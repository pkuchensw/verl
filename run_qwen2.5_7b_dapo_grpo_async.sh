#!/usr/bin/env bash
# ============================================================
# DAPO-style GRPO RL training for Qwen2.5-7B (dense) — COLOCATE ASYNC
#   - Train: DAPO-Math-17k
#   - Eval:  AIME-2024 (pass@8)
#   - Engine: FSDP2 trainer + sglang rollout 同卡 (8 GPU)，训练/推理异步重叠
#   - Trainer mode: colocate_async (和 sync 共用 8 卡 + memory saver，
#                   无需独立 rollout GPU，不会撞卡)
#
# Usage:
#   bash run_qwen2.5_7b_dapo_grpo_async.sh
#
# 与 sync 版区别:
#   - 训练与 rollout 异步重叠（不用等 rollout 完成再训练）
#   - partial rollout: 权重更新时在途请求保存续跑，不丢已生成 token
#   - 权重经 checkpoint_engine(naive) 每步同步，复用 sync 的内存释放路径
#   - colocate_async 不强制 bypass_mode，is_mode 支持 none/tis/mis（同 sync）
# ============================================================

set -xeuo pipefail

export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ====================== User Configurable ======================

# --- Model & Data ---
MODEL_PATH=${MODEL_PATH:-/jizhicfs/pkuhetu/siwei/model/Qwen2.5-Math-7B}
TRAIN_DATA=${TRAIN_DATA:-/jizhicfs/pkuhetu/siwei/data/dapo-math-17k-parquet/train.parquet}
VAL_DATA=${VAL_DATA:-/jizhicfs/pkuhetu/siwei/data/aime-2024.parquet}

# --- Cluster ---
# colocate_async: trainer 和 rollout 同卡，8 卡全用（和 sync 一样），无独立 rollout GPU。
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# --- Training ---
total_epochs=${TOTAL_EPOCHS:-1}
train_batch_size=${TRAIN_BATCH_SIZE:-128}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-16}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-20480}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-8192}

# --- Actor ---
actor_lr=${ACTOR_LR:-1e-6}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
entropy_coeff=${ENTROPY_COEFF:-0}

actor_clip_ratio_low=${ACTOR_CLIP_RATIO_LOW:-0.2}
actor_clip_ratio_high=${ACTOR_CLIP_RATIO_HIGH:-0.28}
actor_clip_ratio_c=${ACTOR_CLIP_RATIO_C:-10.0}

# --- Rollout (sglang, standalone) ---
rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.9}
rollout_n=${ROLLOUT_N:-8}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_top_p=${ROLLOUT_TOP_P:-1.0}

# --- FP8 Rollout Mode ---
#   fp8_mode=weight -> 仅 weight-FP8（quantization=fp8），【推荐，当前最稳】。
#                      不改变 rollout 分布（无长度偏移）。
#   fp8_mode=none   -> 纯 bf16。
#   fp8_mode=kv / both -> 【不推荐】fp8 KV 长序列（max_new>=2048）采样 NaN 崩溃；
#                      且 async 下每次权重同步都会 release/resume KV cache，
#                      加剧数值风险。
fp8_mode=${FP8_MODE:-weight}

# --- Eval (pass@8) ---
test_freq=${TEST_FREQ:-5}
save_freq=${SAVE_FREQ:-50}
val_before_train=${VAL_BEFORE_TRAIN:-False}

# --- Resume ---
#   resume_mode=disable     -> 默认。不读任何 ckpt，从零开始训练，无需改路径。
#   resume_mode=auto        -> 从 default_local_dir 找最新 ckpt 自动续训。
#   resume_mode=resume_path -> 从 resume_from_path 指定的 ckpt 续训（需同时设 RESUME_FROM_PATH）。
resume_mode=${RESUME_MODE:-disable}
resume_from_path=${RESUME_FROM_PATH:-}

# --- Logging ---
project_name=${PROJECT_NAME:-verl_grpo_dapo_math}
experiment_name=${EXPERIMENT_NAME:-qwen2.5_7b_dapo_async}

# ===============================================================

# ---- Step 1: Assemble arguments ----

# --- Importance Sampling Mode ---
#   is_mode=tis     -> Token-level IS with upper-bound clipping (TIS)
#   is_mode=mis     -> Token-level IS with lower+upper bounds (MIS / IcePop)
#   is_mode=none    -> No importance sampling (bypass mode, vanilla loss)
# 注意: async 下 partial rollout 恢复时模型版本可能不同，IS 权重意义有限，
#       建议默认 none。
is_mode=${IS_MODE:-mis}

ALGORITHM=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
)

if [ "$is_mode" = "tis" ]; then
    ALGORITHM+=(
        ++algorithm.rollout_correction.rollout_is=token
        ++algorithm.rollout_correction.rollout_is_threshold=2.0
        ++algorithm.rollout_correction.rollout_is_batch_normalize=false
        ++algorithm.rollout_correction.rollout_rs=null
        ++algorithm.rollout_correction.rollout_rs_threshold=null
        ++algorithm.rollout_correction.bypass_mode=false
        ++algorithm.rollout_correction.loss_type=ppo_clip
    )
    experiment_name=${experiment_name}_tis
elif [ "$is_mode" = "mis" ]; then
    ALGORITHM+=(
        ++algorithm.rollout_correction.rollout_is=token
        ++algorithm.rollout_correction.rollout_is_threshold=0.5_5.0
        ++algorithm.rollout_correction.rollout_is_batch_normalize=false
        ++algorithm.rollout_correction.rollout_rs=null
        ++algorithm.rollout_correction.rollout_rs_threshold=null
        ++algorithm.rollout_correction.bypass_mode=false
        ++algorithm.rollout_correction.loss_type=ppo_clip
    )
    experiment_name=${experiment_name}_mis
else
    # is_mode=none: no importance sampling
    ALGORITHM+=(
        ++algorithm.rollout_correction.bypass_mode=true
        ++algorithm.rollout_correction.loss_type=vanilla
    )
    experiment_name=${experiment_name}_wotis
fi

FP8_ARGS=()
if [ "$fp8_mode" = "both" ]; then
    FP8_ARGS+=(
        ++actor_rollout_ref.rollout.quantization=fp8
        ++actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype=fp8_e4m3
    )
    experiment_name=${experiment_name}_fullfp8
elif [ "$fp8_mode" = "kv" ]; then
    FP8_ARGS+=(
        ++actor_rollout_ref.rollout.engine_kwargs.sglang.kv_cache_dtype=fp8_e4m3
    )
    experiment_name=${experiment_name}_kvfp8
elif [ "$fp8_mode" = "weight" ]; then
    FP8_ARGS+=(
        ++actor_rollout_ref.rollout.quantization=fp8
    )
    experiment_name=${experiment_name}_wfp8
elif [ "$fp8_mode" != "none" ]; then
    echo "ERROR: unknown fp8_mode '$fp8_mode' (expected none|kv|weight|both)" >&2
    exit 1
fi

REWARD=(
    reward_model.reward_manager=dapo
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=True
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
)

DATA=(
    data.train_files="['$TRAIN_DATA']"
    data.val_files="['$VAL_DATA']"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.prompt_key=prompt
)

MODEL=(
    actor_rollout_ref.model.path="$MODEL_PATH"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa
)

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=${entropy_coeff}
    actor_rollout_ref.actor.clip_ratio_low=${actor_clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${actor_clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=${actor_clip_ratio_c}
    # 同卡（colocated）：训练后参数驻留 CPU，腾显存给 sglang resume 权重；
    # optimizer 也 offload，减少 FSDP 常驻显存，8 卡下更宽松。
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=sglang
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.top_p=${rollout_top_p}
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    # 同卡（colocated）：权重同步走 naive checkpoint engine 的
    # release_kv_cache/resume_kv_cache 路径，free_cache_engine 保留 True。
    actor_rollout_ref.rollout.free_cache_engine=True
    # pass@8 validation: each AIME prompt generates 8 responses with sampling
    ++actor_rollout_ref.rollout.val_kwargs.n=8
    ++actor_rollout_ref.rollout.val_kwargs.do_sample=True
    ++actor_rollout_ref.rollout.val_kwargs.temperature=0.7
)

REF=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.ref.fsdp_config.param_offload=True
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["console","wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    +trainer.ray_kwargs.ray_init.address=auto
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
    trainer.val_before_train=${val_before_train}
    trainer.resume_mode=${resume_mode}
    # ---- colocate_async 特有：v1 async trainer，trainer/rollout 同卡 ----
    trainer.v1.trainer_mode=colocate_async
    trainer.v1.colocate_async.num_warmup_batches=1
)

# ---- Step 2: Launch ----
echo ""
echo "============================================================"
echo "  DAPO GRPO RL Training - Qwen2.5-7B (colocate_async)"
echo "  Model:      $MODEL_PATH"
echo "  Train:      $TRAIN_DATA ($(python3 -c "from datasets import Dataset; print(len(Dataset.from_parquet('$TRAIN_DATA')))"))"
echo "  Eval:       $VAL_DATA (pass@8)"
echo "  GPUs:       $NGPUS_PER_NODE (trainer + rollout 同卡共享)"
echo "  Batch:      $train_batch_size  (mini: $ppo_mini_batch_size)"
echo "  Rollout n:  $rollout_n  LR: $actor_lr"
echo "  FP8 Mode:   $fp8_mode"
echo "  Note:       训练/rollout 异步重叠，partial rollout 续跑"
echo "============================================================"
echo ""

sleep 2

python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${FP8_ARGS[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"

echo "Training complete!"
