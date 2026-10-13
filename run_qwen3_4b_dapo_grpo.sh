#!/usr/bin/env bash
# ============================================================
# DAPO-style GRPO RL training for Qwen3-4B-Base (dense)
#   - Train: DAPO-Math-17k
#   - Eval:  AIME-2024 (pass@8)
#   - Engine: FSDP2 training + sglang rollout (single node)
#
# Usage:
#   bash run_qwen3_4b_dapo_grpo.sh
# ============================================================

set -xeuo pipefail

export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ====================== User Configurable ======================

# --- Model & Data ---
MODEL_PATH=${MODEL_PATH:-/jizhicfs/pkuhetu/siwei/model/Qwen3-4B-Base}
TRAIN_DATA=${TRAIN_DATA:-/jizhicfs/pkuhetu/siwei/data/dapo-math-17k-parquet/train.parquet}
VAL_DATA=${VAL_DATA:-/jizhicfs/pkuhetu/siwei/data/aime-2024.parquet}

# --- Cluster ---
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

# --- Rollout (sglang) ---
rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.9}
rollout_n=${ROLLOUT_N:-8}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}
rollout_top_p=${ROLLOUT_TOP_P:-1.0}

# --- Importance Sampling Mode ---
#   is_mode=tis     -> Token-level IS with upper-bound clipping (TIS)
#   is_mode=mis     -> Token-level IS with lower+upper bounds (MIS / IcePop)
#   is_mode=mis_is  -> IcePop mismatch filter + entropy top-20% hard mask (MIS_IS)
#   is_mode=none    -> No importance sampling (bypass mode, vanilla loss)
is_mode=${IS_MODE:-mis}

# --- Eval (pass@8) ---
test_freq=${TEST_FREQ:-5}
save_freq=${SAVE_FREQ:-20}
val_before_train=${VAL_BEFORE_TRAIN:-False}

# --- Logging ---
project_name=${PROJECT_NAME:-verl_grpo_dapo_math}
experiment_name=${EXPERIMENT_NAME:-qwen3_4b_dapo}

# ===============================================================

# ---- Step 1: Assemble arguments ----

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
elif [ "$is_mode" = "mis_is" ]; then
    # MIS_IS: stage-1 IcePop mask (exclude |IS weight| outside [0.5,5.0]),
    #         stage-2 keep entropy top-20% of remaining candidates
    ALGORITHM+=(
        ++algorithm.rollout_correction.rollout_is=null
        ++algorithm.rollout_correction.rollout_is_topk_ratio=0.2
        ++algorithm.rollout_correction.rollout_is_topk_candidate_threshold=0.5_5.0
        ++algorithm.rollout_correction.rollout_is_batch_normalize=false
        ++algorithm.rollout_correction.rollout_rs=null
        ++algorithm.rollout_correction.rollout_rs_threshold=null
        ++algorithm.rollout_correction.bypass_mode=false
        ++algorithm.rollout_correction.loss_type=ppo_clip
    )
    experiment_name=${experiment_name}_mis_is
else
    # is_mode=none: no importance sampling
    ALGORITHM+=(
        ++algorithm.rollout_correction.bypass_mode=true
        ++algorithm.rollout_correction.loss_type=vanilla
    )
    experiment_name=${experiment_name}_wotis
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
    # param_offload=True: 训练完参数放 CPU，腾出 GPU 给 sglang resume 权重
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
)

# ---- Step 2: Launch ----
echo ""
echo "============================================================"
echo "  DAPO GRPO RL Training - Qwen3-4B"
echo "  Model:      $MODEL_PATH"
echo "  Train:      $TRAIN_DATA ($(python3 -c "from datasets import Dataset; print(len(Dataset.from_parquet('$TRAIN_DATA')))"))"
echo "  Eval:       $VAL_DATA (pass@8)"
echo "  GPUs:       $(( NNODES * NGPUS_PER_NODE )) ($NNODES nodes x $NGPUS_PER_NODE)"
echo "  Batch:      $train_batch_size  (mini: $ppo_mini_batch_size)"
echo "  Rollout n:  $rollout_n  LR: $actor_lr"
echo "  IS Mode:    $is_mode (tis/mis/none)"
echo "============================================================"
echo ""

sleep 2

python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"

echo "Training complete!"
