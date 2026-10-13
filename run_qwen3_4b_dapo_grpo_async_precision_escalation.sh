#!/usr/bin/env bash
# ============================================================
# DAPO-style GRPO RL training for Qwen2.5-7B (dense) — COLOCATE ASYNC
#   + MISMATCH-AWARE PARTIAL ROLLOUT TRUNCATION
#   + TRAIN-TIME MISMATCH GATE
#   + MISMATCH-DRIVEN PRECISION ESCALATION   <-- new
#   - Train: DAPO-Math-17k
#   - Eval:  AIME-2024 (pass@8)
#   - Engine: FSDP2 trainer + sglang rollout 同卡 (8 GPU)，训练/推理异步重叠
#   - Trainer mode: colocate_async
#
# 与 run_qwen2.5_7b_dapo_grpo_async_mismatch_trunc.sh 的区别（其余参数完全一致）:
#   额外开启 algorithm.mismatch_precision_escalation。fp8 rollout 引擎与 bf16 训练
#   引擎的分布不一致，truncation / gate 两个机制都在【同一个低精度引擎】上原地修补。
#   一个题组若反复触发这两个机制，说明瓶颈是引擎精度本身而非模型版本漂移——原地
#   重生成只会继续触发。该机制因此：
#     1) 按 (uid, session) 统计回退次数；某个 session 回退 >= num_rollbacks_trigger
#        次即把该 uid 组标记为"升级"，丢弃其此前全部推理并放入缓冲区；
#     2) 缓冲区攒够 batch_size 个组时，在 on_step_end 的正常权重同步+resume 之后
#        执行升级：先把 rollout pipeline 排空（此时 trainer 不会再派发新 prompt，
#        在途请求自然跑完，引擎进入空闲），再把 rollout 引擎整体重启为高精度(bf16)
#        构建，只跑这批升级样本，最后切回原精度并同步权重；
#     3) 切换期间不做别的事（整个引擎换精度，天然串行/阻塞）。排空是必须的：
#        杀掉 server actor 时若还有在途/排队请求，它们会随 actor 一起失败。
#     4) 排空会清空 pipeline，但异步 replay buffer 有 starvation 自愈(refill_fn)，
#        下一步会自行补派，吞吐自恢复，不影响正确性。
#
# 实现:
#   - 控制层:  verl/trainer/ppo/v1/precision_escalation.py (只管策略/计数/缓冲)
#   - 切换层:  verl/trainer/ppo/v1/engine_precision_switch.py (只管重启引擎)
#   - 接线:    trainer_colocate_async.py on_step_end -> _maybe_run_precision_escalation
#   - 回退归属: single_turn_agent_loop 把 uid_session 作为 escalation_key 传给
#     FullyAsyncLLMServerClient，再由 get_inflight_prefixes 带回给 trainer。
#   - 升级样本重建: 只用 uid 列表，prompt 行由 _submit_batch_to_rollout 持久化在
#     TransferQueue 的 uid key 下，届时按 uid 读回即可（无额外 payload 缓存）。
#
# strategy=drop（PREC_ESC_STRATEGY=drop）: 不切引擎。触发的整组被判为"低精度引擎救不动"，
#   由 sampler 在采样前把该组（prompt+轨迹）从 TransferQueue 清掉并 refill 补位，整组不参与
#   训练。相比 switch：省掉排空/引擎重启/重生成的开销、不阻塞训练，但丢数据而非救数据。
#
# Usage:
#   bash run_qwen2.5_7b_dapo_grpo_async_precision_escalation.sh
# ============================================================

set -xeuo pipefail

export OMP_NUM_THREADS=2
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- sglang JIT kernel host compiler ---
# sglang 0.5.x 会对若干 CUDA kernel 做 JIT 编译（tvm_ffi -> ninja -> nvcc），编译参数
# 用 -std=c++20 且 #include <version>，该头文件 GCC 9 才引入。host gcc 为 8.x 的节点
# 会编译失败 -> ninja exited 1 -> 进程退出时 CUDA 分配器析构触发 "CUDA error:
# invalid argument" abort。此时需用 gcc-toolset（>=12）作为 host compiler。
if command -v gcc >/dev/null 2>&1 && [ "$(gcc -dumpversion 2>/dev/null | cut -d. -f1)" -ge 9 ]; then
    : # system gcc is new enough, keep as-is
else
    for _toolset in gcc-toolset-14 gcc-toolset-13 gcc-toolset-12; do
        if [ -x "/opt/rh/${_toolset}/root/usr/bin/g++" ]; then
            echo "host gcc too old; enabling ${_toolset} as nvcc host compiler"
            export PATH="/opt/rh/${_toolset}/root/usr/bin:${PATH}"
            export CC="/opt/rh/${_toolset}/root/usr/bin/gcc"
            export CXX="/opt/rh/${_toolset}/root/usr/bin/g++"
            break
        fi
    done
fi

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

# --- FP8 Rollout Mode ---
#   fp8_mode=weight -> 仅 weight-FP8（quantization=fp8），【推荐，当前最稳】。
#   fp8_mode=none   -> 纯 bf16。注意：此时基础引擎已是 bf16，"升级到 bf16" 无意义，
#                      脚本会拒绝（见下方校验）。
#   fp8_mode=kv / both -> 【不推荐】fp8 KV 长序列采样 NaN 崩溃。
fp8_mode=${FP8_MODE:-none}

# --- sglang engine kwargs ---
sgl_disable_piecewise_cuda_graph=${SGL_DISABLE_PIECEWISE_CUDA_GRAPH:-true}

# --- Mismatch-aware Partial Rollout Truncation ---
mismatch_trunc_enable=${MISMATCH_TRUNC_ENABLE:-true}
log_ratio_threshold=${LOG_RATIO_THRESHOLD:-4.0}
rollback_window=${ROLLBACK_WINDOW:-32}

# --- Training-time Mismatch Screening (train_mismatch_gate) ---
gate_enable=${GATE_ENABLE:-true}
gate_log_ratio_threshold=${GATE_LOG_RATIO_THRESHOLD:-4.0}
gate_rollback_window=${GATE_ROLLBACK_WINDOW:-32}
gate_oversample=${GATE_OVERSAMPLE:-16}
gate_min_llr=${GATE_MIN_LLR:-1.5}
gate_max_force_accept_regens=${GATE_MAX_FORCE_ACCEPT_REGENS:-3}

# --- Mismatch-driven Precision Escalation ---
#   prec_esc_enable      -> 总开关。
#   prec_esc_trigger     -> 单条轨迹（uid_session）自上次重生成以来被回退达到该次数时，
#                          把整个 uid 组升级。3 = 已扛过两次就地回退(dtruncation/gate
#                          本可自行处理)、又第三次触发，才算"引擎精度本身有病"。
#   每个组只升级一次（one-shot）：升到 bf16 重算一次是"这个组是不是被引擎精度卡住"
#   的一次验证；若 bf16 后仍然回退，说明不是精度问题而是题目本身难，就此作罢——否则
#   同一批慢性组会每 prec_esc_trigger 步复活一次，引擎反复重启。这也让总切换次数上界
#   为「被卡住的组数 / batch_size」，而非随时间无限增长。
#   prec_esc_batch       -> 攒够多少个升级组才切引擎重生成一次。单位是 prompt 组
#                          （每组含 rollout.n 条，n=8 时 32 组 = 256 条序列）。
#                          刻意远小于 train_batch_size：一个组要先触发 3 次才进缓冲区，
#                          攒满整个训练 batch 要等太久，少数几个"慢性病"组会一直卡着
#                          拿不到 bf16 重算。取小值让它们能被及时救出来，同时因为
#                          触发门槛高，切换仍然很少。
#   prec_esc_target      -> 升级目标精度，目前只支持 bf16（仅 strategy=switch 用）。
#   prec_esc_strategy    -> switch（默认）：按上面的机制切引擎重生成；
#                           drop：不切引擎，直接把触发的整组从采样池里清掉（由 sampler
#                           在采样前 clear 其 prompt+轨迹 并 refill 补位），该组不参与训练。
#                           代价低、不阻塞、无引擎重启，但丢数据而非救数据；此时
#                           prec_esc_batch / prec_esc_target 不生效。drop 也可用于 bf16 基础引擎。
prec_esc_enable=${PREC_ESC_ENABLE:-true}
prec_esc_strategy=${PREC_ESC_STRATEGY:-switch}
prec_esc_trigger=${PREC_ESC_TRIGGER:-3}
prec_esc_batch=${PREC_ESC_BATCH:-64}
prec_esc_target=${PREC_ESC_TARGET:-bf16}

# --- Eval (pass@8) ---
test_freq=${TEST_FREQ:-5}
save_freq=${SAVE_FREQ:-50}
val_before_train=${VAL_BEFORE_TRAIN:-False}

# --- Resume ---
resume_mode=${RESUME_MODE:-disable}
resume_from_path=${RESUME_FROM_PATH:-}

# --- Logging ---
project_name=${PROJECT_NAME:-verl_grpo_dapo_math}
experiment_name=${EXPERIMENT_NAME:-Qwen3-4B_dapo_async}

# ===============================================================

# ---- Step 1: Assemble arguments ----

# --- Importance Sampling Mode ---
is_mode=${IS_MODE:-none}

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
elif [ "$is_mode" = "nois" ]; then
    ALGORITHM+=(
        ++algorithm.rollout_correction.rollout_is=null
        ++algorithm.rollout_correction.rollout_rs=null
        ++algorithm.rollout_correction.rollout_rs_threshold=null
        ++algorithm.rollout_correction.bypass_mode=false
    )
    experiment_name=${experiment_name}_nois
else
    ALGORITHM+=(
        ++algorithm.rollout_correction.rollout_is=null
        ++algorithm.rollout_correction.rollout_rs=null
        ++algorithm.rollout_correction.bypass_mode=true
        ++algorithm.rollout_correction.loss_type=ppo_clip
    )
    experiment_name=${experiment_name}_wotis
fi

# --- Mismatch-aware Partial Rollout Truncation ---
MISMATCH_TRUNC=()
if [ "$mismatch_trunc_enable" = "true" ]; then
    MISMATCH_TRUNC+=(
        algorithm.partial_rollout_mismatch_truncation.enable=true
        algorithm.partial_rollout_mismatch_truncation.log_ratio_threshold=${log_ratio_threshold}
        algorithm.partial_rollout_mismatch_truncation.rollback_window=${rollback_window}
    )
    experiment_name=${experiment_name}_mtrunc
fi

# --- Training-time Mismatch Screening ---
GATE_ARGS=()
if [ "$gate_enable" = "true" ]; then
    GATE_ARGS+=(
        algorithm.train_mismatch_gate.enable=true
        algorithm.train_mismatch_gate.log_ratio_threshold=${gate_log_ratio_threshold}
        algorithm.train_mismatch_gate.rollback_window=${gate_rollback_window}
        algorithm.train_mismatch_gate.oversample=${gate_oversample}
        algorithm.train_mismatch_gate.min_llr=${gate_min_llr}
        algorithm.train_mismatch_gate.max_force_accept_regens=${gate_max_force_accept_regens}
    )
    experiment_name=${experiment_name}_gate
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

# --- Mismatch-driven Precision Escalation ---
# 升级只有在"基础引擎精度 != 目标精度"时才有意义。基础引擎为 bf16（fp8_mode=none）
# 时升级到 bf16 是空操作，直接拒绝以免白白重启引擎。
PREC_ESC=()
if [ "$prec_esc_enable" = "true" ]; then
    if [ "$prec_esc_strategy" != "switch" ] && [ "$prec_esc_strategy" != "drop" ]; then
        echo "ERROR: unknown PREC_ESC_STRATEGY '$prec_esc_strategy' (expected switch|drop)" >&2
        exit 1
    fi
    # 升级到 bf16 只有在基础引擎是低精度时才有意义；drop 策略不碰引擎，故不要求。
    if [ "$prec_esc_strategy" = "switch" ]; then
        if [ "$fp8_mode" = "none" ]; then
            echo "ERROR: PREC_ESC_STRATEGY=switch 需要低精度基础引擎（fp8_mode=weight/both）；当前 fp8_mode=none 已是 bf16，升级无意义。" >&2
            exit 1
        fi
        if [ "$prec_esc_target" != "bf16" ]; then
            echo "ERROR: 目前只支持 PREC_ESC_TARGET=bf16（当前 '$prec_esc_target'）。" >&2
            exit 1
        fi
    fi
    PREC_ESC+=(
        algorithm.mismatch_precision_escalation.enable=true
        algorithm.mismatch_precision_escalation.strategy=${prec_esc_strategy}
        algorithm.mismatch_precision_escalation.num_rollbacks_trigger=${prec_esc_trigger}
        algorithm.mismatch_precision_escalation.batch_size=${prec_esc_batch}
        algorithm.mismatch_precision_escalation.target_precision=${prec_esc_target}
    )
    experiment_name=${experiment_name}_pesc
    if [ "$prec_esc_strategy" = "drop" ]; then
        experiment_name=${experiment_name}drop
    fi
fi

SGL_ARGS=()
if [ "$sgl_disable_piecewise_cuda_graph" = "true" ]; then
    SGL_ARGS+=(
        ++actor_rollout_ref.rollout.engine_kwargs.sglang.disable_piecewise_cuda_graph=true
    )
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
    trainer.v1.trainer_mode=colocate_async
    trainer.v1.colocate_async.num_warmup_batches=1
)

# ---- Step 2: Launch ----
echo ""
echo "============================================================"
echo "  DAPO GRPO RL Training - Qwen2.5-7B (colocate_async + precision escalation)"
echo "  Model:      $MODEL_PATH"
echo "  Train:      $TRAIN_DATA ($(python3 -c "from datasets import Dataset; print(len(Dataset.from_parquet('$TRAIN_DATA')))"))"
echo "  Eval:       $VAL_DATA (pass@8)"
echo "  GPUs:       $NGPUS_PER_NODE (trainer + rollout 同卡共享)"
echo "  Batch:      $train_batch_size  (mini: $ppo_mini_batch_size)"
echo "  Rollout n:  $rollout_n  LR: $actor_lr"
echo "  FP8 Mode:   $fp8_mode"
echo "  Mismatch Trunc: $mismatch_trunc_enable (tau=$log_ratio_threshold, window=$rollback_window)"
echo "  Mismatch Gate:  $gate_enable (oversample=$gate_oversample)"
echo "  Precision Esc:  $prec_esc_enable (strategy=$prec_esc_strategy, trigger=$prec_esc_trigger, batch=$prec_esc_batch, target=$prec_esc_target)"
echo "============================================================"
echo ""

sleep 2

# --- cgroup 内存上限诊断 ---
echo "=== mem diag ==="
_CGP=$(sed -n 's/.*memory:\(\/.*\)/\1/p' /proc/self/cgroup || true)
echo "  pod cgroup: ${_CGP}"
if [ -n "$_CGP" ] && [ -r "/sys/fs/cgroup${_CGP}/memory.max" ]; then
    _MAX=$(cat "/sys/fs/cgroup${_CGP}/memory.max" 2>/dev/null || true)
    [ -n "$_MAX" ] && echo "  memory.max: $(echo $_MAX/1073741824 | awk '{printf "%.1f GB", $1}')" || echo "  memory.max: (无限制)"
fi
if [ -n "$_CGP" ] && [ -r "/sys/fs/cgroup${_CGP}/memory/memory.limit_in_bytes" ]; then
    _MAXV1=$(cat "/sys/fs/cgroup${_CGP}/memory/memory.limit_in_bytes" 2>/dev/null || true)
    [ -n "$_MAXV1" ] && echo "  memory.limit_in_bytes: $(echo $_MAXV1/1073741824 | awk '{printf "%.1f GB", $1}')"
fi
if [ -n "$_CGP" ] && [ -r "/sys/fs/cgroup${_CGP}/memory.current" ]; then
    _CUR=$(cat "/sys/fs/cgroup${_CGP}/memory.current" 2>/dev/null || true)
    [ -n "$_CUR" ] && echo "  memory.current: $(echo $_CUR/1073741824 | awk '{printf "%.1f GB", $1}')"
fi
echo "  node mem: $(free -g 2>/dev/null | awk '/Mem:/{printf "total=%dG used=%dG avail=%dG", $2, $3, $7}' || echo n/a)"
echo "  self rss: $(awk '/VmRSS/{printf "%.1f GB", $2/1048576}' /proc/self/status 2>/dev/null || echo n/a)"

python3 -m verl.trainer.main_ppo \
    "${ALGORITHM[@]}" \
    "${MISMATCH_TRUNC[@]}" \
    "${GATE_ARGS[@]}" \
    "${PREC_ESC[@]}" \
    "${FP8_ARGS[@]}" \
    "${SGL_ARGS[@]}" \
    "${REWARD[@]}" \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${REF[@]}" \
    "${TRAINER[@]}" \
    "$@"

echo "Training complete!"

# === HF export: merge latest FSDP checkpoint -> HF safetensors model ===
CKPT_ROOT="checkpoints/${project_name}/${experiment_name}"
_LATEST_TXT="${CKPT_ROOT}/latest_checkpointed_iteration.txt"
if [ -f "${_LATEST_TXT}" ]; then
    _STEP=$(cat "${_LATEST_TXT}")
    _ACTOR_DIR="${CKPT_ROOT}/global_step_${_STEP}/actor"
    _HF_DIR="${CKPT_ROOT}/hf_global_step_${_STEP}"
    if [ -d "${_ACTOR_DIR}" ]; then
        echo "=== Merging checkpoint global_step_${_STEP} -> HF safetensors at ${_HF_DIR} ==="
        python3 -m verl.model_merger merge \
            --backend fsdp \
            --local_dir "${_ACTOR_DIR}" \
            --target_dir "${_HF_DIR}"
        echo "HF model saved to ${_HF_DIR}"
    else
        echo "!!! No actor dir at ${_ACTOR_DIR}; skip HF export"
    fi
else
    echo "!!! No checkpoint found (${_LATEST_TXT}); skip HF export"
fi
