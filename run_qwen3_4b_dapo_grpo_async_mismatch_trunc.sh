#!/usr/bin/env bash
# ============================================================
# DAPO-style GRPO RL training for Qwen2.5-7B (dense) — COLOCATE ASYNC
#   + MISMATCH-AWARE PARTIAL ROLLOUT TRUNCATION
#   - Train: DAPO-Math-17k
#   - Eval:  AIME-2024 (pass@8)
#   - Engine: FSDP2 trainer + sglang rollout 同卡 (8 GPU)，训练/推理异步重叠
#   - Trainer mode: colocate_async (和 sync 共用 8 卡 + memory saver，
#                   无需独立 rollout GPU，不会撞卡)
#
# 与 run_qwen2.5_7b_dapo_grpo_async.sh 唯一区别（其余参数完全一致）:
#   - 开启 algorithm.partial_rollout_mismatch_truncation:
#     权重同步时用 pi_{k+1} 对在途 partial-rollout prefix 重算逐 token logprob，
#     与 rollout 保存的 pi_old logprob 比较；首个 |log_ratio|>τ 处回退到
#     b = max(t*-rollback_window, 0)，同步后用新模型从 b 续生，丢弃高 mismatch 后缀。
#   - 不改 GRPO loss / reward / batch，只改 Async Partial Rollout 的保存/恢复方式。
#
# Usage:
#   bash run_qwen2.5_7b_dapo_grpo_async_mismatch_trunc.sh
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

# --- sglang JIT kernel host compiler ---
# sglang 0.5.x 会对若干 CUDA kernel 做 JIT 编译（tvm_ffi -> ninja -> nvcc），编译参数
# 用 -std=c++20 且 #include <version>，该头文件 GCC 9 才引入。host gcc 为 8.x 的节点
# （如 28.59.87.74，gcc 8.5.0）会编译失败（per_token_group_quant_8bit、
# resolve_future_token_ids 等）-> ninja exited 1 -> 进程退出时 CUDA 分配器析构触发
# "CUDA error: invalid argument" abort。此时需用 gcc-toolset（>=12）作为 host compiler。
# 检测系统默认 gcc 若 >=9 则不动；否则启用 gcc-toolset-14（若存在）。
# 说明：这只影响 sglang 内核算子的本地编译，不改变任何训练数值/语义。
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
fp8_mode=${FP8_MODE:-none}

# --- sglang engine kwargs ---
# sglang Piecewise CUDA Graph（0.5.x 默认开启的实验特性）启动时会对一批内核做
# JIT 编译（tvm_ffi -> ninja -> nvcc）。编译用 -std=c++20 且 #include <version>，
# 该头文件 GCC 9 才引入；host gcc 为 8.x 的节点（如 28.59.87.74，gcc 8.5.0）会
# 编译失败（per_token_group_quant_8bit 等）-> ninja exited 1 -> 进程退出时 CUDA
# 分配器析构触发 "CUDA error: invalid argument" abort。与 fp8/bf16 无关。
# 关闭该优化对训练正确性无影响（本脚本本已 enforce_eager + disable_cuda_graph），
# 仅推理性能略降。装 gcc>=9 后可将 SGL_DISABLE_PIECEWISE_CUDA_GRAPH=false 恢复。
sgl_disable_piecewise_cuda_graph=${SGL_DISABLE_PIECEWISE_CUDA_GRAPH:-true}

# --- Mismatch-aware Partial Rollout ---
# 权重同步时对在途 partial-rollout prefix 用 pi_{k+1} 重算 logprob，与 pi_old
# 逐 token 比较，首个 |log_ratio|>τ 处回退到 b=max(t*-rollback_window,0)。
#   mismatch_trunc_enable=false -> 完全等价于原 async 脚本（可做对照）。
#   log_ratio_threshold τ 是全局固定阈值。2026-08-14 校准：fp8-vs-bf16 引擎噪声
#   逐 token max=0.97 / p99.9=0.34（200 条 DAPO-Math prefix, 9600 token），τ>=1.5 无
#   误触发。2026-08-30 观测：τ=2.0 时 truncated 率高（~11.6%）、avg_rollback=1436 过深，
#   长 prefix 上过早触发导致大量推理被丢。放宽到 τ=4.0（噪声 p99.9 的 ~12x 余量），
#   只在真实 drift（math-vs-base anchor 命中 22%，|log_ratio| 明显 > 噪声尾）才回退。
mismatch_trunc_enable=${MISMATCH_TRUNC_ENABLE:-true}
log_ratio_threshold=${LOG_RATIO_THRESHOLD:-4.0}
rollback_window=${ROLLBACK_WINDOW:-32}

# --- Training-time Mismatch Screening (train_mismatch_gate) ---
# 每步多采 gate_oversample 个 prompt 组，用 pi_k 算逐轨迹 per-token 极端 mismatch
# = max_t |pi_k_logprob_t - rollout_logprob_t|，|max|>gate_log_ratio_threshold 的
# 组做"局部回退"（与 truncation 相同：在首个超阈值 token t* 处回退 b=max(t*-window,0)，
# 保留前缀、只重生成尾段），保证每个进入训练的样本都被筛过至少一次。
# 度量与 partial_rollout_mismatch_truncation 对齐（都是"存在某个单 token 偏差超过
# 阈值"），不再用序列求和——求和被响应长度支配（L x 每 token 常偏），导致"最脏的
# 16 组"其实是"最长的 16 组"，dirty 率 ~35%、force_accept 50+。
#   gate_enable=false  -> 完全等价于未启用（对照用）。
#   gate_log_ratio_threshold: per-token 极端偏差阈值，与 truncation 的
#     log_ratio_threshold 同一尺度。默认 4.0（与截断阈值一致）：引擎噪声
#     per-token max=0.97 / p99.9=0.34，4.0 约 4x 余量，只抓明显 drift；
#     若 dirty 率仍高可上调、嫌漏检可下调。
#   gate_rollback_window: 与 truncation 的 rollback_window 相同（默认 32）。
#     b=max(t*-window,0)，保留前缀、丢弃高 mismatch 尾段。组内干净 session 原样
#     重入（seed_is_complete），只对真正脏的 session 回退，绝不整组从零重生成。
#   gate_oversample: 每步多采的余量 = 每步最多强制重采样的组数上限（top-K）。
#     候选不是"超阈值组"，而是【所有 sampled 组按 per-token mismatch 排序取前 K】——
#     相对 top-K 强制重采样，不依赖 τ 精确标定：每步主动把最偏离当前策略的
#     min(K, 达标数) 个组拉回 pi_k 重生成（保留前缀、只重生成尾段）。干净时前 K 的
#     mismatch 也未必超阈值，但只要够"最偏"就会被重采样（受 min_llr 软下限约束）。
#     代价在池子：每步净入池 = gen_batch(128) + 回退组回归 - (128 + oversample) 消耗，
#     oversample > 回退回归率 时池子净流出，靠 warmup 余量撑，余量耗尽即 141/144 或
#     143/144 卡死（2026-08-31 两次实测）。replay_buffer.py 已加自愈守卫：无在途请求
#     且长期凑不满批时自动补发一整批重建余量，因此任何 oversample 都不会死锁，但
#     oversample 越大、补发越频繁、每步多等一个生成周期。
#   gate_min_llr: top-K 的软下限。低于此 per-token mismatch 的组即使排进前 K 也不重采样
#     （引擎噪声地板 ~1.0，防止为凑满 K 而白回退相对噪声）。默认 1.5，可下调激进重采样。
#   gate_log_ratio_threshold: 现在只用于两处——(a) 统计 dirty 率（诊断）；(b) 判定某条
#     轨迹的 rollback 点 b=max(t*-window,0)。不再直接决定"回退哪些组"。
#   redispatch_topk_extra 指标（wandb）：前 K 里真正低于阈值（纯阈值过滤会放行）的组数。
#     持续非零 -> top-K 在补阈值漏掉的尾段；过大 -> 阈值太紧、重采样偏激进。
# 注意：bypass 模式（is_mode=none）下 gate 会自己多跑一次 pi_k 前向；
#       decoupled 模式（is_mode=tis/mis）下直接复用 _compute_old_log_prob 的结果。
gate_enable=${GATE_ENABLE:-true}
gate_log_ratio_threshold=${GATE_LOG_RATIO_THRESHOLD:-4.0}
gate_rollback_window=${GATE_ROLLBACK_WINDOW:-32}
gate_oversample=${GATE_OVERSAMPLE:-16}
gate_min_llr=${GATE_MIN_LLR:-1.5}
gate_max_force_accept_regens=${GATE_MAX_FORCE_ACCEPT_REGENS:-3}

# --- Eval (pass@8) ---
test_freq=${TEST_FREQ:-5}
save_freq=${SAVE_FREQ:-50}
val_before_train=${VAL_BEFORE_TRAIN:-False}

# --- Resume ---
#   resume_mode=auto        -> 默认。从 default_local_dir
#                              (checkpoints/${project_name}/${experiment_name})
#                              找最新 ckpt 自动续训；无 ckpt 时从零开始。
#   resume_mode=disable     -> 强制从零训练（不读任何 ckpt）。
#   resume_mode=resume_path -> 从 resume_from_path 指定的 ckpt 续训（需同时设 RESUME_FROM_PATH）。
resume_mode=${RESUME_MODE:-disable}
resume_from_path=${RESUME_FROM_PATH:-}

# --- Logging ---
project_name=${PROJECT_NAME:-verl_grpo_dapo_math}
experiment_name=${EXPERIMENT_NAME:-Qwen3_4b_dapo_async}

# ===============================================================

# ---- Step 1: Assemble arguments ----

# --- Importance Sampling Mode ---
#   is_mode=tis     -> Token-level IS with upper-bound clipping (TIS)
#   is_mode=mis     -> Token-level IS with lower+upper bounds (MIS / IcePop)
#   is_mode=none    -> bypass: old_log_probs=rollout_log_probs（不重算），loss 用
#                      PPO clip(pi_theta/pi_rollout)。去掉显式 TIS/MIS 权重层，
#                      但 rollout 数值仍作为 ratio 分母(隐式 IS)进 loss。
#   is_mode=nois    -> 【真·无 rollout 校正】训练引擎自己重算 old_log_probs=pi_k，
#                      rollout_is/rollout_rs 全关，loss = PPO clip(pi_theta/pi_k)，
#                      rollout_log_probs 完全不进 loss（只供 gate 筛选与 rollout_corr/* 诊断）。
# 默认 tis。train_mismatch_gate 在 none/nois 下会自己跑一次 pi_k 前向（nois 下复用重算结果）。
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
    # nois: 完全无 rollout IS 校正。bypass_mode=false -> _compute_old_log_prob 走
    # decoupled 分支，用训练引擎对整批重算 old_log_probs=pi_k(近端 anchor)；
    # rollout_is/rollout_rs 全 null -> _compute_advantage 不生成 rollout_is_weights、
    # 不改 response_mask。loss = 标准 PPO clip(pi_theta/pi_k)，rollout 引擎的
    # logprob(含 fp8 噪声/NaN) 完全不进 loss。rollout_log_probs 仍在 batch，
    # 供 train_mismatch_gate 筛选与 rollout_corr/* 诊断指标使用。
    ALGORITHM+=(
        ++algorithm.rollout_correction.rollout_is=null
        ++algorithm.rollout_correction.rollout_rs=null
        ++algorithm.rollout_correction.rollout_rs_threshold=null
        ++algorithm.rollout_correction.bypass_mode=false
    )
    experiment_name=${experiment_name}_nois
else
    # is_mode=none: bypass。_compute_old_log_prob 直接把 rollout_log_probs 当
    # old_log_probs(不重算 pi_k)，loss = PPO clip(pi_theta/pi_rollout)。
    # 去掉了显式 TIS/MIS 权重层，但 rollout 数值作为 ratio 分母(隐式 IS)仍在 loss。
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
echo "  DAPO GRPO RL Training - Qwen2.5-7B (colocate_async + mismatch truncation)"
echo "  Model:      $MODEL_PATH"
echo "  Train:      $TRAIN_DATA ($(python3 -c "from datasets import Dataset; print(len(Dataset.from_parquet('$TRAIN_DATA')))"))"
echo "  Eval:       $VAL_DATA (pass@8)"
echo "  GPUs:       $NGPUS_PER_NODE (trainer + rollout 同卡共享)"
echo "  Batch:      $train_batch_size  (mini: $ppo_mini_batch_size)"
echo "  Rollout n:  $rollout_n  LR: $actor_lr"
echo "  FP8 Mode:   $fp8_mode"
echo "  Mismatch Trunc: $mismatch_trunc_enable (tau=$log_ratio_threshold, window=$rollback_window)"
echo "  Note:       训练/rollout 异步重叠，partial rollout 续跑"
echo "============================================================"
echo ""

sleep 2

# --- cgroup 内存上限诊断 ---
# 训练 pod 的 CPU-RAM 上限由调度器在 pod 创建时定，pod 内不可改。这里把真实
# limit 和当前用量打进日志，便于判断 OOM 是否因 pod 内存 limit 过小（此前多次
# CONSTRAINT_MEMCG 在 driver anon-rss≈54.5GB 处被杀）。
echo "=== mem diag ==="
# 注意: 本脚本开头是 `set -euo pipefail`。cgroup 文件可能因权限/布局不可读，
# 任何 `cat ... | awk` 若 cat 失败都会让脚本整段退出，所以一律用 [ -r ] 保护。
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
# CPU-only (verl.model_merger loads shards to RAM), ~14GB peak for 7B bf16.
# Runs only if a checkpoint exists; safe after training has released the pod wall.
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
