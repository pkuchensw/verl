#!/usr/bin/env bash
# Evaluate multiple verl checkpoints on AIME-2024 with pass@8
# Usage: bash eval_aime_pass8.sh

set -xeuo pipefail

export PATH=/jizhicfs/johnnyslin/anaconda3/envs/csw-verl/bin:$PATH
export PYTHONPATH=/jizhicfs/pkuhetu/siwei/verl:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
export OMP_NUM_THREADS=2

BASE_DIR=/jizhicfs/pkuhetu/siwei/verl
CKPT_DIR=$BASE_DIR/checkpoints/verl_grpo_dapo_math
export AIME_PARQUET=/jizhicfs/pkuhetu/siwei/data/aime-2024.parquet
MERGE_SCRIPT=$BASE_DIR/scripts/legacy_model_merger.py

# Only test global_step_100 and global_step_135 for both models
CONFIGS=(
    "qwen2.5_7b_dapo_mis,global_step_100"
    "qwen2.5_7b_dapo_mis,global_step_135"
)

echo "=========================================="
echo " AIME-2024 Pass@8 Evaluation"
echo "=========================================="

for CONFIG in "${CONFIGS[@]}"; do
    IFS=',' read -r EXPERIMENT STEP <<< "$CONFIG"
    echo ""
    echo "=== Evaluating: $EXPERIMENT $STEP ==="

    export EXPERIMENT STEP
    LOCAL_DIR=$CKPT_DIR/$EXPERIMENT/$STEP/actor
    export TARGET_DIR=/tmp/aime_eval_${EXPERIMENT}_${STEP}

    rm -rf "$TARGET_DIR"

    python3 "$MERGE_SCRIPT" merge \
        --backend fsdp \
        --local_dir "$LOCAL_DIR" \
        --target_dir "$TARGET_DIR" 2>&1

    python3 "$BASE_DIR/eval_aime_pass8.py"

    rm -rf "$TARGET_DIR"
    echo ""
done

echo "=========================================="
echo " All evaluations complete!"
echo "=========================================="
