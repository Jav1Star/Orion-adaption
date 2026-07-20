#!/usr/bin/env bash
set -euo pipefail

T=$(date +%m%d%H%M)
RUN_TS=$(date +%m%d_%H%M)
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
ORION_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

# 关键调用点：stage1 训练固定走这一份 config，避免把不必要的路径选择暴露成日常操作成本。
CFG_PATH=${CFG_PATH:-"$ORION_ROOT/adzoo/orion/configs/orion_stage1_adaption_train.py"}
CONDA_ENV=${CONDA_ENV:-simlingo_310}
GPU_ID=${GPU_ID:-0}
RUN_NAME=${RUN_NAME:-"orion_stage1_${RUN_TS}"}
WORK_DIR=${WORK_DIR:-"$ORION_ROOT/work_dirs/$RUN_NAME"}
TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-"$ORION_ROOT/data/infos/b2d_infos_train.pkl"}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
NUM_EPOCHS=${NUM_EPOCHS:-}
LOG_INTERVAL=${LOG_INTERVAL:-10}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-}

DEFAULT_PYTHON_BIN="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
if [ ! -x "$DEFAULT_PYTHON_BIN" ]; then
    echo "python binary not found for conda env '$CONDA_ENV': $DEFAULT_PYTHON_BIN" >&2
    exit 1
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}

mkdir -p "$WORK_DIR/logs"

echo "[orion-stage1-train] root=$ORION_ROOT"
echo "[orion-stage1-train] env=$CONDA_ENV gpu=$GPU_ID"
echo "[orion-stage1-train] cfg=$CFG_PATH"
echo "[orion-stage1-train] ann_file=$TRAIN_ANN_FILE"
echo "[orion-stage1-train] run_name=$RUN_NAME"
echo "[orion-stage1-train] work_dir=$WORK_DIR"

cd "$ORION_ROOT"
CFG_OPTIONS=(
    "data.train.ann_file=$TRAIN_ANN_FILE"
    "data.workers_per_gpu=$WORKERS_PER_GPU"
    "log_config.interval=$LOG_INTERVAL"
)
if [ -n "$SAMPLES_PER_GPU" ]; then
    CFG_OPTIONS+=("data.samples_per_gpu=$SAMPLES_PER_GPU")
fi
if [ -n "$CHECKPOINT_INTERVAL" ]; then
    CFG_OPTIONS+=("checkpoint_config.interval=$CHECKPOINT_INTERVAL")
fi
if [ -n "$NUM_EPOCHS" ]; then
    CFG_OPTIONS+=("num_epochs=$NUM_EPOCHS")
fi

PYTHONPATH="$ORION_ROOT:${PYTHONPATH:-}" \
"$PYTHON_BIN" -m adzoo.orion.train \
    "$CFG_PATH" \
    --gpu-ids "$GPU_ID" \
    --work-dir "$WORK_DIR" \
    --cfg-options \
    "${CFG_OPTIONS[@]}" \
    "${@}" \
    2>&1 | tee "$WORK_DIR/logs/train.$T"
