#!/usr/bin/env bash
set -euo pipefail

T=$(date +%m%d%H%M)
RUN_TS=$(date +%m%d_%H%M)
SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
ORION_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)

if [ "${1:-}" != "" ] && [[ "${1:-}" != --* ]]; then
    CFG_PATH=$1
    shift
else
    # 关键调用点：无参数时仍默认跑带 collision loss 的 stage1 配置。
    CFG_PATH=${CFG_PATH:-"$ORION_ROOT/adzoo/orion/configs/orion_stage1_adaption_train_with_col_loss.py"}
fi
if [[ "$CFG_PATH" != /* ]]; then
    CFG_PATH="$ORION_ROOT/$CFG_PATH"
fi

REQUESTED_GPUS=
if [ "${1:-}" != "" ] && [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    REQUESTED_GPUS=$1
    shift
fi

CONDA_ENV=${CONDA_ENV:-simlingo_310}
USER_GPU_IDS=${GPU_IDS:-}
USER_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}
if [ "$REQUESTED_GPUS" != "" ] && [ "$USER_GPU_IDS" = "" ] && [ "$USER_CUDA_VISIBLE_DEVICES" = "" ]; then
    GPU_IDS=$(seq -s, 0 $((REQUESTED_GPUS - 1)))
else
    GPU_IDS=${GPU_IDS:-${CUDA_VISIBLE_DEVICES:-0}}
fi
CFG_BASE=$(basename "${CFG_PATH%.*}")
RUN_NAME=${RUN_NAME:-"${CFG_BASE}_dist_${RUN_TS}"}
WORK_DIR=${WORK_DIR:-"$ORION_ROOT/work_dirs/$RUN_NAME"}
TRAIN_ANN_FILE=${TRAIN_ANN_FILE:-"$ORION_ROOT/data/infos/b2d_infos_train.pkl"}
SAMPLES_PER_GPU=${SAMPLES_PER_GPU:-}
WORKERS_PER_GPU=${WORKERS_PER_GPU:-4}
NUM_EPOCHS=${NUM_EPOCHS:-}
LOG_INTERVAL=${LOG_INTERVAL:-10}
CHECKPOINTS_INTERVAL=${CHECKPOINTS_INTERVAL:-}
MASTER_PORT=${MASTER_PORT:-54621}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}

DEFAULT_PYTHON_BIN="$HOME/miniconda3/envs/$CONDA_ENV/bin/python"
if [ ! -x "$DEFAULT_PYTHON_BIN" ]; then
    echo "python binary not found for conda env '$CONDA_ENV': $DEFAULT_PYTHON_BIN" >&2
    exit 1
fi
PYTHON_BIN=${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}

IFS=',' read -r -a GPU_ID_ARRAY <<< "$GPU_IDS"
GPUS_PER_NODE=${#GPU_ID_ARRAY[@]}
if [ "$GPUS_PER_NODE" -le 0 ]; then
    echo "GPU_IDS must contain at least one GPU id, got '$GPU_IDS'" >&2
    exit 1
fi
if [ "$REQUESTED_GPUS" != "" ] && [ "$GPUS_PER_NODE" -ne "$REQUESTED_GPUS" ]; then
    echo "Requested $REQUESTED_GPUS GPUs, but GPU_IDS/CUDA_VISIBLE_DEVICES resolves to '$GPU_IDS' ($GPUS_PER_NODE GPUs)." >&2
    exit 1
fi

VISIBLE_GPUS=$(CUDA_VISIBLE_DEVICES="$GPU_IDS" "$PYTHON_BIN" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)
if [ "$GPUS_PER_NODE" -gt "$VISIBLE_GPUS" ]; then
    echo "Requested GPUS_PER_NODE=${GPUS_PER_NODE}, but only ${VISIBLE_GPUS} CUDA devices are visible." >&2
    exit 1
fi

mkdir -p "$WORK_DIR/logs"

echo "[orion-dist-train] root=$ORION_ROOT"
echo "[orion-dist-train] env=$CONDA_ENV gpu_ids=$GPU_IDS nproc_per_node=$GPUS_PER_NODE"
echo "[orion-dist-train] cfg=$CFG_PATH"
echo "[orion-dist-train] ann_file=$TRAIN_ANN_FILE"
echo "[orion-dist-train] run_name=$RUN_NAME"
echo "[orion-dist-train] work_dir=$WORK_DIR"

cd "$ORION_ROOT"
CFG_OPTIONS=(
    "data.train.ann_file=$TRAIN_ANN_FILE"
    "data.workers_per_gpu=$WORKERS_PER_GPU"
    "log_config.interval=$LOG_INTERVAL"
)
if [ -n "$SAMPLES_PER_GPU" ]; then
    CFG_OPTIONS+=("data.samples_per_gpu=$SAMPLES_PER_GPU")
fi
if [ -n "$CHECKPOINTS_INTERVAL" ]; then
    # 关键调用点：显式 checkpoint iter 间隔统一走 runtime override，避免被 train.py 默认按 epoch 回填覆盖。
    CFG_OPTIONS+=("checkpoint_interval=$CHECKPOINTS_INTERVAL")
fi
if [ -n "$NUM_EPOCHS" ]; then
    CFG_OPTIONS+=("num_epochs=$NUM_EPOCHS")
fi

CUDA_VISIBLE_DEVICES="$GPU_IDS" \
PYTHONPATH="$ORION_ROOT:${PYTHONPATH:-}" \
"$PYTHON_BIN" -m torch.distributed.launch \
    --nproc_per_node="$GPUS_PER_NODE" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    --nnodes="$NNODES" \
    --node_rank="$NODE_RANK" \
    "$SCRIPT_DIR/train.py" \
    "$CFG_PATH" \
    --launcher pytorch \
    --deterministic \
    --work-dir "$WORK_DIR" \
    --cfg-options \
    "${CFG_OPTIONS[@]}" \
    "${@}" \
    2>&1 | tee "$WORK_DIR/logs/train.$T"
