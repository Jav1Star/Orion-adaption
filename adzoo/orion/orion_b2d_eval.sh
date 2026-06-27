#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 CHECKPOINT_PATH [GPU_RANK]" >&2
    exit 1
fi

CHECKPOINT_PATH=$1
GPU_RANK=${2:-0}
CARLA_GPU=${CARLA_GPU:-$GPU_RANK}
MODEL_GPU=${MODEL_GPU:-$GPU_RANK}

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
ORION_ROOT=$(cd -- "$SCRIPT_DIR/../.." && pwd)
B2D_ROOT="$ORION_ROOT/Bench2Drive"
LEADERBOARD_ROOT="$B2D_ROOT/leaderboard"
SCENARIO_RUNNER_ROOT="$B2D_ROOT/scenario_runner"
TEAM_AGENT="$ORION_ROOT/team_code/orion_b2d_agent.py"
ORION_ASSET_ROOT=${ORION_ASSET_ROOT:-/raid/yyj/Orion-adaption/Orion}
B2D_CKPT_DIR="$B2D_ROOT/ckpts"
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-$ORION_ROOT/adzoo/orion/configs/orion_stage3_agent.py}
TEAM_CONFIG="$AGENT_CONFIG_PATH+$CHECKPOINT_PATH"
ROUTES=${ROUTES:-$LEADERBOARD_ROOT/data/bench2drive220.xml}
CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:-$ORION_ROOT/bench2drive_eval/orion_eval.json}
SAVE_PATH=${SAVE_PATH:-$ORION_ROOT/bench2drive_eval/records}
PORT=${PORT:-30000}
TM_PORT=${TM_PORT:-50000}
IS_BENCH2DRIVE=${IS_BENCH2DRIVE:-True}
PLANNER_TYPE=${PLANNER_TYPE:-only_traj}
export CARLA_ROOT=${CARLA_ROOT:-/dataset/carla0915}
export CARLA_SERVER="$CARLA_ROOT/CarlaUE4.sh"
export SCENARIO_RUNNER_ROOT
export LEADERBOARD_ROOT
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-0}
export REPETITIONS=${REPETITIONS:-1}
export RESUME=${RESUME:-True}
export TEAM_AGENT
export TEAM_CONFIG
export CHECKPOINT_ENDPOINT
export SAVE_PATH
export ROUTES
export PORT
export TM_PORT
export IS_BENCH2DRIVE
export PLANNER_TYPE
export GPU_RANK
export CARLA_GPU
export MODEL_GPU
CARLA_EGG=$(find "$CARLA_ROOT/PythonAPI/carla/dist" -maxdepth 1 -name 'carla-*-py3*.egg' | head -n 1 || true)
export PYTHONPATH="$ORION_ROOT:$B2D_ROOT:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:${CARLA_EGG:+$CARLA_EGG:}$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
mkdir -p "$(dirname -- "$CHECKPOINT_ENDPOINT")" "$SAVE_PATH" "$B2D_CKPT_DIR"
if [ -d "$ORION_ASSET_ROOT/pretrain_qformer" ]; then
    ln -sfn "$ORION_ASSET_ROOT/pretrain_qformer" "$B2D_CKPT_DIR/pretrain_qformer"
elif [ -d "$ORION_ROOT/ckpts/pretrain_qformer" ]; then
    ln -sfn "$ORION_ROOT/ckpts/pretrain_qformer" "$B2D_CKPT_DIR/pretrain_qformer"
fi
cd "$B2D_ROOT"
PYTHON_BIN=${PYTHON_BIN:-/raid/yyj/miniconda3/envs/orion/bin/python}
CUDA_VISIBLE_DEVICES=${MODEL_GPU} "$PYTHON_BIN" "$LEADERBOARD_ROOT/leaderboard/leaderboard_evaluator.py"   --routes="$ROUTES"   --repetitions="$REPETITIONS"   --track="$CHALLENGE_TRACK_CODENAME"   --checkpoint="$CHECKPOINT_ENDPOINT"   --agent="$TEAM_AGENT"   --agent-config="$TEAM_CONFIG"   --debug="$DEBUG_CHALLENGE"   --record="${RECORD_PATH:-}"   --resume="$RESUME"   --port="$PORT"   --traffic-manager-port="$TM_PORT"   --gpu-rank="$CARLA_GPU"
