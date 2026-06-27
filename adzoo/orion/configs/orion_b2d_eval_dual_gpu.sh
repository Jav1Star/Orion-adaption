#!/usr/bin/env bash

GPU_RANK_LIST=${GPU_RANK_LIST:-0,1}
CARLA_GPU=${CARLA_GPU:-0,1}
MODEL_GPU=${MODEL_GPU:-0,1}

PORT=${PORT:-30000}
TM_PORT=${TM_PORT:-50000}

CHECKPOINT_ENDPOINT=${CHECKPOINT_ENDPOINT:-/home/yyj/Orion/bench2drive_eval/orion_eval.json}
SAVE_PATH=${SAVE_PATH:-/home/yyj/Orion/bench2drive_eval/records}
ORION_EVAL_VISUALIZATION=${ORION_EVAL_VISUALIZATION:-false}

ROUTES=${ROUTES:-/home/yyj/Orion/Bench2Drive/leaderboard/data/bench2drive220.xml}
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-/home/yyj/Orion/adzoo/orion/configs/orion_stage3_agent.py}

CARLA_ROOT=${CARLA_ROOT:-/home/yyj/data/carla0915}
PYTHON_BIN=${PYTHON_BIN:-/home/yyj/miniconda3/envs/orion/bin/python}

DEBUG_CHALLENGE=${DEBUG_CHALLENGE:-0}
REPETITIONS=${REPETITIONS:-1}
RESUME=${RESUME:-True}

# Optional: uncomment to bind a default checkpoint to this config.
EVAL_CHECKPOINT_PATH=/home/yyj/Orion/ckpts/Orion_full/Orion.pth
