#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 ROUTE_RES_DIR [PYTHON_BIN]" >&2
    echo "Example: $0 /raid/yyj/Orion-adaption/bench2drive_eval/routes/res" >&2
    exit 1
fi

SCRIPT_DIR=$(cd -- "$(dirname -- "$0")" && pwd)
ORION_ROOT="$SCRIPT_DIR"
B2D_ROOT="$ORION_ROOT/Bench2Drive"
LEADERBOARD_ROOT="$B2D_ROOT/leaderboard"
SCENARIO_RUNNER_ROOT="$B2D_ROOT/scenario_runner"
MERGE_SCRIPT="$B2D_ROOT/tools/merge_route_json.py"

ROUTE_RES_DIR=$1
PYTHON_BIN=${2:-${PYTHON_BIN:-python}}

if [[ "$ROUTE_RES_DIR" != /* ]]; then
    ROUTE_RES_DIR="$PWD/$ROUTE_RES_DIR"
fi
ROUTE_RES_DIR=$(cd -- "$ROUTE_RES_DIR" && pwd)

if [ ! -d "$ROUTE_RES_DIR" ]; then
    echo "Route result directory does not exist: $ROUTE_RES_DIR" >&2
    exit 1
fi

if [ ! -f "$MERGE_SCRIPT" ]; then
    echo "merge_route_json.py not found: $MERGE_SCRIPT" >&2
    exit 1
fi

shopt -s nullglob
json_files=("$ROUTE_RES_DIR"/*.json)
shopt -u nullglob
if [ ${#json_files[@]} -eq 0 ]; then
    echo "No json files found under: $ROUTE_RES_DIR" >&2
    exit 1
fi

export ORION_ROOT
export B2D_ROOT
export LEADERBOARD_ROOT
export SCENARIO_RUNNER_ROOT
export PYTHONPATH="$ORION_ROOT:$B2D_ROOT:$LEADERBOARD_ROOT:$SCENARIO_RUNNER_ROOT:${PYTHONPATH:-}"

cd "$ORION_ROOT"
echo "Merging route json files from: $ROUTE_RES_DIR"
"$PYTHON_BIN" "$MERGE_SCRIPT" -f "$ROUTE_RES_DIR"
echo "Merged output written to: $ROUTE_RES_DIR/merged.json"
