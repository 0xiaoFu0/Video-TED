#!/usr/bin/env bash
# Official SEED-Bench Video eight-frame pack; same generation budget as videor1_64.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"
MODE="${1:-run}"
case "${MODE}" in
    preflight|smoke|full|run) ;;
    *) echo "Usage: bash eval/scripts/eval_seedbench.sh [preflight|smoke|full|run]" >&2; exit 2 ;;
esac
if [[ "${MODE}" != preflight ]]; then
    : "${MODEL_PATH:?Set MODEL_PATH to the model you want to evaluate}"
    : "${EXP_NAME:?Set a unique EXP_NAME, e.g. qwen25vl7b_seedbench8}"
fi
RUN_NAME="${EXP_NAME:-seedbench_check}"
export SEEDBENCH_DATA_DIR="${SEEDBENCH_DATA_DIR:-/gemini-3/space/zwj/LMUData/SEED-Bench}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
export BENCHMARKS=seedbench EVAL_PROFILE=seedbench_8
export BACKEND="${BACKEND:-vllm}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}" TP_SIZE="${TP_SIZE:-1}"
export THINK_MODE=1 N=1 TEMPERATURE=0.01 TOP_P=0.001 REPETITION_PENALTY=1.0
export MAX_NEW_TOKENS=1024 MAX_MODEL_LEN=16384
export FPS=2 MIN_FRAMES=4 MIN_PIXELS=100352 MAX_PIXELS=200704
export MAX_FRAMES=8 SEEDBENCH_MAX_FRAMES=8
export VOTE=majority_voting SEED=42 DTYPE=bfloat16 DECODER=decord
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}" TRUST_REMOTE_CODE=0
export SCORE_ONLY=0 RESUME=1 ALLOW_NONSTANDARD_COUNTS="${ALLOW_NONSTANDARD_COUNTS:-0}"

if [[ "${MODE}" == preflight || "${MODE}" == run ]]; then
    PREFLIGHT=1 LIMIT=0 VERIFY_FRAME_IMAGES=1 EXP_NAME="${RUN_NAME}_preflight" \
        bash "${SCRIPT_DIR}/eval_all.sh"
fi
if [[ "${MODE}" == smoke || "${MODE}" == run ]]; then
    PREFLIGHT=0 LIMIT=5 VERIFY_FRAME_IMAGES=1 EXP_NAME="${RUN_NAME}_smoke" \
        bash "${SCRIPT_DIR}/eval_all.sh"
    "${PYTHON_BIN:-python3}" - "${OUTPUT_DIR}/${RUN_NAME}_smoke/summary.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
stats = report["benchmarks"]["seedbench"]
if not report["complete"] or stats["n_done"] != stats["n_total"]:
    raise SystemExit("SEED-Bench smoke incomplete; inspect errors.json and worker logs")
if any(group["invalid_answers"] == group["n_total"] for group in stats["by_task"].values()):
    raise SystemExit("SEED-Bench smoke has a task with no parseable answers; inspect predictions before full run")
print("SEED-Bench smoke passed.")
PY
fi
if [[ "${MODE}" == full || "${MODE}" == run ]]; then
    PREFLIGHT=0 LIMIT=0 VERIFY_FRAME_IMAGES=0 EXP_NAME="${RUN_NAME}_full" \
        bash "${SCRIPT_DIR}/eval_all.sh"
fi
