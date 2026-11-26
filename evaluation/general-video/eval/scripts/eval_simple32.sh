#!/usr/bin/env bash
# Short reasoning prompt, same FPS-floor sampler, at most 32 frames.
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"
: "${MODEL_PATH:?Set MODEL_PATH to the model you want to evaluate}"
: "${EXP_NAME:?Set a new EXP_NAME for the simple prompt / 32-frame run}"
export BENCHMARKS=mvbench,videomme,tempcompass EVAL_PROFILE=simple_32
export MVBENCH_DATA_DIR="${MVBENCH_DATA_DIR:-/path/to/workspace/datasets/MVBench}"
export VIDEOMME_DATA_DIR="${VIDEOMME_DATA_DIR:-/gemini-3/space/zwj/LMUData/VideoMME}"
export TEMPCOMPASS_DATA_DIR="${TEMPCOMPASS_DATA_DIR:-/gemini-3/space/zwj/LMUData/TempCompass}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
export BACKEND="${BACKEND:-vllm}" CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}" TP_SIZE="${TP_SIZE:-1}"
export THINK_MODE=1 N=1 TEMPERATURE=0.01 TOP_P=0.001 REPETITION_PENALTY=1.0
export MAX_NEW_TOKENS=1024 MAX_MODEL_LEN=16384
export FPS=2 MIN_FRAMES=4 MIN_PIXELS=100352 MAX_PIXELS=200704
export MAX_FRAMES=32 MVBENCH_MAX_FRAMES=32 VIDEOMME_MAX_FRAMES=32 TEMPCOMPASS_MAX_FRAMES=32
export VOTE=majority_voting SEED=42 DTYPE=bfloat16 DECODER=decord
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}" TRUST_REMOTE_CODE=0
export PREFLIGHT=0 SCORE_ONLY=0 LIMIT=0 RESUME=1 ALLOW_NONSTANDARD_COUNTS=0 VERIFY_FRAME_IMAGES=0
exec bash "${SCRIPT_DIR}/eval_all.sh"
