#!/usr/bin/env bash
# Paired comparison: shared protocol, separate resumable experiment directories.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_DIR}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct}"
OURS_MODEL_PATH="${OURS_MODEL_PATH:-${MODEL_PATH:-}}"
: "${OURS_MODEL_PATH:?Set OURS_MODEL_PATH to the merged actor/huggingface checkpoint}"
EXP_PREFIX="${EXP_PREFIX:-ar_step2924_vs_qwen25vl7b_videor1_64}"
export BACKEND="${BACKEND:-vllm}"
export BENCHMARKS="${BENCHMARKS:-mvbench,videomme}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/outputs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NUM_GPUS="${NUM_GPUS:-8}" TP_SIZE="${TP_SIZE:-1}"
export PREFLIGHT=0 SCORE_ONLY=0 RESUME=1 ALLOW_NONSTANDARD_COUNTS="${ALLOW_NONSTANDARD_COUNTS:-0}"

# Explicit exports prevent a previous run's generic settings leaking into this
# comparison. The named profile also rejects conflicting CLI/env settings when
# used directly through eval_all.sh.
export EVAL_PROFILE=videor1_64 THINK_MODE=1 N=1
export TEMPERATURE=0.01 TOP_P=0.001 REPETITION_PENALTY=1.0
export MAX_NEW_TOKENS=1024 MAX_MODEL_LEN=16384
export FPS=2 MIN_FRAMES=4 MIN_PIXELS=100352 MAX_PIXELS=200704
export MAX_FRAMES=64 MVBENCH_MAX_FRAMES=64 VIDEOMME_MAX_FRAMES=64 TEMPCOMPASS_MAX_FRAMES=64
export VOTE=majority_voting SEED=42 DTYPE=bfloat16 DECODER=decord

# Both local checkpoints must have exported weights; config/tokenizer files
# alone are insufficient. This check runs before any CUDA initialization.
"${PYTHON_BIN:-python3}" - "${BASE_MODEL_PATH}" "${OURS_MODEL_PATH}" <<'PY'
from pathlib import Path
import json
import sys
for label, value in zip(("base", "ours"), sys.argv[1:]):
    root = Path(value).expanduser()
    weights = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
    if not (root / "config.json").is_file() or not any(p.is_file() and p.stat().st_size > 0 for p in weights):
        raise SystemExit(f"Missing exported {label} model weights or config.json: {root}")
    model_type = json.loads((root / "config.json").read_text()).get("model_type")
    if model_type != "qwen2_5_vl":
        raise SystemExit(f"Expected Qwen2.5-VL for {label}, got model_type={model_type!r}: {root}")
    print(f"{label} checkpoint: {root}", flush=True)
PY

check_smoke() {
    "${PYTHON_BIN:-python3}" - "${OUTPUT_DIR}/$1/summary.json" <<'PY'
from pathlib import Path
import json
import sys
path = Path(sys.argv[1])
report = json.loads(path.read_text())
if report.get('complete') is not True:
    raise SystemExit(f"Incomplete smoke evaluation: {path}")
for name, stats in report['benchmarks'].items():
    total = stats['n_total']
    if total <= 0 or stats['n_done'] != total or stats['invalid_answers'] >= total:
        raise SystemExit(f"Smoke failed: {name} has incomplete or entirely unparseable predictions: {path}")
PY
}

MODEL_PATH="${BASE_MODEL_PATH}" EXP_NAME="${EXP_PREFIX}_base_smoke" LIMIT=5 \
    bash "${SCRIPT_DIR}/eval_all.sh"
check_smoke "${EXP_PREFIX}_base_smoke"
MODEL_PATH="${OURS_MODEL_PATH}" EXP_NAME="${EXP_PREFIX}_ours_smoke" LIMIT=5 \
    bash "${SCRIPT_DIR}/eval_all.sh"
check_smoke "${EXP_PREFIX}_ours_smoke"

MODEL_PATH="${BASE_MODEL_PATH}" EXP_NAME="${EXP_PREFIX}_base_full" LIMIT=0 \
    bash "${SCRIPT_DIR}/eval_all.sh"
MODEL_PATH="${OURS_MODEL_PATH}" EXP_NAME="${EXP_PREFIX}_ours_full" LIMIT=0 \
    bash "${SCRIPT_DIR}/eval_all.sh"

if [[ "${BACKEND}" == mock ]]; then
    "${PYTHON_BIN:-python3}" -m general_video_eval.compare \
        "${OUTPUT_DIR}/${EXP_PREFIX}_base_full" \
        "${OUTPUT_DIR}/${EXP_PREFIX}_ours_full" --allow-mock
else
    "${PYTHON_BIN:-python3}" -m general_video_eval.compare \
        "${OUTPUT_DIR}/${EXP_PREFIX}_base_full" \
        "${OUTPUT_DIR}/${EXP_PREFIX}_ours_full"
fi
