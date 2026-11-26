#!/usr/bin/env bash
# Run once on each node, rank 0 on 192.0.2.10 and rank 1 on 192.0.2.11.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ "${1:-}" != 0 && "${1:-}" != 1 ]]; then
  echo 'Usage: bash scripts/launch_16gpu_node.sh 0|1' >&2
  exit 2
fi
export NODE_COUNT=2 NODE_RANK="$1"
export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
export GPUS_PER_WORKER=1
export CONFIG="${CONFIG:-$PWD/config.server.json}"
export CANDIDATES="${CANDIDATES:-/path/to/workspace/zwj/AR-Annotate-Qwen38-v4/work/candidates_all_v4/candidates.jsonl}"
export RUN_DIR="${RUN_DIR:-$PWD/work/pilot_all_16gpu_v42}"
export LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-10}"
export ROLLOUTS="${ROLLOUTS:-4}"
export MIN_ACCEPTED_ROLLOUTS="${MIN_ACCEPTED_ROLLOUTS:-1}"
export PROMPT_PROFILE=tree_v1
printf 'Node rank %s; shared run directory: %s\n' "$NODE_RANK" "$RUN_DIR"
exec bash scripts/launch_annotation.sh
