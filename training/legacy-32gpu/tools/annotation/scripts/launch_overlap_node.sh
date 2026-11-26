#!/usr/bin/env bash
# Incremental overlap annotation only; uses the existing resumable shard launcher.
# rank 0: 192.0.2.10, rank 1: 192.0.2.11. Both nodes share the run path.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# != 1 || ( "${1:-}" != 0 && "${1:-}" != 1 ) ]]; then
  echo '用法：bash scripts/launch_overlap_node.sh 0|1' >&2
  echo 'rank 0 = 192.0.2.10；rank 1 = 192.0.2.11' >&2
  exit 2
fi
case "${DRY_RUN:-0}" in 0|1) ;; *) echo 'DRY_RUN 必须为 0 或 1。' >&2; exit 2 ;; esac
rank="$1"
if [[ "$rank" == 0 ]]; then node_host=192.0.2.10; else node_host=192.0.2.11; fi
export CANDIDATES="${CANDIDATES:-$PWD/work/candidates_overlap_incremental/candidates.jsonl}"
export RUN_DIR="${RUN_DIR:-$PWD/work/full_overlap_16gpu}"
export CONFIG="${CONFIG:-$PWD/config.server.json}"
printf 'Overlap node rank %s = %s; shared run directory: %s\n' "$rank" "$node_host" "$RUN_DIR"

# This check uses only the standard library: an empty incremental pool must not
# activate the annotation environment or launch any model-loading workers.
candidate_count="$(python3 - "$CANDIDATES" <<'PY'
import json
import sys
try:
    count = 0
    with open(sys.argv[1], encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('sample_id'), str) or not row['sample_id']:
                raise ValueError(f'invalid candidate row at line {line_number}')
            count += 1
except (OSError, ValueError) as error:
    print(f'无法读取增量候选: {sys.argv[1]}: {error}', file=sys.stderr)
    sys.exit(2)
print(count)
PY
)"
if [[ "$candidate_count" == 0 ]]; then
  echo '增量候选为 0：没有需要补造的样本；未启动 worker、未激活注释环境、未加载模型。'
  exit 0
fi

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  local_ips="$(hostname -I)"
  if [[ " $local_ips " != *" $node_host "* ]]; then
    printf 'rank %s 要求节点 IP %s；当前 IP：%s\n' "$rank" "$node_host" "$local_ips" >&2
    exit 2
  fi
fi

export GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a overlap_gpus <<< "$GPU_IDS"
if [[ ${#overlap_gpus[@]} != 8 ]]; then
  echo '补造使用两节点共 16 个 shard；每节点 GPU_IDS 必须包含 8 个 GPU。' >&2
  exit 2
fi
# Override inherited pilot settings. Run provenance and resume behavior remain
# the existing launcher's responsibility; no retry-errors flag is added here.
export LIMIT_PER_DATASET=0 ROLLOUTS=4 MIN_ACCEPTED_ROLLOUTS=1
export PROMPT_PROFILE=tree_v1 GPUS_PER_WORKER=1 RETRY_ERRORS=0
printf 'Incremental candidates: %s; limit=0; rollouts=4; min-accepted=1; prompt=tree_v1\n' "$candidate_count"
exec bash scripts/launch_16gpu_node.sh "$rank"
