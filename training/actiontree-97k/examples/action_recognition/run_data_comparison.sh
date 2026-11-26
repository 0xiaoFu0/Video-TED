#!/usr/bin/env bash
# v3 merged pools: group A trains qualified CoT, group B trains all generated CoT.
# Each group uses the unchanged Qwen2.5 RLSD launcher/configuration (2 x 8 GPUs).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PROJECT_DIR="$PWD"

usage() {
  cat <<'EOF'
Usage: bash examples/action_recognition/run_data_comparison.sh {qualified|full} {head|worker|cluster|smoke|train|check-video} [extra arguments]
Set identical GROUP_A_HEAD, GROUP_A_WORKER, GROUP_B_HEAD, and GROUP_B_WORKER values in all four terminals (four distinct IPv4 addresses).
qualified uses group A; full uses group B. Start the head and worker for each group, then run cluster/smoke/train on that group head.
EXPERIMENT_DATA_ROOT defaults to data/data_comparison_v3 under the project; qualified/full logs and checkpoints are stored separately under OUTPUT_ROOT.
Each mode validates the merged version, both pool sizes, file hashes, reference eligibility, and the shared reference_max_length.
DRY_RUN=1 prints commands without starting Ray, training, or GPU checks.
For both groups, check-video runs the existing single-GPU forward/backward check on the corresponding SFT data.
EOF
}

variant="${1:-}"
mode="${2:-}"
if [[ "$variant" == -h || "$variant" == --help ]]; then usage; exit 0; fi
case "$variant" in qualified|full) ;; *) usage >&2; exit 2 ;; esac
case "$mode" in head|worker|cluster|smoke|train|check-video) ;; *) usage >&2; exit 2 ;; esac
shift 2

host_names=(GROUP_A_HEAD GROUP_A_WORKER GROUP_B_HEAD GROUP_B_WORKER)
hosts=()
for host_name in "${host_names[@]}"; do
  host="${!host_name:-}"
  if [[ -z "$host" ]]; then
    echo "Missing ${host_name}; enter the actual IPv4 address of that machine." >&2
    exit 2
  fi
  if [[ ! "$host" =~ ^(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})\.(0|[1-9][0-9]{0,2})$ ]]; then
    echo "$host_name must be an IPv4 address: $host" >&2
    exit 2
  fi
  IFS=. read -r -a octets <<< "$host"
  for octet in "${octets[@]}"; do
    if (( octet > 255 )); then echo "$host_name has an invalid IPv4 address: $host" >&2; exit 2; fi
  done
  # Bash 3.2 with nounset treats an empty array expansion as unbound.
  for existing in ${hosts[@]+"${hosts[@]}"}; do
    if [[ "$host" == "$existing" ]]; then
      echo "The four machine addresses must be distinct; duplicate address: $host" >&2
      exit 2
    fi
  done
  hosts+=("$host")
done

case "${DRY_RUN:-0}" in 0|1) ;; *) echo "DRY_RUN must be 0 or 1." >&2; exit 2 ;; esac
data_root="${EXPERIMENT_DATA_ROOT:-$PROJECT_DIR/data/data_comparison_v3}"
output_root="${OUTPUT_ROOT:-/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3}"
[[ "$data_root" == /* ]] || data_root="$PROJECT_DIR/$data_root"
[[ "$output_root" == /* ]] || output_root="$PROJECT_DIR/$output_root"

if [[ "$variant" == qualified ]]; then
  head_host="$GROUP_A_HEAD"
  worker_host="$GROUP_A_WORKER"
  ray_port=6383
  nccl_port=29523
else
  head_host="$GROUP_B_HEAD"
  worker_host="$GROUP_B_WORKER"
  ray_port=6384
  nccl_port=29524
fi
train_data="$data_root/$variant/train.rlsd.think.jsonl"
sft_data="$data_root/$variant/train.sft.think.jsonl"
run_name="${RUN_NAME:-ar_qwen25_${variant}_v3_${mode}_$(date +%Y%m%d_%H%M%S)}"
run_env=(
  "TRAIN_ENV_ROOT=${TRAIN_ENV_ROOT:-/path/to/workspace/zwj/.venvs/open-o3-visd-vllm}"
  "MODEL_PATH=${MODEL_PATH:-/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct}"
  "TRAIN_DATA=$train_data"
  "SFT_DATA=$sft_data"
  "OUTPUT_ROOT=$output_root/$variant"
  "RUN_NAME=$run_name"
  "HEAD_HOST=$head_host"
  "WORKER_HOST=$worker_host"
  "RAY_PORT=$ray_port"
  "AR_NCCL_PORT=$nccl_port"
)

# Read-only checks run even for head/worker and DRY_RUN. A v2 export or a partial
# transfer cannot accidentally start the v3 cluster. Production counts are
# pinned by comparison_nodes.env; tests can supply their own smaller fixtures.
reference_max_length="$(python3 - "$data_root/manifest.json" "$data_root" \
  "${AR_EXPECT_OLD_PLANNED:-11964}" "${AR_EXPECT_INCREMENTAL_PLANNED:-7698}" \
  "${AR_EXPECT_QUALIFIED_ROWS:-9014}" "${AR_EXPECT_FULL_ROWS:-19621}" <<'PY'
import hashlib
import json
from collections import Counter
from pathlib import Path, PurePosixPath
import re
import sys


def read_rows(path):
    with open(path, encoding='utf-8') as stream:
        values = [json.loads(line) for line in stream if line.strip()]
    if not values or any(not isinstance(row, dict) for row in values):
        raise ValueError('empty or invalid training dataset: ' + path)
    indexed = {}
    for row in values:
        sid = row['metadata']['sample_id']
        if not isinstance(sid, str) or not sid or sid in indexed:
            raise ValueError('missing or duplicate sample_id: ' + path)
        indexed[sid] = row
    return indexed


def count(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(name + ' must be an integer >= ' + str(minimum))
    return value


def statuses(value):
    if not isinstance(value, dict) or set(value) - {'accepted', 'needs_review', 'error'}:
        raise ValueError('invalid annotation statuses')
    return {name: count(value.get(name, 0), 'status ' + name)
            for name in ('accepted', 'needs_review', 'error')}


def check_pairs(rlsd, sft, variant):
    if rlsd.keys() != sft.keys():
        raise ValueError('RLSD/SFT sample IDs differ: ' + variant)
    observed = Counter()
    for sid, row in rlsd.items():
        status = row['metadata']['annotation_status']
        if status not in ({'accepted'} if variant == 'qualified' else {'accepted', 'needs_review'}):
            raise ValueError('invalid ' + variant + ' annotation_status: ' + sid)
        observed[status] += 1
        reference = row['reference_response']
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError('missing generated CoT reference: ' + sid)
        if row['reference_status'] not in ('unreviewed', 'reviewed', 'verified'):
            raise ValueError('reference is ineligible for CoT supervision: ' + sid)
        count(row['metadata']['reference_tokens_including_eos'], 'reference token count', 1)
        context = row['teacher_context']
        if not isinstance(context, str) or not context.strip():
            raise ValueError('missing CoT teacher context: ' + sid)
        pair = sft[sid]
        messages = pair['messages']
        if (not isinstance(messages, list) or len(messages) != 2
                or messages[0] != dict(role='user', content=row['prompt'])
                or messages[1] != dict(role='assistant', content=reference)
                or pair['answer'] != row['answer'] or pair['videos'] != row['videos']
                or pair['metadata'] != row['metadata']):
            raise ValueError('RLSD/SFT pair mismatch: ' + sid)
    return observed


try:
    with open(sys.argv[1], encoding='utf-8') as stream:
        manifest = json.load(stream)
    if type(manifest['merge_schema_version']) is not int or manifest['merge_schema_version'] != 1:
        raise ValueError('merge_schema_version must be 1; export both pools with the v3 builder')
    if manifest['full_reference_policy'] != 'all_nonerror_generated_trees_with_reference_loss':
        raise ValueError('full_reference_policy is not the all-generated-CoT policy; re-export comparison data')
    if manifest['reference_answer_policy'] != 'original_qa_label' or manifest['error_policy'] != 'omit_without_retry':
        raise ValueError('v3 requires original QA labels and omission of error samples')
    expected_old, expected_incremental, expected_qualified, expected_full = [
        count(int(value), 'configured expected count', 1) for value in sys.argv[3:7]]
    expected_planned = expected_old + expected_incremental
    sources = manifest['source_counts']
    if not isinstance(sources, dict) or set(sources) != {'old', 'incremental'}:
        raise ValueError('source_counts must contain exactly old and incremental')
    combined = Counter()
    for name, expected in [('old', expected_old), ('incremental', expected_incremental)]:
        source = sources[name]
        planned = count(source['planned'], name + ' planned', 1)
        source_statuses = statuses(source['statuses'])
        if planned != expected or sum(source_statuses.values()) != planned:
            raise ValueError('source planned/status count mismatch: ' + name)
        if (count(source['qualified_rows'], name + ' qualified_rows') != source_statuses['accepted']
                or count(source['full_rows'], name + ' full_rows') != planned - source_statuses['error']):
            raise ValueError('source qualified/full count mismatch: ' + name)
        combined.update(source_statuses)
    if (count(manifest['planned'], 'planned', 1) != expected_planned
            or count(manifest['qualified_rows'], 'qualified_rows', 1) != expected_qualified
            or count(manifest['full_rows'], 'full_rows', 1) != expected_full
            or statuses(manifest['statuses']) != dict(combined)
            or combined['accepted'] != expected_qualified
            or expected_planned - combined['error'] != expected_full):
        raise ValueError('merged planned/qualified/full/status count mismatch')
    if 'expected_counts' in manifest:
        for name, expected in [('planned', expected_planned), ('qualified', expected_qualified), ('full', expected_full)]:
            declared = manifest['expected_counts'][name]
            if declared is not None and (type(declared) is not int or declared != expected):
                raise ValueError('manifest expected_counts mismatch: ' + name)
    data_root = Path(sys.argv[2])
    hashes = manifest['output_sha256']
    payloads = {f'{variant}/train.{kind}.think.jsonl' for variant in ('qualified', 'full') for kind in ('rlsd', 'sft')}
    if not isinstance(hashes, dict) or not payloads <= hashes.keys():
        raise ValueError('output_sha256 is missing training files')
    for name, expected in hashes.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('invalid output_sha256 entry: ' + name)
        actual = hashlib.sha256()
        with open(data_root / relative, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                actual.update(chunk)
        if actual.hexdigest() != expected:
            raise ValueError('output SHA256 mismatch: ' + name)
    groups = {}
    for variant, expected in [('qualified', expected_qualified), ('full', expected_full)]:
        rlsd = read_rows(data_root / variant / 'train.rlsd.think.jsonl')
        sft = read_rows(data_root / variant / 'train.sft.think.jsonl')
        if len(rlsd) != expected or len(sft) != expected:
            raise ValueError('actual RLSD/SFT row count mismatch: ' + variant)
        observed = check_pairs(rlsd, sft, variant)
        if observed['accepted'] != expected_qualified or observed['needs_review'] != (expected_full - expected_qualified if variant == 'full' else 0):
            raise ValueError('actual annotation_status count mismatch: ' + variant)
        groups[variant] = rlsd
    accepted_ids = {sid for sid, row in groups['full'].items() if row['metadata']['annotation_status'] == 'accepted'}
    if groups['qualified'].keys() != accepted_ids:
        raise ValueError('qualified IDs differ from accepted full IDs')
    if any(row != groups['full'][sid] for sid, row in groups['qualified'].items()):
        raise ValueError('qualified/full shared reference mismatch')
    observed_max = max(row['metadata']['reference_tokens_including_eos'] for row in groups['full'].values())
    value = count(manifest['reference_max_length'], 'reference_max_length', 1)
    if (count(manifest['max_observed_reference_tokens'], 'max_observed_reference_tokens', 1) != observed_max
            or value != max(1024, observed_max)):
        raise ValueError('shared reference_max_length differs from all-reference token counts')
except (OSError, ValueError, KeyError, TypeError) as error:
    print(f'Comparison training-data validation failed: {sys.argv[1]}: {error}', file=sys.stderr)
    sys.exit(2)
print(value)
PY
)"
if [[ "$mode" == smoke || "$mode" == train ]]; then
  # Preserve the original positive coefficient and make unreviewed generated
  # CoT eligible in both groups, including needs_review rows in the full group.
  set -- "$@" "opsd.ar_reference_max_length=$reference_max_length" \
    opsd.ar_allow_unreviewed_reference=true opsd.ar_reference_coef=0.1
fi

command=(bash "$PROJECT_DIR/examples/action_recognition/run_qwen25.sh" "$mode" "$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  LC_ALL=C printf '%q ' env "${run_env[@]}" "${command[@]}"
  printf '\n'
  exit 0
fi

case "$mode" in
  cluster|smoke|train|check-video)
    if [[ ! -s "$train_data" ]]; then
      echo "Missing or empty $variant RLSD data: $train_data" >&2
      exit 1
    fi
    if [[ "$mode" == check-video && ! -s "$sft_data" ]]; then
      echo "Missing or empty $variant SFT data: $sft_data" >&2
      exit 1
    fi
    ;;
esac
exec env "${run_env[@]}" "${command[@]}"
