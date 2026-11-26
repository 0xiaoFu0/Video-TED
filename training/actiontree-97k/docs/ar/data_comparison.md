# Two training groups after merging two annotation pools (v3)

v3 merges the original pool of 11,964 annotations with the supplementary incremental pool of 7,698, totaling 19,662 planned examples. After excluding error entries from both pools, qualified uses 9,014 automatically verified CoTs; full uses 19,621 non-error generated CoTs, including 10,607 marked needs_review. Each group occupies two machines with 8 GPUs each, and trains independently from the original Qwen2.5-VL-7B-Instruct.

The current RLSD configuration is retained: 1 epoch, random seed 1, 16 videos per batch, 4 online rollouts per video, up to 16 frames, 2 FPS, and 768 generated tokens. The reference coefficient is 0.1, and unreviewed references are allowed. CoTs in full that did not pass verification also enter the teacher context and reference supervision; the final `<answer>` retains the original QA label, and complete verification results remain in the audit. SFT files support the corresponding video forward/backward checks and optional SFT experiments. The current workflow applies RLSD directly to the original model without prior SFT.

| Data group | Examples in file | Steps in 1 epoch | Examples consumed in that epoch | Incomplete final batch |
|---|---:|---:|---:|---:|
| qualified | 9,014 | 563 | 9,008 | 6 |
| full | 19,621 | 1,226 | 19,616 | 5 |

These step counts retain `drop_last=True`; the final remainder contains fewer than 16 examples after random shuffling, rather than reflecting a new quality filter. The two jobs have different step counts because their data sizes differ. Actual reference weights still follow the existing online-rollout difficulty calculation and may be zero when every response in a group is correct; needs_review status alone does not disable reference eligibility.

## 1. Upload and extract the independent v3 package

Upload `AR-training-comparison-v3.zip` to `/path/to/workspace/zwj/`. On `192.0.2.12`, run:

```bash
cd /path/to/workspace/zwj
unzip AR-training-comparison-v3.zip
cd AR-training-comparison-v3
sha256sum -c SHA256SUMS --quiet
```

The training project is `/path/to/workspace/zwj/AR-training-comparison-v3/AR-RLSD`. Extract once if all four machines share this directory. With non-shared storage, place the same package on all four machines, extract it to the same absolute path, and synchronize the new data directory after export. Models and original videos must also be readable through the same paths on all four machines.

v3 uses separate code, data, Ray ports, and output directories. Do not overwrite a running v2 project, run `ray stop`, or terminate other training, evaluation, or annotation processes.

## 2. Export the merged data once

Both annotation pools should have completed all 16 shards. Run the following on just one node that can access the original candidates, records, rollouts, and videos; shared directories do not need repeated exports. This step reads existing generated results, without loading a generation model or regenerating CoTs.

```bash
source /path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh
conda activate ar-qwen38-annotate
cd /path/to/workspace/zwj/AR-training-comparison-v3/tools/annotation

"$CONDA_PREFIX/bin/python" -E -s -m ar_annotate.merged_comparison \
  --pool old \
    /path/to/workspace/zwj/AR-Annotate-Qwen38-v4/work/candidates_all_v4/candidates.jsonl \
    /path/to/workspace/zwj/AR-Annotate-Qwen38-v4.2/work/full_all_16gpu_v42 \
  --pool incremental \
    /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2/work/candidates_overlap_incremental/candidates.jsonl \
    /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2/work/full_overlap_16gpu \
  --out /path/to/workspace/zwj/AR-training-comparison-v3/AR-RLSD/data/data_comparison_v3 \
  --student-model /path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --expect-planned 19662 --expect-qualified 9014 --expect-full 19621
```

The output directory must not already exist. On error, preserve the original directory and run records and inspect the specific count, hash, or missing-file issue. Old single-pool export directories cannot serve as v3 merged data.

```text
data/data_comparison_v3/
  manifest.json
  SHA256SUMS
  selection.audit.jsonl
  qualified/train.rlsd.think.jsonl
  qualified/train.sft.think.jsonl
  qualified/sample_ids.json
  full/train.rlsd.think.jsonl
  full/train.sft.think.jsonl
  full/sample_ids.json
```

After export, check counts and hashes:

```bash
cd /path/to/workspace/zwj/AR-training-comparison-v3/AR-RLSD/data/data_comparison_v3
sha256sum -c SHA256SUMS --quiet
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('manifest.json').read_text())
assert d['merge_schema_version'] == 1
assert d['source_counts']['old']['planned'] == 11964
assert d['source_counts']['incremental']['planned'] == 7698
assert (d['planned'], d['qualified_rows'], d['full_rows']) == (19662, 9014, 19621)
assert d['full_reference_policy'] == 'all_nonerror_generated_trees_with_reference_loss'
for group, expected in [('qualified', 9014), ('full', 19621)]:
    for kind in ('sft', 'rlsd'):
        path = Path(group) / f'train.{kind}.think.jsonl'
        actual = sum(bool(line.strip()) for line in path.open())
        assert actual == expected, (str(path), actual, expected)
        print(path, actual)
print('Shared reference_max_length:', d['reference_max_length'])
PY
```

The main manifest is `manifest.json`; `merge_schema_version=1` identifies the two-pool merged format, and `source_counts` distinguishes old from incremental. The launcher also checks both pool and group counts, actual SFT/RLSD line counts, every payload hash, consistency of paired samples and references, that all qualified entries are accepted, and that all full entries are non-error. It uses a shared `reference_max_length` covering every reference from both pools plus EOS, retaining longer needs_review entries rather than deleting examples or truncating references to the old 768-token budget. The student's online generation limit remains 768.

## 3. Initialize all four machines

| Node | Role | Data group | Ray address |
|---|---|---|---|
| 192.0.2.12 | A head | qualified, 9,014 examples | 192.0.2.12:6383 |
| 192.0.2.13 | A worker | qualified, 9,014 examples | Join A head |
| 192.0.2.10 | B head | full, 19,621 examples | 192.0.2.10:6384 |
| 192.0.2.11 | B worker | full, 19,621 examples | Join B head |

Run in all four terminals:

```bash
cd /path/to/workspace/zwj/AR-training-comparison-v3/AR-RLSD
source examples/action_recognition/comparison_nodes.env
unset RUN_NAME DRY_RUN
hostname -I
test -f "$TRAIN_ENV_ROOT/bin/activate"
"$TRAIN_ENV_ROOT/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() == 8'
nvidia-smi
```

`comparison_nodes.env` only exports addresses, v3 paths, the original model, the existing training environment, and expected counts; it starts no jobs. The default environment is `/path/to/workspace/zwj/.venvs/open-o3-visd-vllm`. If a node already has a compatible training venv elsewhere, explicitly override `TRAIN_ENV_ROOT` after sourcing the file. Do not confuse the data-generation conda environment with the training environment. Key dependencies remain Transformers 4.57.3, vLLM 0.11.0, qwen-vl-utils 0.0.14, torch/torchvision, Ray, decord, flash-attn, and related packages.

Each machine needs 8 idle GPUs. If `nvidia-smi` shows training, evaluation, or annotation processes still occupying them, report the node/process state and continue when resources become available. This workflow neither stops existing jobs nor automatically aborts on busy GPUs. If an existing Ray instance prevents a new head/worker from starting, preserve that instance and report the error.

You can first run a read-only preview. It still validates v3 data and reports missing files or count mismatches, but does not activate the GPU environment or start Ray:

```bash
DRY_RUN=1 bash examples/action_recognition/run_data_comparison.sh qualified train
DRY_RUN=1 bash examples/action_recognition/run_data_comparison.sh full train
```

## 4. Start the two Ray groups

Start each group's head first, then its worker after the head succeeds. A uses Ray 6383 and NCCL 29523; B uses Ray 6384 and NCCL 29524, different from v2's ports.

On `192.0.2.12`:

```bash
bash examples/action_recognition/run_data_comparison.sh qualified head
```

On `192.0.2.13`:

```bash
bash examples/action_recognition/run_data_comparison.sh qualified worker
```

On `192.0.2.10`:

```bash
bash examples/action_recognition/run_data_comparison.sh full head
```

On `192.0.2.11`:

```bash
bash examples/action_recognition/run_data_comparison.sh full worker
```

The existing launcher checks the head/worker's local IP. Each group connects only to its own Ray address. Once started, workers wait for tasks and do not run a training driver.

## 5. Check and train on the two heads

On `192.0.2.12`:

```bash
bash examples/action_recognition/run_data_comparison.sh qualified check-video && \
bash examples/action_recognition/run_data_comparison.sh qualified smoke \
  data.video_max_frames=16 data.max_response_length=768 worker.rollout.enforce_eager=false && \
bash examples/action_recognition/run_data_comparison.sh qualified train
```

On `192.0.2.10`:

```bash
bash examples/action_recognition/run_data_comparison.sh full check-video && \
bash examples/action_recognition/run_data_comparison.sh full smoke \
  data.video_max_frames=16 data.max_response_length=768 worker.rollout.enforce_eager=false && \
bash examples/action_recognition/run_data_comparison.sh full train
```

The two chains can run independently on their respective heads. `check-video` uses the corresponding SFT file for single-GPU video processing and model forward/backward, without an optimizer update. `smoke` checks each group's two nodes, 16 GPUs, code/data/model-configuration/dependency-version consistency, and NCCL all_reduce, then trains for two steps using the full configuration's 16 frames, 768 generated tokens, and CUDA graphs. A nonzero exit prevents the next command from running. Full `train` uses a new run name and starts a fresh 1-epoch run from the original model.

Distinguish the scope of these checks: check-video selects only the first example by default; increasing `CHECK_SAMPLES` adds preprocessing for the first video of each class, while backward still runs only on the first example. Two-step smoke covers only the sampled batches and does not establish GPU validation of every longer needs_review example or every real video. Full CPU reference-length statistics ensure the budget includes these trees; actual teacher/video prompts and GPU-memory requirements still depend on server results.

smoke/train run in the foreground while also saving logs. Use an existing tmux session and keep the training terminal running. Default paths:

```text
/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/qualified/logs/ar_qwen25_qualified_v3_smoke_timestamp.log
/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/qualified/logs/ar_qwen25_qualified_v3_train_timestamp.log
/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/full/logs/ar_qwen25_full_v3_smoke_timestamp.log
/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/full/logs/ar_qwen25_full_v3_train_timestamp.log
```

Checkpoints are under the matching run directory in each group's output root, for example `qualified/ar_qwen25_qualified_v3_train_timestamp/`. Timestamps use `YYYYmmdd_HHMMSS`. Smoke also saves a separate checkpoint at completion; full training does not automatically resume from it. Use the group's `cluster` mode to check the cluster separately.

These instructions describe launch steps; they do not mean server jobs have started. CPU launcher tests and dry runs do not replace actual video, NCCL, and GPU smoke results.
