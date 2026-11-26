# Local verification record

## 2026-10-30 Qwen2.5-VL-7B update

The update checks passed: **169 passed in 20.09 seconds**. The new registry, preflight, launcher, and real Bash-chain checks are recorded in [verification/pytest_qwen25_7b_cpu.txt](verification/pytest_qwen25_7b_cpu.txt). These update checks use the available local Python 3.9 environment and do not replace the target Python/PyTorch/Transformers GPU smoke. The historical full-suite records below retain their original environments and scope. See [QWEN25_7B_UPDATE.md](QWEN25_7B_UPDATE.md).

Date: 2026-10-23. Verification was performed in the new project `AR/AR-MultiModel-97k`; no training-cluster connections or server jobs were started.

## 2026-10-27 connection-update regression

Full CPU regression: **540 passed, 94 subtests passed, 43.53 seconds; no failures or skips**. The command and environment are the same as in the initial record below. See [verification/pytest_connection_cpu.txt](verification/pytest_connection_cpu.txt) for the latest log.

Added coverage includes SSH key-permission copies and foreground authentication, failure without a terminal, private-agent ownership/cleanup, per-node logs, nine-stage diagnostics, stdin and legacy Base64 entry points, stderr notices in the synchronization protocol, and blocking background submission after authentication failure. Actual local Bash execution verifies paths containing special characters, unchanged JSON transfer, and environment setup before Python starts. A real timeout test verifies that Python exits with its launching process. SSH/cluster operations still use test doubles; no server connections or GPUs were used.

Comparison with the original project confirms unchanged AR reward, format protocol, 97k data validation, and training loop. The connection patch excludes `configs/cluster.json`. A simulated extraction of the old ZIP followed by applying the patch verified that it produces the updated code while preserving user configuration and existing output files.

## Initial full CPU regression

Result: **480 passed, 94 subtests passed, 40.68 seconds; no failures or skips**.

```bash
AR_ANNOTATION_SOURCE=/Users/zwj/Desktop/CVPR/AR/AR-action-training-32gpu-v3/tools/annotation \
QWEN3VL_PROCESSOR_PATH=/private/tmp/trsi-qwen3-official-processor \
/private/tmp/trsi-qwen3-venv/bin/python -B -m pytest -q tests
```

The log is saved in [verification/pytest_cpu.txt](verification/pytest_cpu.txt). CPU environment: Python 3.12, torch 2.8.0, torchvision 0.23.0, transformers 4.57.3, qwen-vl-utils 0.0.14, datasets 4.4.1, huggingface-hub 0.36.2, and tensordict 0.10.0. The pip dependency-consistency check passed.

`AR_ANNOTATION_SOURCE` is used only by one inherited integration test for the old annotation pipeline; it is not a dependency of the new training entry point. That test explicitly skips when the variable is absent. `QWEN3VL_PROCESSOR_PATH` points to existing official processor assets without model weights. Three real-asset tests skip when it is absent; the other synthetic-processor/small-model tests still run.

## Coverage

- Architecture/dimension, weight-shard/header, tokenizer/processor/template checks for all three models, including invalid-input rejection.
- Actual execution of the new Bash entry-point chain with a CPU recorder to check final training arguments for all three models, 97k paths, separate reports, fixed MeSD objectives, two-step smoke tests, and full-training save_freq=100. Failed preflight checks prevent the trainer from starting.
- Failure blocking at each launcher stage, exactly one full-training invocation, model/path isolation, prevention of duplicate launches overwriting active-run records, process-identity matching, protection of unrelated Ray services, and cleanup after interrupted startup or partial process exit. SSH/Ray operations use test doubles; the four nodes were not accessed.
- Actual tiny-Qwen3 forward/backward execution, DeepStack gradients, and comparison with HF mRoPE; regression of the original Qwen2.5 video path; image/video and teacher-input replay using the actual official Qwen3 processor.
- Synchronized encoder-cache cleanup. Incompatible vLLM cache layouts or unfinished requests raise errors to prevent use of features from stale visual weights.
- Actual tiny-Qwen2.5/Qwen3 checkpoint merging, HF saving/reloading, and per-tensor numeric comparisons; legacy key conversion, different sharding dimensions/mesh orders, replicated tensors, missing shards, changing checkpoints, and output-overwrite protection.
- Regression of the original AR reward, reference objective, teacher schedule, and data protocol.

New scripts passed Python syntax/undefined-name checks and Bash syntax checks. `bash run.sh models` and `plan` commands for all three models were run locally; no training commands were executed.

## Unchanged components and pending server verification

File-by-file comparison against the source project confirms that `verl/ar/reward.py`, `verl/ar/response_format.py`, `verl/ar/check_merged_97430_data.py`, and `verl/trainer/opsd_trainer.py` are byte-identical. The new project does not introduce Video-t-RSI STGR data, RSI loops, or other reward objectives.

Local results do not establish that full 3B/4B/8B weights execute correctly on the target GPUs, nor do they establish four-node CUDA/NCCL/vLLM compatibility, actual memory/throughput, or training convergence. For the selected model, the server launcher runs the above preflight checks, a 32-rank NCCL check, and two training steps with the full budget before starting the full experiment. No actual benchmark evaluation was run.
