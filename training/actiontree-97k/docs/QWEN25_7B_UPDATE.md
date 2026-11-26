# Qwen2.5-VL-7B Update

Date: 2026-10-30

This release extends the existing fast project with `qwen25-7b`, selecting `Qwen/Qwen2.5-VL-7B-Instruct`. The metadata is checked against the [official model configuration](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/raw/main/config.json). Its text model uses hidden size 3584, intermediate size 18944, 28 layers, 28 attention heads, 4 KV heads, and vocabulary size 152064. Its vision output size is 3584.

Formal runs save a complete checkpoint every 100 steps, retain at most 30 checkpoints, and save the final step. Smoke runs remain two steps. Existing fast settings, data, MeSD objective, and schedules are retained.

## Upgrade the deployed fast project

Upload `AR-MultiModel-97k-qwen25-7b-update.zip` to `/path/to/workspace/zwj/` on head node `192.0.2.12`. Apply the update after the previous training task has finished and released the cluster:

```bash
cd /path/to/workspace/zwj
unzip -o AR-MultiModel-97k-qwen25-7b-update.zip -d /path/to/workspace/zwj
cd /path/to/workspace/zwj/AR-MultiModel-97k
bash run.sh models
```

The update archive includes changed source, tests, documentation, and package metadata. It does not contain `configs/cluster.json`, training data, model weights, work reports, or outputs.

The default model path must already exist on every node:

```text
/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
```

For another location, set the existing cluster configuration's `model_paths.qwen25-7b`. The launcher does not download model weights.

Start on the head node:

```bash
bash run_fast.sh train qwen25-7b && bash run_fast.sh logs qwen25-7b
```

The existing four-node checks, model/data/video checks, NCCL check, and two-step smoke run before fresh formal training. The default output directory is `/gemini-3/space/zwj/AR-MultiModel-97k/outputs/qwen25-7b/<RUN_NAME>/`. Checkpoints remain FSDP training checkpoints; use `bash run.sh trans /path/to/global_step_N` when an independent Hugging Face export is needed.

## Verification boundary

The new profile is checked locally against synthetic metadata and shard fixtures, including flat/nested Qwen2.5 config layouts and wrong-size rejection. CPU recording substitutes exercise the actual Bash chain and check the final arguments for the selected model, save frequency 100, retention 30, full checkpoint state, and fast micro-batch/HSDP overrides. The test helper runs the actual fast entry instead of extracting shell source through process substitution.

Local checks: **169 passed in 20.09 seconds** across model profiles, preflight, launch orchestration, the actual shell chain, transport, and node diagnostics. See `verification/pytest_qwen25_7b_cpu.txt`. Offline construction from the official 7B config also passed with PyTorch 2.8.0 and Transformers 4.57.3: all parameters remained on the meta device and 152064 x 3584 input/output embeddings remained untied. See `verification/qwen25_7b_meta.json`. The older full-suite logs describe their historical release and environment. No complete 7B model, remote SSH connection, or GPU job was run to prepare this update; target-cluster smoke is still required.
