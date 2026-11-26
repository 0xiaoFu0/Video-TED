# AR-MultiModel-97k

Qwen2.5-VL-7B update (2026-10-30): use `bash run_fast.sh train qwen25-7b`; formal runs save every 100 steps and retain at most 30 checkpoints. See [upgrade instructions](docs/QWEN25_7B_UPDATE.md).

An independent multi-model project based on the local `AR-action-training-32gpu-v3/AR-RLSD` project. It preserves the original 97,430 action-recognition examples, tree-structured reference answers, AR protocol reward, and MeSD training objective, and provides single-command launch from the head node and checkpoint conversion to Hugging Face. The model list uses Qwen3-VL **4B**, as confirmed by the user.

| Launch name | Initial model |
| --- | --- |
| `qwen25-3b` | Qwen/Qwen2.5-VL-3B-Instruct |
| `qwen25-7b` | Qwen/Qwen2.5-VL-7B-Instruct |
| `qwen3-4b` | Qwen/Qwen3-VL-4B-Instruct |
| `qwen3-8b` | Qwen/Qwen3-VL-8B-Instruct |

## Launch

Place the **entire new project** on the head node, for example at `/path/to/workspace/zwj/AR-MultiModel-97k`. The head node must be able to SSH non-interactively to the three worker nodes using `/root/.ssh/id_rsa`. Keep the project at the same path on all four nodes. Shared mounts are used directly; otherwise, the launcher synchronizes the project source automatically. Models, original videos, and 97k data must already be readable at their configured locations on all four nodes; the launcher does not download them.

Synchronization does not delete remote files. If removed source files remain remotely after an upgrade, the digest check will fail. Deploy the same new project on all four nodes before launching again.

First check [configs/cluster.json](configs/cluster.json) once: the four node IPs, virtual environment, model root, 97k data path, and output directory. Defaults reuse the four nodes and existing environment you provided; the new project uses separate Ray/NCCL ports. If a model is outside the default directory, fill in `model_paths`, for example:

```json
"model_paths": {"qwen3-8b": "/your/models/Qwen3-VL-8B-Instruct"}
```

On the **192.0.2.12 head node**, enter the new project and choose one of these model commands:

```bash
cd /path/to/workspace/zwj/AR-MultiModel-97k
bash run.sh train qwen25-3b
# Or: bash run.sh train qwen25-7b
# Or: bash run.sh train qwen3-4b
# Or: bash run.sh train qwen3-8b
```

For the shortest wall-clock time with the same experiment, use the fast entry point instead; it runs the identical pipeline and contract but packs each GPU's four trajectories into one forward, shards FSDP within a node, and tokenizes each teacher prompt once (see [docs/FAST_TRAINING.md](docs/FAST_TRAINING.md)):

```bash
bash run_fast.sh train qwen3-4b
```

Run one model at a time on this four-node cluster. To switch models, wait for the previous run to finish, then change the final argument. Each run starts fresh from the selected base model and does not load another model's checkpoint. The pipeline runs in the background automatically; no separate `nohup` or per-node head/worker commands are needed.

Connection handling follows `Video-t-RSI-v8/scripts/video_trsi_cluster.sh`: SSH to all three workers is checked before background submission. First connections use `accept-new`; changes to a previously recorded host key are still rejected. If the configured key permissions are too broad, a temporary local copy with mode 0600 is used. An encrypted key can be unlocked by entering its passphrase once when prompted in an interactive terminal. Set `ssh_key` to `""` to use an existing SSH agent or default key. The virtual environment's `PATH` and `LD_LIBRARY_PATH`, and the project's `PYTHONPATH`, are set before Python starts.

The pipeline runs in this order: source synchronization → model/processor/environment/idle-GPU checks on all four nodes → 97k data and tokenizer fingerprint validation → forward/backward check on a representative video → Ray startup → 32-GPU NCCL check → two-step smoke test → full training. A failure stops the pipeline at that stage. Smoke and full training use the same 16-frame, 768-response-token, and rollout budgets. Full training is reinitialized after smoke completes; smoke-test weights do not carry over.

An existing training job or Ray service causes an error and exit; old training jobs are not cleaned up automatically. When the new pipeline finishes or fails, it cleans up only the Ray services it started and whose process identities it recorded.

```bash
bash run.sh logs qwen3-8b       # Follow logs; Ctrl+C only exits the viewer
bash run.sh status qwen3-8b     # Show the stage, process, and log location
bash run.sh stop qwen3-8b       # Stop this pipeline; does not save the current step
```

Before stopping, check the run's `checkpoint_tracker.json` and wait for any checkpoint write in progress to finish. If a process exits unexpectedly or Ray cleanup remains incomplete, run `bash run.sh cleanup qwen3-8b` on the original head node. Only Ray services matching both this run and the actual recorded process identities will be stopped.

Optional checks:

```bash
bash run.sh models
bash run.sh plan qwen3-8b      # Show the plan only; no SSH, GPU use, or training-output writes
bash run.sh connect qwen3-8b   # Foreground SSH/Python checks only; no source sync or GPU use
bash run.sh check qwen3-8b     # Stop after the four-node, data, and video checks
bash run.sh smoke qwen3-8b     # Stop after two training steps
bash run.sh train qwen3-8b --foreground
```

`check` and `smoke` also run in the background by default; add `--foreground` to watch them in the foreground. To use another cluster configuration, run `bash run.sh --config configs/your_cluster.json train qwen3-8b`. The configuration file supplies the launch settings; they do not depend on stale `MODEL_PATH`, `RUN_NAME`, or Ray/NCCL variables in the terminal.

## trans / Model conversion

Pass a fully saved `global_step_N` directory to detect Qwen2.5-VL / Qwen3-VL automatically and merge the model shards from 32 ranks:

```bash
bash run.sh trans /path/to/RUN/global_step_100
```

`export` and `trans` are equivalent. The default output is `RUN/exports/global_step_100-hf`, ready to use as a Hugging Face / vLLM model directory. You can also specify a separate directory:

```bash
bash run.sh trans /path/to/RUN/global_step_100 --output /path/to/my-model-hf
bash run.sh trans /path/to/RUN/global_step_100 --dry-run
```

Keep `checkpoint_tracker.json` in the original run directory and `dataloader.pt` in the target step directory; do not copy only the `actor` shards. The tool converts only fully saved models, preserves dtype, processor, tokenizer, and chat template, and rejects missing shards, mixed world sizes, changing shards, or overwriting existing output. Conversion runs on CPU and needs enough memory for the complete weights and merging process. HF exports do not include optimizer state; resuming training still requires the original checkpoint. `actor/huggingface` contains only metadata and cannot be used as a complete model by itself.

## Outputs and training settings

The default root is `/gemini-3/space/zwj/AR-MultiModel-97k/outputs`, with separate directories for each model:

```text
outputs/<model_name>/<RUN_NAME>/global_step_100/...
outputs/<model_name>/<RUN_NAME>/checkpoint_tracker.json
outputs/<model_name>/<RUN_NAME>/training_complete.json
outputs/<model_name>/launches/<RUN_NAME>/launcher.log
outputs/<model_name>/launches/<RUN_NAME>/nodes/<node_IP>.<stage>.log
outputs/<model_name>/launches/<RUN_NAME>/run.json
outputs/<model_name>/tensorboard/...
```

Each run saves the resolved model, paths, source fingerprint, and cluster configuration. Data preflight results are stored separately under `work/preflight/<model_name>/<path_fingerprint>/`. Data, tokenizer, and reward-file SHA256 values are checked before reuse; changes to the model or data trigger validation again.

A failed node check displays the IP, stage, exit code, tail of the original error, and full node-log path. Node logs for `check` mark progress through identity/source/idle/versions/imports/cuda/model/data/shared-storage. For example, `Permission denied` indicates an SSH authentication error, whereas missing packages or import failures require fixing the environment based on the actual exception. `exit=1` alone does not establish an SSH problem. Successful SSH/Python checks alone do not mean model/GPU checks have passed.

The package's `data/` directory retains old data files used by the original project's regression tests. Actual 97k training always reads the external `data_root` specified in `configs/cluster.json`. Full videos and model weights are not included in the archive.

| Fixed item | Setting |
| --- | --- |
| Training data | Original 97,430 examples; no relabeling or split changes |
| Hardware | 4 nodes × 8 GPUs, rollout TP=1 |
| MeSD | EMA=0.01, lambda=0.5, decay=1000, teacher disable-after=1000 |
| AR objective | protocol reward, reference coefficient=0.1, sequence mean |
| Batch | rollout prompts=32, n=4, actor global batch=32 |
| Video/length | 2 FPS, up to 16 frames, 100352-pixel budget; prompt=16384, response=768 |
| Optimization | lr=1e-6, one data epoch, gradient checkpointing enabled |
| Saving | Every 100 steps, retain 30 checkpoints with optimizer/resume state; save the final step on completion |

Original parameters are retained for backbone comparisons. The 32B or Video-t-RSI STGR/RSI training configurations were not imported into this project. Visual processing follows each model's processor (Qwen2.5 patch=14, Qwen3 patch=16); the same frame count does not imply the same visual-token count.

## Environment and verification

The runtime stack follows the reference implementation: PyTorch 2.8.0, Transformers 4.57.3, vLLM 0.11.0, and qwen-vl-utils 0.0.14. FlashAttention must import successfully on the server. See [requirements.txt](requirements.txt) for dependencies. To create a new training environment, install the requirements first, then install `flash-attn==2.8.3 --no-build-isolation` compatible with PyTorch/CUDA. The launcher checks the environment; it does not modify an existing one automatically.

See [docs/VERIFICATION.md](docs/VERIFICATION.md) for local verification records and [docs/CHANGES.md](docs/CHANGES.md) for the porting scope. **Package checks are local CPU checks. The Qwen2.5-VL-7B addition requires the target-cluster preflight and smoke test before formal training.** Server preflight checks, smoke tests, and full experiments determine memory use, speed, and final accuracy.

Historical v3 documentation remains in `docs/legacy/`, and old entry points are retained for traceability. Use the `run.sh` commands above for normal operation of this project.
