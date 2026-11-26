# Implementation scope and sources

## 2026-10-30: Qwen2.5-VL-7B and checkpoint retention

- Register `qwen25-7b` with official 7B dimensions and default local-model path.
- Keep full checkpoint saving every 100 formal steps and set retention to 30.
- Extend model-identity and actual-shell coverage; execute the real fast wrapper in its CPU helper.
- See [QWEN25_7B_UPDATE.md](QWEN25_7B_UPDATE.md) for upgrade commands and verification limits.

Date: 2026-10-23.

## 2026-10-27 connection and diagnostics update

Based on `prepare_ssh`, `ssh_check_all`, and `node_setup_env` in `Video-t-RSI/Video-t-RSI-v8/scripts/video_trsi_cluster.sh`: SSH checks before launch, temporary private key copies, unlocking a private agent in an interactive terminal, connection reuse, first-use host-key acceptance, and environment setup before Python starts. SSH resources created by this run are released when it ends; manual Ray cleanup uses a new connection session.

The new `connect` command checks only SSH/Python. Node commands now pass scripts/JSON through stdin while retaining compatibility with the old `--spec` option. Synchronization and node checks save separate logs and display concise exceptions so that long Base64 commands do not hide errors. Per-stage check markers and diagnostics for missing packages/import errors were added.

This update changes only connections, environment initialization, and diagnostics. It does not copy the reference script's global Ray reset or adjust models, data, or training hyperparameters. The screenshots are insufficient to identify the server failure's root cause; new server logs remain necessary.

- Source base: the current local directory `/Users/zwj/Desktop/CVPR/AR/AR-action-training-32gpu-v3/AR-RLSD`, including its already installed 97k entry point; it was not freshly extracted from an old ZIP.
- New directory: `/Users/zwj/Desktop/CVPR/AR/AR-MultiModel-97k`. The source project, My_Project, and Video-t-RSI were not modified.
- References: Qwen3 positional encoding, encoder-cache refresh, and processor checks from `Video-t-RSI/Video-t-RSI-v8-Qwen3VL-8B`; multi-node launch and model-merging ideas from `My_Project/MeSD-Qwen3VL-32B`.

## New entry points

`run.sh` → `tools/launch.py`: a unified model registry, isolated preflight/output directories, a background pipeline on the head node, source synchronization over SSH, status/log/stop commands, and CPU model export. `configs/cluster.json` centralizes deployment settings. The original training script was split into the shared `run_model.sh` and `run_97430.sh`, preserving the original fixed MeSD parameters and data checks.

Static model preflight checks validate dimensions against official configurations, dense architecture, non-quantized weights, shard completeness and safetensors headers, and tokenizer/processor assets. Actual processor loading runs on each node; cluster checks additionally compare SHA256 values for tokenizer, processor, chat-template, and related assets across nodes. Shard-header checks do not read every model tensor or verify remote weight SHA256 values.

## Required Qwen3 adaptations

- `verl/utils/position_ids.py` centralizes Qwen2.5/Qwen3 mRoPE construction; the dataset and teacher reconstruction paths use the same logic.
- Fixed length calculation for Qwen3 text-only rank-1 token inputs; retained the original DeepStack forward implementation.
- `check_video.py` adds actual video-metadata replay, teacher/student input checks, and HF mRoPE consistency checks for Qwen3.
- Clear the vLLM encoder cache after each rollout to prevent stale visual features from being reused after subsequent weight updates. An incompatible engine layout blocks the smoke test rather than continuing silently.
- `opsd_main.py` writes `training_complete.json` only after `trainer.fit()` returns normally. The launcher verifies that smoke actually completed two steps before starting full training.

The AR dataset, reward function, teacher schedule, reference objective, and trainer algorithm were not replaced. Qwen2.5 sampled FPS, Qwen3 video metadata, and FSDP→vLLM key mapping retain the base project's existing logic.

## Export

`tools/export_hf.py` reconstructs weights using the actual DTensor mesh/placements, supports replicated/sharded tensors and legacy Qwen2.5 key mapping, and strictly checks merged keys and shapes against the model's expectations. It writes atomically to a separate HF directory and retains the processor and template. By default, exports are placed outside step directories that checkpoint retention may delete.

## Official configuration sources

- [Qwen2.5-VL-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/blob/main/config.json)
- [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/main/config.json)
- [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct/blob/main/config.json)
- [Official Qwen3-VL model family](https://github.com/QwenLM/Qwen3-VL)
