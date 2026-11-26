# Current release entry point

This is the independent `AR-action-training-32gpu-v2` package, using all 93,583 examples, four nodes with 32 GPUs,
the E1 complete-format reward, and reference coefficient 0.1. Use the deployment instructions and commands in the [package-root README](../README.md).
The older deployment record below is retained for history; its two-node setup and old project paths do not apply to this full-data run.

# Qwen2.5-VL-7B-Instruct: v4 think hierarchy-format update

This update uses the original Qwen2.5-VL-7B-Instruct directly for RLSD + action-tree reference supervision.
SFT is optional; the workflow below starts from the original weights. This update passed 46 CPU tests; see `docs/ar/think_format_validation.json`.
Server logs provided by the user show that v2, with the JSON logging fix, completed single-GPU forward/backward checks and two training steps on two nodes with 16 GPUs.
v4 retains v3's training hyperparameters and dual-branch objective; the outer reasoning tag is standardized to `<think>`, preserving the hierarchical body character for character. GPU training with this version still requires server validation.

## 1. Update the code on both nodes

Upload `AR-RLSD_qwen25_think_20261004_v4.tar.gz` to `/path/to/workspace/zwj/`.
Run the following on both nodes (extract only once if the code directory is shared):

```bash
cd /path/to/workspace/zwj
tar -xzf AR-RLSD_qwen25_think_20261004_v4.tar.gz
cd AR-RLSD-think
sha256sum -c SHA256SUMS --quiet
```

The archive extracts into a separate `AR-RLSD-think/` directory, preserving the server's old `AR-RLSD/` project. Old JSONL files remain unchanged; the new version reads `.think.jsonl` files by default. All current AR source code is included.
The environment, models, and videos use their existing server directories.

## 2. Check one real video and 7B forward/backward on the head node

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
export TRAIN_DATA="$PWD/data/train.rlsd.think.jsonl"
export SFT_DATA="$PWD/data/train.sft.think.jsonl"
export MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
bash examples/action_recognition/run_qwen25.sh check-video
```

By default, this uses GPU 0 on the current node, decodes one training video, and checks that:

- Pixel tensors match across student input, training workers, and rollout input.
- Actual sampled FPS and the video-token grid are passed correctly.
- Qwen2.5-VL position IDs match the results computed by the server's installed vLLM element by element.
- Loading the original 7B weights and performing one supervised forward/backward pass produces finite loss and gradients, with nonzero gradients.

This check performs no optimizer update and saves no training weights. Success prints `GPU forward/backward passed` and
`Video preprocessing passed`. If either line is absent or the process exits nonzero, inspect the exception before proceeding with training.

To expand decoding/preprocessing checks to one video per class (backward still runs only once on the first video):

```bash
CHECK_SAMPLES=51 bash examples/action_recognition/run_qwen25.sh check-video
```

## 3. Start Ray on two nodes

The default head is `192.0.2.12` and the worker is `192.0.2.13`. The script checks that the current machine has the corresponding IP.
If rebuilding a container changes its IP, run `hostname -I` on both nodes first, then set the same values in both terminals:

```bash
export HEAD_HOST=actual_head_node_IP
export WORKER_HOST=actual_worker_node_IP
```

On the head node:

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
bash examples/action_recognition/run_qwen25.sh head
```

On the worker node:

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
bash examples/action_recognition/run_qwen25.sh worker
```

The script does not use SSH or automatically terminate existing Ray jobs. If Ray is already running or a port is occupied, first identify which job owns that instance.
Both nodes need 8 available GPUs and access to the same code, model, and data paths.

## 4. Run a two-step training check on the head node only

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
bash examples/action_recognition/run_qwen25.sh smoke
```

The check first validates node IPs, GPU resources, Python/dependency versions, source hashes, annotation hashes, and model-configuration hashes, then runs a 16-rank NCCL
all_reduce (each GPU should receive 136). Only after all checks pass does it start the single training driver.

The short check uses 8 frames, up to 512 generated tokens, and 2 training steps; its algorithm is the same as full training.
It checks video decoding, vLLM rollout, teacher scoring, the independent reference loss, FSDP backward, and final checkpoint saving.
The short training check passes only if the program completes both steps and exits normally; CPU tests and NCCL success do not replace it.

Log: `/gemini-3/space/zwj/AR-RLSD/outputs/logs/ar_qwen25_think_smoke_timestamp.log`.
Checkpoint: `/gemini-3/space/zwj/AR-RLSD/outputs/ar_qwen25_think_smoke_timestamp/`.
The existing trainer saves a checkpoint at completion even when periodic saving is disabled.

To run the cluster check separately: `bash examples/action_recognition/run_qwen25.sh cluster`.

If two-node Ray is already configured, confirm both nodes can read the updated project, then run the following two-step check on the head node.
It retains the full configuration's 16 frames, 768 tokens, and CUDA graph settings, shortening only the number of training steps:

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
export TRAIN_DATA="$PWD/data/train.rlsd.think.jsonl"
export MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
export RUN_NAME="ar_qwen25_think_check_$(date +%Y%m%d_%H%M%S)"
bash examples/action_recognition/run_qwen25.sh train trainer.max_steps=2
```

The two-step check computes its cosine learning-rate schedule over two steps and checks only execution and memory use. Full training uses a new run name and starts from the original weights.

## 5. Full training

After the short check completes, run the following in a persistent terminal such as tmux on the head node:

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
export TRAIN_DATA="$PWD/data/train.rlsd.think.jsonl"
export MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
export RUN_NAME="ar_qwen25_think_train_$(date +%Y%m%d_%H%M%S)"
bash examples/action_recognition/run_qwen25.sh train
```

By default, a fresh run starts from the original model: 1 epoch, up to 16 frames, target sampling rate 2 FPS, and up to 768 generated tokens.
A batch contains 16 videos, each producing 4 trajectories: 64 trajectories total, 4 per GPU, with microbatch size 1.
Default new run names include a timestamp, and existing outputs cannot be overwritten. Ray reuses the cluster started above.

Training-configuration overrides can be appended, for example to run 20 steps first:

```bash
bash examples/action_recognition/run_qwen25.sh train trainer.max_steps=20
```

## Retained v3 / MeSD parameters

Reference file: `/Users/zwj/Desktop/My_Project/MeSD/configs/train_mesd.yaml`.
The alignment checklist and source hashes are saved in `docs/ar/mesd_config_alignment.json`.

| Parameter | v2 | v3 |
|---|---|---|
| Maximum video frames | 32 | 16 |
| Video min/max pixels per frame | 262144 | 100352 |
| Maximum generated tokens | 1024 | 768 |
| Learning-rate schedule | constant | cosine, decays from 1e-6 to 0, no warmup |
| Gradient-norm clipping | 1.0 | 5.0 |
| Policy-ratio clipping bounds | 0.8 / 1.28 | 0.8 / 1.2 |
| top-p / top-k | 1.0 / -1 | 0.95 / 50 |
| vLLM GPU-memory budget fraction | 0.60 | 0.50 |
| Actor parameter/optimizer offload | Enabled | Disabled |
| Teacher FSDP CPU offload | Enabled | Disabled |
| Training epochs | 10 | 1, approximately 292 steps for 4676 examples |

The settings already matched for 16 videos/batch, 4 trajectories/video, per-GPU microbatch 1, AdamW, learning rate 1e-6, and weight decay 0.01.
Actor FP32 weights/BF16 computation and teacher BF16 are recorded explicitly; saving every 10 steps and retaining 5 checkpoints preserves the confirmed settings.

AR retains the original Qwen2.5-VL, HMDB51 data fields, errors for overlong inputs, copying and freezing the teacher each step, RLSD lambda=0.5,
reference-loss coefficient 0.1, reference-target length 1024, and separate run directories. MeSD's EMA, multi-view fusion, RKL routing,
1000-step annealing, STGR data-loading parameters, and automatic-resume settings were not imported. The AR loader still uses 8 workers.
`max_model_len` automatically uses the input limit plus generation limit, currently 17152; `max_token_len_per_gpu` is unused in fixed-microbatch mode.

v4 does not change the v3 hyperparameters in this section. The new format applies only to newly launched jobs; the old step292 model still corresponds to the old training. Formal comparisons start from the original model with new run names.

## Paths and environment

| Item | Default |
|---|---|
| Environment | `/path/to/workspace/zwj/.venvs/open-o3-visd-vllm` |
| Project | `/path/to/workspace/zwj/AR-RLSD-think` |
| Original model | `/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct` |
| RLSD annotations | `data/train.rlsd.think.jsonl` under the project |
| SFT annotations | `data/train.sft.think.jsonl` under the project |
| Videos | `/gemini-3/space/zwj/LMUData/HMDB51/videos` |
| Output root | `/gemini-3/space/zwj/AR-RLSD/outputs` |
| Ray / NCCL check port | `6379` / `29511` |

Environment variables can override `TRAIN_ENV_ROOT`, `MODEL_PATH`, `TRAIN_DATA`, `SFT_DATA`, `OUTPUT_ROOT`, `HEAD_HOST`,
`WORKER_HOST`, `RAY_PORT`, and `AR_NCCL_PORT`; `CHECK_GPU` selects the GPU for the single-GPU check.
Model and annotation path overrides must match on both nodes. Dependencies do not need reinstalling, and `pip install -e .` is unnecessary.

To place annotations in LMUData as well, copy them to the agreed `HMDB51/annotations/ar_think_v4` directory, then point both nodes'
`TRAIN_DATA` / `SFT_DATA` to the respective JSONL files there. Video paths are already correct, so the annotations need no rewriting.

## Implementation and verification scope

Shared video processing is in `verl/utils/video.py`. The dataset, teacher prompts, FSDP workers, vLLM, and SFT use the same frames and sampling rate.
Qwen2.5-VL uses the processor's patch size and sampled FPS; Qwen3-VL retains its metadata path.
Old caches with dimensions that are not multiples required by the current model raise an error, preventing silent acceptance of incompatible caches.

Video temporal positions read `vision_config.tokens_per_second` from the model configuration. Fractional sampling times are preserved until the final integer conversion,
matching the pinned [vLLM 0.11.0 mRoPE implementation](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/rotary_embedding/mrope.py).
Qwen2.5-VL frame/FPS inputs follow the [Transformers 4.57.3 Processor](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py),
and vLLM model processing follows [vLLM 0.11.0 Qwen2.5-VL](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/models/qwen2_5_vl.py).

The previous 33 CPU tests plus the new checks total 46 passing tests. They cover actual Transformers patch reshaping, FPS and teacher/actor/rollout propagation,
SFT masks/positions, batch rejection conditions, and argument passing/failure blocking through the actual shell launcher with Python/Ray test doubles.
The local machine has no CUDA, real videos, or model weights. Prior GPU success records come from user-provided server screenshots, not local tests.
The data remains existing generated annotations; checking video paths does not establish that action-tree contents have been manually reviewed video by video.

## v4 format changes and data validation

- `reference_response` replaces `<action_tree>` / `</action_tree>` with `<think>` / `</think>`, preserving the indented tree character for character.
- Student and teacher still require hierarchical action analysis, with the final class inside `<answer>`; tree evidence in the teacher context is preserved.
- The reference loss retains coefficient 0.1 and clipped mode; classification reward, RLSD reweighting, and all other training parameters remain unchanged.
- New `train.rlsd.think.jsonl` / `train.sft.think.jsonl` files each contain 4,676 examples across 51 classes. Original data remains in `train.rlsd.jsonl` / `train.sft.jsonl`.
- Explicitly set `TRAIN_DATA` / `SFT_DATA` as above before launch to avoid inheriting old terminal paths. Full training uses the original `MODEL_PATH` and a new `RUN_NAME`.

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
source /path/to/workspace/zwj/.venvs/open-o3-visd-vllm/bin/activate
python -m verl.ar.prepare_data --check data/train.rlsd.think.jsonl --check-media
```

This check validates the text contract and video paths. During the server's two-step training check, also inspect whether generated samples actually use `<think>`, preserve hierarchical organization, and have correctly parsed final answers; format instructions do not guarantee compliance on every generation. Evaluation continues to use the same VideoSTAR-Eval-v5.
