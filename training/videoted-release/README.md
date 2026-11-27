# VideoTED

**Tree-Guided Evidence Distillation for Open-Vocabulary Action Recognition**

[Project page](https://0xiaofu0.github.io/Video-TED/)

VideoTED post-trains Qwen-VL models for video action recognition. Tree-Conditioned Credit Modulation (TCM) uses training-only action-tree evidence to reweight token-level advantages. Reference Trajectory Learning (RTL) learns from a separate tree-derived response and gives it more weight when fewer sampled responses succeed. At inference, the student uses the video and prompt.

This release includes the training code and a synthetic JSONL example. The training annotations and VideoTED model weights will be released separately. To train with this code, supply your own training manifest, videos, and a compatible base model. The core losses are in [`verl/ar/objectives.py`](verl/ar/objectives.py).

## Training efficiency

VideoTED uses vLLM 0.11.0 to batch and sample on-policy student rollouts. PyTorch FSDP handles parameter updates, and Ray coordinates the distributed workers. [vLLM documentation](https://docs.vllm.ai/en/v0.11.0/api/vllm/entrypoints/llm.html)

In the recorded Qwen2.5-VL-7B run, one epoch took 3,044 steps on 32 NVIDIA A800 GPUs (80 GB each), with 32 video prompts and four responses per prompt. The mean step time was **24.74 seconds** across 3,013 steps after excluding the first step and the 30 steps that saved checkpoints.

## Installation

Use Linux with CUDA-capable NVIDIA GPUs and Python 3.11. From the repository root, install the training dependencies and FlashAttention for the matching PyTorch and CUDA environment.

```bash
python -m pip install -r requirements.txt
python -m pip install flash-attn==2.8.3 --no-build-isolation
```

The supported backbones are Qwen2.5-VL-Instruct (3B and 7B) and Qwen3-VL-Instruct (4B and 8B).

## Training data

Use one JSON object per line, following [`examples/train.example.jsonl`](examples/train.example.jsonl).

| Field | Content |
| --- | --- |
| `prompt` | Student instruction with one `<video>` marker and the requested `<think>…</think><answer>…</answer>` format. |
| `videos` | A one-element list containing the absolute path to the video. |
| `answer` | Ground-truth action class. |
| `teacher_context` | Nonempty training-only evidence. |
| `reference_response` | An indented action hierarchy in `<think>` followed by the matching class in `<answer>`. A rejected reference may be empty. |
| `reference_status` | `verified`, `reviewed`, `unreviewed`, or `rejected`. The default recipe enables unreviewed references and excludes rejected references from RTL. |

Keep training-only evidence in `teacher_context`. Additional metadata is optional.
You can check the manifest before training and add `--check-media` to verify video paths.

```bash
python -m verl.ar.prepare_data --check examples/train.example.jsonl
python -m verl.ar.prepare_data --check /data/train.jsonl --check-media
```

## Train

Pass a local checkpoint directory or a Hugging Face model ID, followed by your training JSONL and any configuration overrides.

```bash
bash train.sh Qwen/Qwen2.5-VL-7B-Instruct /data/train.jsonl
```

The default [configuration](configs/videoted.yaml) uses one node with eight GPUs, 32 prompts per batch, and four rollouts per prompt. Provide at least one full batch of examples. The default limits are 16 sampled frames at 2 FPS, 768 generated tokens, and 1,024 reference tokens including EOS. Set `opsd.ar_reference_max_length` if your tokenizer produces longer references. The trainer saves checkpoints every 100 steps and keeps up to 30.

For a short training run, append `trainer.max_steps=2` and provide enough examples for two full batches. For multiple nodes, start a Ray cluster and make the repository, videos, manifest, and any local model files available at the same absolute paths on every node. Run the launcher once with `RAY_ADDRESS` set to the head node.

```bash
RAY_ADDRESS=ray-head:6379 bash train.sh /models/Qwen2.5-VL-7B-Instruct /data/train.jsonl \
  trainer.nnodes=4 trainer.n_gpus_per_node=8 \
  worker.actor.fsdp.fsdp_size=8 worker.ref.fsdp.fsdp_size=8 \
  worker.actor.micro_batch_size_per_device_for_update=4 \
  worker.actor.micro_batch_size_per_device_for_experience=4
```

To export a completed local FSDP checkpoint, use the same PyTorch version as training and keep the run's checkpoint tracker with the shards.

```bash
python tools/export_hf.py /path/to/run/global_step_100 --output /path/to/export
```

## CPU checks

```bash
python -m pip install -r requirements-test.txt
python -m pytest -q
```

## Acknowledgments

The VideoTED training recipe follows selected settings from MeSD (Multi-Evidence Self-Distillation for VideoLLM), including video sampling and optimizer settings. [MeSD code](https://github.com/WhiteVentral/MeSD-code)

The training runtime builds on [RLSD](https://github.com/iie-ycx/RLSD), [EasyVideoR1](https://github.com/cyuQ1n/EasyVideoR1), [EasyR1](https://github.com/hiyouga/EasyR1), and [veRL](https://github.com/volcengine/verl). See [NOTICE](NOTICE) and [LICENSE](LICENSE).
