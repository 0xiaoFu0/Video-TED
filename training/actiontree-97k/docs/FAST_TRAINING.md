# Fast training on 4 × 8 A800-80GB

```bash
cd /path/to/workspace/zwj/AR-MultiModel-97k
bash run_fast.sh train qwen3-4b      # or qwen25-3b / qwen25-7b / qwen3-8b
bash run_fast.sh logs  qwen3-4b      # every run.sh command passes through
```

`run_fast.sh` runs exactly the `run.sh` pipeline (sync → node checks → data
preflight → video check → Ray → NCCL check → 2-step smoke → full training)
with the same model, data, reward, objective and schedule. It only changes
*how* each training step is executed. Nothing in the MeSD contract moves:
32 prompts × n = 4, actor global batch 32, lr 1e-6, one epoch, 16 frames /
100352 px, 768 response tokens, EMA 0.01, λ = 0.5 with 1000-step decay and
teacher cutoff, reference coefficient 0.1, save every 100 steps, retain at most 30 checkpoints.
`verl/ar/validation.py` still rejects any run that deviates from it.

## Why the default configuration underuses the GPUs

Per training step the cluster handles 32 prompts × 4 rollouts = **128
trajectories, i.e. 4 per GPU**. The data is small: the student prompt is
~100 text tokens plus the video (16 frames at 100352 px → 8 temporal groups ×
98 tokens on Qwen3-VL, × 128 on Qwen2.5-VL ≈ 0.8–1.0k tokens), the teacher
prompt adds ~300 tokens, responses are ≤ 768 tokens. Each GPU therefore sees
roughly **4 × 2k tokens of real work per step** on a 3–8B model, which an A800
finishes in well under a second of arithmetic. The step time is dominated by
fixed overheads, and the default settings multiply them:

| Default setting | Effect per step (per GPU) |
| --- | --- |
| `micro_batch_size_per_device_for_update = 1`, `…_for_experience = 1` | The 4 sequences a GPU owns are processed one at a time: 4 old-log-prob forwards, 4 reference-log-prob forwards, then 4 × (student fwd/bwd + teacher fwd + reference fwd/bwd) in `update_actor`. That is ~20 separate full-model passes for ~8k tokens of work. |
| `fsdp_size = -1` (32-way FULL_SHARD across 4 nodes) | Every one of those passes all-gathers the entire model (bf16: ~6 GB for 3B, ~8 GB for 4B, ~16 GB for 8B) **over the inter-node network**, and every backward reduce-scatters over it too. With 1-sequence micro-batches the GPUs mostly wait on NCCL. |
| Driver-side teacher prompts (`construct_teacher_prompts`) | The head process re-ran the Qwen video processor for all 128 rows every step, although the 4 rows of a rollout group share one prompt and one decoded video (32 distinct videos). Single-threaded CPU work while 32 GPUs idle. |
| vLLM rollout: 4 prompts per GPU | Decode at batch 4 is far below what the KV cache (0.5 × 80 GB) can hold. This one is inherent to the 32-prompt contract: with on-policy sampling there is no way to shorten the ~768 sequential decode steps without changing the batch size (see "What was not changed"). |

Memory is not the constraint anywhere: with 32-way sharding the fp32 actor +
optimizer + bf16 EMA teacher need ~2–5 GB per GPU, vLLM takes 40 GB only
while awake, and activations for 2k-token sequences with gradient
checkpointing are a few GB.

## What `run_fast.sh` changes

All overrides are OmegaConf CLI arguments appended by
`examples/action_recognition/run_97430.sh` from `AR_FAST_OVERRIDES`. They sit
*after* the fixed batch contract and *before* the user's own arguments and
the pinned MeSD alignment settings (`run_model.sh`), so they cannot redefine
the experiment and you can still override them on the command line.

1. **Pack each GPU's work into one forward**
   `worker.actor.micro_batch_size_per_device_for_update=4`,
   `worker.actor.micro_batch_size_per_device_for_experience=4`
   (= 32 × 4 / 32 GPUs; `padding_free: true` packs them with varlen
   flash-attention, no cross-sequence attention). The policy, teacher and
   reference losses are normalized over the *global* sequence count
   (`dp_opsd_actor.update_policy_grpo_opsd`, `total_sequence_count` /
   `total_reference_rows` all-reduced across ranks), so the summed gradient
   is the same as with four micro-batches of one, up to floating-point
   summation order. Passes per step drop from ~20 to ~5.
2. **Shard within the node (HSDP)**
   `worker.actor.fsdp.fsdp_size=8`, `worker.ref.fsdp.fsdp_size=8` →
   `ShardingStrategy.HYBRID_SHARD`: parameters are sharded over the 8 GPUs of
   a node and replicated across the 4 nodes. All parameter all-gathers now run
   over NVLink; only the gradient reduce crosses the inter-node fabric, once
   per step. The actor and the EMA teacher must share one mesh
   (`validation.py` checks this), and `_ema_teacher_from_actor`, the vLLM
   weight sync (`full_tensor()`), the sharded checkpoint and
   `tools/export_hf.py` (which handles the `("ddp", "fsdp")` mesh) are all
   mesh-agnostic. Per-GPU memory grows from (model+optimizer)/32 to /8: ≈ 6 GB
   for 3–4B, ≈ 18 GB for 8B — still far below 80 GB alongside vLLM's 40 GB.
3. **Tokenize each distinct teacher prompt once** (`verl/trainer/opsd_algos.py`,
   `construct_teacher_prompts`): a per-call memo keyed on the rendered prompt
   text and the identity of the decoded-media object. `DataProto.repeat`
   duplicates object references, so the 4 members of a rollout group hit the
   memo and the video processor runs 32 times per step instead of 128. Rows
   that do not share both the text and the media object are tokenized exactly
   as before, so the output is unchanged. Covered by
   `tests/ar/test_teacher_prompt_memo.py`.

Resulting change in the trainer's argv (train mode) relative to `run.sh`:

```
… data.rollout_batch_size=32 worker.actor.global_batch_size=32 … trainer.load_checkpoint_path=null
  worker.actor.micro_batch_size_per_device_for_update=4
  worker.actor.micro_batch_size_per_device_for_experience=4
  worker.actor.fsdp.fsdp_size=8 worker.ref.fsdp.fsdp_size=8
  [your extra arguments]
  opsd.ar_alignment_mode=mesd … (fixed MeSD alignment, unchanged)
```

### Knobs

| Variable | Default | Meaning |
| --- | --- | --- |
| `FAST_HSDP` | `8` | GPUs per FSDP shard group. `0` keeps the original 32-way full sharding (use this if a node-local mesh ever misbehaves; you keep change 1 and 3). |
| `FAST_MICRO` | `4` | Sequences per GPU per forward. Must divide 4. |
| `FAST_VLLM_MEM` | unset | Override `worker.rollout.gpu_memory_utilization` (e.g. `0.6`). Not needed at batch 4 per GPU. |

`bash run_fast.sh smoke qwen3-4b --foreground` runs the two-step smoke test
with these settings and prints `timing_s/step`, `timing_s/gen`,
`timing_s/old`, `timing_s/update_actor` and
`timing_s/grpo_opsd_teacher_prompt`; compare them with `bash run.sh smoke …`
on the same model to see the gain on your hardware before committing to a
full run. `gen` should be roughly unchanged, the other three much smaller.

### Checkpoints

Checkpoints written by a fast run carry the 2-D device mesh. Resume with the
same `FAST_HSDP` value (`trainer.load_checkpoint_path=…` appended to the
command works as before); `bash run.sh trans …` exports them unchanged.

## What was not changed, and why

* **Rollout batch / number of optimizer steps.** With 4 prompts per GPU, the
  vLLM decode phase is the remaining floor (~768 sequential decode steps at
  batch 4). Raising `data.rollout_batch_size` to 128–256 would give 4–8× the
  rollout throughput per sample, but it changes the experiment: the number of
  updates drops from 3044 to 761–380, and the EMA rate, the 1000-step teacher
  cutoff, the 1000-step λ decay and the cosine schedule are all counted in
  steps. `validation.py` pins these values on purpose, so a larger batch is a
  different experiment that needs those constants (and probably the learning
  rate) rescaled and agreed first. If you want that variant, it is a one-line
  change to the `fast=( … )` list in `run_fast.sh` plus relaxing the pinned
  values in `verl/ar/validation.py` and `run_model.sh`.
* `trainer.save_freq=100` with full optimizer state (≈ 30 saves of
  ~1–4 GB per rank for 3–8B) is cheap enough; left alone.
* `StatefulDataLoader(num_workers=8)` decodes 32 videos per step with
  prefetching; it was not observed as a bottleneck in the code and is left
  alone.

## Files touched

* `run_fast.sh` — new single-command entry point.
* `examples/action_recognition/run_97430.sh` — reads `AR_FAST_OVERRIDES`
  (2 lines); the default path is unchanged when the variable is unset.
* `verl/trainer/opsd_algos.py` — memo in `construct_teacher_prompts`.
* `tests/ar/test_teacher_prompt_memo.py`,
  `tests/multimodel/test_shell_contract.py` — tests for both.
