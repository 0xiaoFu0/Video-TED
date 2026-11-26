> The [release-root README](../README.md) is authoritative for full-data v3 deployment and alignment. The original AR extension documentation is retained below.

# AR-RLSD: Action-tree supervision and an independent annotation-reference branch

**This release is full-data, four-node v3 aligned with the MeSD mechanism, retaining the E1 format reward.** For current deployment and complete run commands, follow the
[package-root README](../README.md): 93,583 examples, 4 × 8 GPUs, reference coefficient 0.1,
and `run_merged_32gpu.sh` as the official entry point. The earlier project description remains below; its two-node configuration,
4,676 examples, and old deployment paths are historical context.

The current v4 output protocol is `<think>hierarchical action analysis</think><answer>class</answer>`. The hierarchical body and dual-branch objective are preserved; only the answer tags are standardized. New data is in `data/train.rlsd.think.jsonl` / `data/train.sft.think.jsonl`; both old JSONL files remain unchanged. See [UPLOAD_README.md](UPLOAD_README.md) for upload and launch instructions.

The independent project is at `/Users/zwj/Desktop/CVPR/AR/AR-RLSD`. Its algorithm backbone is official RLSD, pinned to commit `aca313b978de4fb3a509ddf1831bf61fe597a2eb`. After network cloning failed, an independent Git clone was created from a clean local checkout whose origin points to the official repository; neither the original RLSD nor the direction-control project was modified. Original official documentation remains in `README.md` and `README_zh.md`.

## Qwen2.5-VL server update (2026-10-03)

See [UPLOAD_README.md](UPLOAD_README.md) for the execution order on two A800 nodes. The default is the original
`/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct`, reusing the old environment. RLSD starts directly from the base model; SFT is optional.
Video decoding, teacher input, actor, and vLLM share video processing, select patch size according to the processor, and pass actual sampled FPS.
Qwen2.5-VL temporal positions read the model configuration and retain fractional time intervals; SFT uses the same position calculation.
The default configuration uses two nodes with 8 GPUs each, 16 videos per rollout batch, 4 generated trajectories per video, and an actor batch of 16 videos.
The updated 33 CPU tests passed. User-provided server logs show that v2 with the logging fix completed real-video/7B forward/backward checks and two training steps on two nodes with 16 GPUs.
v3 follows MeSD's common parameters: 16 frames, 100352 pixels, 768 generated tokens, cosine learning rate, disabled CPU offload, and 1 epoch.
It retains the original model, RLSD/reference losses, and saving every 10 steps with 5 retained checkpoints. See UPLOAD_README.md for GPU verification of the new parameters.

## Implemented training path

Optional small-scale action-tree SFT → RLSD action classification → tree-structured privileged context and an independent annotation-reference loss.

The student receives only the video, question, and generic tree-format requirements. The teacher additionally reads `teacher_context`. The reference trajectory is `reference_response`, scored with teacher forcing under the **student's original context**, without the teacher's privileged information.

```text
video + question ── student sampling ── G responses ── class reward ── group advantage
        │                                   │
        │                         frozen teacher + tree context
        │                                   │
        │                             RLSD token weighting
        │                                   │
        └── annotation reference ── separate loss ── combined actor update
```

The two outer tags are `<think>` / `<answer>`; the body is an indented hierarchy in English. `<action_tree>` in the teacher's `teacher_context` remains an evidence marker, not the assistant's answer format. The current release's main reward requires the correct final class and a complete, valid think/answer structure; it does not award points for tree-node counts or action keywords. The reference objective supervises tree expression; the format reward does not guarantee correct tree content. The current full-data entry point does not perform SFT first.

## Data

New data has been placed in `data/`, with 4,676 examples across 51 classes in each file, without resplitting or filtering:

- `train.sft.think.jsonl`: messages/videos/answer/metadata.
- `train.rlsd.think.jsonl`: prompt/videos/answer/teacher_context/reference_response/reference_status/metadata.
- `train.sft.jsonl` / `train.rlsd.jsonl`: unchanged old `<action_tree>` data, retained only for provenance and format conversion, not used by the new launcher.
- `think_format_manifest.json`: SHA256 values of old/new files, converter hash, and preservation scope. Data hashes in `docs/ar/provenance.json` still refer to the old source files.

To rebuild the think version from old data (the output must be a new directory):

```bash
python -m verl.ar.convert_think --source-dir data --out data_think_rebuild
```

Conversion only replaces the reference answer's opening/closing tags and updates the student's hierarchy-format instruction. Every character, indentation, newline, and answer in the reference body is preserved, as are teacher context, video paths, labels, ordering, metadata, and unreviewed status. Default training does not automatically run separate SFT.

Original Video-STAR `sft.json` supplies samples and classes; observation paragraphs in the companion `sft_cot.json` supply action evidence. Tool selection and candidate scores were not used. English class branches are a project design choice; evidence sentences have not been checked video by video. The training configuration explicitly sets `ar_allow_unreviewed_reference: true`, meaning training uses existing generated annotations, not that they passed visual review. Setting it to false excludes the independent loss for unreviewed references but does not automatically remove teacher context.

The video root is currently `/gemini-3/space/zwj/LMUData/HMDB51/videos`. If the actual extraction layout differs, rebuild into a new directory:

```bash
python -m verl.ar.prepare_data \
  --source-dir data \
  --out data_rebased \
  --video-root /gemini-3/space/zwj/LMUData/HMDB51/actual_video_directory
```

Validate annotations and server media:

```bash
python -m verl.ar.prepare_data --check data/train.rlsd.think.jsonl
python -m verl.ar.prepare_data --check data/train.rlsd.think.jsonl --check-media
```

The training set is not used as validation data. Validation is disabled by default; after supplying independent evaluation data, set `data.val_files`, `trainer.val_before_train=True`, and `trainer.val_freq`. Validation JSONL needs only prompt/videos/answer and optional metadata, without teacher context.

## Training entry points

The server reuses `/path/to/workspace/zwj/.venvs/open-o3-visd-vllm`. Launch scripts automatically use this environment and set the current project's
`PYTHONPATH`; there is no need to rerun `pip install -e .` in the shared old environment. The local `.venv-test` is only for CPU tests.
Use the versions in `requirements.txt`, and check consistency between the two nodes before running.

Optional cold-start SFT (one video per GPU; multi-process execution defaults to FSDP full-shard, with gradient accumulation controlling the batch):

```bash
torchrun --nproc_per_node=8 -m verl.ar.sft \
  --model /path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --data data/train.sft.think.jsonl \
  --output /gemini-3/space/zwj/AR-RLSD/outputs/sft_coldstart \
  --max-samples 512 --epochs 1
```

512 is an adjustable experimental budget; the default `--max-samples 0` uses all examples. Subsets use random sampling with a fixed seed and save sample IDs without changing original data. SFT selects Qwen2.5-VL or Qwen3-VL and FSDP wrap layers according to the model configuration. It masks user/video/padding positions and trains only the assistant. Prefix mismatches or overlong inputs raise errors rather than silently truncating.

RLSD + tree context + independent reference branch:

```bash
# Start two-node Ray as described in UPLOAD_README.md, then run on the head:
bash examples/action_recognition/run_qwen25.sh smoke
```

This launcher defaults to the original Qwen2.5-VL-7B-Instruct and does not require prior SFT. After optional SFT, use MODEL_PATH to select its checkpoint.
The SFT output path above is an example, not an already generated model. Increase step counts after server smoke passes. The launcher checks media paths and output directories and does not automatically overwrite historical runs.

Common ablation overrides:

| Control | CLI override |
|---|---|
| RLSD with tree context, without reference loss | `opsd.ar_reference_coef=0` |
| RLSD + difficulty-weighted reference CE | `opsd.ar_reference_mode=ce` |
| GRPO without teacher reweighting or reference loss | `opsd.rlsd_lambda=0 opsd.ar_reference_coef=0` |

The last option preserves the same training path but still performs teacher forward passes, so it is unsuitable as a GRPO compute-cost baseline. The pure official RLSD path can use upstream configurations. Current AR uses population standard deviation within groups; keep this consistent in algorithm comparisons.

## Exact V1 definition

For each video, use only the binary rewards from G actual student rollouts:

- `A_i = (r_i − mean(r)) / (population_std(r) + 1e−6)`; all-wrong and all-correct groups both yield zero.
- `delta_it = stopgrad(log p_old(y_it | x,z,y_<t) − log p_old(y_it | x,y_<t))`.
- `A'_it = A_i × [(1−lambda) + lambda × clip(exp(sign(A_i) × delta_it), 1−eps, 1+eps)]`.
- `b = stopgrad(kappa × eligible × (1−mean(r))²)`.
- Reference objective: `−b × mean_t min(rho*_t, clip(rho*_t,1−eps_ref,1+eps_ref))`, where `rho* = exp(log p_current(y*|x) − log p_old(y*|x))`.
- Total loss is the original policy loss plus `ar_reference_coef × reference_loss`.

The teacher is refreshed from the actor at the start of each actor update, before any optimizer update, and remains fixed for that iteration, including the first iteration after checkpoint restoration. Old student/reference scores are computed before updates and detached. Actor dropout is disabled during optimizer updates to keep scoring conditions consistent. RLSD clips in log space before exponentiation to avoid overflow.

The reference loss first takes a token mean per sequence, then averages over all groups, without renormalizing by the number of active references. To reuse existing batch dispatch, V1 duplicates the same reference alongside G rollouts. With equal group sizes, the full-batch mean equals one reference per group, but introduces extra forward-pass cost. These references are not sampled rollouts and do not enter the classification-reward mean or advantage standard deviation. The `ce` control uses the same difficulty weights; it is not fixed-weight CE.

This implementation draws on OraRL's annotation-as-rollout idea but does not include its complete directional gain, pruning, or recentering. The reference likelihood ratio also is not unbiased on-policy importance sampling; theoretical guarantees of the complete OraRL/RLSD methods cannot be inherited directly.

## Code locations and verification

- `verl/ar/reward.py`: final answer-tag parsing, class accuracy, and complete-format success reward.
- `verl/ar/response_format.py`: shared think/answer format checks, without judging tree semantics.
- `verl/ar/objectives.py`: group statistics, stable RLSD reweighting, and independent reference loss.
- `verl/ar/trainer.py`, `batch.py`: reference tokens/masks, concatenation with original student prompts, and old-policy scoring.
- `verl/trainer/opsd_trainer.py`: teacher-context wiring and reference preparation.
- `verl/workers/actor/dp_opsd_actor.py`: joint loss, fixed-teacher refresh, and normalization.
- `verl/ar/sft.py`: optional cold-start SFT.
- `examples/action_recognition/rlsd.yaml`, `train.sh`: minimal experimental configuration and launcher.

The fix passing `raw_prompt` through training/validation rollouts was reused from direction-control. Its video processor and vLLM backend already share substantial implementation with the corresponding paths in this official checkout; the multi-view/direction-control logic was not imported wholesale. See `docs/ar/provenance.json` for sources and hashes.

CPU tests:

```bash
python -m pytest tests/ar -q
```

The initial version passed 17 tests in an isolated local environment; the Qwen2.5-VL update passed 33 tests in total. Existing tests cover the complete data contract, reward, numerical stability at extreme values, stop-gradient, EOS/padding, mRoPE concatenation, actual DataProto reference wiring, structured configuration, and the production actor-update method using CPU model forward passes and single-rank communication test doubles. CPU tests do not replace real-video or multi-GPU FSDP validation.

Dynamic microbatching, non-binary reward shaping, online group filtering, and mixed offline rollouts are currently unsupported; configuration validation rejects these combinations. Existing server screenshots validate the v2 short-training workflow; v3 parameters, full video coverage, complete training, and model quality remain to be verified. Run the new configuration for 2 steps with full-training frame/generation-length settings first, then increase the training budget after it passes.
