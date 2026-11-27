# VideoTED

**Tree-Guided Evidence Distillation for Open-Vocabulary Action Recognition**

[Project page](https://0xiaofu0.github.io/Video-TED/)

VideoTED uses verified action trees to connect visible observations, sub-actions and activities. The trees provide two training signals: Tree-Conditioned Credit Modulation (TCM) adjusts token-level credit on sampled responses, and Reference Trajectory Learning (RTL) trains the student on tree-derived reference responses. At inference, the student receives the video and question.

## Collaboration

VideoTED is co-developed by [@0xiaoFu0](https://github.com/0xiaoFu0) and [@WhiteVentral](https://github.com/WhiteVentral). WhiteVentral contributed the [public training release](training/videoted-release/README.md), originally published in [WhiteVentral/VideoTED-code](https://github.com/WhiteVentral/VideoTED-code). This repository is the home for ongoing development, issues, and pull requests.

## Repository layout

| Directory | Purpose |
| --- | --- |
| [annotation](annotation/) | Generate and verify action-evidence annotations and export training records. |
| [training](training/) | Multi-model training on ActionTree-97k and an earlier 32-GPU training snapshot. |
| [training/videoted-release](training/videoted-release/) | Standalone Qwen-VL post-training release with setup instructions, a sample manifest, and CPU checks. |
| [evaluation](evaluation/) | Open-vocabulary action recognition and general video-understanding evaluation. |

## Reproduction

For the standalone training release, start with its [installation and training guide](training/videoted-release/README.md) and run its commands from `training/videoted-release/`. The release includes its own [license](training/videoted-release/LICENSE) and [upstream notices](training/videoted-release/NOTICE).

1. Read the [annotation guide](annotation/README.md) and run the environment and input checks. The annotation package provides candidate preparation, generation, verification and export.
2. Use the [ActionTree-97k training guide](training/actiontree-97k/README.md) for backbone selection and launch configuration. Set your own media, annotation, checkpoint and output paths before running it. The [32-GPU v3 guide](training/legacy-32gpu/README.md) describes an earlier run on 93,583 records.
3. Use the [action recognition guide](evaluation/action-recognition/README.md) and the [general video understanding guide](evaluation/general-video/README.md) for benchmark setup and scoring.

ActionTree-97k contains 97,430 verified examples. Training records use `prompt`, `videos`, `answer`, `teacher_context`, `reference_response`, `reference_status` and `metadata`. The code packages expect the corresponding media and benchmark assets in your own environment.

This repository is prepared from development snapshots. Copy the relevant *.env.example file to its corresponding *.env filename, then set the launchers’ dataset, checkpoint and host settings for your environment. Machine-specific environment files, datasets, checkpoints, case media and review manuscripts are outside this repository. Existing source headers and upstream notices are retained.

