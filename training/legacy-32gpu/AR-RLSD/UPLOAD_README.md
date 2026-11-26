# 当前发布包入口

本包为独立 `AR-action-training-32gpu-v2`，全量 93,583 条、四节点 32 卡、
E1 完整格式奖励及参考系数 0.1。请使用 [包根目录 README](../README.md) 的部署和命令。
下文为保留的旧版部署记录，两节点和旧项目路径不适用于本次全量运行。

# Qwen2.5-VL-7B-Instruct：v4 think 层级格式更新包

本更新包使用原版 Qwen2.5-VL-7B-Instruct 直接进行 RLSD + 动作树参考监督。
SFT 为可选步骤，下面流程从原始权重开始。本次通过 46 项 CPU 测试，记录见 `docs/ar/think_format_validation.json`。
用户提供的服务器日志显示，v2 加 JSON 日志修复后已完成单卡前反向及两节点 16 卡两步训练。
v4 沿用 v3 的训练超参数和双分支目标；回答外层统一为 `<think>`，层级正文逐字符保留。新版 GPU 训练仍需服务器验证。

## 1. 更新两节点代码

将 `AR-RLSD_qwen25_think_20261004_v4.tar.gz` 上传到 `/path/to/workspace/zwj/`。
两节点分别执行以下命令（如代码目录共享，只需解压一次）：

```bash
cd /path/to/workspace/zwj
tar -xzf AR-RLSD_qwen25_think_20261004_v4.tar.gz
cd AR-RLSD-think
sha256sum -c SHA256SUMS --quiet
```

压缩包展开为独立的 `AR-RLSD-think/`，服务器旧 `AR-RLSD/` 工程保留。包内旧版 JSONL 原样保留；新版默认读取 `.think.jsonl` 文件。包含当前全部 AR 源码。
环境、模型、视频沿用服务器现有目录。

## 2. 在主节点验证一个真实视频和 7B 前反向

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
export TRAIN_DATA="$PWD/data/train.rlsd.think.jsonl"
export SFT_DATA="$PWD/data/train.sft.think.jsonl"
export MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
bash examples/action_recognition/run_qwen25.sh check-video
```

默认使用当前节点 GPU 0，解码一个训练视频，并检查：

- 学生输入、训练 worker 与 rollout 输入的 pixel tensors 一致。
- 实际采样 FPS 和视频 token grid 正确传递。
- Qwen2.5-VL 的位置 ID 与服务器安装的 vLLM 计算结果逐项一致。
- 加载原版 7B 权重，进行一次有监督前向和反向，loss 与梯度有限且梯度非零。

该检查没有优化器更新，不保存训练权重。成功时出现 `GPU forward/backward passed` 和
`Video preprocessing passed`。未出现这两行或进程非零退出时，先查看异常，不继续下面训练。

扩大到每类一个视频的解码/预处理检查（仍只对第一个视频做一次反向）：

```bash
CHECK_SAMPLES=51 bash examples/action_recognition/run_qwen25.sh check-video
```

## 3. 启动两节点 Ray

默认主节点 `192.0.2.12`，工作节点 `192.0.2.13`。脚本检查当前机器是否拥有对应 IP。
如容器重建导致 IP 变化，先在两节点分别执行 `hostname -I`，然后在两边终端设置相同的：

```bash
export HEAD_HOST=实际主节点IP
export WORKER_HOST=实际工作节点IP
```

主节点执行：

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
bash examples/action_recognition/run_qwen25.sh head
```

工作节点执行：

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
bash examples/action_recognition/run_qwen25.sh worker
```

脚本不使用 SSH，不自动终止既有 Ray 任务。如果提示 Ray 已运行或端口占用，先确认该实例属于哪个任务。
两节点各需要 8 张可用 GPU，并能访问相同的代码、模型和数据路径。

## 4. 仅在主节点执行 2 步训练验证

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
bash examples/action_recognition/run_qwen25.sh smoke
```

先校验节点 IP、GPU 资源、Python/依赖版本、源码哈希、标注哈希、模型配置哈希，然后执行 16 rank NCCL
all_reduce（每张 GPU 的结果应为 136）。全部通过才启动唯一一个训练 driver。

短步验证使用 8 帧、最多 512 个生成 token，执行 2 个训练 step；算法和完整训练一致。
可检验视频解码、vLLM rollout、教师评分、独立 reference loss、FSDP 反向和最终 checkpoint 保存。
程序完成 2 步并正常退出才算短步训练通过；CPU 测试和 NCCL 成功不能代替此项。

日志：`/gemini-3/space/zwj/AR-RLSD/outputs/logs/ar_qwen25_think_smoke_时间戳.log`。
checkpoint：`/gemini-3/space/zwj/AR-RLSD/outputs/ar_qwen25_think_smoke_时间戳/`。
即使周期保存被关闭，现有训练器仍在结束时保存 checkpoint。

可单独运行集群检查：`bash examples/action_recognition/run_qwen25.sh cluster`。

已配置好两节点 Ray 的用户，确认两节点均能读取新版工程后，在主节点执行下面的两步检查。
它保留正式配置的 16 帧、768 token 和 CUDA graph 设置，只缩短训练步数：

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
export TRAIN_DATA="$PWD/data/train.rlsd.think.jsonl"
export MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
export RUN_NAME="ar_qwen25_think_check_$(date +%Y%m%d_%H%M%S)"
bash examples/action_recognition/run_qwen25.sh train trainer.max_steps=2
```

两步检查的余弦学习率按两步计算，仅检查执行和显存；正式训练使用新任务名从原版权重开始。

## 5. 正式训练

完成短步验证后，在主节点的 tmux 等持久终端执行：

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
export TRAIN_DATA="$PWD/data/train.rlsd.think.jsonl"
export MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
export RUN_NAME="ar_qwen25_think_train_$(date +%Y%m%d_%H%M%S)"
bash examples/action_recognition/run_qwen25.sh train
```

默认从原版模型开始新运行：1 epoch、最多 16 帧、2 FPS 目标采样率、最多 768 个生成 token。
输入是 16 个视频，每个视频生成 4 条轨迹，总共 64 条，每 GPU 4 条；microbatch 为 1。
默认新运行名带时间戳，输出已存在时拒绝覆盖。Ray 集群沿用上一步启动的集群。

可在末尾传训练配置覆盖，例如先跑 20 步：

```bash
bash examples/action_recognition/run_qwen25.sh train trainer.max_steps=20
```

## 沿用的 v3 / MeSD 参数

参考文件：`/Users/zwj/Desktop/My_Project/MeSD/configs/train_mesd.yaml`。
对齐清单与来源哈希保存在 `docs/ar/mesd_config_alignment.json`。

| 参数 | v2 | v3 |
|---|---|---|
| 视频最多帧数 | 32 | 16 |
| 视频每帧 min/max pixels | 262144 | 100352 |
| 最多生成 token | 1024 | 768 |
| 学习率调度 | constant | cosine，1e-6 衰减到 0，无 warmup |
| 梯度范数裁剪 | 1.0 | 5.0 |
| Policy ratio 裁剪上下界 | 0.8 / 1.28 | 0.8 / 1.2 |
| top-p / top-k | 1.0 / -1 | 0.95 / 50 |
| vLLM 显存预算比例 | 0.60 | 0.50 |
| actor 参数/优化器 offload | 开启 | 关闭 |
| teacher FSDP CPU offload | 开启 | 关闭 |
| 训练轮数 | 10 | 1，4676 条数据约 292 step |

16 视频/batch、每视频 4 条轨迹、单卡 microbatch 1、AdamW、学习率 1e-6 和 weight decay 0.01 原本已经一致。
显式记录 actor FP32 权重/BF16 计算和 teacher BF16；每 10 步保存、保留 5 份 checkpoint 沿用已确认设置。

AR 保留原版 Qwen2.5-VL、HMDB51 数据字段、超长输入报错、每步复制并冻结教师、RLSD lambda=0.5、
参考损失系数 0.1、参考目标长度 1024 和独立任务目录。未移植 MeSD 的 EMA、多视角融合、RKL 路由、
1000 步退火、STGR 数据加载参数或自动恢复设置。AR loader 仍为 8 个 workers。
`max_model_len` 自动使用输入上限加生成上限，当前为 17152；`max_token_len_per_gpu` 在固定 microbatch 模式下不使用。

v4 不改变本节的 v3 超参数。新格式仅用于新启动的任务；旧 step292 模型仍对应旧版训练。正式对照从原版模型开始，并使用新的运行名。

## 路径与环境

| 项目 | 默认值 |
|---|---|
| 环境 | `/path/to/workspace/zwj/.venvs/open-o3-visd-vllm` |
| 工程 | `/path/to/workspace/zwj/AR-RLSD-think` |
| 原版模型 | `/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct` |
| RLSD 标注 | 工程下 `data/train.rlsd.think.jsonl` |
| SFT 标注 | 工程下 `data/train.sft.think.jsonl` |
| 视频 | `/gemini-3/space/zwj/LMUData/HMDB51/videos` |
| 输出根目录 | `/gemini-3/space/zwj/AR-RLSD/outputs` |
| Ray / NCCL 检查端口 | `6379` / `29511` |

可用环境变量覆盖 `TRAIN_ENV_ROOT`、`MODEL_PATH`、`TRAIN_DATA`、`SFT_DATA`、`OUTPUT_ROOT`、`HEAD_HOST`、
`WORKER_HOST`、`RAY_PORT`、`AR_NCCL_PORT`；`CHECK_GPU` 选择单卡检查使用的 GPU。
模型路径和标注路径的覆盖在两节点须一致。无需重装依赖或执行 `pip install -e .`。

如果需要把标注也放入 LMUData，复制到约定的 `HMDB51/annotations/ar_think_v4` 后，将两节点
`TRAIN_DATA` / `SFT_DATA` 分别指向那里的 JSONL。视频路径已经正确，不需要重写标注。

## 实现和验证范围

共享视频处理位于 `verl/utils/video.py`。数据集、教师 prompt、FSDP worker、vLLM 和 SFT 使用同一组帧与采样率。
Qwen2.5-VL 使用 Processor 的 patch size 和采样 FPS；Qwen3-VL 保留 metadata 通路。
旧缓存若不是当前模型所需的尺寸倍数会报错，避免静默接受错配缓存。

视频时间位置从模型的 `vision_config.tokens_per_second` 读取；保留小数采样时间，最后再取整数，
与固定使用的 [vLLM 0.11.0 mRoPE 实现](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/layers/rotary_embedding/mrope.py) 对齐。
Qwen2.5-VL 的帧/FPS 输入依据 [Transformers 4.57.3 Processor](https://github.com/huggingface/transformers/blob/v4.57.3/src/transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py)，
vLLM 模型处理依据 [vLLM 0.11.0 Qwen2.5-VL](https://github.com/vllm-project/vllm/blob/v0.11.0/vllm/model_executor/models/qwen2_5_vl.py)。

旧版 33 项 CPU 测试及本次新增检查共 46 项通过，覆盖实际 Transformers patch reshape、FPS 和教师/actor/rollout 传递、
SFT mask/位置、batch 拒绝条件，以及真实 shell 启动器在替身 Python/Ray 下的参数传递和失败阻断。
本机无 CUDA、真实视频和模型权重；此前 GPU 成功记录来自用户服务器截图，不属于本地测试结果。
数据仍为现有生成标注；视频路径检查不代表动作树内容已经人工逐视频审核。

## v4 格式修改与数据校验

- `reference_response` 的 `<action_tree>` / `</action_tree>` 改为 `<think>` / `</think>`，内部缩进树逐字符保留。
- 学生和教师仍要求层级动作分析，最终类别仍在 `<answer>`；教师 context 的树证据保留。
- 参考损失保持系数 0.1、clipped 模式；分类 reward、RLSD 调权和其余训练参数保持不变。
- 新数据 `train.rlsd.think.jsonl` / `train.sft.think.jsonl` 各 4,676 条、51 类。原始数据保留于 `train.rlsd.jsonl` / `train.sft.jsonl`。
- 启动前显式设置上文的 `TRAIN_DATA` / `SFT_DATA`，避免终端继承旧路径。正式训练使用原版 `MODEL_PATH` 和新的 `RUN_NAME`。

```bash
cd /path/to/workspace/zwj/AR-RLSD-think
source /path/to/workspace/zwj/.venvs/open-o3-visd-vllm/bin/activate
python -m verl.ar.prepare_data --check data/train.rlsd.think.jsonl --check-media
```

该检查验证文本契约和视频路径。服务器两步训练还需查看生成样本是否实际使用 `<think>` 并保持层级组织，最终答案是否正常解析；格式指令不保证每次生成都合规。评测继续使用同一套 VideoSTAR-Eval-v5。
