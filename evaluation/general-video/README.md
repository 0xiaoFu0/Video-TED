# 通用视频理解评测：MVBench / Video-MME / TempCompass-MCQ / SEED-Bench Video

**v1.2.2 新增简短 prompt 全套评测。** `simple_64` 用于 MVBench / Video-MME / TempCompass-MCQ，沿用原采样与生成预算，最多 64 帧；`seedbench_simple_8` 对 SEED-Bench 的官方 8 帧使用同一份简短 prompt。完整模型路径、参数和运行记录见 [SIMPLE64.md](SIMPLE64.md)。

**v1.2.1 新增 `simple_32`。** 三个原视频 benchmark 使用现有简短推理 prompt，保持 `videor1_64` 的 2 FPS 向下取偶数均匀采样、像素与生成设置，帧数上限改为 32。启动命令和完整 prompt 见 [SIMPLE32.md](SIMPLE32.md)。这项消融同时改变 prompt 和帧数，不能把分数变化单独归因于其中一项；`generic` 默认采样方式与它不同。

**v1.2.0 新增 SEED-Bench Video。** 使用官方每题 8 张预抽帧，单独以 `EVAL_PROFILE=seedbench_8` 运行，输出 Video Overall（3757 题）和 Action Recognition（1509 题）。保留 Video-R1 prompt、像素与生成预算，但不把预抽帧重复扩展为 64 帧。上传包、全量图片预检、试跑及正式评测指令见 **[SEEDBENCH.md](SEEDBENCH.md)**。

独立评测包，沿用 MeSD 的环境变量和 `eval/scripts/eval_all.sh` 启动方式。支持本地 Qwen2.5-VL、Qwen3-VL（含 MoE）Hugging Face checkpoint，通过 vLLM 推理。多卡默认每卡一个模型副本，全部 benchmark 共用同一批模型进程。无需训练仓库、API judge 或 base/novel/HM 配置。

**v1.1.0 新增两模型统一比较协议。** `EVAL_PROFILE=videor1_64` 参考 Video-R1 的评测 prompt 和论文声明的 64 帧推理档位。两个 benchmark 均使用 **64 帧上限**、目标 2 FPS、全片/标注区间内均匀采样；短片实际输入可少于 64 帧。原 `generic` 协议保留，旧的 65.55 / 57.78 结果属于原设置，不能当作本次新协议结果。

## 两模型统一评测：当前 checkpoint 与原始基座

把 `GeneralVideo-Eval-v1.1.0.zip` 上传到 `/path/to/workspace/zwj/` 后，在服务器运行：

```bash
(
set -e
unzip -o /path/to/workspace/zwj/GeneralVideo-Eval-v1.1.0.zip \
  -d /path/to/workspace/zwj/GeneralVideo-Eval-v1.1.0/
cd /path/to/workspace/zwj/GeneralVideo-Eval-v1.1.0/GeneralVideo-Eval

BASE_MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct \
OURS_MODEL_PATH=/gemini-3/space/zwj/AR-action-training-32gpu-v3/outputs/ar_qwen25_action93583_32gpu_mesd_ema010_ref010_train_20261013_030523_714650/global_step_2924/actor/huggingface \
MVBENCH_DATA_DIR=/path/to/workspace/datasets/MVBench \
VIDEOMME_DATA_DIR=/gemini-3/space/zwj/LMUData/VideoMME \
BENCHMARKS=mvbench,videomme BACKEND=vllm \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 TP_SIZE=1 \
EXP_PREFIX=ar_step2924_vs_qwen25vl7b_videor1_64 \
bash eval/scripts/eval_pair_videor1_64.sh
)
```

基座是 **Qwen2.5-VL-7B-Instruct**。上面的默认基座路径来自此前训练配置；启动脚本会先核对两模型的 `config.json` 和实际权重文件。基座和当前模型先各做每 benchmark 5 题的真实试跑，均成功后依次全量评测基座和当前模型。任何一步失败都会停止；使用同一命令可续跑。

结果目录依次以 `_base_smoke`、`_ours_smoke`、`_base_full`、`_ours_full` 结尾。末尾会核对两次全量实验的协议、题目集合、完整状态，并在 `_ours_full/` 写出 `comparison.md` 和 `comparison.csv`，包括总分、动作分项及相对基座的百分点变化。原实验目录不受影响。

固定配置如下，已记录在每次运行的 `manifest.json` 中：

| 项目 | `videor1_64` |
| --- | --- |
| Prompt | Video-R1 深入思考、自我反思与 `<think>` / `<answer>` 指令；选项统一为 `A. ...`，问题与 Options 间明确换行 |
| 帧预算 | 所选 benchmark 均最多 64 帧，2 FPS，向下取偶数，最少 4 帧 |
| 时间范围 | 覆盖全片；MVBench 保留官方标注的 start/end 与帧目录语义 |
| 每帧像素 | 最小 100352，最大 200704（`128×28²` / `256×28²`） |
| 解码参数 | temperature 0.01，top_p 0.001，repetition_penalty 1.0，N=1 |
| 生成与上下文 | 最大生成 1024 tokens，上下文 16384 |
| 比较启动脚本 | seed 42、BF16、Decord；相同输入、解析与计分规则 |

**协议来源说明：** [Video-R1 论文 §4.1](https://arxiv.org/html/2503.21776v4#S4.SS1) 和 [README](https://github.com/tulerfeng/Video-R1#-inference--evaluation) 声明 temperature=0.01，仓库 [eval_bench.py](https://github.com/tulerfeng/Video-R1/blob/main/src/eval_bench.py) 当前写的是 0.1，本包明确选择论文值。仓库工具默认仍是 16 帧和较低像素上限，README 要求手动改推理预算。本包显式配置 64 帧上限与 200704 像素，不依赖那些隐式默认值。

该协议用于两模型在本包内的公平比较，保留现有新版 vLLM 适配、标注加载、MVBench 时间区间和确定性答案解析，并非旧版 Video-R1 运行时的逐项数值复现。TempCompass 仍只采用原生 1580 道 MCQ；Video-R1 配套 7540 条混合转换题不纳入本包。直接用 `eval_all.sh` 选择命名协议时，若遗留环境变量与固定参数冲突，会明确报错；上面的两模型脚本会显式设置协议参数，避免继承上一轮的温度、帧数或 LIMIT。

论文主表可以直接取以下总分和动作分项；分数都是百分数。

| Benchmark | 总成绩 | 动作分项 | 完整题数 |
| --- | --- | --- | ---: |
| MVBench | `avg`：20 个任务的准确率宏平均 | `fa`：Fine-grained Action | 4,000（FA 200） |
| Video-MME | `overall` | `action_recognition`：Action Recognition | 2,700 |
| TempCompass-MCQ | `overall` | `action_recognition`：Action | 1,580（Action 338） |
| SEED-Bench Video | `overall`：三种视频任务按题目汇总 | `action_recognition`：Action Recognition | 3,757（Action 1,509） |

这里的三个动作分项采用各 benchmark 的官方任务分类，含义和难度各有差异。TempCompass 仅测 `multi-choice`，论文表头请写 **TempCompass-MCQ**，不包含 caption matching、yes/no 和 captioning。Video-MME 采用 **无字幕、无音频** 的视频帧输入。

## 1. 上传和环境

把整个 `GeneralVideo-Eval` 目录上传到服务器任意位置，例如 `/path/to/workspace/zwj/GeneralVideo-Eval`。目录内的导入和脚本路径均相对本包解析。

优先激活之前能运行 MeSD 评测的环境，再执行：

```bash
cd /path/to/workspace/zwj/GeneralVideo-Eval
python -m pip install -r requirements.txt
```

vLLM 接口基线为 `vllm==0.11.0`、`transformers==4.57.3`，与参考 MeSD requirements 一致。新建环境时可使用 Python 3.11，并执行 `python -m pip install -r requirements-vllm.txt`。具体 CUDA 驱动与 PyTorch wheel 需与服务器环境兼容。本包直接解码和抽帧，不依赖 `qwen-vl-utils`。

如当前环境使用 `python` 而不是 `python3`，启动时加 `PYTHON_BIN=python`。

## 2. 数据目录

常见布局会自动识别；不扫描整个数据盘，也不会自动下载数据。

服务器目录已通过用户提供的终端输出确认：MVBench 为 `/path/to/workspace/datasets/MVBench`，Video-MME 为 `/gemini-3/space/zwj/LMUData/VideoMME`。这两项已作为默认路径；TempCompass 的默认路径仍是之前指定的下载目标，尚未确认服务器下载与解压状态。目录存在及标注文件齐全不代表全部视频已核验，完整性请运行下方预检查。

```text
MVBench/
  json/                         # 20 个官方任务 JSON
    action_sequence.json
    fine_grained_action.json
    ...
  video/                        # 也支持根目录下的原视频目录
    star/Charades_v1_480/
    Moments_in_Time_Raw/videos/
    tvqa/frames_fps3_hq/         # Episodic Reasoning，原版按帧存储
    ...

VideoMME/
  videomme/test-00000-of-00001.parquet
  data/<videoID>.mp4           # 也支持 video/data/、videos/、video/

TempCompass/
  multi-choice/test-00000-of-00001.parquet
  videos/<video_id>.mp4
```

MVBench 兼容 `video/data0613/` 解压布局、VLMEvalKit 的 `MVBench.tsv`，也兼容 OpenGVLab 预裁剪 MP4 版的 `test.json`（`root`/`meta`）和 `MVBench_MP4.tsv`。原版带起止时间的题目只抽取对应片段；MP4 转换版按已裁剪视频读取，不重复裁剪。TVQA 原版按 3 FPS、一基帧编号处理。

v1.0.2 修正了原版 MVBench 的单选项兼容问题：官方 Counterfactual Inference 包含 5 道只有一个选项的题目，全部保留在 4,000 题分母中，仍执行模型推理并解析输出。空输出或无效答案计错，不自动预测 A；Video-MME 和 TempCompass 仍要求至少两个选项。

v1.0.3 兼容服务器上 NTU RGB+D 的同名 MP4 转码文件。Fine-grained Pose 的原 AVI 不存在时，按同一视频 ID 查找同名前缀目录中的 MP4，再检查 `nturgbd_convert/`；优先保留存在的原 AVI。转码文件仍按原题目读取和抽帧，不视为预裁剪片段，也不修改标注或视频文件。

Video-MME 也支持根目录 `test*.parquet`、`Video-MME.tsv` 和官方嵌套 JSON。TempCompass 也支持原始 `questions/multi-choice.json`。使用其他标注位置可设置 `MVBENCH_ANNO_FILE`、`VIDEOMME_ANNO_FILE`、`TEMPCOMPASS_ANNO_FILE`；相对路径以对应数据根目录为基准。MVBench 覆盖位置可指向包含全部任务 JSON 的目录。

TempCompass 的下载与解压命令：

```bash
hf download lmms-eval/TempCompass \
  --repo-type dataset \
  --include "multi-choice/*" "tempcompass_videos.zip" "README.md" \
  --local-dir /gemini-3/space/zwj/LMUData/TempCompass/

unzip -n /gemini-3/space/zwj/LMUData/TempCompass/tempcompass_videos.zip \
  -d /gemini-3/space/zwj/LMUData/TempCompass/
```

只下载 ZIP 而不解压会在预检查时报缺失视频。完整 MCQ 需要 410 个视频，不需要下载其他三种题型的标注。

## 3. 先检查数据，再跑服务器小样本

先导出本次评测使用的路径：

```bash
export MODEL_PATH=/gemini-3/space/zwj/AR-action-training-32gpu-v3/outputs/ar_qwen25_action93583_32gpu_mesd_ema010_ref010_train_20261013_030523_714650/global_step_2924/actor/huggingface
export MVBENCH_DATA_DIR=/path/to/workspace/datasets/MVBench
export VIDEOMME_DATA_DIR=/gemini-3/space/zwj/LMUData/VideoMME
export TEMPCOMPASS_DATA_DIR=/gemini-3/space/zwj/LMUData/TempCompass
```

预检查会读取完整标注、核对题数与任务覆盖，并检查所有引用的视频/帧目录是否存在；不会加载模型或初始化 CUDA：

```bash
PREFLIGHT=1 BENCHMARKS=mvbench,videomme,tempcompass \
  bash eval/scripts/eval_all.sh
```

接着用一张 GPU、每个 benchmark 前 5 道题，确认服务器的模型、vLLM 和解码器组合能工作：

```bash
CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
BENCHMARKS=mvbench,videomme,tempcompass \
EXP_NAME=server_smoke LIMIT=5 THINK_MODE=1 N=1 \
  bash eval/scripts/eval_all.sh
```

`LIMIT` 对每个 benchmark 生效；小样本结果标为 `partial_or_nonstandard`，只用于验证运行。MVBench 前 5 题通常不包含 FA，该分项显示为空；全量运行才有完整分项。

## 4. 三个 benchmark 的正式八卡命令

```bash
cd /path/to/workspace/zwj/GeneralVideo-Eval

MODEL_PATH=/gemini-3/space/zwj/AR-action-training-32gpu-v3/outputs/ar_qwen25_action93583_32gpu_mesd_ema010_ref010_train_20261013_030523_714650/global_step_2924/actor/huggingface \
MVBENCH_DATA_DIR=/path/to/workspace/datasets/MVBench \
VIDEOMME_DATA_DIR=/gemini-3/space/zwj/LMUData/VideoMME \
TEMPCOMPASS_DATA_DIR=/gemini-3/space/zwj/LMUData/TempCompass \
BENCHMARKS=mvbench,videomme,tempcompass \
EXP_NAME=ar_qwen25_step2924_general_8gpu \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NUM_GPUS=8 THINK_MODE=1 N=1 \
  bash eval/scripts/eval_all.sh
```

只测目前已有的两个数据集，改为 `BENCHMARKS=mvbench,videomme`。数据路径只检查选中的 benchmark。较大模型需要跨卡放置时设置 `TP_SIZE=2` 或 `4`；例如八卡、`TP_SIZE=2` 会启动四个模型副本。`NUM_GPUS` 必须与 `CUDA_VISIBLE_DEVICES` 的条目数一致，并能被 `TP_SIZE` 整除。

## 5. 参数与统一比较协议

| 环境变量 | 默认值 | 含义 |
| --- | --- | --- |
| `EVAL_PROFILE` | `generic` | 原协议；`videor1_64` 采用上方固定的 Video-R1 参考协议，下面表格列出 generic 默认值 |
| `THINK_MODE` | `1` | `1`：通用视频证据推理 + `<answer>`；`0`：直接输出选项字母 |
| `N` | `1` | 每题采样数；多次采样采用多数投票，平票取最先出现的有效答案 |
| `TEMPERATURE` | N=1 时 0，否则 0.7 | 贪心或随机采样；N>1 时必须大于 0 |
| `MAX_NEW_TOKENS` | think 时 2048，否则 128 | 生成长度上限 |
| `MAX_MODEL_LEN` | `32768` | 输入视频/文字与生成合计的上下文上限 |
| `MVBENCH_MAX_FRAMES` | `32` | MVBench 按参考代码在分段中点抽帧 |
| `VIDEOMME_MAX_FRAMES` | `64` | Video-MME 均匀抽帧上限 |
| `TEMPCOMPASS_MAX_FRAMES` | `32` | TempCompass 均匀抽帧上限 |
| `FPS` / `MIN_FRAMES` | `2` / `4` | 均匀抽帧目标 FPS / 最少帧数；短视频允许重复帧 |
| `MAX_PIXELS` / `MIN_PIXELS` | `262144` / `3136` | 每帧空间像素预算，按模型 patch 合并尺寸对齐 |
| `GPU_MEMORY_UTILIZATION` | `0.8` | vLLM 显存占用比例 |
| `DTYPE` / `SEED` | `bfloat16` / `42` | 模型精度 / 随机种子 |
| `REPETITION_PENALTY` / `TOP_P` | `1.05` / `1` | 生成参数 |
| `DECODER` | `decord` | 可显式设为 `opencv`，需安装 `opencv-python-headless` |
| `WORKER_TIMEOUT_S` | `3600` | 单个 worker 最长无进度时间，含模型加载及单题推理 |
| `OUTPUT_DIR` | `outputs` | 输出根目录 |
| `RESUME` | `1` | 自动跳过同实验已保存的题目 |

`MAX_FRAMES` 可以统一覆盖三个默认上限，单个 benchmark 的变量优先。帧数必须为大于等于 2 的偶数。解码只保留选中的帧，并使用 Pillow bicubic 缩放一次；这不是原 MeSD 各 benchmark 旧配置的逐项数值复现。公平比较时，同一 benchmark 的所有模型必须使用相同抽帧、像素、thinking 与生成配置；与他人论文成绩比较时同时注明该协议。

Qwen3 输入保留原始 FPS、帧索引和时间信息；Qwen2.5 使用抽样后的有效 FPS。实际抽帧索引、时间戳和尺寸都保存在逐题结果中。遇到解码失败会记录错误并退出，不会切换为整段视频解码或跳过该题。

## 6. 输出、续跑与计分

```text
outputs/<EXP_NAME>/
  manifest.json               # 模型/参数/依赖版本/源码与数据指纹
  dataset.jsonl               # 完整的本次待评测题目，含标准答案供离线计分
  predictions/worker-*.jsonl   # 原始输出、解析答案、投票结果、抽帧信息
  logs/worker-*.log            # 模型加载、推理和异常栈
  errors.json                 # 基础设施错误；成功时为空
  summary.json                # 总成绩 + 全部分项 + 分母与完成状态
  summary.csv                 # 主表指标；MVBench 取 avg/fa
  summary.md                  # 可直接查看的汇总表
```

`summary.json` 中还包含 MVBench 的 `by_task`，Video-MME 的 `by_task/by_duration/by_domain`，TempCompass 的 `by_dimension`。总正确数、题数、无效输出数和完成状态可用于核对结果。

重复执行**相同的正式命令**即可续跑。已完成题目逐条落盘，失败题不会写成成功记录；最后一条被中断的 JSONL 可以恢复。修改模型、源码、数据或推理协议后必须使用新的 `EXP_NAME`，避免混入旧结果。同一个实验只允许一个写入/计分进程。

只重新统计已有结果，无需 GPU、模型或原数据目录：

```bash
SCORE_ONLY=1 EXP_NAME=ar_qwen25_step2924_general_8gpu \
  bash eval/scripts/eval_all.sh
```

答案解析支持 `<answer>B</answer>`、裸字母、明确的最终答案陈述和精确选项文本。无效输出计错，不默认变成 A；未完成/失败题始终保留在分母中。worker 报错、无进度超时或收到中断时会停止其余 worker，保存已有结果并返回非零状态；SIGTERM 会清理 worker 进程组。

只有 `scope=full`、`complete=true` 且 `is_mock=false` 的结果才适合作为本包的完整模型成绩。默认要求官方题数和任务覆盖；有意使用自定义标注时可设置 `ALLOW_NONSTANDARD_COUNTS=1`，不符合标准覆盖的结果会明确标为 `partial_or_nonstandard`。

## 7. 本地验证与来源

```bash
python -m unittest discover -s tests -v
bash -n eval/scripts/eval_all.sh
```

`BACKEND=mock` 仅用于 CPU 管线验证：固定输出 A，并在汇总中明确标识 mock，不使用标准答案生成输出。完整验证范围见 [VALIDATION.md](VALIDATION.md)。当前交付已验证 CPU 数据/计分/多进程流程和真实视频解码；CUDA/vLLM 模型推理需要执行上面的服务器小样本命令确认。

实现参考了用户指定的 MeSD、Open-o3-VISD-vLLM 评测入口和 Eval_tele 内的 MVBench 数据定义，未修改这些参考目录。公开接口与任务定义来源：

- [MVBench 官方实现](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2)
- [Video-MME 官方项目](https://github.com/MME-Benchmarks/Video-MME)
- [TempCompass 官方项目](https://github.com/llyx97/TempCompass)
- [TempCompass Hugging Face 数据](https://huggingface.co/datasets/lmms-eval/TempCompass)
- [vLLM 0.11 Qwen3-VL 接口](https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/models/qwen3_vl.html)
