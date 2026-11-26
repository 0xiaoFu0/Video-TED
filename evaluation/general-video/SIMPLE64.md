# TreeSD-7B step2924：简短 prompt，四个 benchmark 全量评测

记录日期：2026-10-17。模型目录采用用户此前已验证的 Hugging Face 导出目录。

| Benchmark | 全量题数 | 动作分项题数 | 输入帧数 |
| --- | ---: | ---: | --- |
| MVBench | 4000 | Fine-grained Action 200 | 最多 64 |
| Video-MME，无字幕/无音频 | 2700 | Action Recognition 313 | 最多 64 |
| TempCompass-MCQ | 1580 | Action 338 | 最多 64 |
| SEED-Bench Video | 3757 | Action Recognition 1509 | 官方固定 8 |

前三项采用 `simple_64`，目标 2 FPS、最少 4 帧、`uniform_fps_floor` 均匀采样，保留官方片段范围。64 是上限，短片可以更少；达到上限时仍覆盖全片/标注片段。
SEED 单独采用 `seedbench_simple_8`，按官方已抽取的 8 帧顺序读取，不重复扩帧或二次裁剪。其接口时间信息仍标记为合成序号，非源视频时间戳。

四项均使用以下简短 prompt，保留简要推理与答案标签；模型自身聊天模板不变：

```text
Select the best answer based on the video. Reason briefly using only visible video evidence inside <think>...</think>, then output the option letter inside <answer>...</answer>.
Question: {question}
Options:
(A) {choice_A}
(B) {choice_B}
...
```

将 `GeneralVideo-Eval-v1.2.2.zip` 上传到 `/path/to/workspace/zwj/`，在已有可用评测环境中执行完整命令：

```bash
(
set -e
unzip -n /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.2.zip \
  -d /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.2
cd /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.2/GeneralVideo-Eval

export PYTHON_BIN=python3
export MODEL_PATH=/gemini-3/space/zwj/AR-action-training-32gpu-v3/outputs/ar_qwen25_action93583_32gpu_mesd_ema010_ref010_train_20261013_030523_714650/global_step_2924/actor/huggingface
export MVBENCH_DATA_DIR=/path/to/workspace/datasets/MVBench
export VIDEOMME_DATA_DIR=/gemini-3/space/zwj/LMUData/VideoMME
export TEMPCOMPASS_DATA_DIR=/gemini-3/space/zwj/LMUData/TempCompass
export SEEDBENCH_DATA_DIR=/gemini-3/space/zwj/LMUData/SEED-Bench
unset MVBENCH_ANNO_FILE VIDEOMME_ANNO_FILE TEMPCOMPASS_ANNO_FILE SEEDBENCH_ANNO_FILE

export OUTPUT_DIR="$PWD/outputs"
export BACKEND=vllm
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 TP_SIZE=1
export THINK_MODE=1 N=1
export TEMPERATURE=0.01 TOP_P=0.001 REPETITION_PENALTY=1.0
export MAX_NEW_TOKENS=1024 MAX_MODEL_LEN=16384
export FPS=2 MIN_FRAMES=4 MIN_PIXELS=100352 MAX_PIXELS=200704
export VOTE=majority_voting SEED=42 DTYPE=bfloat16 DECODER=decord
export GPU_MEMORY_UTILIZATION=0.8 TRUST_REMOTE_CODE=0
export PREFLIGHT=0 SCORE_ONLY=0 LIMIT=0 RESUME=1 ALLOW_NONSTANDARD_COUNTS=0 VERIFY_FRAME_IMAGES=0

# 第一阶段：三个视频 benchmark，全量 8280 题，简短 prompt、最多 64 帧。
BENCHMARKS=mvbench,videomme,tempcompass EVAL_PROFILE=simple_64 \
MAX_FRAMES=64 MVBENCH_MAX_FRAMES=64 VIDEOMME_MAX_FRAMES=64 TEMPCOMPASS_MAX_FRAMES=64 \
EXP_NAME=treesd7b_step2924_v122_simple64_three_full \
  bash eval/scripts/eval_all.sh

# 第二阶段：SEED-Bench Video，全量 3757 题，相同简短 prompt、官方 8 帧。
BENCHMARKS=seedbench EVAL_PROFILE=seedbench_simple_8 \
MAX_FRAMES=8 SEEDBENCH_MAX_FRAMES=8 \
EXP_NAME=treesd7b_step2924_v122_seedbench_simple8_full \
  bash eval/scripts/eval_all.sh
)
```

两阶段依次使用同一台八卡机器，每卡一个模型副本。任何一阶段失败时命令停止；相同代码与配置下重跑可续跑，已完成的阶段不会重复预测。新实验名保留已有结果；评测仍自动核对官方题数、任务覆盖与媒体路径。

也可设置上述模型、数据、输出与 GPU 环境后执行专用启动器，得到相同参数和实验名：

```bash
EXP_PREFIX=treesd7b_step2924_v122 bash eval/scripts/eval_simple64_all.sh
```

结果查看：

```bash
cat /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.2/GeneralVideo-Eval/outputs/treesd7b_step2924_v122_simple64_three_full/summary.md
cat /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.2/GeneralVideo-Eval/outputs/treesd7b_step2924_v122_seedbench_simple8_full/summary.md
```

最终记录四个 benchmark 的 Overall/Avg 和对应动作分项，共八个百分数。请连同 `manifest.json`、`summary.json` 保存；本次是简短 prompt 配置，和原先 Video-R1 prompt 的成绩分别记录。
