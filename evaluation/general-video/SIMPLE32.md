# Video-R1：简短 prompt，最多 32 帧

`simple_32` 用于 MVBench（4000）、Video-MME 无字幕（2700）、TempCompass-MCQ（1580），共 8280 题。
它沿用 `videor1_64` 的 `uniform_fps_floor` 采样、2 FPS 目标、最少 4 帧和标注片段边界，三个 benchmark 的帧数上限均为 32。短视频可以少于 32 帧；达到上限时仍覆盖完整视频/标注片段。

使用当前包已有的 generic 简短推理 prompt（`THINK_MODE=1`）：

```text
Select the best answer based on the video. Reason briefly using only visible video evidence inside <think>...</think>, then output the option letter inside <answer>...</answer>.
Question: {question}
Options:
(A) {choice_A}
(B) {choice_B}
...
```

每帧像素 100352–200704；temperature=0.01、top_p=0.001、repetition_penalty=1、N=1；最大生成 1024 tokens、上下文 16384；seed=42、BF16、Decord。模型仍使用自身的 chat template。这里只更换问题指令与选项格式、帧数上限。

将 `GeneralVideo-Eval-v1.2.1.zip` 上传到 `/path/to/workspace/zwj/`，然后执行：

```bash
(
set -e
unzip -n /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.1.zip \
  -d /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.1
cd /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.1/GeneralVideo-Eval

export PYTHON_BIN=python3
export MODEL_PATH=/gemini-3/space/zwj/models/Video-R1-7B
export MVBENCH_DATA_DIR=/path/to/workspace/datasets/MVBench
export VIDEOMME_DATA_DIR=/gemini-3/space/zwj/LMUData/VideoMME
export TEMPCOMPASS_DATA_DIR=/gemini-3/space/zwj/LMUData/TempCompass
unset MVBENCH_ANNO_FILE VIDEOMME_ANNO_FILE TEMPCOMPASS_ANNO_FILE
export OUTPUT_DIR="$PWD/outputs"
export BACKEND=vllm
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 TP_SIZE=1
export GPU_MEMORY_UTILIZATION=0.8
export EXP_NAME=videor1_7b_simple32_three_v121_full

bash eval/scripts/eval_simple32.sh
)
```

专用脚本显式设置协议及全量参数，避免继承上次的 64 帧、8192 tokens、SEED、LIMIT 或仅计分状态。
运行前使用现有路径预检；不额外加载一次模型。八卡每卡一个模型副本，共用这批进程评测三个 benchmark。相同代码、模型和配置可用同一命令续跑。

启动日志应显示：

```text
Evaluation profile: simple_32; prompt=generic; sampling=uniform_fps_floor; frame caps={'mvbench': 32, 'videomme': 32, 'tempcompass': 32}; temperature=0.01; top_p=0.001; max_new_tokens=1024
```

结果查看：

```bash
cat /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.1/GeneralVideo-Eval/outputs/videor1_7b_simple32_three_v121_full/summary.md
```

保留旧版目录用于原实验续跑。新版本使用独立目录和新实验名；旧结果不会被替换。`simple_32` 会拒绝 SEED-Bench，它的官方预抽帧包应继续单独使用 `seedbench_8`。

这次同时调整 prompt 和帧数，属于一套新评测配置。若用于模型间比较，需让比较模型也采用这套配置；不能将变化单独解释为 prompt 的效果或帧数的效果。
