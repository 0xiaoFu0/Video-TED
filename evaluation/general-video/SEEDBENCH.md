# SEED-Bench Video 评测

v1.2.0 接入 SEED-Bench-1 的视频部分，过滤所有图像题。主表输出两列：

| 指标 | 题数 | 计分 |
| --- | ---: | --- |
| SEED-Bench Video Overall (8 frames) | 3757 | 全部视频题的正确率，按题目汇总 |
| SEED-Bench Action Recognition | 1509 | question_type_id=10 的正确率 |

另在 summary.json 的 by_task 保留 Action Prediction（1225）及 Procedure Understanding（1023）。缺失/无效回答按错误计分，不缩小分母。仅 scope=full、complete=true、is_mock=false 才是完整模型结果。

## 数据布局与协议

官方 ZIP 中央目录已与官方 SEED-Bench.json 逐题核对：3757 个题目目录，30056 张 PNG，全部一一匹配。详细来源与哈希见 [mapping audit](validation/seedbench_mapping_audit.json)。代码按 question_id 读取，不用可能重复的源视频 data_id；已对应题目区间的帧不再按 segment 裁剪。

```text
/gemini-3/space/zwj/LMUData/SEED-Bench/
  SEED-Bench.json
  v1_video/
    task10/ssv2_8_frame/v1/1.png ... 8.png
    task11/kitchen_8_frame/v1743/1.png ... 8.png
    task12/breakfast_8_frame/v3817/1.png ... 8.png
```

`seedbench_8` 采用现有 Video-R1 thinking prompt、temperature=0.01、top_p=0.001、N=1、最大输出1024 tokens、上下文16384、每帧100352–200704 pixels。**按1.png到8.png的顺序输入全部8帧，不再采样、补帧或扩展到64帧。** 此配置只能用于 BENCHMARKS=seedbench；其他三个 benchmark 继续使用原来的 videor1_64。

帧包没有逐帧原始时间戳。传给 Qwen 的2 FPS只是统一的序号时间基准，不表示原始FPS或视频实际长4秒，manifest和逐题video_info明确标记source_timing_known=false与synthetic_ordinal_2fps。所有待比较模型应使用相同配置。本包使用生成答案后的MCQ正确率，不宣称复现原SEED论文的候选答案似然评分。

## 上传与启动

把 GeneralVideo-Eval-v1.2.0.zip 上传到 /path/to/workspace/zwj/，在已有可用的 vLLM 环境执行。代码不需要重新安装，也不需要更换当前模型环境；SEED图片读取使用已在requirements中的Pillow。

```bash
(
set -euo pipefail
unzip -o /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.0.zip \
  -d /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.0/
cd /path/to/workspace/zwj/GeneralVideo-Eval-v1.2.0/GeneralVideo-Eval

export PYTHON_BIN=python3
export SEEDBENCH_DATA_DIR=/gemini-3/space/zwj/LMUData/SEED-Bench
unset SEEDBENCH_ANNO_FILE
export MODEL_PATH=/gemini-3/space/zwj/models/Video-R1-7B
export EXP_NAME=videor1_7b_seedbench8_v120
export BACKEND=vllm ALLOW_NONSTANDARD_COUNTS=0
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NUM_GPUS=8 TP_SIZE=1
export OUTPUT_DIR="$PWD/outputs"
bash eval/scripts/eval_seedbench.sh run
)
```

`run` 依次执行：

1. 全量标注/逐题8帧检查，并逐张解码全部30056张图片；每200题打印进度，不加载模型。
2. 五题真实模型试跑，至少覆盖三个子任务各一题；所有题完成且每个子任务至少有一个可解析答案后继续。
3. 3757题正式评测，8卡默认每卡一个副本，结果逐条落盘。

只检查数据：`bash eval/scripts/eval_seedbench.sh preflight`，不需要MODEL_PATH。只试跑：末尾换成`smoke`。已通过预检/试跑后续跑全量：末尾换成`full`，相同EXP_NAME会跳过已完成题目。更换模型必须更换EXP_NAME。

模型路径可替换为：

```bash
# Qwen2.5-VL-7B
export MODEL_PATH=/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct
export EXP_NAME=qwen25vl7b_seedbench8_v120

# TreeSD-7B（自己的模型）
export MODEL_PATH=/gemini-3/space/zwj/AR-action-training-32gpu-v3/outputs/ar_qwen25_action93583_32gpu_mesd_ema010_ref010_train_20261013_030523_714650/global_step_2924/actor/huggingface
export EXP_NAME=treesd7b_step2924_seedbench8_v120
```

上述为各模型分别使用的变量示例；选择一组后执行同一个run命令。

## 查看及比较结果

正式结果位于 `outputs/${EXP_NAME}_full/summary.md`，同时输出CSV/JSON、manifest、逐题预测和worker日志。成功标准：Questions=3757、Completed=3757、Scope=full、Complete=True；Invalid仍应单独检查。

同一版代码与环境下跑完Qwen和自己的模型后：

```bash
python3 -m general_video_eval.compare \
  outputs/qwen25vl7b_seedbench8_v120_full \
  outputs/treesd7b_step2924_seedbench8_v120_full
```

比较器会核对两次实验的题目、协议、依赖、源码、完整状态及1509题动作分母，输出comparison.md和comparison.csv。旧版本其他benchmark结果无需改动；不要把不同版本/协议结果伪装成严格配对比较。

本地验证包含CPU管线和实际合成PNG解码；开发机没有CUDA/vLLM，服务器真实模型兼容性由上述smoke步骤确认。官方源帧和服务器解压结果的实际完整性以服务器preflight为准。
