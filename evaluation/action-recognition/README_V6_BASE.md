# VideoSTAR-Eval v6：单轮、无工具、Base/Novel 与三个 split

每个模型评测 HMDB51 / UCF101 / SSv2 Base、Novel，并评测 HMDB51 / UCF101 的 split1、split2、split3 全类别。
共十二个划分、45,893 次单轮识别；全部完成后写入一行，包含三组 Base / Novel / HM、六个 split 准确率，以及两个数据集各自的三划分平均和标准差。

原始标注、候选类别、视频处理源码和计分来自固定官方提交。原模型与训练模型使用相同协议。
v6 使用 `protocol.py` 中的 `RECOGNITION_TEMPLATE`：视频证据摘要 + 一个候选类别，不再沿用工具第二轮的身体部位映射、候选评分或类别解释指令。单轮、关闭工具和新的 prompt 是明确的推理改动，不能称为完整复现论文的工具增强 pipeline。
**官方 HMDB 两套标注中有 120 条标签不一致；本包保留各自官方原始标签，不擅自修正。**
详细核对、样本数、已知边界见 [PROTOCOL.md](PROTOCOL.md)，逐条差异见 `ALIGNMENT.json`。

## 上传和校验

上传 `VideoSTAR-Eval_20261004_v6_prompt_update.tar.gz` 到服务器 `/path/to/workspace/zwj/`：

```bash
cd /path/to/workspace/zwj
tar -xzf VideoSTAR-Eval_20261004_v6_prompt_update.tar.gz
cd VideoSTAR-Eval-v5
sha256sum -c SHA256SUMS --quiet
```

此更新包仍解压到 `VideoSTAR-Eval-v5/`，更新代码和 prompt 示例；协议与默认结果目录升级为 v6。旧结果保留，不能用于新任务续跑。更新前先结束正在使用该代码目录的任务。两台服务器确保能访问相同版本；共享挂载只需解压一次。
仍使用 `/path/to/workspace/zwj/.venvs/open-o3-visd-vllm/bin/python`。如环境目录不同，用 `EVAL_PYTHON` 指定现有环境 Python，无需重装训练环境。

## 第一台八卡机器：原版 Qwen2.5-VL

```bash
cd /path/to/workspace/zwj/VideoSTAR-Eval-v5
bash run_v6.sh base
```

这里的 `base` 指原版模型，命令仍会运行全部十二个评测划分。
模型为 `/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct`。

## 第二台八卡机器：已合并的训练后模型

```bash
cd /path/to/workspace/zwj/VideoSTAR-Eval-v5
bash run_v6.sh trained
```

默认模型为旧的 `/gemini-3/space/zwj/AR-RLSD/eval_models/ar_v3_025233_step292`。评测新的 think checkpoint 时，用下文 `--model`、`--model-name`、`--run-name` 指定其已合并目录及独立名称。
两台机器各使用 GPU 0–7、八个单卡 worker、每卡 batch size=2；强制 spawn，不启动 Ray。
两个模型保持同一 GPU 数、batch size、视频配置、种子和依赖版本。
旧评测若还占卡，在其原终端停止后再启动 v6；脚本不会终止其他任务。

默认媒体根目录：

- HMDB：`/gemini-3/space/zwj/LMUData/HMDB51/videos`
- UCF：`/gemini-3/space/zwj/LMUData/UCF101/UCF-101`
- SSv2：`/gemini-3/space/zwj/LMUData/SSv2/20bn-something-something-v2`

保留此前建立的 UCF `HandStandPushups -> HandstandPushups` 目录别名。
启动前自动检查官方文件哈希、数据路径、模型文件和空闲显存；所有检查通过才启动 worker。
如需只检查而不推理，加 `--prepare-only`。如需先跑每个划分 8 条，加 `--limit 8 --run-name qwen25_v6_smoke`，小样本不会进入正式总表。
可加 `--train-data /gemini-3/space/zwj/AR-RLSD/data/train.rlsd.jsonl` 检查实际训练视频重叠；各类是否见过与视频是否重叠是两回事。

## 结果和恢复

```text
/gemini-3/space/zwj/VideoSTAR-Eval-results-v6/
  models.csv
  models.md
  models.json
  runs/qwen25_base_8gpu_v6/
  runs/ar_rlsd_step292_8gpu_v6/
```

总表每个模型一行，包含：
`Model | 三个数据集各自 Base/Novel/HM | HMDB split1/2/3/mean/std | UCF split1/2/3/mean/std | Training overlap note`。
三划分 mean 对三个准确率等权平均；std 为总体标准差（ddof=0），显示单位为百分点。不会把三个 split 合并去重后计分。
`summary.json` 保留未舍入统计量与训练重叠审计，`models.csv` / `models.md` 显示两位小数。
每个划分保留 `run.json / results.json / metrics.json`，worker 日志在任务的 `logs/shard_00.log` 等文件。
日志显示“单轮推理”，不再有第一轮 / 第二轮。prompt 不再要求 Stage 1 / Stage 2，只要求简短视觉证据摘要和最终类别。

```bash
bash run_v6.sh base --resume
# 训练模型所在节点：
bash run_v6.sh trained --resume
# 只重新汇总已完成结果：
bash run_v6.sh base --merge-only
```

只能恢复同一 v6 任务，不能接用 v1/v2/v3/v4/v5 缓存。运行异常会退出并阻止写入正式表；`--resume` 从各 worker 首个失败位置重试。
普通模型答错或无法解析仍计入准确率分母。完整成功的模型会追加一行，相同模型名称与相同权重重汇总只更新该行。

```bash
cat /gemini-3/space/zwj/VideoSTAR-Eval-results-v6/models.md
tail -f /gemini-3/space/zwj/VideoSTAR-Eval-results-v6/runs/qwen25_base_8gpu_v6/logs/shard_00.log
```

评测后续模型：

```bash
bash run_v6.sh trained --model /实际新模型目录 \
  --model-name NewModel-stepXXX --run-name newmodel_stepXXX_8gpu_v6 \
  --model-role unknown
```

`run_v5.sh` 兼容入口也转到 v6，不会恢复旧 prompt。可用 `EVAL_V6_RESULTS_ROOT` 改变统一输出根目录。底层 `run_8gpu.sh` 接受同样的参数，默认评测十二个划分；直接使用该入口时需加 `--allow-train-overlap`。
`run_v6.sh` 已按本次全部三个 split 的要求设置该参数，并分别标记原版 / AR 训练后模型身份。
本地参考 AR 训练集与 HMDB split2 / split3 分别重叠 1,340 / 1,311 条；仍评测整个官方 split，同时在总表中注明。
若提供 `--train-data`，还会记录该文件的哈希及逐划分的实际路径重叠；未提供时采用参考重叠说明，不表示已核对服务器训练文件。

## 只运行指定设置（单卡入口）

```bash
bash run_eval.sh check --benchmarks both --allow-train-overlap --report data_check_v6.json
bash run_eval.sh list
```

`--benchmarks base-novel` 对应六个 Base/Novel；`--benchmarks splits` 对应 HMDB/UCF 全部六个 split；`both` 共十二个划分。
`hmdb51` / `ucf101` 分别展开为各自三个 split；`split1` 仍可显式选择两个数据集的 split1。
单卡 `run_eval.sh run` 也支持这些分组，八卡总表入口固定两套都评测。

## 核验和限制

```bash
python audit_protocol.py --output ALIGNMENT.json
```

使用 Python 标准库即可核对所有官方文件、标注和候选映射。CPU 测试和 Qwen 聊天模板渲染记录见 `VALIDATION.json` / `PROMPT_AUDIT.json`。
本地没有执行真实 GPU 推理，也没有逐字节核对服务器原始视频。依赖版本差异、batch size 和推理协议变化可能改变分数。
HMDB 的参考 AR 训练集已覆盖 51 类，且与 split2/3 有视频重叠；该模型的 Novel 和三划分均值不代表纯粹的微调未见类/未见视频泛化。
本包使用三划分等权均值；标准差明确采用 ddof=0，论文未明确的标准差定义不作一致性承诺。
