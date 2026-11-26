# 补造所有此前漏掉且能精确匹配的 SSv2/K710 CoT（v2）

本包使用独立目录 `/path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2`。节点 0 为 `192.0.2.10`，节点 1 为 `192.0.2.11`，各使用 8 张 GPU。标注模型仍为 `/path/to/workspace/models/Qwen/Qwen3.8-27B`，使用原有 `ar-qwen38-annotate` Conda 环境、`tree_v1` 提示和每视频 4 次独立生成，至少一次通过即 accepted。

包名延续 overlap，但 v2 的任务范围为 `all-missing`：从原 QA 来源重建全部 SSv2/K710 候选，保留能够严格匹配本地视频且尚未在旧全量任务中处理过的样本。它包含因旧标签规则拒绝的标点类别、此前排除的评测重叠，以及其他当前可精确匹配的漏掉 QA。准备过程按 sample ID、实际视频路径和视频内容哈希去重；旧任务的 accepted、needs_review 和 25 条 error 均不重跑。原 QA 标签、评测成员信息与来源审计保留。

HMDB51 已在旧全量任务中生成，本次补造只针对 SSv2/K710。无法匹配、匹配歧义、标签冲突或重复媒体仍进入拒绝/排重审计；K710 必须匹配完整文件名中的时间片段，不能用同一视频 ID 的其他切片替代。新增数量以服务器实际准备得到的 manifest 为准，不能把历史 19,688 的估计数当作恢复目标或新增数量。先前截图只展示部分拒绝原因，不能据此将 v1 的零新增全部归因于标签标点。

## 1. 上传独立包

将 `AR-Annotate-Qwen38-overlap-v2.zip` 上传到 `/path/to/workspace/zwj/`，在 `192.0.2.10` 解压并校验：

```bash
cd /path/to/workspace/zwj
unzip AR-Annotate-Qwen38-overlap-v2.zip
cd AR-Annotate-Qwen38-overlap-v2
sha256sum -c SHA256SUMS --quiet
```

两节点须真正共享这份代码、候选和新运行目录，支持文件锁及原子重命名；相同路径名不等于共享存储。使用旧标注任务的共享盘即可。本包只含标注工具、配置、提示、资产、测试及说明；原标注结果和视频仍从既有路径读取。与正在运行的 `AR-training-comparison-v2` 训练工程使用独立目录。

旧 overlap-v1 包、目录与结果继续保留。本次在 v2 新根目录重新准备候选，不复用或覆盖 v1 的候选与运行目录。

## 2. 只在节点 0 准备增量候选一次

在 `192.0.2.10` 执行：

```bash
cd /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2
source /path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh
conda activate ar-qwen38-annotate

"$CONDA_PREFIX/bin/python" -E -s -m ar_annotate prepare-incremental \
  --scope all-missing \
  --config config.server.json \
  --existing-candidates /path/to/workspace/zwj/AR-Annotate-Qwen38-v4/work/candidates_all_v4/candidates.jsonl \
  --existing-run /path/to/workspace/zwj/AR-Annotate-Qwen38-v4.2/work/full_all_16gpu_v42 \
  --out work/candidates_overlap_incremental
```

`--out` 必须是新目录。准备阶段读取原来源、重新进行严格视频匹配，并以旧全量任务为排重依据；这是 CPU 和文件读取工作，此时不加载生成模型。原候选、旧 records/rollouts、旧训练数据均保留在原目录。

本次明确使用 `--scope all-missing`；旧 `prepare-overlap-incremental` 命令继续保留为仅重叠范围的兼容入口。增量 manifest 的 `planned` 是最终新增数量，`accepted` 是新增候选的数据集计数，`skipped` 记录排重及所选范围的排除原因。`retained_source/` 保留此次重建来源与拒绝清单用于审计；模型只读取顶层 `candidates.jsonl` 中的增量样本。

准备完成后查看新增候选数量、数据集分布和完整 manifest：

```bash
"$CONDA_PREFIX/bin/python" -E -s - <<'PY'
import json
from collections import Counter
from pathlib import Path

root = Path('work/candidates_overlap_incremental')
counts = Counter()
with (root / 'candidates.jsonl').open(encoding='utf-8') as stream:
    for line in stream:
        if line.strip():
            counts[json.loads(line)['dataset']] += 1
print('新增视频数:', sum(counts.values()))
print('各数据集:', dict(counts))
print(json.dumps(json.loads((root / 'manifest.json').read_text()), ensure_ascii=False, indent=2))
PY
```

如果新增数量为 0，本次没有需要补造的样本。专用启动器会直接退出，不加载模型，也不会将旧全量候选作为替代输入。

## 3. 两节点分别启动新增样本的 16 卡标注

在 `192.0.2.10` 执行：

```bash
cd /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2
bash scripts/launch_overlap_node.sh 0
```

在 `192.0.2.11` 执行：

```bash
cd /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2
bash scripts/launch_overlap_node.sh 1
```

启动器自动激活原标注环境。默认只读取 `work/candidates_overlap_incremental/candidates.jsonl`，全量处理这份增量清单，使用相同的新输出目录 `work/full_overlap_16gpu`。节点 0 负责 shard 0–7，节点 1 负责 shard 8–15；每张 GPU 一个模型副本，同一视频的 4 次生成在该副本顺序执行。这里沿用独立标注进程，不需要启动 Ray。

新任务拥有自己的 `run.json`、运行哈希、records 和 rollouts。两个节点的终端都需执行相应命令并保持存活，可在已有 tmux 会话中运行。恢复中断时使用同一命令；它会跳过本增量任务已经完成的结果。修改代码、配置或候选后，应使用新运行目录。

需要先查看分片命令时，在对应节点执行 `DRY_RUN=1 bash scripts/launch_overlap_node.sh 0` 或 `1`；它读取增量清单但不加载模型、不启动 worker。

## 4. 查看所有分片日志和新任务汇总

在任一节点的另一个终端执行：

```bash
cd /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2
tail -n 5 work/full_overlap_16gpu/logs/shard-*.log
```

日志实际命名为 `shard-分片编号-YYYYmmdd_HHMMSS-进程号.log`，位于共享的新运行目录。需要持续跟踪已经创建的全部分片日志：

```bash
tail -n 2 -f work/full_overlap_16gpu/logs/shard-*.log
```

查看全局进度或两节点都退出后的最终汇总：

```bash
source /path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh
conda activate ar-qwen38-annotate
cd /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2

"$CONDA_PREFIX/bin/python" -E -s -m ar_annotate summarize \
  --candidates work/candidates_overlap_incremental/candidates.jsonl \
  --run work/full_overlap_16gpu
```

`planned` 是本次新增视频数；`missing_records=0`、`mismatched_records=0`、`missing_shards=[]` 表示两节点的计划覆盖齐全。`complete` 还要求没有样本级 error；accepted/needs_review/error 和 rollout 分布均以这份新汇总为准。新增 CoT 的原始结果和核验记录留在新目录，旧全量任务继续作为独立来源保留。

若增量候选为 0，启动器不会建立 `run.json`，无需执行 summarize；直接以准备阶段 manifest 为结果。

## 后续训练的数据接线

本包恢复带标点的原 QA 类别用于补造标注。以后将这些新增样本导入训练时，需要同步对齐标签规范化、目标校验与 `<answer>` 奖励解析；旧训练器只接受小写字母和下划线的类别规则，不能直接据此拒绝带标点的有效标签。该接线在新增数据准备完成后处理，当前正在运行的训练作业继续使用原工程与数据。
