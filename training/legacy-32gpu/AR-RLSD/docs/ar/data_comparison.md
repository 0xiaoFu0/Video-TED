# 两池合并后的两组训练（v3）

v3 合并原有 11,964 条标注池与补造的 7,698 条增量池，共 19,662 条计划样本。省略两个池的 error 后，qualified 使用 9,014 条自动核验通过的 CoT；full 使用 19,621 条非错误生成 CoT，包含 10,607 条 needs_review。两组各占两台机器、每台 8 GPU，都从原版 Qwen2.5-VL-7B-Instruct 开始独立训练。

沿用当前 RLSD 配置：1 epoch、随机种子 1、每批 16 个视频、每视频 4 条在线 rollout、最多 16 帧、2 FPS、768 个生成 token。参考系数为 0.1，允许 unreviewed 参考。full 的未通过核验 CoT 同样进入教师上下文和参考监督；最终 `<answer>` 保持原 QA 标签，完整核验结果保留在审计中。SFT 文件用于对应的视频前反向检查及可选 SFT 实验，当前工作流直接从原版模型做 RLSD，无需先训练 SFT。

| 数据组 | 文件样本数 | 1 epoch 步数 | 该 epoch 消费的样本 | 不足完整 batch 的尾部 |
|---|---:|---:|---:|---:|
| qualified | 9,014 | 563 | 9,008 | 6 |
| full | 19,621 | 1,226 | 19,616 | 5 |

上述步数沿用 `drop_last=True`，尾部来自随机打乱后的不足 16 条，不是新的质量筛选。两个作业因数据量不同而具有不同训练步数。参考实际权重仍遵循现有在线 rollout 难度计算，组内全对时可能为零；needs_review 状态本身不关闭参考资格。

## 1. 上传并解压独立 v3 包

把 `AR-training-comparison-v3.zip` 上传至 `/path/to/workspace/zwj/`。在 `192.0.2.12` 执行：

```bash
cd /path/to/workspace/zwj
unzip AR-training-comparison-v3.zip
cd AR-training-comparison-v3
sha256sum -c SHA256SUMS --quiet
```

训练项目为 `/path/to/workspace/zwj/AR-training-comparison-v3/AR-RLSD`。四台机器共享此目录时只解压一次；非共享存储需要四台放置同一包、解压到相同绝对路径，并在导出后同步新的数据目录。模型与原视频在四台机器也必须能通过相同路径读取。

v3 使用独立代码、数据、Ray 端口及输出目录。不要用它覆盖运行中的 v2 工程，不执行 `ray stop` 或终止其他训练、评测、标注进程。

## 2. 一次性导出两池合并数据

两个标注池都应已完成全部 16 个 shard。仅在能访问原候选、记录、rollout 和视频的一个节点执行以下命令；共享目录无需重复导出。此步骤读取已有生成结果，不加载生成模型，也不重新造 CoT。

```bash
source /path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh
conda activate ar-qwen38-annotate
cd /path/to/workspace/zwj/AR-training-comparison-v3/tools/annotation

"$CONDA_PREFIX/bin/python" -E -s -m ar_annotate.merged_comparison \
  --pool old \
    /path/to/workspace/zwj/AR-Annotate-Qwen38-v4/work/candidates_all_v4/candidates.jsonl \
    /path/to/workspace/zwj/AR-Annotate-Qwen38-v4.2/work/full_all_16gpu_v42 \
  --pool incremental \
    /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2/work/candidates_overlap_incremental/candidates.jsonl \
    /path/to/workspace/zwj/AR-Annotate-Qwen38-overlap-v2/work/full_overlap_16gpu \
  --out /path/to/workspace/zwj/AR-training-comparison-v3/AR-RLSD/data/data_comparison_v3 \
  --student-model /path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --expect-planned 19662 --expect-qualified 9014 --expect-full 19621
```

输出目录必须不存在；报错时保留原目录和运行记录，查看具体的数量、哈希或缺失文件原因。旧的单池导出目录不能作为 v3 合并数据使用。

```text
data/data_comparison_v3/
  manifest.json
  SHA256SUMS
  selection.audit.jsonl
  qualified/train.rlsd.think.jsonl
  qualified/train.sft.think.jsonl
  qualified/sample_ids.json
  full/train.rlsd.think.jsonl
  full/train.sft.think.jsonl
  full/sample_ids.json
```

导出后检查数量与哈希：

```bash
cd /path/to/workspace/zwj/AR-training-comparison-v3/AR-RLSD/data/data_comparison_v3
sha256sum -c SHA256SUMS --quiet
python3 - <<'PY'
import json
from pathlib import Path
d = json.loads(Path('manifest.json').read_text())
assert d['merge_schema_version'] == 1
assert d['source_counts']['old']['planned'] == 11964
assert d['source_counts']['incremental']['planned'] == 7698
assert (d['planned'], d['qualified_rows'], d['full_rows']) == (19662, 9014, 19621)
assert d['full_reference_policy'] == 'all_nonerror_generated_trees_with_reference_loss'
for group, expected in [('qualified', 9014), ('full', 19621)]:
    for kind in ('sft', 'rlsd'):
        path = Path(group) / f'train.{kind}.think.jsonl'
        actual = sum(bool(line.strip()) for line in path.open())
        assert actual == expected, (str(path), actual, expected)
        print(path, actual)
print('共同 reference_max_length:', d['reference_max_length'])
PY
```

主 manifest 文件名为 `manifest.json`；`merge_schema_version=1` 标记两池合并格式，`source_counts` 区分 old 和 incremental。启动器还会校验两池及两组数量、实际 SFT/RLSD 行数、所有 payload 哈希、成对样本与参考的一致性、qualified 全部 accepted、full 全部非 error。它使用包含两个池所有参考及 EOS 的共同 `reference_max_length`，保留较长 needs_review，不按旧 768 token 预算删样本或截断参考；学生在线生成上限仍为 768。

## 3. 四台机器初始化

| 节点 | 角色 | 数据组 | Ray 地址 |
|---|---|---|---|
| 192.0.2.12 | A head | qualified，9,014 条 | 192.0.2.12:6383 |
| 192.0.2.13 | A worker | qualified，9,014 条 | 加入 A head |
| 192.0.2.10 | B head | full，19,621 条 | 192.0.2.10:6384 |
| 192.0.2.11 | B worker | full，19,621 条 | 加入 B head |

四个终端都执行：

```bash
cd /path/to/workspace/zwj/AR-training-comparison-v3/AR-RLSD
source examples/action_recognition/comparison_nodes.env
unset RUN_NAME DRY_RUN
hostname -I
test -f "$TRAIN_ENV_ROOT/bin/activate"
"$TRAIN_ENV_ROOT/bin/python" -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available() and torch.cuda.device_count() == 8'
nvidia-smi
```

`comparison_nodes.env` 只导出地址和 v3 路径、原版模型、现有训练环境以及本次预期数量，不启动任何作业。默认环境为 `/path/to/workspace/zwj/.venvs/open-o3-visd-vllm`；若节点已有兼容训练 venv 在别处，可在 source 后显式覆盖 `TRAIN_ENV_ROOT`。不要把造数据的 conda 环境误当训练环境。关键依赖沿用 Transformers 4.57.3、vLLM 0.11.0、qwen-vl-utils 0.0.14、torch/torchvision、Ray、decord、flash-attn 等。

每台机器需要 8 张空闲 GPU。若 `nvidia-smi` 显示训练、评测或标注仍占用这些 GPU，报告节点和进程情况，待资源可用后再继续；本流程不停止现有作业，也没有自动忙卡终止保护。已有 Ray 实例导致新 head/worker 启动失败时也保留该实例，报告错误。

可以先做只读预览；预览仍校验 v3 数据，缺文件或数量不符会报错，但不激活 GPU 环境或启动 Ray：

```bash
DRY_RUN=1 bash examples/action_recognition/run_data_comparison.sh qualified train
DRY_RUN=1 bash examples/action_recognition/run_data_comparison.sh full train
```

## 4. 启动两组 Ray

每组先启动 head，成功后再启动对应 worker。A 使用 Ray 6383、NCCL 29523；B 使用 Ray 6384、NCCL 29524，与 v2 使用不同端口。

在 `192.0.2.12`：

```bash
bash examples/action_recognition/run_data_comparison.sh qualified head
```

在 `192.0.2.13`：

```bash
bash examples/action_recognition/run_data_comparison.sh qualified worker
```

在 `192.0.2.10`：

```bash
bash examples/action_recognition/run_data_comparison.sh full head
```

在 `192.0.2.11`：

```bash
bash examples/action_recognition/run_data_comparison.sh full worker
```

已有启动器检查 head/worker 的本机 IP。两组只连接自己的 Ray 地址；worker 启动完成后等待任务分配，不执行训练 driver。

## 5. 两个 head 检查并正式训练

在 `192.0.2.12`：

```bash
bash examples/action_recognition/run_data_comparison.sh qualified check-video && \
bash examples/action_recognition/run_data_comparison.sh qualified smoke \
  data.video_max_frames=16 data.max_response_length=768 worker.rollout.enforce_eager=false && \
bash examples/action_recognition/run_data_comparison.sh qualified train
```

在 `192.0.2.10`：

```bash
bash examples/action_recognition/run_data_comparison.sh full check-video && \
bash examples/action_recognition/run_data_comparison.sh full smoke \
  data.video_max_frames=16 data.max_response_length=768 worker.rollout.enforce_eager=false && \
bash examples/action_recognition/run_data_comparison.sh full train
```

两条链可在各自 head 独立执行。`check-video` 使用对应 SFT 文件做单卡视频处理和模型前反向，不执行优化器更新。`smoke` 检查各组两节点、16 GPU、代码/数据/模型配置/依赖版本一致性以及 NCCL all_reduce，随后使用正式的 16 帧、768 生成 token 和 CUDA graphs 训练两步。前一步非零退出就不执行下一步。正式 `train` 使用新运行名，重新从原版模型开始 1 epoch。

检查范围需要准确区分：默认 check-video 只选择第一条，增大 `CHECK_SAMPLES` 只增加各类别首条的预处理，反向仍只对第一条；两步 smoke 只覆盖抽到的 batch，不能证明所有较长 needs_review 或全部真实视频都已完成 GPU 验证。参考长度的全量 CPU 统计保证预算包含这些树，实际 teacher/video prompt 与显存仍以服务器运行结果为准。

smoke/train 在前台运行并同时保存日志。可在已有 tmux 会话执行，保持训练终端运行。默认路径：

```text
/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/qualified/logs/ar_qwen25_qualified_v3_smoke_时间戳.log
/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/qualified/logs/ar_qwen25_qualified_v3_train_时间戳.log
/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/full/logs/ar_qwen25_full_v3_smoke_时间戳.log
/gemini-3/space/zwj/AR-RLSD/outputs/data_comparison_v3/full/logs/ar_qwen25_full_v3_train_时间戳.log
```

checkpoint 在同组输出根目录下的同名运行目录，例如 `qualified/ar_qwen25_qualified_v3_train_时间戳/`。时间戳为 `YYYYmmdd_HHMMSS`。smoke 结束也会保存独立 checkpoint，正式训练不会自动从它恢复。单独检查集群可执行本组 `cluster` 模式。

本说明提供启动步骤，不代表服务器作业已启动。CPU 启动器测试和 dry run 不替代实际视频、NCCL 与 GPU smoke 的结果。
