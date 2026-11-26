> 全量 v3 部署和对齐说明以 [发布根目录 README](../README.md) 为准。下面保留原 AR 扩展文档。

# AR-RLSD：动作树监督与独立标注参考分支

**本发布包是全量四节点 MeSD 机制对齐 v3，保留 E1 格式奖励。** 当前部署和完整运行指令以
[包根目录 README](../README.md) 为准：93,583 条数据，4 × 8 卡，参考系数 0.1，
`run_merged_32gpu.sh` 作为正式入口。下文保留早期项目说明，其中两节点配置、
4,676 条数据和旧部署路径均为历史背景。

当前 v4 输出协议为 `<think>层级动作分析</think><answer>类别</answer>`。保留层级正文和双分支目标，只统一回答标签。新版数据为 `data/train.rlsd.think.jsonl` / `data/train.sft.think.jsonl`；旧版两份 JSONL 原样保留。上传与启动见 [UPLOAD_README.md](UPLOAD_README.md)。

独立工程位于 `/Users/zwj/Desktop/CVPR/AR/AR-RLSD`。算法主干为官方 RLSD，固定提交 `aca313b978de4fb3a509ddf1831bf61fe597a2eb`。本次网络克隆失败后，从工作区干净且 origin 指向官方仓库的本地 checkout 建立独立 Git clone；没有改动原 RLSD 或 direction-control 工程。原官方说明保留在 `README.md`、`README_zh.md`。

## Qwen2.5-VL 服务器更新（2026-10-03）

两节点 A800 的执行顺序见 [UPLOAD_README.md](UPLOAD_README.md)。默认使用原版
`/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct`，复用旧环境；从基础模型直接做 RLSD，SFT 为可选步骤。
视频解码、教师输入、actor 与 vLLM 共用视频处理，按 Processor 选择 patch size，传递实际采样 FPS。
Qwen2.5-VL 的时间位置读取模型配置并保留小数时间间隔；SFT 使用相同位置计算。
默认配置为两节点各 8 卡，16 个视频/rollout batch、每个视频 4 条生成轨迹、actor batch 为 16 个视频。
更新后的 33 项 CPU 测试通过。用户服务器日志显示 v2 加日志修复后已完成真实视频/7B 前反向及两节点 16 卡两步训练。
v3 参考 MeSD 的通用参数：16 帧、100352 pixels、768 生成 token、余弦学习率、关闭 CPU offload、1 epoch。
保留原版模型、RLSD/参考损失和每 10 步保存/保留 5 份设置；新参数的 GPU 验证步骤见 UPLOAD_README.md。

## 已实现的训练路线

可选少量动作树 SFT → RLSD 动作分类 → 加入树状 privileged context 与独立 annotation reference loss。

学生输入只有视频、问题和通用树格式要求。教师额外读取 `teacher_context`；参考轨迹为 `reference_response`，在**学生原始上下文**下 teacher-forcing，不接收教师特权信息。

```text
video + question ── student sampling ── G responses ── class reward ── group advantage
        │                                   │
        │                         frozen teacher + tree context
        │                                   │
        │                             RLSD token weighting
        │                                   │
        └── annotation reference ── separate loss ── combined actor update
```

两个外层标记为 `<think>` / `<answer>`；内部是英文缩进层级。教师输入 `teacher_context` 中的 `<action_tree>` 保留为证据标记，不是助手回答格式。当前发布包主奖励要求最终类别正确且 think/answer 完整合法，不按树节点数量或动作关键词加分。参考目标提供树表达监督；格式奖励并不能保证树内容正确。当前全量入口不先做 SFT。

## 数据

已在 `data/` 放入新版数据，各 4,676 条、51 类，无重划分/过滤：

- `train.sft.think.jsonl`：messages/videos/answer/metadata。
- `train.rlsd.think.jsonl`：prompt/videos/answer/teacher_context/reference_response/reference_status/metadata。
- `train.sft.jsonl` / `train.rlsd.jsonl`：旧版 `<action_tree>` 数据，原样保留，仅用于溯源和格式转换，不交给新版启动器。
- `think_format_manifest.json`：原文件和新文件 SHA256、转换器哈希与保留范围。`docs/ar/provenance.json` 中的数据哈希仍指向旧版源文件。

从旧数据重建 think 版本（输出必须是新目录）：

```bash
python -m verl.ar.convert_think --source-dir data --out data_think_rebuild
```

转换仅替换参考回答的两端标签，并同步学生层级格式指令；参考正文的所有字符、缩进、换行与答案保留，教师 context、视频路径、标签、顺序、元数据和未审阅状态保留。默认训练不自动运行独立 SFT。

原始 Video-STAR `sft.json` 提供样本与类别，配套 `sft_cot.json` 的观察段落提供动作证据；没有使用工具选择和候选分数。英文类别分支是项目设计，证据句尚未逐视频核验。训练配置显式设置 `ar_allow_unreviewed_reference: true`，表示按现有生成标注训练，不代表其已通过视觉审核；设为 false 会排除未审阅参考的独立损失，但不会自动移除教师 context。

视频根目录暂为 `/gemini-3/space/zwj/LMUData/HMDB51/videos`。若实际解压布局不同，可重建到新目录：

```bash
python -m verl.ar.prepare_data \
  --source-dir data \
  --out data_rebased \
  --video-root /gemini-3/space/zwj/LMUData/HMDB51/实际视频目录
```

校验 annotation 与服务器媒体：

```bash
python -m verl.ar.prepare_data --check data/train.rlsd.think.jsonl
python -m verl.ar.prepare_data --check data/train.rlsd.think.jsonl --check-media
```

没有用训练集充当验证集。默认关闭验证；接入独立评测数据后设置 `data.val_files`、`trainer.val_before_train=True` 和 `trainer.val_freq`。验证 JSONL 只需 prompt/videos/answer 及可选元数据，不必包含教师 context。

## 训练入口

服务器复用 `/path/to/workspace/zwj/.venvs/open-o3-visd-vllm`。启动脚本自动使用该环境并设置当前工程的
`PYTHONPATH`，无需在共享旧环境中重新执行 `pip install -e .`。本机 `.venv-test` 仅用于 CPU 测试。
依赖版本以 `requirements.txt` 为准，运行前检查两节点一致性。

可选冷启动 SFT（每 GPU 一个视频，多进程默认 FSDP full-shard，通过梯度累积控制 batch）：

```bash
torchrun --nproc_per_node=8 -m verl.ar.sft \
  --model /path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct \
  --data data/train.sft.think.jsonl \
  --output /gemini-3/space/zwj/AR-RLSD/outputs/sft_coldstart \
  --max-samples 512 --epochs 1
```

512 是可调实验预算；默认 `--max-samples 0` 使用全量。子集采用固定种子随机抽样，并保存 sample IDs，不修改原数据。SFT 根据模型配置选择 Qwen2.5-VL 或 Qwen3-VL 及 FSDP 包裹层。SFT 对 user/video/padding 位置 mask，只训练 assistant；前缀对不上或超长会报错，不静默截断。

RLSD + 树 context + 独立参考分支：

```bash
# 先按 UPLOAD_README.md 启动两节点 Ray，再在主节点执行：
bash examples/action_recognition/run_qwen25.sh smoke
```

该启动器默认使用原版 Qwen2.5-VL-7B-Instruct，不要求先做 SFT。若完成可选 SFT，可通过 MODEL_PATH 切换 checkpoint。
上面的 SFT 输出路径是示例，不是已生成模型。完成服务器 smoke 后再提高步数；启动器检查媒体路径与输出目录，不自动覆盖历史运行。

常用消融覆盖参数：

| 对照 | CLI override |
|---|---|
| 树 context 的 RLSD，无参考损失 | `opsd.ar_reference_coef=0` |
| RLSD + 按难度加权的参考 CE | `opsd.ar_reference_mode=ce` |
| 无教师调权、无参考损失的 GRPO | `opsd.rlsd_lambda=0 opsd.ar_reference_coef=0` |

最后一种保留相同训练通路，但仍执行教师前向，不适合作为 GRPO 计算成本基线。纯官方 RLSD 路线可使用上游配置；当前 AR 的组内标准差是总体标准差，应在算法对照中保持一致。

## V1 的精确定义

对每个视频只用 G 条真实学生 rollout 的二元奖励计算：

- `A_i = (r_i − mean(r)) / (population_std(r) + 1e−6)`；全错和全对组均为零。
- `delta_it = stopgrad(log p_old(y_it | x,z,y_<t) − log p_old(y_it | x,y_<t))`。
- `A'_it = A_i × [(1−lambda) + lambda × clip(exp(sign(A_i) × delta_it), 1−eps, 1+eps)]`。
- `b = stopgrad(kappa × eligible × (1−mean(r))²)`。
- 参考目标：`−b × mean_t min(rho*_t, clip(rho*_t,1−eps_ref,1+eps_ref))`，`rho* = exp(log p_current(y*|x) − log p_old(y*|x))`。
- 总损失为原 policy loss 加 `ar_reference_coef × reference_loss`。

教师在每次 actor update 开始、任何优化器更新前从 actor 刷新，在本轮固定，包括 checkpoint 恢复后的首轮。old student/reference scores 在更新前计算并 detach。优化器更新中禁用 actor dropout，保持评分条件一致。RLSD 在 log 空间先裁剪后 exp，避免溢出。

参考损失先逐序列做 token mean，再对全部组求均值，不按“有效参考数量”重归一化。V1 为便于沿用现有 batch 分发，将同一参考随 G 条 rollout 复制；在组大小一致时，全 batch 的均值等价于每组一条参考，但会增加前向开销。它们不是采样 rollout，不进入分类奖励均值或优势标准差。`ce` 对照也使用同一难度权重；不是固定权重 CE。

此实现参考 OraRL 的 annotation-as-rollout 思想，不含其完整 directional gain、pruning、recentering。参考 likelihood ratio 也不构成无偏 on-policy importance sampling；不能直接继承完整 OraRL/RLSD 的理论保证。

## 代码位置与验证

- `verl/ar/reward.py`：最终 answer 标签解析、类别准确率和完整格式成功奖励。
- `verl/ar/response_format.py`：共享 think/answer 格式检查，不判断树语义。
- `verl/ar/objectives.py`：组统计、稳定 RLSD 调权、独立参考 loss。
- `verl/ar/trainer.py`、`batch.py`：参考 token/mask、原始学生 prompt 拼接和旧策略评分。
- `verl/trainer/opsd_trainer.py`：教师 context 接线、参考准备。
- `verl/workers/actor/dp_opsd_actor.py`：联合 loss、固定教师刷新与归一化。
- `verl/ar/sft.py`：可选冷启动 SFT。
- `examples/action_recognition/rlsd.yaml`、`train.sh`：最小实验配置与启动器。

从 direction-control 复用了训练/验证 rollout 传递 `raw_prompt` 的修正；其视频 processor 和 vLLM backend 与本次官方 checkout 对应路径已有大量共同实现，未整体移植多视角/方向控制逻辑。来源与哈希见 `docs/ar/provenance.json`。

CPU 测试：

```bash
python -m pytest tests/ar -q
```

最初版本本机隔离环境通过 17 项测试；本次 Qwen2.5-VL 更新后共通过 33 项测试。原有测试覆盖：全量数据契约、reward、极值数值稳定性、stop-gradient、EOS/padding、mRoPE 拼接、真实 DataProto 参考接线、结构化配置，以及使用 CPU 模型前向/单 rank 通信替身执行生产 actor 更新方法。CPU 测试不能代替真实视频或 FSDP 多卡验证。

当前不支持动态 microbatch、非二元 reward shaping、在线组过滤或混合离线 rollout；配置校验会拒绝这些组合。已有的服务器截图验证了 v2 短训流程；v3 参数、全量视频覆盖、完整训练和模型效果尚待验证。新配置先保留正式帧数/生成长度运行 2 步，无误后再扩大训练预算。
