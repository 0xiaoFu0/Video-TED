# 全量动作识别训练 v3：MeSD 训练机制对齐，保留 AR 任务和参考分支

这个独立项目基于 v2 的完整格式奖励版本，对齐本地 MeSD 的教师更新、退火和主损失计算。
使用服务器已有的 93,583 条合并数据，从原版 `Qwen2.5-VL-7B-Instruct` 开始训练；
不先做 SFT。参考学习系数固定为 `0.1`。

## 奖励和数据检查

主奖励为 `r = 分类正确 × 完整格式有效`，结果仍然是 0 或 1。
完整格式要求：恰好一个非空 `<think>…</think>`，后面恰好一个非空
`<answer>类别</answer>`；答案符合已有标签规则；不能有重复、嵌套、旧版或其他完整 XML
标签，也不能在外层夹带其他正文。允许外层空白和 think 内的普通缩进文本。
合法标签中的数字、标点、括号和否定词沿用全量数据原来的解析规则。

只输出正确的 `<answer>` 得 0；完整格式且答案正确得 1；完整格式但答案错误也得 0。
格式检查不判断动作树内容与视频是否一致，也不检查树的语义或层级质量。
没有额外评分服务或评分模型。

组内优势和参考学习的失败权重使用同一主奖励；参考权重为
`b = kappa × eligible × (1 − mean(r))²`，总损失中的参考系数为 `0.1`。
clipped reference loss、参考资格和失败权重保持 v2 的实现。主损失采用新的优势归一化和序列平均。

## 对齐范围

| 机制 | v2 | v3 |
|---|---|---|
| 教师 | 每个更新开始硬复制 actor | 整个更新结束后做 EMA：`teacher = .99 × teacher + .01 × actor` |
| 教师使用时间 | 一直使用 | 第 1–1000 步；第 1001 步起不构建教师 prompt、不做教师前向或 EMA |
| RLSD 混合 λ | 0.5 固定 | `λ(t)=.5 × max(1−(t−1)/1000,0)`，第 1 步 .5、第 1000 步 .0005 |
| 组内优势 | population std，epsilon 1e-6 | sample std（unbiased=True），epsilon 1e-4 |
| PPO | .2/.2，dual clip=3 | .2/.2，无 dual clip |
| 主损失平均 | 全局 token 平均 | 每条回答先平均 token，再全局平均回答；按全局回答数校正 FSDP/microbatch |
| 超长 prompt | 报错 | 先左裁最旧普通文字，再重新编码；保持完整视觉占位符和聊天结构 |
| teacher prompt budget | 16384 | 18432；student 保持 16384 |
| 断点状态 | actor/optimizer 等 | 增加每 rank 的 EMA teacher 分片，精确恢复 teacher |

EMA 更新率 **固定 .01**。旧教师权重经过 k 次更新的保留比例为 `.99^k`；
线性衰减的是 RLSD 混合 λ。第 1001 步后是 **GRPO + 系数 .1 的 AR 参考学习**，
参考分支不会随教师关闭。

AR 的单个动作树 context、协议奖励、标注参考和旧 student 概率对比逻辑继续保留。
没有迁入 MeSD 的三视角教师、top-k 融合或失败 RKL 路由；这不是完整 MeSD 算法替换。
学习率、AdamW、cosine、PPO clip、精度、视频帧数等已有运行参数已核对；
保留 AR 的基础模型、B32/G4、原生视频、参考长度和 save100/保留5 策略。
sampling 温度 1.0、top_p .95、top_k 50、G4、microbatch 1、grad norm 5.0 和 TP1 也与 MeSD 一致。
B32 是现有 32 ranks 在重复 rollout 前分发 prompt 的要求，不照搬 MeSD 的 B16。

安全左裁按 fast tokenizer 的字符偏移删除文字，避免裁开 Unicode 字符。
student 的 `input_ids`、mRoPE、`raw_prompt_ids` 和 vLLM 的 `raw_prompt` 都来自同一拟合 prompt；
teacher 也先重新编码再计算位置。原始动作树、参考答案和 teacher_question 不被改写。
如果连完整视觉块和聊天结构都超过 budget，会明确报错，要求调整 budget 或视频采样，
不会静默截断视频、减少帧数、丢弃样本。已拟合文字可能丢失最早的文本上下文。

`prepare` 逐条验证全部 RLSD/SFT 配对、标签、完整参考格式、视频路径、来源计数、
manifest 哈希、参考资格和参考长度。它只生成本项目的预检报告和四条来源均衡的视频探针，
不修改、过滤或重建原数据。遇到不兼容格式会给出行号和 sample ID，停止启动。
预检报告绑定数据、tokenizer、标签和格式检查器的哈希；修改它们后必须重新 `prepare`。

| 设置 | 本次值 |
|---|---|
| 数据 | 93,583：V1 10,454；LLaVA 2,905；K700 46,523；SSv2 33,701 |
| 数据根目录 | `/path/to/workspace/zwj/AR-action-training-merged-v1` |
| 初始化模型 | `/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct` |
| 训练 | MeSD 对齐的 RLSD + 独立参考分支，参考系数 0.1，kappa 1.0 |
| 主节点 | `192.0.2.12` |
| 工作节点 | `192.0.2.13`、`192.0.2.10`、`192.0.2.11` |
| 规模 | 4 × 8 卡；FSDP full shard；rollout TP=1 |
| 视频 batch / actor global batch | 32 / 32 |
| 每视频 rollout | 4；每更新 128 条轨迹，每 GPU 4 条 |
| actor 更新 microbatch | 每 GPU 1 条 |
| 学习率 / epoch / seed | 1e-6 / 1 / 1 |
| 视频 | 最多 16 帧，2 fps，每帧最多 100352 pixels |
| student / teacher prompt / response 上限 | 16384 / 18432 / 768 tokens |
| 参考长度上限 | 至少 1024，按完整数据实测最长参考（包括 EOS）提高 |
| 保存 | 每 100 步，保留最后 5 份；包含 actor、EMA teacher、优化器、scheduler、RNG 和 dataloader 状态 |
| Ray / NCCL 集群检查端口 | 6401 / 29541 |
| 输出根目录 | `/gemini-3/space/zwj/AR-action-training-32gpu-v3/outputs` |

沿用原项目的 `drop_last=True`：一个 epoch 为 **2,924 步**，访问 93,568 条，
最后不足一个 batch 的 15 条不参加这个 epoch 的更新。全部 93,583 条仍保留在数据文件中。
这是原训练入口的批次行为，没有筛掉数据。

## 部署

将 `AR-action-training-32gpu-v3.zip` 和 `.zip.sha256` 上传到 `/path/to/workspace/zwj`。
共享目录只需解压一次；不共享时，四台节点均部署同一份包，并保证数据、视频、模型和
Python 环境的绝对路径一致。包内 `AR-RLSD/data/` 是保留的旧版 4,676 条测试数据，
**不包含服务器的 93,583 条导出**。本次启动器明确使用上表的数据根目录。

```bash
cd /path/to/workspace/zwj
sha256sum -c AR-action-training-32gpu-v3.zip.sha256
unzip -n AR-action-training-32gpu-v3.zip
cd AR-action-training-32gpu-v3
sha256sum --quiet -c SHA256SUMS
```

新项目使用独立输出目录、端口和 head 临时 session 目录，并清理外部继承的 Ray 路由变量。
请等当前 v2 训练完成、四台机器可用后，再启动 v3；新端口不能释放已有 GPU 任务。
本项目不自动执行全局 `ray stop`、杀进程或删除旧权重。每台节点可以先用 `nvidia-smi` 查看实际占用。
本次交付没有连接服务器、停止当前训练或删除任何权重。

## ① 主节点：检查全量数据，启动 head

在 **192.0.2.12**：

```bash
cd /path/to/workspace/zwj/AR-action-training-32gpu-v3/AR-RLSD
source examples/action_recognition/merged_32gpu.env
bash examples/action_recognition/run_merged_32gpu.sh prepare && \
bash examples/action_recognition/run_merged_32gpu.sh head
```

## ② 三个工作节点：加入集群

在 **192.0.2.13**：

```bash
cd /path/to/workspace/zwj/AR-action-training-32gpu-v3/AR-RLSD
source examples/action_recognition/merged_32gpu.env
NODE_HOST=192.0.2.13 bash examples/action_recognition/run_merged_32gpu.sh worker
```

在 **192.0.2.10**：

```bash
cd /path/to/workspace/zwj/AR-action-training-32gpu-v3/AR-RLSD
source examples/action_recognition/merged_32gpu.env
NODE_HOST=192.0.2.10 bash examples/action_recognition/run_merged_32gpu.sh worker
```

在 **192.0.2.11**：

```bash
cd /path/to/workspace/zwj/AR-action-training-32gpu-v3/AR-RLSD
source examples/action_recognition/merged_32gpu.env
NODE_HOST=192.0.2.11 bash examples/action_recognition/run_merged_32gpu.sh worker
```

如果代码目录不共享，各工作节点应先运行一次 `prepare`，再运行上面的 `worker`。
head/worker 的 `ray start` 以后台服务形式启动，命令返回终端是正常的；只有主节点启动训练 driver。

## ③ 主节点：检查、试跑、正式训练

回到 **192.0.2.12**：

```bash
(
  set -euo pipefail
  cd /path/to/workspace/zwj/AR-action-training-32gpu-v3/AR-RLSD
  source examples/action_recognition/merged_32gpu.env
  unset RUN_NAME

  bash examples/action_recognition/run_merged_32gpu.sh cluster
  bash examples/action_recognition/run_merged_32gpu.sh check-video
  bash examples/action_recognition/run_merged_32gpu.sh smoke

  export RUN_NAME="ar_qwen25_action93583_32gpu_mesd_ema010_ref010_train_$(date +%Y%m%d_%H%M%S)_$$"
  mkdir -p "$OUTPUT_ROOT/logs"

  nohup bash examples/action_recognition/run_merged_32gpu.sh train \
    > "$OUTPUT_ROOT/logs/$RUN_NAME.launcher.log" 2>&1 < /dev/null &

  printf '训练 PID：%s\n运行名称：%s\n日志：%s\n' \
    "$!" "$RUN_NAME" "$OUTPUT_ROOT/logs/$RUN_NAME.launcher.log"
)
```

集群检查要求四个指定 IP、每台 8 个可分配的 Ray GPU 资源，以及代码、数据、模型配置、
依赖一致；全部 32 ranks 的 NCCL all-reduce 应输出 `sum=528.0`。
实际显存占用还应以 `nvidia-smi` 为准。`check-video` 检查四个来源的探针并执行前反向；
`smoke` 按正式 16 帧 / 768 token 参数执行两步，不周期保存；结束时仍保存独立 smoke checkpoint。
正式训练使用新的运行目录，从基础模型重新初始化，不继承 smoke 权重或 E1/E2 旧权重。

启动日志打印实际完整配置，并在用户参数后固定以下对齐字段：

```text
opsd.ar_alignment_mode=mesd
opsd.ar_teacher_update_mode=ema
opsd.ar_teacher_update_rate=0.01
opsd.ar_teacher_update_interval=1
opsd.ar_teacher_disable_after_step=1000
opsd.ar_teacher_max_prompt_length=18432
opsd.rlsd_lambda=0.5
opsd.rlsd_lambda_warmup_steps=0
opsd.rlsd_lambda_decay_steps=1000
data.truncation=left
worker.actor.loss_avg_mode=seq
opsd.ar_reward_mode=protocol
worker.reward.reward_function_kwargs={"reward_mode":"protocol"}
opsd.ar_reference_coef=0.1
```

日志分别记录 `reward/accuracy`、`reward/answer_valid`、`reward/protocol_valid`、
`reward/overall` 和 `ar/group_joint_success_mean`，以区分类别准确率与完整格式成功率。
RLSD 消费 `train.rlsd.think.jsonl`；配对 SFT 文件用于验证和探针，没有额外 SFT 训练阶段。
入口在构造 dataloader 前验证配置；`filter_overlong_prompts=True` 会被拒绝，超长 prompt 按文字裁剪处理。

查看最新正式训练日志：

```bash
source /path/to/workspace/zwj/AR-action-training-32gpu-v3/AR-RLSD/examples/action_recognition/merged_32gpu.env
tail -n 100 -f "$(ls -t "$OUTPUT_ROOT"/logs/*mesd_ema010_ref010_train*.launcher.log | head -n 1)"
```

完整 epoch 完成后，权重应在 `$OUTPUT_ROOT/<RUN_NAME>/global_step_2924/actor`。

## 中断后的恢复

重建健康的四节点集群后，选择**这次新项目**已保存完整的 `global_step_*` 目录
（不是其中的 `actor`）。v3 的 actor 子目录还应有 32 个
`teacher_world_size_32_rank_<0..31>.pt`。缺失、为空或损坏的 teacher 分片会拒绝恢复；
**不能把 v2 checkpoint 当作 v3 的精确续训 checkpoint**。每份 checkpoint 新增 EMA 教师权重，会增加磁盘容量和保存 I/O。保持 32 ranks、数据、batch 和设置一致，
使用新的 RUN_NAME：

```bash
cd /path/to/workspace/zwj/AR-action-training-32gpu-v3/AR-RLSD
source examples/action_recognition/merged_32gpu.env
export RUN_NAME="ar_qwen25_action93583_32gpu_mesd_ema010_ref010_resume_$(date +%Y%m%d_%H%M%S)_$$"
: "${AR_RESUME_CKPT:?请先设置为本次已保存完整的 global_step_* 目录}"
mkdir -p "$OUTPUT_ROOT/logs"
nohup bash examples/action_recognition/run_merged_32gpu.sh train \
  "trainer.load_checkpoint_path=$AR_RESUME_CKPT" \
  trainer.find_last_checkpoint=false \
  > "$OUTPUT_ROOT/logs/$RUN_NAME.launcher.log" 2>&1 < /dev/null &
```

本地验证结果见 [VALIDATION.md](VALIDATION.md)，改动来源和保留文件哈希见
[PROVENANCE.json](PROVENANCE.json)。本地验证不代替服务器 93,583 条数据检查、真实解码、
NCCL 和完整训练，按上面的 `prepare/cluster/check-video/smoke` 顺序运行。

`tools/annotation` 原样保留旧项目的真实 merger 集成测试依赖；上述训练命令不会调用它生成新标注。

## TensorBoard

默认日志在本项目的 `AR-RLSD/tensorboard_log`，按 project/run 分目录。
在主节点执行（使用额外终端）：

```bash
cd /path/to/workspace/zwj/AR-action-training-32gpu-v3/AR-RLSD
/path/to/workspace/zwj/.venvs/open-o3-visd-vllm/bin/tensorboard \
  --logdir "$PWD/tensorboard_log" --host 127.0.0.1 --port 6007
```

在本地终端执行，再打开 `http://127.0.0.1:6007`：

```bash
ssh -N -L 6007:127.0.0.1:6007 root@192.0.2.12
```

除格式/正确率、长度和 reference_loss 外，重点看：
`ar/teacher_active`、`ar/teacher_ema_applied`、`grpo_opsd/stgca_effective_lambda`、
`ar/advantage_sample_std`、`ar/advantage_epsilon`、`ar/prompt_trimmed_fraction`、
`ar/prompt_removed_text_tokens_mean`、`ar/teacher_prompt_tokens_max`。
前 1000 步 active/EMA 为 1；第 1001 步起为 0，λ 为 0，`ar/reference_coef` 继续为 .1。
参考损失是否非零还取决于组内失败和资格，不能仅凭系数判断每步都有参考梯度。
