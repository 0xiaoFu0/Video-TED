# Qwen3.8 动作证据标注工具

独立项目：为现有 AR-RLSD-think 统一标注 HMDB51 / SSv2 / K710 数据，使用本地
`/path/to/workspace/models/Qwen/Qwen3.8-27B`。不更改训练工程、不下载视频、不改写原标注。

v4 统一标注池生成成对的 `pool.rlsd.think.jsonl` 和 `pool.sft.think.jsonl`；
旧扩数据合并模式仍使用 `train.*.think.jsonl`。
学生 prompt 直接沿用现有工程；保留原分类答案，补齐教师 context 和层级动作证据。
**`accepted` 只是模型一致性与格式检查通过，不代表人工审核；导出仍是 `unreviewed`。**

## 当前部署版本 v4.2：四次生成中有一次通过就保留

**本节优先于下面的旧版说明。** 上传 `AR-Annotate-Qwen38-v4.2.zip`，解压到独立目录，
两节点均使用 v4.2。复用现有 Conda 环境，不重装包，不重新准备三数据集候选。

- 默认每视频 4 次，每次都生成并核验；至少 1 次通过即可接收。
- 其余 rollout 即使是技术错误，也不否决已经独立通过核验的 CoT。错误计数、原始输出仍保留。
- 所有计划的 rollout 仍须有结果文件；中断导致的缺失不会被当作完整任务。
- 多条通过时导出编号最小的一条，并保留全部候选；不是模型评分选出的“最佳”一条。
- 没有任何通过结果时：全为语义分歧则 needs_review；含技术错误则 error，按原规则排查/重试。
  默认 export 仍要求所有样本有结果且无未解决的样本级 error；显式 `--allow-partial` 才可导出未完成批次的合格子集。
- 样本已 accepted 时恢复不会重试其余错误 rollout；已满足保留目标。错误仍在 rollout_summary 与逐次文件中。
- 包含 v4.1 格式反馈重试与三次格式尝试修复，保留树状语义核验。
- `complete=true` 表示样本级无未解决错误和无缺失；逐 rollout 的 error 计数可能非零，这是保留的诊断信息。

在共享目录解压一次：

```bash
cd /path/to/workspace/zwj
unzip -n AR-Annotate-Qwen38-v4.2.zip
```

在 `192.0.2.10`：

```bash
cd /path/to/workspace/zwj/AR-Annotate-Qwen38-v4.2
ROLLOUTS=4 MIN_ACCEPTED_ROLLOUTS=1 bash scripts/launch_16gpu_node.sh 0
```

在 `192.0.2.11`：

```bash
cd /path/to/workspace/zwj/AR-Annotate-Qwen38-v4.2
ROLLOUTS=4 MIN_ACCEPTED_ROLLOUTS=1 bash scripts/launch_16gpu_node.sh 1
```

默认仍是每数据集 10 条、最多 30 视频。默认候选路径固定为
`/path/to/workspace/zwj/AR-Annotate-Qwen38-v4/work/candidates_all_v4/candidates.jsonl`，
新版任务目录是 `/path/to/workspace/zwj/AR-Annotate-Qwen38-v4.2/work/pilot_all_16gpu_v42`。
代码、阈值与格式重试策略变化，因此不要覆盖旧 run；旧结果保留，不把旧记录改写成新版结果。

汇总命令（激活原标注环境后执行）：

```bash
python -m ar_annotate summarize \
  --candidates /path/to/workspace/zwj/AR-Annotate-Qwen38-v4/work/candidates_all_v4/candidates.jsonl \
  --run work/pilot_all_16gpu_v42
```

仍无统一动态进度条；当前终端显示节点分配，日志显示每视频和每次 rollout 进度。
可在另一个终端查看 `tail -n 5 work/pilot_all_16gpu_v42/logs/*.log`，无需再次启动标注。

小批确认后运行全量，两节点分别在新版目录执行（最后的 rank 分别为 0、1）：

```bash
# 节点 0
LIMIT_PER_DATASET=0 ROLLOUTS=4 MIN_ACCEPTED_ROLLOUTS=1 \
RUN_DIR="$PWD/work/full_all_16gpu_v42" bash scripts/launch_16gpu_node.sh 0
# 节点 1，在另一个节点单独执行
LIMIT_PER_DATASET=0 ROLLOUTS=4 MIN_ACCEPTED_ROLLOUTS=1 \
RUN_DIR="$PWD/work/full_all_16gpu_v42" bash scripts/launch_16gpu_node.sh 1
```

结束后仅在一节点导出：

```bash
python -m ar_annotate export --pool \
  --candidates /path/to/workspace/zwj/AR-Annotate-Qwen38-v4/work/candidates_all_v4/candidates.jsonl \
  --run work/full_all_16gpu_v42 --out work/export_pool_all_v42
```

## v4.1：针对格式失败的修复

服务器 v4 pilot 中一个 K710 rollout 的两次格式尝试分别缺少 `sufficient_evidence`、
超过全树 8 条 observation。失败发生在证据校验阶段，未进入分类核验。
v4.1 在下一次结构重试中反馈具体校验错误，保留字段和总条数约束，不自动补 true 或截断动作树。
默认格式尝试上限从 2 增至 3。核验 JSON 格式出错时复用已合法的证据，只重新生成核验 JSON。
有效的语义分歧仍不重试。重试 prompt 及错误阶段写入 attempts 供审计。

新 ZIP 根目录为 `AR-Annotate-Qwen38-v4.1/`，保留旧代码与数据。因为代码/配置指纹发生变化，
不得用新版覆盖旧 run 后直接 `--retry-errors`，不能把新结果悄悄填入旧 pilot。
准备好的候选可原样复用；新增 `--sample-id` 用来只复测指定视频，不修改候选清单和 manifest。

```bash
cd /path/to/workspace/zwj
unzip -n AR-Annotate-Qwen38-v4.1.zip
cd AR-Annotate-Qwen38-v4.1
source /path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh
conda activate ar-qwen38-annotate
CUDA_VISIBLE_DEVICES=0 "$CONDA_PREFIX/bin/python" -E -s -u -m ar_annotate annotate \
  --candidates /path/to/workspace/zwj/AR-Annotate-Qwen38-v4/work/candidates_all_v4/candidates.jsonl \
  --sample-id k710_b15ca58fc8f777a6da18 \
  --rollouts 4 --min-accepted-rollouts 3 \
  --out work/format_smoke
```

在有空闲 GPU 的任一节点执行一次即可；该命令只处理一个视频的四次 rollout，不是重跑 16 卡。
验证标准是无技术/格式 error，结果仍可能是 needs_review，不能为了让结果 accepted 而放宽事实核验。
后续若用两个节点执行新版完整 pilot，仍要使用共享新输出目录并显式设置上述旧候选的绝对路径。
下面 v4 的节点分工和准备/导出规则不变，但代码目录应替换为 v4.1，旧候选无需重复准备。

## v4：三数据集统一标注池，两个节点共 16 卡

v4 是本次部署入口；后面的 v3/v2 章节保留旧流程背景。ZIP 根目录为 `AR-Annotate-Qwen38-v4/`。
默认配置采用双方讨论的 `tree_v1`：**不给原标签，也不给旧 CoT，看视频构建树，再核验**。
根节点概括整体活动，分支组织子动作/交互/协调/时序阶段，叶节点必须具体支持父节点。
不强制所有视频都有“开始—中间—结束”。新增 `hierarchy_supported` 核验字段，与视觉支持、
证据充分性、类别匹配一起作为接收条件。使用相同模型核验仍不能保证正确性。
默认每视频 4 次 rollout、至少 3 次通过，每视频最多导出一条代表性 CoT。

三数据集放在同一候选清单、同一运行目录，按 `metadata.dataset` 保留来源：

- HMDB51：读取现有 `AR-RLSD-think/data/train.rlsd.think.jsonl` 的视频与类别，重新生成 CoT；
  本地源清单为 4,676 条 / 51 类。实际服务器可用数以准备报告为准，缺失媒体明确写入拒绝清单。
  不追加验证/测试视频，不把旧 `reference_response`、`teacher_context` 输入模型。
  核验选项为该训练源的完整 51 类词表，选项顺序打乱；不指定正确选项。
- SSv2 / K710：复用已准备的 `work/candidates_v1/candidates.jsonl`，校验 manifest、逐行哈希、
  视频大小/mtime 和评测排除资产；不重复扫描大型媒体目录。旧 rollout 结果不迁移到新 prompt 任务。
- 完成后导出 `pool.rlsd.think.jsonl` / `pool.sft.think.jsonl`，只含本次通过结果，不再追加旧 HMDB51。
  原标签、源样本 ID、源清单哈希、dataset、匹配方式、rollout 记录和评测集合成员信息均保留。
  这是待后续拆分的标注池，不自动随机划分训练/验证/测试。未通过的原始标注仍保存在运行目录。

本地核对现有 HMDB51 4,676 条与当前 VideoSTAR-Eval-v5：base / novel / split1 重叠为 0，
split2 重叠 1,340、split3 重叠 1,311、full 重叠 3,570；所有评测集合并集重叠 3,896。
因此不能把这批数据直接用于所有 split 的训练再评测。`assets/hmdb51_eval_membership.json`
保存视频级集合成员关系及来源哈希；导出 metadata 的 `evaluation_overlaps_json` 保留这些关系。
后续按选定评测协议过滤，不能只随机切分。SSv2/K710 继续沿用已有视频排除策略。

### 部署和准备（只在一个节点执行一次）

节点分工：`192.0.2.10` 为 rank 0，`192.0.2.11` 为 rank 1，每节点按 8 卡配置。
本地尝试 SSH 时两者都关闭了连接，**未远程部署、未确认 GPU/共享盘、未启动任务**。
下面由用户在可登录的服务器终端执行，沿用已经成功的标注环境，不要重建或重装。

上传 ZIP 到 `/path/to/workspace/zwj/` 后：

```bash
cd /path/to/workspace/zwj
unzip -n AR-Annotate-Qwen38-v4.zip
cd AR-Annotate-Qwen38-v4
source /path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh
conda activate ar-qwen38-annotate
"$CONDA_PREFIX/bin/python" -E -s -m ar_annotate prepare-combined \
  --existing-candidates /path/to/workspace/zwj/AR-Annotate-Qwen38/work/candidates_v1/candidates.jsonl \
  --out work/candidates_all_v4
```

确认报告包含三个数据集的实际数量。准备目录存在时不会覆盖；中断造成的不完整目录应换新名字。
如果旧候选清单实际在 v3 目录，修改 `--existing-candidates` 即可。
config 中 HMDB51 视频根目录为 `/gemini-3/space/zwj/LMUData/HMDB51/videos`，
只按照原 `video_relpath` 拼接，不在其它目录模糊匹配替代文件。

**两节点必须真正共享同一代码、模型、候选与输出目录，支持 POSIX 文件锁和原子重命名。**
相同路径名不等于共享文件。可在节点 0 创建 `work/shared_probe_v4.txt` 写入本次随机字符串，
节点 1 读取核对，再从节点 1 追加另一个字符串，回节点 0 检查，两边均应可见。
每节点运行 `nvidia-smi -L` 和 `python -m ar_annotate doctor` 确认同一环境与 GPU；
doctor 不加载权重，最终仍须真实小批。若目录不共享，应先配置共享存储，不能复制两个输出目录冒充共享盘。

### 双节点小批（每数据集 10 条，总计最多 30 视频 / 120 次 rollout）

两节点均进入 `/path/to/workspace/zwj/AR-Annotate-Qwen38-v4`，然后分别执行：

```bash
# root@192.0.2.10
bash scripts/launch_16gpu_node.sh 0
```

```bash
# root@192.0.2.11
bash scripts/launch_16gpu_node.sh 1
```

默认使用同一绝对路径 `work/pilot_all_16gpu_v4`。rank 0 负责全局分片 0..7，rank 1 负责 8..15。
不要用各节点独立的时间戳生成 RUN_DIR。脚本不会自动 SSH 到另一个节点，两个命令都要执行。
一张 GPU 一个模型副本；同一视频的 4 次在该副本上顺序生成，共享其采样帧。
`DRY_RUN=1 bash scripts/launch_16gpu_node.sh 0` 可以先检查分配，无 GPU 推理。
运行日志在共享 `RUN_DIR/logs/`；保持终端会话存活，或在已有 tmux 会话中执行。

两节点都结束后，在任一节点汇总：

```bash
python -m ar_annotate summarize \
  --candidates work/candidates_all_v4/candidates.jsonl \
  --run work/pilot_all_16gpu_v4
```

应有 `complete: true`、缺失/不匹配记录为 0、`missing_shards: []`；查看三个数据集分别的
accepted / needs_review / error，以及逐 rollout 计数。通过率不是准确率，还应抽查视频和树。

### 全量、恢复和统一池导出

新小批确认后，两节点分别执行（共享目录名完全相同）：

```bash
# root@192.0.2.10
LIMIT_PER_DATASET=0 RUN_DIR="$PWD/work/full_all_16gpu_v4" \
  bash scripts/launch_16gpu_node.sh 0
```

```bash
# root@192.0.2.11
LIMIT_PER_DATASET=0 RUN_DIR="$PWD/work/full_all_16gpu_v4" \
  bash scripts/launch_16gpu_node.sh 1
```

恢复用原命令；只重试技术错误再加 `RETRY_ERRORS=1`。代码、环境、prompt、帧参数、种子、
rollout 数、筛选阈值或总分片数变化时必须换运行目录。原 v2/v3 目录和结果保留。

```bash
python -m ar_annotate export --pool \
  --candidates work/candidates_all_v4/candidates.jsonl \
  --run work/full_all_16gpu_v4 \
  --out work/export_pool_all_v4
```

产物是两份 `pool.*.think.jsonl`、逐样本转换审计、review.csv 和 manifest；
`base_rows` 必须为 0。export 会检查全部 16 个分片的汇总、全部选中记录、各次 rollout 与媒体哈希。
缺节点/缺记录/有技术错误时默认拒绝完整导出。之后再根据 dataset 和评测成员信息拆分。
筛选与 token 上限可能减少最终数量，manifest 会列明，不能把计划视频数当作合格 CoT 数。

## v3 历史说明：多次 rollout 与可选 prompt 草案

每个视频现在支持固定 N 次独立随机生成；每次包含“证据生成 + 视频核验”两次调用。
`attempts=2` 仍只表示一次 rollout 内结构错误的重试，不等于 rollout 数量。
v3 默认是 1 次；v4 配置已设为 4 次。`--rollouts` / `ROLLOUTS` 可覆盖，但阈值也要与次数一致。
不因提前通过或语义不一致提前结束。
每次使用独立的可复现种子，不读取其他 rollout 的回答；同一视频共享解码后的帧和打乱的选项顺序。
独立随机采样不等于独立标注者；同一模型、同一组帧的系统性误判仍可能反复出现。

- 未指定阈值时，要求严格多数通过：4 次至少 3 次。可用 `--min-accepted-rollouts` / `MIN_ACCEPTED_ROLLOUTS` 调整。
  这不是准确率保证，也不承诺提升通过率；更严格的阈值可能减少产量。
- 原标签始终保留。通过条件仍是类别匹配、视觉证据充分、核验支持；语义不一致计入分母。
- 有任一次技术错误，样本汇总为 `error`，不能把剩余几次当作完整 N 次。
  `--retry-errors` / `RETRY_ERRORS=1` 只重试技术错误，归档旧错误，不重跑有效的分歧结果。
- 每次独立保存到 `rollouts/<sample_id>/0000.json` 等文件；中断后只补缺失次数。
  汇总仍放在 `records/<sample_id>.json`，增加 `rollout_summary` 和每次结果的哈希。
- 导出重新校验所有 rollout 与汇总。每视频仅导出一条：采用编号最小的通过结果，
  这是确定性代表，不是经过质量排序的“最佳 CoT”。所有其余结果保留，可供后续人工筛选。
  汇总未通过的样本，即使某一次通过，也不自动加入训练。
- 默认温度 0.7 / top_p 0.8 / top_k 20，可在 annotation 配置中调整；参数、种子、prompt 均进入任务指纹。
- 8/16 卡仍按视频分片，每卡顺序处理该视频的 N 次，不同时放入 N 个视频批次。
  推理调用次数约为单次的 N 倍；节省重复解码不等于无成本扩增。

先保持原 prompt 做 rollout 对照：

```bash
# v3 ZIP 解压后进入新目录；复用已经跑通的 Conda 环境，不重装。
cd /path/to/workspace/zwj/AR-Annotate-Qwen38-v3
source /path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh
conda activate ar-qwen38-annotate

CANDIDATES=/path/to/workspace/zwj/AR-Annotate-Qwen38/work/candidates_v1/candidates.jsonl \
ROLLOUTS=4 MIN_ACCEPTED_ROLLOUTS=3 LIMIT_PER_DATASET=100 \
PROMPT_PROFILE=legacy RUN_DIR="$PWD/work/pilot_rollout4_legacy" \
  bash scripts/launch_annotation.sh
```

换 prompt 的对照使用相同样本、种子、4 次 rollout 与阈值，只改两个参数：

```bash
CANDIDATES=/path/to/workspace/zwj/AR-Annotate-Qwen38/work/candidates_v1/candidates.jsonl \
ROLLOUTS=4 MIN_ACCEPTED_ROLLOUTS=3 LIMIT_PER_DATASET=100 \
PROMPT_PROFILE=temporal_v1 RUN_DIR="$PWD/work/pilot_rollout4_temporal" \
  bash scripts/launch_annotation.sh
```

不要同时启动这两条占满相同 GPU 的命令。新任务的目录名固定下来后，恢复时使用完全相同的命令。
可先把 `LIMIT_PER_DATASET=1`，输出到另一个 smoke 目录确认兼容性，再启动 100 条对照。
同一目录不允许修改 N、阈值、prompt、模型、依赖或分片数。v2 任务不混入 v3，旧目录与结果保留。
新 ZIP 根目录是 `AR-Annotate-Qwen38-v3/`，不会覆盖现有 `AR-Annotate-Qwen38/`。

### Prompt 讨论稿（尚未经真实视频对照验证）

`legacy` 完整保留原 evidence / verify prompt，学生训练 prompt 始终保持原版。
`temporal_v1` 只更换造数据与核验 prompt，仍然不告知原标签，仍输出原有 JSON schema：

1. 首轮围绕“初始状态 → 可见交互/变化 → 最后可见状态”组织证据；只写可见阶段，不强制补齐。
2. SSv2 强调接触、位移、方向、假动作、相机转动与平移、碰撞前后各物体运动。
   K710 强调人体动作、人与物体交互和协调，不能仅凭道具或场景判断动作。
3. 动作关键条件不清楚时必须降低充分性判断；不把品牌、颜色、背景细节写成主要推理。
4. 核验逐项检查选项限定条件；即使选中了正确类别，草稿缺少决定性时序证据也不能自动通过。
   核验理由不能临时添加事件来修补草稿。

待讨论：是否改为给原标签后写 CoT。该模式可能提高标签对应率，也可能诱导补写视频里不存在的动作。
本次未擅自切换为给答案生成。若以后增加该模式，需要让核验不接收正确答案，并单独记录生成模式。
多次 rollout 也无法补回所有抽样都漏掉的关键帧。本次保持 2 fps / 最多 16 帧，方便区分随机采样与 prompt 的影响。

具体草案文件：`prompts/evidence_temporal_v1.txt`、`evidence_ssv2_v1.txt`、
`evidence_k710_v1.txt`、`verify_temporal_v1.txt`。

## 本版实现范围

- `prepare`：读取 Flash 原 JSONL，只选 SSv2 和 K710；严格匹配唯一、非空的完整文件名。
  K710 仅扫描已找到的 Kinetics-700 **train** 目录，另核对类别目录。
  不用“同 ID、不同时段”替代原片段，不使用 `/gemini-6`，不纳入 MVBench。
- 自带从本地 VideoSTAR-Eval-v5 提取的评测视频排除清单：SSv2 24,777 个视频 ID，
  K400/K600 合并 29,575 个视频 ID。Kinetics 排除整个原视频 ID，而非只排除一个时间段。
  原标注与排除清单的来源、哈希都保留。**它只覆盖列出的本地评测，不能保证所有外部 benchmark 都无重叠。**
- 首轮只看视频，不给类别、选项、真实文件名，生成简洁英文动作证据 JSON。
- 次轮重新使用同一组帧，给打乱后的选项与首轮证据，不告知正确选项，检查分类和证据支持度。
  不一致进入 `needs_review`。语义不一致不重试，不把模型预测写回原标签。
- 原始生成文本、采样帧索引/时间/哈希、视频 SHA256、源记录位置、模型与依赖指纹都保存。
- 每视频独立原子落盘，支持断点恢复、分片、多 GPU 进程；解码在可超时终止的子进程运行。
  结构不合法最多尝试两次；连续五个错误停止当前分片，避免整批空跑。
- 导出时按**学生模型 tokenizer**限制参考长度，检查媒体哈希、ID 重复、SFT/RLSD 对应关系。
  不截掉答案来满足长度；超长样本留在审计中。默认不允许把未完成任务当作完整导出。

在准备时就会排除评测重叠、重复、标签冲突和不兼容标签，所以新增数不承诺是 15,012。
SSv2 的 train 来源依据 Flash 路径，不等于已独立核验官方 split 清单。
加入 SSv2/Kinetics 后，已有 base/novel **未见类别**评测的含义会改变：视频去重不等于类别未见。

## 环境：专用 Conda 环境与电信 pip 镜像（v2）

建议把整个文件夹上传到：

```text
/path/to/workspace/zwj/AR-Annotate-Qwen38
```

开发机完成 CPU 数据流测试。用户提供的 v2 服务器截图已显示 A800 上两条真实视频 smoke 与
200 条八卡 pilot 完成；v3 多次 rollout 尚未在服务器执行。截图不等于本地复现或质量验证。
v3 将 torch 2.6.0 / torchvision 0.21.0 / Transformers 5.8.0 固定为该成功环境的版本，
避免安装脚本再次自动升级到此前与驱动不兼容的 CUDA 13 版本；其余依赖仍非完整锁定。
已跑通的 `ar-qwen38-annotate` 使用 Python 3.10，直接复用即可。下面 setup 是首次部署说明，
不需要为升级代码重新创建环境，也不要重新执行它来恢复旧任务。

建议 Linux x86_64 / Python 3.11，使用原版 BF16 权重。27B 级权重约占 54 GB（仅参数量估算），
还需视觉编码器、激活和缓存空间。为了吞吐，先测试 **1 张空闲 80 GB GPU**；
能放下就每卡一副本，不能放下则切换两卡一副本。没有承诺单卡一定装得下。
主模型使用 Transformers 的 Qwen3_5ForConditionalGeneration：官方 Qwen3.8 使用该架构。
采用 SDPA，不要求先编译 flash-attn / flash-linear-attention；未装优化内核时可能较慢。
推理环境与训练环境分离；不使用旧训练环境启动这个标注工具。
使用你指定的 `/path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh`，
新环境名为 **`ar-qwen38-annotate`**，默认在该 Miniconda 的 envs 目录下。

```bash
cd /path/to/workspace/zwj/AR-Annotate-Qwen38
nvidia-smi
bash scripts/setup_env.sh
source /path/to/workspace/zwj/miniconda3/etc/profile.d/conda.sh
conda activate ar-qwen38-annotate
python -m ar_annotate doctor
```

脚本创建 Conda Python 3.11 + pip 后，在新环境中设置：

```bash
python -m pip config --site set global.index-url http://pypi.chinatelecom.ai/simple
python -m pip config --site set install.trusted-host pypi.chinatelecom.ai
```

`--site` 把同样的配置键写入这个 Python 环境的 pip.conf，避免更改其他环境的用户级配置。
安装命令也明确指定该镜像，清空额外索引；**torch/torchvision 同样从该镜像安装，不回退到外网 CUDA wheel 站点**。
镜像没有满足版本的包时会报错；需同步对应 wheel 或提供离线包，不会静默安装不支持模型的旧版本。
Conda 创建 Python 本身走已有 Conda channel 配置，pip 镜像不能替代 Conda channel。

应确认实际安装 wheel 的 CUDA runtime 与驱动匹配，`doctor` 会打印 torch/CUDA/GPU 信息。
安装后的版本快照为 `$CONDA_PREFIX/environment.freeze.txt` 和 `environment.conda.yml`。
已有同名环境默认不更改；中断后明确重试本标注环境，可用 `RESUME_INSTALL=1 bash scripts/setup_env.sh`。
可用 `ANNOTATION_ENV_NAME` 自定义新名字，不能使用 base 或已有训练环境。
默认模型加载均为 `local_files_only=True`。

`doctor` 只检查 CUDA、权重分片清单和 processor，**不代表真实视频已跑通**。
如果两台服务器的 Miniconda 环境目录不共享，需要在两台分别安装，且使用同一依赖版本快照。
如果环境目录共享，只安装一次，不要两台同时对同一环境执行 pip 安装。

## 1. 准备新增样本

`config.server.json` 已填写会话中确认的服务器路径，原 RLSD 数据指向：
`/path/to/workspace/zwj/AR-RLSD-think/data/train.rlsd.think.jsonl`。
首次使用检查该文件中的路径即可。准备步骤不加载模型，可用系统 Python 执行。

```bash
python -m ar_annotate prepare --out work/candidates_v1
```

产物：

```text
work/candidates_v1/
  candidates.jsonl    # 已筛选、尚未做视觉标注的清单
  rejected.jsonl      # 未匹配/重名/类别冲突/评测重叠等，含原因
  manifest.json       # 实际数量、扫描范围、输入哈希
```

输出目录必须是新目录，不覆盖历史清单。数据目录软链接不会递归进入，跳过的目录写入 manifest；
根目录不可访问或扫描发生访问错误会直接报错。不要把跳过范围或未匹配理解为全盘不存在。
SSv2 与 K710 答案会统一为小写下划线；含不受当前 reward 支持的标点的标签被拒绝并记录，
不会默默删掉标点、缩写动作或合并类别。

评测清单更新后，可从新的本地/服务器评测工程重新生成排除文件，再更新配置路径：

```bash
python -m ar_annotate exclusions \
  --eval-root /实际路径/VideoSTAR-Eval-v5 \
  --out assets/eval_exclusions.updated.json
```

## 2. 小批标注与查看

第一遍各数据集只取 5 条，确认模型/视频/输出格式。样本按固定哈希顺序抽取，不只取源文件前几条。

```bash
CUDA_VISIBLE_DEVICES=0 python -m ar_annotate annotate \
  --candidates work/candidates_v1/candidates.jsonl \
  --out work/smoke_v1 --limit-per-dataset 5
```

再做每个数据集 100 条的小批，人工看原视频和动作描述。只用 Qwen3.8，不做其他模型对比：

```bash
CUDA_VISIBLE_DEVICES=0 python -m ar_annotate annotate \
  --candidates work/candidates_v1/candidates.jsonl \
  --out work/pilot_v1 --limit-per-dataset 100
```

每条结果位于 `work/pilot_v1/records/<sample_id>.json`，里面有 `evidence`、`verdict`、
原始响应、采样信息和状态。候选清单用 sample_id 关联视频路径与原标签。
结果还记录各可见 GPU 的峰值已分配显存（不等于 nvidia-smi 的总占用）和设备映射。
若单卡出现 OOM/CPU offload，换 `CUDA_VISIBLE_DEVICES=0,1` 并使用新的 smoke 输出目录。
重点抽查 SSv2 的方向、先后、假动作/真实动作；K710 的片段内容与类别是否吻合。
第二遍核验仍是同一个模型，不能视为独立人工真值。

自动标注器关闭模型原生 thinking，要求输出 JSON；程序将可见动作证据整理为训练的 `<think>` 层级。
两者用途不同，不保存模型自由展开的思维链作为训练目标。

### 帧采样

默认 2 fps、最多 16 帧、约 100352 pixels，与当前训练设置对应。
qwen-vl-utils 提前解码采样，processor 使用 `do_resize=False, do_sample_frames=False`，
并传入真实视频 metadata，避免二次抽帧和时间丢失。
Qwen3.8 与学生的空间 patch 尺寸不同，**不声称预处理像素或所有库版本下的采样逐位一致**；
真实帧索引/时间戳都写入结果，服务器小批时应确认。
使用 decord 是为了接近当前训练读取方式；解码子进程超时 120 秒后记为错误，不会挂死全批。
v2 每个标注进程复用一个常驻 CPU 解码进程，避免每条视频重新导入 PyTorch；
发生解码错误/超时会终止并重建这个进程。两轮标注复用同一组帧。

## 3. 全量与断点续跑

单卡小批通过后，可手动单进程运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m ar_annotate annotate \
  --candidates work/candidates_v1/candidates.jsonl --out work/full_v1
```

中断后重复同一命令，已有结果会跳过。重跑错误样本追加 `--retry-errors`，旧错误记录会归档。
`needs_review` 不会自动重跑到符合原答案，也不会自动进入训练。
更改 prompt、配置、依赖、模型文件或分片总数，必须用新输出目录，避免混合不同版本。
pilot 与 full 是不同任务目录；本版不自动把 pilot 结果迁入 full。

### 单机 8 卡

默认 **每卡一个副本，8 个进程处理互不重叠的样本**。不是把一个 27B 模型拆到 8 卡。
脚本自动激活 Conda 环境、划分 GPU、生成日志，并检查子进程退出码：

```bash
RUN_DIR="$PWD/work/full_8gpu_v2" bash scripts/launch_annotation.sh
```

先检查分配、不加载模型：

```bash
DRY_RUN=1 bash scripts/launch_annotation.sh
```

单卡显存不足时，`GPUS_PER_WORKER=2` 表示 4 个副本、每副本可见两张卡：

```bash
GPUS_PER_WORKER=2 RUN_DIR="$PWD/work/full_8gpu_pairs_v2" bash scripts/launch_annotation.sh
```

两卡副本依然是 Transformers device_map 层分配，不是高效 tensor parallel，
也不保证两卡都有高利用率；单卡能运行时优先单卡副本。吞吐提升需实际测量，不承诺线性加速。

### 双机各 8 卡，共 16 卡

两台需要相同代码、配置、依赖版本，以及相同绝对路径可访问的模型/视频/候选清单。
输出目录必须是真正共享的文件系统，支持 POSIX 原子重命名和跨节点 advisory file locks。
不同节点不要使用同一个 NODE_RANK。脚本不通过 SSH 自动启动另一台。

节点 0：

```bash
NODE_COUNT=2 NODE_RANK=0 RUN_DIR=/path/to/workspace/zwj/AR-Annotate-Qwen38/work/full_16gpu_v2 \
  bash scripts/launch_annotation.sh
```

节点 1：

```bash
NODE_COUNT=2 NODE_RANK=1 RUN_DIR=/path/to/workspace/zwj/AR-Annotate-Qwen38/work/full_16gpu_v2 \
  bash scripts/launch_annotation.sh
```

默认分别负责全局分片 0..7 和 8..15，共 16 个单卡副本。
两卡一副本时两台都设置 `GPUS_PER_WORKER=2`，总计 8 个副本。
两台每台的 GPU_IDS 数量、GPUS_PER_WORKER、NODE_COUNT 必须一致。
若是一台 16 卡机器，可设置 `GPU_IDS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`，保留 NODE_COUNT=1。

中断后使用完全相同的命令恢复；只重试技术错误时添加 `RETRY_ERRORS=1`。
每分片有锁；只有完整、无错误的分片正常退出。一个节点退出成功不代表另一节点成功，
最终 export 会检查所有选中样本。日志在 `RUN_DIR/logs/`，每次启动新建日志文件。
v1 已开始的任务应继续用 v1 代码；v2 改了任务指纹，需要新的 run 目录，不能混入旧结果。

## 4. 导出并接入原训练工程

完成全量标注后，在**标注 Conda 环境**运行；`--run` 填实际运行目录，例如 8 卡：

```bash
python -m ar_annotate export \
  --candidates work/candidates_v1/candidates.jsonl \
  --run work/full_8gpu_v2 \
  --out /path/to/workspace/zwj/AR-RLSD-think/data/expanded_qwen38_v1
```

默认合并配置里的原始 4,676 条训练数据，原文件不改；生成成对 RLSD/SFT、审计 JSONL、
`review.csv` 和数量/哈希 manifest。媒体会重新核对哈希，不能在标注后悄悄替换视频。
本版自动接收标签一致且证据支持的条目，所有新增 reference_status 都是 `unreviewed`。
现有训练配置允许这种参考参与损失，正式训练前应人工抽查，而不是将其当作已核验真值。

默认参考长度上限是学生 tokenizer 的 768 tokens（含 EOS），与当前 rollout 上限对齐，
不会照搬标注模型 1536 的输出预算。训练 prompt 沿用原文件，不包含类别、选项或文件名。
长度超限和待审查样本不导出，计数不会隐藏。

缺记录/技术错误时默认拒绝导出；若明确只想拿部分成功数据，可用 `--allow-partial`，
manifest 会标明未完成数量。为 pilot 导出时使用独立的 pilot 输出名，不要误当全量。

随后换回你原来的 RLSD 环境做检查与 smoke：

```bash
conda deactivate
cd /path/to/workspace/zwj/AR-RLSD-think
source /path/to/workspace/zwj/.venvs/open-o3-visd-vllm/bin/activate
python -m verl.ar.prepare_data --check data/expanded_qwen38_v1/train.rlsd.think.jsonl --check-media
python -m verl.ar.prepare_data --check data/expanded_qwen38_v1/train.sft.think.jsonl --check-media
export TRAIN_DATA="$PWD/data/expanded_qwen38_v1/train.rlsd.think.jsonl"
export SFT_DATA="$PWD/data/expanded_qwen38_v1/train.sft.think.jsonl"
bash examples/action_recognition/run_qwen25.sh smoke
```

上面的 smoke 仍需按原工程启动 Ray、配置正确节点 IP，并确保训练节点也能访问相同视频。
标注运行不自动启动训练，也不改变原工程的参考损失、优化器或评测设置。
新增数据会改变采样比例与一个 epoch 的步数；应使用导出 manifest 的实际数量记录实验预算。

## 测试与来源

```bash
python3 -m unittest discover -s tests -v
```

CPU 测试覆盖严格匹配、评测重叠排除、标签冲突、无标签首轮提示、核验不一致不重试、
格式失败、媒体变更、断点续跑、任务指纹、未完成导出拒绝、长度限制、SFT/RLSD 对齐。
在旁边有 AR-RLSD 工程时，导出样本还通过其真实 `prepare_data.validate_rows` 校验。
GPU 模型加载、真实帧理解、显存峰值、每小时吞吐需在服务器 smoke 验证。

官方文档、第三方参考及采用/未采用的内容见 [SOURCES.md](SOURCES.md)。
