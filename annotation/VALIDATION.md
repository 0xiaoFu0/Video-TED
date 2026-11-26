# 验证记录

2026-10-05，v4.2 本地 CPU 回归共 42 项通过：

- 默认阈值为 1；四次中仅一次通过可保留，无有效通过的样本仍为 needs_review/error。
- 一次通过、三次格式错误时正常聚合并导出通过结果，完整保留三次错误与选中 rollout 编号。
- 汇总 complete 可为 true，同时逐 rollout 错误仍准确计数；错误没有被改写为成功。
- 两节点启动脚本默认均传入 min-accepted-rollouts=1，复用 v4 候选，使用新运行目录。
- 尚未在服务器跑 v4.2 真实权重与视频；CPU 假模型测试验证程序逻辑，不证明视觉判断准确率。

2026-10-05，v4.1 本地 CPU 回归共 39 项通过：

- 模拟服务器错误顺序：缺字段 → 9 条 observation → 合法输出；确认后续 prompt 收到具体校验反馈。
- 核验格式错误时仅重试核验，复用合法证据；现有语义分歧不重试测试保持通过。
- 指定 sample-id 仅运行该候选，未知 ID 拒绝；原候选文件哈希保持不变。
- v4.1 未在服务器进行真实 GPU 复测；不承诺增加格式反馈后所有样本一定成功。

2026-10-05，v4 本地 CPU 回归验证，共 36 项通过：

- 复用原 SSv2/K710 候选逐行保持不变；加入 HMDB51 视频/标签与原样本 ID，旧 CoT 不进入候选。
- HMDB51 缺失媒体、冲突标签、路径穿越、候选与 manifest 不一致均有明确处理。
- `tree_v1` 在无标签首轮生成树；正确类别但层级不合格仍为 needs_review，缺少层级核验字段视为格式错误。
- 模拟 16 个全局分片完整、无重复覆盖三个数据集；每视频四次，一视频导出一条，SFT/RLSD 配对一致。
- `--pool` 不读取或追加旧 base；普通合并模式遇到 HMDB51 新候选会拒绝，避免旧/新 HMDB 重复。
- 全局统计检查 16 个汇总；即使记录齐全，缺任一分片汇总也拒绝默认完整导出。
- 两节点 wrapper dry-run 验证 rank 0 = shards 0..7，rank 1 = shards 8..15。
- prepare-combined / export / summarize CLI 帮助、三个 shell 脚本语法与 Python compileall 通过。
- 本地实际读取旧 HMDB51 源清单：4,676 条、51 类；评测重叠：base/novel/split1 为 0，
  split2 = 1,340，split3 = 1,311，full = 3,570；并集为 3,896。视频在远端，本地没有验证媒体内容。
- 分别尝试 SSH 到用户提供的两节点，均得到 Connection closed，未验证远端 GPU/共享盘，未部署或启动。

v4 待验证：两节点共享文件系统、实际 16 卡环境、新树 prompt 的质量、HMDB51 视频解码和全量运行。

2026-10-05，v3 本地 CPU 回归验证（实际命令输出，不加载模型）：

- 原有 20 项和新增 10 项测试共 30 项通过。
- 新增固定 rollout 预算、2/4 不通过与 3/4 通过、每视频一次解码、独立种子、
  中途打断只补缺失、只重试技术错误、保留有效分歧、修改采样/结果拒绝恢复或导出。
- 错误重试再次中断时可恢复；正常完成与打断均关闭解码 backend。
- 模拟端到端 4 次 rollout 后每视频只导出一条，SFT/RLSD 对齐并保留汇总元数据。
- 八卡启动脚本将 rollout 数、阈值、seed 与 prompt profile 传给所有 worker。
- v3 真实模型兼容性、速度和 prompt 效果仍需服务器 smoke 与同批对照。

用户在会话提供的 v2 服务器截图（非本地复现）：

- Python 3.10 专用 Conda 环境；Transformers 5.8.0；torch 2.6.0 / torchvision 0.21.0；A800 80GB。
- CUDA 矩阵运算和两条视频标注成功；Decord 失败时 torchvision 回退成功。
- 八卡 pilot 共 200 条，缺失 0、任务不匹配 0、八份分片汇总。
  K710 自动通过 77/100，SSv2 自动通过 33/100；这不是准确率。

以下保留 v2 发布时的本地验证记录和当时的待验证项。

2026-10-04，本地 CPU 验证，未使用服务器权重或媒体。

- `python3 -m unittest discover -s tests -v`：v2 共 20 项通过。
- `python3 -m ar_annotate --help`：正常，提供 exclusions / prepare / annotate / export / doctor。
- `bash -n scripts/setup_env.sh`、`bash -n scripts/launch_annotation.sh`：通过。
- `PYTHONPYCACHEPREFIX=/tmp/ar_qwen38_pycache python3 -m compileall -q ar_annotate tests`：通过。
- 模拟的端到端流程覆盖准备、两次模型调用、原子结果、恢复、分片、错误归档、合并导出。
- 成对导出样本通过实际 `AR-RLSD/verl/ar/prepare_data.py` 的 validator；未修改训练工程。
- 评测排除资产实际读取本地七份评测 JSON，保留源路径、行数和 SHA256。
- v2 增加 8/16 卡分片计划、双节点不重复分片、双卡回退、非法 GPU 参数测试。
- v2 实际启动 CPU 模拟解码子进程，验证进程复用、超时终止及异常后重建。

尚待服务器验证：

- 安装后的 torch / Transformers / qwen-vl-utils 与 NVIDIA 驱动组合。
- Qwen3.8-27B BF16 权重加载、设备分配、视频 processor、CUDA 前向。
- 真实 SSv2 WebM 与 Kinetics MP4 的读取及帧时间信息。
- 小批生成质量、核验误判、显存峰值、吞吐和全量覆盖。
- Conda 创建和电信镜像包可用性、跨节点共享文件系统锁语义。

部署命令与小批流程见 README。CPU 测试的假模型只验证程序行为，不证明实际模型的准确率。
