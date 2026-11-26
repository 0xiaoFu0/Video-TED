# 本地验证记录（2026-10-13）

独立 v3 项目 **351 个测试全部通过，无失败或跳过，45.09 秒**。命令：

```bash
cd AR-RLSD
/Users/zwj/Desktop/CVPR/AR/AR-RLSD/.venv-test/bin/python -m pytest \
  tests/ar tests/test_protocol_reward_cpu.py \
  tests/test_merged_32gpu.py tests/test_cluster_topology.py -q
```

使用工作区已有 CPU 环境：Python 3.9.6、Torch 2.8.0、pytest 8.4.2、OmegaConf 2.3.1。
实际 fast tokenizer 测试使用已有 tokenizers 0.22.2，构造本地 ByteLevel BPE，未下载模型或安装依赖。
仅有既有 urllib3/LibreSSL 环境警告。

验证数值与梯度：sample std/1e-4、短长回答等权序列平均、不同 microbatch 划分等价、
负优势 ratio>3 时不执行 dual clip、参考失败权重仍按同一二元主奖励计算。
实际生产 actor 验证教师在完整更新末尾只 EMA 一次、精度转换/浮点与整数 buffers、
布局不一致时拒绝更新、teacher cutoff 后无教师张量/前向但参考梯度继续。

教师 checkpoint 测试执行真实 Torch DCP get/set state、torch.save/load，确认恢复的是
独立教师状态，拒绝本地/其他 rank 缺失、空或损坏分片。集体通信使用测试替身，模型在 CPU；
这些测试不代替实际 CUDA FSDP 断点保存/恢复验证。

生产 trainer 构造检查第 1/1000/1001 步的 λ、active 和 reference 保留；
左裁检查实际 tokenizer 中文重叠字符偏移、完整视觉块和聊天结构、相同 media 对象、
重新编码后 mRoPE/位置和 raw_prompt/input_ids 的一致性，以及原教师/参考字段保留。
无法容纳完整视觉骨架时明确报错。

结构化 PPO 配置、四节点启动路由、固定对齐参数与旧 alias 冲突检查通过。
实际 runner + 实际 validator 证明非法超长过滤在 tokenizer/dataloader/worker 分配前被拒绝。
旧 baseline、protocol 动态加载、宽标签、参考分支、真实 merger、节点/NCCL 调度替身和错误传播均通过。

另检查 116 个 Python 源文件 AST、4 个动作识别 Bash 脚本及 README 的 10 个 Bash 命令块。
93,583 条实际数据不在本地包内；包内 data、annotation、reward/format 和 teacher/student 模板
59 个文件与 v2 逐字节相同。v2 原有 223 个 manifest 项全部仍匹配，没有更改当前训练项目。
来源、变更文件和 SHA256 在 PROVENANCE.json/SHA256SUMS 中，zip 打包时校验 CRC 和解压后的 manifest。

**未运行服务器数据检查、GPU 视频前反向、真实 NCCL、32 卡 FSDP、完整训练或效果评测。**
按 README 依次执行 prepare、cluster、check-video、smoke；需等当前 v2 训练结束、四台节点可用。
smoke 结束仍保存独立 checkpoint，正式训练从基础模型新建运行。
EMA teacher checkpoint 增加磁盘容量/I/O；精确续训必须保留全部 teacher 分片和相同 32 ranks。
