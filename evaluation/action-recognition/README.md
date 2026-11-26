# Qwen3-VL-8B-Instruct：v6 动作识别评测

这是独立的 `VideoSTAR-Eval-qwen3-v6` 目录。目标模型已经确认：
`/path/to/workspace/models/Qwen/Qwen3-VL-8B-Instruct`。
原来的 `VideoSTAR-Eval-v5` 和正在进行的 SFT 评测无需改动。

## 上传与启动

把 `VideoSTAR-Eval_qwen3_v6_20261004.tar.gz` 上传到服务器 `/path/to/workspace/zwj`，
在可用的 GPU 节点执行：

```bash
cd /path/to/workspace/zwj
tar -xzf VideoSTAR-Eval_qwen3_v6_20261004.tar.gz
cd VideoSTAR-Eval-qwen3-v6
sha256sum -c SHA256SUMS --quiet && bash run_qwen3.sh smoke
```

`smoke` 默认只用 GPU 0，对每个划分跑 2 个样本，共 24 个评测条目，不写入正式总表。
正常完成后启动八卡全量评测：

```bash
bash run_qwen3.sh run
```

默认八张卡是 0–7，每卡一个独立模型副本；batch size=2，12 个划分共 45,893 个评测条目。
可以用 `--gpus 8,9,10,11,12,13,14,15` 改成同一机器上另一组卡。
`bash run_qwen3.sh check` 只检查路径与分配方案，不证明模型可在 GPU 上推理。

中断后用相同设置执行 `bash run_qwen3.sh run --resume`。
不要把 smoke 的结果当全量结果，也不要把 smoke 的 run-name 用于全量任务。

## 结果位置

```text
/gemini-3/space/zwj/VideoSTAR-Eval-results-v6-qwen3/
  models.csv
  models.md
  models.json
  runs/qwen3vl_8b_instruct_smoke_v6/summary.json
  runs/qwen3vl_8b_instruct_8gpu_v6/summary.json
  runs/qwen3vl_8b_instruct_8gpu_v6/logs/shard_00.log
```

仍然每个完整模型一行，列名与已有 v6 总表一致：三个数据集 Base/Novel/HM，
HMDB/UCF 各三个 split、均值、总体标准差及训练重叠说明。
输入适配代码和配置哈希与旧版不同，因此使用单独的结果目录，保持旧总表的严格校验。
后续整理 Excel 时可并列展示，并注明 Qwen3 原生视频输入适配。

## 与当前 v6 的关系

- 标注文件、候选标签及顺序、12 个划分、v6 prompt、单轮无工具和计分函数均保留。
- 保留 vendored `smart_nframes`、目标 2 FPS、通常 4–16 帧（极短视频沿用原规则），
  `torch.linspace(...).round()` 的均匀抽帧位置及像素预算计算。
- 保留 temperature=0.8、top_p=0.001、max_tokens=4096、max_model_len=16384，
  种子为 seed + worker 编号。这是沿用现有 v6 比较设置，不是 Qwen 官方推荐解码参数。
- Qwen3 的 patch=16、merge=2，空间尺寸按 32 的倍数对齐；Qwen2.5 按 28 对齐。
  原始抽帧及像素预算规则一致，不代表最终尺寸、视觉 token 数或模型聊天模板字节一致。
- 视频以 `(TCHW frames, metadata)` 传给 vLLM；metadata 保存原始 FPS、真实帧索引、
  原视频帧数和解码器。设置 `do_sample_frames=False`、`do_resize=False`，避免再次抽帧或缩放。
- 媒体文件路径不进入 prompt 文本；只保留视频占位符和统一识别指令。
- 沿用原参考训练重叠说明与 HMDB 官方标注差异报告，见 `PROTOCOL.md` / `ALIGNMENT.json`。

运行清单额外保存 `model_input`、适配器源码哈希和 processor/tokenizer/聊天模板文件哈希，
配置变动会拒绝恢复旧任务。`upstream/` 官方文件保持不变。

## 环境和验证边界

默认复用 `/path/to/workspace/zwj/.venvs/open-o3-visd-vllm/bin/python`。
适配器检查 `vllm>=0.11.0`、`transformers>=4.57.0`；解码仍使用现有 decord / torchvision。
不要求升级安装的 qwen-vl-utils：本适配器使用已有 vendored 采样函数并补充时间元数据。
如需选择另一环境，设置 `EVAL_PYTHON=/实际环境/bin/python`，不会自动安装或升级软件。

本地已经验证真实小视频的 CPU 解码、相同抽帧位置、模型原生空间网格，以及官方
Qwen3-VL-8B-Instruct processor 对所有 12 套 prompt 的处理。具体测试见 `VALIDATION.json`。
本地没有执行 Qwen3 模型权重的真实 GPU 推理，也没有检查服务器全部视频，首次须先跑 smoke。
`VALIDATION_V6_BASE.json` / `PROMPT_AUDIT_V6_BASE.json` 是复制前 v6 的历史验证记录。

接口来源：
- [Qwen3-VL 官方视频处理说明](https://github.com/QwenLM/Qwen3-VL#new-qwen-vl-utils-usage)
- [vLLM 0.11.0 Qwen3-VL 视频元数据与时间戳接口](https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/models/qwen3_vl.html)

主要改动文件：`qwen3_video.py`、`evaluate.py`、`run_qwen3.sh`、`audit_protocol.py`；
新增验证：`tests/test_qwen3.py`。
