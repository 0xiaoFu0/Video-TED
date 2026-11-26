# 来源与实现取舍

查阅日期：2026-10-04。程序为本项目新写，没有复制第三方标注项目实现。

| 来源 | 用途与边界 |
|---|---|
| [Qwen3.8-27B 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B) | 确认图像/视频输入、默认 thinking 及关闭方式。非 thinking 采样采用 temperature=0.7 / top_p=0.8 / top_k=20；Transformers 本版未实现 API 的 presence_penalty，不能宣称逐项等价于官方 API 配置。 |
| [Transformers Qwen3.5 架构文档](https://huggingface.co/docs/transformers/model_doc/qwen3_5) | 使用 Qwen3_5ForConditionalGeneration 多模态类；不是只加载语言模型的 ForCausalLM。Qwen3.8 使用该架构。 |
| [vLLM Qwen3.8 recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) | 确认 config/processor 的 Transformers 5.8.0 起点。recipe 明确已验证的是文本 serving；没有据此宣称其 A800 视频路径验证通过。本工具未依赖 vLLM。 |
| [Qwen3-VL 官方工具说明](https://github.com/QwenLM/Qwen3-VL) | qwen-vl-utils 的视频帧与 metadata 接口、patch 参数、避免 processor 二次缩放。文档提示 decord 可能挂起，所以本项目隔离解码进程并设超时。 |
| [ShareGPT4Video / ShareCaptioner-Video](https://github.com/ShareGPT4Omni/ShareGPT4Video) | 参考批量视频 caption 与时序描述的任务拆分；没有复用其旧版训练环境、权重或长视频采样策略。这里输出短动作证据，贴合 AR-RLSD 的 16 帧输入。 |
| [Qwen Video Desk](https://github.com/jackzhenguo/qwen38-vllm-video) | 社区实际视频输入案例。其验证环境是 RTX5090 / NVFP4，文档报告请求级 fps 参数兼容问题。未照搬为本项目 A800 / BF16 的部署保证，也未采用其 Web UI。 |
| [PyTorch 官方安装页](https://pytorch.org/get-started/locally/) | 安装的 CUDA wheel 需与服务器驱动相容；以独立环境的 pip check、freeze 和真实 smoke 为准。v2 按用户指定镜像获取 pip 包。 |
| [pip 配置文档](https://pip.pypa.io/en/stable/topics/configuration/) | --site 将 global.index-url / install.trusted-host 写入环境级配置；安装命令显式指定镜像以覆盖外部 index-url。 |
| [Conda create 文档](https://docs.conda.io/projects/conda/en/latest/commands/create.html) | 使用用户给定 Miniconda 新建独立 Python 3.11 环境，不复用训练环境。 |

本地项目依据：

- `../AR-RLSD/data/train.rlsd.think.jsonl`：学生 prompt 与原始字段契约；旧训练数据未复制进本工具。
- `../AR-RLSD/verl/ar/reward.py`：类别规范化及精确匹配。
- `../AR-RLSD/verl/ar/prepare_data.py`：数据/参考回答格式校验。
- `../VideoSTAR-Eval-v5/benchmarks.json` 及相应 annotations：生成 `assets/eval_exclusions.json`；每个源文件哈希写入该资产。

当前工具没有完成官方 train split 的独立复核、跨所有 benchmark 的视频去重、人工视觉审核。
这些限制在候选 manifest、导出 manifest 与 README 中保留；目录名、模型一致性都不当作人工真值。
