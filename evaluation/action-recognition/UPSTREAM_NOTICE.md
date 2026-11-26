# 来源说明

官方项目：https://github.com/AMAP-ML/Video-STAR

固定提交：`8b5576ca3dcf690399a440816a6742b5eb6fb6ee`

本包为用户本地实验整理官方已公开的评测源码与标注。
`upstream/AScripts/`、`upstream/src/qwen-vl-utils/`、`upstream/README.md`、
`upstream/requirements.txt` 及 `AFile/` 保持来源文件字节不变。
官方仓库的 `src/r1-v/LICENSE` 按原路径附带；不将该子目录的许可自动扩大解释为整个仓库的许可。
原作者署名、论文引用和项目说明见 `upstream/README.md`。

本包新增顶层启动、路径检查、恢复、对比和导出代码；改动范围详见顶层 README。
v6 的 `protocol.py` 使用新编写的通用无工具识别 prompt，不再沿用官方第二轮模板正文。
官方候选类别提取、采样及计分仍保持；聊天模板只接收视频类型占位符，真实路径只提供给媒体读取器。
上述改动不写回 `upstream/`；独立 v6 协议标识阻止旧 prompt 分数混入新表，不声称精确复现论文数值。
模型合并流程参考当前 AR-RLSD 的 `scripts/model_merger.py`，增加独立导出目录与分片完整性检查。
Base/Novel 与 split1/2/3 都保留官方标注；HMDB 两套标注有 120 条差异，见 PROTOCOL.md。
