# Qwen3-VL 适配补充

本目录沿用下文 v6 的标注、候选、prompt、生成参数和计分规则；Qwen3 视频输入由
`qwen3_video.py` 处理。保留原版抽帧位置和像素预算规则，空间网格改为模型要求的 32，
并传入原始 FPS 和实际帧索引，关闭重复抽帧/缩放。不能宣称与 Qwen2.5 输入像素/视觉 token 完全一致。
运行清单记录这些差异；默认结果目录是 `VideoSTAR-Eval-results-v6-qwen3`。
完整使用说明和验证边界见 [README.md](README.md)。下面保留原 v6 协议说明。

# v6 单轮无工具评测：对齐范围与已知差异

上游：<https://github.com/AMAP-ML/Video-STAR>，固定提交 `8b5576ca3dcf690399a440816a6742b5eb6fb6ee`。
论文：<https://arxiv.org/html/2510.08480v2>，表 1 为 Base-to-Novel，表 2 为 Full / 三个 Split 的平均。

2026-10-04 从 GitHub API 获取该提交的完整 Git tree，随包保存为 `UPSTREAM_TREE.json`。
`SOURCE_MANIFEST.json` 中 73 个官方文件全部通过远端 Git blob SHA-1 及本地 SHA-256 核对。
包含原始 JSON 标注、候选类别文件、启动脚本、推理模板、视频处理器、计分代码；原文件没有修改。
运行 `python audit_protocol.py` 可重新核验并生成 `ALIGNMENT.json`。

## 十二个评测划分

| 设置 | 官方标注（相对 AFile/） | 视频数 | 候选类别数 |
| --- | --- | ---: | ---: |
| HMDB51 Base | `json/hmdb51/val.json` | 780 | 26 |
| HMDB51 Novel | `json/hmdb51/test.json` | 750 | 25 |
| UCF101 Base | `json/ucf101/val.json` | 2,088 | 51 |
| UCF101 Novel | `json/ucf101/test.json` | 1,695 | 50 |
| SSv2 Base | `json/SSv2/val.json` | 17,562 | 87 |
| SSv2 Novel | `json/SSv2/test.json` | 7,215 | 87 |
| HMDB51 split1（全部 51 类） | `json_split/hmdb51/hmdb_exp_split_1.json` | 1,530 | 51 |
| HMDB51 split2（全部 51 类） | `json_split/hmdb51/hmdb_exp_split_2.json` | 1,530 | 51 |
| HMDB51 split3（全部 51 类） | `json_split/hmdb51/hmdb_exp_split_3.json` | 1,530 | 51 |
| UCF101 split1（全部 101 类） | `json_split/ucf101/ucf_split_1.json` | 3,783 | 101 |
| UCF101 split2（全部 101 类） | `json_split/ucf101/ucf_split_2.json` | 3,734 | 101 |
| UCF101 split3（全部 101 类） | `json_split/ucf101/ucf_split_3.json` | 3,696 | 101 |

候选列表分别来自上游对应启动脚本的 `--json-file-path`，保留文件顺序，不按样本标签删选或重排。
每个模型共 45,893 次单轮识别，涉及 40,233 个不同的“数据集 + 视频路径”。
HMDB/UCF Base+Novel 与各自 split1 的视频路径集合相同，但候选集合不同，必须分别推理。
SSv2 这两组是公开代码对原始验证集的类别划分，不能因文件名为 `test.json` 就称为原始无标签测试集。

## 官方 HMDB 标注存在的矛盾

逐条比较发现：同一视频在 Base/Novel 与 split1 文件中有 120 条标签不一致，四组各 30 条：

| 视频所属目录 | Base/Novel 标注 | split1 标注 |
| --- | --- | --- |
| `sword` | Sword exercise | sword |
| `sword_exercise` | Sword | sword exercise |
| `kick` | Kick ball | kick |
| `kick_ball` | Kick | kick ball |

v6 **保留每套官方标注原样**，不会为得到一致结果而偷偷改标签。全部差异列在 `ALIGNMENT.json` 的
`official_annotation_conflicts.hmdb51`。这些是公开文件之间可复核的矛盾，不能仅据此认定论文使用了同样的标注。
UCF101 两套设置的同一视频标签在官方计分归一化后全部一致。

## 推理与计分

输入原始视频和当前划分的全部候选类别；每个样本只调用一次生成。
使用本包 `protocol.py` 的 `RECOGNITION_TEMPLATE`，即通用无工具动作识别指令；输出 `<think>简短视觉证据摘要</think><answer>一个候选类别</answer>`。
不执行第一轮工具选择，不调用姿态估计、人物检测、在线检索或类别解释工具；固定使用官方候选函数的 `False` 分支，即仅类别名。
v6 不再使用官方第二轮模板：不要求身体部位映射、2–3 个候选评分、引用未提供的动作解释，也不包含标注视频说明。
统一考虑可见运动、时序、人物/物体交互及物体位置/状态变化，不强制层级树结构。仅候选类别列表随划分变化，两个模型使用相同的指令。
模型可见文本不含媒体路径、文件名或单条样本的 problem / explain / solution。
真实路径仅传给视频解码器；聊天模板仅收到视频类型占位符。

沿用上游修改过的 qwen-vl-utils 视频读取、帧采样、缩放与 FPS 元数据处理。
其默认帧数根据 2 FPS 估计，通常限制在 4–16 帧，并沿全视频均匀选取；极短视频仍受实际总帧数约束。
不替换成训练时的帧数设置。默认使用 decord，环境变量覆盖会进入任务记录，模型间必须保持一致。
沿用 `temperature=0.8, top_p=0.001, max_tokens=4096, max_model_len=16384`。
八个 GPU worker 各 TP=1、batch size=2，种子为 `seed + shard_index`。这是工程并行方式的变化，不能承诺与上游 batch size=50 逐 token 一致。

最终答案使用官方提取与归一化函数，计算样本加权 Top-1。HM = 2 × Base × Novel / (Base + Novel)，用未舍入准确率计算，两者都为零时 HM=0。
每个 split 内按视频数加权合并八个 worker；三个 split 之间采用准确率的等权算术平均。
总体标准差为 sqrt(sum((accuracy_i - mean)^2) / 3)，即 ddof=0；CSV 以百分点显示。此 std 定义是本包显式约定，未声称论文明确使用同一定义。
同一视频如果属于多个 split，会分别参加每个 split 的评测，不跨 split 去重、混合分母或复用输出。

无法解析的输出仍在分母中；推理/解码异常记录并阻止写入完整总表，不把失败样本静默丢掉。

## 对齐声明的边界

- 已核实：上述十二个划分的公开标注、候选文件、顺序、官方采样源码、生成参数及计分规则。
- 明确改动：一次生成、工具全关、类别解释全关、使用通用无工具 prompt、无路径段落；不是完整的 Video-STAR 工具增强 pipeline，也不是论文分数的严格复现。
- HMDB/UCF 采用公开 split1/2/3 文件，分别报告准确率和三划分等权平均；本次不新增 Full 或 K600。
- 本地 AR 训练标注有 4,676 条、51 类，与 HMDB Base/Novel 和 split1 的视频路径无重叠，但覆盖了 Novel 类别。
  与 HMDB split2 / split3 则分别有 1,340 / 1,311 个视频路径重叠。按用户要求仍完整运行三个 split，不删除这些视频；总表记录重叠说明。
  因此 HMDB Novel 是官方分组名，不能描述为该模型微调未见类泛化；含训练视频的三划分均值也不是纯粹未见视频泛化。
  以上是本地参考 AR 训练文件的核对结果；提供服务器实际 `--train-data` 后会另记该文件哈希和实际重叠数。训练协议不会因更换评测文件而自动对齐论文。
- 本地不能验证服务器上每段视频的字节内容与解码成功率；服务器启动时检查路径，推理中检查解码错误。路径都存在不等于逐字节验证了全部视频。
- 本次本地测试覆盖 CPU 流程、八进程控制、断点恢复、分片合并和模板渲染；真实 GPU 验证由服务器运行完成。

两个待比较模型都使用此 v6 协议；v1/v2/v3/v4/v5 的缓存和总表不能并入。
原版 v6 目录为 `VideoSTAR-Eval-v5`，运行入口为 `run_v6.sh`，结果写入 `VideoSTAR-Eval-results-v6`。
本 Qwen3 适配目录、入口和结果目录按本文开头及 README 执行。
