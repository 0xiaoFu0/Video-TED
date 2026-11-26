# 交付验证范围

验证日期：2026-10-17。开发机为 macOS，未安装 CUDA/vLLM；以下本地检查不代表新增协议已完成真实模型 GPU 推理。

## v1.2.2 简短 prompt / 64 帧与 SEED 8 帧

- 新增 `simple_64` 和 `seedbench_simple_8`，分别保留原 `videor1_64` 和 `seedbench_8` 的帧输入与生成参数，只改变 prompt 和协议描述。现有命名配置保持原有含义。
- 66 项相关 CPU 测试通过，其中新增 8 项验证两个新配置、所有 benchmark 的相同简短 prompt、参数冲突拒绝、64 帧/短片实际图片解码、SEED 8 帧保持与禁止二次采样、数据采样来源记录和双阶段 shell 启动/失败停止。此次未重复运行旧版已知受操作系统权限影响的孤儿进程清理测试。
- 专用启动器 CPU mock 顺序完成 8280/8280 和 3757/3757 题，共 12037 题，两阶段均完整。前三项采用合成标注及占位媒体；SEED 采用官方标注与合成图片。检查保存配置、样本采样标记及输出题数；再次执行两阶段均 0 remaining，所有预测文件字节不变。
- 记录见 `validation/simple64_cpu_audit.json`。这些是配置与管线检查，不能作为模型成绩或真实 GPU 推理已成功的证据。

## v1.2.1 简短 prompt / 32 帧

- 新增 `simple_32` 配置和 `eval_simple32.sh`：保留原 2 FPS 向下取偶数的均匀采样、片段边界、像素与生成参数，采用 generic 简短推理 prompt 和 32 帧上限。SEED-Bench 显式拒绝。
- 142 项测试中 141 项通过，含全部 6 项新增检查。已有的孤儿进程清理测试在 macOS 环境中因 `os.killpg` 返回 `EPERM` 报错，单独重试仍同样报错；本次未修改该清理逻辑，不能声称全套测试全部通过。
- 新增检查覆盖完整 prompt、其他参数与 `videor1_64` 一致、冲突覆盖拒绝、短片少于 32 帧、长片覆盖首尾、MVBench 帧目录实际解码/片段边界、启动脚本清理遗留的仅计分/小样本状态及协议 metadata。已有续跑隔离测试增加新配置验证，拒绝覆盖旧实验。
- 实际运行专用 shell 启动器，采用合成标注与占位媒体，CPU mock 双 worker 完成 8280/8280 题（4000+2700+1580）；三项均完整。再次运行显示 8280 saved / 0 remaining，预测文件字节不变。审计见 `validation/simple32_cpu_audit.json`，这些输出不是模型成绩，也未进行真实 GPU 推理。

## v1.2.0 SEED-Bench Video

- **136项CPU单元/集成测试全部通过**，涵盖原三个benchmark与新增SEED路径、过滤、计分、8帧解码、启动脚本、续跑和离线计分。
- 实际读取官方ZIP的中央目录，与官方SEED-Bench.json逐题核对。生产loader生成的30056条帧路径与官方目录完全一致，对应3757题，三个任务1509/1225/1023题。仅下载中央目录与标注，未下载官方PNG内容；见`validation/seedbench_mapping_audit.json`。
- 官方真实标注+合成PNG的双worker CPU mock全量运行两次，各3757/3757题、scope=full；逐题固定回答A，正确数由标注独立核验为Overall924、Action378、Prediction279、Procedure267。它们是管线测试，不是模型成绩。
- 全部30056张合成PNG通过图片预检查；RGB解码、顺序与像素预算另有实际JPEG/PNG测试。缺失第8帧和坏图片会失败，禁止二次裁剪、补帧、64帧覆盖及越界路径。
- 再次运行全量显示3757 saved / 0 remaining，预测文件哈希不变。比较器输出Overall/Action的Markdown和CSV；见`validation/seedbench_cpu_audit.json`。
- `seedbench_8`和原`videor1_64`隔离；启动脚本覆盖旧环境的64帧/8192token等参数。五题smoke覆盖三个任务，全量图像检查在GPU初始化前完成。
- 模型接口的时间信息标为synthetic_ordinal_2fps，不冒充原视频时间戳。尚未在服务器执行SEED真实GPU推理；请执行`SEEDBENCH.md`中的run命令完成服务器检查和试跑。

## 既有版本记录

v1.1.0 新增 `videor1_64` 独立协议与两模型比较启动脚本。协议固定 Video-R1 参考 prompt、64 帧上限/2 FPS 向下偶数采样、100352–200704 每帧像素、temperature=0.01、top_p=0.001、repetition_penalty=1、最大生成 1024 与上下文 16384。记录论文与仓库温度差异；MVBench 官方片段边界、TVQA 帧目录、NTU MP4 兼容与 TempCompass 原生 MCQ 范围保持。

此前 v1.0.3 服务器全量输出已由用户截图确认：MVBench 4000/4000、Avg 65.55、FA 46.50；Video-MME 2700/2700、Overall 57.78、Action 56.87，两项 Complete=True。它们属于旧协议；不作为 v1.1.0 新协议的 GPU 验证结果。

v1.0.1 根据服务器终端输出修正 MVBench 默认路径为 `/path/to/workspace/datasets/MVBench`，并确认 `json/` 下 20 个任务文件的原版布局。Video-MME 根目录及 parquet 标注位置也已由服务器输出确认；远程题数、视频完整性与模型推理仍需预检查及小样本运行验证。

v1.0.2 修复服务器发现的单选项校验错误。官方 MVBench revision `230a2d4fac8900333c61754641c7a13e069ac9c6` 的 20 个 JSON 共 4,000 题，其中 Counterfactual Inference 索引 `81/98/111/137/162` 的 5 题只有一个选项。所有题目均成功加载；完整双 worker mock 管线验证通过。该 MVBench 测试使用媒体路径占位文件，不代表真实 MVBench 视频解码或 GPU 推理已验证。逐任务校验和与计数记录在 `validation/mvbench_official_annotation_audit.json`。

v1.0.3 根据服务器目录输出增加 Fine-grained Pose 同 ID MP4 兼容：`nturgbd/` 和 `nturgbd_convert/` 各有 200 个 MP4，与全部 200 条 AVI 标注同名匹配。原 AVI 优先、MP4 备选、转换目录备选、其他任务不变、缺失文件仍报错均有针对性测试。

## 已执行

- **114 项 CPU 单元与集成测试全部通过**：数据布局、时间区间、TVQA 帧目录、预裁剪视频、可变选项数、答案解析、分项计分、多进程和启动脚本。新增命名协议、环境变量冲突、uniform_fps_floor、两模型配置一致性、对比拒绝错误数据/协议和双模型启动顺序检查；包含 8 项单选项及 5 项 NTU MP4 兼容回归检查。
- 新两模型启动脚本完整 CPU mock 验证：两次 smoke 各 10 条、两次全量各 6700 条，均完整；自动比较与 Markdown/CSV 输出成功。再次执行全部显示 0 remaining，不重复预测。MVBench 使用真实官方 4000 条标注和占位媒体；Video-MME 使用明确的合成 2700 条/900 个占位视频，所以这些 mock 分数不是模型成绩。
- 新采样与像素预算的真实视频解码：OpenCV 读取实际 TempCompass 视频，6.875 秒输入得到 12 帧、308×588，采样覆盖首尾；最大帧数和像素预算均满足协议。审计记录在 `validation/videor1_64_cpu_audit.json`。
- 两模型脚本会在试跑某个 benchmark 全部答案无法解析时停止；可解析但答错的试跑不会因准确率低而被阻止。
- 三 benchmark 双 worker shell 集成：保留全部预期题目，输出 JSON/CSV/Markdown；重复启动不重复预测，修改协议会拒绝混用旧结果。
- 断点恢复：中断 JSONL 尾行可恢复；完整行损坏、重复/未知 ID 被拒绝；离线计分不需要原数据或模型。
- 失败路径：worker 异常后保留已完成预测；缺失题不缩小计分分母；无进度超时返回非零；SIGTERM 清理 worker；已退出 group leader 的后代进程也被终止；同实验并发写入被拒绝。
- 真实 TempCompass-MCQ 标注与全部视频：两 worker mock shell 全量运行 **1,580/1,580 题、410 个视频**，无缺失预测、无无效答案、无运行错误；只包含 `multi-choice`。
- 真实 MP4 解码：使用显式 OpenCV 解码器验证均匀抽帧、MVBench 时间裁剪、分段中点采样和 Qwen 时间元数据，并与独立逐帧读取及缩放结果逐像素比较。
- Shell 语法与全部 Python 源文件语法检查。

原始 MCQ parquet SHA256：

```text
c4dfa7b87007d83aa2b6e11b1ad7e6a176ab628cefb5e42bd89c913ca73e6a78
```

固定输出 A 的 mock 控制用于独立核对计分器，不是模型实验成绩：

| 分组 | 题数 | A 为标准答案的题数 |
| --- | ---: | ---: |
| Overall | 1,580 | 461 |
| Action | 338 | 81 |
| Direction | 335 | 103 |
| Speed | 317 | 86 |
| Order | 302 | 110 |
| Attribute change | 288 | 81 |

这些计数由原始 parquet 独立计算，与评测结果逐项相符。证据保存在 `validation/tempcompass_audit.json` 和 `validation/real_video_decode.json`。临时验证数据与输出路径仅用于说明来源，不是运行依赖；视频文件和完整数据集不包含在代码包中。

## 复现

在安装数据处理依赖后运行：

```bash
python -m unittest discover -s tests -v
bash -n eval/scripts/eval_all.sh
```

使用本地完整 TempCompass 数据做 CPU 全量管线检查：

```bash
BACKEND=mock BENCHMARKS=tempcompass \
TEMPCOMPASS_DATA_DIR=/gemini-3/space/zwj/LMUData/TempCompass \
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 \
EXP_NAME=tempcompass_mcq_cpu_check THINK_MODE=1 N=1 \
  bash eval/scripts/eval_all.sh
```

mock 模式不初始化 GPU，设备编号仅用于验证多 worker 分片。汇总明确显示 `is_mock=true`。

## 服务器还需验证

真实模型 checkpoint、GPU 显存、服务器解码器和 CUDA/vLLM 组合，需要按 README 执行 `LIMIT=5` 的真实推理命令。本地未执行 Decord 的 Linux GPU 环境解码或完整 MVBench/Video-MME 真实视频推理；其读取布局已通过对应参考代码与 CPU fixtures 核对。

固定 benchmark、抽帧/像素预算、thinking、生成长度和投票配置后，再对各待比较模型进行完整评测。本包保存这些设置及源码/数据指纹供核对。
