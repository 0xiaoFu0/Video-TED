# 已有 v6 输出的 CPU 重评分

2026-10-05 的 Qwen3 输出样例在最后一行给出了正确的 `Brush hair`，但缺少 `<answer>` 标签。旧版提取器将其保存为 `predict=""`，使动作正确的回答仍被计为错误。HMDB split1 的截图显示 1,256/1,530 条无效答案；这不表示这些答案全部可恢复或全部正确。

此补丁增加独立的后处理程序，不修改原始预测、原表、提示词、数据集、推理代码或已发布的 v6 严格标签评分。新的评分版本为 `v6-answer-tags-or-exact-final-label-v1`。

## 执行

将 `rescore_answers.py` 放到已有的 `VideoSTAR-Eval-qwen3-v6` 目录，使用 Python 3.9 或以上版本即可，不需要 CUDA、vLLM、Transformers 或重新加载权重。

```bash
cd /path/to/workspace/zwj/VideoSTAR-Eval-qwen3-v6
python3 rescore_answers.py \
  --run-dir /gemini-3/space/zwj/VideoSTAR-Eval-results-v6-qwen3/runs/qwen3vl_8b_instruct_8gpu_v6
```

默认输出到原 run 内的新目录 `answer_rescore_v1/`。已有输出目录不会被覆盖；如需再次执行，指定不同的 `--output-dir`。

## 解析规则

1. 使用每个划分实际给模型的候选类别；解析函数不接收标准答案或视频路径。
2. 完整的 `<answer>` 标签大小写不敏感，内容必须匹配一个候选类别。多个互相冲突的标签、损坏标签、空标签或非候选答案均不恢复。
3. 无答案标签时，只有正常结束（`finish_reason=stop`）的回答才允许回退：最后一个非空行必须整体精确匹配一个候选类别。不同的独立候选行会判为歧义。
4. 不从解释段落中搜索类别，不做模糊匹配、同义词替换、语义裁判或根据标准答案挑选候选。未闭合的 `<think>` 不作为最终答案。
5. 仅沿用原评分器已有的规范化：忽略大小写、空格、下划线、换行和星号。`Answer: Brush hair` 或 `The action is brush hair.` 不属于本版末行回退规则。
6. 未解析、截断和错误样本仍保留在总分母中；完整明确答案标签可以在截断回答中计分，裸标签截断回答不恢复。

## 输出

- `models.csv` / `models.md`：新解析规则下的一行完整指标，显式记录评分版本和是否为子集。
- `models_strict.csv`：根据原始 `predict` 重算并核对的旧版严格标签分数。
- `audit.csv`：每个划分的前后准确率、无效/未解决答案数、新增正确数、丢失正确数和截断数。准确率列使用 0–1，模型总表使用 %。
- `summary.json`：两套分数、解析规则、源文件 SHA-256、原 manifest、训练重叠备注。
- `*.answers.jsonl`：逐条新旧答案、路径、索引、结束原因和解析方式。原始完整回答仍在输入 run 的 `results.json` 中。

程序验证十二个划分、标注及候选文件哈希、条目数量/顺序、标签、模型身份、原始汇总指标和原始提取器输出。默认只接受完整评测；`--allow-subset` 仅用于明确标记的诊断子集。检查失败不会发布不完整报告。

## 比较口径

新分数是“候选动作识别准确率”，旧分数包含严格标签格式要求，两者不得混用。比较 Qwen2.5、RLSD、SFT、Qwen3 前，对各自已有的 v6 原始输出运行同一脚本即可，不需要重做 GPU 推理。此脚本可在 Qwen3 代码目录中读取 Qwen2.5 的 v6 run。

此修复不证明所有剩余预测都正确，也不证明 GPU 输入链路不存在其他问题。请先查看重评分后的 `unresolved`、`truncated` 和逐条样例，再决定是否需要小样本补测。
