"""One row per model, backed by verified full-split prediction counts."""
from __future__ import annotations

import csv
import fcntl
import io
import statistics
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from protocol import EVAL_SPLITS, PROMPT_PROTOCOL, atomic_json, read_json

SPLITS = EVAL_SPLITS
COLUMNS = ("Model", "HMDB51 Base (%)", "HMDB51 Novel (%)", "HMDB51 HM (%)",
           "UCF101 Base (%)", "UCF101 Novel (%)", "UCF101 HM (%)",
           "SSv2 Base (%)", "SSv2 Novel (%)", "SSv2 HM (%)",
           "HMDB51 split1 all 51 classes (%)", "HMDB51 split2 (%)", "HMDB51 split3 (%)",
           "HMDB51 3-split mean (%)", "HMDB51 3-split std (pp, ddof=0)",
           "UCF101 split1 all 101 classes (%)", "UCF101 split2 (%)", "UCF101 split3 (%)",
           "UCF101 3-split mean (%)", "UCF101 3-split std (pp, ddof=0)", "Training overlap note")


def split_statistics(metrics):
    """Equal weight per official split, not one pooled/deduplicated video set."""
    result = {}
    for dataset in ("hmdb51", "ucf101"):
        values = [metrics[f"{dataset}-split{i}"]["accuracy"] for i in (1, 2, 3)]
        result[dataset] = {"split_accuracies": values, "mean_accuracy": statistics.mean(values),
                           "std_accuracy": statistics.pstdev(values), "std_ddof": 0,
                           "aggregation": "arithmetic_mean_of_three_split_accuracies"}
    return result


def training_note(plan):
    audit = plan.get("training_audit", {})
    role = audit.get("model_role", "unknown")
    if role == "base":
        return "Original model; not fine-tuned on this AR set; pretraining overlap unknown"
    if role == "ar-trained":
        if audit.get("provided_train_file"):
            parts = [f"{key}={item['provided_train_overlap_selected']}" for key, item in audit["counts"].items()
                     if item.get("provided_train_overlap_selected")]
            return "AR train video overlap: " + (", ".join(parts) if parts else "0 by provided paths")
        return "Reference AR train overlap: HMDB split2=1340/1530, split3=1311/1530; 3-split mean includes seen videos"
    return "Training history unspecified; reference AR overlaps HMDB split2=1340, split3=1311"


def table_values(metrics):
    values = []
    for dataset in ("hmdb51", "ucf101", "ssv2"):
        base = 100 * metrics[dataset + "-base"]["accuracy"]
        novel = 100 * metrics[dataset + "-novel"]["accuracy"]
        hm = 2 * base * novel / (base + novel) if base + novel else 0.0
        values.extend((base, novel, hm))
    for stats in split_statistics(metrics).values():
        values.extend(100 * value for value in stats["split_accuracies"])
        values.extend((100 * stats["mean_accuracy"], 100 * stats["std_accuracy"]))
    return [f"{value:.2f}" for value in values]


@contextmanager
def file_lock(path, blocking=False):
    """The OS releases the lock even after an interrupted coordinator."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise ValueError("另一个进程正在更新此任务/表格：" + str(path)) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def atomic_text(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def table_contract(plan):
    # Model weights/name and physical GPU IDs can differ; evaluation settings,
    # data, prompts, partition strategy, and seed schedule must be identical.
    if set(plan["splits"]) != set(SPLITS) or any(plan["splits"][key].get("prompt_protocol") != PROMPT_PROTOCOL for key in SPLITS):
        raise ValueError("旧版/未知提示词结果不能写入 v6 总表；请用 v6 新任务重新评测")
    return {"num_shards": plan["num_shards"], "environment": plan["environment"], "partition": "strided-v1",
            "seed_schedule": "base_plus_shard", "splits": {
                k: {field: value for field, value in plan["splits"][k].items() if field != "model"}
                for k in SPLITS}}


def check_table_compatibility(directory, plan):
    """Reject old-protocol destinations before spending time on GPU inference."""
    contract = table_contract(plan)
    directory = Path(directory)
    path = directory / "models.json"
    if path.exists():
        with file_lock(directory / ".models.lock", blocking=True):
            table = read_json(path)
            if table.get("schema_version") != 3 or table.get("contract") != contract:
                raise ValueError("目标总表包含旧版或不同设置的结果；请指定新的 --table-dir，例如 VideoSTAR-Eval-results-v6")


def update_table(directory, plan, metrics, run_directory):
    if set(metrics) != set(SPLITS):
        raise ValueError("总表需要三个数据集 Base/Novel 和 HMDB/UCF split1/2/3 共十二个划分")
    for key in SPLITS:
        m = metrics[key]
        expected = plan["splits"][key]["official_split_size"]
        if (not m["complete"] or m["subset_only"] or m["expected"] != expected
                or m["completed"] != expected or m["inference_errors"]):
            raise ValueError("未完整完成或存在运行错误，拒绝写入正式总表：" + key)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    contract = table_contract(plan)
    entry = {"model_name": plan["model_name"], "model": plan["splits"][SPLITS[0]]["model"],
             "run_directory": str(Path(run_directory).resolve()), "metrics": metrics,
             "three_split_statistics": split_statistics(metrics),
             "training_audit": plan.get("training_audit", {}), "training_overlap_note": training_note(plan),
             "updated_at": datetime.now(timezone.utc).isoformat()}
    if not entry["model_name"].strip() or any(c in entry["model_name"] for c in "\n\r"):
        raise ValueError("模型名称不能为空或包含换行")
    with file_lock(directory / ".models.lock", blocking=True):
        path = directory / "models.json"
        table = read_json(path) if path.exists() else {"schema_version": 3, "contract": contract, "rows": []}
        if table["schema_version"] != 3 or table["contract"] != contract:
            raise ValueError("评测设置与总表已有模型不同；请保持相同设置或指定新的 --table-dir")
        matching = [i for i, row in enumerate(table["rows"]) if row["model_name"] == entry["model_name"]]
        if matching:
            index = matching[0]
            if table["rows"][index]["model"] != entry["model"]:
                raise ValueError("此模型名称已有不同权重的结果，请为新模型使用独立 --model-name")
            table["rows"][index] = entry
        else:
            table["rows"].append(entry)
        atomic_json(path, table)
        output = io.StringIO(newline="")
        writer = csv.writer(output)
        writer.writerow(COLUMNS)
        markdown = ["| " + " | ".join(COLUMNS) + " |", "| --- | " + " | ".join(["---:"] * (len(COLUMNS) - 2)) + " | --- |"]
        for row in table["rows"]:
            values = [row["model_name"]] + table_values(row["metrics"]) + [row["training_overlap_note"]]
            writer.writerow(values)
            values[0] = values[0].replace("\\", "\\\\").replace("|", "\\|")
            markdown.append("| " + " | ".join(values) + " |")
        atomic_text(directory / "models.csv", output.getvalue())
        atomic_text(directory / "models.md", "\n".join(markdown) + "\n")
    print("\n".join(markdown), flush=True)
    print("总表：" + str(directory / "models.csv"), flush=True)
    return directory / "models.csv"
