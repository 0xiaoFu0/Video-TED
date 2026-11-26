#!/usr/bin/env python3
"""Single-node data-parallel evaluation and one-row-per-model reporting."""
from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import evaluate
from audit_protocol import verify_sources
from model_table import SPLITS, check_table_compatibility, file_lock, split_statistics, update_table
from protocol import (PROMPT_PROTOCOL, ROOT, atomic_json, digest, load_dataset, load_progress,
                      partition_rows, read_json, registry, summarize)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="/path/to/workspace/models/Qwen/Qwen2.5-VL-7B-Instruct")
    p.add_argument("--model-name", required=True, help="Unique table row label, including checkpoint step")
    p.add_argument("--run-name", required=True, help="Output directory name; reuse with --resume")
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7", help="Physical GPU IDs on this node")
    p.add_argument("--batch-size", type=int, default=2, help="Videos per worker batch")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--limit", type=int, default=0, help="Subset per split BEFORE partitioning; subsets never enter the table")
    p.add_argument("--output-root", default=str(ROOT / "outputs"))
    p.add_argument("--table-dir", default=str(ROOT / "results"))
    p.add_argument("--resume", action="store_true", help="Skip completed samples and retry runtime-error suffixes")
    p.add_argument("--prepare-only", action="store_true", help="Check paths/model files and print allocation without loading GPUs")
    p.add_argument("--merge-only", action="store_true", help="Verify/merge completed workers and refresh the table on CPU")
    p.add_argument("--train-data", help="Optional current AR training JSONL, for video-overlap checks")
    p.add_argument("--allow-train-overlap", action="store_true", help="Run all requested splits, retaining training-overlap metadata")
    p.add_argument("--model-role", choices=("base", "ar-trained", "unknown"), default="unknown",
                   help="Metadata only: distinguish original model from the current AR model")
    for dataset in ("hmdb51", "ucf101", "ssv2"):
        p.add_argument("--" + dataset + "-root")
    return p


def gpu_ids(value):
    ids = [part.strip() for part in value.split(",")]
    if not ids or any(not re.fullmatch(r"[0-9]+|GPU-[A-Za-z0-9-]+", p) for p in ids) or len(set(ids)) != len(ids):
        raise ValueError("--gpus 必须是互不重复的 GPU 编号，例如 0,1,2,3,4,5,6,7")
    return ids


def worker_argv(args, rank, count, run_directory, roots):
    command = ["run", "--model", str(Path(args.model).expanduser().resolve()),
               "--run-name", f"shard_{rank:02d}", "--output-root", str(run_directory / "workers"),
               "--benchmarks", *SPLITS,
               "--batch-size", str(args.batch_size), "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--tensor-parallel-size", "1", "--seed", str(args.seed + rank),
               "--num-shards", str(count), "--shard-index", str(rank), "--limit", str(args.limit)]
    for dataset, root in roots.items():
        command.extend(["--" + dataset + "-root", str(root)])
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.train_data:
        command.extend(["--train-data", str(Path(args.train_data).expanduser().resolve())])
    if args.allow_train_overlap:
        command.append("--allow-train-overlap")
    if args.resume:
        command.extend(["--resume", "--retry-errors"])
    return command


def prepare(args):
    verify_sources()
    ids = gpu_ids(args.gpus)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) or args.run_name in {".", ".."}:
        raise ValueError("--run-name 只能使用字母、数字、下划线、横线和点")
    if not args.model_name.strip() or any(c in args.model_name for c in "\n\r"):
        raise ValueError("--model-name 不能为空或包含换行")
    if args.batch_size < 1 or not 0 < args.gpu_memory_utilization < 1 or args.limit < 0:
        raise ValueError("Invalid batch size, memory fraction or limit")
    run_directory = Path(args.output_root).expanduser().resolve() / args.run_name
    roots = {d: getattr(args, d + "_root") or os.environ.get(d.upper() + "_ROOT") or registry()["roots"][d]
             for d in ("hmdb51", "ucf101", "ssv2")}
    global_args = evaluate.parser().parse_args(worker_argv(args, 0, 1, run_directory, roots))
    prepared, reports = evaluate.selected_data(global_args)
    if any(report["missing"] for report in reports):
        raise ValueError("视频缺失，未启动 GPU：" + str(reports))
    if not args.allow_train_overlap and any(report["train_overlap"] or report["known_overlap_with_packaged_ar_train"] for report in reports):
        raise ValueError("评测视频与训练集重叠，未启动 GPU：" + str(reports))
    identity = evaluate.model_identity(args.model)
    roots = {spec["dataset"]: root for _, spec, _, _, _, root, _ in prepared}
    plan = {"format_version": 1, "model_name": args.model_name, "num_shards": len(ids),
            "limit_per_split": args.limit, "environment": evaluate.environment_versions(), "splits": {}, "workers": []}
    plan["training_audit"] = {
        "model_role": args.model_role, "allow_train_overlap": args.allow_train_overlap,
        "provided_train_file": str(Path(args.train_data).resolve()) if args.train_data else None,
        "provided_train_sha256": digest(Path(args.train_data)) if args.train_data else None,
        "counts": {r["benchmark"]: {"selected": r["selected"], "provided_train_overlap_selected": r["train_overlap"],
                   "reference_ar_overlap_full_split": r["known_overlap_with_packaged_ar_train"]} for r in reports}}
    for selected in prepared:
        plan["splits"][selected[0]] = evaluate.signature_for(global_args, selected, identity)
    commands = []
    for rank, gpu in enumerate(ids):
        argv = worker_argv(args, rank, len(ids), run_directory, roots)
        worker_args = evaluate.parser().parse_args(argv)
        signatures = {}
        for selected in prepared:
            key, spec, rows, actions, count, root, annotated = selected
            local = partition_rows(rows, rank, len(ids))
            signatures[key] = evaluate.signature_for(worker_args, (key, spec, local, actions, count, root, annotated), identity)
        plan["workers"].append(signatures)
        commands.append(([sys.executable, str(ROOT / "evaluate.py"), *argv], gpu))
        print(f"GPU {gpu}: shard {rank + 1}/{len(ids)}, " + ", ".join(
            f'{key}={signatures[key]["selected"]}' for key in SPLITS), flush=True)
    return plan, commands, run_directory


def check_gpu_memory(ids, fraction):
    command = ["nvidia-smi", "-i", ",".join(ids),
               "--query-gpu=index,uuid,memory.total,memory.free", "--format=csv,noheader,nounits"]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("无法读取 GPU 显存状态：" + str(exc)) from exc
    records = {}
    for line in output.splitlines():
        parts = [v.strip() for v in line.split(",")]
        if len(parts) != 4:
            raise ValueError("Unexpected nvidia-smi output: " + line)
        index, uuid, total, free = parts
        records[index] = records[uuid] = (uuid, float(total), float(free))
    occupied, seen = [], set()
    for gpu in ids:
        if gpu not in records:
            raise ValueError("GPU not found: " + gpu)
        uuid, total, free = records[gpu]
        if uuid in seen:
            raise ValueError("同一物理 GPU 被重复指定：" + gpu)
        seen.add(uuid)
        required = math.ceil(total * fraction) + 512  # CUDA initialization headroom, MiB
        if free < required:
            occupied.append(f"GPU {gpu}: free={free:.0f} MiB, required~{required} MiB")
    if occupied:
        raise ValueError("显存不足，未启动任何评测进程；请先释放对应 GPU：\n" + "\n".join(occupied))


def stop_workers(processes):
    # Each worker owns a new process group, including its vLLM children.
    # Never kill unrelated Python, Ray, SSH, or GPU processes.
    def send(sig):
        for process in processes:
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
    send(signal.SIGTERM)
    deadline = time.monotonic() + 5
    while any(p.poll() is None for p in processes) and time.monotonic() < deadline:
        time.sleep(0.1)
    send(signal.SIGKILL)
    for process in processes:
        process.wait()


def print_progress(run_directory, plan, done):
    completed = 0
    total = sum(s["selected"] for s in plan["splits"].values())
    for rank in range(plan["num_shards"]):
        for key in SPLITS:
            path = run_directory / "workers" / f"shard_{rank:02d}" / key / "progress.json"
            if path.exists():
                progress = read_json(path)
                completed += progress["completed"]
    print(f"单轮推理 {completed}/{total}；已结束 {done}/{plan['num_shards']} 个 worker", flush=True)


def supervise(commands, run_directory, plan):
    processes, handles = [], []
    logdir = run_directory / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise KeyboardInterrupt()
    old_handler = signal.signal(signal.SIGTERM, interrupted)
    try:
        for rank, (command, gpu) in enumerate(commands):
            handle = (logdir / f"shard_{rank:02d}.log").open("a", encoding="utf-8")
            handles.append(handle)
            handle.write(f"\n=== worker {rank}, GPU {gpu}, {time.ctime()} ===\n")
            handle.flush()
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES=gpu, VLLM_WORKER_MULTIPROC_METHOD="spawn", PYTHONUNBUFFERED="1")
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                                              stderr=subprocess.STDOUT, start_new_session=True))
        next_report = 0.0
        while True:
            statuses = [p.poll() for p in processes]
            for rank, status in enumerate(statuses):
                if status is not None and status != 0:
                    raise RuntimeError(f"worker {rank} 失败（exit={status}）；日志：{logdir / f'shard_{rank:02d}.log'}")
            done = sum(status == 0 for status in statuses)
            if time.monotonic() >= next_report or done == len(processes):
                print_progress(run_directory, plan, done)
                next_report = time.monotonic() + 15
            if done == len(processes):
                return
            time.sleep(0.5)
    finally:
        stop_workers(processes)
        for handle in handles:
            handle.close()
        signal.signal(signal.SIGTERM, old_handler)


def merge_run(run_directory, plan):
    """Validate each exact partition, then score the original ordered union."""
    merged, metrics = {}, {}
    for rank in range(plan["num_shards"]):
        environment = run_directory / "workers" / f"shard_{rank:02d}" / "environment.json"
        if read_json(environment) != plan["environment"]:
            raise ValueError("Worker environment differs: " + str(environment))
    for key in SPLITS:
        signature = plan["splits"][key]
        spec = registry()["datasets"][key]
        if digest(ROOT / spec["annotations"]) != signature["annotations_sha256"] or digest(ROOT / spec["classes"]) != signature["classes_sha256"]:
            raise ValueError("评测标注或类别文件已变化：" + key)
        rows, _, full_count = load_dataset(spec, plan["limit_per_split"])
        if signature["selected"] != len(rows) or signature["official_split_size"] != full_count:
            raise ValueError("Global sample count differs: " + key)
        predictions = {}
        for rank in range(plan["num_shards"]):
            directory = run_directory / "workers" / f"shard_{rank:02d}" / key
            if read_json(directory / "run.json") != plan["workers"][rank][key]:
                raise ValueError("Worker model/data/settings mismatch: " + str(directory))
            local = partition_rows(rows, rank, plan["num_shards"])
            results = load_progress(directory / "results.json", local)
            if len(results) != len(local):
                raise ValueError("分片未完成，拒绝生成总表：" + str(directory))
            for b in results:
                if b["path"] in predictions:
                    raise ValueError("Duplicate prediction: " + b["path"])
                predictions[b["path"]] = b
        ordered = [predictions[row["path"]] for row in rows]
        metrics[key] = summarize(ordered, len(rows), full_count)
        merged[key] = ordered
    # Every partition is verified before publishing any merged result.
    for key, results in merged.items():
        directory = run_directory / key
        atomic_json(directory / "run.json", {**plan["splits"][key],
                    "parallel": {"num_shards": plan["num_shards"], "strategy": "strided-v1", "seed_schedule": "base_plus_shard"}})
        atomic_json(directory / "results.json", results)
        atomic_json(directory / "metrics.json", metrics[key])
    atomic_json(run_directory / "summary.json", {"model_name": plan["model_name"], "metrics": metrics,
                "three_split_statistics": split_statistics(metrics), "training_audit": plan.get("training_audit", {})})
    if any(m["inference_errors"] for m in metrics.values()):
        raise ValueError("存在解码/推理错误，结果已保存，但未写入正式总表；用 --resume 重试失败部分")
    return metrics


def main(argv=None):
    args = parser().parse_args(argv)
    plan, commands, run_directory = prepare(args)
    print("提示词协议：" + PROMPT_PROTOCOL + "（单轮，无工具，无类别解释，文本不包含视频路径）", flush=True)
    print("保留官方原始标注：HMDB Base/Novel 与 split1 有 120 条标签不一致，详见 PROTOCOL.md / ALIGNMENT.json。", flush=True)
    print("已选择 HMDB/UCF 三个 split；HMDB split2/3 与参考 AR 训练集重叠 1340/1311 条，结果将记录此信息。", flush=True)
    if not args.limit:
        check_table_compatibility(args.table_dir, plan)
    if args.prepare_only:
        print("数据/模型路径检查通过；尚未启动 GPU。", flush=True)
        return 0
    with file_lock(Path(args.output_root).expanduser().resolve() / ".locks" / (args.run_name + ".lock")):
        path = run_directory / "launch.json"
        if path.exists():
            if not (args.resume or args.merge_only):
                raise ValueError("任务已存在，请使用 --resume 或新的 --run-name")
            if read_json(path) != plan:
                raise ValueError("任务的模型/数据/参数已变化，不能恢复；请使用新的 --run-name")
        elif args.merge_only:
            raise ValueError("找不到 launch.json，无法合并：" + str(run_directory))
        elif run_directory.exists():
            raise ValueError("任务目录已存在但没有 launch.json；请选择新的 --run-name")
        if not args.merge_only:
            # Also covers a failed initialization where no result was written.
            check_gpu_memory(gpu_ids(args.gpus), args.gpu_memory_utilization)
            if not path.exists():
                atomic_json(path, plan)
            print("启动独立单卡进程；日志目录：" + str(run_directory / "logs"), flush=True)
            supervise(commands, run_directory, plan)
        metrics = merge_run(run_directory, plan)
        if args.limit:
            print("小样本评测完成；未加入全量模型总表。汇总：" + str(run_directory / "summary.json"), flush=True)
        else:
            update_table(args.table_dir, plan, metrics, run_directory)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("评测已中断，已停止本次 worker；可用相同命令加 --resume 继续。", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        raise SystemExit(1)
