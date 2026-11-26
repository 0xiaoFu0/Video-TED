#!/usr/bin/env python3
"""Standalone Video-STAR evaluation; imports GPU dependencies only for `run`."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path

from protocol import (EVAL_SPLITS, PROMPT_PROTOCOL, ROOT, atomic_json, digest, expand_benchmarks, load_dataset,
                      load_progress, manifest_rows, messages, multimodal_inputs, official,
                      partition_rows, read_json, registry, resolve_video_root, summarize)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List official splits and convenient benchmark groups")
    for command in ("check", "run"):
        q = sub.add_parser(command)
        q.add_argument("--benchmarks", nargs="+", default=list(EVAL_SPLITS))
        q.add_argument("--limit", type=int, default=0, help="First N videos per split; 0 means full split")
        for dataset in ("hmdb51", "ucf101", "ssv2", "k400", "k600"):
            q.add_argument("--" + dataset + "-root", default=None)
        q.add_argument("--annotated-root", help="Legacy metadata only; v6 never loads or exposes annotated media")
        q.add_argument("--train-data", help="Optional AR JSONL to audit video-path overlap")
        q.add_argument("--allow-train-overlap", action="store_true", help="Explicitly permit a diagnostic evaluation on overlapping videos")
        q.add_argument("--report", help="Write media/annotation check report")
    q = sub.choices["run"]
    q.add_argument("--model", required=True, help="Local Hugging Face weights directory, not an FSDP actor directory")
    q.add_argument("--run-name", required=True, help="Distinct name per model/checkpoint")
    q.add_argument("--output-root", default=str(ROOT / "outputs"))
    q.add_argument("--batch-size", type=int, default=50)
    q.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    q.add_argument("--tensor-parallel-size", type=int, default=1)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--resume", action="store_true")
    q.add_argument("--enforce-eager", action="store_true")
    q.add_argument("--num-shards", type=int, default=1)
    q.add_argument("--shard-index", type=int, default=0)
    q.add_argument("--retry-errors", action="store_true", help="Retry from the first cached runtime error")
    q = sub.add_parser("score")
    q.add_argument("results", help="A split's results.json, next to run.json")
    return p


def selected_data(args):
    if args.limit < 0:
        raise ValueError("--limit must be nonnegative")
    specs = expand_benchmarks(args.benchmarks)
    reg = registry()
    train_paths = set()
    if args.train_data:
        for line in Path(args.train_data).read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                for video in item.get("videos", []):
                    path = Path(video)
                    train_paths.add("/".join(path.parts[-2:]))
    prepared, reports = [], []
    for key, spec in specs:
        rows, actions, full_count = load_dataset(spec, args.limit)
        all_rows = rows
        rows = partition_rows(rows, getattr(args, "shard_index", 0), getattr(args, "num_shards", 1))
        base = getattr(args, spec["dataset"] + "_root") or os.environ.get(spec["root_env"]) or reg["roots"][spec["dataset"]]
        root, missing = resolve_video_root(base, rows or all_rows)
        if not rows:
            missing = []
        # AR's training JSONL uses absolute class/video paths. Other datasets
        # must not collide with HMDB's relative names in the audit.
        overlap = [r["path"] for r in rows if spec["dataset"] == "hmdb51" and r["path"] in train_paths]
        annotated = Path(args.annotated_root) if args.annotated_root else root.parent / (root.name + "_pose")
        report = {"benchmark": key, "annotations": spec["annotations"], "classes": len(actions),
                  "official_split_size": full_count, "selected": len(rows), "video_root": str(root),
                  "found": len(rows) - len(missing), "missing": len(missing), "missing_examples": missing[:10],
                  "train_overlap": len(overlap) if args.train_data else None,
                  "known_overlap_with_packaged_ar_train": spec.get("ar_train_overlap"),
                  "annotated_prompt_root": str(annotated)}
        reports.append(report)
        prepared.append((key, spec, rows, actions, full_count, root, annotated))
    return prepared, reports


def model_identity(path):
    path = Path(path).expanduser().resolve()
    if not (path / "config.json").is_file():
        raise ValueError("模型目录缺少 config.json；FSDP checkpoint 请先用 export_model.sh 导出。")
    weights = sorted(list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin")))
    if not weights:
        raise ValueError("模型目录没有完整权重；checkpoint 下的 huggingface 目录在合并前只有配置文件。")
    return {"path": str(path), "config_sha256": digest(path / "config.json"),
            "weights": [{"name": f.name, "size": f.stat().st_size, "mtime_ns": f.stat().st_mtime_ns} for f in weights]}


def environment_versions():
    versions = {}
    for package in ("torch", "torchvision", "transformers", "vllm", "decord", "qwen-vl-utils"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    return versions


class Backend:
    def __init__(self, args):
        # The upstream repo includes modified video sampling defaults. Isolate
        # those exact files from the training environment's installed package.
        utils = ROOT / "upstream/src/qwen-vl-utils/src"
        sys.path.insert(0, str(utils))
        from qwen_vl_utils import process_vision_info
        from qwen_vl_utils import vision_process
        from qwen3_video import check_runtime
        from transformers import AutoProcessor, AutoTokenizer
        from vllm import LLM, SamplingParams
        self.qwen3 = read_json(Path(args.model) / "config.json").get("model_type") == "qwen3_vl"
        if self.qwen3:
            check_runtime()
        self.legacy_vision = vision_process
        self.vision = process_vision_info
        self.processor = AutoProcessor.from_pretrained(args.model)
        self.processor.tokenizer = AutoTokenizer.from_pretrained(args.model)
        self.processor.tokenizer.padding_side = "left"
        self.sampling = SamplingParams(temperature=0.8, top_p=0.001, max_tokens=4096,
                                       stop_token_ids=[])
        self.llm = LLM(model=args.model, tensor_parallel_size=args.tensor_parallel_size,
                       max_model_len=16384, gpu_memory_utilization=args.gpu_memory_utilization,
                       limit_mm_per_prompt={"image": 1, "video": 1}, seed=args.seed,
                       enforce_eager=args.enforce_eager)

    def generate(self, batch):
        if self.qwen3:
            from qwen3_video import multimodal_inputs as qwen3_inputs
            inputs = qwen3_inputs(batch, self.processor, self.legacy_vision)
        else:
            inputs = multimodal_inputs(batch, self.processor, self.vision)
        outputs = self.llm.generate(inputs, sampling_params=self.sampling)
        if len(outputs) != len(batch):
            raise RuntimeError("Generation returned a different number of samples")
        return [(o.outputs[0].text, o.outputs[0].finish_reason) for o in outputs]


def generation(backend, batch):
    try:
        return [(text, reason, None) for text, reason in backend.generate(batch)]
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        print(f"inference batch failed: {error}", file=sys.stderr, flush=True)
        return [("<answer>error</answer>", "error", error) for _ in batch]


def run_split(backend, args, selected, output, signature):
    key, spec, rows, actions, full_count, root, annotated = selected
    api = official(spec["entry"], actions, root, annotated)
    result_path = output / "results.json"
    results = load_progress(result_path, rows)
    if getattr(args, "retry_errors", False):
        bad = next((i for i, r in enumerate(results) if r.get("inference_error")), len(results))
        if bad < len(results):
            results = results[:bad]
            atomic_json(result_path, results)
    def progress():
        atomic_json(output / "progress.json", {"expected": len(rows), "completed": len(results),
                    "inference_passes": 1, "prompt_protocol": PROMPT_PROTOCOL})
    progress()
    for i in range(len(results), len(rows), args.batch_size):
        subset = rows[i:i + args.batch_size]
        batch = [messages(row, api) for row in subset]
        for row, (text, reason, error) in zip(subset, generation(backend, batch)):
            record = dict(row)
            record.update(inference_output=text, predict=api["extract_answer"](text).replace("\n", "").strip(),
                          inference_error=error, finish_reason=reason)
            results.append(record)
        atomic_json(result_path, results)
        progress()
        atomic_json(output / "metrics.json", summarize(results, len(rows), full_count))
        print(f"{key}: inference {len(results)}/{len(rows)}", flush=True)
    # Publish [] as well: an eight-way partition can have empty workers.
    if not result_path.exists():
        atomic_json(result_path, results)
    metrics = summarize(results, len(rows), full_count)
    atomic_json(output / "metrics.json", metrics)
    print(json.dumps({"benchmark": key, **metrics}, ensure_ascii=False), flush=True)
    return metrics


def signature_for(args, selected, identity):
    key, spec, rows, _, full_count, root, annotated = selected
    signature = {"benchmark": key, "upstream_commit": read_json(ROOT / "SOURCE_MANIFEST.json")["commit"],
            "prompt_protocol": PROMPT_PROTOCOL, "inference_passes": 1, "tools": False, "class_explanations": False,
            "allow_train_overlap": bool(getattr(args, "allow_train_overlap", False)),
            "annotations": spec["annotations"], "annotations_sha256": digest(ROOT / spec["annotations"]),
            "classes_sha256": digest(ROOT / spec["classes"]), "model": identity,
            "video_root": str(root), "annotated_prompt_root": str(annotated),
            "selected": len(rows), "official_split_size": full_count,
            "batch_size": args.batch_size, "temperature": 0.8, "top_p": 0.001, "max_tokens": 4096,
            "max_model_len": 16384, "tensor_parallel_size": args.tensor_parallel_size, "seed": args.seed,
            "enforce_eager": args.enforce_eager, "gpu_memory_utilization": args.gpu_memory_utilization,
            "video_pixel_budget_env": os.environ.get("VIDEO_MAX_PIXELS"),
            "video_reader_env": os.environ.get("FORCE_QWENVL_VIDEO_READER"),
            "source": {str(p.relative_to(ROOT)): digest(p) for p in [ROOT / "evaluate.py", ROOT / "protocol.py",
                ROOT / "upstream/AScripts" / spec["entry"] / "4subtol.py",
                ROOT / "upstream/AScripts/cresult/1.py",
                ROOT / "upstream/src/qwen-vl-utils/src/qwen_vl_utils/vision_process.py"]}}
    if read_json(Path(identity["path"]) / "config.json").get("model_type") == "qwen3_vl":
        from qwen3_video import input_profile
        signature["model_input"] = input_profile()
        signature["source"]["qwen3_video.py"] = digest(ROOT / "qwen3_video.py")
        signature["processor_files"] = {p.name: digest(p) for p in Path(identity["path"]).iterdir()
            if p.is_file() and (p.name in {"preprocessor_config.json", "processor_config.json",
                "video_preprocessor_config.json", "tokenizer_config.json", "tokenizer.json",
                "special_tokens_map.json", "vocab.json", "merges.txt", "chat_template.json",
                "generation_config.json"} or p.suffix == ".jinja")}
    if getattr(args, "num_shards", 1) != 1:
        signature["sharding"] = {"strategy": "strided-v1", "count": args.num_shards,
                                  "index": args.shard_index, "limit_per_split": args.limit}
    return signature


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "list":
        for key, spec in registry()["datasets"].items():
            print(f"{key:20s} {spec['rows']:6d} videos  {spec['annotations']}")
        print("Groups:", json.dumps(registry()["groups"], ensure_ascii=False))
        return 0
    if args.command == "score":
        path = Path(args.results)
        manifest = read_json(path.parent / "run.json")
        spec = registry()["datasets"][manifest["benchmark"]]
        if digest(ROOT / spec["annotations"]) != manifest["annotations_sha256"]:
            raise ValueError("Annotation hash differs from evaluation run")
        rows, _, full_count = manifest_rows(spec, manifest)
        results = load_progress(path, rows)
        metrics = summarize(results, len(rows), full_count)
        print(json.dumps(metrics, indent=2, ensure_ascii=False))
        return int(not metrics["complete"] or metrics["inference_errors"] > 0)
    prepared, reports = selected_data(args)
    print(json.dumps(reports, ensure_ascii=False, indent=2), flush=True)
    if args.report:
        atomic_json(args.report, reports)
    if any(r["missing"] for r in reports):
        print("视频未全部找到。请解压数据，或用 --ucf101-root / --ssv2-root 等指定视频根目录。未启动 GPU。", file=sys.stderr)
        return 2
    if not args.allow_train_overlap and any(r["train_overlap"] or r["known_overlap_with_packaged_ar_train"] for r in reports):
        print("所选 HMDB 划分与当前 AR 训练集存在重叠。请优先选择 hmdb51（split 1）。", file=sys.stderr)
        return 2
    if args.command == "check":
        print("视频路径检查通过；尚未验证解码或 GPU 推理。")
        return 0
    if args.batch_size <= 0 or args.tensor_parallel_size <= 0 or not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("Invalid batch size / tensor parallel size / GPU memory fraction")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.run_name) or args.run_name in {".", ".."}:
        raise ValueError("Use a simple run name without path separators")
    identity = model_identity(args.model)
    args.model = identity["path"]
    runs = []
    # Validate every requested output before allocating the model or writing files.
    for selected in prepared:
        key = selected[0]
        output = Path(args.output_root) / args.run_name / key
        signature = signature_for(args, selected, identity)
        manifest_path = output / "run.json"
        if output.exists():
            if not args.resume or not manifest_path.is_file():
                raise ValueError("Output exists; use --resume or a new --run-name: " + str(output))
            if read_json(manifest_path) != signature:
                raise ValueError("Resume model, source, data or settings differ: " + str(output))
            load_progress(output / "results.json", selected[2])
        runs.append((selected, output, signature))
    for _, output, signature in runs:
        atomic_json(output / "run.json", signature)
    atomic_json(Path(args.output_root) / args.run_name / "environment.json", environment_versions())
    needs_generation = any(
        len(load_progress(output / "results.json", selected[2])) < len(selected[2])
        or (args.retry_errors and any(r.get("inference_error")
            for r in load_progress(output / "results.json", selected[2])))
        for selected, output, _ in runs)
    backend = Backend(args) if needs_generation else None
    failed = False
    for selected, output, signature in runs:
        metrics = run_split(backend, args, selected, output, signature)
        failed |= bool(metrics["inference_errors"] or not metrics["complete"])
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(2)
