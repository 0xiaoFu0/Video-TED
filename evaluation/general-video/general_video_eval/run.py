"""Local-data, multi-GPU evaluation with durable predictions and bounded waits."""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re
import signal
import sys
import time
import traceback

from . import __version__
from .loaders import load_dataset
from .schema import Sample, media_exists
from .protocol import DEFAULT_FRAMES, PROFILE_NAMES, apply_eval_profile, build_prompt


EXPECTED_COUNTS = {"mvbench": 4000, "videomme": 2700, "tempcompass": 1580, "seedbench": 3757}
TEMPCOMPASS_COUNTS = {"action": 338, "direction": 335, "speed": 317, "order": 302, "attribute_change": 288}
VIDEOMME_TASKS = {"Temporal Perception", "Spatial Perception", "Attribute Perception", "Action Recognition",
                 "Object Recognition", "OCR Problems", "Counting Problem", "Temporal Reasoning",
                 "Spatial Reasoning", "Action Reasoning", "Object Reasoning", "Information Synopsis"}
DATA_ENV = {"mvbench": "MVBENCH_DATA_DIR", "videomme": "VIDEOMME_DATA_DIR", "tempcompass": "TEMPCOMPASS_DATA_DIR", "seedbench": "SEEDBENCH_DATA_DIR"}
DEFAULT_DATA_DIRS = {"mvbench": "/path/to/workspace/datasets/MVBench",
                     "videomme": "/gemini-3/space/zwj/LMUData/VideoMME",
                     "tempcompass": "/gemini-3/space/zwj/LMUData/TempCompass",
                     "seedbench": "/gemini-3/space/zwj/LMUData/SEED-Bench"}


def env_bool(name, default=False):
    value = os.environ.get(name, str(int(default))).lower()
    if value not in {"0", "1", "true", "false"}:
        raise ValueError(f"{name} must be 0 or 1")
    return value in {"1", "true"}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-profile", choices=PROFILE_NAMES, default=os.getenv("EVAL_PROFILE", "generic"))
    p.add_argument("--model-path", default=os.getenv("MODEL_PATH", ""))
    p.add_argument("--benchmarks", default=os.getenv("BENCHMARKS", "mvbench,videomme,tempcompass"))
    p.add_argument("--exp-name", default=os.getenv("EXP_NAME", "general_video"))
    p.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "outputs"))
    p.add_argument("--backend", choices=["vllm", "mock"], default=os.getenv("BACKEND", "vllm"))
    p.add_argument("--limit", type=int, default=int(os.getenv("LIMIT", "0")), help="Maximum questions per benchmark; 0 evaluates all")
    p.add_argument("--preflight", action="store_true", default=env_bool("PREFLIGHT"), help="Load all selected annotations and check media without loading a model")
    p.add_argument("--verify-frame-images", action="store_true", default=env_bool("VERIFY_FRAME_IMAGES"),
                   help="Fully decode every selected SEED-Bench image during preflight, before GPU initialization")
    p.add_argument("--score-only", action="store_true", default=env_bool("SCORE_ONLY"))
    p.add_argument("--no-resume", action="store_true", default=not env_bool("RESUME", True))
    p.add_argument("--allow-nonstandard-counts", action="store_true", default=env_bool("ALLOW_NONSTANDARD_COUNTS"))
    p.add_argument("--num-gpus", type=int, default=int(os.getenv("NUM_GPUS", "0")))
    p.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("TP_SIZE", "1")))
    p.add_argument("--worker-timeout", type=float, default=float(os.getenv("WORKER_TIMEOUT_S", "3600")))
    p.add_argument("--think-mode", type=int, choices=[0, 1], default=int(os.environ["THINK_MODE"]) if "THINK_MODE" in os.environ else None)
    p.add_argument("--n", "--N", type=int, default=int(os.environ["N"]) if "N" in os.environ else None)
    p.add_argument("--temperature", type=float, default=float(os.environ["TEMPERATURE"]) if "TEMPERATURE" in os.environ else None)
    p.add_argument("--top-p", type=float, default=float(os.environ["TOP_P"]) if "TOP_P" in os.environ else None)
    p.add_argument("--repetition-penalty", type=float, default=float(os.environ["REPETITION_PENALTY"]) if "REPETITION_PENALTY" in os.environ else None)
    p.add_argument("--max-new-tokens", type=int, default=int(os.environ["MAX_NEW_TOKENS"]) if "MAX_NEW_TOKENS" in os.environ else None)
    p.add_argument("--max-model-len", type=int, default=int(os.environ["MAX_MODEL_LEN"]) if "MAX_MODEL_LEN" in os.environ else None)
    p.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8")))
    p.add_argument("--fps", type=float, default=float(os.environ["FPS"]) if "FPS" in os.environ else None)
    p.add_argument("--max-pixels", type=int, default=int(os.environ["MAX_PIXELS"]) if "MAX_PIXELS" in os.environ else None)
    p.add_argument("--min-pixels", type=int, default=int(os.environ["MIN_PIXELS"]) if "MIN_PIXELS" in os.environ else None)
    p.add_argument("--min-frames", type=int, default=int(os.environ["MIN_FRAMES"]) if "MIN_FRAMES" in os.environ else None)
    p.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    p.add_argument("--seed", type=int, default=int(os.getenv("SEED", "42")))
    p.add_argument("--decoder", choices=["decord", "opencv"], default=os.getenv("DECODER", "decord"))
    p.add_argument("--trust-remote-code", action="store_true", default=env_bool("TRUST_REMOTE_CODE"))
    for name in EXPECTED_COUNTS:
        default_dir = (str(Path(os.environ["DATA_ROOT"]) / {"mvbench": "MVBench", "videomme": "VideoMME", "tempcompass": "TempCompass", "seedbench": "SEED-Bench"}[name])
                       if "DATA_ROOT" in os.environ else DEFAULT_DATA_DIRS[name])
        p.add_argument(f"--{name}-data-dir", default=os.getenv(DATA_ENV[name], default_dir))
        p.add_argument(f"--{name}-anno-file", default=os.getenv(name.upper() + "_ANNO_FILE"))
        frames_override = os.getenv(name.upper() + "_MAX_FRAMES", os.getenv("MAX_FRAMES"))
        p.add_argument(f"--{name}-max-frames", type=int, default=int(frames_override) if frames_override is not None else None)
    return p


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def experiment_lock(run_dir):
    """Serialize writers and scoring for one experiment on Linux/macOS."""
    import fcntl
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".eval.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Experiment is already running or being scored: {run_dir}. Use a different EXP_NAME.") from exc
        try:
            stream.seek(0)
            stream.truncate()
            stream.write(str(os.getpid()) + "\n")
            stream.flush()
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def model_identity(model_path):
    root = Path(model_path).expanduser()
    if not root.is_dir():
        return {"model_id": model_path}
    identity = {"path": str(root.resolve()), "files": {}}
    for name in ("config.json", "tokenizer_config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"):
        path = root / name
        if path.is_file():
            identity["files"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    weights = list(root.glob("*.safetensors")) + list(root.glob("pytorch_model*.bin"))
    identity["weights"] = {p.name: [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(weights)}
    return identity


def source_identity():
    root = Path(__file__).parent
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def make_config(args, benchmarks):
    profile_config = apply_eval_profile(args, benchmarks)
    if args.temperature is None:
        args.temperature = 0.0 if args.n == 1 else 0.7
    if args.max_new_tokens is None:
        args.max_new_tokens = 2048 if args.think_mode else 128
    if args.n < 1 or args.limit < 0 or args.worker_timeout <= 0:
        raise ValueError("N must be positive; LIMIT nonnegative; WORKER_TIMEOUT_S positive")
    if args.n > 1 and args.temperature == 0:
        raise ValueError("N>1 requires TEMPERATURE>0; use N=1 for greedy evaluation")
    if args.tensor_parallel_size < 1 or args.max_new_tokens < 1 or args.max_model_len <= args.max_new_tokens:
        raise ValueError("Invalid TP_SIZE / MAX_NEW_TOKENS / MAX_MODEL_LEN")
    if not 0 < args.top_p <= 1 or not 0 < args.gpu_memory_utilization < 1 or args.temperature < 0:
        raise ValueError("Invalid TOP_P / GPU_MEMORY_UTILIZATION / TEMPERATURE")
    if args.fps <= 0 or args.min_frames < 2 or args.min_frames % 2 or args.min_pixels < 1 or args.max_pixels < args.min_pixels:
        raise ValueError("Invalid FPS / MIN_FRAMES / pixel budgets")
    frame_limits = {b: getattr(args, b + "_max_frames") for b in benchmarks}
    if any(n < args.min_frames or n % 2 for n in frame_limits.values()):
        raise ValueError("MIN_FRAMES and benchmark MAX_FRAMES must be even, with MAX_FRAMES >= MIN_FRAMES >= 2")
    vote = os.getenv("VOTE", "majority_voting")
    if vote not in {"majority", "majority_voting"}:
        raise ValueError("VOTE must be majority_voting; confidence weighting is not supported")
    versions = {}
    for package in ("torch", "vllm", "transformers", "numpy", "Pillow", "pyarrow", "decord", "opencv-python-headless", "opencv-python"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "model_path": args.model_path, "model_identity": model_identity(args.model_path),
        "backend": args.backend, "think_mode": bool(args.think_mode), "n": args.n,
        **profile_config,
        "temperature": args.temperature, "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
        "max_new_tokens": args.max_new_tokens, "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization, "tensor_parallel_size": args.tensor_parallel_size,
        "seed": args.seed, "dtype": args.dtype, "trust_remote_code": args.trust_remote_code,
        "fps": args.fps, "min_frames": args.min_frames, "max_frames": max(frame_limits.values()),
        "max_frames_by_benchmark": frame_limits, "max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
        "decoder": args.decoder, "benchmarks": benchmarks, "limit": args.limit,
        "python_version": sys.version.split()[0], "dependencies": versions,
        "protocol": f"video_frames_only_no_subtitles; {profile_config['prompt_style']}; deterministic_option_parser; majority_vote",
        "package_version": __version__, "source_sha256": source_identity(),
    }


def prepare_samples(args, benchmarks):
    from collections import Counter
    from .loaders.mvbench import TASKS
    samples, audit = [], {}
    for benchmark in benchmarks:
        root = Path(getattr(args, benchmark + "_data_dir")).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"{DATA_ENV[benchmark]} does not exist: {root}")
        full = load_dataset(benchmark, str(root), getattr(args, benchmark + "_anno_file"))
        if not full:
            raise ValueError(f"{benchmark}: annotation contains no questions")
        official_count = len(full) == EXPECTED_COUNTS[benchmark]
        if benchmark == "mvbench":
            groups = dict(Counter(sample.metadata.get("task") for sample in full))
            official_structure = groups == {task: 200 for task in TASKS}
        elif benchmark == "tempcompass":
            groups = dict(Counter(sample.metadata.get("dim") for sample in full))
            official_structure = groups == TEMPCOMPASS_COUNTS
        elif benchmark == "seedbench":
            from .loaders.seedbench import TASK_COUNTS
            groups = dict(Counter(sample.metadata.get("task") for sample in full))
            official_structure = groups == TASK_COUNTS
        else:
            groups = dict(Counter(sample.metadata.get("duration") for sample in full))
            official_structure = groups == {duration: 900 for duration in ("short", "medium", "long")} and {
                sample.metadata.get("task_type") for sample in full} == VIDEOMME_TASKS
        if not official_count and not args.allow_nonstandard_counts:
            raise ValueError(f"{benchmark}: loaded {len(full)} questions, expected {EXPECTED_COUNTS[benchmark]}. "
                             "Check annotation path/schema. Intentional custom datasets require ALLOW_NONSTANDARD_COUNTS=1.")
        if not official_structure and not args.allow_nonstandard_counts:
            raise ValueError(f"{benchmark}: annotation task/duration coverage differs from the official benchmark: {groups}. "
                             "Check annotations. Intentional custom datasets require ALLOW_NONSTANDARD_COUNTS=1.")
        selected = full[:args.limit] if args.limit else full
        if benchmark == "seedbench" and args.limit:
            # The official JSON is grouped by task. A tiny smoke should exercise
            # all three layouts instead of reading only Action Recognition.
            heads = [next(s for s in full if s.metadata["task"] == task)
                     for task in groups]
            head_ids = {s.uid for s in heads}
            selected = (heads + [s for s in full if s.uid not in head_ids])[:args.limit]
        if args.eval_profile in {"videor1_64", "simple_32", "simple_64"}:
            selected = [replace(sample, metadata={**sample.metadata,
                        "sampling": "uniform_fps_floor", "eval_profile": args.eval_profile})
                        for sample in selected]
        samples.extend(selected)
        audit[benchmark] = {"annotation_questions": len(full), "selected_questions": len(selected),
                            "official_expected_questions": EXPECTED_COUNTS[benchmark],
                            "scope": "full" if official_count and official_structure and not args.limit else "partial_or_nonstandard",
                            "annotation_groups": groups, "data_dir": str(root.resolve())}
    ids = [sample.uid for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs; loaders must namespace IDs with the benchmark name")
    missing = {}
    seen = set()
    for sample in samples:
        key = (sample.media_type, sample.media_path)
        if key not in seen:
            seen.add(key)
            if not media_exists(sample):
                missing.setdefault(sample.benchmark, []).append(sample.media_path)
    if missing:
        lines = [f"{b}: {len(paths)} missing media paths; examples:\n  " + "\n  ".join(paths[:12]) for b, paths in missing.items()]
        raise FileNotFoundError("Media preflight failed before GPU initialization:\n" + "\n".join(lines) +
                                "\nExtract dataset video archives first; missing items are never silently omitted from scores.")
    seed_samples = [sample for sample in samples if sample.benchmark == "seedbench"]
    if seed_samples:
        from .loaders.seedbench import check_frame_coverage
        check_frame_coverage(seed_samples, decode=args.verify_frame_images)
    return samples, audit


def read_predictions(run_dir, repair_tail=False):
    records = []
    for path in sorted((Path(run_dir) / "predictions").glob("worker-*.jsonl")):
        raw = path.read_bytes()
        lines = raw.splitlines(keepends=True)
        offset = 0
        for index, line in enumerate(lines):
            if not line.strip():
                offset += len(line)
                continue
            try:
                record = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                    with path.open("r+b") as stream:
                        stream.truncate(offset)
                    print(f"Recovered interrupted final JSONL line: {path}", flush=True)
                    break
                raise ValueError(f"Corrupt prediction JSONL: {path}, line {index + 1}")
            # Ensure append always starts on a fresh line, including hand-produced JSONL files.
            if repair_tail and index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("ab") as stream:
                    stream.write(b"\n")
            records.append(record)
            offset += len(line)
    return records


def worker_main(worker_id, devices, config, shard, run_dir, events):
    if hasattr(os, "setsid"):
        os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(devices)
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    log = Path(run_dir) / "logs" / f"worker-{worker_id}.log"
    with log.open("a", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        events.put({"event": "started", "worker": worker_id})
        current = None
        try:
            from .backend import MockBackend, VLLMBackend
            from .scoring import extract_answer, majority_vote
            backend = MockBackend(config) if config["backend"] == "mock" else VLLMBackend(config)
            events.put({"event": "ready", "worker": worker_id})
            with (Path(run_dir) / "predictions" / f"worker-{worker_id}.jsonl").open("a", encoding="utf-8", buffering=1) as output:
                for sample in shard:
                    current = sample
                    started = time.monotonic()
                    prompt = build_prompt(sample, config["think_mode"], config.get("prompt_style", "generic"))
                    result = backend.generate(sample, prompt, config["max_frames_by_benchmark"][sample.benchmark])
                    responses = result["responses"]
                    if len(responses) != config["n"] or any(not isinstance(x, str) for x in responses):
                        raise ValueError(f"{sample.uid}: backend returned invalid completion count/type")
                    parsed = [extract_answer(response, sample.choices) for response in responses]
                    prediction = majority_vote(parsed)
                    record = {"uid": sample.uid, "benchmark": sample.benchmark, "prediction": prediction,
                              "responses": responses, "parsed_answers": parsed, "finish_reasons": result.get("finish_reasons", []),
                              "video_info": result.get("video_info", {}), "elapsed_s": round(time.monotonic() - started, 4),
                              "worker": worker_id}
                    output.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                    output.flush()
                    events.put({"event": "sample", "worker": worker_id, "uid": sample.uid})
            if hasattr(backend, "close"):
                backend.close()
            events.put({"event": "done", "worker": worker_id})
        except BaseException as exc:
            error = {"event": "error", "worker": worker_id, "uid": current.uid if current else None,
                     "benchmark": current.benchmark if current else None, "error": f"{type(exc).__name__}: {exc}",
                     "traceback": traceback.format_exc(), "log": str(log)}
            traceback.print_exc()
            events.put(error)
            # The full exception is already saved in the worker log and event.
            # Exit nonzero without multiprocessing printing to a closed log.
            raise SystemExit(1)


def stop_workers(processes):
    def stop_group(process, sig):
        # A worker can exit before its vLLM descendants. Its process group is
        # still ours even when its leader no longer exists.
        if hasattr(os, "killpg"):
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        if process.is_alive():
            try:
                process.terminate() if sig == signal.SIGTERM else process.kill()
            except ProcessLookupError:
                pass
    for process in processes:
        stop_group(process, signal.SIGTERM)
    for process in processes:
        process.join(timeout=3)
        stop_group(process, signal.SIGKILL)
        process.join(timeout=3)


def run_workers(args, config, pending, run_dir):
    if not pending:
        return []
    devices = [x.strip() for x in os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",") if x.strip()]
    if len(set(devices)) != len(devices) or not devices or devices == ["-1"]:
        raise ValueError("CUDA_VISIBLE_DEVICES must list unique usable GPU identifiers")
    num_gpus = args.num_gpus or len(devices)
    if num_gpus != len(devices):
        raise ValueError("NUM_GPUS must equal the number of CUDA_VISIBLE_DEVICES entries")
    tp = args.tensor_parallel_size
    if num_gpus % tp:
        raise ValueError("NUM_GPUS must be divisible by TP_SIZE")
    groups = [devices[i:i + tp] for i in range(0, num_gpus, tp)]
    groups = groups[:min(len(groups), len(pending))]
    ctx = mp.get_context("spawn")
    events = ctx.Queue()
    processes, errors = [], []
    last_event = {i: time.monotonic() for i in range(len(groups))}
    done = set()
    progress = 0
    last_print = 0.0
    # Scheduler cancellation must unwind the parent and stop its independent
    # worker process groups (including vLLM's own child processes).
    def on_sigterm(_signum, _frame):
        raise KeyboardInterrupt
    previous_sigterm = signal.signal(signal.SIGTERM, on_sigterm)
    try:
        for i, group in enumerate(groups):
            process = ctx.Process(target=worker_main, args=(i, group, config, pending[i::len(groups)], str(run_dir), events), name=f"eval-worker-{i}")
            process.start()
            processes.append(process)
        while len(done) < len(processes):
            try:
                message = events.get(timeout=1)
            except queue.Empty:
                message = None
            now = time.monotonic()
            if message:
                i = message["worker"]
                last_event[i] = now
                if message["event"] == "sample":
                    progress += 1
                elif message["event"] == "done":
                    done.add(i)
                elif message["event"] == "error":
                    errors.append(message)
                    break
                elif message["event"] == "ready":
                    print(f"Worker {i} model ready on GPU(s) {','.join(groups[i])}", flush=True)
            for i, process in enumerate(processes):
                if process.exitcode is not None and process.exitcode != 0:
                    errors.append({"worker": i, "error": f"Worker exited with code {process.exitcode}", "log": str(run_dir / 'logs' / f'worker-{i}.log')})
                    break
                if i not in done and now - last_event[i] > args.worker_timeout:
                    errors.append({"worker": i, "error": f"No worker progress for {args.worker_timeout}s; see worker log", "log": str(run_dir / 'logs' / f'worker-{i}.log')})
                    break
            if errors:
                break
            if now - last_print >= 5:
                print(f"New predictions: {progress}/{len(pending)}; workers finished: {len(done)}/{len(processes)}", flush=True)
                last_print = now
            # A successful process with no completion event must not leave the parent waiting forever.
            if not message and all(p.exitcode is not None for p in processes) and len(done) < len(processes):
                errors.append({"error": "Workers exited before all completion events were received"})
                break
        if not errors:
            for i, process in enumerate(processes):
                process.join(timeout=30)
                if process.is_alive() or process.exitcode != 0:
                    errors.append({"worker": i, "error": "Worker shutdown failed or timed out", "exitcode": process.exitcode})
    except KeyboardInterrupt:
        errors.append({"error": "Evaluation interrupted; completed predictions are resumable"})
    finally:
        stop_workers(processes)
        events.close()
        signal.signal(signal.SIGTERM, previous_sigterm)
    return errors


def save_report(run_dir, samples, records, manifest, errors):
    from .scoring import summarize
    report = summarize(samples, records, errors=errors)
    report["experiment"] = manifest["exp_name"]
    report["backend"] = manifest["config"]["backend"]
    report["eval_profile"] = manifest["config"].get("eval_profile", "generic")
    report["is_mock"] = report["backend"] == "mock"
    report["scope"] = manifest["datasets"]
    report["updated_at"] = utc_now()
    atomic_json(run_dir / "summary.json", report)
    columns = ["benchmark", "scope", "n_total", "n_done", "overall", "avg", "fa", "action_recognition", "invalid_answers", "complete"]
    with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        import csv
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for name, stats in report["benchmarks"].items():
            writer.writerow({"benchmark": name, "scope": manifest["datasets"][name]["scope"], **{k: stats.get(k) for k in columns[2:]}})
    config = manifest["config"]
    lines = ["MOCK OUTPUT — pipeline verification only" if report["is_mock"] else "Video benchmark results", "",
             f"Profile: {report['eval_profile']}; prompt: {config.get('prompt_style', 'generic')}; "
             f"sampling: {config.get('video_sampling', 'benchmark_default')}; "
             f"maximum frames: {config.get('max_frames_by_benchmark', {})}.", "",
             "| Benchmark | Scope | Questions | Completed | Overall (%) | Action recognition (%) | Invalid | Complete |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |"]
    for name, stats in report["benchmarks"].items():
        def fmt(value):
            return "—" if value is None else f"{value:.2f}"
        scope = manifest["datasets"][name]["scope"]
        overall = stats.get("avg") if name == "mvbench" else stats.get("overall")
        display = {"mvbench": "MVBench (Avg / FA)", "tempcompass": "TempCompass-MCQ",
                   "videomme": "Video-MME (no subtitles)", "seedbench": "SEED-Bench Video"}[name]
        lines.append(f"| {display} | {scope} | {stats['n_total']} | {stats['n_done']} | {fmt(overall)} | {fmt(stats.get('action_recognition'))} | {stats.get('invalid_answers', 0)} | {stats['complete']} |")
    lines += ["", "Action recognition: MVBench Fine-grained Action; Video-MME Action Recognition; TempCompass-MCQ Action; SEED-Bench Action Recognition.",
              "MVBench Overall in this table is the task macro average (Avg); its JSON overall field is pooled accuracy.",
              "Missing/invalid predictions score zero. Full annotation coverage requires Scope=full and Complete=True."]
    if "seedbench" in report["benchmarks"]:
        lines += ["SEED-Bench: exactly 8 official pre-extracted frames in order, no resampling. "
                  "2 FPS is a synthetic ordinal timebase, not the source-video FPS. "
                  "Generated MCQ accuracy; not the original candidate-likelihood protocol."]
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print(f"Results: {run_dir.resolve()}", flush=True)
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.exp_name):
        raise ValueError("EXP_NAME must contain only letters, digits, underscores, dots and hyphens")
    run_dir = Path(args.output_dir).expanduser() / args.exp_name
    if args.preflight and not args.score_only:
        return execute(args, run_dir)
    with experiment_lock(run_dir):
        return execute(args, run_dir)


def execute(args, run_dir):
    if args.score_only:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        samples = [Sample(**json.loads(line)) for line in (run_dir / "dataset.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        existing_errors = json.loads((run_dir / "errors.json").read_text()) if (run_dir / "errors.json").is_file() else []
        report = save_report(run_dir, samples, read_predictions(run_dir), manifest, existing_errors)
        return 0 if report["complete"] else 1
    benchmarks = [x.strip().lower() for x in args.benchmarks.split(",") if x.strip()]
    benchmarks = ["tempcompass" if b in {"tempcompass_mcq", "tempcompass-mcq"} else b for b in benchmarks]
    if not benchmarks or len(set(benchmarks)) != len(benchmarks) or set(benchmarks) - set(EXPECTED_COUNTS):
        raise ValueError("BENCHMARKS must be a unique comma-separated list of mvbench,videomme,tempcompass,seedbench")
    if args.backend == "vllm" and not args.model_path and not args.preflight:
        raise ValueError("Set MODEL_PATH to the local checkpoint directory or model ID")
    config = make_config(args, benchmarks)
    print(f"Evaluation profile: {config['eval_profile']}; prompt={config['prompt_style']}; "
          f"sampling={config['video_sampling']}; frame caps={config['max_frames_by_benchmark']}; "
          f"temperature={config['temperature']}; top_p={config['top_p']}; "
          f"max_new_tokens={config['max_new_tokens']}", flush=True)
    samples, audit = prepare_samples(args, benchmarks)
    for name, info in audit.items():
        print(f"{name}: {info['selected_questions']}/{info['annotation_questions']} questions, scope={info['scope']}", flush=True)
    if args.preflight:
        print("Data preflight passed. No model was loaded.", flush=True)
        return 0
    sample_text = "".join(json.dumps(sample.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for sample in samples)
    dataset_sha = hashlib.sha256(sample_text.encode()).hexdigest()
    fingerprint = hashlib.sha256(json.dumps({"config": config, "dataset_sha256": dataset_sha}, sort_keys=True).encode()).hexdigest()
    manifest = {"exp_name": args.exp_name, "created_at": utc_now(), "fingerprint": fingerprint, "dataset_sha256": dataset_sha,
                "config": config, "datasets": audit}
    previous = run_dir / "manifest.json"
    if previous.exists():
        if args.no_resume:
            raise FileExistsError(f"Output experiment already exists: {run_dir}. Use a new EXP_NAME.")
        old = json.loads(previous.read_text(encoding="utf-8"))
        if old["fingerprint"] != fingerprint:
            raise ValueError("Resume refused: model, code, dataset, or generation protocol changed. Use a new EXP_NAME.")
        manifest["created_at"] = old["created_at"]
        manifest["resumed_at"] = utc_now()
    elif (run_dir / "predictions").exists() and any((run_dir / "predictions").iterdir()):
        raise ValueError("Prediction files exist without a manifest; use a new EXP_NAME")
    for directory in (run_dir, run_dir / "predictions", run_dir / "logs"):
        directory.mkdir(parents=True, exist_ok=True)
    atomic_json(run_dir / "manifest.json", manifest)
    (run_dir / "dataset.jsonl").write_text(sample_text, encoding="utf-8")
    records = read_predictions(run_dir, repair_tail=True)
    from .scoring import summarize
    summarize(samples, records)  # reject duplicate/foreign IDs before launching expensive workers
    completed = {record["uid"] for record in records}
    pending = [sample for sample in samples if sample.uid not in completed]
    print(f"Resume: {len(completed)} saved, {len(pending)} remaining. Backend={args.backend}", flush=True)
    errors = run_workers(args, config, pending, run_dir)
    atomic_json(run_dir / "errors.json", errors)
    report = save_report(run_dir, samples, read_predictions(run_dir, repair_tail=True), manifest, errors)
    if errors:
        print("Evaluation failed; completed predictions were saved. Details: " + str(run_dir / "errors.json"), file=sys.stderr)
    return 0 if report["complete"] and not errors else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
