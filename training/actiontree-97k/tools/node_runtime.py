#!/usr/bin/env python3
"""Per-node checks and ownership-scoped Ray lifecycle for the AR launcher."""
from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"__pycache__", ".pytest_cache", ".git", "work", "outputs", "tensorboard_log", "data", ".venv"}


def source_files(root=ROOT):
    paths = []
    for directory, dirs, names in os.walk(root):
        dirs[:] = [name for name in dirs if name not in SKIP]
        paths.extend(Path(directory) / name for name in names
                     if name != ".DS_Store" and not name.endswith(".pyc"))
    return sorted(p for p in paths if p.is_file())


def raylet_identity():
    """Bind cleanup to the exact process, including its creation time."""
    import psutil
    matches = []
    for proc in psutil.process_iter(["pid", "name", "create_time", "cmdline"]):
        if proc.info["name"] in {"raylet", "gcs_server"}:
            matches.append({"pid": proc.info["pid"], "created": proc.info["create_time"],
                            "cmdline": proc.info["cmdline"]})
    return sorted(matches, key=lambda item: item["pid"])


def source_digest(root=ROOT):
    result = hashlib.sha256()
    for path in source_files(root):
        result.update(str(path.relative_to(root)).encode() + b"\0")
        result.update(path.read_bytes())
    return result.hexdigest()


def runtime_env(spec, host):
    # Do not inherit a previous experiment's model, Ray, CUDA, or NCCL routing.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAY_", "NCCL_", "TORCH_NCCL_", "GLOO_"))}
    for key in ("MY_HOST_IPV6", "RAY_ADDRESS", "VLLM_HOST_IP", "CUDA_VISIBLE_DEVICES"):
        env.pop(key, None)
    env.update(spec.get("environment", {}))
    env.update(PATH=str(Path(spec["train_env_root"]) / "bin") + os.pathsep + env.get("PATH", ""),
               LD_LIBRARY_PATH=os.pathsep.join(filter(None, [str(Path(spec["train_env_root"]) / "lib"),
                                                            env.get("LD_LIBRARY_PATH", "")])),
               PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               MY_HOST_IP=host, VLLM_HOST_IP=host, VLLM_ENABLE_V1_MULTIPROCESSING="0",
               VLLM_USE_MODELSCOPE="0", USE_MODELSCOPE_HUB="0", OMP_NUM_THREADS="1",
               FORCE_QWENVL_VIDEO_READER="decord",
               CUDA_VISIBLE_DEVICES=",".join(map(str, range(spec["gpus_per_node"]))),
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False")
    bypass = ["localhost", "127.0.0.1", spec["head_host"], *spec["worker_hosts"]]
    env["NO_PROXY"] = ",".join(filter(None, [env.get("NO_PROXY", ""), *bypass]))
    env["no_proxy"] = env["NO_PROXY"]
    if "model_output" in spec:
        env["TENSORBOARD_DIR"] = str(Path(spec["model_output"]) / "tensorboard")
    return env


def running_ray():
    result = subprocess.run(["pgrep", "-x", "raylet|gcs_server"], capture_output=True, text=True)
    if result.returncode not in (0, 1):
        raise RuntimeError("Cannot inspect existing raylet processes")
    return result.returncode == 0


def assert_idle():
    if running_ray():
        raise RuntimeError("Ray services are already running; stop the previous task and release Ray first. This launcher will not stop a cluster it does not own.")
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError("GPUs already have compute processes: " + result.stdout.strip())


@contextmanager
def _check_phase(host, phase):
    print(f"[{host}] check {phase}: BEGIN", flush=True)
    yield
    print(f"[{host}] check {phase}: OK", flush=True)


def _distribution_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"Required package {name!r} is not installed for {sys.executable} (environment {sys.prefix})"
        ) from error


def _import_runtime_module(name):
    try:
        return importlib.import_module(name)
    except Exception as error:
        raise RuntimeError(
            f"Cannot import {name!r} with {sys.executable}: {type(error).__name__}: {error}"
        ) from error


def check_node(spec, host, env):
    with _check_phase(host, "identity"):
        print(f"[{host}] python={sys.executable}; environment={sys.prefix}; code={ROOT}", flush=True)
        addresses = subprocess.check_output(["hostname", "-I"], text=True).split()
        if host not in addresses:
            raise RuntimeError(f"The current node addresses {addresses} do not include configured IP {host}")
    with _check_phase(host, "source"):
        if source_digest() != spec["source_digest"]:
            raise RuntimeError("Source files differ across the four nodes; synchronize the new project.")
    with _check_phase(host, "idle"):
        assert_idle()
    with _check_phase(host, "versions"):
        versions = {name: _distribution_version(name) for name in
                    ("torch", "torchvision", "transformers", "vllm", "ray", "qwen-vl-utils", "flash-attn")}
        print(f"[{host}] versions: " + json.dumps(versions, sort_keys=True), flush=True)
        for name, version in {"torch": "2.8.0", "transformers": "4.57.3", "vllm": "0.11.0", "qwen-vl-utils": "0.0.14"}.items():
            if versions[name].split("+")[0] != version:
                raise RuntimeError(f"{name}={versions[name]}; this integration requires {version}; Python: {sys.executable}")
    with _check_phase(host, "imports"):
        torch = _import_runtime_module("torch")
        _import_runtime_module("flash_attn")  # Detect extension/ABI failures before Ray.
        _import_runtime_module("vllm")
    with _check_phase(host, "cuda"):
        available = torch.cuda.is_available()
        count = torch.cuda.device_count() if available else 0
        if not available or count != spec["gpus_per_node"]:
            raise RuntimeError(f"Expected {spec['gpus_per_node']} visible CUDA GPUs, found {count}; CUDA available={available}")
    with _check_phase(host, "model"):
        from preflight_model import validate_model

        model = validate_model(spec["model_path"], spec["model_key"], check_processor=True)
    with _check_phase(host, "data"):
        for name in ("train.rlsd.think.jsonl", "train.sft.think.jsonl", "manifest.json"):
            path = Path(spec["data_root"]) / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError("Data file is missing or empty: " + str(path))
    with _check_phase(host, "shared-storage"):
        probe = Path(spec["shared_probe"])
        if probe.read_text() != spec["token"]:
            raise RuntimeError("Output storage is not shared at the same mount path on all four nodes")
    print(json.dumps({"host": host, "versions": versions, "model": model}, ensure_ascii=False), flush=True)


def ray_action(action, spec, host, env):
    owner = Path(tempfile.gettempdir()) / f"ar-multimodel-ray-{spec['ray_port']}.json"
    ray = str(Path(spec["train_env_root"]) / "bin/ray")
    if action == "start":
        if host not in subprocess.check_output(["hostname", "-I"], text=True).split():
            raise RuntimeError("The node IP does not match the configured Ray startup IP")
        assert_idle()
        if owner.exists():
            raise RuntimeError(f"A Ray ownership record already exists: {owner}; verify and remove records for finished tasks")
        record = {"token": spec["token"], "host": host, "code_dir": str(ROOT)}
        # Exclusive owner file also prevents two launchers from starting this node.
        with owner.open("x") as stream:
            json.dump(record, stream)
        args = [ray, "start", f"--node-ip-address={host}", f"--num-gpus={spec['gpus_per_node']}", "--disable-usage-stats"]
        if host == spec["head_host"]:
            temp = tempfile.mkdtemp(prefix="ar-mm-ray-")
            args += ["--head", f"--port={spec['ray_port']}", "--include-dashboard=false", f"--temp-dir={temp}"]
        else:
            args += [f"--address={spec['head_host']}:{spec['ray_port']}"]
        # Keep ownership on failure so the coordinator can clean partial starts.
        def interrupted(signum, frame):
            raise KeyboardInterrupt("Ray startup interrupted")
        previous = signal.signal(signal.SIGTERM, interrupted)
        try:
            subprocess.run(args, check=True, env=env, timeout=180)
        finally:
            record["raylets"] = raylet_identity()
            owner.write_text(json.dumps(record))
            signal.signal(signal.SIGTERM, previous)
    elif action == "stop":
        if not owner.exists():
            return
        record = json.loads(owner.read_text())
        if (record.get("token") != spec["token"] or record.get("host") != host
                or record.get("code_dir") != str(ROOT)):
            raise RuntimeError("Refusing to stop Ray services owned by another task")
        current = raylet_identity()
        if any(process not in record.get("raylets", []) for process in current):
            raise RuntimeError("Ray process identities have changed; refusing to stop the current cluster using an old ownership record")
        subprocess.run([ray, "stop"], check=True, env=env, timeout=120)
        if running_ray():
            raise RuntimeError("Ray is still running; retaining the ownership record for later cleanup")
        owner.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "start", "stop"])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help="Legacy base64-encoded JSON run specification")
    source.add_argument("--spec-stdin", action="store_true", help="Read the JSON run specification from stdin")
    parser.add_argument("--host", required=True)
    args = parser.parse_args()
    try:
        spec = json.load(sys.stdin) if args.spec_stdin else json.loads(base64.urlsafe_b64decode(args.spec))
        if not isinstance(spec, dict):
            raise ValueError("Run specification must be a JSON object")
        env = runtime_env(spec, args.host)
        os.environ.clear()
        os.environ.update(env)
        if args.action == "check":
            check_node(spec, args.host, env)
        else:
            ray_action(args.action, spec, args.host, env)
    except Exception as error:
        print(f"[{args.host}] {args.action} FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
