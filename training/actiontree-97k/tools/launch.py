#!/usr/bin/env python3
"""One head-node entry point: models, plan, connect, check, smoke, train, status, logs, stop, export."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from model_profiles import MODELS, resolve_model, resolve_model_path
from node_runtime import runtime_env, source_digest, source_files
from ssh_transport import SSHSession, ssh_args


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def load_config(path):
    value = json.loads(Path(path).read_text())
    hosts = [value["head_host"], *value["worker_hosts"]]
    if len(hosts) != 4 or len(set(hosts)) != 4 or value["gpus_per_node"] != 8:
        raise ValueError("The training configuration requires four nodes with eight GPUs per node.")
    for host in hosts:
        if str(ipaddress.IPv4Address(host)) != host:
            raise ValueError("Node addresses must use canonical IPv4 notation")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value["ssh_user"]):
        raise ValueError("Invalid SSH username")
    for key in ("train_env_root", "model_root", "data_root", "output_root"):
        if not isinstance(value[key], str) or not Path(value[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path")
    for key, path in value.get("model_paths", {}).items():
        resolve_model(key)
        if not isinstance(path, str) or not Path(path).is_absolute():
            raise ValueError("Model paths in model_paths must be absolute paths")
    for key in ("ray_port", "nccl_port"):
        if type(value[key]) is not int or not 1024 <= value[key] <= 65535:
            raise ValueError(f"{key} must be an integer in 1024..65535")
    if value["ray_port"] == value["nccl_port"]:
        raise ValueError("Ray and NCCL must use different ports")
    if type(value.get("code_sync")) is not bool:
        raise ValueError("code_sync must be true or false")
    for key, val in value.get("environment", {}).items():
        if not key.startswith(("NCCL_", "TORCH_NCCL_", "GLOO_")) or not isinstance(val, str):
            raise ValueError("environment only accepts string values for NCCL_/TORCH_NCCL_/GLOO_ settings")
    return value


def build_spec(config, model_key, mode, run_id=None):
    profile = resolve_model(model_key)
    explicit_path = config.get("model_paths", {}).get(model_key) or str(Path(config["model_root"]) / profile["name"])
    model_path = resolve_model_path(model_key, model_path=explicit_path,
                                    model_root=config["model_root"])
    token = uuid.uuid4().hex
    run_id = run_id or f"ar_{model_key}_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{token[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError("Invalid run name")
    # Full token/config/data hash validation is performed by check_report, not by this path name.
    import hashlib
    identity = hashlib.sha256((str(model_path) + "\0" + config["data_root"]).encode()).hexdigest()[:12]
    spec = dict(config, model_key=model_key, model_path=str(model_path), hf_id=profile["hf_id"],
                mode=mode, run_id=run_id, token=token, code_dir=str(ROOT), source_digest=source_digest())
    model_output = Path(config["output_root"]) / model_key
    spec.update(model_output=str(model_output), run_dir=str(model_output / run_id),
                report=str(ROOT / "work/preflight" / model_key / identity / "report.json"))
    spec["state_dir"] = str(model_output / "launches" / run_id)
    spec["shared_probe"] = str(Path(spec["state_dir"]) / "shared_probe")
    return spec


def plan(spec):
    return {"model": spec["hf_id"], "model_path": spec["model_path"],
            "nodes": [spec["head_host"], *spec["worker_hosts"]], "gpus": 32,
            "data_root": spec["data_root"], "output": spec["run_dir"], "preflight_report": spec["report"],
            "contract": {"rows": 97430, "alignment": "mesd", "ema_rate": 0.01,
                         "decay_steps": 1000, "teacher_disable_after": 1000, "reference_coef": 0.1,
                         "rollout_batch_size": 32, "global_batch_size": 32, "rollout_n": 4,
                         "video_frames": 16, "fps": 2.0, "learning_rate": 1e-6},
            "stages": ["foreground SSH check", "sync/check four nodes", "97k data/tokenizer preflight", "video forward/backward",
                       "start owned Ray cluster", "32-rank NCCL check", "two-step smoke",
                       *(["fresh formal training"] if spec["mode"] == "train" else []), "release owned Ray"]}


def bootstrap(spec, *, require_code=True):
    """Set loader paths before Python starts, as in video_trsi_cluster.sh."""
    env_root = Path(spec["train_env_root"])
    code = spec["code_dir"]
    return ("set -euo pipefail\n"
            f"export PATH={shlex.quote(str(env_root / 'bin'))}:\"$PATH\"\n"
            f"export LD_LIBRARY_PATH={shlex.quote(str(env_root / 'lib'))}${{LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}}\n"
            f"export PYTHONPATH={shlex.quote(code)}\n"
            f"test -x {shlex.quote(str(env_root / 'bin/python'))} || "
            f"{{ echo {shlex.quote('Training environment Python is missing or not executable: ' + str(env_root / 'bin/python'))} >&2; exit 1; }}\n"
            + (f"cd -- {shlex.quote(code)}\n" if require_code else "cd /\n"))


def node_command(spec, host, action):
    # The script and JSON travel over stdin; errors never dump an encoded spec.
    return ["bash", "-s"] if host == spec["head_host"] else ssh_args(spec, host) + ["bash -s"]


def node_script(spec, host, action):
    command = [str(Path(spec["train_env_root"]) / "bin/python"), "-u", "-B",
               str(Path(spec["code_dir"]) / "tools/node_runtime.py"), action, "--spec-stdin", "--host", host]
    remote_spec = {key: value for key, value in spec.items() if key != "_ssh"}
    return (bootstrap(spec) + "exec " + shlex.join(command) + " <<'AR_SPEC_JSON'\n"
            + json.dumps(remote_spec, ensure_ascii=False) + "\nAR_SPEC_JSON\n")


def run_node_logged(spec, host, action, command, *, script=None, stdin=None, timeout=60):
    log = Path(spec["state_dir"]) / "nodes" / f"{host}.{action}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{action}] {host} BEGIN — {log}", flush=True)
    failure = None
    with log.open("wb") as stream:
        try:
            result = subprocess.run(command, input=script, stdin=stdin, text=script is not None,
                                    stdout=stream, stderr=subprocess.STDOUT,
                                    env=SSHSession(spec).environment(), timeout=timeout, cwd=ROOT)
            if result.returncode:
                failure = f"exit={result.returncode}"
        except subprocess.TimeoutExpired:
            failure = f"timed out after {timeout}s"
        except OSError as error:
            failure = str(error)
    output = log.read_text(errors="replace")
    if failure:
        tail = "\n".join(line[:1200] for line in output.splitlines()[-30:])
        raise RuntimeError(f"{host} [{action}] failed ({failure})\nLog: {log}\n{tail}") from None
    print(f"[{action}] {host} OK", flush=True)
    return output


def node(spec, host, action):
    return run_node_logged(spec, host, action, node_command(spec, host, action),
                           script=node_script(spec, host, action),
                           timeout={"check": 600, "start": 210, "stop": 150}[action])


def remote_python(spec, host, action, code, arguments=()):
    command = [str(Path(spec["train_env_root"]) / "bin/python"), "-B", "-c", code, *arguments]
    return run_node_logged(spec, host, action, ssh_args(spec, host) + ["bash -s"],
                           script=bootstrap(spec, require_code=False) + "exec " + shlex.join(command) + "\n")


def synchronize(spec):
    if not spec["code_sync"]:
        return
    # Shared mounts need no copy. For private disks copy only this project's sources.
    # Never rsync --delete; refuse an unrelated nonempty target directory.
    probe_code = """import sys,pathlib,hashlib
r=pathlib.Path(sys.argv[1]); expected=sys.argv[2]
if r.exists() and any(r.iterdir()) and (not (r/'PROJECT_ID').is_file() or (r/'PROJECT_ID').read_text().strip()!='ar-multimodel-97k-v1'):
 raise SystemExit('Refusing to overwrite an unrelated directory: '+str(r))
if r.is_dir() and (r/'tools/node_runtime.py').is_file():
 sys.path.insert(0,str(r/'tools'))
 from node_runtime import source_digest
 print('AR_MM_SYNC_STATUS=' + ('MATCH' if source_digest(r)==expected else 'COPY'))
else: print('AR_MM_SYNC_STATUS=COPY')
"""
    archive = None
    try:
        for host in spec["worker_hosts"]:
            result = remote_python(spec, host, "sync-probe", probe_code, [str(ROOT), spec["source_digest"]])
            markers = [line.strip() for line in result.splitlines() if line.startswith("AR_MM_SYNC_STATUS=")]
            if markers == ["AR_MM_SYNC_STATUS=MATCH"]:
                continue
            if markers != ["AR_MM_SYNC_STATUS=COPY"]:
                raise RuntimeError(f"{host} returned an invalid source-sync probe result; see the node sync-probe log")
            # Refuse synchronization while another job has allocated this worker.
            idle = """import subprocess
r=subprocess.run(['pgrep','-x','raylet|gcs_server'],capture_output=True)
if r.returncode!=1: raise SystemExit('Ray already running; source sync refused')
out=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
if out.strip(): raise SystemExit('GPU busy; source sync refused')
"""
            remote_python(spec, host, "sync-idle", idle)
            if archive is None:
                archive = tempfile.TemporaryFile()
                with tarfile.open(fileobj=archive, mode="w") as tar:
                    for path in source_files():
                        tar.add(path, arcname=str(path.relative_to(ROOT)), recursive=False)
            archive.seek(0)
            command = "mkdir -p -- " + shlex.quote(str(ROOT)) + " && tar -xf - -C " + shlex.quote(str(ROOT))
            run_node_logged(spec, host, "sync-copy", ssh_args(spec, host) + [command],
                            stdin=archive, timeout=300)
    finally:
        if archive is not None:
            archive.close()


def stage_env(spec, run_name=None):
    env = runtime_env(spec, spec["head_host"])
    # Explicit profiles win over stale exports in an interactive terminal.
    env.update(MODEL_KEY=spec["model_key"], MODEL_PATH=spec["model_path"],
               TRAIN_ENV_ROOT=spec["train_env_root"], MERGED_DATA_ROOT=spec["data_root"],
               TRAIN_DATA=str(Path(spec["data_root"]) / "train.rlsd.think.jsonl"),
               SFT_DATA=str(Path(spec["data_root"]) / "train.sft.think.jsonl"),
               OUTPUT_ROOT=spec["model_output"], AR_PREFLIGHT_REPORT=spec["report"],
               HEAD_HOST=spec["head_host"], WORKER_HOSTS=",".join(spec["worker_hosts"]),
               NNODES="4", N_GPUS="8", RAY_PORT=str(spec["ray_port"]), AR_NCCL_PORT=str(spec["nccl_port"]),
               RAY_ADDRESS=f"{spec['head_host']}:{spec['ray_port']}", RUN_NAME=run_name or spec["run_id"],
               CHECK_GPU="0", FORCE_QWENVL_VIDEO_READER="decord",
               TENSORBOARD_DIR=str(Path(spec["model_output"]) / "tensorboard"))
    return env


def run_stage(spec, action, run_name=None):
    print(f"\n=== {spec['model_key']}: {action} ===", flush=True)
    subprocess.run(["bash", str(ROOT / "examples/action_recognition/run_97430.sh"), action],
                   env=stage_env(spec, run_name), cwd=ROOT, check=True)


def prepare(spec):
    python = str(Path(spec["train_env_root"]) / "bin/python")
    command = [python, "-m", "verl.ar.check_merged_97430_data", "--data-root", spec["data_root"],
               "--model", spec["model_path"], "--report", spec["report"], "--check-report"]
    result = subprocess.run(command, env=stage_env(spec), cwd=ROOT, capture_output=True, text=True)
    if result.returncode:
        print("Preparing the 97k data/tokenizer validation record for this model.", flush=True)
        run_stage(spec, "prepare")
    else:
        print("The 97k data, tokenizer, and reward-protocol fingerprints match; reusing this model preflight report.", flush=True)


def verify_completion(spec, run_name, expected_steps=None):
    path = Path(spec["model_output"]) / run_name / "training_complete.json"
    data = json.loads(path.read_text())
    if (data["model_path"] != spec["model_path"] or data["experiment_name"] != run_name
            or data["global_step"] != data["training_steps"] or data["global_step"] < 1
            or (expected_steps is not None and data["global_step"] != expected_steps)):
        raise RuntimeError("Training completion marker does not match: " + str(path))


def set_state(spec, status, **extra):
    write_json(Path(spec["state_dir"]) / "status.json", dict(status=status, run_id=spec["run_id"],
               model=spec["model_key"], updated=datetime.now().isoformat(), **extra))


def pipeline(spec):
    if str(ROOT) != spec["code_dir"] or source_digest() != spec["source_digest"]:
        raise RuntimeError("Source files changed after submission; restart the launcher")
    hosts = [spec["head_host"], *spec["worker_hosts"]]
    attempted = []
    success = False
    def terminate(signum, frame):
        raise KeyboardInterrupt("received signal " + str(signum))
    previous = signal.signal(signal.SIGTERM, terminate)
    set_state(spec, "preflight", pid=os.getpid())
    try:
        synchronize(spec)
        with ThreadPoolExecutor(max_workers=4) as pool:
            # Consume every result; a failed node prevents all training stages.
            results = [pool.submit(node, spec, host, "check") for host in hosts]
            errors = []
            for result in results:
                try:
                    result.result()
                except Exception as error:
                    errors.append(str(error))
            if errors:
                raise RuntimeError("Node preflight failed: " + "\n".join(errors))
        prepare(spec)
        run_stage(spec, "check-video")
        if spec["mode"] == "check":
            success = True
            return
        for host in hosts:
            attempted.append(host)
            node(spec, host, "start")
        run_stage(spec, "cluster")
        set_state(spec, "smoke", pid=os.getpid())
        smoke_id = spec["run_id"] + "_smoke"
        run_stage(spec, "smoke", smoke_id)
        verify_completion(spec, smoke_id, expected_steps=2)
        if spec["mode"] == "train":
            set_state(spec, "training", pid=os.getpid())
            run_stage(spec, "train")
            verify_completion(spec, spec["run_id"])
        success = True
    except BaseException as error:
        set_state(spec, "failed", error=str(error), pid=os.getpid())
        raise
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        cleanup = []
        for host in reversed(attempted):
            try:
                node(spec, host, "stop")
            except Exception as error:
                cleanup.append(f"{host}: {error}")
        Path(spec["shared_probe"]).unlink(missing_ok=True)
        if cleanup:
            set_state(spec, "cleanup_failed", errors=cleanup, training_completed=success)
            print("Ray cleanup is incomplete; run bash run.sh cleanup " + spec["model_key"], flush=True)
        elif success:
            set_state(spec, "completed")
        signal.signal(signal.SIGTERM, previous)


def head_only(spec):
    if sys.platform != "linux":
        raise RuntimeError("Training runs on the Linux head node; models / plan / export --dry-run are available locally.")
    ips = subprocess.check_output(["hostname", "-I"], text=True).split()
    if spec["head_host"] not in ips:
        raise RuntimeError("Run the launch command only on head node " + spec["head_host"] + "")


def start(spec, foreground=False):
    head_only(spec)
    python = Path(spec["train_env_root"]) / "bin/python"
    if not python.is_file():
        raise ValueError("Training environment does not exist: " + str(python))
    state = Path(spec["state_dir"])
    state.mkdir(parents=True, exist_ok=False)
    Path(spec["shared_probe"]).write_text(spec["token"])
    session = SSHSession(spec)
    try:
        set_state(spec, "connecting")
        # Authentication is resolved while the user's terminal is still attached.
        session.prepare(interactive=True)
        env = runtime_env(spec, spec["head_host"])
        for key in ("SSH_AUTH_SOCK", "SSH_AGENT_PID"):
            if key in session.environment():
                env[key] = session.environment()[key]
        _start_prepared(spec, foreground, env)
    except BaseException as error:
        if not (state / "process.json").exists():
            set_state(spec, "failed", error=str(error))
            Path(spec["shared_probe"]).unlink(missing_ok=True)
            session.cleanup()
        raise


def _start_prepared(spec, foreground, env):
    python = Path(spec["train_env_root"]) / "bin/python"
    state = Path(spec["state_dir"])
    spec_path = state / "run.json"
    write_json(spec_path, spec)
    log = state / "launcher.log"
    command = [str(python), "-u", str(ROOT / "tools/launch.py"), "_pipeline", str(spec_path)]
    if foreground:
        with log.open("xb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       env=env, start_new_session=True)
            write_json(state / "process.json", {"pid": process.pid, "spec": str(spec_path)})
            try:
                for line in iter(process.stdout.readline, b""):
                    stream.write(line); stream.flush()
                    sys.stdout.buffer.write(line); sys.stdout.buffer.flush()
                code = process.wait()
            except KeyboardInterrupt:
                os.killpg(process.pid, signal.SIGTERM)
                code = process.wait()
            if code:
                raise RuntimeError(f"Pipeline failed (exit={code}); log: {log}")
    else:
        with log.open("xb") as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        write_json(state / "process.json", {"pid": process.pid, "spec": str(spec_path)})
        # A rejected concurrent launch must not replace the active run pointer.
        for _ in range(50):
            if (state / "accepted.json").exists():
                break
            if process.poll() is not None:
                raise RuntimeError(f"Pipeline startup failed; see {log}")
            time.sleep(0.1)
        mode_text = {"train": "formal training starts automatically after preflight and the two-step smoke test pass", "smoke": "running preflight and the two-step smoke test",
                     "check": "running four-node, data, and video checks"}[spec["mode"]]
        print(f"Pipeline submitted (PID {process.pid}); {mode_text}.\nModel: {spec['hf_id']}\n"
              f"Output: {spec['run_dir']}\nLog: {log}\nView: bash run.sh logs {spec['model_key']}")


def connect(spec):
    """Read-only connection/Python diagnostic; no source sync or GPU work."""
    head_only(spec)
    state = Path(spec["state_dir"])
    state.mkdir(parents=True, exist_ok=False)
    session = SSHSession(spec)
    try:
        session.prepare(interactive=True)
        code = "import sys,socket; print('host='+socket.gethostname()); print('python='+sys.executable); print(sys.version)"
        for host in [spec["head_host"], *spec["worker_hosts"]]:
            command = [str(Path(spec["train_env_root"]) / "bin/python"), "-B", "-c", code]
            output = run_node_logged(spec, host, "python", node_command(spec, host, "python"),
                                     script=bootstrap(spec, require_code=False) + "exec " + shlex.join(command) + "\n")
            print(output.strip(), flush=True)
        print(f"SSH/Python connection checks passed on all four nodes. Log: {state / 'nodes'}", flush=True)
    finally:
        session.cleanup()


def latest(config, key):
    resolve_model(key)
    pointer = Path(config["output_root"]) / "latest" / (key + ".json")
    state = Path(json.loads(pointer.read_text())["state_dir"])
    spec = json.loads((state / "run.json").read_text())
    return state, spec


def process_is_owned(pid, spec_path):
    proc = Path(f"/proc/{pid}/cmdline")
    try:
        if proc.is_file():
            argv = proc.read_bytes().decode().rstrip("\0").split("\0")
        else:
            result = subprocess.run(["ps", "-p", str(pid), "-o", "args="], capture_output=True, text=True)
            if result.returncode:
                return False
            argv = shlex.split(result.stdout)
        index = argv.index(str(ROOT / "tools/launch.py"))
        return argv[index + 1:] == ["_pipeline", str(spec_path)]
    except (OSError, ValueError):
        return False


def pipeline_entry(spec):
    session = SSHSession(spec)
    try:
        head_only(spec)
        lock_path = Path(spec["output_root"]) / ".cluster.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                set_state(spec, "failed", error="A task from this project is already running on this four-node cluster")
                raise RuntimeError("A task from this project is already running on this four-node cluster")
            write_json(Path(spec["output_root"]) / "latest" / (spec["model_key"] + ".json"),
                       {"state_dir": spec["state_dir"]})
            write_json(Path(spec["state_dir"]) / "accepted.json", {"pid": os.getpid()})
            pipeline(spec)
    except BaseException as error:
        status_file = Path(spec["state_dir"]) / "status.json"
        old = json.loads(status_file.read_text()) if status_file.exists() else {}
        if old.get("status") not in ("failed", "cleanup_failed"):
            set_state(spec, "failed", error=str(error))
        raise
    finally:
        Path(spec["shared_probe"]).unlink(missing_ok=True)
        session.cleanup()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "_pipeline":
        pipeline_entry(json.loads(Path(argv[1]).read_text()))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "configs/cluster.json"))
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("models", help="List supported models")
    for name in ("plan", "connect", "check", "smoke", "train", "status", "logs", "stop", "cleanup"):
        sub = subs.add_parser(name)
        sub.add_argument("model", choices=list(MODELS))
        if name in ("check", "smoke", "train"):
            sub.add_argument("--foreground", action="store_true", help="Run in the foreground (default: background)")
    export = subs.add_parser("export", aliases=["trans"], help="Convert a complete FSDP checkpoint to a Hugging Face model")
    export.add_argument("checkpoint")
    export.add_argument("--output")
    export.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "models":
        for key, profile in MODELS.items():
            print(f"{key:12s} {profile['hf_id']}")
        return
    config = load_config(args.config)
    if args.command in ("export", "trans"):
        # Metadata-only dry-run works on a laptop; actual merge uses training dependencies.
        python = sys.executable if args.dry_run else str(Path(config["train_env_root"]) / "bin/python")
        command = [python, str(ROOT / "tools/export_hf.py"), args.checkpoint]
        if args.output:
            command += ["--output", args.output]
        if args.dry_run:
            command += ["--dry-run"]
        subprocess.run(command, check=True)
        return
    if args.command in ("plan", "connect", "check", "smoke", "train"):
        spec = build_spec(config, args.model, "train" if args.command == "plan" else args.command)
        if args.command == "plan":
            print(json.dumps(plan(spec), indent=2, ensure_ascii=False))
        elif args.command == "connect":
            connect(spec)
        else:
            start(spec, args.foreground)
        return
    state, spec = latest(config, args.model)
    if args.command == "status":
        status = json.loads((state / "status.json").read_text()) if (state / "status.json").exists() else {"status": "submitted"}
        process = json.loads((state / "process.json").read_text())
        status["process_alive"] = process_is_owned(process["pid"], state / "run.json")
        status["log"] = str(state / "launcher.log")
        print(json.dumps(status, indent=2, ensure_ascii=False))
    elif args.command == "logs":
        try:
            subprocess.run(["tail", "-n", "80", "-F", str(state / "launcher.log")], check=True)
        except KeyboardInterrupt:
            pass
    elif args.command == "stop":
        head_only(spec)
        process = json.loads((state / "process.json").read_text())
        if not process_is_owned(process["pid"], state / "run.json"):
            raise RuntimeError("The pipeline is not running or its PID has changed; no signal was sent")
        # Signal coordinator + its local training children. Coordinator cleans only its own Ray services.
        if os.getpgid(process["pid"]) == process["pid"]:
            os.killpg(process["pid"], signal.SIGTERM)
        else:
            os.kill(process["pid"], signal.SIGTERM)
        print("Stop requested for this pipeline; only previously completed checkpoints are preserved.")
    elif args.command == "cleanup":
        head_only(spec)
        process = json.loads((state / "process.json").read_text())
        if process_is_owned(process["pid"], state / "run.json"):
            raise RuntimeError("The pipeline is still running; use stop first")
        # Previous pipeline's private SSH session was already released.
        spec = {key: value for key, value in spec.items() if key != "_ssh"}
        session = SSHSession(spec)
        try:
            session.prepare(interactive=True)
            for host in reversed([spec["head_host"], *spec["worker_hosts"]]):
                node(spec, host, "stop")
        finally:
            session.cleanup()
        status_file = state / "status.json"
        old = json.loads(status_file.read_text()) if status_file.exists() else {}
        completed = old.get("training_completed") or old.get("status") == "completed"
        set_state(spec, "completed" if completed else "stopped", cleanup_completed=True)
        print("Cleanup completed for the Ray services owned by this run.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
