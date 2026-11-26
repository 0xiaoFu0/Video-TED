"""CPU-only orchestration checks; no subprocess, SSH, Ray, or GPU job is started."""

import importlib
import json
from pathlib import Path
import signal
import subprocess

import pytest

from tools import launch

node_runtime = importlib.import_module("node_runtime")


@pytest.fixture
def config(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(launch, "ROOT", project)
    monkeypatch.setattr(node_runtime, "ROOT", project)
    monkeypatch.setattr(launch, "source_digest", lambda: "fixture-source")
    monkeypatch.setattr(launch.signal, "signal", lambda *args: signal.SIG_DFL)
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_ROOT", raising=False)
    return {
        "head_host": "10.0.0.1", "worker_hosts": ["10.0.0.2", "10.0.0.3", "10.0.0.4"],
        "ssh_user": "root", "ssh_key": "", "train_env_root": str(tmp_path / "venv"),
        "model_root": str(tmp_path / "models"), "model_paths": {},
        "data_root": str(tmp_path / "data"), "output_root": str(tmp_path / "outputs"),
        "gpus_per_node": 8, "ray_port": 6411, "nccl_port": 29551, "code_sync": False,
        "environment": {"NCCL_DEBUG": "WARN", "NCCL_SOCKET_IFNAME": "=eth0"},
    }


@pytest.fixture
def spec(config):
    result = launch.build_spec(config, "qwen3-4b", "train", run_id="fixture_train")
    Path(result["state_dir"]).mkdir(parents=True)
    Path(result["shared_probe"]).write_text(result["token"])
    return result


def status(spec):
    return json.loads((Path(spec["state_dir"]) / "status.json").read_text())


def install_pipeline_fakes(monkeypatch, *, failed_node=None, failed_stage=None, bad_smoke_marker=False,
                           missing_smoke_marker=False, failed_start=None, failed_stop=None):
    events = []

    def node(spec, host, action):
        events.append(("node", action, host))
        if (action == "check" and host == failed_node) or (action == "start" and host == failed_start):
            raise subprocess.CalledProcessError(1, ["fake", action, host])
        if action == "stop" and host == failed_stop:
            raise RuntimeError("fixture cleanup refused")

    def stage(spec, action, run_name=None):
        events.append(("stage", action, run_name))
        if action == failed_stage:
            raise subprocess.CalledProcessError(1, ["fake", action])
        if action in ("smoke", "train"):
            if action == "smoke" and missing_smoke_marker:
                return
            name = run_name or spec["run_id"]
            steps = 1 if action == "smoke" and bad_smoke_marker else (2 if action == "smoke" else 3)
            launch.write_json(Path(spec["model_output"]) / name / "training_complete.json", {
                "model_path": spec["model_path"], "experiment_name": name,
                "global_step": steps, "training_steps": steps,
            })

    monkeypatch.setattr(launch, "node", node)
    monkeypatch.setattr(launch, "run_stage", stage)
    monkeypatch.setattr(launch, "synchronize", lambda spec: events.append(("synchronize",)))
    monkeypatch.setattr(launch, "prepare", lambda spec: events.append(("prepare",)))
    return events


@pytest.mark.parametrize("failed_host", ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"])
def test_any_node_preflight_failure_blocks_all_training(spec, monkeypatch, failed_host):
    events = install_pipeline_fakes(monkeypatch, failed_node=failed_host)
    with pytest.raises(RuntimeError, match="Node preflight failed"):
        launch.pipeline(spec)
    assert {event[2] for event in events if event[:2] == ("node", "check")} == {
        spec["head_host"], *spec["worker_hosts"]}
    assert not any(event[0] in ("stage", "prepare") for event in events)
    assert not any(event[:2] == ("node", "start") for event in events)
    assert status(spec)["status"] == "failed"
    assert not Path(spec["shared_probe"]).exists()


@pytest.mark.parametrize("failure", ["process", "missing_marker", "wrong_steps"])
def test_smoke_failure_or_incomplete_smoke_never_starts_formal_training(spec, monkeypatch, failure):
    events = install_pipeline_fakes(monkeypatch,
        failed_stage="smoke" if failure == "process" else None,
        missing_smoke_marker=failure == "missing_marker", bad_smoke_marker=failure == "wrong_steps")
    with pytest.raises((subprocess.CalledProcessError, FileNotFoundError, RuntimeError)):
        launch.pipeline(spec)
    assert len([event for event in events if event[:2] == ("stage", "smoke")]) == 1
    assert not any(event[:2] == ("stage", "train") for event in events)
    assert [event[2] for event in events if event[:2] == ("node", "stop")] == list(reversed(
        [spec["head_host"], *spec["worker_hosts"]]))
    assert status(spec)["status"] == "failed"


def test_success_trains_once_after_verified_smoke_and_releases_all_owned_nodes(spec, monkeypatch):
    events = install_pipeline_fakes(monkeypatch)
    launch.pipeline(spec)
    stages = [event[1] for event in events if event[0] == "stage"]
    assert stages == ["check-video", "cluster", "smoke", "train"]
    smoke_event = next(event for event in events if event[:2] == ("stage", "smoke"))
    assert smoke_event[2] == spec["run_id"] + "_smoke"
    assert [event[2] for event in events if event[:2] == ("node", "start")] == [spec["head_host"], *spec["worker_hosts"]]
    assert [event[2] for event in events if event[:2] == ("node", "stop")] == list(reversed(
        [spec["head_host"], *spec["worker_hosts"]]))
    assert status(spec)["status"] == "completed"
    assert not Path(spec["shared_probe"]).exists()


@pytest.mark.parametrize("mode,expected_stages", [
    ("check", ["check-video"]), ("smoke", ["check-video", "cluster", "smoke"]),
])
def test_check_and_smoke_commands_do_not_train_formally(spec, monkeypatch, mode, expected_stages):
    spec["mode"] = mode
    events = install_pipeline_fakes(monkeypatch)
    launch.pipeline(spec)
    assert [event[1] for event in events if event[0] == "stage"] == expected_stages
    if mode == "check":
        assert not any(event[:2] == ("node", "start") for event in events)
    assert status(spec)["status"] == "completed"


def test_partial_start_failure_cleans_only_attempted_nodes(spec, monkeypatch):
    failed_host = spec["worker_hosts"][0]
    events = install_pipeline_fakes(monkeypatch, failed_start=failed_host)
    with pytest.raises(subprocess.CalledProcessError):
        launch.pipeline(spec)
    assert [event[2] for event in events if event[:2] == ("node", "stop")] == [failed_host, spec["head_host"]]
    assert not any(event[:2] in (("stage", "smoke"), ("stage", "train")) for event in events)


@pytest.mark.parametrize("stage,expected_stops", [("check-video", 0), ("cluster", 4)])
def test_video_or_cluster_failure_blocks_smoke_and_formal(spec, monkeypatch, stage, expected_stops):
    events = install_pipeline_fakes(monkeypatch, failed_stage=stage)
    with pytest.raises(subprocess.CalledProcessError):
        launch.pipeline(spec)
    assert not any(event[:2] in (("stage", "smoke"), ("stage", "train")) for event in events)
    assert len([event for event in events if event[:2] == ("node", "stop")]) == expected_stops


def test_cleanup_error_is_persisted_without_hiding_other_cleanup(spec, monkeypatch):
    events = install_pipeline_fakes(monkeypatch, failed_stop=spec["worker_hosts"][1])
    launch.pipeline(spec)
    assert len([event for event in events if event[:2] == ("node", "stop")]) == 4
    final = status(spec)
    assert final["status"] == "cleanup_failed"
    assert final["training_completed"] is True
    assert "fixture cleanup refused" in final["errors"][0]


def test_model_output_reports_and_state_are_isolated_between_profiles(config):
    specs = [launch.build_spec(config, key, "train", run_id="same_name") for key in launch.MODELS]
    for field in ("model_path", "model_output", "run_dir", "report", "state_dir"):
        assert len({item[field] for item in specs}) == len(launch.MODELS), field
    assert len({item["token"] for item in specs}) == len(launch.MODELS)
    changed = dict(config, model_paths={"qwen3-4b": "/alternate/model"})
    other = launch.build_spec(changed, "qwen3-4b", "train")
    original = next(item for item in specs if item["model_key"] == "qwen3-4b")
    assert other["report"] != original["report"]


def test_stale_model_path_environment_cannot_override_config_profile(config, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/stale/Qwen2.5-VL-7B-Instruct")
    monkeypatch.setenv("MODEL_ROOT", "/stale/models")
    result = launch.build_spec(config, "qwen3-4b", "train")
    assert result["model_path"] == str(Path(config["model_root"]) / "Qwen3-VL-4B-Instruct")
    config["model_paths"] = {"qwen3-4b": "/explicit/four_b"}
    assert launch.build_spec(config, "qwen3-4b", "train")["model_path"] == "/explicit/four_b"


def test_stage_environment_pins_model_data_output_and_cluster(spec, monkeypatch):
    stale = {
        "MODEL_PATH": "/stale/model", "MODEL_KEY": "qwen25-3b", "TRAIN_ENV_ROOT": "/stale/env",
        "MERGED_DATA_ROOT": "/stale/data", "TRAIN_DATA": "/stale/train", "SFT_DATA": "/stale/sft",
        "OUTPUT_ROOT": "/stale/output", "RUN_NAME": "old_run", "NNODES": "1", "N_GPUS": "1",
        "AR_PREFLIGHT_REPORT": "/stale/report", "HEAD_HOST": "9.9.9.9", "WORKER_HOSTS": "9.9.9.8",
        "RAY_ADDRESS": "9.9.9.9:1", "RAY_REDIS_ADDRESS": "9.9.9.9:2", "RAY_PORT": "6551",
        "CUDA_VISIBLE_DEVICES": "7", "MY_HOST_IP": "9.9.9.9", "MY_HOST_IPV6": "::1",
        "VLLM_HOST_IP": "9.9.9.9", "NCCL_DEBUG": "TRACE", "NCCL_IB_HCA": "stale_hca",
        "GLOO_SOCKET_IFNAME": "stale0", "PYTHONPATH": "/stale/code", "TENSORBOARD_DIR": "/stale/tb",
    }
    for key, value in stale.items():
        monkeypatch.setenv(key, value)
    env = launch.stage_env(spec, "unique_smoke")
    expected = {
        "MODEL_PATH": spec["model_path"], "MODEL_KEY": spec["model_key"],
        "TRAIN_ENV_ROOT": spec["train_env_root"], "MERGED_DATA_ROOT": spec["data_root"],
        "TRAIN_DATA": str(Path(spec["data_root"]) / "train.rlsd.think.jsonl"),
        "SFT_DATA": str(Path(spec["data_root"]) / "train.sft.think.jsonl"),
        "OUTPUT_ROOT": spec["model_output"], "RUN_NAME": "unique_smoke", "NNODES": "4", "N_GPUS": "8",
        "AR_PREFLIGHT_REPORT": spec["report"], "HEAD_HOST": spec["head_host"],
        "WORKER_HOSTS": ",".join(spec["worker_hosts"]), "RAY_PORT": str(spec["ray_port"]),
        "RAY_ADDRESS": f"{spec['head_host']}:{spec['ray_port']}", "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
        "MY_HOST_IP": spec["head_host"], "VLLM_HOST_IP": spec["head_host"],
        "NCCL_DEBUG": "WARN", "NCCL_SOCKET_IFNAME": "=eth0", "PYTHONPATH": str(launch.ROOT),
        "TENSORBOARD_DIR": str(Path(spec["model_output"]) / "tensorboard"),
    }
    for key, value in expected.items():
        assert env[key] == value, key
    for key in ("MY_HOST_IPV6", "RAY_REDIS_ADDRESS", "NCCL_IB_HCA", "GLOO_SOCKET_IFNAME"):
        assert key not in env
    assert all(host in env["NO_PROXY"].split(",") for host in [spec["head_host"], *spec["worker_hosts"]])


def test_ray_stop_without_owner_never_runs_global_ray_stop(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(node_runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(node_runtime, "running_ray", lambda: True)
    monkeypatch.setattr(node_runtime.subprocess, "run", lambda *a, **k: pytest.fail("must not stop unowned Ray"))
    node_runtime.ray_action("stop", spec, spec["head_host"], {})


@pytest.mark.parametrize("changed_field,foreign_value", [
    ("token", "foreign-token"), ("host", "10.0.0.99"), ("code_dir", "/other/project"),
])
def test_ray_stop_rejects_foreign_ownership_record(spec, tmp_path, monkeypatch, changed_field, foreign_value):
    monkeypatch.setattr(node_runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    owner = tmp_path / f"ar-multimodel-ray-{spec['ray_port']}.json"
    record = {"token": spec["token"], "host": spec["head_host"], "code_dir": str(node_runtime.ROOT)}
    record[changed_field] = foreign_value
    launch.write_json(owner, record)
    calls = []
    monkeypatch.setattr(node_runtime.subprocess, "run", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(node_runtime, "running_ray", lambda: False)
    with pytest.raises(RuntimeError):
        node_runtime.ray_action("stop", spec, spec["head_host"], {})
    assert calls == []
    assert owner.exists()


def test_start_refuses_existing_ray_without_attempting_start_or_stop(spec, monkeypatch):
    monkeypatch.setattr(node_runtime.subprocess, "check_output", lambda *a, **k: spec["head_host"])
    monkeypatch.setattr(node_runtime, "running_ray", lambda: True)
    monkeypatch.setattr(node_runtime.subprocess, "run", lambda *a, **k: pytest.fail("existing Ray must not be changed"))
    with pytest.raises(RuntimeError, match="Ray services are already running"):
        node_runtime.ray_action("start", spec, spec["head_host"], {})


def create_owner(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(node_runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    owner = tmp_path / f"ar-multimodel-ray-{spec['ray_port']}.json"
    identities = [{"pid": 1234, "created": 100.0, "cmdline": ["/raylet", "--fixture"]}]
    launch.write_json(owner, {
        "token": spec["token"], "host": spec["head_host"], "code_dir": str(node_runtime.ROOT),
        "raylets": identities,
    })
    return owner, identities


@pytest.mark.parametrize("field,value", [("pid", 5678), ("created", 200.0), ("cmdline", ["/foreign/raylet"])])
def test_stale_ray_record_cannot_stop_replacement_cluster(spec, tmp_path, monkeypatch, field, value):
    owner, recorded = create_owner(spec, tmp_path, monkeypatch)
    current = [{**recorded[0], field: value}]
    monkeypatch.setattr(node_runtime, "raylet_identity", lambda: current)
    monkeypatch.setattr(node_runtime.subprocess, "run", lambda *a, **k: pytest.fail("foreign Ray must not be stopped"))
    with pytest.raises(RuntimeError, match="process identities have changed"):
        node_runtime.ray_action("stop", spec, spec["head_host"], {})
    assert owner.exists()


@pytest.mark.parametrize("still_running", [False, True])
def test_owned_ray_stop_deletes_record_only_after_processes_exit(spec, tmp_path, monkeypatch, still_running):
    owner, identities = create_owner(spec, tmp_path, monkeypatch)
    monkeypatch.setattr(node_runtime, "raylet_identity", lambda: identities)
    monkeypatch.setattr(node_runtime, "running_ray", lambda: still_running)
    calls = []
    monkeypatch.setattr(node_runtime.subprocess, "run", lambda args, **kwargs: calls.append(args))
    if still_running:
        with pytest.raises(RuntimeError, match="Ray is still running"):
            node_runtime.ray_action("stop", spec, spec["head_host"], {})
    else:
        node_runtime.ray_action("stop", spec, spec["head_host"], {})
    assert calls == [[str(Path(spec["train_env_root"]) / "bin/ray"), "stop"]]
    assert owner.exists() is still_running


def test_stop_accepts_surviving_subset_of_owned_ray_processes(spec, tmp_path, monkeypatch):
    owner, survivors = create_owner(spec, tmp_path, monkeypatch)
    record = json.loads(owner.read_text())
    record["raylets"].append({"pid": 2345, "created": 101.0, "cmdline": ["/gcs_server", "--fixture"]})
    launch.write_json(owner, record)
    # The GCS process has already exited. The surviving raylet still has exactly
    # its recorded PID, creation time, and command line and remains safe to stop.
    monkeypatch.setattr(node_runtime, "raylet_identity", lambda: survivors)
    monkeypatch.setattr(node_runtime, "running_ray", lambda: False)
    calls = []
    monkeypatch.setattr(node_runtime.subprocess, "run", lambda args, **kwargs: calls.append(args))
    node_runtime.ray_action("stop", spec, spec["head_host"], {})
    assert calls == [[str(Path(spec["train_env_root"]) / "bin/ray"), "stop"]]
    assert not owner.exists()


def test_interrupted_ray_start_records_process_identity_and_can_be_cleaned(spec, tmp_path, monkeypatch):
    monkeypatch.setattr(node_runtime.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(node_runtime.tempfile, "mkdtemp", lambda **kwargs: str(tmp_path / "ray-session"))
    monkeypatch.setattr(node_runtime.subprocess, "check_output", lambda *args, **kwargs: spec["head_host"])
    monkeypatch.setattr(node_runtime, "assert_idle", lambda: None)
    identities = [{"pid": 3456, "created": 102.0, "cmdline": ["/raylet", "--partially-started"]}]
    monkeypatch.setattr(node_runtime, "raylet_identity", lambda: identities)
    monkeypatch.setattr(node_runtime, "running_ray", lambda: False)
    handlers = {signal.SIGTERM: signal.SIG_DFL}

    def record_handler(signum, handler):
        previous = handlers[signum]
        handlers[signum] = handler
        return previous

    calls = []

    def interrupted_command(args, **kwargs):
        calls.append(args)
        if args[1] == "start":
            handler = handlers[signal.SIGTERM]
            assert callable(handler)
            handler(signal.SIGTERM, None)

    monkeypatch.setattr(node_runtime.signal, "signal", record_handler)
    monkeypatch.setattr(node_runtime.subprocess, "run", interrupted_command)
    with pytest.raises(KeyboardInterrupt, match="Ray startup interrupted"):
        node_runtime.ray_action("start", spec, spec["head_host"], {})
    owner = tmp_path / f"ar-multimodel-ray-{spec['ray_port']}.json"
    record = json.loads(owner.read_text())
    assert record["token"] == spec["token"]
    assert record["host"] == spec["head_host"]
    assert record["raylets"] == identities
    assert handlers[signal.SIGTERM] == signal.SIG_DFL
    node_runtime.ray_action("stop", spec, spec["head_host"], {})
    assert [args[1] for args in calls] == ["start", "stop"]
    assert not owner.exists()


@pytest.mark.parametrize("owned", [True, False])
def test_stop_cli_signals_only_verified_pipeline(spec, config, tmp_path, monkeypatch, owned):
    state = Path(spec["state_dir"])
    launch.write_json(state / "process.json", {"pid": 23456})
    monkeypatch.setattr(launch, "load_config", lambda path: config)
    monkeypatch.setattr(launch, "latest", lambda config, key: (state, spec))
    monkeypatch.setattr(launch, "head_only", lambda spec: None)
    monkeypatch.setattr(launch, "process_is_owned", lambda pid, path: owned)
    calls = []
    monkeypatch.setattr(launch.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(launch.os, "killpg", lambda *args: calls.append(("group", *args)))
    monkeypatch.setattr(launch.os, "kill", lambda *args: calls.append(("process", *args)))
    if owned:
        launch.main(["stop", "qwen3-4b"])
        assert calls == [("group", 23456, signal.SIGTERM)]
    else:
        with pytest.raises(RuntimeError, match="no signal was sent"):
            launch.main(["stop", "qwen3-4b"])
        assert calls == []


@pytest.mark.parametrize("command,expected", [
    ("python {script} _pipeline {spec}", True),
    ("python {script} _pipeline {spec}.foreign", False),
    ("python /other/job.py --note '_pipeline {spec}'", False),
    ("python {script} plan qwen3-4b", False),
])
def test_process_identity_matches_exact_pipeline_and_spec(spec, monkeypatch, command, expected):
    path = Path(spec["state_dir"]) / "run.json"
    output = command.format(script=launch.ROOT / "tools/launch.py", spec=path)
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0, output))
    assert launch.process_is_owned(23456, path) is expected


def test_linux_proc_identity_uses_exact_nul_delimited_argv(spec, tmp_path, monkeypatch):
    real_path = Path
    spec_path = real_path(spec["state_dir"]) / "run.json"
    proc_path = tmp_path / "fake-proc-cmdline"
    monkeypatch.setattr(launch, "Path", lambda value: proc_path if value == "/proc/23456/cmdline" else real_path(value))
    monkeypatch.setattr(launch.subprocess, "run", lambda *a, **k: pytest.fail("proc identity must not need ps"))
    argv = ["/venv/bin/python", "-u", str(launch.ROOT / "tools/launch.py"), "_pipeline", str(spec_path)]
    proc_path.write_bytes(("\0".join(argv) + "\0").encode())
    assert launch.process_is_owned(23456, spec_path)
    argv[-1] += ".foreign"
    proc_path.write_bytes(("\0".join(argv) + "\0").encode())
    assert not launch.process_is_owned(23456, spec_path)


def test_rejected_concurrent_run_does_not_replace_active_pointer(spec, monkeypatch):
    spec_path = Path(spec["state_dir"]) / "run.json"
    launch.write_json(spec_path, spec)
    pointer = Path(spec["output_root"]) / "latest" / "qwen3-4b.json"
    launch.write_json(pointer, {"state_dir": "/active/run"})
    monkeypatch.setattr(launch, "head_only", lambda spec: None)
    def locked(*args):
        raise BlockingIOError("fixture lock is held")
    monkeypatch.setattr(launch.fcntl, "flock", locked)
    monkeypatch.setattr(launch, "pipeline", lambda spec: pytest.fail("concurrent run must not enter pipeline"))
    with pytest.raises(RuntimeError, match="A task from this project is already running on this four-node cluster"):
        launch.main(["_pipeline", str(spec_path)])
    assert json.loads(pointer.read_text()) == {"state_dir": "/active/run"}
    assert not (Path(spec["state_dir"]) / "accepted.json").exists()
    assert status(spec)["status"] == "failed"


@pytest.mark.parametrize("result", ["MATCH", "COPY"])
def test_sync_protocol_accepts_status_amid_stderr_warnings(spec, monkeypatch, result):
    spec["code_sync"] = True
    events = []
    def remote(spec, host, action, code, arguments=()):
        events.append((host, action))
        if action == "sync-probe":
            return f"startup warning\nAR_MM_SYNC_STATUS={result}\nloader warning\n"
        return ""
    monkeypatch.setattr(launch, "remote_python", remote)
    monkeypatch.setattr(launch, "source_files", lambda: [])
    monkeypatch.setattr(launch, "run_node_logged", lambda spec, host, action, *a, **kw: events.append((host, action)))
    launch.synchronize(spec)
    for host in spec["worker_hosts"]:
        assert (host, "sync-probe") in events
        assert ((host, "sync-copy") in events) is (result == "COPY")
        assert ((host, "sync-idle") in events) is (result == "COPY")


@pytest.mark.parametrize("output", ["", "MATCH", "AR_MM_SYNC_STATUS=unknown", "AR_MM_SYNC_STATUS=MATCH\nAR_MM_SYNC_STATUS=COPY"])
def test_sync_protocol_rejects_missing_or_ambiguous_status(spec, monkeypatch, output):
    spec["code_sync"] = True
    monkeypatch.setattr(launch, "remote_python", lambda *a, **kw: output)
    monkeypatch.setattr(launch, "run_node_logged", lambda *a, **kw: pytest.fail("must not copy on invalid probe"))
    with pytest.raises(RuntimeError, match="invalid source-sync probe result"):
        launch.synchronize(spec)
