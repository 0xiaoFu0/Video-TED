"""Local transport/cleanup checks: real bash, fake Python, no SSH/Ray/GPU work."""

import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

import pytest

from tools import launch


@pytest.fixture
def transport_spec(tmp_path, monkeypatch):
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(launch, "ROOT", project)
    monkeypatch.setattr(launch, "source_digest", lambda: "transport-fixture-source")
    monkeypatch.setattr(launch, "head_only", lambda spec: None)
    config = {
        "head_host": "10.0.0.1", "worker_hosts": ["10.0.0.2", "10.0.0.3", "10.0.0.4"],
        "ssh_user": "root", "ssh_key": "", "train_env_root": str(tmp_path / "venv"),
        "model_root": str(tmp_path / "models"), "model_paths": {},
        "data_root": str(tmp_path / "data"), "output_root": str(tmp_path / "outputs"),
        "gpus_per_node": 8, "ray_port": 6411, "nccl_port": 29551, "code_sync": True,
        "environment": {"NCCL_DEBUG": "WARN"},
    }
    python = Path(config["train_env_root"]) / "bin/python"
    python.parent.mkdir(parents=True)
    python.write_text("#!/bin/bash\nexit 0\n")
    python.chmod(0o755)
    return launch.build_spec(config, "qwen3-4b", "train", run_id="transport_fixture")


def fake_session(monkeypatch, spec, tmp_path, *, prepare_error=None, prepared=False):
    """Only this fake session's own resources may be released by the caller."""
    events = []
    owned = tmp_path / "owned-session"
    foreign = tmp_path / "foreign-session"
    foreign.mkdir()
    (foreign / "keep").write_text("another run owns this")
    if prepared:
        owned.mkdir()
        spec["_ssh"] = {"fixture_owner": spec["token"]}

    class Session:
        def __init__(self, received):
            assert received is spec
            events.append("construct")

        def environment(self):
            return dict(os.environ, AR_TRANSPORT_SESSION="own-session")

        def prepare(self, interactive=False):
            events.append(("prepare", interactive))
            owned.mkdir(exist_ok=True)
            spec["_ssh"] = {"fixture_owner": spec["token"]}
            if prepare_error is not None:
                raise prepare_error

        def cleanup(self):
            events.append("cleanup")
            if owned.exists():
                owned.rmdir()

    monkeypatch.setattr(launch, "SSHSession", Session)
    return events, owned, foreign


def install_fake_python(spec, tmp_path):
    # These names are literal filesystem names. An unquoted expansion would
    # run touch, alter the name, or break the shell before reaching Python.
    env_root = tmp_path / "venv space ' $(touch injected_env) \" tail"
    code = tmp_path / "project space ' $(touch injected_code) \" tail"
    (env_root / "bin").mkdir(parents=True)
    (env_root / "lib").mkdir()
    (code / "tools").mkdir(parents=True)
    spec.update(train_env_root=str(env_root), code_dir=str(code))
    python = env_root / "bin/python"
    python.write_text(
        "#!/bin/bash\nset -euo pipefail\n"
        '[[ "$PATH" == "$EXPECT_BIN":* ]] || { echo "PATH was not bootstrapped" >&2; exit 81; }\n'
        '[[ "$LD_LIBRARY_PATH" == "$EXPECT_LD" ]] || { echo "loader path was not bootstrapped" >&2; exit 82; }\n'
        '[[ "$PYTHONPATH" == "$EXPECT_CODE" ]] || { echo "PYTHONPATH was not bootstrapped" >&2; exit 83; }\n'
        + "exec " + shlex.quote(sys.executable) + ' "$@"\n'
    )
    python.chmod(0o755)
    (code / "tools/node_runtime.py").write_text(
        "import json, os, sys\n"
        "print(json.dumps({'argv': sys.argv[1:], 'spec': json.load(sys.stdin), "
        "'cwd': os.getcwd(), 'PATH': os.environ['PATH'], 'LD_LIBRARY_PATH': os.environ['LD_LIBRARY_PATH']}))\n"
    )
    return env_root, code


@pytest.mark.parametrize("inherited_library_path", [None, "/older/lib one:/older/lib'two"])
def test_real_bash_bootstraps_before_exec_and_preserves_stdin_json(
    transport_spec, tmp_path, inherited_library_path,
):
    spec = transport_spec
    env_root, code = install_fake_python(spec, tmp_path)
    spec["data_root"] = "data ' \" $(touch injected_json) `touch injected_backtick`\nAR_SPEC_JSON\n\u4e2d\u6587"
    spec["_ssh"] = {"effective_key": "/private/key", "owner_token": "must-stay-on-head"}
    env = dict(os.environ, EXPECT_BIN=str(env_root / "bin"), EXPECT_CODE=str(code))
    env.pop("LD_LIBRARY_PATH", None)
    if inherited_library_path is not None:
        env["LD_LIBRARY_PATH"] = inherited_library_path
    env["EXPECT_LD"] = str(env_root / "lib") + (":" + inherited_library_path if inherited_library_path else "")
    result = subprocess.run(["/bin/bash", "-s"], input=launch.node_script(spec, spec["head_host"], "check"),
                            capture_output=True, text=True, env=env, cwd=tmp_path, timeout=10)
    assert result.returncode == 0, result.stderr
    received = json.loads(result.stdout)
    assert received["spec"] == {key: value for key, value in spec.items() if key != "_ssh"}
    assert received["argv"] == ["check", "--spec-stdin", "--host", spec["head_host"]]
    assert received["cwd"] == str(code)
    assert received["PATH"].split(":")[0] == str(env_root / "bin")
    assert received["LD_LIBRARY_PATH"] == env["EXPECT_LD"]
    for name in ("injected_env", "injected_code", "injected_json", "injected_backtick"):
        assert not (tmp_path / name).exists()
        assert not (code / name).exists()


def test_bootstrap_connection_does_not_require_remote_code(transport_spec, tmp_path):
    env_root, code = install_fake_python(transport_spec, tmp_path)
    missing_code = code / "not-yet-synchronized"
    transport_spec["code_dir"] = str(missing_code)
    env = dict(os.environ, EXPECT_BIN=str(env_root / "bin"), EXPECT_CODE=str(missing_code),
               EXPECT_LD=str(env_root / "lib"))
    env.pop("LD_LIBRARY_PATH", None)
    command = [str(env_root / "bin/python"), "-B", "-c", "import os; print(os.getcwd())"]
    result = subprocess.run(["/bin/bash", "-s"],
                            input=launch.bootstrap(transport_spec, require_code=False) + shlex.join(command) + "\n",
                            capture_output=True, text=True, env=env, cwd=tmp_path, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/"
    assert not missing_code.exists()


def test_node_argv_keeps_script_and_spec_out_of_ssh_command(transport_spec, monkeypatch):
    spec = transport_spec
    host = spec["worker_hosts"][0]
    monkeypatch.setattr(launch, "ssh_args", lambda received, target: ["fixture-ssh", target])
    assert launch.node_command(spec, spec["head_host"], "check") == ["bash", "-s"]
    assert launch.node_command(spec, host, "check") == ["fixture-ssh", host, "bash -s"]
    assert spec["token"] not in " ".join(launch.node_command(spec, host, "check"))


def test_run_node_logged_success_keeps_stdout_and_stderr(transport_spec, tmp_path, monkeypatch, capsys):
    spec = transport_spec
    fake_session(monkeypatch, spec, tmp_path)
    output = launch.run_node_logged(spec, spec["head_host"], "check", ["/bin/bash", "-s"],
                                   script="printf 'node stdout\\n'; printf 'node stderr\\n' >&2\n")
    log = Path(spec["state_dir"]) / "nodes" / f"{spec['head_host']}.check.log"
    assert output == log.read_text() == "node stdout\nnode stderr\n"
    console = capsys.readouterr().out
    assert spec["head_host"] in console and "BEGIN" in console and "OK" in console


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_node_failure_logs_stderr_without_dumping_command_or_spec(
    transport_spec, tmp_path, monkeypatch, capsys, failure,
):
    spec = transport_spec
    fake_session(monkeypatch, spec, tmp_path)
    host = spec["worker_hosts"][1]
    command_marker = "private-full-command-must-not-be-reported"
    spec_marker = "private-json-spec-must-not-be-reported"
    script = (f"# {spec_marker}\n"
              "printf 'permission or loader diagnosis\\n' >&2\n"
              + ("exit 29\n" if failure == "exit" else "exec /bin/sleep 5\n"))
    with pytest.raises(RuntimeError) as caught:
        launch.run_node_logged(spec, host, "check", ["/bin/bash", "-s", "--", command_marker],
                               script=script, timeout=0.2 if failure == "timeout" else 5)
    log = Path(spec["state_dir"]) / "nodes" / f"{host}.check.log"
    message = str(caught.value)
    assert host in message and "[check]" in message and str(log) in message
    assert "permission or loader diagnosis" in message
    assert "permission or loader diagnosis" in log.read_text()
    assert ("exit=29" if failure == "exit" else "timed out after") in message
    combined = message + log.read_text() + capsys.readouterr().out
    assert command_marker not in combined
    assert spec_marker not in combined
    assert spec["token"] not in combined


def test_node_script_timeout_terminates_the_python_runtime(transport_spec, tmp_path, monkeypatch):
    """Killing a bash wrapper must not leave its Python preflight child running."""
    spec = transport_spec
    env_root, code = install_fake_python(spec, tmp_path)
    fake_session(monkeypatch, spec, tmp_path)
    pid_file = tmp_path / "runtime.pid"
    monkeypatch.setenv("EXPECT_BIN", str(env_root / "bin"))
    monkeypatch.setenv("EXPECT_LD", str(env_root / "lib"))
    monkeypatch.setenv("EXPECT_CODE", str(code))
    monkeypatch.setenv("AR_TEST_PID_FILE", str(pid_file))
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    (code / "tools/node_runtime.py").write_text(
        "import os, pathlib, sys, time\n"
        "pathlib.Path(os.environ['AR_TEST_PID_FILE']).write_text(str(os.getpid()))\n"
        "print('Python runtime entered', file=sys.stderr, flush=True)\n"
        "time.sleep(30)\n"
    )
    processes = []
    real_popen = subprocess.Popen
    def record_process(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(launch.subprocess, "Popen", record_process)
    try:
        with pytest.raises(RuntimeError, match="timed out after") as caught:
            launch.run_node_logged(
                spec, spec["head_host"], "check", launch.node_command(spec, spec["head_host"], "check"),
                script=launch.node_script(spec, spec["head_host"], "check"), timeout=1,
            )
        assert pid_file.is_file(), "Python must have started before the timeout"
        runtime_pid = int(pid_file.read_text())
        assert len(processes) == 1
        # exec must retain the launcher's PID through bash, the fake venv
        # executable, and Python, so subprocess.run kills the actual runtime.
        assert runtime_pid == processes[0].pid
        assert processes[0].returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(runtime_pid, 0)
        assert "Python runtime entered" in str(caught.value)
    finally:
        # If the regression returns, clean the orphan before reporting failure.
        if pid_file.is_file():
            runtime_pid = int(pid_file.read_text())
            if runtime_pid != os.getpid():
                try:
                    os.kill(runtime_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_start_ssh_failure_never_detaches_and_cleans_own_resources(
    transport_spec, tmp_path, monkeypatch, failure_type,
):
    spec = transport_spec
    events, owned, foreign = fake_session(monkeypatch, spec, tmp_path,
                                           prepare_error=failure_type("SSH authentication failed"))
    monkeypatch.setattr(launch.subprocess, "Popen", lambda *a, **kw: pytest.fail("failed SSH must never detach"))
    with pytest.raises(failure_type, match="SSH authentication failed"):
        launch.start(spec)
    assert events == ["construct", ("prepare", True), "cleanup"]
    state = Path(spec["state_dir"])
    assert json.loads((state / "status.json").read_text())["status"] == "failed"
    assert not Path(spec["shared_probe"]).exists()
    assert not (state / "process.json").exists()
    assert not (state / "run.json").exists()
    assert not owned.exists()
    assert (foreign / "keep").read_text() == "another run owns this"


def test_pipeline_lock_rejection_releases_session_and_probe_without_replacing_active_run(
    transport_spec, tmp_path, monkeypatch,
):
    spec = transport_spec
    state = Path(spec["state_dir"])
    state.mkdir(parents=True)
    Path(spec["shared_probe"]).write_text(spec["token"])
    events, owned, foreign = fake_session(monkeypatch, spec, tmp_path, prepared=True)
    pointer = Path(spec["output_root"]) / "latest" / f"{spec['model_key']}.json"
    launch.write_json(pointer, {"state_dir": "/other/active/run"})
    def lock_held(*args):
        raise BlockingIOError("another pipeline owns the lock")
    monkeypatch.setattr(launch.fcntl, "flock", lock_held)
    monkeypatch.setattr(launch, "pipeline", lambda value: pytest.fail("rejected launch must not run pipeline"))
    with pytest.raises(RuntimeError, match="A task from this project is already running on this four-node cluster"):
        launch.pipeline_entry(spec)
    assert events == ["construct", "cleanup"]
    assert not Path(spec["shared_probe"]).exists()
    assert not owned.exists() and (foreign / "keep").exists()
    assert json.loads(pointer.read_text()) == {"state_dir": "/other/active/run"}
    assert not (state / "accepted.json").exists()
    assert json.loads((state / "status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("failure", [None, "authentication", "python"])
def test_connect_only_checks_connections_and_always_releases_session(
    transport_spec, tmp_path, monkeypatch, capsys, failure,
):
    spec = transport_spec
    spec["mode"] = "connect"
    events, owned, foreign = fake_session(monkeypatch, spec, tmp_path,
        prepare_error=RuntimeError("SSH failed") if failure == "authentication" else None)
    def forbidden(*args, **kwargs):
        pytest.fail("connect must never synchronize sources, start Ray, or train")
    for name in ("synchronize", "prepare", "run_stage", "node", "pipeline", "start"):
        monkeypatch.setattr(launch, name, forbidden)
    monkeypatch.setattr(launch.subprocess, "Popen", forbidden)
    monkeypatch.setattr(launch, "ssh_args", lambda value, host: ["fixture-ssh", host])
    calls = []
    def diagnostic(received, host, action, command, **kwargs):
        assert received is spec and action == "python"
        script = kwargs["script"]
        assert "node_runtime.py" not in script
        assert "import sys,socket" in script
        assert "cd /\n" in script
        assert "--spec-stdin" not in script
        calls.append(host)
        if failure == "python" and host == spec["worker_hosts"][0]:
            raise RuntimeError("remote Python loader failed")
        return "host=fixture\npython=/fixture/python\n"
    monkeypatch.setattr(launch, "run_node_logged", diagnostic)
    if failure:
        with pytest.raises(RuntimeError, match="SSH failed|remote Python loader failed"):
            launch.connect(spec)
    else:
        launch.connect(spec)
        assert "connection checks passed" in capsys.readouterr().out
    expected_hosts = [] if failure == "authentication" else [spec["head_host"], *spec["worker_hosts"]]
    if failure == "python":
        expected_hosts = expected_hosts[:2]
    assert calls == expected_hosts
    assert events == ["construct", ("prepare", True), "cleanup"]
    assert not owned.exists() and (foreign / "keep").exists()
    assert not Path(spec["shared_probe"]).exists()
    assert not (Path(spec["state_dir"]) / "process.json").exists()
    assert not (Path(spec["output_root"]) / "latest").exists()
