"""Local transport lifecycle tests: never connect to a host or start an agent."""
import json
import os
from pathlib import Path
import stat
import subprocess

import pytest

from tools import ssh_transport as transport


@pytest.fixture
def spec(tmp_path, monkeypatch):
    key = tmp_path / "configured-key"
    key.write_text("fixture-sensitive-private-key-material")
    key.chmod(0o600)
    actual_mkdtemp = transport.tempfile.mkdtemp
    monkeypatch.setattr(transport.tempfile, "mkdtemp",
                        lambda prefix, dir: actual_mkdtemp(prefix=prefix, dir=tmp_path))
    monkeypatch.setattr(transport.SSHSession, "_can_prompt", staticmethod(lambda: False))
    monkeypatch.setenv("SSH_AUTH_SOCK", "/existing-user-agent")
    monkeypatch.setenv("SSH_AGENT_PID", "999")
    return {"ssh_user": "root", "ssh_key": str(key),
            "worker_hosts": ["10.0.0.2", "10.0.0.3", "10.0.0.4"],
            "state_dir": str(tmp_path / "state")}


def install_runner(monkeypatch, probe=None, agent_output=None, add_code=0):
    calls = []

    def run(args, **kwargs):
        calls.append((list(args), dict(kwargs)))
        if args[0] == "ssh":
            if "-O" in args:
                return subprocess.CompletedProcess(args, 0, "", "")
            return probe(args, kwargs) if probe else subprocess.CompletedProcess(args, 0, "", "")
        if args[0] == "ssh-agent":
            if "-k" in args:
                return subprocess.CompletedProcess(args, 0, "", "")
            socket = args[args.index("-a") + 1]
            output = agent_output if agent_output is not None else (
                f"SSH_AUTH_SOCK={socket}; export SSH_AUTH_SOCK;\n"
                "SSH_AGENT_PID=12345; export SSH_AGENT_PID;\necho Agent pid 12345;\n")
            return subprocess.CompletedProcess(args, 0, output, "")
        if args[0] == "ssh-add":
            return subprocess.CompletedProcess(args, add_code, "", "")
        pytest.fail("Unexpected command: " + repr(args))

    monkeypatch.setattr(transport.subprocess, "run", run)
    return calls


def test_options_keep_host_verification_batch_mode_and_identity_fallback(spec):
    args = transport.ssh_args(spec, "10.0.0.2")
    assert args[:2] == ["ssh", "-T"]
    assert args[-1] == "root@10.0.0.2"
    assert args[args.index("-i") + 1] == spec["ssh_key"]
    for option in ("BatchMode=yes", "ConnectTimeout=15", "ServerAliveInterval=15",
                   "ServerAliveCountMax=4", "StrictHostKeyChecking=accept-new", "LogLevel=ERROR"):
        assert option in args
    assert "IdentitiesOnly=yes" not in args
    assert "StrictHostKeyChecking=no" not in args


def test_wide_key_permissions_use_private_copy_and_preserve_original(spec, monkeypatch):
    original = Path(spec["ssh_key"])
    original.chmod(0o644)
    calls = install_runner(monkeypatch)
    session = transport.SSHSession(spec)
    record = session.prepare()
    copied = Path(record["effective_key"])
    directory = Path(record["temp_dir"])
    assert copied != original and copied.read_bytes() == original.read_bytes()
    assert stat.S_IMODE(copied.stat().st_mode) == 0o600
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(original.stat().st_mode) == 0o644
    assert not copied.is_relative_to(Path(spec["state_dir"]))
    assert all("ControlMaster=auto" in args for args, _ in calls)
    assert all(kwargs["stdin"] == subprocess.DEVNULL for _, kwargs in calls)
    assert all(kwargs["env"]["SSH_AUTH_SOCK"] == "/existing-user-agent" for _, kwargs in calls)
    for host in spec["worker_hosts"]:
        log = Path(spec["state_dir"]) / "nodes" / f"ssh-{host}.log"
        assert "exit=0" in log.read_text()
        assert original.read_text() not in log.read_text()
    assert original.read_text() not in json.dumps(record)
    session.cleanup()
    assert not directory.exists() and original.exists()
    assert not any(args[:2] == ["ssh-agent", "-k"] for args, _ in calls)


@pytest.mark.parametrize("mode", [0o400, 0o600])
def test_already_private_key_is_not_rewritten(spec, monkeypatch, mode):
    key = Path(spec["ssh_key"])
    key.chmod(mode)
    install_runner(monkeypatch)
    session = transport.SSHSession(spec)
    record = session.prepare()
    assert record["effective_key"] == str(key)
    session.cleanup()
    assert key.read_text() == "fixture-sensitive-private-key-material"
    assert stat.S_IMODE(key.stat().st_mode) == mode


def test_empty_key_uses_existing_agent_and_default_identities(spec, monkeypatch):
    spec["ssh_key"] = ""
    calls = install_runner(monkeypatch)
    session = transport.SSHSession(spec)
    assert session.environment()["SSH_AUTH_SOCK"] == "/existing-user-agent"
    session.prepare()
    assert all("-i" not in args for args, _ in calls)
    session.cleanup()
    assert not any(args[0] == "ssh-agent" for args, _ in calls)


def test_noninteractive_auth_failure_records_each_host_and_never_prompts(spec, monkeypatch):
    def denied(args, kwargs):
        return subprocess.CompletedProcess(args, 255, "", "Permission denied (publickey).")

    calls = install_runner(monkeypatch, probe=denied)
    session = transport.SSHSession(spec)
    with pytest.raises(RuntimeError, match="No interactive terminal") as caught:
        session.prepare(interactive=True)  # The actual TTY check still fails.
    assert all(host in str(caught.value) for host in spec["worker_hosts"])
    assert "ssh-add" in str(caught.value)
    assert not any(args[0] in ("ssh-agent", "ssh-add") for args, _ in calls)
    assert not Path(spec["_ssh"]["temp_dir"]).exists()
    assert len(list((Path(spec["state_dir"]) / "nodes").glob("ssh-*.log"))) == 3


def test_detached_mode_never_prompts_even_if_stdin_looks_interactive(spec, monkeypatch):
    monkeypatch.setattr(transport.SSHSession, "_can_prompt", staticmethod(lambda: True))
    calls = install_runner(monkeypatch, probe=lambda args, kwargs:
                           subprocess.CompletedProcess(args, 255, "", "Permission denied"))
    with pytest.raises(RuntimeError, match="ssh-add"):
        transport.SSHSession(spec).prepare(interactive=False)
    assert not any(args[0] in ("ssh-agent", "ssh-add") for args, _ in calls)


def test_private_agent_unlocks_once_and_detached_child_cleans_its_own_resources(spec, monkeypatch):
    monkeypatch.setattr(transport.SSHSession, "_can_prompt", staticmethod(lambda: True))

    def probe(args, kwargs):
        unlocked = kwargs["env"]["SSH_AUTH_SOCK"] != "/existing-user-agent"
        return subprocess.CompletedProcess(args, 0 if unlocked else 255, "", "" if unlocked else "Permission denied")

    calls = install_runner(monkeypatch, probe=probe)
    parent = transport.SSHSession(spec)
    record = parent.prepare(interactive=True)
    assert len([args for args, _ in calls if args[0] == "ssh-add"]) == 1
    assert record["agent_pid"] == 12345
    assert parent.environment()["SSH_AGENT_PID"] == "12345"
    assert os.environ["SSH_AGENT_PID"] == "999"
    first_log = Path(spec["state_dir"]) / "nodes" / "ssh-10.0.0.2.log"
    assert "exit=255" in first_log.read_text() and "exit=0" in first_log.read_text()
    # JSON round-trip models parent -> run.json -> detached coordinator.
    child_spec = json.loads(json.dumps(spec))
    child = transport.SSHSession(child_spec)
    assert child.environment()["SSH_AUTH_SOCK"] == record["agent_env"]["SSH_AUTH_SOCK"]
    child.cleanup()
    kills = [kwargs for args, kwargs in calls if args == ["ssh-agent", "-k"]]
    assert len(kills) == 1 and kills[0]["env"]["SSH_AGENT_PID"] == "12345"
    assert not Path(record["temp_dir"]).exists()
    before = len(calls)
    child.cleanup()
    assert len(calls) == before


def test_failed_key_unlock_cleans_private_agent_and_keeps_diagnostic(spec, monkeypatch):
    monkeypatch.setattr(transport.SSHSession, "_can_prompt", staticmethod(lambda: True))
    calls = install_runner(monkeypatch, probe=lambda args, kwargs:
                           subprocess.CompletedProcess(args, 255, "", "Permission denied"), add_code=1)
    with pytest.raises(RuntimeError, match="SSH 10.0.0.2.*ssh-add failed"):
        transport.SSHSession(spec).prepare(interactive=True)
    assert any(args == ["ssh-agent", "-k"] for args, _ in calls)
    assert not Path(spec["_ssh"]["temp_dir"]).exists()
    assert "Key unlock failed" in (Path(spec["state_dir"]) / "nodes" / "ssh-10.0.0.2.log").read_text()


def test_changed_host_key_is_reported_without_removing_known_hosts(spec, monkeypatch):
    calls = install_runner(monkeypatch, probe=lambda args, kwargs: subprocess.CompletedProcess(
        args, 255, "", "WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED!"))
    with pytest.raises(RuntimeError, match="verify the host identity"):
        transport.SSHSession(spec).prepare()
    assert all(args[0] == "ssh" for args, _ in calls)


def test_timeouts_are_logged_with_host_and_do_not_hide_original_stderr(spec, monkeypatch):
    def timeout(args, kwargs):
        raise subprocess.TimeoutExpired(args, 45, output=b"partial output", stderr=b"Connection timed out")

    install_runner(monkeypatch, probe=timeout)
    with pytest.raises(RuntimeError, match="10.0.0.2.*exit=timeout.*Connection timed out"):
        transport.SSHSession(spec).prepare()
    log = Path(spec["state_dir"]) / "nodes" / "ssh-10.0.0.2.log"
    assert "partial output" in log.read_text() and "Connection timed out" in log.read_text()


def test_unreadable_key_fails_before_any_connection_and_removes_private_directory(spec, monkeypatch):
    spec["ssh_key"] += ".missing"
    calls = install_runner(monkeypatch)
    with pytest.raises(RuntimeError, match="not readable"):
        transport.SSHSession(spec).prepare()
    assert not any(args[-1] == "true" for args, _ in calls)
    assert not Path(spec["_ssh"]["temp_dir"]).exists()


def test_cleanup_refuses_foreign_directory_and_never_kills_inherited_agent(spec, tmp_path, monkeypatch):
    directory = tmp_path / "ar-mm-ssh-foreign"
    directory.mkdir()
    (directory / ".owner").write_text("somebody-else")
    spec["_ssh"] = {"temp_dir": str(directory), "owner_token": "wrong-owner",
                    "agent_pid": 999, "agent_env": {"SSH_AGENT_PID": "999", "SSH_AUTH_SOCK": "/existing-user-agent"}}
    calls = install_runner(monkeypatch)
    assert transport.SSHSession(spec).cleanup() == []
    assert directory.exists() and calls == []


def test_cleanup_errors_do_not_mask_original_failure(spec, monkeypatch):
    install_runner(monkeypatch)
    session = transport.SSHSession(spec)
    session.prepare()

    def missing_ssh(*args, **kwargs):
        raise FileNotFoundError("ssh executable disappeared")

    monkeypatch.setattr(transport.subprocess, "run", missing_ssh)
    problems = session.cleanup()
    assert len(problems) == 3
    assert not Path(spec["_ssh"]["temp_dir"]).exists()


def test_agent_output_is_never_evaluated_as_shell_code(spec, monkeypatch):
    monkeypatch.setattr(transport.SSHSession, "_can_prompt", staticmethod(lambda: True))
    calls = install_runner(monkeypatch, probe=lambda args, kwargs:
                           subprocess.CompletedProcess(args, 255, "", "Permission denied"),
                           agent_output="SSH_AUTH_SOCK=$(touch /unexpected); export SSH_AUTH_SOCK;\n"
                                        "SSH_AGENT_PID=12345; export SSH_AGENT_PID;\n")
    monkeypatch.setattr(transport.os, "system", lambda *args: pytest.fail("must not evaluate agent output"))
    with pytest.raises(RuntimeError, match="unexpected ssh-agent output"):
        transport.SSHSession(spec).prepare(interactive=True)
    assert not any(args[0] == "ssh-add" for args, _ in calls)
    assert not Path(spec["_ssh"]["temp_dir"]).exists()
