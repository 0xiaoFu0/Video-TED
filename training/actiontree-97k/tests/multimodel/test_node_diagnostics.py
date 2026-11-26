"""CPU tests for remote node input, phase diagnostics, and failure reporting."""

import base64
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import node_runtime


@pytest.fixture
def spec(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    for name in ("train.rlsd.think.jsonl", "train.sft.think.jsonl", "manifest.json"):
        (data / name).write_text("fixture\n")
    probe = tmp_path / "shared_probe"
    probe.write_text("owned-token")
    return {
        "head_host": "10.0.0.1", "worker_hosts": ["10.0.0.2", "10.0.0.3", "10.0.0.4"],
        "train_env_root": str(tmp_path / "venv"), "gpus_per_node": 8, "environment": {},
        "source_digest": "fixture-digest", "model_path": str(tmp_path / "model"),
        "model_key": "qwen3-8b", "data_root": str(data), "shared_probe": str(probe),
        "token": "owned-token",
    }


@pytest.fixture
def passing_checks(spec, monkeypatch):
    calls = []
    monkeypatch.setattr(node_runtime.subprocess, "check_output", lambda *a, **k: spec["head_host"])
    monkeypatch.setattr(node_runtime, "source_digest", lambda: spec["source_digest"])
    monkeypatch.setattr(node_runtime, "assert_idle", lambda: calls.append("idle"))
    versions = {"torch": "2.8.0+cu128", "torchvision": "0.23.0", "transformers": "4.57.3",
                "vllm": "0.11.0", "ray": "2.56.1", "qwen-vl-utils": "0.0.14", "flash-attn": "2.8.3"}
    monkeypatch.setattr(node_runtime.importlib.metadata, "version", lambda name: versions[name])
    torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True, device_count=lambda: 8))

    def import_module(name):
        calls.append("import:" + name)
        return torch if name == "torch" else SimpleNamespace()

    monkeypatch.setattr(node_runtime, "_import_runtime_module", import_module)

    def validate_model(path, key, **kwargs):
        calls.append(("model", path, key, kwargs))
        return {"model_key": key, "processor_checked": kwargs["check_processor"]}

    monkeypatch.setitem(sys.modules, "preflight_model", SimpleNamespace(validate_model=validate_model))
    return calls


def test_successful_node_check_reports_every_phase_with_host(spec, passing_checks, capsys):
    node_runtime.check_node(spec, spec["head_host"], {})
    lines = capsys.readouterr().out.splitlines()
    phases = ["identity", "source", "idle", "versions", "imports", "cuda", "model", "data", "shared-storage"]
    expected = [f"[{spec['head_host']}] check {phase}: {state}" for phase in phases for state in ("BEGIN", "OK")]
    assert [line for line in lines if "] check " in line] == expected
    assert any(sys.executable in line and "python=" in line for line in lines)
    report = json.loads(lines[-1])
    assert report["host"] == spec["head_host"]
    assert report["versions"]["torch"] == "2.8.0+cu128"
    assert report["model"]["processor_checked"] is True
    assert passing_checks[-1] == ("model", spec["model_path"], "qwen3-8b", {"check_processor": True})


@pytest.mark.parametrize("phase", ["source", "idle", "data", "shared-storage"])
def test_failed_phase_is_visible_and_later_phases_do_not_run(spec, passing_checks, monkeypatch, capsys, phase):
    if phase == "source":
        monkeypatch.setattr(node_runtime, "source_digest", lambda: "wrong-digest")
    elif phase == "idle":
        def busy():
            raise RuntimeError("GPU busy fixture")
        monkeypatch.setattr(node_runtime, "assert_idle", busy)
    elif phase == "data":
        (Path(spec["data_root"]) / "manifest.json").unlink()
    else:
        Path(spec["shared_probe"]).write_text("another-token")
    with pytest.raises(RuntimeError):
        node_runtime.check_node(spec, spec["head_host"], {})
    lines = capsys.readouterr().out.splitlines()
    phase_lines = [line for line in lines if "] check " in line]
    assert phase_lines[-1] == f"[{spec['head_host']}] check {phase}: BEGIN"
    assert f"[{spec['head_host']}] check {phase}: OK" not in lines


def test_missing_package_names_distribution_and_selected_interpreter(spec, passing_checks, monkeypatch, capsys):
    normal_version = node_runtime.importlib.metadata.version

    def missing_flash(name):
        if name == "flash-attn":
            raise node_runtime.importlib.metadata.PackageNotFoundError(name)
        return normal_version(name)

    monkeypatch.setattr(node_runtime.importlib.metadata, "version", missing_flash)
    with pytest.raises(RuntimeError) as error:
        node_runtime.check_node(spec, spec["head_host"], {})
    assert "flash-attn" in str(error.value)
    assert "not installed" in str(error.value)
    assert sys.executable in str(error.value)
    output = capsys.readouterr().out
    assert "check versions: BEGIN" in output
    assert "check imports: BEGIN" not in output


def test_wrong_version_reports_actual_expected_and_interpreter(spec, passing_checks, monkeypatch):
    normal_version = node_runtime.importlib.metadata.version
    monkeypatch.setattr(node_runtime.importlib.metadata, "version",
                        lambda name: "4.56.0" if name == "transformers" else normal_version(name))
    with pytest.raises(RuntimeError) as error:
        node_runtime.check_node(spec, spec["head_host"], {})
    assert all(part in str(error.value) for part in ("transformers=4.56.0", "4.57.3", sys.executable))


@pytest.mark.parametrize("module_name", ["torch", "flash_attn", "vllm"])
def test_import_failure_keeps_module_executable_and_original_loader_error(monkeypatch, module_name):
    def broken_import(name):
        raise ImportError("libfixture_cuda.so: cannot open shared object file")

    monkeypatch.setattr(node_runtime.importlib, "import_module", broken_import)
    with pytest.raises(RuntimeError) as error:
        node_runtime._import_runtime_module(module_name)
    assert all(part in str(error.value) for part in (module_name, sys.executable, "ImportError", "libfixture_cuda.so"))
    assert isinstance(error.value.__cause__, ImportError)


@pytest.mark.parametrize("existing", [None, "/cuda/lib64:/custom/lib"])
def test_runtime_library_path_prepend_preserves_existing_environment(spec, monkeypatch, existing):
    monkeypatch.setenv("EXISTING_APPLICATION_SETTING", "keep-me")
    if existing is None:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    else:
        monkeypatch.setenv("LD_LIBRARY_PATH", existing)
    env = node_runtime.runtime_env(spec, spec["head_host"])
    expected = str(Path(spec["train_env_root"]) / "lib") + (":" + existing if existing else "")
    assert env["LD_LIBRARY_PATH"] == expected
    assert env["EXISTING_APPLICATION_SETTING"] == "keep-me"
    assert env["PATH"].startswith(str(Path(spec["train_env_root"]) / "bin") + ":")


def cli_input(monkeypatch, spec, action="check", transport="stdin"):
    args = ["node_runtime.py", action, "--host", spec["head_host"]]
    if transport == "stdin":
        args += ["--spec-stdin"]
        monkeypatch.setattr(node_runtime.sys, "stdin", io.StringIO(json.dumps(spec)))
    else:
        args += ["--spec", base64.urlsafe_b64encode(json.dumps(spec).encode()).decode()]
    monkeypatch.setattr(node_runtime.sys, "argv", args)
    # main intentionally replaces its process environment; use a private mapping
    # so these in-process tests cannot erase the pytest process environment.
    monkeypatch.setattr(node_runtime.os, "environ", {"ORIGINAL_SETTING": "preserved"})


@pytest.mark.parametrize("transport", ["stdin", "legacy"])
@pytest.mark.parametrize("action", ["check", "start", "stop"])
def test_main_accepts_new_stdin_and_legacy_spec_inputs(spec, monkeypatch, transport, action):
    cli_input(monkeypatch, spec, action, transport)
    calls = []
    monkeypatch.setattr(node_runtime, "check_node", lambda *args: calls.append(("check", *args)))
    monkeypatch.setattr(node_runtime, "ray_action", lambda *args: calls.append(args))
    assert node_runtime.main() == 0
    assert len(calls) == 1
    called_action, actual_spec, host, env = calls[0]
    assert (called_action, actual_spec, host) == (action, spec, spec["head_host"])
    assert env["ORIGINAL_SETTING"] == "preserved"
    assert env["LD_LIBRARY_PATH"] == str(Path(spec["train_env_root"]) / "lib")


def test_main_reports_host_action_error_type_message_and_traceback(spec, monkeypatch, capsys):
    cli_input(monkeypatch, spec)

    def failure(*args):
        raise RuntimeError("worker model directory is missing")

    monkeypatch.setattr(node_runtime, "check_node", failure)
    assert node_runtime.main() == 1
    error = capsys.readouterr().err
    assert error.startswith(f"[{spec['head_host']}] check FAILED: RuntimeError: worker model directory is missing\n")
    assert "Traceback (most recent call last):" in error
    assert "--spec" not in error


def test_main_preserves_keyboard_interrupt(spec, monkeypatch, capsys):
    cli_input(monkeypatch, spec, action="start")

    def interrupted(*args):
        raise KeyboardInterrupt("startup interrupted fixture")

    monkeypatch.setattr(node_runtime, "ray_action", interrupted)
    with pytest.raises(KeyboardInterrupt, match="startup interrupted fixture"):
        node_runtime.main()
    assert "FAILED" not in capsys.readouterr().err


@pytest.mark.parametrize("payload", ["[]", "not-json"])
def test_malformed_stdin_fails_without_dispatch(spec, monkeypatch, capsys, payload):
    cli_input(monkeypatch, spec)
    monkeypatch.setattr(node_runtime.sys, "stdin", io.StringIO(payload))
    monkeypatch.setattr(node_runtime, "check_node", lambda *args: pytest.fail("invalid spec must not run checks"))
    assert node_runtime.main() == 1
    assert f"[{spec['head_host']}] check FAILED:" in capsys.readouterr().err


@pytest.mark.parametrize("arguments", [[], ["--spec", "e30=", "--spec-stdin"]])
def test_cli_requires_exactly_one_spec_transport(monkeypatch, arguments):
    monkeypatch.setattr(node_runtime.sys, "argv", ["node_runtime.py", "check", "--host", "10.0.0.1", *arguments])
    with pytest.raises(SystemExit) as error:
        node_runtime.main()
    assert error.value.code == 2


def test_real_cli_returns_exit_one_for_invalid_stdin_and_prints_concise_failure():
    script = Path(node_runtime.__file__)
    result = subprocess.run([sys.executable, str(script), "check", "--spec-stdin", "--host", "10.0.0.2"],
                            input="[]", text=True, capture_output=True, timeout=10)
    assert result.returncode == 1
    assert result.stderr.startswith("[10.0.0.2] check FAILED: ValueError: Run specification must be a JSON object\n")
    assert "Traceback (most recent call last):" in result.stderr
