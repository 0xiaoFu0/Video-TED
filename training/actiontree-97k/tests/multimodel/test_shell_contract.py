"""Execute the real Bash chain with recording CPU substitutes for Python/hostname.

These tests verify the final argv delivered to the trainer across all supported
profiles, including override precedence and shell failure propagation. No model,
Ray cluster, distributed worker, or GPU process is created.
"""

import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

import pytest

from tools import launch


ROOT = Path(__file__).resolve().parents[2]
ENTRY = ROOT / "examples/action_recognition/run_97430.sh"
RECORDER = r'''
import json
import os
from pathlib import Path
import sys

name = Path(sys.argv[0]).name
args = sys.argv[1:]
keys = ["MODEL_KEY", "MODEL_PATH", "MERGED_DATA_ROOT", "TRAIN_DATA", "SFT_DATA",
        "OUTPUT_ROOT", "AR_PREFLIGHT_REPORT", "RUN_NAME", "RAY_ADDRESS"]
with Path(os.environ["AR_TEST_RECORD"]).open("a") as stream:
    stream.write(json.dumps({"program": name, "argv": args,
                            "env": {key: os.environ.get(key) for key in keys}}) + "\n")
if name == "hostname":
    if args != ["-I"]:
        raise SystemExit(98)
    print(os.environ["HEAD_HOST"])
    raise SystemExit(0)
if name != "python" or len(args) < 2 or args[0] != "-m":
    raise SystemExit(98)
module = args[1]
if module == os.environ.get("AR_TEST_FAIL_MODULE"):
    print("CPU fixture: requested failure at " + module, file=sys.stderr)
    raise SystemExit(37)
if module == "verl.ar.check_merged_97430_data":
    if "--check-report" in args:
        print(os.environ.get("AR_TEST_REFERENCE_LENGTH", "2048"))
elif module not in {"verl.ar.check_cluster", "verl.ar.prepare_data", "verl.trainer.opsd_main"}:
    print("CPU fixture: unexpected Python module " + module, file=sys.stderr)
    raise SystemExit(98)
'''


@pytest.fixture
def shell_fixture(tmp_path, monkeypatch):
    environment = tmp_path / "mock training env"
    binaries = environment / "bin"
    binaries.mkdir(parents=True)
    for name in ("python", "hostname", "ray"):
        executable = binaries / name
        executable.write_text(f"#!{sys.executable}\n" + RECORDER)
        executable.chmod(0o755)
    (binaries / "activate").write_text(
        f"export VIRTUAL_ENV={shlex.quote(str(environment))}\n"
        f"export PATH={shlex.quote(str(binaries))}:\"$PATH\"\n"
    )
    config = launch.load_config(ROOT / "configs/cluster.json")
    config.update(train_env_root=str(environment), model_root=str(tmp_path / "model assets"),
                  data_root=str(tmp_path / "97k data"), output_root=str(tmp_path / "training outputs"),
                  code_sync=False, model_paths={})
    record_path = tmp_path / "executed_commands.jsonl"
    monkeypatch.setattr(launch, "source_digest", lambda: "shell-contract-fixture")
    monkeypatch.delenv("MODEL_PATH", raising=False)
    monkeypatch.delenv("MODEL_ROOT", raising=False)

    def run(key="qwen3-4b", mode="train", *, extra_args=(), overrides=None, run_id=None):
        spec = launch.build_spec(config, key, mode, run_id=run_id or f"shell_{key}_{mode}")
        env = launch.stage_env(spec)
        env.pop("AR_FAST_OVERRIDES", None)  # only the explicit fast tests opt in
        env["AR_TEST_RECORD"] = str(record_path)
        if overrides:
            env.update(overrides)
        if record_path.exists():
            record_path.unlink()
        result = subprocess.run(["bash", str(ENTRY), mode, *extra_args], env=env,
                                cwd=tmp_path, text=True, capture_output=True, timeout=20)
        records = [json.loads(line) for line in record_path.read_text().splitlines()] if record_path.exists() else []
        return result, records, spec

    return run


def module_records(records, module):
    return [record for record in records if record["argv"][:2] == ["-m", module]]


def final_overrides(trainer_record):
    # OmegaConf's CLI merge applies repeated keys in argv order; the last wins.
    return dict(arg.split("=", 1) for arg in trainer_record["argv"][2:] if "=" in arg)


@pytest.mark.parametrize("model_key", list(launch.MODELS))
@pytest.mark.parametrize("mode", ["smoke", "train"])
def test_actual_shell_chain_preserves_each_model_training_contract(shell_fixture, model_key, mode):
    result, records, spec = shell_fixture(model_key, mode)
    assert result.returncode == 0, result.stdout + result.stderr
    modules = [record["argv"][1] for record in records if record["program"] == "python"]
    assert modules == ["verl.ar.check_merged_97430_data", "verl.ar.check_cluster",
                       "verl.ar.prepare_data", "verl.trainer.opsd_main"]
    trainer = module_records(records, "verl.trainer.opsd_main")
    assert len(trainer) == 1
    values = final_overrides(trainer[0])
    expected = {
        "worker.actor.model.model_path": spec["model_path"],
        "data.train_files": str(Path(spec["data_root"]) / "train.rlsd.think.jsonl"),
        "trainer.experiment_name": spec["run_id"], "trainer.save_checkpoint_path": spec["run_dir"],
        "trainer.nnodes": "4", "trainer.n_gpus_per_node": "8",
        "data.rollout_batch_size": "32", "worker.actor.global_batch_size": "32",
        "data.video_max_frames": "16", "data.max_response_length": "768", "worker.rollout.n": "4",
        "worker.actor.optim.lr": "1e-6", "trainer.total_epochs": "1",
        "opsd.ar_alignment_mode": "mesd", "opsd.ar_teacher_update_mode": "ema",
        "opsd.ar_teacher_update_rate": "0.01", "opsd.ar_teacher_update_interval": "1",
        "opsd.ar_teacher_disable_after_step": "1000", "opsd.ar_teacher_max_prompt_length": "18432",
        "opsd.rlsd_lambda": "0.5", "opsd.rlsd_lambda_warmup_steps": "0", "opsd.rlsd_lambda_decay_steps": "1000",
        "opsd.ar_reward_mode": "protocol", "opsd.ar_reference_coef": "0.1",
        "opsd.ar_reference_max_length": "2048", "data.truncation": "left", "worker.actor.loss_avg_mode": "seq",
        "trainer.find_last_checkpoint": "false", "trainer.load_checkpoint_path": "null",
    }
    for key, value in expected.items():
        assert values[key] == value, (model_key, mode, key, values.get(key))
    assert json.loads(values["worker.reward.reward_function_kwargs"]) == {"reward_mode": "protocol"}
    if mode == "smoke":
        assert values["trainer.max_steps"] == "2"
        assert values["trainer.save_freq"] == "-1"
    else:
        assert "trainer.max_steps" not in values
        assert values["trainer.save_freq"] == "100"
        assert values["trainer.save_limit"] == "30"
        assert values["trainer.save_model_only"] == "false"
    preflight = module_records(records, "verl.ar.check_merged_97430_data")[0]
    assert preflight["argv"][2:] == ["--data-root", spec["data_root"], "--model", spec["model_path"],
                                    "--report", spec["report"], "--check-report"]
    assert Path(spec["report"]).parts[-3] == model_key
    assert trainer[0]["env"]["AR_PREFLIGHT_REPORT"] == spec["report"]
    assert trainer[0]["env"]["SFT_DATA"] == str(Path(spec["data_root"]) / "train.sft.think.jsonl")
    assert trainer[0]["env"]["RAY_ADDRESS"] == f"{spec['head_host']}:{spec['ray_port']}"
    assert "97430" in result.stdout
    assert "reference length 2048" in result.stdout
    assert (Path(spec["model_output"]) / "logs" / (spec["run_id"] + ".log")).is_file()


def test_fixed_mesd_arguments_override_conflicting_user_arguments_in_real_shell(shell_fixture):
    result, records, _ = shell_fixture(extra_args=[
        "opsd.ar_alignment_mode=other", "opsd.ar_teacher_update_rate=0.9",
        "opsd.ar_teacher_disable_after_step=5", "opsd.rlsd_lambda_decay_steps=5",
        "opsd.ar_reference_coef=0.9", "opsd.ar_reward_mode=wrong",
        'worker.reward.reward_function_kwargs={"reward_mode":"wrong"}',
    ])
    assert result.returncode == 0, result.stdout + result.stderr
    values = final_overrides(module_records(records, "verl.trainer.opsd_main")[0])
    assert values["opsd.ar_alignment_mode"] == "mesd"
    assert values["opsd.ar_teacher_update_rate"] == "0.01"
    assert values["opsd.ar_teacher_disable_after_step"] == "1000"
    assert values["opsd.rlsd_lambda_decay_steps"] == "1000"
    assert values["opsd.ar_reference_coef"] == "0.1"
    assert values["opsd.ar_reward_mode"] == "protocol"
    assert json.loads(values["worker.reward.reward_function_kwargs"]) == {"reward_mode": "protocol"}


@pytest.mark.parametrize("mode,failed_module", [
    ("prepare", "verl.ar.check_merged_97430_data"),
    ("smoke", "verl.ar.check_merged_97430_data"),
    ("train", "verl.ar.check_merged_97430_data"),
    ("smoke", "verl.ar.prepare_data"),
    ("train", "verl.ar.prepare_data"),
    ("train", "verl.ar.check_cluster"),
])
def test_actual_shell_preparation_failure_never_reaches_trainer(shell_fixture, mode, failed_module):
    result, records, _ = shell_fixture(mode=mode, overrides={"AR_TEST_FAIL_MODULE": failed_module})
    assert result.returncode == 37, result.stdout + result.stderr
    assert module_records(records, failed_module)
    assert not module_records(records, "verl.trainer.opsd_main")
    if failed_module == "verl.ar.check_merged_97430_data":
        assert not module_records(records, "verl.ar.check_cluster")


@pytest.mark.parametrize("reference_length", ["", "0", "-1", "2048 extra"])
def test_malformed_reference_budget_stops_shell_before_cluster_or_training(shell_fixture, reference_length):
    result, records, _ = shell_fixture(overrides={"AR_TEST_REFERENCE_LENGTH": reference_length})
    assert result.returncode != 0
    assert "reference-length check returned an invalid result" in result.stderr
    assert not module_records(records, "verl.ar.check_cluster")
    assert not module_records(records, "verl.trainer.opsd_main")


def test_existing_run_name_is_not_reused_by_real_shell(shell_fixture):
    first, records, spec = shell_fixture(run_id="repeat_forbidden")
    assert first.returncode == 0
    assert len(module_records(records, "verl.trainer.opsd_main")) == 1
    second, records, _ = shell_fixture(run_id="repeat_forbidden")
    assert second.returncode != 0
    assert "RUN_NAME" in second.stderr
    assert not module_records(records, "verl.trainer.opsd_main")


def fast_overrides(**overrides):
    """Execute the real wrapper with a CPU recorder in place of launch Python."""
    with tempfile.TemporaryDirectory(prefix="ar-fast-test-") as directory:
        recorder = Path(directory) / "record-launch-python"
        recorder.write_text(
            f"#!{sys.executable}\n"
            "import os\n"
            "print(os.environ['AR_FAST_OVERRIDES'], end='')\n"
        )
        recorder.chmod(0o755)
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("FAST_") and key != "AR_FAST_OVERRIDES"}
        env.update(overrides)
        env["AR_LAUNCH_PYTHON"] = str(recorder)
        result = subprocess.run(["bash", str(ROOT / "run_fast.sh"), "models"],
                                cwd=ROOT, env=env, text=True, capture_output=True)
        assert result.returncode == 0, result.stderr
        return result.stdout


def test_run_fast_defaults_pack_each_gpu_and_shard_within_node():
    assert fast_overrides().split() == [
        "worker.actor.micro_batch_size_per_device_for_update=4",
        "worker.actor.micro_batch_size_per_device_for_experience=4",
        "worker.actor.fsdp.fsdp_size=8", "worker.ref.fsdp.fsdp_size=8"]
    assert fast_overrides(FAST_HSDP="0").split() == [
        "worker.actor.micro_batch_size_per_device_for_update=4",
        "worker.actor.micro_batch_size_per_device_for_experience=4"]
    assert "worker.rollout.gpu_memory_utilization=0.6" in fast_overrides(FAST_VLLM_MEM="0.6").split()
    for bad in ({"FAST_HSDP": "5"}, {"FAST_MICRO": "3"}, {"FAST_VLLM_MEM": "1.5"}):
        result = subprocess.run(["bash", "run_fast.sh", "models"], cwd=ROOT, env={**dict(os.environ), **bad},
                                text=True, capture_output=True)
        assert result.returncode == 2, (bad, result.stderr)


@pytest.mark.parametrize("model_key", ["qwen3-4b", "qwen25-7b"])
@pytest.mark.parametrize("mode", ["smoke", "train"])
def test_fast_overrides_sit_between_batch_contract_and_fixed_alignment(shell_fixture, mode, model_key):
    result, records, _ = shell_fixture(key=model_key, mode=mode, overrides={"AR_FAST_OVERRIDES": fast_overrides()},
                                       extra_args=["worker.actor.fsdp.fsdp_size=16",
                                                   "opsd.ar_teacher_disable_after_step=5"])
    assert result.returncode == 0, result.stdout + result.stderr
    trainer = module_records(records, "verl.trainer.opsd_main")[0]
    values = final_overrides(trainer)
    assert values["worker.actor.micro_batch_size_per_device_for_update"] == "4"
    assert values["worker.actor.micro_batch_size_per_device_for_experience"] == "4"
    assert values["worker.ref.fsdp.fsdp_size"] == "8"
    assert values["worker.actor.fsdp.fsdp_size"] == "16"  # explicit user argument still wins
    assert values["opsd.ar_teacher_disable_after_step"] == "1000"  # fixed contract still last
    assert values["data.rollout_batch_size"] == "32" and values["worker.actor.global_batch_size"] == "32"
    keys = [arg.split("=", 1)[0] for arg in trainer["argv"][2:] if "=" in arg]
    assert (keys.index("data.rollout_batch_size") < keys.index("worker.actor.micro_batch_size_per_device_for_update")
            < keys.index("opsd.ar_alignment_mode"))


def test_default_chain_has_no_fast_keys(shell_fixture):
    _, records, _ = shell_fixture()
    values = final_overrides(module_records(records, "verl.trainer.opsd_main")[0])
    assert not any(key.startswith(("worker.actor.micro_batch", "worker.actor.fsdp", "worker.ref.fsdp")) for key in values)
