"""CPU tests: real subprocess orchestration with an explicit fake backend.

No synthetic prediction from these tests is packaged as a benchmark result.
"""
import copy
import csv
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import evaluate
import parallel_eval as parallel
from model_table import SPLITS, check_table_compatibility, file_lock, update_table
from protocol import (ROOT, atomic_json, load_dataset, manifest_rows, partition_rows,
                      read_json, registry, summarize)


@pytest.mark.parametrize("key", SPLITS)
def test_full_eight_way_partition_has_no_missing_or_repeated_videos(key):
    rows, _, _ = load_dataset(registry()["datasets"][key])
    shards = [partition_rows(rows, rank, 8) for rank in range(8)]
    paths = [r["path"] for shard in shards for r in shard]
    assert len(paths) == len(set(paths)) == len(rows)
    assert set(paths) == {r["path"] for r in rows}
    assert max(map(len, shards)) - min(map(len, shards)) <= 1


def setup_media(tmp_path, limit=9):
    args = ["--model", str(tmp_path / "model"), "--model-name", "TEST-ONLY",
            "--run-name", "test_run", "--output-root", str(tmp_path / "outputs"),
            "--table-dir", str(tmp_path / "table"), "--limit", str(limit), "--allow-train-overlap"]
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_text("CPU TEST FIXTURE, NOT REAL WEIGHTS")
    for dataset in ("hmdb51", "ucf101", "ssv2"):
        root = tmp_path / dataset
        root.mkdir()
        args.extend(["--" + dataset + "-root", str(root)])
    for key in SPLITS:
        spec = registry()["datasets"][key]
        rows, _, _ = load_dataset(spec, limit)
        for row in rows:
            path = tmp_path / spec["dataset"] / row["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return args


def fake_worker(path):
    # Exercise the actual evaluate.main() in eight OS processes, bypassing only
    # the GPU backend; errors and predictions still use real persistence code.
    path.write_text("""
import os, sys
from pathlib import Path
sys.path.insert(0, ROOT_PATH)
import evaluate
from protocol import load_dataset, registry
class CPUFakeBackend:
    def __init__(self, args):
        self.answers = {}
        self.wrong = set()
        from protocol import EVAL_SPLITS
        for key in EVAL_SPLITS:
            rows, _, _ = load_dataset(registry()['datasets'][key], args.limit)
            self.wrong.add(Path(rows[0]['path']).name)
            self.answers.update((Path(r['path']).name, r['solution']) for r in rows)
    def generate(self, batch):
        marker = os.environ.get('FAKE_WORKER_FAIL_ONCE')
        if marker and os.environ['CUDA_VISIBLE_DEVICES'] == '0' and not Path(marker).exists():
            Path(marker).touch()
            raise RuntimeError('TEST ONLY: one decode failure')
        outputs = []
        for msg in batch:
            if 'three independent tools' in msg[0]['content'][1]['text']:
                text = '<pose_estimation>no</pose_estimation><person_detection>no</person_detection><noun_explanation>no</noun_explanation>'
            else:
                name = Path(msg[0]['content'][0]['video']).name
                text = '<answer>TEST-WRONG</answer>' if name in self.wrong else self.answers[name]
            outputs.append((text, 'stop'))
        return outputs
evaluate.Backend = CPUFakeBackend
raise SystemExit(evaluate.main())
""".replace("ROOT_PATH", repr(str(ROOT))))


def cpu_launcher(monkeypatch, tmp_path):
    worker = tmp_path / "cpu_fake_worker.py"
    fake_worker(worker)
    real_prepare = parallel.prepare
    def prepare(args):
        plan, commands, directory = real_prepare(args)
        for command, _ in commands:
            command[1] = str(worker)
        return plan, commands, directory
    monkeypatch.setattr(parallel, "prepare", prepare)
    monkeypatch.setattr(parallel, "check_gpu_memory", lambda *a: None)


@pytest.mark.parametrize("limit", [2, 9])
def test_real_eight_process_run_merges_uneven_or_empty_shards_and_resumes(tmp_path, monkeypatch, limit):
    args = setup_media(tmp_path, limit)
    cpu_launcher(monkeypatch, tmp_path)
    assert parallel.main(args) == 0
    directory = tmp_path / "outputs/test_run"
    for key in SPLITS:
        metrics = read_json(directory / key / "metrics.json")
        assert metrics["completed"] == limit
        assert metrics["correct"] == limit - 1
        assert metrics["accuracy"] == (limit - 1) / limit  # weighted by videos, not mean of shard accuracies
        assert metrics["inference_errors"] == 0
        original, _, _ = load_dataset(registry()["datasets"][key], limit)
        result = read_json(directory / key / "results.json")
        assert [r["path"] for r in result] == [r["path"] for r in original]
        assert not (tmp_path / "table/models.csv").exists()  # subsets cannot enter full table
    before = (directory / "workers/shard_00/hmdb51-split1/results.json").stat().st_mtime_ns
    assert parallel.main([*args, "--resume"]) == 0
    assert (directory / "workers/shard_00/hmdb51-split1/results.json").stat().st_mtime_ns == before
    assert parallel.main([*args, "--merge-only"]) == 0
    assert evaluate.main(["score", str(directory / "workers/shard_00/hmdb51-split1/results.json")]) == 0


def test_runtime_failure_then_resume_retries_and_leaves_no_fake_row(tmp_path, monkeypatch):
    args = setup_media(tmp_path, 9)
    cpu_launcher(monkeypatch, tmp_path)
    monkeypatch.setenv("FAKE_WORKER_FAIL_ONCE", str(tmp_path / "failed_once"))
    with pytest.raises(RuntimeError, match="worker 0"):
        parallel.main(args)
    assert not (tmp_path / "table/models.csv").exists()
    assert parallel.main([*args, "--resume"]) == 0
    summary = read_json(tmp_path / "outputs/test_run/summary.json")
    assert all(m["complete"] and not m["inference_errors"]
               for m in summary["metrics"].values())


def materialize_plan(tmp_path):
    args = parallel.parser().parse_args(setup_media(tmp_path, 9))
    plan, _, directory = parallel.prepare(args)
    for rank in range(8):
        worker = directory / "workers" / f"shard_{rank:02d}"
        atomic_json(worker / "environment.json", plan["environment"])
        for key in SPLITS:
            signature = plan["workers"][rank][key]
            rows, _, _ = manifest_rows(registry()["datasets"][key], signature)
            records = [{**r, "inference_error": None, "finish_reason": "stop",
                        "inference_output": r["solution"], "predict": r["solution"]} for r in rows]
            atomic_json(worker / key / "run.json", signature)
            atomic_json(worker / key / "results.json", records)
    return plan, directory


@pytest.mark.parametrize("damage", ["missing", "duplicate", "label", "signature", "inference_error", "environment"])
def test_merge_rejects_partial_wrong_or_failed_worker(tmp_path, damage):
    plan, directory = materialize_plan(tmp_path)
    worker = directory / "workers/shard_00"
    path = worker / "hmdb51-split1/results.json"
    records = read_json(path)
    if damage == "missing":
        atomic_json(path, records[:-1])
    elif damage == "duplicate":
        atomic_json(path, [records[0], records[0]])
    elif damage == "label":
        records[0]["solution"] = "<answer>WRONG LABEL</answer>"
        atomic_json(path, records)
    elif damage == "signature":
        manifest = worker / "hmdb51-split1/run.json"
        value = read_json(manifest)
        value["seed"] += 1
        atomic_json(manifest, value)
    elif damage == "environment":
        atomic_json(worker / "environment.json", {})
    else:
        records[0]["inference_error"] = "failed"
        atomic_json(path, records)
    with pytest.raises(ValueError):
        parallel.merge_run(directory, plan)


def table_fixture(tmp_path):
    plan, directory = materialize_plan(tmp_path)
    metrics = parallel.merge_run(directory, plan)
    # Deliberately synthetic full-dataset metadata solely for table unit tests.
    for key in SPLITS:
        plan["splits"][key]["official_split_size"] = 9
        metrics[key]["official_split_size"] = 9
        metrics[key]["subset_only"] = False
    return plan, metrics, directory


def test_table_adds_second_model_preserves_first_and_upserts_same_model(tmp_path):
    plan, metrics, directory = table_fixture(tmp_path)
    destination = tmp_path / "table"
    update_table(destination, plan, metrics, directory)
    baseline = read_json(destination / "models.json")["rows"][0]
    other = copy.deepcopy(plan)
    other["model_name"] = "AR-RLSD-step292"
    for key in SPLITS:
        other["splits"][key]["model"]["path"] = "other-model"
    update_table(destination, other, metrics, directory)
    table = read_json(destination / "models.json")
    assert len(table["rows"]) == 2 and table["rows"][0] == baseline
    update_table(destination, other, metrics, directory)
    assert len(read_json(destination / "models.json")["rows"]) == 2
    with (destination / "models.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert [r["Model"] for r in rows] == ["TEST-ONLY", "AR-RLSD-step292"]
    assert rows[0]["HMDB51 split1 all 51 classes (%)"] == "100.00"
    assert (destination / "models.md").read_text().count("AR-RLSD-step292") == 1


@pytest.mark.parametrize("damage", ["subset", "partial", "runtime", "settings", "weights"])
def test_table_rejects_unhealthy_or_incompatible_row_without_changing_csv(tmp_path, damage):
    plan, metrics, directory = table_fixture(tmp_path)
    destination = tmp_path / "table"
    update_table(destination, plan, metrics, directory)
    before = (destination / "models.csv").read_bytes()
    if damage == "subset": metrics[SPLITS[0]]["subset_only"] = True
    elif damage == "partial": metrics[SPLITS[0]]["completed"] -= 1
    elif damage == "runtime": metrics[SPLITS[0]]["inference_errors"] = 1
    elif damage == "settings": plan["splits"][SPLITS[0]]["seed"] += 1
    else: plan["splits"][SPLITS[0]]["model"]["path"] = "different-weights"
    with pytest.raises(ValueError):
        update_table(destination, plan, metrics, directory)
    assert (destination / "models.csv").read_bytes() == before


def test_gpu_preflight_rejects_occupied_or_aliased_cards(monkeypatch):
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: "0, GPU-aaa, 81251, 420\n1, GPU-bbb, 81251, 81000\n")
    with pytest.raises(ValueError, match="GPU 0"):
        parallel.check_gpu_memory(["0", "1"], 0.9)
    parallel.check_gpu_memory(["1"], 0.9)
    with pytest.raises(ValueError, match="重复"):
        parallel.check_gpu_memory(["1", "GPU-bbb"], 0.9)


def test_failed_worker_stops_other_launched_process_groups(tmp_path):
    commands = [([sys.executable, "-c", "raise SystemExit(7)"], "0"),
                ([sys.executable, "-c", "import time; time.sleep(60)"], "1")]
    plan = {"num_shards": 2, "splits": {key: {"selected": 1} for key in SPLITS}}
    before = time.monotonic()
    with pytest.raises(RuntimeError, match="exit=7"):
        parallel.supervise(commands, tmp_path, plan)
    assert time.monotonic() - before < 10


def test_run_lock_rejects_duplicate_coordinator(tmp_path):
    path = tmp_path / "lock"
    with file_lock(path):
        with pytest.raises(ValueError, match="另一个"):
            with file_lock(path):
                pass


def test_v2_run_cannot_resume_or_merge_with_v3(tmp_path, monkeypatch):
    argv = setup_media(tmp_path, 2)
    plan, _, directory = parallel.prepare(parallel.parser().parse_args(argv))
    old_plan = copy.deepcopy(plan)
    for specs in [old_plan["splits"], *old_plan["workers"]]:
        for signature in specs.values():
            signature.pop("prompt_protocol")
    atomic_json(directory / "launch.json", old_plan)
    def forbidden(*a, **k):
        raise AssertionError("Must reject old manifests before any GPU operation")
    monkeypatch.setattr(parallel, "check_gpu_memory", forbidden)
    monkeypatch.setattr(parallel, "supervise", forbidden)
    for flag in ("--resume", "--merge-only"):
        with pytest.raises(ValueError, match="不能恢复"):
            parallel.main([*argv, flag])


def test_v2_table_rejected_before_gpu_and_v2_plan_cannot_publish(tmp_path, monkeypatch):
    plan, metrics, directory = table_fixture(tmp_path)
    destination = tmp_path / "table"
    update_table(destination, plan, metrics, directory)
    old_table = read_json(destination / "models.json")
    for signature in old_table["contract"]["splits"].values():
        signature.pop("prompt_protocol")
    atomic_json(destination / "models.json", old_table)
    before = (destination / "models.csv").read_bytes()
    with pytest.raises(ValueError, match="旧版"):
        check_table_compatibility(destination, plan)
    monkeypatch.setattr(parallel, "prepare", lambda args: (plan, [], directory))
    def forbidden(*a, **k):
        raise AssertionError("Must reject old table before any GPU operation")
    monkeypatch.setattr(parallel, "check_gpu_memory", forbidden)
    monkeypatch.setattr(parallel, "supervise", forbidden)
    with pytest.raises(ValueError, match="旧版"):
        parallel.main(["--model-name", "TEST", "--run-name", "NEW", "--table-dir", str(destination)])
    plan["splits"][SPLITS[0]].pop("prompt_protocol")
    with pytest.raises(ValueError, match="旧版/未知"):
        update_table(tmp_path / "new_table", plan, metrics, directory)
    assert (destination / "models.csv").read_bytes() == before


def test_resume_after_interruption_during_error_recovery(tmp_path):
    from types import SimpleNamespace
    spec = registry()["datasets"]["hmdb51-split1"]
    rows, actions, count = load_dataset(spec, 3)
    records = [{**rows[0], "predict": rows[0]["solution"], "inference_error": None,
                "inference_output": rows[0]["solution"], "finish_reason": "stop"}]
    atomic_json(tmp_path / "results.json", records)
    class Fake:
        def generate(self, batch):
            return [(rows[0]["solution"], "stop") for _ in batch]
    metrics = evaluate.run_split(Fake(), SimpleNamespace(batch_size=2, retry_errors=True),
                                 ("hmdb51-split1", spec, rows, actions, count, tmp_path, tmp_path), tmp_path, {})
    assert metrics["complete"] and metrics["correct"] == 3
    assert not metrics["inference_errors"]
    assert not (tmp_path / "first_turn.json").exists()
