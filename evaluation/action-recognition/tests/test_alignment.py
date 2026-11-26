import copy
import csv
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_protocol import audit
from model_table import COLUMNS, SPLITS, table_values, update_table
from protocol import EVAL_SPLITS, PROMPT_PROTOCOL, atomic_json, read_json
from test_parallel import table_fixture, setup_media
import parallel_eval as parallel


def test_official_byte_identity_split_sizes_and_annotation_conflicts():
    report = audit()
    assert report["verified_unchanged_official_files"] == 73
    assert report["total_evaluation_items"] == 45893
    assert report["unique_dataset_video_pairs"] == 40233
    assert [(s["rows"], s["classes"]) for s in report["splits"].values()] == [
        (780, 26), (750, 25), (2088, 51), (1695, 50), (17562, 87), (7215, 87), (1530, 51), (1530, 51), (1530, 51), (3783, 101), (3734, 101), (3696, 101)]
    # Do not hide the four swapped class labels in official HMDB annotations.
    assert len(report["official_annotation_conflicts"]["hmdb51"]) == 120
    assert report["split_relationships"]["ucf101"]["normalized_target_mismatches"] == 0


def test_harmonic_mean_uses_unrounded_accuracies_and_handles_zero():
    metrics = {k: {"accuracy": 0.0} for k in SPLITS}
    metrics["hmdb51-base"]["accuracy"] = 0.8
    metrics["hmdb51-novel"]["accuracy"] = 0.2
    metrics["ucf101-base"]["accuracy"] = 1 / 3
    metrics["ucf101-novel"]["accuracy"] = 2 / 7
    values = table_values(metrics)
    assert values[:3] == ["80.00", "20.00", "32.00"]
    assert values[5] == f"{100 * 2 * (1/3) * (2/7) / (1/3 + 2/7):.2f}"
    assert values[6:9] == ["0.00", "0.00", "0.00"]
    assert len(values) == len(COLUMNS) - 2 == 19


@pytest.mark.parametrize("old_protocol", ["videostar-v3-no-path", "videostar-v5-single-no-tools-no-path-3split"])
def test_full_table_contains_both_protocols_and_no_old_results(tmp_path, old_protocol):
    plan, metrics, directory = table_fixture(tmp_path)
    metrics["hmdb51-base"]["accuracy"] = .8
    metrics["hmdb51-novel"]["accuracy"] = .2
    update_table(tmp_path / "table", plan, metrics, directory)
    with (tmp_path / "table/models.csv").open() as f:
        row = list(csv.DictReader(f))[0]
    assert row["HMDB51 HM (%)"] == "32.00"
    assert row["HMDB51 split1 all 51 classes (%)"] == "100.00"
    bad = copy.deepcopy(plan)
    for s in bad["splits"].values():
        s["prompt_protocol"] = old_protocol
    with pytest.raises(ValueError, match="旧版"):
        update_table(tmp_path / "table", bad, metrics, directory)


def test_all_workers_use_same_twelve_splits_and_single_pass(tmp_path):
    args = parallel.parser().parse_args(setup_media(tmp_path, 2))
    plan, commands, _ = parallel.prepare(args)
    for command, _ in commands:
        index = command.index("--benchmarks") + 1
        assert command[index:index + 12] == list(EVAL_SPLITS)
    for signatures in [plan["splits"], *plan["workers"]]:
        for sig in signatures.values():
            assert sig["inference_passes"] == 1
            assert sig["tools"] is False and sig["class_explanations"] is False
            assert sig["prompt_protocol"] == PROMPT_PROTOCOL


def test_actual_train_overlap_stops_all_workers_before_gpu(tmp_path, monkeypatch):
    from protocol import load_dataset, registry
    argv = setup_media(tmp_path, 2)
    argv.remove("--allow-train-overlap")
    rows, _, _ = load_dataset(registry()["datasets"]["hmdb51-base"], 1)
    train = tmp_path / "train.jsonl"
    import json
    train.write_text(json.dumps({"videos": ["/training/" + rows[0]["path"]]}) + "\n")
    def forbidden(*args):
        raise AssertionError("No GPU action should occur")
    monkeypatch.setattr(parallel, "check_gpu_memory", forbidden)
    monkeypatch.setattr(parallel, "supervise", forbidden)
    with pytest.raises(ValueError, match="训练集重叠"):
        parallel.main([*argv, "--train-data", str(train)])
