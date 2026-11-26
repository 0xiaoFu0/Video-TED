"""Three-split reporting uses three independent denominators and retains overlap notes."""
import copy
import csv
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import parallel_eval as parallel
from model_table import SPLITS, split_statistics, table_values, update_table
from protocol import load_dataset, read_json, registry
from test_parallel import setup_media, table_fixture


def test_three_split_mean_is_not_pooled_or_deduplicated():
    metrics = {key: {"accuracy": 0.0} for key in SPLITS}
    for split, n, accuracy in [(1, 3783, 0.0), (2, 3734, .5), (3, 3696, 1.0)]:
        metrics[f"ucf101-split{split}"] = {"accuracy": accuracy, "expected": n}
    stats = split_statistics(metrics)["ucf101"]
    assert stats["mean_accuracy"] == .5
    assert stats["std_accuracy"] == pytest.approx(math.sqrt(1 / 6))
    assert stats["std_ddof"] == 0
    pooled = (3734 * .5 + 3696) / (3783 + 3734 + 3696)
    assert stats["mean_accuracy"] != pooled
    assert table_values(metrics)[14:19] == ["0.00", "50.00", "100.00", "50.00", "40.82"]


def test_three_split_statistics_do_not_accept_missing_split():
    metrics = {key: {"accuracy": .5} for key in SPLITS}
    del metrics["hmdb51-split3"]
    with pytest.raises(KeyError):
        split_statistics(metrics)


def test_allow_overlap_reaches_every_worker_and_is_saved(tmp_path):
    argv = setup_media(tmp_path, 2)
    args = parallel.parser().parse_args([*argv, "--model-role", "ar-trained"])
    plan, commands, _ = parallel.prepare(args)
    assert len(plan["splits"]) == 12
    assert len(plan["workers"]) == 8
    assert all("--allow-train-overlap" in cmd for cmd, _ in commands)
    for worker in plan["workers"]:
        assert all(spec["allow_train_overlap"] for spec in worker.values())
    counts = plan["training_audit"]["counts"]
    assert counts["hmdb51-split2"]["reference_ar_overlap_full_split"] == 1340
    assert counts["hmdb51-split3"]["reference_ar_overlap_full_split"] == 1311
    assert plan["training_audit"]["model_role"] == "ar-trained"


def test_table_has_per_split_mean_std_and_model_specific_overlap_note(tmp_path):
    plan, metrics, directory = table_fixture(tmp_path)
    plan["training_audit"]["model_role"] = "base"
    dest = tmp_path / "table"
    update_table(dest, plan, metrics, directory)
    trained = copy.deepcopy(plan)
    trained["model_name"] = "AR-step292"
    trained["training_audit"]["model_role"] = "ar-trained"
    update_table(dest, trained, metrics, directory)
    with (dest / "models.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2 and len(rows[0]) == 21
    for ds in ("HMDB51", "UCF101"):
        assert rows[0][ds + " 3-split mean (%)"] == "100.00"
        assert rows[0][ds + " 3-split std (pp, ddof=0)"] == "0.00"
        assert rows[0][ds + " split2 (%)"] == "100.00"
        assert rows[0][ds + " split3 (%)"] == "100.00"
    assert "Original model" in rows[0]["Training overlap note"]
    assert "1340" in rows[1]["Training overlap note"] and "1311" in rows[1]["Training overlap note"]
    assert "includes seen videos" in rows[1]["Training overlap note"]
    assert "three_split_statistics" in read_json(dest / "models.json")["rows"][1]


def test_incomplete_third_split_cannot_enter_table(tmp_path):
    plan, metrics, directory = table_fixture(tmp_path)
    metrics["ucf101-split3"]["completed"] -= 1
    with pytest.raises(ValueError, match="未完整完成"):
        update_table(tmp_path / "table", plan, metrics, directory)
    assert not (tmp_path / "table/models.csv").exists()


def test_all_three_splits_use_official_annotations_without_deduplication():
    assert registry()["groups"]["hmdb51"] == [f"hmdb51-split{i}" for i in (1, 2, 3)]
    assert registry()["groups"]["ucf101"] == [f"ucf101-split{i}" for i in (1, 2, 3)]
    for ds, expected in [("hmdb51", [1530, 1530, 1530]), ("ucf101", [3783, 3734, 3696])]:
        sizes = [load_dataset(registry()["datasets"][f"{ds}-split{i}"])[2] for i in (1, 2, 3)]
        assert sizes == expected
