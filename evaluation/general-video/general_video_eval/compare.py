"""Compare two completed runs only when data and evaluation settings agree.

Usage: python -m general_video_eval.compare BASE_RUN_DIR OURS_RUN_DIR
The existing manifests and summaries are read-only. New comparison.md/csv files
are written to OURS_RUN_DIR; mock comparisons require an explicit test flag.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


EXPECTED_COUNTS = {"mvbench": 4000, "videomme": 2700, "tempcompass": 1580, "seedbench": 3757}
EXPECTED_ACTION_COUNTS = {"seedbench": 1509}
DISPLAY_NAMES = {"mvbench": "MVBench", "videomme": "Video-MME", "tempcompass": "TempCompass-MCQ",
                 "seedbench": "SEED-Bench Video"}
METRICS = {"mvbench": (("Avg", "avg"), ("FA", "fa")),
           "videomme": (("Overall", "overall"), ("Action", "action_recognition")),
           "tempcompass": (("Overall", "overall"), ("Action", "action_recognition")),
           "seedbench": (("Overall", "overall"), ("Action", "action_recognition"))}


class ComparisonError(ValueError):
    """An incompatible, incomplete, or malformed result pair."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ComparisonError(message)


def _read_object(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    _require(isinstance(result, dict), f"{path}: expected a JSON object")
    return result


def _different_paths(left: Any, right: Any, prefix="") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(left.keys() | right.keys()):
            path = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(path)
            else:
                result.extend(_different_paths(left[key], right[key], path))
        return result
    return [] if left == right else [prefix]


def _count(value: Any, description: str, *, positive=False) -> int:
    _require(type(value) is int and value >= (1 if positive else 0),
             f"{description}: expected {'positive' if positive else 'nonnegative'} integer count")
    return value


def _percent(value: Any, description: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool)
             and math.isfinite(value) and 0 <= value <= 100,
             f"{description}: expected a finite percentage in [0, 100]")
    return float(value)


def _validate_run(directory: Path, manifest: dict, summary: dict, *, allow_mock: bool) -> None:
    label = str(directory)
    config = manifest.get("config")
    datasets = manifest.get("datasets")
    stats = summary.get("benchmarks")
    _require(isinstance(config, dict), f"{label}: missing manifest config")
    _require(isinstance(config.get("eval_profile"), str) and bool(config["eval_profile"])
             and isinstance(config.get("video_sampling"), str) and bool(config["video_sampling"]),
             f"{label}: missing saved evaluation profile/sampling provenance")
    _require(isinstance(config.get("source_sha256"), str)
             and re.fullmatch(r"[0-9a-fA-F]{64}", config["source_sha256"]) is not None,
             f"{label}: missing/invalid source_sha256")
    _require(isinstance(config.get("dependencies"), dict) and bool(config["dependencies"]),
             f"{label}: missing saved dependency versions")
    _require(isinstance(datasets, dict) and datasets, f"{label}: missing manifest datasets")
    _require(isinstance(stats, dict) and set(stats) == set(datasets),
             f"{label}: summary benchmark set differs from manifest datasets")
    _require(not set(stats) - set(EXPECTED_COUNTS), f"{label}: unsupported benchmark names")
    _require(config.get("benchmarks") is not None and set(config["benchmarks"]) == set(stats),
             f"{label}: config benchmark set differs from summary")
    _require(summary.get("scope") == datasets, f"{label}: summary scope/counts differ from manifest datasets")
    _require(summary.get("complete") is True, f"{label}: summary is incomplete")
    _require(not summary.get("errors") and summary.get("n_errors", 0) == 0,
             f"{label}: summary contains evaluation errors")
    _require(type(summary.get("is_mock")) is bool, f"{label}: summary must explicitly identify is_mock")
    is_mock = summary["is_mock"]
    _require(config.get("backend") in {"mock", "vllm"}, f"{label}: unsupported/missing backend")
    _require(summary.get("backend") == config["backend"] and is_mock == (config["backend"] == "mock"),
             f"{label}: inconsistent mock/backend labels")
    _require(not is_mock or allow_mock, f"{label}: MOCK results require --allow-mock and are pipeline tests only")
    _require(summary.get("experiment") == manifest.get("exp_name") and bool(manifest.get("exp_name")),
             f"{label}: summary experiment differs from manifest exp_name")
    _require(isinstance(manifest.get("dataset_sha256"), str)
             and re.fullmatch(r"[0-9a-fA-F]{64}", manifest["dataset_sha256"]) is not None,
             f"{label}: missing/invalid dataset_sha256")
    total = done = correct = 0
    for benchmark, bucket in stats.items():
        info = datasets[benchmark]
        _require(isinstance(info, dict) and isinstance(bucket, dict), f"{label}: invalid {benchmark} metadata")
        prefix = f"{label}/{benchmark}"
        n_total = _count(bucket.get("n_total"), f"{prefix} n_total", positive=True)
        n_done = _count(bucket.get("n_done"), f"{prefix} n_done")
        n_correct = _count(bucket.get("n_correct"), f"{prefix} n_correct")
        _require(n_done == n_total and n_correct <= n_total and bucket.get("complete") is True,
                 f"{prefix}: incomplete/inconsistent summary counts")
        _require(bucket.get("n_errors", 0) == 0, f"{prefix}: benchmark contains errors")
        selected = _count(info.get("selected_questions"), f"{prefix} selected_questions", positive=True)
        annotation = _count(info.get("annotation_questions"), f"{prefix} annotation_questions", positive=True)
        _require(selected == n_total and selected <= annotation,
                 f"{prefix}: summary counts differ from selected/annotation question counts")
        if not is_mock:
            _require(info.get("scope") == "full", f"{prefix}: real comparisons require scope=full")
            _require(n_total == annotation == info.get("official_expected_questions") == EXPECTED_COUNTS[benchmark],
                     f"{prefix}: full counts must equal the official {EXPECTED_COUNTS[benchmark]} questions")
        else:
            _require(info.get("scope") in {"full", "partial_or_nonstandard", "partial"},
                     f"{prefix}: unrecognized mock scope")
        action_total = _count(bucket.get("action_recognition_n_total"), f"{prefix} action count", positive=not is_mock)
        if not is_mock and benchmark in EXPECTED_ACTION_COUNTS:
            _require(action_total == EXPECTED_ACTION_COUNTS[benchmark],
                     f"{prefix}: action count must equal the official {EXPECTED_ACTION_COUNTS[benchmark]} questions")
        action_done = _count(bucket.get("action_recognition_n_done"), f"{prefix} action completed count")
        action_correct = _count(bucket.get("action_recognition_n_correct"), f"{prefix} action correct count")
        _require(action_total <= n_total and action_done == action_total and action_correct <= action_total,
                 f"{prefix}: incomplete/inconsistent action counts")
        overall = _percent(bucket.get("overall"), f"{prefix} overall")
        action = _percent(bucket.get("action_recognition"), f"{prefix} action_recognition") if action_total else None
        _require(action_total > 0 or bucket.get("action_recognition") is None,
                 f"{prefix}: an empty mock action category must have no score")
        _require(math.isclose(overall, 100 * n_correct / n_total, rel_tol=0, abs_tol=1e-7),
                 f"{prefix}: overall percentage disagrees with counts")
        _require(not action_total or math.isclose(action, 100 * action_correct / action_total, rel_tol=0, abs_tol=1e-7),
                 f"{prefix}: action percentage disagrees with counts")
        if benchmark == "mvbench":
            _percent(bucket.get("avg"), f"{prefix} avg")
            fa = _percent(bucket.get("fa"), f"{prefix} fa") if action_total else None
            _require((fa is None and bucket.get("fa") is None) if action is None else
                     math.isclose(fa, action, rel_tol=0, abs_tol=1e-7), f"{prefix}: FA disagrees with action score")
        total += n_total
        done += n_done
        correct += n_correct
    _require(summary.get("n_total") == total and summary.get("n_done") == done
             and summary.get("n_correct") == correct,
             f"{label}: summary total counts disagree with benchmark counts")


def compare_runs(base_run_dir, ours_run_dir, *, allow_mock=False) -> dict:
    """Read and validate two runs; do not write or modify any files."""
    directories = [Path(path).expanduser().resolve() for path in (base_run_dir, ours_run_dir)]
    _require(directories[0] != directories[1], "BASE_RUN_DIR and OURS_RUN_DIR must be different runs")
    pairs = [(_read_object(directory / "manifest.json"), _read_object(directory / "summary.json"))
             for directory in directories]
    for directory, (manifest, summary) in zip(directories, pairs):
        _validate_run(directory, manifest, summary, allow_mock=allow_mock)
    (base_manifest, base_summary), (ours_manifest, ours_summary) = pairs
    _require(base_summary["is_mock"] == ours_summary["is_mock"], "Cannot compare MOCK and real inference results")
    configs = [{key: value for key, value in manifest["config"].items()
                if key not in {"model_path", "model_identity"}} for manifest, _summary in pairs]
    differences = _different_paths(configs[0], configs[1])
    _require(not differences, "Evaluation config differs (only model_path/model_identity may differ): "
             + ", ".join(differences))
    _require(base_manifest["dataset_sha256"] == ours_manifest["dataset_sha256"],
             "Dataset mismatch: dataset_sha256 differs")
    differences = _different_paths(base_manifest["datasets"], ours_manifest["datasets"])
    _require(not differences, "Dataset scope/counts differ: " + ", ".join(differences))
    rows = []
    for benchmark in EXPECTED_COUNTS:
        if benchmark not in base_summary["benchmarks"]:
            continue
        base = base_summary["benchmarks"][benchmark]
        ours = ours_summary["benchmarks"][benchmark]
        for field in ("n_total", "n_done", "action_recognition_n_total", "action_recognition_n_done"):
            _require(base[field] == ours[field], f"{benchmark}: compared summary counts differ at {field}")
        for index, (metric, key) in enumerate(METRICS[benchmark]):
            base_value = float(base[key]) if base[key] is not None else None
            ours_value = float(ours[key]) if ours[key] is not None else None
            _require((base_value is None) == (ours_value is None), f"{benchmark}/{metric}: inconsistent missing scores")
            rows.append({"benchmark": DISPLAY_NAMES[benchmark], "metric": metric,
                         "questions": base["n_total"] if index == 0 else base["action_recognition_n_total"],
                         "base_percent": base_value, "ours_percent": ours_value,
                         "delta_pp": ours_value - base_value if base_value is not None else None,
                         "scope": base_manifest["datasets"][benchmark]["scope"],
                         "is_mock": base_summary["is_mock"]})
    return {"base_run_dir": str(directories[0]), "ours_run_dir": str(directories[1]),
            "base_experiment": base_manifest["exp_name"], "ours_experiment": ours_manifest["exp_name"],
            "base_model": base_manifest["config"].get("model_path", ""),
            "ours_model": ours_manifest["config"].get("model_path", ""),
            "dataset_sha256": base_manifest["dataset_sha256"], "config": configs[0],
            "is_mock": base_summary["is_mock"], "rows": rows}


def render_markdown(comparison: dict) -> str:
    config = comparison["config"]
    title = "MOCK COMPARISON — pipeline verification only" if comparison["is_mock"] else "Video benchmark comparison"
    budgets = ", ".join(f"{DISPLAY_NAMES.get(name, name)}={count}" for name, count in
                        config.get("max_frames_by_benchmark", {}).items())
    lines = [title, "", f"Base: `{comparison['base_experiment']}` (`{comparison['base_model']}`)",
             f"Ours: `{comparison['ours_experiment']}` (`{comparison['ours_model']}`)", "",
             f"Profile: `{config.get('eval_profile', 'unspecified')}`; sampling: `{config.get('video_sampling', 'unspecified')}`; "
             f"FPS={config.get('fps', 'unspecified')}; min frames={config.get('min_frames', 'unspecified')}; frame caps: {budgets or 'unspecified'}.",
             f"Pixels/frame: {config.get('min_pixels', 'unspecified')}..{config.get('max_pixels', 'unspecified')}; "
             f"THINK_MODE={config.get('think_mode', 'unspecified')}; N={config.get('n', 'unspecified')}; "
             f"temperature={config.get('temperature', 'unspecified')}; max new tokens={config.get('max_new_tokens', 'unspecified')}.",
             f"Protocol: `{config.get('protocol', 'unspecified')}`", "",
             "| Benchmark | Metric | Questions | Scope | Base (%) | Ours (%) | Delta (pp) |",
             "| --- | --- | ---: | --- | ---: | ---: | ---: |"]
    for row in comparison["rows"]:
        base = "—" if row["base_percent"] is None else f"{row['base_percent']:.2f}"
        ours = "—" if row["ours_percent"] is None else f"{row['ours_percent']:.2f}"
        delta = "—" if row["delta_pp"] is None else f"{row['delta_pp']:+.2f}"
        lines.append(f"| {row['benchmark']} | {row['metric']} | {row['questions']} | {row['scope']} | "
                     f"{base} | {ours} | {delta} |")
    lines += ["", "Delta is Ours minus Base in percentage points. FA is MVBench Fine-grained Action.",
              *(["SEED-Bench Video Action is the Action Recognition task (question type 10)."]
                if "seedbench" in config.get("benchmarks", []) else []),
              "Both runs are complete; question counts include invalid predictions as incorrect.",
              "All saved evaluation settings, dependencies, source code identity and dataset hashes match, except model identity.",
              f"Dataset SHA-256: `{comparison['dataset_sha256']}`", ""]
    return "\n".join(lines)


def render_csv(comparison: dict) -> str:
    stream = io.StringIO(newline="")
    config = comparison["config"]
    provenance_fields = ["eval_profile", "video_sampling", "fps", "min_frames", "max_frames", "min_pixels", "max_pixels"]
    fields = ["benchmark", "metric", "questions", "scope", "base_percent", "ours_percent", "delta_pp", "is_mock",
              *provenance_fields, "base_model", "ours_model"]
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    benchmark_names = {value: key for key, value in DISPLAY_NAMES.items()}
    for row in comparison["rows"]:
        metadata = {key: config.get(key, "") for key in provenance_fields}
        metadata["max_frames"] = config.get("max_frames_by_benchmark", {}).get(benchmark_names[row["benchmark"]], "")
        writer.writerow({**row, **metadata, "base_model": comparison["base_model"], "ours_model": comparison["ours_model"],
                         **{key: "" if row[key] is None else f"{row[key]:.6f}"
                            for key in ("base_percent", "ours_percent", "delta_pp")}})
    return stream.getvalue()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_run_dir")
    parser.add_argument("ours_run_dir")
    parser.add_argument("--allow-mock", action="store_true", help="Allow two completed mock runs; outputs remain explicitly labeled MOCK")
    args = parser.parse_args(argv)
    try:
        comparison = compare_runs(args.base_run_dir, args.ours_run_dir, allow_mock=args.allow_mock)
        markdown = render_markdown(comparison)
        csv_text = render_csv(comparison)
        destination = Path(comparison["ours_run_dir"])
        (destination / "comparison.md").write_text(markdown, encoding="utf-8")
        (destination / "comparison.csv").write_text(csv_text, encoding="utf-8")
        print(markdown, end="")
        print(f"\nSaved: {destination / 'comparison.md'}\nSaved: {destination / 'comparison.csv'}")
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: comparison refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
