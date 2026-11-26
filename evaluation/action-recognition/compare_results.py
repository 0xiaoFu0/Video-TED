#!/usr/bin/env python3
"""Compare completed runs; recompute scores from predictions and preserve coverage."""
import argparse
import csv
import json
from pathlib import Path

from protocol import ROOT, atomic_json, digest, manifest_rows, load_progress, read_json, registry, summarize


def compare(name, output_root):
    root = Path(output_root)
    results = {}
    for model in ("base", "trained"):
        directory = root / (name + "_" + model)
        results[model] = {}
        for manifest_path in sorted(directory.glob("*/run.json")):
            manifest = read_json(manifest_path)
            key = manifest["benchmark"]
            spec = registry()["datasets"][key]
            if digest(ROOT / spec["annotations"]) != manifest["annotations_sha256"]:
                raise ValueError("Annotation changed: " + key)
            data, _, full_count = manifest_rows(spec, manifest)
            predictions = load_progress(manifest_path.parent / "results.json", data)
            metrics = summarize(predictions, len(data), full_count)
            results[model][key] = (manifest, metrics)
    rows = []
    okay = True
    keys = sorted(set(results["base"]) | set(results["trained"]))
    if not keys:
        raise ValueError("No evaluation outputs found for " + name)
    for key in keys:
        if key not in results["base"] or key not in results["trained"]:
            raise ValueError("Missing comparison model for " + key)
        left, lm = results["base"][key]
        right, rm = results["trained"][key]
        if {k: v for k, v in left.items() if k != "model"} != {k: v for k, v in right.items() if k != "model"}:
            raise ValueError("Model runs used different data or inference settings: " + key)
        healthy = all(m["complete"] and not m["inference_errors"] for m in (lm, rm))
        okay &= healthy
        rows.append({"benchmark": key, "samples": left["selected"], "subset_only": lm["subset_only"],
                     "base_accuracy": lm["accuracy"], "trained_accuracy": rm["accuracy"],
                     "delta_percentage_points": 100 * (rm["accuracy"] - lm["accuracy"]),
                     "base_completed": lm["completed"], "trained_completed": rm["completed"],
                     "base_errors": lm["inference_errors"],
                     "trained_errors": rm["inference_errors"],
                     "base_invalid_answers": lm["invalid_answers"], "trained_invalid_answers": rm["invalid_answers"],
                     "complete_without_runtime_errors": healthy})
    atomic_json(root / (name + "_comparison.json"), rows)
    with (root / (name + "_comparison.csv")).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0 if okay else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    # run_compare forwards common evaluation arguments; only output-root matters here.
    args, _ = parser.parse_known_args()
    raise SystemExit(compare(args.name, args.output_root))
