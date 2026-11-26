#!/usr/bin/env python3
"""CPU-only, ground-truth-blind answer recovery for existing v6 runs.

Keep the original run and strict scores unchanged. Publish a separately named
scoring protocol; apply it to every model before comparing recovered scores.
Only standard-library and existing CPU evaluation modules are imported.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import re
import tempfile

from model_table import COLUMNS, split_statistics, table_values, training_note
from protocol import (EVAL_SPLITS, PROMPT_PROTOCOL, ROOT, atomic_json, digest,
                      load_progress, manifest_rows, official, official_scorer,
                      read_json, registry, summarize)

SCORING_PROTOCOL = "v6-answer-tags-or-exact-final-label-v1"
RULES = [
    "Use the original candidates for each split; never use the target or video path for parsing.",
    "Use the existing scorer's normalization: lowercase, ignore spaces/underscores/newlines/asterisks.",
    "Ignore closed think blocks; reject an unclosed think block.",
    "Accept complete case-insensitive answer tags only when every answer is the same candidate.",
    "Do not fall back past malformed, conflicting, empty or out-of-candidate answer tags.",
    "Without answer tags, accept only the last nonempty line of a normally stopped response when it exactly matches a candidate.",
    "Reject conflicting standalone candidate lines; never search for labels inside prose or use fuzzy/semantic matching.",
    "Keep all samples in the denominator, including unresolved, truncated and erroneous responses.",
]
SCORER = official_scorer()
ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.I | re.S)
ANSWER_MARKER = re.compile(r"<\s*/?\s*answer\b", re.I)
THINK = re.compile(r"<think>.*?</think>", re.I | re.S)
THINK_MARKER = re.compile(r"<\s*/?\s*think\b", re.I)


class CandidateParser:
    def __init__(self, actions):
        self.labels = {}
        for value in actions.values():
            label = value.split(":", 1)[0].strip()
            key = SCORER["normalize"](label)
            if not key or key in self.labels:
                raise ValueError("Empty or ambiguous normalized candidate: " + label)
            self.labels[key] = label

    def parse(self, text, finish_reason, inference_error=None):
        if inference_error:
            return "", "inference_error"
        if not isinstance(text, str):
            raise ValueError("inference_output must be a string")
        outside = THINK.sub("", text).strip()
        if THINK_MARKER.search(outside):
            return "", "unclosed_think"
        tags = ANSWER.findall(outside)
        markers = ANSWER_MARKER.findall(outside)
        if markers:
            if not tags or len(markers) != 2 * len(tags):
                return "", "malformed_answer_tag"
            names = [SCORER["normalize"](tag.strip()) for tag in tags]
            if any(name not in self.labels for name in names):
                return "", "tag_not_candidate"
            if len(set(names)) != 1:
                return "", "conflicting_answer_tags"
            return self.labels[names[0]], "answer_tag"
        if finish_reason != "stop":
            return "", "no_tag_nonstop"
        lines = [line.strip() for line in outside.splitlines() if line.strip()]
        if not lines:
            return "", "empty_output"
        key = SCORER["normalize"](lines[-1])
        if key not in self.labels:
            return "", "final_line_not_candidate"
        standalone = {SCORER["normalize"](line) for line in lines} & self.labels.keys()
        if len(standalone) != 1:
            return "", "conflicting_candidate_lines"
        return self.labels[key], "exact_final_line"


def normalized_answer(text):
    return SCORER["normalize"](SCORER["extract_answer"](text))


def recover_records(records, actions):
    parser = CandidateParser(actions)
    recovered, methods = [], Counter()
    gained = lost = recovered_labels = 0
    for row in records:
        # The parser has no access to solution, path, or the annotation row.
        label, method = parser.parse(row["inference_output"], row["finish_reason"], row["inference_error"])
        predict = "<answer>" + label + "</answer>" if label else ""
        target = normalized_answer(row["solution"])
        before = bool(target) and normalized_answer(row["predict"]) == target
        after = bool(target) and normalized_answer(predict) == target
        gained += int(after and not before)
        lost += int(before and not after)
        recovered_labels += int(bool(label) and not normalized_answer(row["predict"]))
        methods[method] += 1
        recovered.append({**row, "strict_predict": row["predict"], "predict": predict,
                          "answer_parse_method": method, "scoring_protocol": SCORING_PROTOCOL})
    return recovered, {"parse_methods": dict(methods), "newly_correct": gained,
                       "no_longer_correct": lost, "previously_empty_now_resolved": recovered_labels}


def csv_text(headers, rows):
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    return output.getvalue()


def rescore(run_dir, output_dir=None, allow_subset=False):
    run_dir = Path(run_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() if output_dir else run_dir / "answer_rescore_v1"
    if output_dir.exists():
        raise ValueError("Output already exists; keep it and choose another --output-dir: " + str(output_dir))
    summary_path = run_dir / "summary.json"
    source_summary = read_json(summary_path)
    if set(source_summary.get("metrics", {})) != set(EVAL_SPLITS):
        raise ValueError("Expected a merged v6 run containing all twelve splits")
    source_hashes = {str(summary_path): digest(summary_path)}
    specs = registry()["datasets"]
    old_metrics, new_metrics, audits, manifests = {}, {}, {}, {}
    identity = None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".answer-rescore-", dir=output_dir.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        for key in EVAL_SPLITS:
            manifest_path = run_dir / key / "run.json"
            result_path = run_dir / key / "results.json"
            source_hashes.update({str(p): digest(p) for p in (manifest_path, result_path)})
            manifest = read_json(manifest_path)
            spec = specs[key]
            if (manifest.get("benchmark") != key or manifest.get("prompt_protocol") != PROMPT_PROTOCOL
                    or manifest.get("inference_passes") != 1 or manifest.get("sharding")
                    or manifest.get("scoring_protocol")):
                raise ValueError("Expected an original merged single-pass v6 split: " + key)
            for field, path_key in (("annotations_sha256", "annotations"), ("classes_sha256", "classes")):
                if digest(ROOT / spec[path_key]) != manifest.get(field):
                    raise ValueError("Annotation/candidate hash mismatch: " + key)
            if identity is None:
                identity = manifest["model"]
            elif identity != manifest["model"]:
                raise ValueError("Different model identities across splits")
            expected, actions, full_count = manifest_rows(spec, manifest)
            records = load_progress(result_path, expected)
            strict = summarize(records, len(expected), full_count)
            if not strict["complete"] or strict["inference_errors"]:
                raise ValueError("Incomplete split or inference errors: " + key)
            if strict["subset_only"] and not allow_subset:
                raise ValueError("Subset run; use --allow-subset for diagnostics only: " + key)
            if strict != source_summary["metrics"][key]:
                raise ValueError("Stored summary disagrees with original predictions: " + key)
            extract = official(spec["entry"])["extract_answer"]
            if any(extract(r["inference_output"]).replace("\n", "").strip() != r["predict"] for r in records):
                raise ValueError("Stored predictions do not match the original v6 extractor: " + key)
            fixed, audit = recover_records(records, actions)
            metrics = summarize(fixed, len(expected), full_count)
            metrics["scoring_protocol"] = SCORING_PROTOCOL
            audit.update(expected=len(expected), strict_accuracy=strict["accuracy"],
                         label_accuracy=metrics["accuracy"], strict_invalid=strict["invalid_answers"],
                         unresolved=metrics["invalid_answers"], truncated=strict["truncated"])
            old_metrics[key], new_metrics[key], audits[key], manifests[key] = strict, metrics, audit, manifest
            with (stage / (key + ".answers.jsonl")).open("w", encoding="utf-8") as handle:
                for index, row in enumerate(fixed):
                    # Raw generation remains in the original results.json, addressed by index/path.
                    answer = {"index": index, **{k: row[k] for k in ("path", "solution", "strict_predict",
                        "predict", "finish_reason", "answer_parse_method")}}
                    handle.write(json.dumps(answer, ensure_ascii=False) + "\n")
            print(f"{key}: strict={100*strict['accuracy']:.2f}% -> label={100*metrics['accuracy']:.2f}%; "
                  f"invalid {strict['invalid_answers']} -> {metrics['invalid_answers']}; "
                  f"newly correct={audit['newly_correct']}", flush=True)
        # Refuse to publish a report if a source is still being changed.
        if any(digest(Path(path)) != checksum for path, checksum in source_hashes.items()):
            raise ValueError("Source files changed during rescoring; wait for inference to finish")
        subset = any(m["subset_only"] for m in new_metrics.values())
        report = {"schema_version": 1, "model_name": source_summary["model_name"], "model": identity,
                  "source_run": str(run_dir), "source_files_sha256": source_hashes,
                  "script_sha256": digest(Path(__file__).resolve()),
                  "prompt_protocol": PROMPT_PROTOCOL, "scoring_protocol": SCORING_PROTOCOL,
                  "created_at": datetime.now(timezone.utc).isoformat(), "rules": RULES,
                  "strict_metrics": old_metrics, "metrics": new_metrics, "audit": audits,
                  "three_split_statistics": split_statistics(new_metrics),
                  "training_audit": source_summary.get("training_audit", {}),
                  "original_manifests": manifests, "subset_only": subset}
        atomic_json(stage / "summary.json", report)
        note = training_note(report)
        headers = [COLUMNS[0], "Scoring protocol", "Subset only", *COLUMNS[1:]]
        new_row = [report["model_name"], SCORING_PROTOCOL, subset, *table_values(new_metrics), note]
        old_row = [report["model_name"], "original-v6-strict-answer-tags", subset, *table_values(old_metrics), note]
        (stage / "models.csv").write_text(csv_text(headers, [new_row]), encoding="utf-8")
        (stage / "models_strict.csv").write_text(csv_text(headers, [old_row]), encoding="utf-8")
        audit_keys = ["expected", "strict_accuracy", "label_accuracy", "strict_invalid", "unresolved",
                      "newly_correct", "no_longer_correct", "previously_empty_now_resolved", "truncated"]
        (stage / "audit.csv").write_text(csv_text(["benchmark", *audit_keys],
            [[k, *[audits[k][field] for field in audit_keys]] for k in EVAL_SPLITS]), encoding="utf-8")
        def md(row):
            return "| " + " | ".join(str(x).replace("|", "\\|") for x in row) + " |"
        (stage / "models.md").write_text(md(headers) + "\n" + md(["---"]*len(headers)) + "\n" + md(new_row) + "\n", encoding="utf-8")
        (stage / "README.txt").write_text(
            "CPU-only rescoring; original predictions and scores were not overwritten.\n"
            "models.csv: candidate-label accuracy under " + SCORING_PROTOCOL + ".\n"
            "models_strict.csv: original strict-tag scores. audit.csv: per-split differences.\n"
            "summary.json: rules, source hashes, manifests and both sets of metrics.\n"
            "*.answers.jsonl: every parsed decision; original raw text is in source_run/<split>/results.json.\n"
            "Use the SAME scoring protocol for Qwen2.5, RLSD and SFT before model comparisons.\n"
            "This is not a rerun or an unchanged reproduction of the upstream strict scoring protocol.\n",
            encoding="utf-8")
        stage.rename(output_dir)
    print("\n重评分完成（未调用模型/GPU），新表：" + str(output_dir / "models.csv"), flush=True)
    print("原始严格分数：" + str(output_dir / "models_strict.csv"), flush=True)
    print("比较其他模型前，请用同一脚本重评分它们的 v6 原始输出。", flush=True)
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", required=True, help="Completed v6 run directory containing summary.json")
    p.add_argument("--output-dir", help="New report directory; default: RUN_DIR/answer_rescore_v1")
    p.add_argument("--allow-subset", action="store_true", help="Permit explicitly marked diagnostic subset reports")
    args = p.parse_args(argv)
    rescore(args.run_dir, args.output_dir, args.allow_subset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
