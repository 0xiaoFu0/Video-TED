"""Deterministic multiple-choice parsing and coverage-aware benchmark scoring.

An unparseable completion is never converted to A. Missing predictions remain
in every applicable accuracy denominator; the result records coverage separately.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import re
from typing import Any

from .schema import Sample


_CLOSED_THINK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.IGNORECASE | re.DOTALL)
_ANSWER_TAG = re.compile(r"<answer\b[^>]*>(.*?)</answer\s*>", re.IGNORECASE | re.DOTALL)
_BARE_LETTER = re.compile(r"^(?:\(([A-Za-z])\)|([A-Za-z]))[.)]?$", re.DOTALL)
_EXPLICIT_MARKER = re.compile(
    r"\b(?:final\s+answer|best\s+option|correct\s+(?:answer|option)|answer)\s*(?::|=|is\b)\s*([^\n]*)",
    re.IGNORECASE,
)


def _plain(text: str) -> str:
    text = text.strip()
    # Markdown emphasis and a single inline-code wrapper are presentation only.
    for left, right in (("**", "**"), ("`", "`")):
        if text.startswith(left) and text.endswith(right) and len(text) >= 2 * len(left):
            text = text[len(left):-len(right)].strip()
    return text


def _content_key(text: str) -> str:
    return " ".join(text.split()).casefold()


def _decode_candidate(text: str, choices: tuple[str, ...]) -> str | None:
    text = _plain(text)
    labels = tuple(chr(65 + index) for index in range(len(choices)))
    match = _BARE_LETTER.fullmatch(text)
    if match:
        candidate = (match.group(1) or match.group(2)).upper()
        return candidate if candidate in labels else None
    matches = [labels[index] for index, choice in enumerate(choices)
               if _content_key(text) == _content_key(choice)]
    if len(matches) == 1:
        return matches[0]
    # Also accept an option label followed by its exact option text. A free-form
    # sentence that merely starts with an uppercase character is not an answer.
    match = re.fullmatch(r"(?:\(([A-Za-z])\)|([A-Za-z])[.)])\s*(.+)", text, re.DOTALL)
    if match:
        candidate = (match.group(1) or match.group(2)).upper()
        if candidate in labels and _content_key(match.group(3)) == _content_key(choices[labels.index(candidate)]):
            return candidate
    return None


def extract_answer(text, choices) -> str | None:
    """Extract a valid option, without ground-truth labels or an API judge.

    Structured answer tags take precedence and work even after truncated think
    output. Plaintext inside completed reasoning is ignored. Unclosed reasoning
    without an answer tag is invalid, even if it mentions an option letter.
    """
    choices = tuple(str(choice) for choice in choices)
    if not 1 <= len(choices) <= 26:
        raise ValueError("MCQ parsing requires 1..26 options")
    if not isinstance(text, str) or not text.strip():
        return None
    visible = _CLOSED_THINK.sub("", text)
    tags = _ANSWER_TAG.findall(visible)
    if tags:
        # A later answer tag expresses the final choice. An invalid final tag
        # does not recover an earlier choice from reasoning or self-correction.
        return _decode_candidate(tags[-1], choices)
    if re.search(r"<think\b", visible, re.IGNORECASE):
        return None
    # Some chat templates prefill <think>; generated text begins after it.
    if re.search(r"</think\s*>", visible, re.IGNORECASE):
        visible = re.split(r"</think\s*>", visible, flags=re.IGNORECASE)[-1]
    candidate = _decode_candidate(visible, choices)
    if candidate is not None:
        return candidate
    markers = _EXPLICIT_MARKER.findall(visible)
    if not markers:
        return None
    explicit = _plain(markers[-1])
    candidate = _decode_candidate(explicit, choices)
    if candidate is not None:
        return candidate
    # An explicit final-answer statement can be followed by an explanation.
    # Reject lists/alternatives such as "B or C" rather than choosing one.
    match = re.match(r"^(?:\(([A-Z])\)|([A-Z]))([.)]?)(?=\s|$)(.*)$", explicit, re.DOTALL)
    if match:
        label = (match.group(1) or match.group(2)).upper()
        suffix = match.group(4).lstrip()
        if re.match(r"^(?:or\b|and\b|/|,|\|)", suffix, re.IGNORECASE):
            return None
        if suffix and not (match.group(1) or match.group(3) or
                           re.match(r"^(?:because\b|since\b|as\b|[-:\u2014\u2013])", suffix, re.IGNORECASE)):
            return None
        if label in {chr(65 + index) for index in range(len(choices))}:
            return label
    return None


def majority_vote(answers: list[str | None]) -> str | None:
    """Ignore invalid completions; break vote ties by first valid completion."""
    valid = [answer.strip().upper() for answer in answers
             if isinstance(answer, str) and re.fullmatch(r"[A-Za-z]", answer.strip())]
    if not valid:
        return None
    counts = Counter(valid)
    maximum = max(counts.values())
    return next(answer for answer in valid if counts[answer] == maximum)


def _prediction(record: dict[str, Any] | None, sample: Sample) -> str | None:
    if record is None:
        return None
    raw = record.get("prediction")
    if isinstance(raw, str) and raw in sample.labels:
        return raw
    return None


def _stats(samples: list[Sample], records: dict[str, dict[str, Any]],
           errors: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    total = len(samples)
    done = sum(sample.uid in records for sample in samples)
    correct = sum(_prediction(records.get(sample.uid), sample) == sample.answer for sample in samples)
    invalid = sum(sample.uid in records and _prediction(records[sample.uid], sample) is None for sample in samples)
    return {
        "n_total": total,
        "n_done": done,
        "n_correct": correct,
        "accuracy": 100.0 * correct / total if total else None,
        "invalid_answers": invalid,
        "n_missing": total - done,
        "complete": done == total and not errors,
    }


def _canonical_benchmark(name: str) -> str:
    return name.casefold().replace("-", "").replace("_", "")


def _is_action(sample: Sample) -> bool:
    benchmark = _canonical_benchmark(sample.benchmark)
    if benchmark == "mvbench":
        return sample.metadata.get("task") == "Fine-grained Action"
    if benchmark == "videomme":
        return sample.metadata.get("task_type") == "Action Recognition"
    if benchmark == "seedbench":
        return sample.metadata.get("task") == "Action Recognition"
    if benchmark in {"tempcompass", "tempcompassmcq"}:
        return str(sample.metadata.get("dim", sample.metadata.get("dimension", ""))).casefold() == "action"
    return False


def _group_stats(samples: list[Sample], records: dict[str, dict[str, Any]], key: str,
                 errors: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        value = sample.metadata.get(key)
        if value is not None and str(value).strip():
            groups[str(value)].append(sample)
    result = {}
    for value, group in groups.items():
        uids = {sample.uid for sample in group}
        group_errors = [error for error in errors if error.get("uid") in uids or not error.get("uid")]
        result[value] = _stats(group, records, group_errors)
    return result


def summarize(samples: list[Sample], records: list[dict], errors: list[dict] | None = None) -> dict:
    """Score expected questions, rejecting duplicate or unrelated records.

    ``n_done`` counts inference records, including unparseable outputs.
    ``invalid_answers`` counts recorded invalid outputs; ``n_missing`` counts
    absent/failed questions. Both contribute zero to expected-set accuracy.
    """
    errors = list(errors or [])
    by_uid: dict[str, Sample] = {}
    for sample in samples:
        if sample.uid in by_uid:
            raise ValueError(f"Duplicate expected sample ID: {sample.uid}")
        by_uid[sample.uid] = sample
    record_map: dict[str, dict] = {}
    for record in records:
        uid = record.get("uid")
        if uid not in by_uid:
            raise ValueError(f"Unknown prediction sample ID: {uid!r}")
        if uid in record_map:
            raise ValueError(f"Duplicate prediction sample ID: {uid}")
        if record.get("benchmark", by_uid[uid].benchmark) != by_uid[uid].benchmark:
            raise ValueError(f"Prediction benchmark does not match expected sample {uid}")
        record_map[uid] = record
    error_uids = set()
    benchmarks = {sample.benchmark for sample in samples}
    for error in errors:
        uid = error.get("uid")
        if uid is not None:
            if uid not in by_uid:
                raise ValueError(f"Unknown failed sample ID: {uid!r}")
            if uid in error_uids:
                raise ValueError(f"Duplicate failed sample ID: {uid}")
            error_uids.add(uid)
            if error.get("benchmark", by_uid[uid].benchmark) != by_uid[uid].benchmark:
                raise ValueError(f"Error benchmark does not match expected sample {uid}")
        elif error.get("benchmark") is not None and error["benchmark"] not in benchmarks:
            raise ValueError(f"Unknown error benchmark: {error['benchmark']!r}")
    report = _stats(samples, record_map, errors)
    report["overall"] = report.pop("accuracy")
    report["n_errors"] = len(errors)
    report["errors"] = errors
    report["benchmarks"] = {}
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.benchmark].append(sample)
    for benchmark, group in grouped.items():
        uids = {sample.uid for sample in group}
        benchmark_errors = [error for error in errors
                            if error.get("uid") in uids or error.get("benchmark") == benchmark
                            or (not error.get("uid") and not error.get("benchmark"))]
        metrics = _stats(group, record_map, benchmark_errors)
        metrics["overall"] = metrics.pop("accuracy")
        metrics["n_errors"] = len(benchmark_errors)
        action = _stats([sample for sample in group if _is_action(sample)], record_map, benchmark_errors)
        metrics["action_recognition"] = action["accuracy"]
        for field in ("n_total", "n_done", "n_correct"):
            metrics[f"action_recognition_{field}"] = action[field]
        canonical = _canonical_benchmark(benchmark)
        if canonical == "mvbench":
            metrics["by_task"] = _group_stats(group, record_map, "task", benchmark_errors)
            scores = [bucket["accuracy"] for bucket in metrics["by_task"].values() if bucket["accuracy"] is not None]
            metrics["avg"] = sum(scores) / len(scores) if scores else None
            metrics["fa"] = metrics["action_recognition"]
            metrics["n_tasks"] = len(metrics["by_task"])
        elif canonical == "videomme":
            metrics["by_task"] = _group_stats(group, record_map, "task_type", benchmark_errors)
            metrics["by_duration"] = _group_stats(group, record_map, "duration", benchmark_errors)
            metrics["by_domain"] = _group_stats(group, record_map, "domain", benchmark_errors)
        elif canonical == "seedbench":
            metrics["by_task"] = _group_stats(group, record_map, "task", benchmark_errors)
        elif canonical in {"tempcompass", "tempcompassmcq"}:
            dimension_key = "dim" if any("dim" in sample.metadata for sample in group) else "dimension"
            metrics["by_dimension"] = _group_stats(group, record_map, dimension_key, benchmark_errors)
        report["benchmarks"][benchmark] = metrics
    return report
