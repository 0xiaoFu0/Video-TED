#!/usr/bin/env python3
"""CPU-only audit against the pinned official Git tree, launchers, and labels."""
import argparse
import ast
import hashlib
from pathlib import Path

from protocol import (EVAL_SPLITS, PROMPT_PROTOCOL, ROOT, atomic_json, digest,
                      load_dataset, messages, official, official_scorer, read_json, registry)


def verify_sources():
    """Reject changed official data/helpers before any GPU work."""
    manifest = read_json(ROOT / "SOURCE_MANIFEST.json")
    tree = read_json(ROOT / "UPSTREAM_TREE.json")
    assert tree["sha"] == manifest["commit"] and not tree.get("truncated")
    blobs = {item["path"]: item["sha"] for item in tree["tree"] if item["type"] == "blob"}
    for item in manifest["files"]:
        data = (ROOT / item["package_path"]).read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == blobs[item["upstream_path"]] == item["git_blob"]
        assert hashlib.sha256(data).hexdigest() == item["sha256"]
    return manifest


def audit():
    from qwen3_video import input_profile
    manifest = verify_sources()
    reg = registry()
    scorer = official_scorer()
    normalize = scorer["normalize"]
    splits, samples, candidates = {}, {}, {}
    for key in EVAL_SPLITS:
        spec = reg["datasets"][key]
        rows, classes, count = load_dataset(spec)
        launch = (ROOT / "upstream" / spec["upstream_launch"]).read_text()
        assert spec["annotations"] in launch and spec["classes"] in launch
        api = official(spec["entry"], classes, "/MEDIA-ROOT", "/ANNOTATED-ROOT")
        names = [value.split(":")[0].strip() for value in classes.values()]
        assert api["get_allowed_answers_str"](False).splitlines() == names
        candidates[key] = {normalize(name) for name in names}
        assert len(candidates[key]) == len(names)
        samples[key] = {r["path"]: normalize(scorer["extract_answer"](r["solution"])) for r in rows}
        assert set(samples[key].values()) <= candidates[key]
        prompt = messages(rows[0], api)[0]["content"][1]["text"]
        for row in rows:
            current = messages(row, api)[0]["content"][1]["text"]
            assert current == prompt
            assert row["path"] not in current and Path(row["path"]).name not in current
        assert "/MEDIA-ROOT" not in prompt and "/ANNOTATED-ROOT" not in prompt
        # Generation controls are copied from each official entry, not guessed
        # from the paper or from our training configuration.
        tree_ast = ast.parse((ROOT / "upstream/AScripts" / spec["entry"] / "4subtol.py").read_text())
        call = next(n.value for n in tree_ast.body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "sampling_params" for t in n.targets))
        sampling = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        assert sampling == {"temperature": 0.8, "top_p": 0.001, "max_tokens": 4096, "stop_token_ids": []}
        splits[key] = {"rows": count, "classes": len(classes), "entry": spec["entry"],
                       "reference_ar_training_overlap_full_split": spec.get("ar_train_overlap"),
                       "upstream_launch": spec["upstream_launch"],
                       "annotations": spec["annotations"], "annotations_sha256": digest(ROOT / spec["annotations"]),
                       "class_file": spec["classes"], "classes_sha256": digest(ROOT / spec["classes"]),
                       "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "generation": sampling}
    unions, conflicts = {}, {}
    for dataset in ("hmdb51", "ucf101", "ssv2"):
        base, novel = dataset + "-base", dataset + "-novel"
        assert not samples[base].keys() & samples[novel].keys()
        assert not candidates[base] & candidates[novel]
        combined = {**samples[base], **samples[novel]}
        if dataset != "ssv2":
            full = samples[dataset + "-split1"]
            assert combined.keys() == full.keys()
            assert candidates[base] | candidates[novel] == candidates[dataset + "-split1"]
            mismatch = [{"path": path, "base_novel_target_normalized": target,
                         "split1_target_normalized": full[path]}
                        for path, target in combined.items() if target != full[path]]
            unions[dataset] = {"same_video_paths": True, "candidate_spaces_differ": True,
                               "normalized_target_mismatches": len(mismatch)}
            if mismatch:
                conflicts[dataset] = mismatch
    return {"prompt_protocol": PROMPT_PROTOCOL, "qwen3_input_adapter": input_profile(), "repository": manifest["repository"],
            "commit": manifest["commit"], "upstream_tree_sha256": digest(ROOT / "UPSTREAM_TREE.json"),
            "verified_unchanged_official_files": len(manifest["files"]), "splits": splits,
            "split_relationships": unions,
            "official_annotation_conflicts": conflicts,
            "conflict_policy": "Keep each official annotation byte-for-byte; report mismatches; do not relabel or reuse predictions across protocols",
            "total_evaluation_items": sum(s["rows"] for s in splits.values()),
            "unique_dataset_video_pairs": len({(reg["datasets"][key]["dataset"], path)
                                                 for key in EVAL_SPLITS for path in samples[key]}),
            "inference": "single recognition pass; class names only; no tools, class explanations or text media paths",
            "preserved": ["official annotation bytes, order and labels", "official per-split candidate files and order",
                          "vendored temporal sampling and pixel-budget rules; Qwen3 uses native factor-32 resizing",
                          "temperature=0.8, top_p=0.001, max_tokens=4096, max_model_len=16384",
                          "official answer extraction/normalization and sample-weighted top-1"],
            "explicit_differences": ["tool-selection turn skipped; noun explanations fixed off",
                                     "Qwen3 uses original FPS and sampled frame indices, factor 32 instead of Qwen2.5 factor 28; no processor resampling or resizing",
                                     "shared tool-free recognition prompt replaces official second-turn instructions; concise evidence summary and one candidate label",
                                     "label-bearing annotated-video paragraph absent",
                                     "eight independent GPU workers, batch size 2, seed + shard index",
                                     "all three HMDB/UCF splits evaluated; means are equal-weight averages of split accuracies; population std (ddof=0)",
                                     "HMDB split2/3 overlap the reference AR training set by 1340/1311 videos; annotated in output",
                                     "AR training covers all 51 HMDB classes, so Novel is a split name, not unseen to this model"],
            "not_verified": ["identity of every remote raw video byte and successful decoding of every video",
                             "paper experiment's unpublished runtime/version details", "Qwen3 real GPU execution"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="ALIGNMENT.json")
    args = parser.parse_args()
    report = audit()
    atomic_json(args.output, report)
    print(f"Verified {report['verified_unchanged_official_files']} official files, "
          f"{len(report['splits'])} splits, {report['total_evaluation_items']} evaluation items. Report: {args.output}")
