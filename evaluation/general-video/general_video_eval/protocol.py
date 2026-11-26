"""Explicit, fingerprinted evaluation protocols for paired model comparisons.

The Video-R1 reference profile combines its published 64-frame inference budget
and temperature with the public evaluation prompt and generation limits. It is
not a claim of reproducing the old upstream runtime, dataset conversion, or
scorer. Official benchmark clip boundaries and this package's scoring remain.
"""
from __future__ import annotations

from typing import Any


DEFAULT_FRAMES = {"mvbench": 32, "videomme": 64, "tempcompass": 32, "seedbench": 8}
PROFILE_NAMES = ("generic", "videor1_64", "seedbench_8", "simple_32", "simple_64", "seedbench_simple_8")
SIMPLE_VIDEO_FRAMES = {"simple_32": 32, "simple_64": 64}
SEED_PROFILES = {"seedbench_8", "seedbench_simple_8"}
GENERIC_DEFAULTS = {
    "think_mode": 1, "n": 1, "top_p": 1.0, "max_model_len": 32768,
    "fps": 2.0, "min_frames": 4, "max_pixels": 262144,
    "min_pixels": 3136, "repetition_penalty": 1.05,
}
VIDEOR1_SETTINGS = {
    "think_mode": 1, "n": 1, "temperature": 0.01, "top_p": 0.001,
    "max_new_tokens": 1024, "max_model_len": 16384,
    "repetition_penalty": 1.0, "fps": 2.0, "min_frames": 4,
    "min_pixels": 128 * 28 * 28, "max_pixels": 256 * 28 * 28,
}
VIDEOR1_QUESTION_TEMPLATE = (
    "{Question}\n"
    "Please think about this question as if you were a human pondering deeply. "
    "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions "
    "It's encouraged to include self-reflection or verification in the reasoning process. "
    "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags."
)
VIDEOR1_MCQ_SUFFIX = (
    " Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer> </answer> tags."
)
VIDEOR1_REFERENCES = {
    "paper": "https://arxiv.org/html/2503.21776v4#S4.SS1",
    "prompt_and_generation": "https://github.com/tulerfeng/Video-R1/blob/main/src/eval_bench.py",
    "sampling": "https://github.com/tulerfeng/Video-R1/blob/main/src/qwen-vl-utils/src/qwen_vl_utils/vision_process.py",
    "checked_on": "2026-10-15",
    "temperature_note": "Uses paper/README temperature 0.01; checked upstream eval_bench.py uses 0.1.",
    "coverage_note": "Official local benchmark annotations; MVBench clips retained; TempCompass native MCQ only.",
    "runtime_note": "Uses this package's current vLLM adapter, decoder and deterministic scorer; not an exact upstream reproduction.",
}


def apply_eval_profile(args: Any, benchmarks: list[str]) -> dict[str, Any]:
    """Fill unspecified settings and reject overrides of a named protocol.

    argparse supplies None for unspecified protocol knobs. CLI/environment
    overrides therefore cannot silently turn a named profile into another run.
    """
    profile = getattr(args, "eval_profile", "generic")
    if profile not in PROFILE_NAMES:
        raise ValueError(f"Unknown evaluation profile: {profile!r}")
    if profile in SEED_PROFILES and benchmarks != ["seedbench"]:
        raise ValueError(f"EVAL_PROFILE={profile} requires BENCHMARKS=seedbench")
    if (profile == "videor1_64" or profile in SIMPLE_VIDEO_FRAMES) and "seedbench" in benchmarks:
        seed_profile = "seedbench_simple_8" if profile in SIMPLE_VIDEO_FRAMES else "seedbench_8"
        raise ValueError(f"SEED-Bench provides eight pre-extracted frames; use BENCHMARKS=seedbench EVAL_PROFILE={seed_profile}")
    reference_prompt = profile in {"videor1_64", "seedbench_8"}
    fixed_settings = reference_prompt or profile in SIMPLE_VIDEO_FRAMES or profile == "seedbench_simple_8"
    settings = VIDEOR1_SETTINGS if fixed_settings else GENERIC_DEFAULTS
    for key, expected in settings.items():
        actual = getattr(args, key, None)
        if actual is None:
            setattr(args, key, expected)
        elif fixed_settings and actual != expected:
            raise ValueError(
                f"EVAL_PROFILE={profile} requires {key}={expected}, got {actual}. "
                "Remove the conflicting CLI/environment override or use EVAL_PROFILE=generic."
            )
    for benchmark in benchmarks:
        key = benchmark + "_max_frames"
        actual = getattr(args, key, None)
        expected = 64 if profile == "videor1_64" else SIMPLE_VIDEO_FRAMES.get(profile, DEFAULT_FRAMES[benchmark])
        if actual is None:
            setattr(args, key, expected)
        elif (fixed_settings or benchmark == "seedbench") and actual != expected:
            raise ValueError(f"EVAL_PROFILE={profile} requires {key}={expected}, got {actual}.")
    config = {
        "eval_profile": profile,
        "prompt_style": "videor1" if reference_prompt else "generic",
        "video_sampling": "seedbench_official_8" if profile in SEED_PROFILES else "uniform_fps_floor" if profile == "videor1_64" or profile in SIMPLE_VIDEO_FRAMES else "benchmark_default",
        "protocol_reference": dict(VIDEOR1_REFERENCES) if reference_prompt else {},
    }
    if profile in SIMPLE_VIDEO_FRAMES:
        frame_cap = SIMPLE_VIDEO_FRAMES[profile]
        config["protocol_reference"] = {
            "kind": "local_prompt_and_frame_budget_variant",
            "parent_profile": "videor1_64",
            "changes": f"generic short reasoning prompt (THINK_MODE=1); frame cap {frame_cap}",
            "sampling_note": f"Retains uniform_fps_floor, target 2 FPS, min 4 and official clip boundaries; short clips can use fewer than {frame_cap} frames.",
            "generation_note": "Generation, pixel and context budgets match videor1_64; this is not the Video-R1 reference prompt protocol.",
        }
    if profile == "seedbench_simple_8":
        config["protocol_reference"] = {
            "kind": "local_prompt_variant", "parent_profile": "seedbench_8",
            "changes": "generic short reasoning prompt (THINK_MODE=1)",
            "generation_note": "Generation, pixel and context budgets match seedbench_8; eight official frames remain unchanged.",
        }
    if "seedbench" in benchmarks:
        config["protocol_reference"]["seedbench_coverage_note"] = (
            "SEED-Bench-1 video MCQ only; eight official frames per question, no source-video resampling."
        )
        config["seedbench_input"] = {
            "frames": 8, "sampling": "seedbench_official_8", "resample": False,
            "timing_source": "synthetic_ordinal_2fps", "source_timing_known": False,
            "scoring": "generated_mcq_pooled_accuracy_not_candidate_likelihood",
            "source": "https://huggingface.co/datasets/AILab-CVC/SEED-Bench",
        }
    return config


def build_prompt(sample: Any, think_mode: bool, prompt_style: str = "generic") -> str:
    if prompt_style == "videor1":
        if not think_mode:
            raise ValueError("Video-R1 prompt requires THINK_MODE=1")
        # Labels are reconstructed from normalized options, never metadata or
        # ground truth. A separating newline avoids upstream's accidental
        # question/Options concatenation for questions without a trailing one.
        options = "\n".join(f"{label}. {choice}" for label, choice in zip(sample.labels, sample.choices))
        question = f"{sample.question}\nOptions:\n{options}\n"
        return VIDEOR1_QUESTION_TEMPLATE.format(Question=question) + VIDEOR1_MCQ_SUFFIX
    if prompt_style != "generic":
        raise ValueError(f"Unknown prompt style: {prompt_style!r}")
    options = "\n".join(f"({label}) {choice}" for label, choice in zip(sample.labels, sample.choices))
    if think_mode:
        instruction = ("Select the best answer based on the video. Reason briefly using only visible video evidence "
                       "inside <think>...</think>, then output the option letter inside <answer>...</answer>.")
    else:
        instruction = "Select the best answer based on the video. Respond with only the option letter."
    return f"{instruction}\nQuestion: {sample.question}\nOptions:\n{options}"
