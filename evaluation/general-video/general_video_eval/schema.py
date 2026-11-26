"""Dependency-free shared contracts. Dataset labels never enter inference prompts."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import re
from typing import Any, Optional


@dataclass
class Sample:
    uid: str
    benchmark: str
    question: str
    choices: tuple[str, ...]
    answer: str
    media_path: str
    media_type: str = "video"  # video, frames, gif
    start: Optional[float] = None  # seconds, inclusive
    end: Optional[float] = None    # seconds, inclusive
    metadata: dict[str, Any] = field(default_factory=dict)
    frame_fps: float = 3.0

    def __post_init__(self):
        self.choices = tuple(self.choices)
        if not self.uid or not self.question.strip():
            raise ValueError("Sample ID and question must be nonempty")
        # The official MVBench release includes five one-option questions.
        minimum = 1 if self.benchmark.casefold() == "mvbench" else 2
        if not minimum <= len(self.choices) <= 26:
            raise ValueError(f"{self.uid}: expected {minimum}..26 choices, got {len(self.choices)}")
        if self.answer not in self.labels:
            raise ValueError(f"{self.uid}: invalid ground-truth answer {self.answer!r}")
        if self.media_type not in {"video", "frames", "gif"}:
            raise ValueError(f"{self.uid}: unknown media type {self.media_type!r}")
        if self.start is not None and self.start < 0:
            raise ValueError(f"{self.uid}: negative clip start")
        if self.end is not None and (self.end < 0 or self.end < (self.start or 0)):
            raise ValueError(f"{self.uid}: invalid clip interval")

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(chr(65 + i) for i in range(len(self.choices)))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_choices(raw: Any) -> tuple[str, ...]:
    if isinstance(raw, dict):
        raw = [raw[k] for k in sorted(raw)]
    if not isinstance(raw, (list, tuple)) and hasattr(raw, "tolist"):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        raise ValueError("Options must be a list of strings or an option dictionary")
    return tuple(re.sub(r"^\s*(?:\([A-Z]\)|[A-Z][.)])\s*", "", str(x)).strip() for x in raw)


def normalize_answer(raw: Any, choices: tuple[str, ...]) -> str:
    labels = tuple(chr(65 + i) for i in range(len(choices)))
    if isinstance(raw, int) and not isinstance(raw, bool):
        if 0 <= raw < len(labels):
            return labels[raw]
    value = str(raw).strip()
    if value.upper() in labels:
        return value.upper()
    match = re.match(r"^(?:\(([A-Z])\)|([A-Z])[.)])(?:\s|$)", value)
    if match and (match.group(1) or match.group(2)) in labels:
        return match.group(1) or match.group(2)
    matches = [labels[i] for i, choice in enumerate(choices) if value.casefold() == choice.casefold()]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(f"Cannot resolve ground-truth answer {raw!r} against {choices!r}")


def media_exists(sample: Sample) -> bool:
    path = Path(sample.media_path)
    return path.is_dir() if sample.media_type == "frames" else path.is_file()
