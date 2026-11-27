"""Validate a VideoTED training manifest and, optionally, its video paths."""

import argparse
import json
from pathlib import Path

from .response_format import validate_reference_format, validate_student_format
from .reward import parse_answer, target_label


def read_rows(path):
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object on line {line_number}")
            yield row


def validate_rows(rows, check_media=False):
    row_count = 0
    classes = set()
    missing_count = 0
    missing_examples = []

    for row_count, row in enumerate(rows, start=1):
        try:
            target = target_label(row["answer"])
            videos = row["videos"]
            if (not isinstance(videos, list) or len(videos) != 1
                    or not isinstance(videos[0], str) or not videos[0]):
                raise ValueError("Expected one video path per sample")
            video = videos[0]
            prompt = row["prompt"]
            if not isinstance(prompt, str) or prompt.count("<video>") != 1:
                raise ValueError("Expected exactly one video marker")
            validate_student_format(prompt)

            teacher_context = row["teacher_context"]
            if not isinstance(teacher_context, str) or not teacher_context.strip():
                raise ValueError("Missing teacher context")
            status = row["reference_status"]
            if status not in ("unreviewed", "reviewed", "verified", "rejected"):
                raise ValueError("Unknown reference status")
            reference = row["reference_response"]
            if not isinstance(reference, str):
                raise ValueError("Reference must be text")
            if reference or status != "rejected":
                if parse_answer(reference) != target:
                    raise ValueError("Reference/target mismatch")
                validate_reference_format(reference)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Row {row_count}: {error}") from error

        classes.add(row["answer"])
        if check_media and not Path(video).is_file():
            missing_count += 1
            if len(missing_examples) < 3:
                missing_examples.append(video)

    if not row_count:
        raise ValueError("Empty dataset")
    if missing_count:
        raise FileNotFoundError(f"{missing_count} video files missing; examples: {missing_examples}")
    return {"rows": row_count, "classes": len(classes), "media_checked": check_media}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, required=True, metavar="JSONL")
    parser.add_argument("--check-media", action="store_true")
    args = parser.parse_args()
    try:
        stats = validate_rows(read_rows(args.check), check_media=args.check_media)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
