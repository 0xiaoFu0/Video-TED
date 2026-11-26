"""The driver tokenizes each distinct (teacher prompt, media object) once per step.

Rollout groups repeat the same prompt and the same decoded video n times, so
``construct_teacher_prompts`` must not run the video processor n times per
group, and the memoized rows must be exactly what per-row tokenization gives.
"""
import ast
from pathlib import Path

from jinja2 import Template
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]


class FakeDataProto:
    def __init__(self, tensors, non_tensors):
        self.batch, self.non_tensor_batch = tensors, non_tensors

    @classmethod
    def from_dict(cls, tensors, non_tensors):
        return cls(tensors, non_tensors)


def load_construct(calls):
    path = ROOT / "verl/trainer/opsd_algos.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in ("_build_messages_from_prompt_text", "construct_teacher_prompts")]
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                              *functions], type_ignores=[])
    ast.fix_missing_locations(module)

    def tokenize(prompt, tokenizer, processor, mm_item, max_prompt_length, *args, **kwargs):
        calls.append((prompt, id(mm_item)))
        ids = torch.tensor([ord(c) for c in prompt][-max_prompt_length:])
        pad = max_prompt_length - len(ids)
        ids = torch.cat([torch.zeros(pad, dtype=torch.long), ids])
        mask = torch.cat([torch.zeros(pad, dtype=torch.long), torch.ones(len(ids) - pad, dtype=torch.long)])
        return ids, mask, torch.arange(max_prompt_length), ids[pad:].tolist(), prompt

    class Tokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            content = messages[0]["content"]
            return content if isinstance(content, str) else content[-1]["text"]

    namespace = dict(torch=torch, np=np, random=__import__("random"), Template=Template,
                     DataProto=FakeDataProto, _tokenize_prompt=tokenize,
                     __name__="verl.trainer.ar_memo_test", __package__="verl.trainer")
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["construct_teacher_prompts"], Tokenizer()


def test_group_rows_sharing_prompt_and_media_are_tokenized_once_and_identically():
    calls = []
    construct, tokenizer = load_construct(calls)
    n = 4
    videos = [{"video": [torch.zeros(2, 3, 4, 4)]} for _ in range(3)]
    # DataProto.repeat(interleave=True) semantics: object references repeated n times.
    multi_modal = np.repeat(np.array(videos, dtype=object), n)
    questions = np.repeat(np.array(["q0", "q1", "q2"], dtype=object), n)
    hints = [[f"ctx{i // n}"] for i in range(3 * n)]
    out = construct(questions, hints, np.array([""] * 12, dtype=object), 0.0, tokenizer, None,
                    template_text="{{ question }}|{{ hint }}", max_prompt_length=16,
                    multi_modal_data=multi_modal)
    assert len(calls) == 3, calls
    assert len({prompt for prompt, _ in calls}) == 3
    assert out.batch["input_ids"].shape == (12, 16)
    for group in range(3):
        rows = out.batch["input_ids"][group * n:(group + 1) * n]
        assert torch.equal(rows, rows[0].expand_as(rows))
        assert all(p == f"q{group}|ctx{group}" for p in out.non_tensor_batch["raw_prompt"][group * n:(group + 1) * n])
    # Row order and content match a run without memoization.
    calls.clear()
    reference, _ = load_construct(calls)
    expected = reference(questions, hints, np.array([""] * 12, dtype=object), 0.0, tokenizer, None,
                         template_text="{{ question }}|{{ hint }}", max_prompt_length=16,
                         multi_modal_data=np.array([dict(v) for v in multi_modal], dtype=object))
    assert len(calls) == 12  # distinct media objects: no sharing, no memo hits
    for key in ("input_ids", "attention_mask", "position_ids"):
        assert torch.equal(out.batch[key], expected.batch[key])
    assert list(out.non_tensor_batch["raw_prompt"]) == list(expected.non_tensor_batch["raw_prompt"])


def test_memo_distinguishes_prompt_text_and_media_identity():
    calls = []
    construct, tokenizer = load_construct(calls)
    shared = {"video": [torch.zeros(2, 3, 4, 4)]}
    multi_modal = np.array([shared, shared, {"video": [torch.zeros(2, 3, 4, 4)]}], dtype=object)
    construct(np.array(["q", "q", "q"], dtype=object), [["a"], ["b"], ["a"]],
              np.array([""] * 3, dtype=object), 0.0, tokenizer, None,
              template_text="{{ question }}|{{ hint }}", max_prompt_length=8, multi_modal_data=multi_modal)
    # same media, different hint -> 2 calls; same hint, different media object -> 1 more.
    assert len(calls) == 3


def test_text_only_rows_are_never_memoized():
    calls = []
    construct, tokenizer = load_construct(calls)
    construct(np.array(["q", "q"], dtype=object), [["a"], ["a"]], np.array(["", ""], dtype=object),
              0.0, tokenizer, None, template_text="{{ question }}|{{ hint }}", max_prompt_length=8)
    assert len(calls) == 2
