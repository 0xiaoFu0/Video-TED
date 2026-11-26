"""Ensure vision-weight updates cannot retain stale vLLM encoder outputs.

Load only the production method so these state-machine checks run without
CUDA/vLLM. The server smoke test is still required for the actual engine layout.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


ROOT = Path(__file__).resolve().parents[2]


def drain(manager):
    path = ROOT / 'verl/workers/sharding_manager/fsdp_vllm.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FSDPVLLMShardingManager')
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_drain_vllm_encoder_cache')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {}
    exec(compile(module, str(path), 'exec'), namespace)
    namespace[method.name](manager)


def engine_state(*, unfinished=False, incomplete=False):
    runner = NS(encoder_cache={'old_visual_hash': object()})
    cache = NS(cache_size=64, num_free_slots=32, num_freeable_slots=48,
               cached={'old_visual_hash': 32}, freed=['old_visual_hash'], freeable={'old_visual_hash': 32})
    if incomplete:
        del cache.cached
    scheduler = NS(encoder_cache_manager=cache, has_unfinished_requests=lambda: unfinished)
    engine = NS(model_executor=NS(driver_worker=NS(worker=NS(model_runner=runner))),
                engine_core=NS(engine_core=NS(scheduler=scheduler)))
    return NS(inference_engine=NS(llm_engine=engine)), runner, cache


def test_finished_generation_clears_runner_and_scheduler_together():
    manager, runner, cache = engine_state()
    drain(manager)
    assert runner.encoder_cache == {}
    assert cache.cached == {} and cache.freed == [] and cache.freeable == {}
    assert cache.num_free_slots == cache.num_freeable_slots == cache.cache_size
    drain(manager)  # Empty state is valid on the next generation as well.


def test_running_generation_retains_consistent_cache_state():
    manager, runner, cache = engine_state(unfinished=True)
    with pytest.raises(RuntimeError, match="unfinished requests"):
        drain(manager)
    assert runner.encoder_cache and cache.cached
    assert cache.num_free_slots == 32


def test_unknown_engine_layout_fails_without_half_clearing():
    manager, runner, cache = engine_state(incomplete=True)
    with pytest.raises(RuntimeError, match='refusing to continue'):
        drain(manager)
    assert runner.encoder_cache and cache.freed
    assert cache.num_free_slots == 32


@pytest.mark.parametrize('field,value', [
    ('cached', None), ('freeable', []), ('freed', {}),
    ('cache_size', -1), ('num_free_slots', 1.5), ('num_freeable_slots', None),
])
def test_malformed_accounting_never_half_clears(field, value):
    manager, runner, cache = engine_state()
    setattr(cache, field, value)
    with pytest.raises(RuntimeError, match='Cannot safely reset'):
        drain(manager)
    assert runner.encoder_cache
    assert getattr(cache, field) is value
    if field != 'cached':
        assert cache.cached


def test_ordered_freeable_matches_pinned_vllm_layout():
    from collections import OrderedDict

    manager, runner, cache = engine_state()
    cache.freeable = OrderedDict(cache.freeable)
    drain(manager)
    assert not runner.encoder_cache and not cache.freeable
    assert cache.num_free_slots == cache.cache_size
