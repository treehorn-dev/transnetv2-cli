from types import SimpleNamespace

import pytest

from transnetv2_cli.backend import BackendUnavailableError, normalize_transitions, resolve_device


def test_resolve_device_prefers_mps_then_cuda_then_cpu():
    torch = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: True),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    assert resolve_device(torch) == "mps"

    torch = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False),
        cuda=SimpleNamespace(is_available=lambda: True),
    )
    assert resolve_device(torch) == "cuda:0"

    torch = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    assert resolve_device(torch) == "cpu"


def test_resolve_device_rejects_unavailable_requested_cuda():
    torch = SimpleNamespace(
        mps=SimpleNamespace(is_available=lambda: False),
        cuda=SimpleNamespace(is_available=lambda: False),
    )

    with pytest.raises(BackendUnavailableError, match="CUDA was requested"):
        resolve_device(torch, requested="cuda")


def test_normalize_transitions_filters_short_shots_and_emits_ms():
    shots, total_frames, duration_ms = normalize_transitions(
        [(0, 10, 0.9), (10, 40, 0.8)],
        fps=10.0,
        min_shot_ms=1500,
        include_probs=True,
    )

    assert len(shots) == 1
    assert shots[0]["start_ms"] == 1000
    assert shots[0]["end_ms"] == 4000
    assert shots[0]["probability"] == 0.8
    assert total_frames == 40
    assert duration_ms == 4000
