from __future__ import annotations

import contextlib
import importlib.resources
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from transnetv2_cli.shots import build_shot


class BackendUnavailableError(RuntimeError):
    pass


@dataclass
class DetectionResult:
    backend: str
    model: str
    fps: float
    source_video: str
    shots: list[dict[str, Any]]
    total_frames: int | None = None
    duration_ms: int | None = None
    resolved_device: str | None = None
    cuda_available: bool | None = None
    decode_backend: str = "ffmpeg-cpu"


def resolve_device(torch_module: Any, requested: str | None = None) -> str:
    mps = getattr(getattr(torch_module, "mps", None), "is_available", lambda: False)
    cuda = getattr(getattr(torch_module, "cuda", None), "is_available", lambda: False)
    if requested:
        requested_lower = requested.lower()
        if requested_lower.startswith("cuda") and not cuda():
            raise BackendUnavailableError("CUDA was requested but torch.cuda.is_available() is false.")
        if requested_lower == "mps" and not mps():
            raise BackendUnavailableError("MPS was requested but torch.backends.mps.is_available() is false.")
        return requested
    if mps():
        return "mps"
    if cuda():
        return "cuda:0"
    return "cpu"


def normalize_transitions(
    transitions: list[tuple[int, int, float] | list[float]],
    *,
    fps: float,
    min_shot_ms: int,
    include_probs: bool,
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    shots: list[dict[str, Any]] = []
    last_end_frame: int | None = None
    for index, item in enumerate(transitions):
        start_frame = int(item[0])
        end_frame = int(item[1])
        probability = float(item[2]) if len(item) > 2 else None
        shot = build_shot(
            index=index,
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            probability=probability if include_probs else None,
        )
        if shot.duration_ms < min_shot_ms:
            continue
        shots.append(shot.to_dict(include_probs=include_probs))
        last_end_frame = end_frame
    total_frames = last_end_frame
    duration_ms = int(round((last_end_frame / fps) * 1000)) if last_end_frame is not None else None
    return shots, total_frames, duration_ms


def load_transnetv2_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import ffmpeg
        import numpy as np
        import torch
        from transnetv2pt.inference import predict_raw, predictions_to_scenes
        from transnetv2pt.transnetv2_pytorch import TransNetV2
    except Exception as exc:  # pragma: no cover - exercised through error surface
        raise BackendUnavailableError(
            "TransNetV2 backend dependencies are unavailable. Install the transnetv2 extra."
        ) from exc
    return ffmpeg, np, torch, predict_raw, predictions_to_scenes, TransNetV2


def probe_fps(ffmpeg_module: Any, video_path: Path) -> float:
    probe = ffmpeg_module.probe(str(video_path))
    video_info = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
    fps_str = video_info.get("r_frame_rate", "30/1")
    num, den = map(int, fps_str.split('/'))
    return num / den if den else 30.0


def decode_video(ffmpeg_module: Any, np_module: Any, video_path: Path):
    video_stream, _ = ffmpeg_module.input(str(video_path)).output(
        "pipe:", format="rawvideo", pix_fmt="rgb24", s="48x27"
    ).run(capture_stdout=True, capture_stderr=True)
    return np_module.frombuffer(video_stream, np_module.uint8).reshape([-1, 27, 48, 3])


def load_model(TransNetV2: Any, torch_module: Any, device: str):
    model = TransNetV2()
    try:
        with importlib.resources.path('transnetv2pt', 'transnetv2-pytorch-weights.pth') as weights_path:
            model.load_state_dict(torch_module.load(str(weights_path), map_location=torch_module.device(device)))
    except FileNotFoundError as exc:  # pragma: no cover - real package condition
        raise BackendUnavailableError('TransNetV2 weights were not found in the installed package.') from exc
    model.eval()
    return model


def detect_with_transnetv2(
    video_path: Path,
    *,
    threshold: float,
    min_shot_ms: int,
    include_probs: bool,
    device: str | None,
) -> DetectionResult:
    ffmpeg, np, torch, predict_raw, predictions_to_scenes, TransNetV2 = load_transnetv2_dependencies()
    resolved_device = resolve_device(torch, requested=device)
    fps = probe_fps(ffmpeg, video_path)
    video = decode_video(ffmpeg, np, video_path)
    model = load_model(TransNetV2, torch, resolved_device)
    # transnetv2pt emits progress to stdout; redirect it away from the JSON CLI channel.
    with contextlib.redirect_stdout(sys.stderr):
        _, single_frame_pred, _ = predict_raw(model, video, device=torch.device(resolved_device))
        transitions = predictions_to_scenes(single_frame_pred, threshold=threshold, probs=True)
    shots, total_frames, duration_ms = normalize_transitions(
        transitions, fps=fps, min_shot_ms=min_shot_ms, include_probs=include_probs
    )
    cuda_available = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
    return DetectionResult(
        backend="transnetv2",
        model="transnetv2pt",
        fps=fps,
        source_video=str(video_path),
        shots=shots,
        total_frames=total_frames,
        duration_ms=duration_ms,
        resolved_device=resolved_device,
        cuda_available=cuda_available,
        decode_backend="ffmpeg-cpu",
    )
