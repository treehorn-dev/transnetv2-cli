from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Shot:
    index: int
    start_ms: int
    end_ms: int
    duration_ms: int
    start_frame: int | None = None
    end_frame: int | None = None
    probability: float | None = None

    def to_dict(self, include_probs: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "index": self.index,
            "start_ms": self.start_ms,
            "end_ms": self.end_ms,
            "duration_ms": self.duration_ms,
        }
        if self.start_frame is not None:
            payload["start_frame"] = self.start_frame
        if self.end_frame is not None:
            payload["end_frame"] = self.end_frame
        if include_probs and self.probability is not None:
            payload["probability"] = self.probability
        return payload


def build_shot(
    *,
    index: int,
    start_frame: int,
    end_frame: int,
    fps: float,
    probability: float | None = None,
) -> Shot:
    start_ms = int(round((start_frame / fps) * 1000))
    end_ms = int(round((end_frame / fps) * 1000))
    duration_ms = max(0, end_ms - start_ms)
    return Shot(
        index=index,
        start_ms=start_ms,
        end_ms=end_ms,
        duration_ms=duration_ms,
        start_frame=start_frame,
        end_frame=end_frame,
        probability=probability,
    )


def build_shots_payload(
    *,
    input_path: Path,
    backend: str,
    model: str,
    fps: float,
    shots: list[Shot],
    total_frames: int | None = None,
    duration_ms: int | None = None,
) -> dict[str, Any]:
    return {
        "schema": "shot-intel/shots@v1",
        "backend": backend,
        "model": model,
        "source_video": str(input_path),
        "fps": fps,
        "total_frames": total_frames,
        "duration_ms": duration_ms,
        "shots": [shot.to_dict() for shot in shots],
    }
