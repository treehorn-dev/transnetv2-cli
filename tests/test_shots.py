from pathlib import Path

from transnetv2_cli.shots import build_shot, build_shots_payload


def test_build_shot_converts_frames_to_ms():
    shot = build_shot(index=0, start_frame=25, end_frame=50, fps=25.0, probability=0.8)

    assert shot.start_ms == 1000
    assert shot.end_ms == 2000
    assert shot.duration_ms == 1000
    assert shot.probability == 0.8


def test_build_shots_payload_emits_expected_schema():
    shot = build_shot(index=0, start_frame=0, end_frame=24, fps=24.0)
    payload = build_shots_payload(
        input_path=Path("/tmp/video.mp4"),
        backend="transnetv2",
        model="transnetv2pt",
        fps=24.0,
        shots=[shot],
        total_frames=240,
        duration_ms=10000,
    )

    assert payload["schema"] == "shot-intel/shots@v1"
    assert payload["backend"] == "transnetv2"
    assert payload["model"] == "transnetv2pt"
    assert payload["shots"][0]["start_ms"] == 0
