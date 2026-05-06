import json
from pathlib import Path

import transnetv2_cli.cli as cli
from transnetv2_cli.backend import DetectionResult


def test_root_command_returns_command_tree(capsys):
    exit_code = cli.main([])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["command"]["parsed"]["path"] == []
    assert payload["result"]["description"] == "TransNetV2 shot boundary detection CLI"
    assert any(command["name"] == "detect" for command in payload["result"]["commands"])


def test_detect_rejects_unsupported_backend(capsys, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")

    exit_code = cli.main(["detect", "--input", str(video), "--backend", "othernet"])
    assert exit_code == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "UNSUPPORTED_BACKEND"


def test_detect_rejects_missing_input(capsys):
    exit_code = cli.main(["detect", "--input", "/tmp/does-not-exist.mp4"])
    assert exit_code == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INPUT_NOT_FOUND"


def test_detect_writes_shots_json(monkeypatch, capsys, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "clip.shots.json"

    monkeypatch.setattr(
        cli,
        "detect_with_transnetv2",
        lambda *args, **kwargs: DetectionResult(
            backend="transnetv2",
            model="transnetv2pt",
            fps=25.0,
            source_video=str(video),
            total_frames=100,
            duration_ms=4000,
            resolved_device="cuda",
            cuda_available=True,
            decode_backend="ffmpeg-cpu",
            shots=[
                {
                    "index": 0,
                    "start_ms": 0,
                    "end_ms": 1000,
                    "duration_ms": 1000,
                    "start_frame": 0,
                    "end_frame": 25,
                    "probability": 0.9,
                }
            ],
        ),
    )

    exit_code = cli.main(["detect", "--input", str(video), "--output", str(output), "--include-probs"])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["result"]["output"] == str(output)
    assert payload["result"]["resolved_device"] == "cuda"
    assert payload["result"]["cuda_available"] is True
    assert payload["result"]["decode_backend"] == "ffmpeg-cpu"
    assert output.exists()

    artifact = json.loads(output.read_text())
    assert artifact["schema"] == "shot-intel/shots@v1"
    assert artifact["backend"] == "transnetv2"
    assert len(artifact["shots"]) == 1


def test_detect_output_is_clean_json_even_if_backend_prints(monkeypatch, capsys, tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    output = tmp_path / "clip.shots.json"

    def fake_detect(*args, **kwargs):
        print("backend noise that should not be on stdout")
        return DetectionResult(
            backend="transnetv2",
            model="transnetv2pt",
            fps=25.0,
            source_video=str(video),
            total_frames=100,
            duration_ms=4000,
            resolved_device="cpu",
            cuda_available=False,
            decode_backend="ffmpeg-cpu",
            shots=[
                {
                    "index": 0,
                    "start_ms": 0,
                    "end_ms": 1000,
                    "duration_ms": 1000,
                }
            ],
        )

    monkeypatch.setattr(cli, "detect_with_transnetv2", fake_detect)
    exit_code = cli.main(["detect", "--input", str(video), "--output", str(output)])
    assert exit_code == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
