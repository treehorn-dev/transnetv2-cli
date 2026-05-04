from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
from pathlib import Path
from typing import Any

from transnetv2_cli import __version__
from transnetv2_cli.backend import BackendUnavailableError, detect_with_transnetv2
from transnetv2_cli.shots import Shot, build_shots_payload

SUPPORTED_BACKENDS = {"transnetv2"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    subparsers = parser.add_subparsers(dest="command")

    detect_parser = subparsers.add_parser("detect", add_help=False)
    detect_parser.add_argument("--input", required=True)
    detect_parser.add_argument("--output")
    detect_parser.add_argument("--backend", default="transnetv2")
    detect_parser.add_argument("--threshold", type=float, default=0.5)
    detect_parser.add_argument("--min-shot-ms", type=int, default=0)
    detect_parser.add_argument("--device")
    detect_parser.add_argument("--include-probs", action="store_true")

    return parser


def main(argv: list[str] | None = None, executable: str = "transnetv2-cli") -> int:
    argv = list(argv or [])
    raw = command_raw(executable, argv)
    if not argv:
        emit(root_payload(raw, executable))
        return 0

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        emit(
            error_response(
                raw,
                {"path": argv[:1], "options": {}, "flags": {}},
                "INVALID_ARGUMENTS",
                "Invalid command arguments.",
                f"Run {executable} with no arguments to inspect supported commands.",
                [{"command": executable, "description": "Show the root command tree."}],
                executable,
            )
        )
        return 2

    if args.command == "detect":
        return detect_main(args, raw, executable)

    emit(root_payload(raw, executable))
    return 0


def main_entry() -> None:
    raise SystemExit(main(sys.argv[1:], executable="transnetv2-cli"))


def detect_main(args: argparse.Namespace, raw: str, executable: str) -> int:
    parsed = {
        "path": ["detect"],
        "options": {
            "input": args.input,
            "output": args.output,
            "backend": args.backend,
            "threshold": args.threshold,
            "min_shot_ms": args.min_shot_ms,
            "device": args.device,
        },
        "flags": {
            "include_probs": bool(args.include_probs),
        },
    }

    if args.backend not in SUPPORTED_BACKENDS:
        emit(
            error_response(
                raw,
                parsed,
                "UNSUPPORTED_BACKEND",
                f"Unsupported backend: {args.backend}",
                "Use --backend transnetv2. That is the only supported backend right now.",
                [
                    {
                        "command": f"{executable} detect --input {args.input} --backend transnetv2",
                        "description": "Run the supported TransNetV2 detection path.",
                    }
                ],
                executable,
            )
        )
        return 1

    input_path = Path(args.input)
    if not input_path.exists():
        emit(
            error_response(
                raw,
                parsed,
                "INPUT_NOT_FOUND",
                f"Input video not found: {input_path}",
                "Pass an existing file path to --input.",
                [],
                executable,
            )
        )
        return 1

    output_path = Path(args.output) if args.output else input_path.with_name(f"{input_path.stem}.shots.json")

    try:
        # Backends may print progress; keep stdout clean for the JSON envelope.
        with contextlib.redirect_stdout(io.StringIO()):
            detection = detect_with_transnetv2(
                input_path,
                threshold=args.threshold,
                min_shot_ms=args.min_shot_ms,
                include_probs=args.include_probs,
                device=args.device,
            )
    except BackendUnavailableError as exc:
        emit(
            error_response(
                raw,
                parsed,
                "BACKEND_UNAVAILABLE",
                str(exc),
                "Wire the TransNetV2 backend in this repo before using detect.",
                [],
                executable,
            )
        )
        return 1

    shots = [Shot(**shot) if not isinstance(shot, Shot) else shot for shot in detection.shots]
    payload = build_shots_payload(
        input_path=input_path,
        backend=detection.backend,
        model=detection.model,
        fps=detection.fps,
        shots=shots,
        total_frames=detection.total_frames,
        duration_ms=detection.duration_ms,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n")

    emit(
        success_response(
            raw,
            {
                **parsed,
                "options": {**parsed["options"], "output": str(output_path)},
            },
            {
                "backend": detection.backend,
                "model": detection.model,
                "output": str(output_path),
                "shots": len(shots),
                "schema": payload["schema"],
            },
            [
                {
                    "command": f"cat {output_path}",
                    "description": "Inspect the generated shots artifact.",
                }
            ],
            executable,
        )
    )
    return 0


def root_payload(raw: str, executable: str) -> dict[str, Any]:
    return success_response(
        raw,
        {"path": [], "options": {}, "flags": {}},
        {
            "description": "TransNetV2 shot boundary detection CLI",
            "commands": [
                {
                    "name": "detect",
                    "description": "Detect shot boundaries and write a shots.json artifact.",
                    "usage": f"{executable} detect --input <video> [--output <shots.json>]",
                }
            ],
        },
        [
            {
                "command": f"{executable} detect --input /path/to/video.mp4",
                "description": "Run the supported TransNetV2 shot detection path.",
            }
        ],
        executable,
    )


def success_response(raw: str, parsed: dict[str, Any], result: dict[str, Any], next_actions: list[dict[str, str]], executable: str) -> dict[str, Any]:
    return {
        "ok": True,
        "command": {
            "raw": raw,
            "parsed": parsed,
            "resolved": {
                "executable": executable,
                "cwd": os.getcwd(),
                "version": __version__,
            },
        },
        "result": result,
        "next_actions": next_actions,
    }


def error_response(raw: str, parsed: dict[str, Any], code: str, message: str, fix: str, next_actions: list[dict[str, str]], executable: str) -> dict[str, Any]:
    return {
        "ok": False,
        "command": {
            "raw": raw,
            "parsed": parsed,
            "resolved": {
                "executable": executable,
                "cwd": os.getcwd(),
                "version": __version__,
            },
        },
        "error": {"code": code, "message": message},
        "fix": fix,
        "next_actions": next_actions,
    }


def command_raw(executable: str, argv: list[str]) -> str:
    return " ".join([executable, *argv]).strip()


def emit(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))
