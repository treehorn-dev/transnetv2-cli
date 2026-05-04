from pathlib import Path


def test_repo_has_initial_layout() -> None:
    assert Path("pyproject.toml").exists()
    assert Path("README.md").exists()
    assert Path("Makefile").exists()
    assert Path("Dockerfile.cpu").exists()
    assert Path("scripts/smoke-cpu.sh").exists()
    assert Path("transnetv2_cli/cli.py").exists()
    assert Path("transnetv2_cli/backend.py").exists()
    assert Path("transnetv2_cli/shots.py").exists()
    assert Path("tests/test_cli.py").exists()
    assert Path("tests/test_shots.py").exists()


def test_pyproject_declares_script_and_pytest() -> None:
    text = Path("pyproject.toml").read_text()

    assert 'transnetv2-cli = "transnetv2_cli.cli:main_entry"' in text
    assert '[tool.pytest.ini_options]' in text


def test_readme_mentions_detect_contract() -> None:
    text = Path("README.md").read_text()

    assert 'transnetv2-cli detect --input /path/to/video.mp4' in text
    assert 'shots.json' in text


def test_dockerfile_and_makefile_expose_cpu_image() -> None:
    dockerfile = Path("Dockerfile.cpu").read_text()
    makefile = Path("Makefile").read_text()

    assert "FROM python:3.11-slim-bookworm" in dockerfile
    assert "ffmpeg" in dockerfile
    assert "pip install --no-cache-dir -e .[transnetv2]" in dockerfile
    assert 'ENTRYPOINT ["transnetv2-cli"]' in dockerfile
    assert "build-cpu:" in makefile
    assert "smoke-cpu:" in makefile
    assert "docker build -t transnetv2-cli:cpu -f Dockerfile.cpu ." in makefile
