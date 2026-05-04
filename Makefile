.PHONY: test build-cpu smoke-cpu

test:
	. .venv/bin/activate && python -m pytest -q

build-cpu:
	docker build -t transnetv2-cli:cpu -f Dockerfile.cpu .

smoke-cpu:
	docker run --rm --entrypoint bash transnetv2-cli:cpu /app/scripts/smoke-cpu.sh
