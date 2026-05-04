#!/usr/bin/env bash
set -euo pipefail

command -v ffmpeg >/dev/null
command -v transnetv2-cli >/dev/null

transnetv2-cli >/tmp/transnetv2-cli-root.json
python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('/tmp/transnetv2-cli-root.json').read_text())
assert payload['ok'] is True
assert payload['command']['resolved']['executable'] == 'transnetv2-cli'
assert payload['result']['description'] == 'TransNetV2 shot boundary detection CLI'
PY

echo "ffmpeg: $(ffmpeg -version | head -n 1)"
echo "transnetv2-cli: ok"
