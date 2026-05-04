# transnetv2-cli

Standalone agent-friendly CLI for shot boundary detection with a TransNetV2 backend.

## Current Scope

This repo currently defines:
- a machine-readable root command contract
- a `detect` command contract
- a reusable `shots.json` output schema
- a backend seam for TransNetV2 integration

The initial backend wiring is intentionally narrow. It supports a single backend identity: `transnetv2`.

## CLI

Show the command tree:

```bash
transnetv2-cli
```

Detect shots:

```bash
transnetv2-cli detect --input /path/to/video.mp4 --output /path/to/video.shots.json
```

## Output Contract

Successful commands return a JSON envelope with:
- `ok`
- `command.raw`
- `command.parsed`
- `command.resolved`
- `result`
- `next_actions`

## Development

```bash
make test
```

## Backend

The current backend identity is fixed to `transnetv2`.

To enable the real backend stack later:

```bash
pip install -e .[transnetv2]
```

## CPU Image

Build the standalone CPU image:

```bash
make build-cpu
```

Smoke test it:

```bash
make smoke-cpu
```
