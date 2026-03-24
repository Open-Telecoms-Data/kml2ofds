#!/bin/sh
# Run the kml2ofds web service locally.
# Requires: uv sync --extra web (or pip install -e ".[web]")
# Tries: .venv from uv, then uv run, then python3 -m uvicorn (for systemd / minimal PATH).
cd "$(dirname "$0")/.."
if [ -x ".venv/bin/uvicorn" ]; then
    exec .venv/bin/uvicorn web.app:app --host 127.0.0.1 --port 8000
elif command -v uv >/dev/null 2>&1; then
    exec uv run uvicorn web.app:app --host 127.0.0.1 --port 8000
else
    exec python3 -m uvicorn web.app:app --host 127.0.0.1 --port 8000
fi
