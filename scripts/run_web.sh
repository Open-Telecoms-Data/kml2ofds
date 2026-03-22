#!/bin/sh
# Run the kml2ofds web service locally.
# Requires: uv sync --extra web (or pip install -e ".[web]")
cd "$(dirname "$0")/.."
exec uv run uvicorn web.app:app --host 127.0.0.1 --port 8000
