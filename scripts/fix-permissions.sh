#!/bin/sh
# Fix permissions for SFTP group-writable deployment.
# Run as root after deploy. EXCLUDES .venv so executables stay runnable.
cd "$(dirname "$0")/.." || exit 1
find . -path ./.venv -prune -o -path ./.git -prune -o -type d -print | xargs chmod 775
find . -path ./.venv -prune -o -path ./.git -prune -o -type f -print | xargs chmod 664
chown -R www-data:www-data .
echo "Done. .venv left unchanged so uvicorn/python stay executable."
