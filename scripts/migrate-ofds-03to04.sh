#!/usr/bin/env bash

set -u

ROOT="."
APPLY=0
VALIDATE=1

usage() {
  cat <<'EOF'
Usage: migrate-ofds-03to04.sh [options]

Recursively find *_ofds-json_*.json files and run ofds-03to04 for each dataset.
It expects sibling files named:
  *_ofds-nodes_*.geojson
  *_ofds-spans_*.geojson

Options:
  --root DIR       Root directory to scan (default: current directory)
  --apply          Apply changes in place (default is dry run)
  --no-validate    Pass --no-validate to ofds-03to04
  --help           Show this help

Examples:
  # Dry run (default)
  ./scripts/migrate-ofds-03to04.sh --root /path/to/datasets

  # Apply changes
  ./scripts/migrate-ofds-03to04.sh --root /path/to/datasets --apply
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)
      if [[ $# -lt 2 ]]; then
        echo "Error: --root requires a directory argument." >&2
        exit 2
      fi
      ROOT="$2"
      shift 2
      ;;
    --apply)
      APPLY=1
      shift
      ;;
    --no-validate)
      VALIDATE=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown option '$1'" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ ! -d "$ROOT" ]]; then
  echo "Error: root directory not found: $ROOT" >&2
  exit 2
fi

processed=0
skipped=0
failed=0

if [[ $APPLY -eq 0 ]]; then
  echo "Mode: DRY RUN"
else
  echo "Mode: APPLY"
fi
echo "Root: $ROOT"
echo

while IFS= read -r json_file; do
  nodes_file="${json_file/_ofds-json_/_ofds-nodes_}"
  nodes_file="${nodes_file%.json}.geojson"
  spans_file="${json_file/_ofds-json_/_ofds-spans_}"
  spans_file="${spans_file%.json}.geojson"
  tmp_json="${json_file}.tmp"

  if [[ ! -f "$nodes_file" || ! -f "$spans_file" ]]; then
    echo "[SKIP] Missing sibling nodes/spans for: $json_file"
    skipped=$((skipped + 1))
    continue
  fi

  validate_flag="--validate"
  if [[ $VALIDATE -eq 0 ]]; then
    validate_flag="--no-validate"
  fi

  if [[ $APPLY -eq 0 ]]; then
    echo "[DRY RUN] uv run ofds-03to04 --input \"$json_file\" --output \"$tmp_json\" --nodes-geojson-output \"$nodes_file\" --spans-geojson-output \"$spans_file\" $validate_flag"
    echo "[DRY RUN] mv \"$tmp_json\" \"$json_file\""
    processed=$((processed + 1))
    continue
  fi

  echo "[RUN ] $json_file"
  if uv run ofds-03to04 \
      --input "$json_file" \
      --output "$tmp_json" \
      --nodes-geojson-output "$nodes_file" \
      --spans-geojson-output "$spans_file" \
      "$validate_flag"; then
    if mv "$tmp_json" "$json_file"; then
      echo "[ OK ] $json_file"
      processed=$((processed + 1))
    else
      echo "[FAIL] Could not replace original JSON: $json_file" >&2
      failed=$((failed + 1))
      rm -f "$tmp_json"
    fi
  else
    echo "[FAIL] Conversion failed: $json_file" >&2
    failed=$((failed + 1))
    rm -f "$tmp_json"
  fi
done < <(rg --files "$ROOT" -g "**/*_ofds-json_*.json")

echo
echo "Summary:"
echo "  Processed: $processed"
echo "  Skipped:   $skipped"
echo "  Failed:    $failed"

if [[ $failed -gt 0 ]]; then
  exit 1
fi

