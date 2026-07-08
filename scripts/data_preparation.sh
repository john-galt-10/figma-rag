#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="figma-navigator"

RAW_OUTPUT_FOLDER_PATH="data/raw/figma_docs"
RAW_HTML_FOLDER_NAME="help_center"
RAW_HTMLS_MANIFEST_PATH="${RAW_OUTPUT_FOLDER_PATH}/manifest.jsonl"

PROCESSED_OUTPUT_FOLDER_PATH="data/processed/figma_docs"
PROCESSED_MD_FOLDER_NAME="help_center_mds"


if command -v python >/dev/null 2>&1; then
  PYTHON_CMD=(python)
elif command -v python.exe >/dev/null 2>&1; then
  PYTHON_CMD=(python.exe)
elif command -v conda >/dev/null 2>&1; then
  PYTHON_CMD=(conda run -n "$CONDA_ENV_NAME" python)
else
  echo "Python executable not found. Activate conda or install Python on this shell PATH." >&2
  exit 1
fi

echo "Python command: ${PYTHON_CMD[*]}"

"${PYTHON_CMD[@]}" scripts/download_help_center.py \
  --output-dir "${RAW_OUTPUT_FOLDER_PATH}/${RAW_HTML_FOLDER_NAME}" \
  --manifest-path "$RAW_HTMLS_MANIFEST_PATH"


"${PYTHON_CMD[@]}" scripts/convert_help_center_to_markdown.py \
  --manifest-path "$RAW_HTMLS_MANIFEST_PATH" \
  --output-dir "${PROCESSED_OUTPUT_FOLDER_PATH}/${PROCESSED_MD_FOLDER_NAME}" \
  --processed-manifest-path "${PROCESSED_OUTPUT_FOLDER_PATH}/manifest.jsonl"
