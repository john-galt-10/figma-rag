#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV_NAME="figma-navigator"

PROCESSED_OUTPUT_FOLDER_PATH="data/processed/figma_docs"
PROCESSED_MANIFEST_PATH="data/processed/figma_docs/manifest.jsonl"

CHUNK_STRATEGY="hierarchical"
CHUNK_MODEL="BAAI/bge-small-en-v1.5"
CHUNK_MAX_TOKENS="320"
CHUNK_OVERLAP_TOKENS="40"
CHUNKS_FILE_NAME="chunks_hierarchical_bge-small-en-v1.5_t320_o40.jsonl"
CHUNKS_PATH="${PROCESSED_OUTPUT_FOLDER_PATH}/indexing/${CHUNKS_FILE_NAME}"

CHROMA_PERSIST_DIR="${PROCESSED_OUTPUT_FOLDER_PATH}/indexing/chroma"
CHROMA_COLLECTION_NAME="hierarchical-bge"
EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"

BM25_PERSIST_DIR="${PROCESSED_OUTPUT_FOLDER_PATH}/indexing/bm25"
BM25_INDEX_NAME="hierarchical-bge-stemmed-english"
BM25_STEMMER_LANGUAGE="english"

command_runs() {
  "$@" --version >/dev/null 2>&1
}

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

echo "[index_preparation.sh] Step 1/3: chunking processed Markdown documents"
"${PYTHON_CMD[@]}" scripts/chunk_figma_docs.py \
  --manifest-path "$PROCESSED_MANIFEST_PATH" \
  --output-path "$CHUNKS_PATH" \
  --strategy "$CHUNK_STRATEGY" \
  --model "$CHUNK_MODEL" \
  --max-tokens "$CHUNK_MAX_TOKENS" \
  --overlap-tokens "$CHUNK_OVERLAP_TOKENS"

echo "[index_preparation.sh] Step 2/3: building Chroma vector index"
"${PYTHON_CMD[@]}" scripts/build_vector_index.py \
  --chunks-path "$CHUNKS_PATH" \
  --persist-dir "$CHROMA_PERSIST_DIR" \
  --collection-name "$CHROMA_COLLECTION_NAME" \
  --model "$EMBEDDING_MODEL" \
  --recreate

echo "[index_preparation.sh] Step 3/3: building BM25 keyword index"
"${PYTHON_CMD[@]}" scripts/build_bm25_index.py \
  --chunks-path "$CHUNKS_PATH" \
  --persist-dir "$BM25_PERSIST_DIR" \
  --index-name "$BM25_INDEX_NAME" \
  --stemmer-language "$BM25_STEMMER_LANGUAGE" \
  --recreate
