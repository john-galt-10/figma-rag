"""Entrypoint for building a persistent LlamaIndex BM25 index."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.indexing import build_bm25_index, build_default_bm25_index_name

DEFAULT_CHUNKS_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "figma_docs"
    / "chunks_hierarchical_bge-small-en-v1.5_t320_o40_20260630-1601.jsonl"
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command line parser for BM25 indexing."""

    parser = argparse.ArgumentParser(
        description="Build a persistent BM25 keyword index from Figma chunk JSONL."
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the chunk JSONL file to index.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "bm25",
        help="Root directory where BM25 indexes are persisted.",
    )
    parser.add_argument(
        "--index-name",
        default=None,
        help=(
            "Name of the BM25 index directory to create or update. By default, "
            "a name is built from the chunking artifact, stemming settings, and "
            "current UTC timestamp."
        ),
    )
    parser.add_argument(
        "--disable-stemming",
        action="store_true",
        help="Build the BM25 index without PyStemmer stemming.",
    )
    parser.add_argument(
        "--stemmer-language",
        default="english",
        help="PyStemmer language to use when stemming is enabled.",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help=(
            "Delete only the selected BM25 index directory before rebuilding it. "
            "Other BM25 indexes are left untouched."
        ),
    )
    return parser


def main() -> int:
    """Build a BM25 index and print the persisted artifact summary."""

    args = build_parser().parse_args()
    stemming_enabled = not args.disable_stemming
    index_name = args.index_name or build_default_bm25_index_name(
        chunks_path=args.chunks_path,
        stemming_enabled=stemming_enabled,
        stemmer_language=args.stemmer_language,
    )

    summary = build_bm25_index(
        chunks_path=args.chunks_path,
        persist_dir=args.persist_dir,
        index_name=index_name,
        stemming_enabled=stemming_enabled,
        stemmer_language=args.stemmer_language,
        recreate=args.recreate,
    )

    print(f"Chunks path: {summary.chunks_path.as_posix()}")
    print(f"Persist directory: {summary.persist_dir.as_posix()}")
    print(f"Index name: {summary.index_name}")
    print(f"Index directory: {summary.index_dir.as_posix()}")
    print(f"Recreated index: {summary.recreated}")
    print(f"Chunks indexed: {summary.total_chunks}")
    print(f"Stemming enabled: {summary.stemming_enabled}")
    print(f"Stemmer language: {summary.stemmer_language or 'N.A.'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
