"""Chunk processed Figma documentation for retrieval."""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.chunking import available_strategies, chunk_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Chunk processed Figma Markdown documents for retrieval."
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=REPO_ROOT / "data" / "processed" / "figma_docs" / "manifest.jsonl",
        help="Path to the processed JSONL manifest.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Path to the generated chunk JSONL corpus. By default, a versioned "
            "filename is built from the strategy, model, parameters, and timestamp."
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=available_strategies(),
        default="hierarchical",
        help="Chunking strategy to use (default: hierarchical).",
    )
    parser.add_argument(
        "--model",
        # default="Alibaba-NLP/gte-modernbert-base",
       default="BAAI/bge-small-en-v1.5",
        help="Hugging Face model whose tokenizer defines token limits.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        # default=600, # 320
        default=320,
        help="Maximum tokens in final embedding text (default: 320).",
    )
    parser.add_argument(
        "--overlap-tokens",
        type=int,
        # default=60, # 40
        default=40,
        help="Approximate overlap for split sections (default: 40).",
    )
    return parser


def build_default_output_path(
    strategy: str,
    model: str,
    max_tokens: int,
    overlap_tokens: int,
) -> Path:
    """Build a readable, collision-resistant filename for one chunking run."""

    model_name = model.rstrip("/").rsplit("/", maxsplit=1)[-1]
    model_slug = re.sub(r"[^A-Za-z0-9.-]+", "-", model_name).strip("-.").lower()
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M")
    filename = (
        f"chunks_{strategy}_{model_slug}_t{max_tokens}_o{overlap_tokens}_"
        f"{timestamp}.jsonl"
    )
    # return REPO_ROOT / "data" / "processed" / "figma_docs" / filename
    return REPO_ROOT / "data" / "processed_w_breadcrumbs" / "figma_docs" / filename


def main() -> int:
    args = build_parser().parse_args()
    output_path = args.output_path or build_default_output_path(
        strategy=args.strategy,
        model=args.model,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    summary = chunk_corpus(
        manifest_path=args.manifest_path,
        output_path=output_path,
        repository_root=REPO_ROOT,
        strategy_name=args.strategy,
        model_name=args.model,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
    )

    for failure in summary.failures:
        print(
            f"Failed: {failure.document_id} ({failure.processed_file_path}): "
            f"{failure.error}",
            file=sys.stderr,
        )

    print(f"Strategy: {summary.strategy}")
    print(f"Manifest documents: {summary.total_documents}")
    print(f"Chunked documents: {summary.chunked_documents}")
    print(f"Failed documents: {summary.failed_documents}")
    print(f"Chunks: {summary.total_chunks}")
    print(
        "Token distribution: "
        f"min={summary.min_tokens}, median={summary.median_tokens:g}, "
        f"p90={summary.p90_tokens}, p95={summary.p95_tokens}, "
        f"max={summary.max_tokens}"
    )
    print(f"Output: {summary.output_path.as_posix()}")
    return 1 if summary.failed_documents else 0


if __name__ == "__main__":
    raise SystemExit(main())
