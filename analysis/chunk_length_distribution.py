"""Analyze and plot the token length distribution of chunk JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHUNKS_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "figma_docs"
    / "chunks_hierarchical_bge-small-en-v1.5_t320_o40_20260629-1709.jsonl"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "analysis" / "plots" / "chunk_length_distribution.png"
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the analysis script."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute token length statistics for a chunk JSONL file and plot "
            "standard plus cumulative histograms."
        )
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Path to the input chunk JSONL file.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=40,
        help="Number of bins to use in the histograms.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the histogram PNG will be written.",
    )
    return parser


def token_length_from_record(record: dict[str, Any], line_number: int) -> int:
    """Return a record token length, falling back to whitespace text count."""
    token_count = record.get("token_count")
    if isinstance(token_count, int) and token_count >= 0:
        return token_count

    text = record.get("text")
    if isinstance(text, str):
        return len(text.split())

    raise ValueError(
        f"Line {line_number} has no non-negative integer token_count "
        "and no text field for fallback counting."
    )


def read_token_lengths(chunks_path: Path) -> list[int]:
    """Read token lengths from a JSONL chunk artifact."""
    token_lengths: list[int] = []
    fallback_count = 0

    with chunks_path.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                record = json.loads(stripped_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Line {line_number} is not valid JSON: {error}") from error

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} is not a JSON object.")

            used_token_count = isinstance(record.get("token_count"), int)
            token_lengths.append(token_length_from_record(record, line_number))
            if not used_token_count:
                fallback_count += 1

    if fallback_count:
        print(f"Warning: used whitespace fallback for {fallback_count} chunks.")

    return token_lengths


def percentile(sorted_values: list[int], quantile: float) -> float:
    """Compute a linearly interpolated percentile from sorted values."""
    if not sorted_values:
        raise ValueError("Cannot compute percentiles for an empty value list.")

    position = quantile * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value + (upper_value - lower_value) * fraction


def print_summary(token_lengths: list[int]) -> None:
    """Print compact descriptive statistics for chunk token lengths."""
    if not token_lengths:
        raise ValueError("No chunks were found in the input JSONL file.")

    sorted_lengths = sorted(token_lengths)

    print("Chunk token length distribution")
    print(f"Chunks: {len(token_lengths)}")
    print(f"Min: {min(token_lengths):.0f}")
    print(f"Max: {max(token_lengths):.0f}")
    print(f"Mean: {statistics.fmean(token_lengths):.2f}")
    print(f"Median: {statistics.median(token_lengths):.2f}")
    print("")
    print("Quantiles:")
    for quantile in QUANTILES:
        value = percentile(sorted_lengths, quantile)
        print(f"  p{int(quantile * 100):02d}: {value:.2f}")


def plot_histograms(token_lengths: list[int], bins: int, output_path: Path) -> None:
    """Write a PNG with regular and cumulative token length histograms."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("")
        print("matplotlib is not installed; skipping PNG plot generation.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5), constrained_layout=True)
    figure.suptitle("Chunk Token Length Distribution")

    axes[0].hist(token_lengths, bins=bins, color="#4c78a8", edgecolor="white")
    axes[0].set_title("Histogram")
    axes[0].set_xlabel("Tokens per chunk")
    axes[0].set_ylabel("Chunk count")

    axes[1].hist(
        token_lengths,
        bins=bins,
        cumulative=True,
        color="#f58518",
        edgecolor="white",
    )
    axes[1].set_title("Cumulative histogram")
    axes[1].set_xlabel("Tokens per chunk")
    axes[1].set_ylabel("Cumulative chunk count")

    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    print("")
    print(f"Plot written to: {output_path}")


def main() -> int:
    """Run the chunk length analysis command."""
    args = build_parser().parse_args()

    if args.bins <= 0:
        raise ValueError("--bins must be greater than zero.")

    token_lengths = read_token_lengths(args.chunks_path)
    print_summary(token_lengths)
    plot_histograms(
        token_lengths=token_lengths,
        bins=args.bins,
        output_path=args.output_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
