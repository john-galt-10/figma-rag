"""Plot retrieval metrics comparisons from aggregate evaluation JSON files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "analysis" / "plots"
)
FILENAME_TIMESTAMP_PATTERN = re.compile(r"(\d{8}T\d{4})")


@dataclass(frozen=True)
class MetricsRun:
    """One retrieval metrics artifact loaded for plotting."""

    path: Path
    label: str
    metadata: dict[str, Any]
    metrics_by_k: dict[str, dict[str, float]]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for retrieval metrics plotting."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped bar charts and metric-vs-k line charts comparing "
            "retrieval metrics across one or more aggregate metrics JSON files."
        )
    )
    parser.add_argument(
        "metrics_files",
        type=Path,
        nargs="*",
        help="Aggregate retrieval metrics JSON files to compare.",
    )
    parser.add_argument(
        "--metrics-list",
        type=Path,
        default=None,
        help=(
            ".lst file containing one metrics JSON path per line. Blank lines "
            "and lines starting with # are ignored. Relative paths are resolved "
            "from the .lst file directory."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where timestamped comparison PNG files are written.",
    )
    parser.add_argument(
        "--k",
        type=int,
        nargs="+",
        default=None,
        help="Specific k values to plot. Defaults to the union found in the inputs.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help=(
            "Specific metric names to plot. Defaults to metrics common to all "
            "loaded runs, preserving the first file's metric order."
        ),
    )
    parser.add_argument(
        "--title-prefix",
        default=None,
        help="Optional prefix added to each plot title.",
    )
    parser.add_argument(
        "--disable-bar-plots",
        "--disable-histogram-plots",
        dest="disable_bar_plots",
        action="store_true",
        help="Do not write grouped bar charts for each selected k value.",
    )
    parser.add_argument(
        "--disable-line-plots",
        action="store_true",
        help="Do not write metric-vs-k line charts for each selected metric.",
    )
    return parser


def resolve_metrics_paths(
    metrics_files: list[Path],
    metrics_list: Path | None,
) -> list[Path]:
    """Resolve input metrics paths from direct arguments or a .lst file."""

    has_direct_paths = bool(metrics_files)
    has_list_path = metrics_list is not None
    if has_direct_paths == has_list_path:
        raise ValueError(
            "Provide exactly one input mode: direct JSON paths or --metrics-list."
        )

    if has_direct_paths:
        return [path.resolve() for path in metrics_files]

    assert metrics_list is not None
    if metrics_list.suffix.lower() != ".lst":
        raise ValueError("--metrics-list must point to a .lst file.")
    if not metrics_list.exists():
        raise FileNotFoundError(f"Metrics list file not found: {metrics_list}")

    base_dir = metrics_list.resolve().parent
    paths = []
    with metrics_list.open("r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith("#"):
                continue

            path = Path(stripped_line)
            if not path.is_absolute():
                path = base_dir / path
            paths.append(path.resolve())

    if not paths:
        raise ValueError(f"No metrics JSON paths found in {metrics_list}.")
    return paths


def load_metrics_runs(paths: list[Path]) -> list[MetricsRun]:
    """Load and validate retrieval metrics JSON files."""

    runs = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Metrics file not found: {path}")
        if not path.is_file():
            raise ValueError(f"Metrics path is not a file: {path}")

        with path.open("r", encoding="utf-8") as file:
            try:
                payload = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

        if not isinstance(payload, dict):
            raise ValueError(f"Metrics file must contain a JSON object: {path}")

        metadata = payload.get("metadata")
        metrics_by_k = payload.get("metrics_by_k")
        if not isinstance(metadata, dict):
            raise ValueError(f"Missing or invalid metadata object in {path}")
        if not isinstance(metrics_by_k, dict):
            raise ValueError(f"Missing or invalid metrics_by_k object in {path}")

        runs.append(
            MetricsRun(
                path=path,
                label=build_run_label(metadata),
                metadata=metadata,
                metrics_by_k=parse_metrics_by_k(metrics_by_k, path),
            )
        )

    if not runs:
        raise ValueError("No metrics files were loaded.")
    return sorted(
        make_unique_run_labels(runs),
        key=lambda run: run.label.casefold(),
    )


def parse_metrics_by_k(
    raw_metrics_by_k: dict[str, Any],
    path: Path,
) -> dict[str, dict[str, float]]:
    """Validate and normalize metrics_by_k values."""

    metrics_by_k: dict[str, dict[str, float]] = {}
    for raw_k, raw_metrics in raw_metrics_by_k.items():
        k = str(raw_k)
        if not isinstance(raw_metrics, dict):
            raise ValueError(f"metrics_by_k[{k!r}] must be an object in {path}")

        metrics = {}
        for metric_name, raw_value in raw_metrics.items():
            if not isinstance(metric_name, str):
                raise ValueError(f"Metric names must be strings in {path}")
            if not isinstance(raw_value, int | float):
                raise ValueError(
                    f"Metric value for {metric_name!r} at k={k} must be numeric in {path}"
                )
            metrics[metric_name] = float(raw_value)
        metrics_by_k[k] = metrics

    if not metrics_by_k:
        raise ValueError(f"No k values found in metrics_by_k for {path}")
    return metrics_by_k


def build_run_label(metadata: dict[str, Any]) -> str:
    """Build a compact run label from retrieval metadata."""

    pipeline = metadata.get("retrieval_pipeline")
    if not isinstance(pipeline, dict):
        pipeline = {}

    components = pipeline.get("components")
    if isinstance(components, list) and components:
        component_label = "+".join(str(component) for component in components)
    else:
        component_label = "components=unknown"

    pipeline_reranking = pipeline.get("reranking")
    metadata_reranking = metadata.get("reranking")
    if not isinstance(pipeline_reranking, dict):
        pipeline_reranking = {}
    if not isinstance(metadata_reranking, dict):
        metadata_reranking = {}

    reranking_enabled = bool(
        metadata_reranking.get("enabled", pipeline_reranking.get("enabled", False))
    )
    aggregation_strategy = str(pipeline.get("aggregation_strategy", "unknown"))

    label_parts = [
        component_label,
        f"rerank={'yes' if reranking_enabled else 'no'}",
        f"agg={aggregation_strategy}",
    ]

    component_weights = pipeline.get("component_weights")
    if aggregation_strategy == "weighted_rrf" and isinstance(component_weights, dict):
        label_parts.append(f"w={format_component_weights(component_weights)}")

    if reranking_enabled:
        reranker_model = metadata_reranking.get(
            "model",
            pipeline_reranking.get("model"),
        )
        if reranker_model:
            label_parts.append(f"rr={short_model_name(str(reranker_model))}")

    candidate_k = metadata_reranking.get("candidate_k")
    label_parts.append(f"cand={candidate_k if candidate_k is not None else 'none'}")
    return " | ".join(label_parts)


def format_component_weights(component_weights: dict[str, Any]) -> str:
    """Format component weights for inclusion in a run label."""

    formatted_weights = []
    for component_name in sorted(component_weights):
        raw_weight = component_weights[component_name]
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            formatted_weight = str(raw_weight)
        else:
            formatted_weight = f"{weight:g}"
        formatted_weights.append(f"({component_name} {formatted_weight})")
    return "-".join(formatted_weights)


def short_model_name(model_name: str) -> str:
    """Return the final path segment of a model name."""

    return model_name.rstrip("/").split("/")[-1] or model_name


def make_unique_run_labels(runs: list[MetricsRun]) -> list[MetricsRun]:
    """Append deterministic suffixes when generated run labels collide."""

    duplicated_labels = {
        run.label for run in runs if sum(other.label == run.label for other in runs) > 1
    }
    used_labels = set()
    label_counts: dict[str, int] = {}
    unique_runs = []
    for run in runs:
        occurrence = label_counts.get(run.label, 0) + 1
        label_counts[run.label] = occurrence

        if run.label not in duplicated_labels:
            unique_runs.append(run)
            used_labels.add(run.label)
            continue

        suffix = filename_timestamp_suffix(run.path) or f"run{occurrence}"
        candidate_label = f"{run.label} | {suffix}"
        if candidate_label in used_labels:
            candidate_label = f"{run.label} | run{occurrence}"

        unique_runs.append(
            MetricsRun(
                path=run.path,
                label=candidate_label,
                metadata=run.metadata,
                metrics_by_k=run.metrics_by_k,
            )
        )
        used_labels.add(candidate_label)
    return unique_runs


def filename_timestamp_suffix(path: Path) -> str | None:
    """Extract a compact timestamp suffix from a metrics filename."""

    match = FILENAME_TIMESTAMP_PATTERN.search(path.stem)
    if not match:
        return None
    return match.group(1)


def select_k_values(runs: list[MetricsRun], requested_k: list[int] | None) -> list[str]:
    """Return sorted k values selected for plotting."""

    if requested_k is not None:
        if any(k <= 0 for k in requested_k):
            raise ValueError("All --k values must be greater than zero.")
        return [str(k) for k in sorted(set(requested_k))]

    k_values = set()
    for run in runs:
        for k in run.metrics_by_k:
            try:
                k_values.add(int(k))
            except ValueError:
                print(
                    f"Warning: ignoring non-integer k value {k!r} in {run.path}",
                    file=sys.stderr,
                )

    if not k_values:
        raise ValueError("No integer k values found in the input metrics files.")
    return [str(k) for k in sorted(k_values)]


def select_metric_names(
    runs: list[MetricsRun],
    requested_metrics: list[str] | None,
) -> list[str]:
    """Return metric names selected for plotting."""

    if requested_metrics is not None:
        unique_metrics = list(dict.fromkeys(requested_metrics))
        if not unique_metrics:
            raise ValueError("At least one --metrics value is required.")
        return unique_metrics

    first_run_order = metric_order_for_run(runs[0])
    common_metrics = metric_names_for_run(runs[0])
    for run in runs[1:]:
        common_metrics &= metric_names_for_run(run)

    selected_metrics = [
        metric_name for metric_name in first_run_order if metric_name in common_metrics
    ]
    if not selected_metrics:
        raise ValueError("No common metrics found across the input metrics files.")
    return selected_metrics


def metric_order_for_run(run: MetricsRun) -> list[str]:
    """Return metric names in first-seen order for a run."""

    ordered_metrics: list[str] = []
    seen_metrics = set()
    for metrics in run.metrics_by_k.values():
        for metric_name in metrics:
            if metric_name not in seen_metrics:
                seen_metrics.add(metric_name)
                ordered_metrics.append(metric_name)
    return ordered_metrics


def metric_names_for_run(run: MetricsRun) -> set[str]:
    """Return all metric names present in a run."""

    metric_names = set()
    for metrics in run.metrics_by_k.values():
        metric_names.update(metrics)
    return metric_names


def warn_for_missing_values(
    runs: list[MetricsRun],
    k_values: list[str],
    metric_names: list[str],
) -> None:
    """Print warnings for missing k values or metrics that will leave empty bars."""

    for run in runs:
        for k in k_values:
            metrics = run.metrics_by_k.get(k)
            if metrics is None:
                print(
                    f"Warning: {run.path.name} has no metrics for k={k}; "
                    "its bars will be empty for that plot.",
                    file=sys.stderr,
                )
                continue

            for metric_name in metric_names:
                if metric_name not in metrics:
                    print(
                        f"Warning: {run.path.name} has no {metric_name!r} "
                        f"metric for k={k}; its bar will be empty.",
                        file=sys.stderr,
                    )


def plot_metrics_for_k(
    runs: list[MetricsRun],
    metric_names: list[str],
    k: str,
    output_path: Path,
    title_prefix: str | None = None,
) -> None:
    """Write a grouped bar chart for one k value."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate retrieval metrics plots."
        ) from exc

    run_count = len(runs)
    metric_count = len(metric_names)
    bar_width = min(0.8 / max(run_count, 1), 0.22)
    group_centers = list(range(metric_count))
    figure_width = max(10.0, metric_count * max(1.2, run_count * 0.45))
    figure_height = max(5.5, 3.8 + run_count * 0.22)

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    title = f"Retrieval Metrics Comparison @ k={k}"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    axis.set_title(title)

    for run_index, run in enumerate(runs):
        offset = (run_index - (run_count - 1) / 2) * bar_width
        x_values = []
        y_values = []
        for metric_index, metric_name in enumerate(metric_names):
            metrics = run.metrics_by_k.get(k)
            value = metrics.get(metric_name) if metrics else None
            if value is None:
                continue
            x_values.append(group_centers[metric_index] + offset)
            y_values.append(value)

        # Skipping missing values preserves each run's bar slot while leaving it empty.
        axis.bar(x_values, y_values, width=bar_width, label=run.label)

    axis.set_xlabel("Metric")
    axis.set_ylabel("Metric value")
    axis.set_ylim(0, 1)
    axis.set_xticks(group_centers)
    axis.set_xticklabels(metric_names, rotation=25, ha="right")
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_all_k_values(
    runs: list[MetricsRun],
    k_values: list[str],
    metric_names: list[str],
    output_dir: Path,
    title_prefix: str | None,
    timestamp: str,
) -> list[Path]:
    """Write one timestamped plot for each selected k value."""

    output_paths = []
    for k in k_values:
        output_path = output_dir / f"retrieval_metrics_comparison_{timestamp}_k{k}.png"
        plot_metrics_for_k(
            runs=runs,
            metric_names=metric_names,
            k=k,
            output_path=output_path,
            title_prefix=title_prefix,
        )
        output_paths.append(output_path)
    return output_paths


def plot_metric_trend(
    runs: list[MetricsRun],
    k_values: list[str],
    metric_name: str,
    output_path: Path,
    title_prefix: str | None = None,
) -> None:
    """Write a line chart showing one metric across selected k values."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate retrieval metrics plots."
        ) from exc

    figure_width = max(10.0, len(k_values) * 1.1)
    figure_height = max(5.5, 3.8 + len(runs) * 0.22)

    figure, axis = plt.subplots(
        figsize=(figure_width, figure_height),
        constrained_layout=True,
    )
    title = f"{metric_name} by k"
    if title_prefix:
        title = f"{title_prefix} - {title}"
    axis.set_title(title)

    for run in runs:
        x_values = []
        y_values = []
        for k in k_values:
            metrics = run.metrics_by_k.get(k)
            value = metrics.get(metric_name) if metrics else None
            if value is None:
                continue
            x_values.append(int(k))
            y_values.append(value)

        if x_values:
            axis.plot(x_values, y_values, marker="o", linewidth=1.8, label=run.label)

    axis.set_xlabel("k")
    axis.set_ylabel(metric_name)
    axis.set_ylim(0, 1)
    axis.set_xticks([int(k) for k in k_values])
    axis.grid(axis="both", linestyle="--", alpha=0.35)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncols=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def plot_all_metric_trends(
    runs: list[MetricsRun],
    k_values: list[str],
    metric_names: list[str],
    output_dir: Path,
    title_prefix: str | None,
    timestamp: str,
) -> list[Path]:
    """Write one timestamped metric-vs-k line chart for each selected metric."""

    output_paths = []
    for metric_name in metric_names:
        output_path = (
            output_dir
            / f"retrieval_metric_trend_{timestamp}_{slugify_filename(metric_name)}.png"
        )
        plot_metric_trend(
            runs=runs,
            k_values=k_values,
            metric_name=metric_name,
            output_path=output_path,
            title_prefix=title_prefix,
        )
        output_paths.append(output_path)
    return output_paths


def slugify_filename(value: str) -> str:
    """Return a filesystem-friendly filename segment."""

    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip(".-_").lower()
    return slug or "metric"


def main() -> int:
    """Run the retrieval metrics plotting command."""

    parser = build_parser()
    args = parser.parse_args()

    try:
        if args.disable_bar_plots and args.disable_line_plots:
            raise ValueError(
                "At least one plot type must be enabled. Remove one disable flag."
            )

        metrics_paths = resolve_metrics_paths(
            metrics_files=args.metrics_files,
            metrics_list=args.metrics_list,
        )
        runs = load_metrics_runs(metrics_paths)
        k_values = select_k_values(runs, args.k)
        metric_names = select_metric_names(runs, args.metrics)
        warn_for_missing_values(runs, k_values, metric_names)

        if not any(
            metric_name in run.metrics_by_k.get(k, {})
            for run in runs
            for k in k_values
            for metric_name in metric_names
        ):
            raise ValueError(
                "No selected metrics are available for the selected k values."
            )

        timestamp = datetime.now().strftime("%Y%m%dT%H%M")
        output_paths = []
        if not args.disable_bar_plots:
            output_paths.extend(
                plot_all_k_values(
                    runs=runs,
                    k_values=k_values,
                    metric_names=metric_names,
                    output_dir=args.output_dir,
                    title_prefix=args.title_prefix,
                    timestamp=timestamp,
                )
            )
        if not args.disable_line_plots:
            output_paths.extend(
                plot_all_metric_trends(
                    runs=runs,
                    k_values=k_values,
                    metric_names=metric_names,
                    output_dir=args.output_dir,
                    title_prefix=args.title_prefix,
                    timestamp=timestamp,
                )
            )
    except Exception as exc:
        parser.exit(status=1, message=f"Error: {exc}\n")

    print("Wrote retrieval metrics plots:")
    for output_path in output_paths:
        print(f"  {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
