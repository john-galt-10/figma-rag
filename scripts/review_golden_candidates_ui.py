"""Gradio UI for approving or rejecting candidate RAG evaluation questions."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "eval" / "golden_candidates.jsonl"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "eval" / "golden_candidates_approved.jsonl"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 7860


@dataclass(frozen=True)
class ReviewSession:
    """Runtime data shared by Gradio event handlers."""

    candidates: list[dict[str, Any]]
    output_path: Path
    start_index: int


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the review UI."""

    parser = argparse.ArgumentParser(
        description="Launch a Gradio UI for reviewing golden RAG candidates."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="JSONL candidate file to review.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Preferred JSONL file for approved candidates.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help="Host interface used by the Gradio server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help="Port used by the Gradio server.",
    )
    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help=(
            "One-based row number in the candidates JSONL to show first. "
            "Defaults to 1."
        ),
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Enable Gradio public sharing.",
    )
    return parser


def resolve_repo_path(path: Path) -> Path:
    """Resolve an absolute path or a path relative to the repository root."""

    expanded_path = path.expanduser()
    if expanded_path.is_absolute():
        return expanded_path
    return REPO_ROOT / expanded_path


def choose_non_destructive_output_path(path: Path) -> Path:
    """Return path, or a timestamped sibling path when path already exists."""

    if not path.exists():
        return path

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidate_path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    suffix_index = 1
    while candidate_path.exists():
        candidate_path = path.with_name(
            f"{path.stem}_{timestamp}_{suffix_index}{path.suffix}"
        )
        suffix_index += 1
    return candidate_path


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows and require each non-empty row to be a JSON object."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue
            try:
                row = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(row)

    if not rows:
        raise ValueError(f"No candidates found in {path}")
    return rows


def validate_start_row(start_row: int, candidate_count: int) -> int:
    """Validate a one-based start row and return its zero-based index."""

    if start_row < 1:
        raise ValueError("--start-row must be 1 or greater")
    if start_row > candidate_count:
        raise ValueError(
            f"--start-row {start_row} is beyond the last candidate row "
            f"({candidate_count})"
        )
    return start_row - 1


def parse_answer_points(text: str) -> list[str]:
    """Parse one-answer-point-per-line UI text into a non-empty string list."""

    answer_points = [line.strip() for line in text.splitlines() if line.strip()]
    if not answer_points:
        raise ValueError("expected_answer_points must include at least one non-empty line")
    return answer_points


def parse_targets_json(text: str) -> list[dict[str, Any]]:
    """Parse and validate the editable targets JSON textbox value."""

    try:
        targets = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"targets must be valid JSON: {exc}") from exc

    if not isinstance(targets, list) or not targets:
        raise ValueError("targets must be a non-empty JSON list")

    normalized_targets: list[dict[str, Any]] = []
    for target_index, target in enumerate(targets, start=1):
        normalized_targets.append(validate_target_shape(target, target_index))
    return normalized_targets


def validate_target_shape(target: Any, target_index: int) -> dict[str, Any]:
    """Validate one target object without reading its referenced document."""

    if not isinstance(target, dict):
        raise ValueError(f"target {target_index} must be a JSON object")
    if set(target) != {"document_path", "char_span"}:
        raise ValueError(
            f"target {target_index} must contain exactly document_path and char_span"
        )

    document_path = target.get("document_path")
    if not isinstance(document_path, str) or not document_path.strip():
        raise ValueError(f"target {target_index} document_path must be non-empty text")

    char_span = target.get("char_span")
    if (
        not isinstance(char_span, list)
        or len(char_span) != 2
        or not all(isinstance(position, int) for position in char_span)
    ):
        raise ValueError(f"target {target_index} char_span must be [start, end] integers")

    return {
        "document_path": document_path,
        "char_span": [int(char_span[0]), int(char_span[1])],
    }


def read_document_span(target: dict[str, Any], target_index: int) -> str:
    """Read the Markdown document referenced by target and return its span text."""

    document_path = target["document_path"]
    resolved_path = resolve_repo_path(Path(document_path)).resolve()
    if not resolved_path.exists():
        raise ValueError(f"target {target_index} document does not exist: {document_path}")
    if not resolved_path.is_file():
        raise ValueError(f"target {target_index} document is not a file: {document_path}")

    document_text = resolved_path.read_text(encoding="utf-8")
    start, end = target["char_span"]
    if start < 0 or end < 0 or end <= start:
        raise ValueError(f"target {target_index} char_span must satisfy 0 <= start < end")
    if end > len(document_text):
        raise ValueError(
            f"target {target_index} char_span ends beyond document length "
            f"({end} > {len(document_text)})"
        )

    span_text = document_text[start:end]
    if not span_text.strip():
        raise ValueError(f"target {target_index} char_span extracts empty text")
    return span_text


def build_preview_markdown(targets_text: str) -> str:
    """Build Markdown previews for every target in the editable targets JSON."""

    targets = parse_targets_json(targets_text)
    preview_sections = []
    for target_index, target in enumerate(targets, start=1):
        span_text = read_document_span(target, target_index)
        start, end = target["char_span"]
        preview_sections.append(
            "\n".join(
                [
                    f"### Target {target_index}",
                    f"`{target['document_path']}`",
                    f"`char_span`: [{start}, {end}]",
                    "",
                    "```text",
                    span_text,
                    "```",
                ]
            )
        )
    return "\n\n".join(preview_sections)


def validate_candidate(
    query: str,
    targets_text: str,
    expected_answer_points_text: str,
) -> dict[str, Any]:
    """Validate edited UI values and return a JSONL-ready candidate row."""

    normalized_query = query.strip()
    if not normalized_query:
        raise ValueError("query must be non-empty")

    targets = parse_targets_json(targets_text)
    for target_index, target in enumerate(targets, start=1):
        read_document_span(target, target_index)

    return {
        "query": normalized_query,
        "targets": targets,
        "expected_answer_points": parse_answer_points(expected_answer_points_text),
    }


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file without touching existing content."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as file:
        file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
        file.write("\n")


def format_targets(targets: Any) -> str:
    """Format target data as editable pretty JSON."""

    return json.dumps(targets, ensure_ascii=False, indent=2)


def format_answer_points(answer_points: Any) -> str:
    """Format answer points as one point per line for editing."""

    if not isinstance(answer_points, list):
        return ""
    return "\n".join(str(point) for point in answer_points)


def clamp_index(index: int, total_count: int) -> int:
    """Clamp a sample index to the available candidate range."""

    if total_count <= 0:
        return 0
    return max(0, min(index, total_count - 1))


def get_candidate(session: ReviewSession, index: int) -> dict[str, Any]:
    """Return the candidate at a clamped index."""

    return session.candidates[clamp_index(index, len(session.candidates))]


def render_candidate(
    session: ReviewSession,
    index: int,
    status_message: str = "",
) -> tuple[str, str, str, str, str, str, int]:
    """Render all editable fields and previews for one candidate."""

    normalized_index = clamp_index(index, len(session.candidates))
    candidate = get_candidate(session, normalized_index)
    query = str(candidate.get("query", ""))
    targets_text = format_targets(candidate.get("targets", []))
    answer_points_text = format_answer_points(candidate.get("expected_answer_points", []))

    try:
        preview_markdown = build_preview_markdown(targets_text)
        validation_message = status_message
    except ValueError as exc:
        preview_markdown = ""
        validation_message = f"Preview error: {exc}"

    progress = (
        f"Sample {normalized_index + 1} / {len(session.candidates)} | "
        f"Approved output: `{session.output_path}`"
    )
    if validation_message:
        progress = f"{progress}\n\n{validation_message}"

    return (
        progress,
        "",
        query,
        answer_points_text,
        targets_text,
        preview_markdown,
        normalized_index,
    )


def render_current_edits(
    session: ReviewSession,
    index: int,
    status_message: str,
    query: str,
    expected_answer_points_text: str,
    targets_text: str,
) -> tuple[str, str, str, str, str, str, int]:
    """Render validation feedback while preserving the user's edited values."""

    normalized_index = clamp_index(index, len(session.candidates))
    progress = (
        f"Sample {normalized_index + 1} / {len(session.candidates)} | "
        f"Approved output: `{session.output_path}`"
    )
    try:
        preview_markdown = build_preview_markdown(targets_text)
        preview_status = ""
    except ValueError as exc:
        preview_markdown = ""
        preview_status = f"Preview error: {exc}"

    return (
        progress,
        status_message if status_message else preview_status,
        query,
        expected_answer_points_text,
        targets_text,
        preview_markdown,
        normalized_index,
    )


def preview_targets(targets_text: str) -> tuple[str, str]:
    """Refresh target previews from disk after target JSON edits."""

    try:
        return build_preview_markdown(targets_text), ""
    except ValueError as exc:
        return "", f"Preview error: {exc}"


def approve_candidate(
    session: ReviewSession,
    query: str,
    expected_answer_points_text: str,
    targets_text: str,
    index: int,
) -> tuple[str, str, str, str, str, str, int]:
    """Append the edited candidate to the output JSONL and advance."""

    try:
        approved_row = validate_candidate(query, targets_text, expected_answer_points_text)
    except ValueError as exc:
        return render_current_edits(
            session,
            int(index),
            f"Approval blocked: {exc}",
            query,
            expected_answer_points_text,
            targets_text,
        )

    append_jsonl(session.output_path, approved_row)
    next_index = int(index) + 1
    if next_index >= len(session.candidates):
        return render_done(session, "Approved final sample.")
    return render_candidate(session, next_index, "Approved sample.")


def reject_candidate(
    session: ReviewSession,
    index: int,
) -> tuple[str, str, str, str, str, str, int]:
    """Reject the current candidate without writing output and advance."""

    next_index = int(index) + 1
    if next_index >= len(session.candidates):
        return render_done(session, "Rejected final sample.")
    return render_candidate(session, next_index, "Rejected sample.")


def render_done(
    session: ReviewSession,
    status_message: str,
) -> tuple[str, str, str, str, str, str, int]:
    """Render the completed review state after the last candidate."""

    progress = (
        f"{status_message}\n\n"
        f"Review complete: {len(session.candidates)} / {len(session.candidates)}\n\n"
        f"Approved output: `{session.output_path}`"
    )
    return progress, "", "", "", "[]", "", len(session.candidates) - 1


def build_app(session: ReviewSession):
    """Build the Gradio Blocks app for the candidate review workflow."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The 'gradio' package is required for this review UI. "
            "Install it in the figma-navigator environment, for example: "
            "pip install gradio"
        ) from exc

    def load_initial_candidate() -> tuple[str, str, str, str, str, str, int]:
        """Render the configured starting candidate when the Gradio app opens."""

        return render_candidate(session, session.start_index)

    def handle_approve(
        query_value: str,
        answer_points_value: str,
        targets_value: str,
        index_value: int,
    ) -> tuple[str, str, str, str, str, str, int]:
        """Approve the current edited candidate from UI inputs."""

        return approve_candidate(
            session,
            query_value,
            answer_points_value,
            targets_value,
            index_value,
        )

    def handle_reject(index_value: int) -> tuple[str, str, str, str, str, str, int]:
        """Reject the current candidate from the UI state."""

        return reject_candidate(session, index_value)

    with gr.Blocks(title="Figma RAG Golden Candidate Review") as app:
        index_state = gr.State(0)

        gr.Markdown("# Figma RAG Golden Candidate Review")
        progress = gr.Markdown()
        validation_status = gr.Markdown()

        query = gr.Textbox(label="Query", lines=2)
        expected_answer_points = gr.Textbox(
            label="Expected answer points, one per line",
            lines=8,
        )
        targets = gr.Textbox(
            label="Targets JSON",
            lines=12,
            max_lines=24,
        )
        gr.Markdown("## Target span previews")
        target_preview = gr.Markdown()

        with gr.Row():
            approve_button = gr.Button("APPROVE", variant="primary")
            reject_button = gr.Button("REJECT", variant="stop")

        render_outputs = [
            progress,
            validation_status,
            query,
            expected_answer_points,
            targets,
            target_preview,
            index_state,
        ]

        app.load(
            fn=load_initial_candidate,
            inputs=[],
            outputs=render_outputs,
        )
        approve_button.click(
            fn=handle_approve,
            inputs=[query, expected_answer_points, targets, index_state],
            outputs=render_outputs,
        )
        reject_button.click(
            fn=handle_reject,
            inputs=[index_state],
            outputs=render_outputs,
        )
        targets.change(
            fn=preview_targets,
            inputs=[targets],
            outputs=[target_preview, validation_status],
        )
        if hasattr(targets, "input"):
            targets.input(
                fn=preview_targets,
                inputs=[targets],
                outputs=[target_preview, validation_status],
            )

    return app


def main() -> int:
    """Load candidates and launch the Gradio review UI."""

    args = build_parser().parse_args()
    input_path = resolve_repo_path(args.input_path).resolve()
    preferred_output_path = resolve_repo_path(args.output_path).resolve()
    output_path = choose_non_destructive_output_path(preferred_output_path)

    candidates = load_jsonl(input_path)
    start_index = validate_start_row(args.start_row, len(candidates))
    session = ReviewSession(
        candidates=candidates,
        output_path=output_path,
        start_index=start_index,
    )
    app = build_app(session)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
