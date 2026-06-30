"""Gradio UI for reviewing retrieval results against existing chunk labels."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from figma_rag.retrieval import (  # noqa: E402
    ChromaRetriever,
    RetrievalRequest,
    RetrievalResult,
    build_retrieval_pipeline,
    parse_metadata_filter_set,
    resolve_collection_name,
)

DEFAULT_TEST_SET_PATH = (
    REPO_ROOT
    / "data"
    / "eval"
    / "retrieval_test"
    / "golden_set_relevant_chunks_hierarchical_bge-small-en-v1.5_20260626-1212.jsonl"
)
DEFAULT_CHUNKS_PATH = (
    REPO_ROOT
    / "data"
    / "processed"
    / "figma_docs"
    / "chunks_hierarchical_bge-small-en-v1.5_t320_o40_20260629-1709.jsonl"
)
DEFAULT_PERSIST_DIR = REPO_ROOT / "data" / "processed" / "figma_docs" / "chroma"
DEFAULT_COLLECTION_NAME = "hierarchical-bge-w-product"
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 20
DEFAULT_METADATA_FILTERS = [
    "token_count>30",
    "product=figma-design-or-general",
]


@dataclass(frozen=True)
class TestSetQuery:
    """One query from a retrieval test set."""

    query_id: str
    query: str
    query_type: str
    answer_type: str
    relevant_chunk_ids: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalSettings:
    """Runtime settings that determine a retrieval result set."""

    test_set_path: str
    chunks_path: str
    persist_dir: str
    collection_name: str
    model: str
    top_k: int
    metadata_filters_text: str
    disable_metadata_filters: bool
    retrieval_components: tuple[str, ...]


def build_parser() -> argparse.ArgumentParser:
    """Build CLI parser for Gradio startup defaults."""

    parser = argparse.ArgumentParser(
        description="Launch a Gradio UI for reviewing retrieval labels."
    )
    parser.add_argument(
        "--test-set-path",
        type=Path,
        default=DEFAULT_TEST_SET_PATH,
        help="JSONL retrieval test set to review.",
    )
    parser.add_argument(
        "--chunks-path",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help=(
            "Chunk JSONL used only to infer the default collection name when "
            "--collection-name is omitted."
        ),
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIR,
        help="Directory containing the persistent Chroma database.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help="Name of the Chroma collection to query.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Sentence Transformers model used to embed each query.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of nearest chunks to retrieve per query.",
    )
    parser.add_argument(
        "--metadata-filter",
        action="append",
        default=DEFAULT_METADATA_FILTERS.copy(),
        metavar="FILTER",
        help=(
            "Metadata filter to apply before vector ranking. Supports =, !=, <, "
            "<=, >, and >=. Can be repeated and filters are combined with AND."
        ),
    )
    parser.add_argument(
        "--disable-metadata-filters",
        action="store_true",
        help="Parse metadata filters but do not apply them during retrieval.",
    )
    parser.add_argument(
        "--retrieval-component",
        action="append",
        choices=["chroma"],
        default=None,
        help="Retrieval component to enable. Defaults to chroma.",
    )
    return parser


def load_test_set(path: Path) -> list[TestSetQuery]:
    """Load retrieval review queries from a JSONL file."""

    queries: list[TestSetQuery] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped_line = line.strip()
            if not stripped_line:
                continue

            try:
                row = json.loads(stripped_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc

            queries.append(parse_test_set_row(row, path, line_number))

    if not queries:
        raise ValueError(f"No queries found in {path}")
    return queries


def parse_test_set_row(row: Any, path: Path, line_number: int) -> TestSetQuery:
    """Validate and parse one JSONL row from the review test set."""

    if not isinstance(row, dict):
        raise ValueError(f"Expected JSON object at {path}:{line_number}")

    query_id = row.get("query_id")
    query = row.get("query")
    chunks = row.get("chunks", [])
    if not isinstance(query_id, str) or not query_id:
        raise ValueError(f"Row {line_number} must include a non-empty query_id")
    if not isinstance(query, str) or not query:
        raise ValueError(f"Row {line_number} must include a non-empty query")
    if not isinstance(chunks, list):
        raise ValueError(f"Row {line_number} chunks must be a list when present")

    relevant_chunk_ids = []
    seen_chunk_ids = set()
    for chunk_index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Row {line_number} chunk {chunk_index} must be a JSON object"
            )

        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError(
                f"Row {line_number} chunk {chunk_index} must include a chunk_id"
            )
        if chunk_id not in seen_chunk_ids:
            seen_chunk_ids.add(chunk_id)
            relevant_chunk_ids.append(chunk_id)

    return TestSetQuery(
        query_id=query_id,
        query=query,
        query_type=str(row.get("query_type", "")),
        answer_type=str(row.get("answer_type", "")),
        relevant_chunk_ids=tuple(relevant_chunk_ids),
    )


def parse_metadata_filters_text(value: str) -> list[str]:
    """Parse one-filter-per-line UI text into CLI-style filter strings."""

    return [line.strip() for line in value.splitlines() if line.strip()]


def normalize_path(value: str) -> Path:
    """Return an absolute path, resolving relative values from the repository root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def build_settings(
    test_set_path: str,
    chunks_path: str,
    persist_dir: str,
    collection_name: str,
    model: str,
    top_k: int | float,
    metadata_filters_text: str,
    disable_metadata_filters: bool,
    retrieval_component: str,
) -> RetrievalSettings:
    """Build normalized retrieval settings from UI values."""

    normalized_top_k = int(top_k)
    if normalized_top_k <= 0:
        raise ValueError("top-k must be greater than zero")

    components = (retrieval_component or "chroma",)
    return RetrievalSettings(
        test_set_path=str(normalize_path(test_set_path)),
        chunks_path=str(normalize_path(chunks_path)),
        persist_dir=str(normalize_path(persist_dir)),
        collection_name=collection_name.strip(),
        model=model.strip(),
        top_k=normalized_top_k,
        metadata_filters_text=metadata_filters_text.strip(),
        disable_metadata_filters=bool(disable_metadata_filters),
        retrieval_components=components,
    )


def retrieve_results(
    query: TestSetQuery,
    settings: RetrievalSettings,
) -> list[RetrievalResult]:
    """Run the configured retrieval pipeline for one query."""

    metadata_filters = parse_metadata_filter_set(
        parse_metadata_filters_text(settings.metadata_filters_text)
    )
    collection_name = resolve_collection_name(
        chunks_path=Path(settings.chunks_path),
        model_name=settings.model,
        collection_name=settings.collection_name or None,
    )

    retriever = ChromaRetriever(
        persist_dir=Path(settings.persist_dir),
        collection_name=collection_name,
        model_name=settings.model,
    )
    pipeline = build_retrieval_pipeline(
        component_names=list(settings.retrieval_components),
        chroma_retriever=retriever,
    )
    return pipeline.retrieve(
        RetrievalRequest(
            query=query.query,
            top_k=settings.top_k,
            metadata_filters=metadata_filters,
            metadata_filters_enabled=not settings.disable_metadata_filters,
        )
    )


def cache_key(settings: RetrievalSettings, query_id: str) -> str:
    """Return a stable cache key for a query and retrieval configuration."""

    return json.dumps(
        {
            "query_id": query_id,
            "settings": settings.__dict__,
        },
        sort_keys=True,
    )


def get_or_retrieve_results(
    query: TestSetQuery,
    settings: RetrievalSettings,
    cache: dict[str, list[RetrievalResult]],
) -> list[RetrievalResult]:
    """Return cached retrieval results or run retrieval for the query."""

    key = cache_key(settings, query.query_id)
    if key not in cache:
        cache[key] = retrieve_results(query=query, settings=settings)
    return cache[key]


def format_query_markdown(
    query: TestSetQuery,
    query_index: int,
    query_count: int,
) -> str:
    """Render the current query details as Markdown."""

    metadata_parts = []
    if query.query_type:
        metadata_parts.append(f"Query type: `{query.query_type}`")
    if query.answer_type:
        metadata_parts.append(f"Answer type: `{query.answer_type}`")

    metadata_text = " | ".join(metadata_parts) if metadata_parts else "No query metadata."
    relevant_ids = "\n".join(
        f"- `{chunk_id}`" for chunk_id in query.relevant_chunk_ids
    )
    if not relevant_ids:
        relevant_ids = "_No relevant chunk IDs are present in this test-set row._"

    return (
        f"### Query {query_index + 1} / {query_count}: `{query.query_id}`\n\n"
        f"{metadata_text}\n\n"
        f"**Query**\n\n{query.query}\n\n"
        f"**Existing relevant chunk IDs**\n\n{relevant_ids}"
    )


def format_result_markdown(
    query: TestSetQuery,
    results: list[RetrievalResult],
    result_index: int,
) -> str:
    """Render the selected retrieval result as Markdown."""

    if not results:
        return "### No retrieved chunks\n\nThe configured retrieval returned no results."

    result_index = clamp_index(result_index, len(results))
    result = results[result_index]
    is_relevant = result.chunk_id in set(query.relevant_chunk_ids)
    status = "YES - already labelled relevant" if is_relevant else "NO - not labelled relevant"
    source_url = result.source_url or ""
    source_link = f"[{source_url}]({source_url})" if source_url else "_Missing source URL_"
    processed_file_path = result.metadata.get("processed_file_path", "")

    return (
        f"### Retrieved chunk {result_index + 1} / {len(results)}\n\n"
        f"**Relevant in test set:** `{status}`\n\n"
        f"**Rank:** {result.rank}  \n"
        f"**Chunk ID:** `{result.chunk_id}`  \n"
        f"**Title:** {result.title}  \n"
        f"**Section:** {result.section}  \n"
        f"**Distance:** {result.distance:.4f}  \n"
        f"**Source URL:** {source_link}  \n"
        f"**Processed markdown path:** `{processed_file_path}`\n\n"
        f"**Chunk text**\n\n"
        f"```text\n{escape_newlines(result.text)}\n```"
        # f"```text\n{escape_newlines(result.content)}\n```"
    )


def escape_newlines(value: str) -> str:
    """Render newline characters as literal backslash-n sequences."""

    return value.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def clamp_index(index: int, item_count: int) -> int:
    """Clamp an index to the valid range for a sequence."""

    if item_count <= 0:
        return 0
    return max(0, min(index, item_count - 1))


def review_state(
    test_set_path: str,
    chunks_path: str,
    persist_dir: str,
    collection_name: str,
    model: str,
    top_k: int | float,
    metadata_filters_text: str,
    disable_metadata_filters: bool,
    retrieval_component: str,
    query_index: int,
    result_index: int,
    cache: dict[str, list[RetrievalResult]] | None,
) -> tuple[str, str, int, int, dict[str, list[RetrievalResult]]]:
    """Render the current query and retrieval result state."""

    cache = cache or {}
    settings = build_settings(
        test_set_path=test_set_path,
        chunks_path=chunks_path,
        persist_dir=persist_dir,
        collection_name=collection_name,
        model=model,
        top_k=top_k,
        metadata_filters_text=metadata_filters_text,
        disable_metadata_filters=disable_metadata_filters,
        retrieval_component=retrieval_component,
    )
    queries = load_test_set(Path(settings.test_set_path))
    query_index = clamp_index(int(query_index), len(queries))
    query = queries[query_index]
    results = get_or_retrieve_results(query=query, settings=settings, cache=cache)
    result_index = clamp_index(int(result_index), len(results))
    return (
        format_query_markdown(query, query_index, len(queries)),
        format_result_markdown(query, results, result_index),
        query_index,
        result_index,
        cache,
    )


def move_query(
    delta: int,
    test_set_path: str,
    chunks_path: str,
    persist_dir: str,
    collection_name: str,
    model: str,
    top_k: int | float,
    metadata_filters_text: str,
    disable_metadata_filters: bool,
    retrieval_component: str,
    query_index: int,
    cache: dict[str, list[RetrievalResult]] | None,
) -> tuple[str, str, int, int, dict[str, list[RetrievalResult]]]:
    """Move to another query and reset result navigation to the first chunk."""

    next_query_index = int(query_index) + delta
    return review_state(
        test_set_path,
        chunks_path,
        persist_dir,
        collection_name,
        model,
        top_k,
        metadata_filters_text,
        disable_metadata_filters,
        retrieval_component,
        next_query_index,
        0,
        cache,
    )


def move_result(
    delta: int,
    test_set_path: str,
    chunks_path: str,
    persist_dir: str,
    collection_name: str,
    model: str,
    top_k: int | float,
    metadata_filters_text: str,
    disable_metadata_filters: bool,
    retrieval_component: str,
    query_index: int,
    result_index: int,
    cache: dict[str, list[RetrievalResult]] | None,
) -> tuple[str, str, int, int, dict[str, list[RetrievalResult]]]:
    """Move to another retrieved chunk for the current query."""

    return review_state(
        test_set_path,
        chunks_path,
        persist_dir,
        collection_name,
        model,
        top_k,
        metadata_filters_text,
        disable_metadata_filters,
        retrieval_component,
        int(query_index),
        int(result_index) + delta,
        cache,
    )


def build_app(args: argparse.Namespace):
    """Build the Gradio Blocks app."""

    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError(
            "The 'gradio' package is required for this review UI. "
            "Install it in the figma-navigator environment, for example: "
            "pip install gradio"
        ) from exc

    metadata_filter_text = "\n".join(args.metadata_filter or [])
    retrieval_component = (args.retrieval_component or ["chroma"])[0]

    with gr.Blocks(title="Figma RAG Retrieval Label Review") as app:
        cache_state = gr.State({})
        query_index_state = gr.State(0)
        result_index_state = gr.State(0)

        gr.Markdown("# Figma RAG Retrieval Label Review")
        with gr.Row():
            with gr.Column(scale=1):
                test_set_path = gr.Textbox(
                    label="Test set path",
                    value=str(args.test_set_path),
                )
                chunks_path = gr.Textbox(
                    label="Chunks path",
                    value=str(args.chunks_path),
                )
                persist_dir = gr.Textbox(
                    label="Persist dir",
                    value=str(args.persist_dir),
                )
                collection_name = gr.Textbox(
                    label="Collection name",
                    value=args.collection_name,
                )
                model = gr.Textbox(label="Model", value=args.model)
                top_k = gr.Number(label="Top K", value=args.top_k, precision=0)
                metadata_filters = gr.Textbox(
                    label="Metadata filters, one per line",
                    value=metadata_filter_text,
                    lines=4,
                )
                disable_metadata_filters = gr.Checkbox(
                    label="Disable metadata filters",
                    value=args.disable_metadata_filters,
                )
                retrieval_component_input = gr.Dropdown(
                    label="Retrieval component",
                    choices=["chroma"],
                    value=retrieval_component,
                )

                load_button = gr.Button("Load / Refresh Query")
                with gr.Row():
                    previous_query = gr.Button("< Previous Query")
                    next_query = gr.Button("Next Query >")
                with gr.Row():
                    previous_result = gr.Button("< Previous Chunk")
                    next_result = gr.Button("Next Chunk >")

            with gr.Column(scale=2):
                query_display = gr.Markdown()
                result_display = gr.Markdown()

        settings_inputs = [
            test_set_path,
            chunks_path,
            persist_dir,
            collection_name,
            model,
            top_k,
            metadata_filters,
            disable_metadata_filters,
            retrieval_component_input,
        ]
        outputs = [
            query_display,
            result_display,
            query_index_state,
            result_index_state,
            cache_state,
        ]

        load_button.click(
            fn=review_state,
            inputs=settings_inputs
            + [query_index_state, result_index_state, cache_state],
            outputs=outputs,
        )
        previous_query.click(
            fn=lambda *values: move_query(-1, *values),
            inputs=settings_inputs + [query_index_state, cache_state],
            outputs=outputs,
        )
        next_query.click(
            fn=lambda *values: move_query(1, *values),
            inputs=settings_inputs + [query_index_state, cache_state],
            outputs=outputs,
        )
        previous_result.click(
            fn=lambda *values: move_result(-1, *values),
            inputs=settings_inputs
            + [query_index_state, result_index_state, cache_state],
            outputs=outputs,
        )
        next_result.click(
            fn=lambda *values: move_result(1, *values),
            inputs=settings_inputs
            + [query_index_state, result_index_state, cache_state],
            outputs=outputs,
        )
        app.load(
            fn=review_state,
            inputs=settings_inputs
            + [query_index_state, result_index_state, cache_state],
            outputs=outputs,
        )

    return app


def main() -> int:
    """Launch the Gradio retrieval review UI."""

    args = build_parser().parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be greater than zero")

    app = build_app(args)
    app.launch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
