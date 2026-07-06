import pandas as pd
import argparse
from datetime import datetime
from pathlib import Path
import os

DEFAULT_EVAL_DIR = Path(r"C:\Users\samue\git\figma-rag\codex_annotation")


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line parser for completing an annotated set."""
    parser = argparse.ArgumentParser(
        description="Download raw Figma Help Center HTML and append manifest records."
    )
    parser.add_argument(
        "--last-query-idx",
        type=int,
        required=True,
        help="Idx of the first query to be annotated (check the last query_id in the golden set).",
    )

    parser.add_argument(
        "--annotated-set-path",
        type=Path,
        default=r"C:\Users\samue\git\figma-rag\codex_annotation\golden_candidates_approved.jsonl",
        help="Annotated set to complete.",
    )

    parser.add_argument(
        "--ref-manifest-path",
        type=Path,
        default=r"C:\Users\samue\git\figma-rag\data\processed\figma_docs\manifest.jsonl",
        help="Manifest of documentation files, useful for finding file paths.",
    )

    parser.add_argument(
        "--output-set-path",
        type=Path,
        default=None,
        help=(
            "Annotated set with automatically inferred fields. When omitted, "
            "the output is written to output_set_path/ with the input filename, "
            "'complete', and the current timestamp up to minutes."
        ),
    )

    return parser


def build_default_output_set_path(annotated_set_path: Path) -> Path:
    """Build a timestamped default output path in the evaluation data folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output_name = f"{annotated_set_path.stem}_complete_{timestamp}{annotated_set_path.suffix}"
    return DEFAULT_EVAL_DIR / output_name


parser = build_parser()
args = parser.parse_args()
annotated_set_path = Path(args.annotated_set_path)
ref_manifest_path = Path(args.ref_manifest_path)
output_set_path = args.output_set_path or build_default_output_set_path(annotated_set_path)

annotated_set = pd.read_json(annotated_set_path, lines=True)
ref_manifest = pd.read_json(ref_manifest_path, lines=True)

new_objects = []
for idx, row in annotated_set.iterrows():

    new_targets = []
    for target in row["targets"]:
        span_start, span_end = target["char_span"][0], target["char_span"][1]
        document_path = target["document_path"]#.replace("\\","/")

        matching_row = ref_manifest[ref_manifest.processed_file_path == document_path].iloc[0]


        relevant_url = matching_row.source_url
        document_id = matching_row.document_id

        with open(document_path, "r", encoding="utf-8") as f:
            processed_md_text = f.read()

        span_idx = (span_start, span_end)
        span_text = processed_md_text[span_start:span_end]

        new_targets.append({
            "document_id": document_id,
            "document_path": document_path,
            "relevant_span": span_text,
            "relevant_url": relevant_url,
            "char_span": [span_start, span_end],
        })
        
    new_objects.append({
        "query": row["query"],
        "query_id": f"{args.last_query_idx+idx+1:05d}",
        "query_type": "general",
        "answer_type": "easy",
        "targets": new_targets,
        "expected_answer_points": row["expected_answer_points"]
    })

pd.DataFrame(new_objects).to_json(output_set_path, orient='records', lines=True)
