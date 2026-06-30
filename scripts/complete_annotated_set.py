import pandas as pd
import argparse
from pathlib import Path

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download raw Figma Help Center HTML and append manifest records."
    )
    parser.add_argument(
        "--annotated-set-path",
        type=Path,
        default=r"C:\Users\samue\git\figma-rag\data\eval\golden_set_manual.json",
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
        default=r"C:\Users\samue\git\figma-rag\data\eval\golden_set.json",
        help="Annotated set with automatically inferred fields.",
    )

    return parser


parser = build_parser()
args = parser.parse_args()

annotated_set = pd.read_json(args.annotated_set_path, lines=True)
ref_manifest = pd.read_json(args.ref_manifest_path, lines=True)

new_objects = []
for idx, row in annotated_set.iterrows():

    new_targets = []
    for target in row["targets"]:
        relevant_span = target["relevant_span"]
        relevant_url = target["relevant_url"]

        matching_row = ref_manifest[ref_manifest.source_url == relevant_url].iloc[0]

        processed_md_path = matching_row.processed_file_path
        document_id = matching_row.document_id

        with open(processed_md_path, "r", encoding="utf-8") as f:
            processed_md_text = f.read()

        x = processed_md_text.find(relevant_span)
        span = (x, x + len(relevant_span))

        if (x == -1): raise Exception(f"Something has gone wrong with index {idx}\n{relevant_span}")

        new_targets.append({
            "document_id": document_id,
            "document_path": processed_md_path,
            "relevant_span": relevant_span,
            "relevant_url": relevant_url,
            "char_span": span,
        })
        
    new_objects.append({
        "query": row["query"],
        "query_id": f"{idx+1:05d}",
        "query_type": row["query_type"],
        "answer_type": row["answer_type"],
        "targets": new_targets,
        "expected_answer_points": row["expected_answer_points"]
    })


pd.DataFrame(new_objects).to_json(args.output_set_path, orient='records', lines=True)