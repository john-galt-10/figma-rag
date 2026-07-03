# Data Pipeline and Retrieval Evaluation

This document recaps the data steps needed to build the local retrieval pipeline and evaluate it against a manually annotated test set.

Run the commands from the repository root with the `figma-navigator` environment active.

## Retrieval Pipeline

The retrieval pipeline turns raw Figma Help Center pages into chunk vectors stored in a local Chroma index.

### 1. Download raw Help Center HTML

```powershell
python scripts\download_help_center.py
```

Input: Figma Help Center article URLs discovered from the sitemap.

Output:

- raw HTML pages in `data\raw\figma_docs\help_center\`
- a raw manifest at `data\raw\figma_docs\manifest.jsonl`

Underlying mechanism: the script fetches candidate Help Center URLs, stores each article HTML file, and appends metadata records to the raw manifest.

Useful parameters:

- `--output-dir`: where raw HTML files are written.
- `--manifest-path`: where raw document metadata is written.
- `--delay-seconds`: delay between requests.
- `--timeout-seconds`: HTTP timeout for fetches.
- `--limit`: optional cap for a smaller test download.
- `--overwrite`: re-download pages that already exist.

### 2. Convert raw HTML to clean Markdown

```powershell
python scripts\convert_help_center_to_markdown.py
```

Input: the raw manifest and raw HTML pages from step 1.

Output:

- Markdown documents in `data\processed\figma_docs\help_center\`
- a processed manifest at `data\processed\figma_docs\manifest.jsonl`

Underlying mechanism: the script parses each raw Help Center HTML page, extracts the main article content, normalizes it into Markdown, and writes a processed manifest that points to the cleaned files.

Useful parameters:

- `--manifest-path`: raw JSONL manifest to read.
- `--output-dir`: where Markdown documents are written.
- `--processed-manifest-path`: where processed metadata is written.

### 3. Chunk the processed Markdown documents

```powershell
python scripts\chunk_figma_docs.py
```

Input: `data\processed\figma_docs\manifest.jsonl` and the cleaned Markdown documents.

Output: a versioned chunk JSONL artifact under `data\processed\figma_docs\`, for example:

```text
data\processed\figma_docs\chunks_hierarchical_bge-small-en-v1.5_t320_o40_20260626-1212.jsonl
```

Underlying mechanism: the script applies the selected chunking strategy, currently `hierarchical` by default, and uses the selected embedding model tokenizer to keep chunks within the configured token budget. Each chunk keeps document metadata, heading context, text, token count, and character spans back into the processed document.

Useful parameters:

- `--manifest-path`: processed manifest to read.
- `--output-path`: explicit chunk JSONL path. If omitted, the script creates a timestamped artifact name.
- `--strategy`: chunking strategy. Default is `hierarchical`.
- `--model`: tokenizer model used for token counting. Default is `BAAI/bge-small-en-v1.5`.
- `--max-tokens`: maximum final chunk size. Default is `320`.
- `--overlap-tokens`: overlap for split sections. Default is `40`.

### 4. Build the vector index

```powershell
python scripts\build_vector_index.py --chunks-path data\processed\figma_docs\chunks_hierarchical_bge-small-en-v1.5_t320_o40_20260626-1212.jsonl --collection-name hierarchical-bge-w-product --recreate
```

Input: a chunk JSONL artifact from step 3.

Output: a persistent Chroma collection under `data\processed\figma_docs\chroma\`.

Underlying mechanism: the script embeds each chunk text with Sentence Transformers and upserts the vectors plus chunk metadata into Chroma. Use the same embedding model here that you intend to use at retrieval and evaluation time.

The embedding loader requires `python-dotenv` in the active `figma-navigator` environment; install it with `pip install python-dotenv` if needed. To prefer local model weights for a model ID, add a repo-local `.env` file with a JSON mapping:

```dotenv
FIGMA_RAG_MODEL_PATHS_JSON={"BAAI/bge-base-en-v1.5":"C:/Users/samue/models/bge-base-en-v1.5"}
```

When the mapped directory exists, the loader uses it. When the mapped directory is missing, it falls back to the model ID and the normal Hugging Face cache/download behavior.

Useful parameters:

- `--chunks-path`: chunk JSONL file to index.
- `--persist-dir`: Chroma persistence directory.
- `--collection-name`: collection name to create or update. If omitted, it is derived from the chunk artifact and embedding model.
- `--model`: Sentence Transformers embedding model. Default is `BAAI/bge-small-en-v1.5`.
- `--batch-size`: chunks embedded and upserted per batch.
- `--recreate`: delete only the selected collection before rebuilding it.

## Evaluation Pipeline

The evaluation pipeline starts from document-level annotations, maps them onto retrieval chunks, runs the retriever, and writes metrics.

### 1. Prepare a document-level golden set

Create a JSONL test set like `data\eval\golden_set.json`.

Each row should represent one query and include:

- `query_id`: stable ID for the question.
- `query`: user-facing question.
- `query_type`: broad question type, such as `general` or `instructions`.
- `answer_type`: difficulty or answer category.
- `expected_answer_points`: list of facts the final answer should cover.
- `targets`: one or more relevant document targets.

Each target should include:

- `document_id`: processed document ID from `data\processed\figma_docs\manifest.jsonl`.
- `document_path`: processed Markdown path.
- `relevant_span`: exact relevant text from the processed Markdown document.
- `relevant_url`: source URL for traceability.
- `char_span`: half-open `[start, end]` character span of `relevant_span` inside the processed Markdown document.

If you start from a manually annotated file that only has `relevant_url` and `relevant_span`, use the processed manifest to fill `document_id`, `document_path`, and `char_span`. The existing `scripts\complete_annotated_set.py` does this enrichment for the current annotation format.

### 2. Map document-level labels to chunk-level labels

```powershell
python scripts\map_relevant_chunks.py --test-set-path data\eval\golden_set.json --chunks-path data\processed\figma_docs\chunks_hierarchical_bge-small-en-v1.5_t320_o40_20260626-1212.jsonl
```

Input:

- document-level golden set with `targets[*].char_span`
- chunk JSONL artifact produced by the same chunking run you want to evaluate

Output: a mapped retrieval test set under `data\eval\retrieval_test\`, for example:

```text
data\eval\retrieval_test\golden_set_relevant_chunks_hierarchical_bge-small-en-v1.5_20260626-1212.jsonl
```

Underlying mechanism: the script groups chunks by `document_id`, checks which chunk character spans overlap each annotated answer span, and writes relevant `chunk_id` labels with overlap scores. It does not modify the original golden set.

Useful parameters:

- `--test-set-path`: document-level annotated JSONL file.
- `--chunks-path`: chunk JSONL file to map against.
- `--output-path`: explicit output path. If omitted, the script builds a traceable filename under `data\eval\retrieval_test\`.

Important: this mapped file is tied to one chunk artifact. If you re-run chunking with different parameters, re-run this mapping step before evaluation.

### 3. Evaluate retrieval quality

```powershell
python scripts\evaluate_retriever.py --test-set-path data\eval\retrieval_test\golden_set_relevant_chunks_hierarchical_bge-small-en-v1.5_20260626-1212.jsonl --collection-name hierarchical-bge-w-product --top-k 1 3 5 10 --save-details
```

Input:

- mapped retrieval test set from step 2
- Chroma collection built from the matching chunk artifact

Output:

- aggregate metrics JSON in `data\eval\retrieval_test\test_results\`
- optional per-query CSV details when `--save-details` is used

Underlying mechanism: the script embeds each query, retrieves chunks from the configured collection, compares retrieved chunk IDs against the mapped relevant chunk IDs, and reports metrics at each requested `top-k`.

Useful parameters:

- `--test-set-path`: mapped retrieval test-set JSONL.
- `--persist-dir`: Chroma persistence directory.
- `--collection-name`: Chroma collection to query.
- `--model`: query embedding model. Use the same model used to build the index.
- `--top-k`: one or more cutoffs to evaluate.
- `--output-dir`: directory for metrics and details.
- `--save-details`: write per-query debug CSV rows.
- `--metadata-filter`: metadata filter applied before vector ranking. Can be repeated.
- `--disable-metadata-filters`: evaluate without metadata filters.
- `--retrieval-component`: retrieval backend to enable. Current default is `chroma`.
- `--seed`: reproducibility seed.

## Artifact Alignment Checklist

Keep these artifacts aligned for a meaningful evaluation:

- The processed manifest should match the Markdown documents used for annotation.
- The chunk JSONL used by `map_relevant_chunks.py` should be the same chunk artifact used to build the evaluated Chroma collection.
- The embedding model used by `build_vector_index.py` should match the model used by `evaluate_retriever.py`.
- If chunking parameters, processed documents, or annotation spans change, regenerate the mapped retrieval test set.
- If indexed chunks or embedding model change, rebuild or recreate the relevant Chroma collection.
