# Figma RAG System

This repository contains the Retrieval-Augmented Generation (RAG) system for a personal assistant focused on explaining Figma’s interface and behavior.

The broader project aims to help users understand UI elements in Figma by grounding answers in official documentation. This repository is dedicated specifically to the RAG layer: document collection, normalization, chunking, retrieval, and grounded answer generation.

## Project goal

The goal of this project is to build a learning-oriented RAG system that can answer questions about Figma features accurately and with traceable sources.

The long-term target workflow is:

1. capture UI context from Figma, such as a screenshot, pointer location, or structured context
2. identify the likely Figma UI element or feature being referenced
3. retrieve the most relevant official Figma documentation
4. generate a grounded answer that explains what the feature does, how to use it, and what related constraints or caveats matter

This repository focuses on steps 3 and 4 first, using a text-based baseline before adding multimodal input.

## Scope of this repository

This repo is responsible for:

* collecting and storing official Figma documentation
* cleaning and normalizing raw documents
* splitting documents into retrieval-friendly chunks
* attaching metadata to documents and chunks
* indexing content for retrieval
* answering user questions using retrieved context
* supporting evaluation of retrieval and answer quality

This repo does **not** initially focus on:

* screenshot analysis
* OCR or computer vision
* Figma plugin-side UI capture
* production deployment
* monetization or market-facing polish

## Data sources

The initial corpus should use only official Figma sources:

* Figma Help Center
* Figma Developer Documentation

Third-party blogs, tutorials, forum posts, and community content are excluded from the first version of the dataset.

## Development approach

This project is primarily a learning project. The main priorities are:

* learning how to design and implement a practical RAG pipeline
* understanding ingestion, chunking, metadata, and retrieval tradeoffs
* experimenting with evaluation methods
* building a clear and extensible baseline before adding multimodal complexity

The implementation philosophy is:

* start simple
* prefer readable Python scripts and explicit data artifacts
* keep each pipeline stage separable
* optimize for clarity and iteration speed rather than premature complexity

## Planned pipeline

The current plan is to build the system in stages:

1. gather official documentation from curated seed URLs
2. store raw content and metadata
3. clean and normalize documents into structured text or markdown
4. chunk documents by semantic sections
5. index chunks for retrieval
6. build a baseline question-answering pipeline
7. evaluate retrieval and answer quality
8. later extend the system to work with multimodal Figma context

## Current milestone

The current baseline covers the full local retrieval loop for the initial Figma Help Center corpus: ingestion, Markdown normalization, hierarchical chunking, Chroma indexing, semantic retrieval, chunk-level label mapping, and retrieval evaluation.

The retrieval pipeline currently:

* reads cleaned Markdown documents from the processed manifest
* chunks documents with the heading-aware hierarchical strategy
* measures chunk limits with the selected Sentence Transformers tokenizer
* preserves document metadata and heading paths in each chunk
* writes versioned JSONL artifacts for reproducible experiments
* embeds chunk text with `BAAI/bge-small-en-v1.5`
* stores vectors and source metadata in local persistent Chroma collections
* retrieves through the reusable pipeline in `src/figma_rag/retrieval/`
* supports metadata filtering, currently including `token_count > 30`
* applies the default topic filter for relevant Figma Help Center areas

The evaluation pipeline maps document-level answer spans onto chunk IDs with `scripts/map_relevant_chunks.py`, then compares retrieved chunk IDs against those labels with `scripts/evaluate_retriever.py`. The latest saved retrieval run uses 20 manually annotated queries against the `hierarchical-bge-w-topic` collection:

* source artifact: `data/eval/retrieval_test/test_results/retrieval_metrics_hierarchical_bge-small-en-v1.5_20260629-1709_k1-3-5-9-15-20_20260701T1558.json`
* Hit@1: `0.40`
* Hit@5: `0.75`
* Hit@9: `0.85`
* Hit@15 and Hit@20: `0.90`
* Recall@5: `0.595`
* Recall@15 and Recall@20: `0.807`

See `docs/data_pipeline_and_evaluation.md` for the full command-level pipeline and artifact alignment checklist.

## Build the local vector index

From the `figma-navigator` environment, build the persistent Chroma collection:

```powershell
python scripts/build_vector_index.py
```

The script reads the selected chunk JSONL, embeds each chunk's `text` field with Sentence Transformers, and upserts the vectors plus source metadata into `data/processed/figma_docs/chroma/`. By default, the collection name is derived from the chunking artifact and embedding model, so indexing a different chunking run creates a separate collection. It expects `chromadb` and `sentence-transformers` to be available in the active environment.

Query the local collection with the simple retrieval example:

```powershell
python scripts/retrieval_example.py "How do variables work in prototypes?"
```

The example uses the reusable Chroma retriever in `src/figma_rag/retrieval/`: it embeds the query with the selected Sentence Transformers model, searches the matching Chroma collection, and prints the nearest chunks with source metadata. Use `--chunks-path` and `--model` to target a specific indexed collection, `--collection-name` to override the generated collection name, and `--top-k` to choose how many chunks to return.

## Repository intent

This repository is intentionally narrow in scope: it is the RAG core of the larger Figma UI assistant project.

Its role is to provide a reliable, inspectable, and extensible foundation for grounded answers before integrating richer context such as screenshots, cursor position, accessibility information, or plugin-derived signals.
