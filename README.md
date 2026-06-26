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

Documentation ingestion and Markdown normalization are complete for the initial Figma Help Center corpus. The current milestone is producing retrieval-ready chunks and preparing them for embedding and indexing.

The chunking pipeline currently:

* reads cleaned Markdown documents from the processed manifest
* supports selectable chunking strategies through the CLI
* uses heading-aware hierarchical chunking as the first strategy
* measures chunk limits with the selected embedding model's tokenizer
* preserves document metadata and heading paths in each chunk
* writes versioned JSONL artifacts for reproducible experiments

The baseline uses `Alibaba-NLP/gte-modernbert-base`, a maximum of 600 tokens per chunk, and 60 tokens of overlap when oversized sections must be split. The next objective is to embed these chunks and build the first local retrieval index.

## Build the local vector index

From the `figma-navigator` environment, build the persistent Chroma collection:

```powershell
python scripts/build_vector_index.py
```

The script reads the selected chunk JSONL, embeds each chunk's `text` field with Sentence Transformers, and upserts the vectors plus source metadata into `data/processed/figma_docs/chroma/`. By default, the collection name is derived from the chunking artifact and embedding model, for example `figma_hierarchical_gte-modernbert-base_t600_o60`, so indexing a different chunking run creates a separate collection. It expects `chromadb` and `sentence-transformers` to be available in the active environment.

Query the local collection with the simple retrieval example:

```powershell
python scripts/retrieval_example.py "How do variables work in prototypes?"
```

The example uses the reusable Chroma retriever in `src/figma_rag/retrieval/`: it embeds the query with the selected Sentence Transformers model, searches the matching Chroma collection, and prints the nearest chunks with source metadata. Use `--chunks-path` and `--model` to target a specific indexed collection, `--collection-name` to override the generated collection name, and `--top-k` to choose how many chunks to return.

## Repository intent

This repository is intentionally narrow in scope: it is the RAG core of the larger Figma UI assistant project.

Its role is to provide a reliable, inspectable, and extensible foundation for grounded answers before integrating richer context such as screenshots, cursor position, accessibility information, or plugin-derived signals.
