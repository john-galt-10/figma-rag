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

## Milestone recap

* **🌱 2026-06-21: Project foundation and ingestion baseline**  
  Initial repository structure, README, Figma Help Center ingestion module, download script, and ingestion tests.

* **🧹 2026-06-22: Markdown normalization and hierarchical chunking**  
  Added the HTML-to-Markdown cleaning pipeline, structured document normalization, and the first heading-aware chunking pipeline.

* **🔎 2026-06-26: Local semantic retrieval baseline**  
  Added Chroma vector indexing, Sentence Transformers embeddings, the reusable Chroma retriever, and the retrieval example script.

* **🎯 2026-06-26: Chunk span tracking for evaluation alignment**  
  Added source span tracking to chunks so document-level annotations can be mapped back to chunk IDs.

* **📊 2026-06-29: Retrieval evaluation and filtering pipeline**  
  Added label-to-chunk mapping, retrieval metrics evaluation, static metadata filtering, and the declared retrieval pipeline.

* **🧭 2026-06-30: Retrieval analysis and annotation workflow**  
  Added metadata filtering improvements, breadcrumbs, a retrieval label review UI, annotation completion tooling, and pipeline documentation.

* **🧱 2026-07-01 to 2026-07-02: BM25 lexical retrieval baseline**  
  Added BM25 index building and BM25 retrieval support alongside the semantic retriever.

* **⚡ 2026-07-02: Hybrid retrieval aggregation**  
  Added aggregation logic to combine retrieval outputs, enabling hybrid search experiments over semantic and lexical retrieval results.

* **📈 2026-07-03: Reranking experiments and retrieval metric visualization**  
  Added optional cross-encoder reranking to the retrieval pipeline, evaluation script, and retrieval example, with candidate-k controls, rerank score metadata, latency summaries, and plotting utilities for comparing aggregate retrieval metrics.

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
