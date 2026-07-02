## Repository structure

Use the existing repository layout and do not introduce new top-level folders unless necessary.

- `scripts/`: entrypoint scripts and one-off pipeline commands
- `src/`: reusable Python modules and pipeline logic
- `data/raw/`: raw fetched documentation and source artifacts
- `data/processed/`: cleaned documents, chunks, and derived artifacts
- `tests/`: automated tests
- `docs/`: design notes, architecture, and planning documents

When implementing new functionality:
- put reusable logic in `src/`
- keep CLI or task entrypoints in `scripts/`
- store raw fetched Figma documentation under `data/raw/figma_docs/`
- keep generated metadata close to the dataset it describes
- avoid creating additional top-level directories unless explicitly needed

You can find general info about the project in `README.md`.

## Technology stack

Use a Python-first stack for the entire RAG system.

Core technologies:

- Python 3.11.15 (use figma-navigator conda environment) 
- `sentence-transformers` for text embeddings
- `requests` or `httpx` for document fetching
- `beautifulsoup4` for HTML parsing when needed
- `numpy` for vector handling and evaluation utilities
- simple local storage using files and JSONL

Guidelines:

- prefer plain Python over framework-heavy RAG stacks
- do not introduce LangChain, LlamaIndex, or workflow frameworks unless explicitly requested
- keep ingestion, cleaning, chunking, indexing, retrieval, and evaluation as separate modules
- prefer local, file-based artifacts during early development
- keep dependencies minimal and easy to understand
- design the code so retrieval components can later be swapped without restructuring the whole project


Retrieval approach:

- use Sentence Transformers for the initial embedding-based retrieval baseline
- keep the first retriever implementation simple and transparent
- additional retrieval improvements, such as reranking or hybrid retrieval, can be added later only after the baseline works

The default goal is not production scale. The goal is to build a clear, inspectable RAG baseline that supports learning and iteration.

## General indications

- Don't create tests if not explicitly asked. Not everything needs tests.
- Run code (tests included) only if explicitly asked.
- Everytime you implement a script, present in the chat how to run it and the meaning of the CLI parameters. Also write a very short summary of the functioning (e.g., input, output, underlying mechanism).
- Comment the code you write. I want a docstring for each class and function, if not obvious or self-explanatory.
- Whenever adding new features or making significant changes, prompt me whether we should update the "Milestone recap" section in the README.md of the repo.
