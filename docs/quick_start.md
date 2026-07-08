1. Run scripts/data_preparation.py to download and clean HTMLs, producing clean .md files
2. Run scripts/index_preparation.py to chunk documents, add them to a chroma and a bm25 index
3. Propagate labels from spans to chunks, using scripts/map_relevant_chunks.py
4. Add to a .env file a GITHUB_MODELS_TOKEN key, to run both generation and LLM judge

Now you should be able to run:
- scripts/retrieval_example.py
- scripts/evaluate_retriever.py
- scripts/generate_answer.py
- scripts/retrieval_example.py