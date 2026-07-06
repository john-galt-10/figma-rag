# Dataset creation

## Candidate generation

This is the prompt for codex to come up with new candidate QA for evaluation.

```
Create a candidate RAG golden dataset as JSONL. Read data\eval\codex_set.jsonl and append 100 new samples.


For each example, include the same set of fields of the entries already present in data\eval\codex_set.jsonl.
 

Rules:
- The supporting_span must be copied exactly from the source file.
- The question should sound like a real user question, not a docs heading. Prefer practical questions users would actually ask: put yourself in the shoes of a Figma user with doubts regarding the functioning of some tool who wants to ask a question to a virtual assistant. 
- DON'T ASK QUESTIONS REGARDING THE DOCUMENTATION, only practical software questions
- Include a mix of lookup, troubleshooting, comparison, configuration, and limitation questions 
- Avoid near-duplicates.
- Generate no more than 3 examples per heading.
- The expected_answer must be fully supported by the supporting_span.
- Exclude questions regarding Figma Sites/Make/Slides/FigJam/community/admin.
- Avoid files including tutorials

Use: 
- scripts/validate_rag_eval_set.py` to validate JSONL syntax, schema, document paths, spans, duplicate queries, and print span previews for manual semantic review.
- `data\eval\codex_reviews_files_2.lst` to keep track of which files have already been considered
- C:\Users\samue\miniconda3\envs\figma-navigator\python.exe as python runtime
```


## Candidate approval

Use python scripts/review_golden_candidates_ui.py to visualize candidates in a useful UI. 

The UI allows to modify the query wording, the answer span and the expected answer points.

Usage example:

```
python scripts/review_golden_candidates_ui.py
    --input-path eval/golden_candidates.jsonl `
    --output-path eval/golden_candidates_approved.jsonl `
    --start-row 42
```

## Candidate formatting

Run `scripts\format_approved_candidates.py` to turn the candidates into the complete format.


## Merging with older ones

Just copy and paste, appending to the old test set.