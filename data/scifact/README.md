# SciFact Data

This folder contains the converted SciFact files used by the experiment.

- `documents.jsonl`: 5,183 biomedical abstracts.
- `queries_all.jsonl`: all converted SciFact queries available in this project copy.
- `queries_150_seed0.jsonl`: fixed 150-query sample created with seed `0`.

Each document line looks like:

```json
{"doc_id": "...", "text": "..."}
```

Each query line looks like:

```json
{"query_id": "...", "text": "...", "relevant_doc_ids": ["..."], "reference_answer": "..."}
```

Where the information comes from:

- SciFact is a BEIR benchmark dataset.
- The relevant document ids come from BEIR/SciFact qrels.
- The reference answer is the gold relevant document text, used here for Token F1.

## Why We Use A 150-Query Sample

The full dataset is useful, but local LLM evaluation can be slow. The fixed
150-query sample gives us a reproducible evaluation set:

- same queries every run
- seed `0`
- easier to compare methods fairly

The final report should always state how many queries were used.

## Possible Extra Datasets Later

The same JSONL format can also be used for:

- NFCorpus
- FiQA
- TREC-COVID

Those would help test whether the adaptive controller generalizes beyond SciFact.
