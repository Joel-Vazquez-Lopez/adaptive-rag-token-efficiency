#!/usr/bin/env python3
"""
Simple HotpotQA converter.

Why this file exists:
Our experiment code expects every dataset to look the same:

1. documents.jsonl
   {"doc_id": "...", "text": "..."}

2. queries.jsonl
   {"query_id": "...", "text": "...", "relevant_doc_ids": [...], "reference_answer": "..."}

HotpotQA is useful for us because it has real question-answer pairs.
That fixes one limitation of SciFact, where we had to build a reference answer
from the gold evidence document.

This converter supports two input options:

Option A: Use HuggingFace datasets.
    python3 scripts/prepare_hotpotqa_simple.py \
      --from-huggingface \
      --split validation \
      --output-dir data/hotpotqa \
      --max-queries 150

Option B: Use a downloaded HotpotQA JSON or JSONL file.
    python3 scripts/prepare_hotpotqa_simple.py \
      --input-file /path/to/hotpotqa.json \
      --output-dir data/hotpotqa \
      --max-queries 150

The output can then be used directly by scripts/run_experiment.py.
"""

import argparse
import json
from pathlib import Path


def write_jsonl(path, rows):
    # Save a list of dictionaries as JSONL.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_local_rows(path):
    # Read either a JSON list file or a JSONL file.
    text = path.read_text(encoding="utf-8").strip()

    if not text:
        return []

    if text.startswith("["):
        return json.loads(text)

    rows = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def read_huggingface_rows(split):
    # Import here so the rest of the project does not require HuggingFace.
    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "To load HotpotQA directly from HuggingFace, install datasets first:\n"
            "python3 -m pip install datasets"
        ) from error

    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    return [dict(row) for row in dataset]


def normalize_context(raw_context):
    # HotpotQA context is usually:
    # [["Title A", ["sentence 1", "sentence 2"]], ["Title B", [...]]]
    #
    # HuggingFace can also expose it as:
    # {"title": [...], "sentences": [[...], [...]]}
    #
    # This function converts both shapes into:
    # [(title, text), (title, text), ...]
    documents = []

    if isinstance(raw_context, dict):
        titles = raw_context.get("title", [])
        sentence_groups = raw_context.get("sentences", [])
        for title, sentences in zip(titles, sentence_groups):
            text = " ".join(str(sentence) for sentence in sentences)
            documents.append((str(title), text))
        return documents

    if isinstance(raw_context, list):
        for item in raw_context:
            if not isinstance(item, list) or len(item) < 2:
                continue
            title = str(item[0])
            sentences = item[1]
            if isinstance(sentences, list):
                text = " ".join(str(sentence) for sentence in sentences)
            else:
                text = str(sentences)
            documents.append((title, text))
        return documents

    return documents


def supporting_fact_titles(row):
    # HotpotQA gives supporting facts as document titles plus sentence ids.
    # We use the titles as gold relevant document ids.
    raw = row.get("supporting_facts", [])

    titles = []
    if isinstance(raw, dict):
        for title in raw.get("title", []):
            titles.append(str(title))
    elif isinstance(raw, list):
        for item in raw:
            if isinstance(item, list) and item:
                titles.append(str(item[0]))

    return sorted(set(titles))


def row_id(row, index):
    # Different HotpotQA versions use different id names.
    return str(row.get("_id") or row.get("id") or f"query_{index}")


def convert_rows(rows, output_dir, max_queries=None):
    # We store each query's context paragraphs as retrievable documents.
    # This is the HotpotQA distractor setup:
    # the system must choose useful paragraphs from the provided context.
    documents = []
    queries = []
    seen_doc_ids = set()

    for index, row in enumerate(rows):
        if max_queries is not None and len(queries) >= max_queries:
            break

        query_id = row_id(row, index)
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        contexts = normalize_context(row.get("context", []))

        if not question or not answer or not contexts:
            continue

        gold_titles = supporting_fact_titles(row)
        relevant_doc_ids = []

        for doc_index, (title, text) in enumerate(contexts):
            # Include the query id so same-titled pages from different questions
            # stay separate and do not accidentally merge.
            doc_id = f"{query_id}::doc_{doc_index}::{title}"
            doc_text = f"{title}. {text}".strip()

            if doc_id not in seen_doc_ids:
                documents.append({"doc_id": doc_id, "text": doc_text})
                seen_doc_ids.add(doc_id)

            if title in gold_titles:
                relevant_doc_ids.append(doc_id)

        # If supporting facts are missing, fall back to all context docs.
        # That keeps the query usable, but real HotpotQA should usually have them.
        if not relevant_doc_ids:
            relevant_doc_ids = [f"{query_id}::doc_{i}::{title}" for i, (title, _text) in enumerate(contexts)]

        queries.append(
            {
                "query_id": query_id,
                "text": question,
                "relevant_doc_ids": relevant_doc_ids,
                "reference_answer": answer,
            }
        )

    write_jsonl(output_dir / "documents.jsonl", documents)
    write_jsonl(output_dir / "queries.jsonl", queries)

    print(f"Documents written: {len(documents)}")
    print(f"Queries written:   {len(queries)}")
    print(f"Documents file:    {output_dir / 'documents.jsonl'}")
    print(f"Queries file:      {output_dir / 'queries.jsonl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=Path, default=None)
    parser.add_argument("--from-huggingface", action="store_true")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=150)
    args = parser.parse_args()

    if args.from_huggingface:
        rows = read_huggingface_rows(args.split)
    elif args.input_file is not None:
        rows = read_local_rows(args.input_file)
    else:
        raise RuntimeError("Use either --from-huggingface or --input-file.")

    convert_rows(rows, args.output_dir, args.max_queries)


if __name__ == "__main__":
    main()
