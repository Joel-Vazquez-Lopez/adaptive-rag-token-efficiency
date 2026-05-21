#!/usr/bin/env python3
"""
Simple BEIR dataset converter.

This file converts a BEIR-style dataset into the simple format used by our
adaptive RAG experiment.

BEIR usually gives us three files:

1. corpus.jsonl
   Contains the documents.

2. queries.jsonl
   Contains the questions/claims.

3. qrels/test.tsv
   Contains the relevance labels:
   which documents are relevant for each query.

Our project wants only two files:

1. documents.jsonl
   {"doc_id": "...", "text": "..."}

2. queries.jsonl
   {"query_id": "...", "text": "...", "relevant_doc_ids": [...], "reference_answer": "..."}

This script assumes the raw BEIR files are already downloaded and extracted.
It does not download anything. It only converts files.
"""

import csv
import json
import argparse
from pathlib import Path


def read_jsonl(path):
    # Read a JSONL file.
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    # Write rows as JSONL.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_document_text(row):
    # BEIR documents normally have title and text.
    # We join both because both can help the retriever.
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    parts = [title, text]
    return " ".join(part for part in parts if part)


def read_qrels(path):
    # Read relevance labels.
    #
    # Output example:
    # {
    #   "query_1": ["doc_1", "doc_7"],
    #   "query_2": ["doc_3"]
    # }
    qrels = {}

    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file, delimiter="\t")

        for row in reader:
            score = int(row.get("score", 0))
            if score <= 0:
                continue

            query_id = row["query-id"]
            doc_id = row["corpus-id"]

            if query_id not in qrels:
                qrels[query_id] = []

            qrels[query_id].append(doc_id)

    return qrels


def convert(raw_dir, output_dir, max_queries=None):
    # These are the normal BEIR file names.
    corpus_path = raw_dir / "corpus.jsonl"
    queries_path = raw_dir / "queries.jsonl"
    qrels_path = raw_dir / "qrels" / "test.tsv"

    # Read the raw dataset.
    corpus_rows = read_jsonl(corpus_path)
    query_rows = read_jsonl(queries_path)
    qrels = read_qrels(qrels_path)

    documents = []
    document_text_by_id = {}

    # Convert documents into our simple format.
    for row in corpus_rows:
        doc_id = str(row["_id"])
        text = make_document_text(row)

        documents.append(
            {
                "doc_id": doc_id,
                "text": text,
            }
        )

        document_text_by_id[doc_id] = text

    queries = []

    # Convert queries into our simple format.
    for row in query_rows:
        query_id = str(row["_id"])

        # If a query has no gold relevant documents, we cannot evaluate it.
        if query_id not in qrels:
            continue

        relevant_doc_ids = []
        for doc_id in qrels[query_id]:
            if doc_id in document_text_by_id:
                relevant_doc_ids.append(doc_id)

        if not relevant_doc_ids:
            continue

        # BEIR is mainly for retrieval, so it usually does not give a short
        # answer. We use the gold relevant document text as our reference.
        reference_answer = ""
        for doc_id in relevant_doc_ids:
            reference_answer += document_text_by_id[doc_id] + " "
        reference_answer = reference_answer.strip()

        queries.append(
            {
                "query_id": query_id,
                "text": str(row["text"]),
                "relevant_doc_ids": relevant_doc_ids,
                "reference_answer": reference_answer,
            }
        )

        if max_queries is not None and len(queries) >= max_queries:
            break

    write_jsonl(output_dir / "documents.jsonl", documents)
    write_jsonl(output_dir / "queries.jsonl", queries)

    print(f"Documents written: {len(documents)}")
    print(f"Queries written:   {len(queries)}")
    print(f"Documents file:    {output_dir / 'documents.jsonl'}")
    print(f"Queries file:      {output_dir / 'queries.jsonl'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-queries", type=int, default=None)
    args = parser.parse_args()

    convert(args.raw_dir, args.output_dir, args.max_queries)


if __name__ == "__main__":
    main()

