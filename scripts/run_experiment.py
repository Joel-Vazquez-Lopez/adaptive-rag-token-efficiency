#!/usr/bin/env python3
"""
Clean runner for the real Adaptive Context model.

This file is small on purpose, but it calls the same tested model code in:

    src/adaptive_retrieval/llm_budget.py

So the GitHub project uses our real method:

1. Fixed top-k baselines.
2. Basic adaptive budget.
3. Safe Adaptive Context:
   - first pass: compact evidence with evidence_ngram_neighbors
   - safety check: inspect the generated answer
   - fallback: expand to full top-10 if the answer looks weak

How to read this file:

- The first section imports the real model code.
- METHODS_TO_RUN decides which experimental systems are evaluated.
- FINAL_TABLE_ROWS decides which rows appear in the clean final report table.
- build_final_table() converts the detailed experiment output into a small table.
- main() loads the data, configures Ollama/Mistral, runs the experiment, and saves outputs.

This file does not implement the model itself. It is the control script.
The model implementation is in src/adaptive_retrieval/llm_budget.py.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path


# Let this script import the project package from src/.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptive_retrieval.data import load_documents, load_queries
from adaptive_retrieval.llm_budget import LLMConfig, run_llm_budget_experiment, write_llm_outputs


# These are the methods we want in the final report table.
# fixed_10 is the expensive baseline.
# learned_budget is the basic adaptive model.
# answer_aware_fallback is our stronger Safe Adaptive model.
METHODS_TO_RUN = [
    "no_retrieval",
    "fixed_3",
    "fixed_5",
    "fixed_7",
    "fixed_10",
    "heuristic_rules",
    "learned_budget",
    "answer_aware_fallback",
]


# The real experiment outputs several rows because it tests full context and compact evidence.
# These are the exact rows we want to show as the clean comparison.
FINAL_TABLE_ROWS = {
    "no_retrieval_full": "No Retrieval",
    "fixed_3_full": "Fixed Top-3",
    "fixed_5_full": "Fixed Top-5",
    "fixed_7_full": "Fixed Top-7",
    "fixed_10_full": "Fixed Top-10",
    "heuristic_rules_full": "Heuristic Rules",
    "learned_budget_evidence_ngram_neighbors": "Basic Adaptive + Compact Evidence",
    "answer_aware_fallback": "Safe Adaptive Context",
}


def as_float(row, key):
    # CSV/table values can be strings, so this converts them safely.
    return float(row[key])


def percent(value):
    # Format decimals as percentages.
    return f"{value * 100:.1f}%"


def build_final_table(answer_summary, dataset_name):
    # Make one small table for the report.
    # The baseline for token/time reduction is fixed_10_full.
    row_by_mode = {str(row["mode"]): row for row in answer_summary}
    baseline = row_by_mode["fixed_10_full"]

    baseline_f1 = as_float(baseline, "answer_f1")
    baseline_tokens = as_float(baseline, "total_tokens")
    baseline_time = as_float(baseline, "generation_time_ms")

    rows = []
    for mode, method_name in FINAL_TABLE_ROWS.items():
        row = row_by_mode[mode]

        f1 = as_float(row, "answer_f1")
        tokens = as_float(row, "total_tokens")
        time_ms = as_float(row, "generation_time_ms")
        fallback_rate = as_float(row, "fallback_rate")

        rows.append(
            {
                "dataset": dataset_name,
                "method": method_name,
                "code_mode": mode,
                "ndcg_at_10": round(as_float(row, "ndcg_at_10"), 6),
                "answer_f1": round(f1, 6),
                "f1_retained_vs_top10": percent(f1 / baseline_f1 if baseline_f1 else 0),
                "total_tokens": round(tokens, 2),
                "token_reduction_vs_top10": percent(1 - (tokens / baseline_tokens) if baseline_tokens else 0),
                "generation_time_ms": round(time_ms, 2),
                "time_reduction_vs_top10": percent(1 - (time_ms / baseline_time) if baseline_time else 0),
                "fallback_rate": percent(fallback_rate),
            }
        )

    return rows


def write_csv(path, rows):
    # Save a table as CSV so it can be opened in Excel or Google Sheets.
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    # Save the same table as Markdown for the report.
    if not rows:
        return
    headers = list(rows[0].keys())
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        lines.append("| " + " | ".join(str(row[header]) for header in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_table(rows):
    # Print the final table in the terminal.
    if not rows:
        return
    headers = list(rows[0].keys())
    print(" | ".join(headers))
    print(" | ".join("-" * len(header) for header in headers))
    for row in rows:
        print(" | ".join(str(row[header]) for header in headers))


def set_seed(seed):
    # Make every Python-side choice deterministic.
    #
    # The current experiment is already mostly deterministic because:
    # - the query sample file is fixed
    # - the train/eval split sorts by query id
    # - the learned budget model has no random initialization
    #
    # This seed is still useful because it documents reproducibility and protects
    # future changes if random sampling is added later.
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    parser = argparse.ArgumentParser(description="Run the real Adaptive RAG experiment.")

    # Dataset paths.
    parser.add_argument("--documents", type=Path, default=Path("data/scifact/documents.jsonl"))
    parser.add_argument("--queries", type=Path, default=Path("data/scifact/queries_150_seed0.jsonl"))
    parser.add_argument("--dataset-name", default="SciFact")

    # Output folder.
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scifact_run"))

    # LLM settings.
    parser.add_argument("--model", default="mistral")
    parser.add_argument("--api-url", default="http://localhost:11434/v1")
    parser.add_argument("--no-api-key", action="store_true", default=True)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--request-timeout-seconds", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-provider-tokens",
        action="store_true",
        help="Use this for final runs so token numbers must come from Ollama/provider usage.",
    )

    # Experiment size.
    parser.add_argument("--max-eval-queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    set_seed(args.seed)

    print("Loading data...")
    documents = load_documents(args.documents)
    queries = load_queries(args.queries)

    print("Configuring model...")
    config = LLMConfig(
        model=args.model,
        api_url=args.api_url,
        api_key_env=args.api_key_env,
        require_api_key=not args.no_api_key,
        dry_run=args.dry_run,
        request_timeout_seconds=args.request_timeout_seconds,
        require_provider_tokens=args.require_provider_tokens,
    )

    print("Running experiment...")
    answer_rows, answer_summary, retrieval_summary = run_llm_budget_experiment(
        documents=documents,
        queries=queries,
        dev_ratio=0.4,
        config=config,
        max_eval_queries=args.max_eval_queries,
        modes=METHODS_TO_RUN,
        compression_modes=["full", "evidence_ngram_neighbors"],
        oracle_strategy="minimum_sufficient",
        sufficiency_ratio=0.95,
        threshold_strategy="heuristic",
    )

    # Save detailed outputs from the real model pipeline.
    write_llm_outputs(args.output_dir, answer_rows, answer_summary, retrieval_summary)

    # Save the clean final table.
    final_rows = build_final_table(answer_summary, args.dataset_name)
    write_csv(args.output_dir / "final_table.csv", final_rows)
    write_markdown(args.output_dir / "final_table.md", final_rows)

    print("\nFinal table")
    print_table(final_rows)
    print(f"\nWrote outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
