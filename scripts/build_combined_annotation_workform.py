#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import pandas as pd


METHODS = [
    "fixed_10_full",
    "heuristic_rules_full",
    "answer_aware_fallback",
]

METHOD_DISPLAY = {
    "fixed_10_full": "Fixed Top-10",
    "heuristic_rules_full": "Heuristic Rules",
    "answer_aware_fallback": "Main Adaptive Method",
}

DATASET_CONFIGS = {
    "scifact": {
        "n_queries": 13,
        "queries_candidates": [
            "data/scifact/queries_150_seed0_llm_gold_v2.jsonl",
            "data/scifact/queries_150_seed0_llm_gold.jsonl",
            "data/scifact/queries_150_seed0.jsonl",
            "data/scifact/queries_all.jsonl",
        ],
        "answers_candidates": [
            "saved_results/scifact_llama70b_final_eval100/llm_answers_by_query.csv",
            "outputs/scifact_llama70b_merged_eval100/llm_answers_by_query.csv",
        ],
    },
    "hotpotqa": {
        "n_queries": 13,
        "queries_candidates": [
            "data/hotpotqa_classmate/queries_150.jsonl",
            "data/hotpotqa_final/queries_150.jsonl",
            "data/hotpotqa/queries.jsonl",
        ],
        "answers_candidates": [
            "saved_results/hotpotqa_llama70b_final_eval100/llm_answers_by_query.csv",
            "saved_results/hotpotqa_llama70b_final_eval/llm_answers_by_query.csv",
            "saved_results/hotpotqa_llama70b_final/llm_answers_by_query.csv",
        ],
    },
    "bioasq": {
        "n_queries": 8,
        "queries_candidates": [
            "data/bioasq_candidate/queries.jsonl",
        ],
        "answers_candidates": [
            "saved_results/bioasq_llama70b_final_eval100/llm_answers_by_query.csv",
            "outputs/bioasq_llama70b_80_eval100_merged/llm_answers_by_query.csv",
        ],
    },
}


def find_existing_path(candidates, label):
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Could not find {label}. Tried:\n" + "\n".join(f"  - {c}" for c in candidates)
    )


def read_queries(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            rows[str(row["query_id"])] = row
    return rows


def clean_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).replace("\r\n", "\n").replace("\r", "\n").strip()


def parse_selected_doc_ids(value):
    text = clean_text(value)
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return []


def complete_query_ids(df: pd.DataFrame, methods: list[str]) -> list[str]:
    sub = df[df["mode"].isin(methods)]
    counts = sub.groupby("query_id")["mode"].nunique()
    return sorted(counts[counts == len(methods)].index.astype(str).tolist())


def score_interesting_queries(df: pd.DataFrame, qids: list[str]) -> list[tuple[float, str]]:
    scored = []

    for qid in qids:
        sub = df[df["query_id"] == qid].set_index("mode")

        fixed = sub.loc["fixed_10_full"]
        heur = sub.loc["heuristic_rules_full"]
        adapt = sub.loc["answer_aware_fallback"]

        adapt_f1 = float(adapt.get("answer_f1", 0.0))
        fixed_f1 = float(fixed.get("answer_f1", 0.0))
        heur_f1 = float(heur.get("answer_f1", 0.0))

        adapt_sem = float(adapt.get("semantic_similarity", 0.0))
        fixed_sem = float(fixed.get("semantic_similarity", 0.0))
        heur_sem = float(heur.get("semantic_similarity", 0.0))

        f1_disagreement = max(
            abs(adapt_f1 - fixed_f1),
            abs(adapt_f1 - heur_f1),
            abs(fixed_f1 - heur_f1),
        )
        sem_disagreement = max(
            abs(adapt_sem - fixed_sem),
            abs(adapt_sem - heur_sem),
            abs(fixed_sem - heur_sem),
        )
        low_quality = 1.0 - adapt_f1

        score = (0.45 * f1_disagreement) + (0.35 * sem_disagreement) + (0.20 * low_quality)
        scored.append((score, qid))

    return sorted(scored, reverse=True)


def sample_query_ids(df: pd.DataFrame, n_queries: int, seed: int) -> list[str]:
    qids = complete_query_ids(df, METHODS)
    if len(qids) < n_queries:
        raise RuntimeError(f"Only {len(qids)} complete queries found; need {n_queries}")

    rng = random.Random(seed)

    ranked = score_interesting_queries(df, qids)

    # Roughly half interesting/disagreement cases, half random cases.
    n_interesting = max(1, n_queries // 2)
    interesting = [qid for _, qid in ranked[:n_interesting]]

    remaining = [qid for qid in qids if qid not in set(interesting)]
    random_part = rng.sample(remaining, n_queries - len(interesting))

    return sorted(interesting + random_part)


def build_dataset_rows(dataset: str, config: dict, seed: int):
    queries_path = find_existing_path(config["queries_candidates"], f"{dataset} queries")
    answers_path = find_existing_path(config["answers_candidates"], f"{dataset} answers")

    print(f"\n[{dataset}]")
    print(f"Queries: {queries_path}")
    print(f"Answers: {answers_path}")

    queries = read_queries(queries_path)
    answers = pd.read_csv(answers_path)

    required = {"query_id", "mode", "answer"}
    missing = required - set(answers.columns)
    if missing:
        raise ValueError(f"{answers_path} missing required columns: {sorted(missing)}")

    answers["query_id"] = answers["query_id"].astype(str)
    answers["mode"] = answers["mode"].astype(str)

    n_queries = config["n_queries"]
    sampled_qids = sample_query_ids(answers, n_queries=n_queries, seed=seed)

    print(f"Sampled queries: {len(sampled_qids)}")

    annotation_rows = []
    key_rows = []

    # Dataset-specific blinded system mapping.
    # This prevents System A from always meaning the same method across datasets.
    rng = random.Random(seed + sum(ord(ch) for ch in dataset))
    shuffled_methods = METHODS[:]
    rng.shuffle(shuffled_methods)
    system_labels = [f"System {chr(ord('A') + i)}" for i in range(len(shuffled_methods))]
    method_to_system = dict(zip(shuffled_methods, system_labels))

    print("Private system mapping:")
    for method, system in method_to_system.items():
        print(f"  {system}: {method}")

    for local_idx, qid in enumerate(sampled_qids, start=1):
        qrow = queries.get(qid, {})
        query_text = clean_text(qrow.get("text", ""))
        reference_answer = clean_text(qrow.get("reference_answer", ""))
        relevant_doc_ids = qrow.get("relevant_doc_ids", [])

        sub = answers[(answers["query_id"] == qid) & (answers["mode"].isin(METHODS))].copy()
        sub = sub.sample(frac=1.0, random_state=seed + local_idx)

        for _, ans in sub.iterrows():
            mode = ans["mode"]
            system = method_to_system[mode]
            annotation_id = f"{dataset}_{local_idx:03d}_{system.replace(' ', '')}"

            annotation_rows.append(
                {
                    "annotation_id": annotation_id,
                    "dataset": dataset,
                    "query_id": qid,
                    "system": system,
                    "query_text": query_text,
                    "reference_answer": reference_answer,
                    "generated_answer": clean_text(ans.get("answer", "")),
                    "selected_doc_ids": json.dumps(
                        parse_selected_doc_ids(ans.get("selected_doc_ids", "")),
                        ensure_ascii=False,
                    ),
                    "relevant_doc_ids": json.dumps(relevant_doc_ids, ensure_ascii=False),
                    "auto_answer_f1": ans.get("answer_f1", ""),
                    "auto_answer_coverage": ans.get("answer_coverage", ""),
                    "auto_semantic_similarity": ans.get("semantic_similarity", ""),
                    "total_tokens": ans.get("total_tokens", ""),
                    "fallback_used": ans.get("fallback_used", ""),
                    "fallback_reason": clean_text(ans.get("fallback_reason", "")),
                    "human_correctness": "",
                    "human_failure_type": "",
                    "human_notes": "",
                }
            )

            key_rows.append(
                {
                    "annotation_id": annotation_id,
                    "dataset": dataset,
                    "query_id": qid,
                    "system": system,
                    "mode": mode,
                    "method_name": METHOD_DISPLAY.get(mode, mode),
                }
            )

    metadata = {
        "dataset": dataset,
        "queries_path": str(queries_path),
        "answers_path": str(answers_path),
        "sampled_queries": len(sampled_qids),
        "annotation_rows": len(annotation_rows),
    }

    return annotation_rows, key_rows, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="annotation_workform/combined_llama70b_final")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_annotation_rows = []
    all_key_rows = []
    metadata_rows = []

    for dataset, config in DATASET_CONFIGS.items():
        annotation_rows, key_rows, metadata = build_dataset_rows(dataset, config, args.seed)
        all_annotation_rows.extend(annotation_rows)
        all_key_rows.extend(key_rows)
        metadata_rows.append(metadata)

    annotation_df = pd.DataFrame(all_annotation_rows)
    key_df = pd.DataFrame(all_key_rows)
    metadata_df = pd.DataFrame(metadata_rows)

    annotation_path = out_dir / "annotation_items_blinded.csv"
    key_path = out_dir / "annotation_key_private.csv"
    metadata_path = out_dir / "source_files.csv"

    annotation_df.to_csv(annotation_path, index=False)
    key_df.to_csv(key_path, index=False)
    metadata_df.to_csv(metadata_path, index=False)

    readme = f"""# Combined Llama-70B Human Annotation Workform

This folder contains the combined blinded annotation workform for the final Llama-70B experiments.

## Annotation design

The workform samples answers from three datasets:

- SciFact: 13 queries x 3 systems = 39 answers
- HotpotQA: 13 queries x 3 systems = 39 answers
- BioASQ: 8 queries x 3 systems = 24 answers

Total: {len(annotation_df)} answers.

Only one LLM is used: `meta-llama/Llama-3.3-70B-Instruct`.

## Systems included

For each sampled query, annotators compare three blinded systems:

- Fixed Top-10
- Heuristic Rules
- Main Adaptive Method

Only the final report modes are included:

- `fixed_10_full`
- `heuristic_rules_full`
- `answer_aware_fallback`

Extra raw experimental rows such as `fixed_3_full`, `fixed_5_full`, compressed fixed baselines, and no-retrieval rows are intentionally excluded from annotation.

## Files

- `annotation_items_blinded.csv`: give this to annotators.
- `annotation_key_private.csv`: private method mapping; do not give this to annotators during blind annotation.
- `source_files.csv`: records which result/query files were used.
- `README.md`: this file.

## Annotation labels

Fill the `human_correctness` column with one of:

- `correct`: the answer contains the required information and does not contradict the reference.
- `partial`: the answer contains some required information but is incomplete, vague, or misses an important condition.
- `wrong`: the answer contradicts the reference or gives a different answer.
- `unclear`: the answer cannot be judged confidently from the query and reference.

Optionally fill `human_failure_type` with one of:

- `retrieval_failure`
- `context_selection_failure`
- `generation_failure`
- `evaluation_mismatch`
- `insufficient_evidence`
- `none`

## Annotation instructions

Judge the generated answer against the query and the reference answer.

The automatic metrics are included for later analysis, but human labels should be based on the answer content, not on the metric values.

The `annotation_key_private.csv` file should remain hidden until annotation is complete.
"""

    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"\nWrote combined annotation workform to: {out_dir}")
    print(f"Rows: {len(annotation_df)}")
    print("\nRows by dataset:")
    print(annotation_df["dataset"].value_counts().to_string())
    print("\nRows by blinded system:")
    print(annotation_df["system"].value_counts().to_string())


if __name__ == "__main__":
    main()
