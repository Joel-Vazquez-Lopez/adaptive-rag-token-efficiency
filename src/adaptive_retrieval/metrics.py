"""
Evaluation metrics.

This file contains small metric functions used across the project.

Important metrics:

- precision_at_k:
  Of the selected documents, how many are relevant?

- recall:
  Of all relevant documents, how many did we retrieve?

- reciprocal_rank / MRR:
  Did a relevant document appear near the top?

- ndcg_at_k:
  Are relevant documents ranked high in the selected context?

- token_f1:
  Word-overlap F1 between generated answer and reference text.

- answer_coverage:
  How much of the reference vocabulary appears in the generated answer?

Token F1 is useful for comparing systems, but it is not perfect. It can punish
answers that are semantically correct but phrased differently.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from adaptive_retrieval.text import tokenize


@dataclass
class RunMetrics:
    mode: str
    iteration: int
    query_id: str
    precision_at_k: float
    recall: float
    mrr: float
    ndcg_at_10: float
    docs_used: int
    tokens_used: int
    context_precision: float
    context_noise_tokens: float
    answer_f1: float
    answer_coverage: float


def precision_at_k(selected_doc_ids: list[str], relevant_doc_ids: frozenset[str]) -> float:
    if not selected_doc_ids:
        return 0.0
    return len(set(selected_doc_ids) & relevant_doc_ids) / len(selected_doc_ids)


def recall(selected_doc_ids: list[str], relevant_doc_ids: frozenset[str]) -> float:
    if not relevant_doc_ids:
        return 0.0
    return len(set(selected_doc_ids) & relevant_doc_ids) / len(relevant_doc_ids)


def reciprocal_rank(ranked_doc_ids: list[str], relevant_doc_ids: frozenset[str]) -> float:
    for index, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return 1 / index
    return 0.0


def ndcg_at_k(ranked_doc_ids: list[str], relevant_doc_ids: frozenset[str], k: int = 10) -> float:
    if not relevant_doc_ids:
        return 0.0

    gains = [1.0 if doc_id in relevant_doc_ids else 0.0 for doc_id in ranked_doc_ids[:k]]
    dcg = sum(gain / math_log2(rank + 2) for rank, gain in enumerate(gains))

    ideal_relevant = min(len(relevant_doc_ids), k)
    ideal_dcg = sum(1.0 / math_log2(rank + 2) for rank in range(ideal_relevant))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def math_log2(value: int) -> float:
    # Tiny wrapper keeps ndcg_at_k readable without importing all of math at call sites.
    import math

    return math.log2(value)


def token_f1(candidate: str, reference: str) -> float:
    candidate_terms = tokenize(candidate)
    reference_terms = tokenize(reference)
    if not candidate_terms or not reference_terms:
        return 0.0

    candidate_counts = Counter(candidate_terms)
    reference_counts = Counter(reference_terms)
    overlap = sum((candidate_counts & reference_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(candidate_terms)
    answer_recall = overlap / len(reference_terms)
    return (2 * precision * answer_recall) / (precision + answer_recall)


def answer_coverage(candidate: str, reference: str) -> float:
    candidate_terms = set(tokenize(candidate))
    reference_terms = set(tokenize(reference))
    if not reference_terms:
        return 0.0
    return len(candidate_terms & reference_terms) / len(reference_terms)
