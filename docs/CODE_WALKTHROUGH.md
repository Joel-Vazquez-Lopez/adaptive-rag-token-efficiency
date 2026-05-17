# Code Walkthrough

This document explains the important parts of the code in plain language.

The project has one main goal:

> Compare fixed-context RAG with adaptive-context RAG, and test whether Safe Adaptive Context can save tokens while keeping answer quality close to fixed top-10.

## 1. The Main Script

File:

```text
scripts/run_experiment.py
```

This is the file you run from the terminal.

It does four things:

1. Loads SciFact documents and queries.
2. Configures the model, usually Mistral through Ollama.
3. Runs the experiment methods.
4. Saves the results tables.

The important method list is:

```python
METHODS_TO_RUN = [
    "fixed_3",
    "fixed_5",
    "fixed_10",
    "learned_budget",
    "answer_aware_fallback",
]
```

Meaning:

- `no_retrieval`: closed-book baseline, k = 0
- `fixed_3`: always use top-3 full documents
- `fixed_5`: always use top-5 full documents
- `fixed_7`: always use top-7 full documents
- `fixed_10`: always use top-10 full documents
- `heuristic_rules`: choose k with simple query-length and score-gap rules
- `learned_budget`: basic adaptive budget model
- `answer_aware_fallback`: our Safe Adaptive model

The final clean table is created by:

```python
build_final_table(...)
```

That function takes the detailed experiment output and keeps only the rows we
want for the report.

The runner also sets:

```python
set_seed(args.seed)
```

This fixes Python-side randomness. The current model is already mostly
deterministic, but the seed makes reproducibility explicit.

## 2. Loading The Dataset

File:

```text
src/adaptive_retrieval/data.py
```

This file defines:

```python
Document
Query
load_documents(...)
load_queries(...)
```

The project uses a simple JSONL format.

Documents look like:

```json
{"doc_id": "123", "text": "document text..."}
```

Queries look like:

```json
{
  "query_id": "1",
  "text": "question or claim...",
  "relevant_doc_ids": ["123"],
  "reference_answer": "gold reference text..."
}
```

Why this matters:

The code does not need to know the original BEIR format. Everything is converted
into this simple format first.

## 3. Text Processing

File:

```text
src/adaptive_retrieval/text.py
```

This file contains small helper functions:

- `tokenize(...)`: turns text into lowercase word tokens
- `estimate_tokens(...)`: rough backup token estimate
- `build_idf(...)`: builds IDF scores for TF-IDF retrieval
- `tfidf_vector(...)`: turns text into a TF-IDF vector
- `cosine_similarity(...)`: compares query and document vectors

Why this matters:

The retrieval system uses TF-IDF cosine similarity to decide which documents
look relevant to the query.

## 4. Retrieval

File:

```text
src/adaptive_retrieval/retriever.py
```

The main retrieval function is:

```python
retrieve(...)
```

It ranks documents by similarity to the query.

For each query:

1. Convert query to a TF-IDF vector.
2. Compare it to every document vector.
3. Sort documents by score.
4. Return the top documents.

Important distinction:

Retrieval is not the same as context selection.

- Retrieval asks: "Which documents might be useful?"
- Context selection asks: "How much of this evidence should we send to the LLM?"

Our project is mainly about the second question.

## 5. Basic Adaptive Budget

File:

```text
src/adaptive_retrieval/learned_budget.py
```

This is the basic adaptive model.

It predicts a context budget using cheap features available before generation.

Examples of features:

- query length
- top retrieval score
- score gaps between retrieved documents
- score entropy
- whether evidence looks concentrated or spread across documents

The model is intentionally simple:

- no neural network
- no deep learning
- no external training API

It trains a small nearest-centroid style classifier from oracle labels.

The helper file:

```text
src/adaptive_retrieval/budget_experiment.py
```

computes retrieval-side metrics used by this model. It is now intentionally
small and only keeps the part needed for the final experiment.

Important idea:

The oracle is not deployable because it uses evaluation information, but it can
teach a small model what kind of query usually needs more context.

## 6. Evidence Compression

File:

```text
src/adaptive_retrieval/llm_budget.py
```

Important function:

```python
compress_document(...)
```

The compact evidence mode we use is:

```text
evidence_ngram_neighbors
```

It works like this:

1. Split a document into sentences.
2. Score sentences by word and phrase overlap with the query.
3. Keep useful evidence sentences.
4. Also keep neighboring sentences, because isolated sentences can lose context.

Why this matters:

Instead of always sending full documents, the system can send smaller evidence
spans that are more focused on the query.

## 7. LLM Prompting

File:

```text
src/adaptive_retrieval/llm_budget.py
```

Important function:

```python
build_prompt(...)
```

This function takes:

- the query
- the selected documents or evidence spans

and turns them into a prompt for Mistral.

The default prompt tells the model:

- answer using only the provided documents
- say evidence is insufficient if the documents do not answer the question

## 8. Calling Ollama / Mistral

File:

```text
src/adaptive_retrieval/llm_budget.py
```

Important function:

```python
call_openai_chat(...)
```

Ollama provides an OpenAI-compatible endpoint, so the code sends a chat
completion request to:

```text
http://localhost:11434/v1/chat/completions
```

The model returns:

- generated answer
- prompt tokens
- completion tokens
- total tokens

For final results, we use:

```bash
--require-provider-tokens
```

That forces the run to use real token counts from the model/provider.

## 9. Safe Adaptive Context

File:

```text
src/adaptive_retrieval/llm_budget.py
```

Most important function:

```python
answer_aware_fallback_run(...)
```

This is our main model.

It works in two stages.

### Stage 1: Compact First Pass

The system:

1. retrieves top documents
2. uses the adaptive budget to choose the first context size
3. compresses the context with `evidence_ngram_neighbors`
4. asks Mistral to answer

This is the cheap attempt.

### Stage 2: Safety Check

The function:

```python
answer_needs_fallback(...)
```

checks whether the first answer looks risky.

Risk signals include:

- empty answer
- "insufficient evidence"
- "cannot determine"
- "not mentioned"
- very short answer
- answer barely overlaps with the evidence
- answer barely overlaps with the query

### Stage 3: Fallback

If the answer looks risky:

1. the system expands to full top-10 documents
2. asks Mistral again
3. returns the fallback answer
4. counts both the first-pass tokens and fallback tokens

If the answer does not look risky:

1. the system keeps the compact answer
2. avoids paying for full top-10 context

This is the core idea:

> Use small context when possible, but expand when the answer looks unsafe.

## 10. Metrics

File:

```text
src/adaptive_retrieval/metrics.py
```

Important metric:

```python
token_f1(...)
```

Token F1 compares the generated answer to the reference text by word overlap.

The final table reports:

- answer F1
- total tokens
- token reduction vs fixed top-10
- generation time
- time reduction vs fixed top-10
- fallback rate

## 11. Result Files

After running the experiment, the output folder contains:

```text
llm_answers_by_query.csv
llm_summary.csv
retrieval_summary.csv
final_table.csv
final_table.md
```

The most important file for the report is:

```text
final_table.csv
```

## 12. How To Explain The Project In One Minute

Most RAG systems use a fixed amount of context, like top-10 documents. This is
simple but can waste tokens.

Our system retrieves candidate documents, builds a smaller evidence context, and
asks the LLM to answer. Then it checks whether the answer looks weak. If it does,
the system expands to full top-10 context and tries again.

So the model is adaptive:

- easy queries stay cheap
- hard queries get more context
- final results compare quality, tokens, time, and fallback rate
