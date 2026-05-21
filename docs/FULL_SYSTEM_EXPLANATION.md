# Full System Explanation

This document explains the project from the beginning, in the simplest possible way.

The main idea is:

> Instead of always sending a fixed number of retrieved documents to the LLM, we test whether the system can send less context while keeping answer quality.

The project is a Retrieval-Augmented Generation system, also called RAG.

A normal RAG system does this:

```text
question -> retrieve top documents -> put documents in prompt -> LLM answers
```

Our project does this:

```text
question -> retrieve candidate documents -> choose/compact context -> LLM answers -> measure quality, tokens, and time
```

The most important part is the context controller.

It decides how much retrieved information should go into the prompt.


## 1. What The Dataset Looks Like

The project uses SciFact in a simplified JSONL format.

JSONL means:

> one JSON object per line

The code uses two files:

```text
data/scifact/documents.jsonl
data/scifact/queries_150_seed0.jsonl
```


### documents.jsonl

Each line is one document.

Shape:

```json
{
  "doc_id": "4983",
  "text": "Full scientific abstract text..."
}
```

Meaning:

- `doc_id`: the unique id of the scientific document
- `text`: the document text that can be retrieved and sent to the LLM


### queries_150_seed0.jsonl

Each line is one question/query.

Shape:

```json
{
  "query_id": "100",
  "text": "All hematopoietic stem cells segregate their chromosomes randomly.",
  "relevant_doc_ids": ["4381486"],
  "reference_answer": "Gold/reference text from the relevant document..."
}
```

Meaning:

- `query_id`: the unique id of the query
- `text`: the user question or scientific claim
- `relevant_doc_ids`: the gold documents that the dataset says are relevant
- `reference_answer`: the text used for answer-quality comparison

The important thing:

> The dataset tells us which documents are relevant. That lets us evaluate whether our system kept useful evidence.


## 2. What Happens When We Run The Experiment

The main command runs:

```bash
python3 scripts/run_experiment.py
```

That script controls the experiment.

It does not contain all the model logic itself. It calls the real implementation in:

```text
src/adaptive_retrieval/llm_budget.py
```

The experiment flow is:

```text
1. Load documents
2. Load queries
3. Build TF-IDF retrieval vectors
4. Retrieve top candidate documents for each query
5. Run several methods:
   - no retrieval
   - fixed top-3
   - fixed top-5
   - fixed top-7
   - fixed top-10
   - heuristic rules
   - basic adaptive + compact evidence
   - safe adaptive context
6. Send prompts to Mistral through Ollama
7. Read the LLM answer and token usage
8. Compute metrics
9. Save final tables
```


## 3. Retrieval: How Documents Are Found

Retrieval happens in:

```text
src/adaptive_retrieval/retriever.py
```

The system uses TF-IDF cosine similarity.

Simple explanation:

1. Convert the query into important words with weights.
2. Convert each document into important words with weights.
3. Compare query and document vectors.
4. Rank documents from most similar to least similar.

The retrieval stage asks:

> Which documents look relevant to this query?

It does not decide how much context to send to the LLM.

That is the job of the context-budget methods.


## 4. The Methods We Compare

The methods are listed in:

```text
scripts/run_experiment.py
```


### No Retrieval

Code mode:

```text
no_retrieval_full
```

Meaning:

> The LLM answers without retrieved documents.

This is the closed-book baseline.

It shows what the model can answer from its own knowledge only.


### Fixed Top-k

Code modes:

```text
fixed_3_full
fixed_5_full
fixed_7_full
fixed_10_full
```

Meaning:

> Always send the same number of top-ranked documents.

Examples:

- `fixed_3_full`: always send top 3 documents
- `fixed_10_full`: always send top 10 documents

Fixed top-10 is the expensive baseline.

We compare against it because many simple RAG systems use a fixed retrieval depth.


### Heuristic Rules

Code mode:

```text
heuristic_rules_full
```

Meaning:

> Use simple rules to choose how many documents to send.

The current heuristic looks at:

- query length
- score gap between ranked documents

If the top documents look clearly better than the rest, it can stop earlier.

This is an explainable adaptive baseline.


### Basic Adaptive + Compact Evidence

Code mode:

```text
learned_budget_evidence_ngram_neighbors
```

Meaning:

> Predict a document budget, then compress selected documents into evidence spans.

It saves tokens in two ways:

1. It may select fewer documents.
2. It does not send full documents. It sends relevant sentence spans instead.

The compression mode is:

```text
evidence_ngram_neighbors
```

This means:

1. Find sentences that overlap with the query.
2. Use unigram/bigram/trigram overlap.
3. Keep neighboring sentences too.

Neighboring sentences matter because a single sentence can be too isolated.


### Safe Adaptive Context

Code mode:

```text
answer_aware_fallback
```

This is our stronger adaptive method.

It works like this:

```text
1. Start with compact adaptive evidence.
2. Ask the LLM to answer.
3. Check if the answer looks weak or unsupported.
4. If it looks okay, keep the answer.
5. If it looks risky, expand to full top-10 context and ask again.
```

The goal is:

> Save tokens when compact context is enough, but protect quality when the answer looks risky.

The current fallback checker looks for:

- weak phrases like "insufficient evidence" or "cannot determine"
- very short answers
- low overlap between answer terms and context terms
- low overlap between answer terms and query terms

This is not a perfect checker.

But it is deployable because it only uses information available at runtime.

It does not use gold labels.


## 5. How Compression Works

Compression happens in:

```text
src/adaptive_retrieval/llm_budget.py
```

The important compression mode is:

```text
evidence_ngram_neighbors
```

It does this:

```text
For each selected document:
1. Split the document into sentences.
2. Score sentences by query overlap.
3. Prefer sentences with phrase overlap.
4. Keep the best evidence sentences.
5. Also keep one neighboring sentence before and after.
6. Build a shorter evidence block.
```

So instead of sending this:

```text
Full abstract with many unrelated sentences...
```

The LLM receives something like:

```text
Phrase-aware evidence spans:
- sentence before the evidence
- main evidence sentence
- sentence after the evidence
```

This is why token use drops.


## 6. How The Prompt Is Built

Prompt building happens in:

```text
build_prompt(...)
```

The default prompt says:

```text
Answer the question using only the provided documents.
If the documents do not contain enough evidence, say that the evidence is insufficient.

Question:
...

Documents:
...

Answer:
```

So each method changes what documents/evidence are placed inside the prompt.

The LLM call itself is the same.


## 7. How Ollama Is Used

The code talks to Ollama through an OpenAI-compatible endpoint.

The URL is:

```text
http://localhost:11434/v1/chat/completions
```

The model is:

```text
mistral
```

The code sends a JSON request shaped like the OpenAI Chat Completions API:

```json
{
  "model": "mistral",
  "temperature": 0.0,
  "max_tokens": 220,
  "messages": [
    {"role": "system", "content": "You are a careful retrieval-augmented QA assistant."},
    {"role": "user", "content": "Question + retrieved evidence"}
  ]
}
```

Ollama returns:

- the generated answer
- prompt token count
- completion token count
- total token count


## 8. How We Get Real Token Usage

Token usage is collected in:

```text
call_openai_chat(...)
```

After the LLM responds, the code reads:

```text
usage.prompt_tokens
usage.completion_tokens
usage.total_tokens
```

If those values exist, the table says:

```text
token_source = provider
```

That means the token numbers came from the actual model/API.

If the provider does not return token usage, the code can estimate tokens locally.

But for final results, we should prefer:

```text
token_source = provider
```

That is why final runs should use:

```bash
--require-provider-tokens
```

when the provider supports it.


## 9. How Answer F1 Is Computed

Answer F1 is computed in:

```text
src/adaptive_retrieval/metrics.py
```

Function:

```text
token_f1(candidate, reference)
```

It compares:

```text
LLM answer
vs
reference_answer from the dataset
```

It is word-overlap F1.

It measures:

- precision: how many answer words are also in the reference
- recall: how many reference words are covered by the answer
- F1: balance between precision and recall

Important limitation:

> Token F1 is lexical. It can punish correct paraphrases.

Example:

Reference:

```text
Vitamin D reduces inflammation.
```

Answer:

```text
Vitamin D lowers inflammatory markers.
```

This may be semantically correct, but token F1 may not be very high because the wording changed.


## 10. How nDCG@10 And MRR@10 Work

These metrics measure whether selected documents are relevant.

They use:

```text
relevant_doc_ids
```

from the dataset.


### nDCG@10

Question:

> Are relevant documents ranked high in the selected context?

Higher is better.

It rewards relevant documents more if they appear near the top.


### MRR@10

Question:

> How early does the first relevant document appear?

Examples:

- first relevant document at rank 1 -> MRR = 1.0
- first relevant document at rank 2 -> MRR = 0.5
- no relevant document -> MRR = 0.0


### Important Naming Note

In the current code, `ndcg_at_10` and `mrr_at_10` are computed on the documents selected into the prompt/context.

So they are best understood as:

```text
context_nDCG@10
context_MRR@10
```

They do not only measure the original retriever.

They measure whether the method kept useful documents in its final context.


## 11. What Each Output File Means

Each run creates an output folder like:

```text
outputs/scifact_real_10
```

Inside, the important files are:


### final_table.md

Clean report table.

This is the easiest file to show in meetings.


### final_table.csv

Same table, but spreadsheet format.


### llm_summary.csv

Aggregated results for each method.

This is used to build the final table.


### llm_answers_by_query.csv

Detailed per-query results.

This is useful for debugging.

It shows:

- query id
- method
- selected document ids
- generated answer
- tokens
- F1
- fallback used or not


### retrieval_summary.csv

Retrieval-side summary from the learned-budget pipeline.


## 12. How To Explain The Final Table

Example columns:

```text
dataset
method
code_mode
ndcg_at_10
mrr_at_10
answer_f1
f1_retained_vs_top10
total_tokens
token_reduction_vs_top10
generation_time_ms
time_reduction_vs_top10
fallback_rate
```

Meaning:

| Column | Meaning |
|---|---|
| `dataset` | Which dataset was tested |
| `method` | Human-readable method name |
| `code_mode` | Internal method name |
| `ndcg_at_10` | Whether selected context keeps relevant docs ranked high |
| `mrr_at_10` | How early the first relevant doc appears |
| `answer_f1` | Word-overlap answer quality |
| `f1_retained_vs_top10` | Method F1 compared with fixed top-10 F1 |
| `total_tokens` | Average prompt + answer tokens |
| `token_reduction_vs_top10` | Token savings compared with fixed top-10 |
| `generation_time_ms` | Average generation time |
| `time_reduction_vs_top10` | Time savings compared with fixed top-10 |
| `fallback_rate` | How often Safe Adaptive expanded context |


## 13. How To Explain The 10-Query Result

The 10-query SciFact result showed:

```text
Fixed Top-10:
answer_f1 = 0.202104
total_tokens = 3726.2

Safe Adaptive Context:
answer_f1 = 0.202268
total_tokens = 1277.2
token_reduction = 65.7%
fallback_rate = 10.0%
```

Plain explanation:

> On this small real run, Safe Adaptive matched fixed top-10 answer F1 while using around two-thirds fewer tokens.

But we should also say:

> This is only a 10-query sanity run. Final claims need larger runs.


## 14. Why 2 Queries Do Not Prove Anything

A 2-query run is a smoke test.

It checks:

```text
Does the code run?
Does the table generate?
Are all methods called?
```

It does not prove performance.

Why?

Because with 2 queries:

> each query controls 50% of the score

One easy or hard query can completely change the result.

So:

- 2 queries = code check
- 10 queries = small sanity test
- 50 queries = useful early evidence
- 100-150 queries = stronger final evidence


## 15. The Full Story In One Minute

If you need to explain the system quickly:

> We built a RAG pipeline on SciFact. First, it retrieves candidate documents using TF-IDF similarity. Then different methods decide how much context to send to Mistral through Ollama. Fixed baselines always send top-3, top-5, top-7, or top-10 full documents. Our adaptive methods try to reduce unnecessary context. Basic Adaptive selects a smaller budget and compresses documents into evidence spans. Safe Adaptive first answers with compact evidence, checks whether the answer looks weak or unsupported, and only then falls back to full top-10 context. We measure answer quality with token F1, context quality with nDCG/MRR, and efficiency with real provider token counts from Ollama. The goal is to show that adaptive context can reduce token usage and latency while preserving answer quality.


## 16. The Safest Scientific Claim

Do not claim:

> This always beats top-10.

Claim:

> Early results suggest that adaptive context selection can substantially reduce token usage, and in some runs preserve answer quality close to fixed top-10. Larger runs across more datasets and models are needed to validate generalization.

That is honest and defensible.

