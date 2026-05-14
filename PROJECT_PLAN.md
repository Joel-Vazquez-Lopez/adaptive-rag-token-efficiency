# Project Plan — Adaptive RAG for Token Efficiency

Course: Language Technology / Information Retrieval  
Dataset: SciFact / BEIR  
Model: Mistral 7B through Ollama  

Start date: __  
Submission date: __  
Repository: __  

## Motivation

Most RAG systems send a fixed amount of retrieved context to the LLM. A common baseline is to retrieve the top-10 documents and place all of them in the prompt.

This is simple, but it can waste tokens and increase latency.

Our question:

Can an adaptive context controller reduce token usage while preserving answer quality?

## Research Questions

1. Does adaptive context selection reduce prompt token usage compared with fixed top-k retrieval?
2. Does token reduction hurt answer quality, measured with Token F1?
3. Does a safety fallback help recover quality on harder queries?
4. How often does the Safe Adaptive model need to expand to full context?

## Experimental Configurations

| Method | Description |
|---|---|
| No Retrieval | Closed-book baseline; Mistral answers without retrieved documents |
| Fixed Top-3 | Always pass 3 full documents |
| Fixed Top-5 | Always pass 5 full documents |
| Fixed Top-10 | Always pass 10 full documents; expensive baseline |
| Heuristic Adaptive | Planned teammate method; rule-based context controller |
| Basic Adaptive + Compact Evidence | Predict context budget, then pass compact evidence spans |
| Safe Adaptive Context | First answer with compact adaptive evidence; if answer looks weak, expand to full top-10 |

## Team Method Structure

The project can be divided by method ownership:

| Owner | Method | Role |
|---|---|---|
| Shared / baseline | No Retrieval | Measures what the model can answer without RAG |
| Fixed-k teammate | Fixed Top-3 / Top-5 / Top-10 | Standard RAG baselines |
| Heuristic teammate | Heuristic Adaptive | Simple rule-based adaptive controller |
| Adaptive system | Basic Adaptive + Compact Evidence | Learned budget plus evidence compression |
| My part | Safe Adaptive Context | Answer-aware fallback for safer token savings |

This structure creates a clear progression:

1. no context
2. fixed context
3. heuristic adaptive context
4. learned adaptive context
5. safe adaptive context

The current GitHub version already includes enough baselines to test Safe
Adaptive end-to-end. The heuristic adaptive method can be integrated later as a
teammate-owned module.

Example heuristic controller:

```text
if top retrieval score is very low:
    k = 0
elif query length is short:
    k = 3
elif top document score is clearly dominant:
    k = 5
else:
    k = 8 or 10
```

## Full-Corpus Baseline Discussion

We considered a baseline that does not use top-k and instead sends the whole
corpus to the LLM.

This is theoretically interesting, but not practical for the main experiment:

- SciFact contains 5,183 abstracts.
- The complete corpus is too large for the local Mistral context window.
- It would make token usage and latency explode.
- Most deployed RAG systems do not send an entire corpus to the model.

So the realistic "large context" baseline is:

```text
Fixed Top-10
```

This gives the model a lot of retrieved evidence while staying runnable and
comparable across methods.

In the report, we can explain:

> Full-corpus prompting is the theoretical maximum-context baseline, but fixed
> top-10 is the practical full-context RAG baseline used in our experiments.

## Our Main Model: Safe Adaptive Context

Safe Adaptive Context is the model we are responsible for.

It is not just a fixed-k system. It is a two-stage controller:

1. First pass:
   - choose an adaptive budget
   - compress selected documents into phrase-aware evidence spans
   - generate an answer
2. Safety check:
   - check if the answer is empty, uncertain, too short, or poorly grounded in evidence
3. Fallback:
   - if risky, regenerate using full top-10 context
   - otherwise keep the compact answer

This means the system can save tokens on easier queries while protecting quality on harder queries.

## Evidence Compression

The compact context mode is called:

```text
evidence_ngram_neighbors
```

It works by:

1. splitting documents into sentences
2. scoring sentences by query word and phrase overlap
3. keeping the strongest evidence sentence plus neighboring sentences
4. preserving enough local context so the LLM can understand the evidence

## Evaluation Metrics

Answer quality:

- Token F1: token overlap between generated answer and reference text.

Efficiency:

- average prompt tokens
- average completion tokens
- average total tokens
- token reduction compared with Fixed Top-10

Latency:

- generation time in milliseconds
- time reduction compared with Fixed Top-10

Adaptive behavior:

- fallback rate for Safe Adaptive Context

## Dataset

Source: SciFact from BEIR.

Corpus:

- 5,183 biomedical abstracts.

Evaluation:

- fixed 150-query sample
- seed: `0`

The runner also accepts `--seed 0` for reproducibility. Most of the pipeline is
deterministic already, but the seed makes this explicit.

## Main Evaluation Plan

The main course dataset is SciFact, but the stronger research version should
test whether the method generalizes across datasets and models.

### Datasets

We plan to evaluate on three datasets:

| Dataset | Domain | Reason |
|---|---|---|
| SciFact | scientific / biomedical claims | main course dataset and first proof-of-concept |
| NFCorpus | biomedical / nutrition | more diverse and less low-entropy than SciFact |
| FiQA | finance | tests whether the system works outside science/biomedicine |

Optional fourth dataset if time allows:

| Dataset | Domain | Reason |
|---|---|---|
| TREC-COVID | scientific / medical COVID literature | another scientific retrieval setting and useful stress test |

The purpose of multiple datasets is:

1. Check whether Safe Adaptive still saves tokens.
2. Check whether answer F1 remains close to fixed top-10.
3. Check whether fallback rate changes when the dataset is harder.
4. Show that the method is not only tuned to SciFact.

### Models

We plan to evaluate with two model types:

| Model | Purpose |
|---|---|
| Mistral 7B through Ollama | local, free, reproducible development model |
| Stronger BergetAI model | hosted confirmation model to test whether the trend holds on a stronger LLM |

Good BergetAI candidates:

| Model | Why |
|---|---|
| `google/gemma-4-31b-it` | good cost/performance choice |
| `meta-llama/Llama-3.3-70B-Instruct` | stronger final confirmation if budget allows |

### Final Comparison Table

The ideal final table should use this shape:

| Dataset | Model | Method | F1 | F1 retained vs Top-10 | Tokens | Token reduction | Time | Time reduction | Fallback rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

This table directly shows the quality/cost tradeoff.

### Related-Method Comparison

The report should compare our method conceptually against related adaptive
RAG/compression work:

- fixed top-k retrieval
- adaptive retrieval / adaptive RAG
- context compression
- answer-aware/self-reflective fallback methods

Our angle is the combination:

> adaptive budget prediction + compact evidence construction + answer-aware fallback + real token/latency measurement.

We should be careful not to claim that each individual component is new. The
contribution is the simple combined controller and the empirical comparison.

## Must-Ship Version

The final version should include:

- working SciFact pipeline
- TF-IDF retriever
- fixed baselines
- Basic Adaptive model
- Safe Adaptive model
- Ollama/Mistral generation
- real provider token counts from Ollama
- one clean results table
- clear README and comments

## Stronger Research Version

If we have time and budget, the stronger version should include:

- SciFact, NFCorpus, and FiQA
- Mistral through Ollama
- one stronger BergetAI model
- 50-query pilot runs for each dataset/model pair
- 150-query final runs where cost/time allows
- final combined table across datasets and models
- short failure analysis of where Safe Adaptive loses quality

## Code Map

| File | Purpose |
|---|---|
| `scripts/run_experiment.py` | Runs the full experiment and creates final tables |
| `src/adaptive_retrieval/data.py` | Loads documents and queries |
| `src/adaptive_retrieval/text.py` | Tokenization, TF-IDF, cosine similarity, token estimates |
| `src/adaptive_retrieval/retriever.py` | Retrieves top documents |
| `src/adaptive_retrieval/budget_experiment.py` | Computes retrieval-side metrics used by the budget model |
| `src/adaptive_retrieval/learned_budget.py` | Basic adaptive budget model |
| `src/adaptive_retrieval/llm_budget.py` | Main LLM pipeline and Safe Adaptive Context |
| `src/adaptive_retrieval/metrics.py` | Answer and retrieval metrics |
| `docs/CODE_WALKTHROUGH.md` | Plain-language explanation of how the code works |

The most important implementation for my part is:

```text
src/adaptive_retrieval/llm_budget.py
answer_aware_fallback_run(...)
```

## Stretch Work

If time allows:

- run the same model on more queries
- compare with another dataset
- add semantic evaluation alongside Token F1
- test a stronger hosted LLM
