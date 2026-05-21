# HotpotQA Setup

HotpotQA is our recommended second dataset.

Why:

- it has real questions
- it has real gold answers
- it has supporting evidence paragraphs
- it tests whether adaptive context works beyond SciFact

This helps with one SciFact limitation:

SciFact does not provide short gold answers, so our SciFact F1 is really
evidence-overlap F1. HotpotQA has a real `answer` field, so answer F1 is easier
to explain.

## Dataset Shape

HotpotQA examples normally contain:

```text
question
answer
context
supporting_facts
```

In our converter:

- `question` becomes the query text
- each context paragraph becomes a retrievable document
- supporting fact titles become relevant document ids
- `answer` becomes the reference answer

The converted files are:

```text
data/hotpotqa/documents.jsonl
data/hotpotqa/queries.jsonl
```

Then the normal experiment script can run without changing the model code.

## Prepare HotpotQA

If you want to load directly from HuggingFace:

```bash
cd /Users/joelvazquezlopez/Desktop/adaptive_rag_github

python3 -m pip install datasets

python3 scripts/prepare_hotpotqa_simple.py \
  --from-huggingface \
  --split validation \
  --output-dir data/hotpotqa \
  --max-queries 150
```

If you already downloaded a HotpotQA JSON or JSONL file:

```bash
cd /Users/joelvazquezlopez/Desktop/adaptive_rag_github

python3 scripts/prepare_hotpotqa_simple.py \
  --input-file /path/to/hotpotqa.json \
  --output-dir data/hotpotqa \
  --max-queries 150
```

## Small Dry Run

This checks the pipeline without calling an LLM:

```bash
cd /Users/joelvazquezlopez/Desktop/adaptive_rag_github

python3 scripts/run_experiment.py \
  --documents data/hotpotqa/documents.jsonl \
  --queries data/hotpotqa/queries.jsonl \
  --dataset-name hotpotqa \
  --max-eval-queries 5 \
  --output-dir outputs/hotpotqa_dry_5 \
  --dry-run
```

## Small Ollama Run

This checks the real local model:

```bash
cd /Users/joelvazquezlopez/Desktop/adaptive_rag_github

ollama run mistral "test"

caffeinate -dimsu python3 scripts/run_experiment.py \
  --documents data/hotpotqa/documents.jsonl \
  --queries data/hotpotqa/queries.jsonl \
  --dataset-name hotpotqa \
  --model mistral \
  --api-url http://localhost:11434/v1 \
  --no-api-key \
  --max-eval-queries 10 \
  --output-dir outputs/hotpotqa_mistral_10 \
  --request-timeout-seconds 300 \
  --require-provider-tokens
```

## Strong Model Check

This checks Berget Llama 70B:

```bash
cd /Users/joelvazquezlopez/Desktop/adaptive_rag_github

export OPENAI_API_KEY="YOUR_REAL_BERGET_KEY"

python3 scripts/run_experiment.py \
  --documents data/hotpotqa/documents.jsonl \
  --queries data/hotpotqa/queries.jsonl \
  --dataset-name hotpotqa \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --api-url https://api.berget.ai/v1 \
  --max-eval-queries 10 \
  --output-dir outputs/hotpotqa_llama70b_10 \
  --request-timeout-seconds 300 \
  --require-provider-tokens
```

## How To Read The Result

The main comparison is:

```text
Fixed Top-10
vs
Safe Adaptive Context
```

The useful result would be:

```text
Safe Adaptive keeps similar answer F1
Safe Adaptive uses fewer total tokens
token_source = provider
```

If this works on HotpotQA, it is stronger than SciFact alone because HotpotQA
has real answers.
