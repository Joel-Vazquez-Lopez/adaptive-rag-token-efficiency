# SciFact Mistral Final Evaluation Run

This folder contains the merged final SciFact evaluation for the local Mistral model through Ollama.

This run uses the same max output token setting as the hosted Llama SciFact evaluation.

## Run setup

- Dataset: SciFact
- Prepared query file: `data/scifact/queries_150_seed0_llm_gold_v2.jsonl`
- Documents: `data/scifact/documents.jsonl`
- Total prepared queries: 150
- Development/calibration split: 50 queries
- Held-out evaluation split: 100 queries
- Evaluated model: `mistral`
- Provider: local Ollama OpenAI-compatible API
- API URL: `http://localhost:11434/v1`
- Run method: 10 batches of 10 held-out evaluation queries, merged at per-query level
- Max output tokens: 80
- Seed: 0
- Token source: provider-reported usage

## Files

- `final_table.csv`: final report table in CSV format
- `final_table.md`: final report table in Markdown format
- `llm_summary.csv`: averaged metrics per method
- `llm_answers_by_query.csv`: detailed per-query answers and metrics
