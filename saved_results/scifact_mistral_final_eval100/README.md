# SciFact Mistral Final Evaluation Run

This folder contains the final SciFact evaluation for the local Mistral model through Ollama.

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
- Run method: single full run over the held-out evaluation split
- Seed: 0
- Max output tokens: 220
- Token source: provider-reported usage

## Main result

Safe Adaptive Context substantially reduces token usage while improving answer quality over Fixed Top-10.

From `final_table.csv`:

- Fixed Top-10 total tokens: 3858.31
- Safe Adaptive total tokens: 1113.72
- Token reduction: 71.1%
- Fixed Top-10 Answer F1: 0.155448
- Safe Adaptive Answer F1: 0.208471
- Fixed Top-10 semantic similarity: 0.391760
- Safe Adaptive semantic similarity: 0.466747
- Safe Adaptive fallback rate: 7.0%

Compared with the heuristic baseline, Safe Adaptive has similar answer quality while using far fewer tokens:

- Heuristic total tokens: 2700.78
- Safe Adaptive total tokens: 1113.72
- Safe Adaptive uses about 58.8% fewer tokens than Heuristic.
- Heuristic Answer F1: 0.213437
- Safe Adaptive Answer F1: 0.208471
- Heuristic semantic similarity: 0.459969
- Safe Adaptive semantic similarity: 0.466747

## Files

- `final_table.csv`: final report table in CSV format
- `final_table.md`: final report table in Markdown format
- `llm_summary.csv`: averaged metrics per method
- `retrieval_summary.csv`: retrieval metrics per method
- `llm_answers_by_query.csv`: detailed per-query answers and metrics
