# SciFact Llama-70B Final Evaluation Run

This folder contains the merged final SciFact evaluation for the hosted Llama model.

## Run setup

- Dataset: SciFact
- Prepared query file: `data/scifact/queries_150_seed0_llm_gold_v2.jsonl`
- Documents: `data/scifact/documents.jsonl`
- Total prepared queries: 150
- Development/calibration split: 50 queries
- Held-out evaluation split: 100 queries
- Evaluated model: `meta-llama/Llama-3.3-70B-Instruct`
- Provider: Berget AI OpenAI-compatible API
- Run method: 10 batches of 10 held-out evaluation queries, merged at per-query level
- Max output tokens: 80
- Token source: provider-reported usage

## Main result

Safe Adaptive Context keeps almost the same answer quality as Fixed Top-10 while using far fewer tokens.

From `final_table.csv`:

- Fixed Top-10 total tokens: 3411.76
- Safe Adaptive total tokens: 933.48
- Token reduction: 72.6%
- Fixed Top-10 Answer F1: 0.261904
- Safe Adaptive Answer F1: 0.265189
- Fixed Top-10 semantic similarity: 0.56934
- Safe Adaptive semantic similarity: 0.568971
- Safe Adaptive fallback rate: 5.0%

## Files

- `final_table.csv`: final report table in CSV format
- `final_table.md`: final report table in Markdown format
- `llm_summary.csv`: averaged metrics per method
- `llm_answers_by_query.csv`: detailed per-query answers and metrics
