# Combined Llama-70B Human Annotation Workform

This folder contains the combined blinded annotation workform for the final Llama-70B experiments.

## Annotation design

The workform samples answers from three datasets:

- SciFact: 13 queries x 3 systems = 39 answers
- HotpotQA: 13 queries x 3 systems = 39 answers
- BioASQ: 8 queries x 3 systems = 24 answers

Total: 102 answers.

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
