# SciFact LLM-Generated Gold Answers

We created an auxiliary SciFact reference-answer file because SciFact gives claims, gold documents, and relevance labels, but not short natural-language gold answers.

The LLM is not creating new relevance labels. It only rewrites the original SciFact gold evidence into a short reference answer.

Model used:
openai/gpt-oss-120b

Provider:
Berget AI OpenAI-compatible API

Temperature:
0

Input per query:
- SciFact claim
- gold relevant document evidence only

Output per query:
- short reference_answer

Prompt used for every query:

You are creating a short reference answer for evaluation.

Use only the gold evidence below.
Do not add outside knowledge.
Write one concise answer that directly checks the claim.
If the evidence supports the claim, restate the correct claim.
If the evidence contradicts the claim, say that the evidence contradicts it and give the correct meaning.
If the evidence is unclear, write: The evidence is insufficient.

Claim:
[claim]

Gold evidence:
[gold evidence]

Reference answer:

Important:
These LLM-generated reference answers are auxiliary evaluation references. They are not original SciFact labels. They are used only after generation to calculate answer-level metrics.
