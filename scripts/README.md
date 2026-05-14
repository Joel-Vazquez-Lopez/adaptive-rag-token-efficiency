# Scripts

Run the experiment from the project root:

```bash
python3 scripts/run_experiment.py --dry-run --max-queries 5 --output-dir outputs/dry_run
```

Use Ollama/Mistral:

```bash
python3 scripts/run_experiment.py \
  --model mistral \
  --api-url http://localhost:11434/v1 \
  --max-queries 50 \
  --require-provider-tokens \
  --output-dir outputs/scifact_mistral_50
```

For final results, keep `--require-provider-tokens`. This makes sure token
counts come from the model/API response instead of local approximation.

The output file to use in the report is:

```text
outputs/scifact_mistral_50/final_table.csv
```
