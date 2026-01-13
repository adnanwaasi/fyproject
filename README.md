# fyproject

`initial.py` generates Python code and saves it locally. It can optionally re-generate and re-run the generated file a few times until it executes successfully.

## Setup

```bash
pip install -r requirements.txt
```

You also need Ollama running locally and an available model (default: `qwen3:14b`).

## Run

Interactive prompt (asks what to generate):

```bash
python initial.py
```

Generate + run + auto-fix loop (retries on failure):

```bash
python initial.py --loop --max-attempts 5 --run-timeout 15
```

## Outputs

- Writes the generated solution into `real/` (or `--out-dir`).
- Prints the parsed structured output as JSON.
