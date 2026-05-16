# fyproject

AI-powered code synthesis pipeline that generates, tests, and self-repairs Python programs from natural language descriptions.

## How It Works

Given a natural language prompt (e.g., *"generate fibonacci series for n terms"*), the pipeline:

1. **Prompt Processing** — Extracts structured problem specifications (inputs, outputs, constraints, edge cases)
2. **Code Generation** — Produces Python code via Ollama (`gemma4:e4b`)
3. **Test Generation** — Automatically creates test cases from the spec and generated code
4. **Test Execution** — Runs tests against the generated code and collects pass/fail results
5. **Error Analysis** — Diagnoses root causes of failing tests using the LLM
6. **Self-Repair** — Rewrites code to fix identified issues, with rollback if quality degrades

The cycle repeats until all tests pass or the acceptance threshold (≥85%) is met.

## Setup

```bash
pip install -r requirements.txt
```

Requires Ollama running locally with model `gemma4:e4b`.

## Run

### CLI Pipeline

```bash
python pipeline.py "generate fibonacci series for n terms"
```

### FastAPI Backend

```bash
python -m uvicorn backend.app:app --reload --port 8000
```

#### Endpoints

- `POST /api/generate` - Async job submission
- `POST /api/generate/sync` - Synchronous generation
- `POST /api/generate/stream` - SSE streaming
- `GET /api/jobs/{job_id}` - Job status
- `GET /api/health` - Health check

## Configuration

- Default repair iterations: 2
- Acceptance threshold: 85% (≥85% tests pass = accepted)
- Output directory: `real/`

## Environment Variables

- `CODE_SYNTH_API_KEY` - Optional API key for protected endpoints
- `CODE_SYNTH_OUTPUT_BASE` - Base directory for generated code (default: `real`)
