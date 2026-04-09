# fyproject

Code synthesis pipeline using Ollama (gemma4:e4b) with automatic repair and test execution.

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
