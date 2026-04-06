"""FastAPI Backend for Code Synthesis Pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import traceback
import uuid
from collections import defaultdict
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import run_pipeline, PipelineConfig, PipelineResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("code_synthesis")

SERVER_START_TIME = datetime.now()

app = FastAPI(
    title="Code Synthesis API",
    description="AI-powered code generation and testing pipeline",
    version="2.0.0",
)

# SECURITY: In production, configure allowed origins via environment variable
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Job-ID"],
)

jobs: dict[str, dict] = {}

_rate_limit_store: dict[str, list[float]] = defaultdict(list)
RATE_LIMIT_MAX_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60

PIPELINE_TIMEOUT_SECONDS = 300


@app.middleware("http")
async def add_request_id_header(request: Request, call_next):
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(ValidationError)
async def pydantic_validation_handler(request: Request, exc: ValidationError):
    logger.warning("Validation error: %s", exc.error_count())
    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation Error",
            "detail": exc.errors(),
            "timestamp": datetime.now().isoformat(),
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception: %s\n%s", exc, traceback.format_exc())
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "detail": str(exc),
            "timestamp": datetime.now().isoformat(),
        },
    )


async def rate_limit(request: Request):
    """Simple in-memory rate limiter: max requests per minute per client IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    _rate_limit_store[client_ip] = [
        ts for ts in _rate_limit_store[client_ip] if ts > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_MAX_REQUESTS:
        logger.warning("Rate limit exceeded for %s", client_ip)
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Max 10 generation requests per minute.",
        )

    _rate_limit_store[client_ip].append(now)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    max_iterations: int = Field(default=1, ge=1, le=10)
    model: str = Field(default="gemma4:e4b", min_length=1)
    output_dir: str = Field(default="real", min_length=1)


class JobStatus(BaseModel):
    job_id: str
    status: str
    created_at: str
    completed_at: Optional[str] = None
    prompt: str
    current_step: Optional[str] = None
    duration_seconds: Optional[float] = None
    result: Optional[dict] = None
    error: Optional[str] = None


class PaginatedJobs(BaseModel):
    jobs: list[JobStatus]
    total: int
    limit: int
    offset: int


class TestResultResponse(BaseModel):
    test_id: str
    passed: bool
    expected: str
    actual: str
    error: Optional[str] = None


def serialize_pipeline_result(result: PipelineResult) -> dict:
    test_results = []
    if result.test_results:
        for tr in result.test_results:
            test_results.append(
                {
                    "test_id": tr.test_id,
                    "passed": tr.passed,
                    "expected": str(tr.expected_output) if tr.expected_output else "",
                    "actual": str(tr.actual_output) if tr.actual_output else "",
                    "error": tr.error_message,
                }
            )

    return {
        "success": result.success,
        "repair_iterations": result.repair_iterations,
        "final_code": result.final_code,
        "output_file": result.output_file,
        "problem_spec": (
            result.problem_spec.model_dump() if result.problem_spec else None
        ),
        "test_results": test_results,
        "error_analysis": result.error_analysis,
        "test_cases": {
            "count": (len(result.test_cases.test_cases) if result.test_cases else 0),
            "cases": [
                {
                    "test_id": tc.test_id,
                    "description": tc.description,
                    "test_type": tc.test_type,
                    "input_data": tc.input_data,
                    "expected_output": tc.expected_output,
                }
                for tc in (result.test_cases.test_cases if result.test_cases else [])
            ],
        },
    }


def _create_job_record(job_id: str, prompt: str) -> dict:
    return {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "completed_at": None,
        "prompt": prompt,
        "current_step": None,
        "duration_seconds": None,
        "result": None,
        "error": None,
    }


def _compute_duration(job: dict) -> Optional[float]:
    if job.get("created_at") and job.get("completed_at"):
        try:
            start = datetime.fromisoformat(job["created_at"])
            end = datetime.fromisoformat(job["completed_at"])
            return round((end - start).total_seconds(), 2)
        except (ValueError, TypeError):
            return None
    return None


def _job_to_status(job: dict) -> JobStatus:
    duration = job.get("duration_seconds") or _compute_duration(job)
    return JobStatus(
        job_id=job["job_id"],
        status=job["status"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        prompt=job["prompt"],
        current_step=job.get("current_step"),
        duration_seconds=duration,
        result=job.get("result"),
        error=job.get("error"),
    )


async def run_pipeline_async(job_id: str, prompt: str, config: PipelineConfig):
    jobs[job_id]["status"] = "running"
    jobs[job_id]["current_step"] = "starting"
    logger.info("Job %s started | prompt=%.100s", job_id, prompt)

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: run_pipeline(prompt, config)),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["current_step"] = "completed"
        jobs[job_id]["result"] = serialize_pipeline_result(result)
        jobs[job_id]["duration_seconds"] = _compute_duration(jobs[job_id])
        logger.info("Job %s completed successfully", job_id)

    except asyncio.TimeoutError:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["current_step"] = "timeout"
        jobs[job_id]["error"] = (
            f"Pipeline timed out after {PIPELINE_TIMEOUT_SECONDS} seconds"
        )
        jobs[job_id]["duration_seconds"] = _compute_duration(jobs[job_id])
        logger.error("Job %s timed out", job_id)

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["completed_at"] = datetime.now().isoformat()
        jobs[job_id]["current_step"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["duration_seconds"] = _compute_duration(jobs[job_id])
        logger.error("Job %s failed: %s\n%s", job_id, e, traceback.format_exc())


def _sse_event(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


async def _stream_pipeline(job_id: str, prompt: str, config: PipelineConfig):
    queue: asyncio.Queue[Optional[dict]] = asyncio.Queue()
    loop = asyncio.get_event_loop()

    def _run_and_report():
        def queue_put(evt: Optional[dict]):
            asyncio.run_coroutine_threadsafe(queue.put(evt), loop)

        def on_progress(step: str, status: str, message: str, progress: float):
            queue_put(
                {
                    "step": step,
                    "status": status,
                    "message": message,
                    "progress": progress,
                }
            )

        try:
            result = run_pipeline(prompt, config, on_progress=on_progress)

            queue_put(
                {
                    "step": "completed",
                    "status": "completed",
                    "message": "Pipeline completed successfully"
                    if result.success
                    else f"Pipeline finished with {result.repair_iterations} repair iterations",
                    "progress": 1.0,
                    "result": serialize_pipeline_result(result),
                }
            )

        except Exception as e:
            queue_put(
                {
                    "step": "failed",
                    "status": "failed",
                    "message": str(e),
                    "progress": 0.0,
                }
            )

        queue_put(None)  # sentinel: end of stream

    loop.run_in_executor(None, _run_and_report)

    while True:
        try:
            event = await asyncio.wait_for(
                queue.get(), timeout=PIPELINE_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            yield _sse_event(
                {
                    "step": "timeout",
                    "status": "failed",
                    "message": f"Pipeline timed out after {PIPELINE_TIMEOUT_SECONDS}s",
                    "progress": 0.0,
                }
            )
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            jobs[job_id]["error"] = "Pipeline timed out"
            jobs[job_id]["duration_seconds"] = _compute_duration(jobs[job_id])
            break

        if event is None:
            break

        jobs[job_id]["current_step"] = event.get("step", "unknown")
        if event.get("step") == "completed" and event.get("result") is not None:
            jobs[job_id]["status"] = "completed"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            jobs[job_id]["result"] = event.get("result")
            jobs[job_id]["duration_seconds"] = _compute_duration(jobs[job_id])
        elif event.get("step") == "failed":
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["completed_at"] = datetime.now().isoformat()
            jobs[job_id]["error"] = event.get("message")
            jobs[job_id]["duration_seconds"] = _compute_duration(jobs[job_id])

        yield _sse_event(event)


@app.get("/")
async def root():
    return {"message": "Code Synthesis API", "version": "2.0.0"}


@app.post("/api/generate", response_model=JobStatus, dependencies=[Depends(rate_limit)])
async def generate_code(request: GenerateRequest, background_tasks: BackgroundTasks):
    prompt = request.prompt.strip()
    job_id = str(uuid.uuid4())

    config = PipelineConfig(
        output_dir=request.output_dir,
        max_repair_iterations=request.max_iterations,
        model=request.model,
        verbose=True,
    )

    jobs[job_id] = _create_job_record(job_id, prompt)
    logger.info(
        "Job %s queued | model=%s max_iter=%d prompt=%.100s",
        job_id,
        request.model,
        request.max_iterations,
        prompt,
    )

    background_tasks.add_task(run_pipeline_async, job_id, prompt, config)

    return _job_to_status(jobs[job_id])


@app.post("/api/generate/stream", dependencies=[Depends(rate_limit)])
async def generate_code_stream(request: GenerateRequest):
    prompt = request.prompt.strip()
    job_id = str(uuid.uuid4())

    config = PipelineConfig(
        output_dir=request.output_dir,
        max_repair_iterations=request.max_iterations,
        model=request.model,
        verbose=True,
    )

    jobs[job_id] = _create_job_record(job_id, prompt)
    jobs[job_id]["status"] = "running"
    logger.info(
        "Job %s streaming | model=%s prompt=%.100s",
        job_id,
        request.model,
        prompt,
    )

    async def event_generator():
        yield _sse_event(
            {"job_id": job_id, "step": "started", "status": "running", "progress": 0.0}
        )
        async for event in _stream_pipeline(job_id, prompt, config):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Job-ID": job_id,
        },
    )


@app.post("/api/generate/sync", dependencies=[Depends(rate_limit)])
async def generate_code_sync(request: GenerateRequest):
    prompt = request.prompt.strip()
    logger.info("Sync generation | prompt=%.100s", prompt)

    config = PipelineConfig(
        output_dir=request.output_dir,
        max_repair_iterations=request.max_iterations,
        model=request.model,
        verbose=True,
    )

    try:
        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(
            loop.run_in_executor(None, lambda: run_pipeline(prompt, config)),
            timeout=PIPELINE_TIMEOUT_SECONDS,
        )
        return serialize_pipeline_result(result)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"Pipeline timed out after {PIPELINE_TIMEOUT_SECONDS} seconds",
        )
    except Exception as e:
        logger.error("Sync generation failed: %s\n%s", e, traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    return _job_to_status(jobs[job_id])


@app.get("/api/jobs", response_model=PaginatedJobs)
async def list_jobs(
    status: Optional[str] = Query(
        default=None,
        description="Filter by status: pending, running, completed, failed, cancelled",
    ),
    limit: int = Query(default=20, ge=1, le=100, description="Max jobs per page"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    all_jobs = list(jobs.values())

    if status:
        all_jobs = [j for j in all_jobs if j["status"] == status]

    total = len(all_jobs)

    all_jobs.sort(key=lambda j: j["created_at"], reverse=True)

    page = all_jobs[offset : offset + limit]

    return PaginatedJobs(
        jobs=[_job_to_status(j) for j in page],
        total=total,
        limit=limit,
        offset=offset,
    )


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]
    if job["status"] not in ("pending", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Cannot cancel job with status '{job['status']}'",
        )

    job["status"] = "cancelled"
    job["completed_at"] = datetime.now().isoformat()
    job["current_step"] = "cancelled"
    job["duration_seconds"] = _compute_duration(job)
    logger.info("Job %s cancelled", job_id)

    return {"message": "Job cancelled", "job_id": job_id}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    del jobs[job_id]
    logger.info("Job %s deleted", job_id)
    return {"message": "Job deleted"}


@app.get("/api/health")
async def health_check():
    active = sum(1 for j in jobs.values() if j["status"] in ("pending", "running"))
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "startup_time": SERVER_START_TIME.isoformat(),
        "active_jobs": active,
    }


@app.get("/api/metrics")
async def metrics():
    all_jobs = list(jobs.values())
    total = len(all_jobs)
    completed = [j for j in all_jobs if j["status"] == "completed"]
    failed = [j for j in all_jobs if j["status"] == "failed"]

    durations = [d for j in completed if (d := _compute_duration(j)) is not None]
    avg_duration = round(sum(durations) / len(durations), 2) if durations else 0.0

    return {
        "total_jobs": total,
        "completed_jobs": len(completed),
        "failed_jobs": len(failed),
        "cancelled_jobs": sum(1 for j in all_jobs if j["status"] == "cancelled"),
        "active_jobs": sum(
            1 for j in all_jobs if j["status"] in ("pending", "running")
        ),
        "average_duration_seconds": avg_duration,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
