"""
FastAPI server for the Email Triage OpenEnv environment.

Exposes the full OpenEnv spec interface plus the required additional endpoints:
  POST /reset     — reset environment, return initial observation
  POST /step      — step through environment, return (obs, reward, done, info)
  GET  /state     — return current env state
  GET  /tasks     — task list + action schema
  POST /grader    — grade a completed episode
  POST /baseline  — run baseline inference on all 3 tasks
  GET  /          — serves the frontend dashboard (index.html)

Port 7860 is used for Hugging Face Spaces compatibility.
"""

from __future__ import annotations
import os
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from env.email_env import EmailTriageEnv
from env.models import EmailAction, EmailObservation, EmailReward, EnvState
from env.tasks import list_tasks, TASKS


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Email Triage OpenEnv",
    description=(
        "An OpenEnv-compliant environment for training and evaluating AI agents "
        "on real-world email triage tasks: classify, prioritize, route, and respond."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static frontend assets
_FRONTEND = Path(__file__).parent.parent / "frontend"
if _FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND)), name="static")

# One global env instance (stateful per session)
_env = EmailTriageEnv()


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "single_label_classification"
    seed: int = 42


class StepResponse(BaseModel):
    observation: Optional[EmailObservation]
    reward: EmailReward
    done: bool
    info: Dict[str, Any]


class GraderRequest(BaseModel):
    """Optional — if empty, grades the current in-progress episode."""
    pass


class BaselineResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_time_seconds: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    """Serve the frontend dashboard."""
    index = _FRONTEND / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {
        "status": "ok",
        "service": "Email Triage OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "Email Triage OpenEnv",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.post("/reset", response_model=EmailObservation, tags=["openenv"])
async def reset(request: Request, body: Optional[ResetRequest] = None):
    """
    Reset the environment for a specific task and seed.
    Returns the first email observation.
    Accepts empty body — defaults to task='single_label_classification', seed=42.
    """
    # Parse body manually so an empty/missing body never raises a validation error
    if body is None:
        raw = await request.body()
        if raw:
            try:
                import json as _json
                data = _json.loads(raw)
                body = ResetRequest(**data)
            except Exception:
                body = ResetRequest()
        else:
            body = ResetRequest()
    try:
        obs = _env.reset(task_id=body.task_id, seed=body.seed)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return obs


@app.post("/step", response_model=StepResponse, tags=["openenv"])
async def step(action: EmailAction):
    """
    Submit an action for the current email.
    Returns next observation (None if done), reward, done flag, and info.
    """
    try:
        obs, reward, done, info = _env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return StepResponse(observation=obs, reward=reward, done=done, info=info)


@app.get("/state", response_model=EnvState, tags=["openenv"])
async def state():
    """Return the full current environment state."""
    return _env.state()


@app.get("/tasks", tags=["openenv"])
async def tasks():
    """
    List all available tasks with descriptions, difficulty, and the action schema
    required for a step() call.
    """
    return {"tasks": list_tasks()}


@app.post("/grader", tags=["openenv"])
async def grader():
    """
    Grade the current (or completed) episode.
    Returns a score strictly in (0, 1) exclusive, and a detailed breakdown.
    """
    score, details = _env.grade()
    # Clamp strictly to (0, 1) exclusive as required by the validator
    score = max(0.001, min(0.999, float(score)))
    return {"score": round(score, 4), "details": details}


@app.post("/baseline", response_model=BaselineResponse, tags=["openenv"])
async def baseline():
    """
    Run the baseline inference script against all 3 tasks using the OpenAI API.
    Requires the OPENAI_API_KEY environment variable to be set.
    Returns reproducible scores for all tasks.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY environment variable is not set."
        )

    start = time.time()
    results = await _run_baseline_all_tasks(api_key)
    elapsed = round(time.time() - start, 2)
    return BaselineResponse(results=results, total_time_seconds=elapsed)


# ---------------------------------------------------------------------------
# Baseline logic (async, uses openai)
# ---------------------------------------------------------------------------

async def _run_baseline_all_tasks(api_key: str) -> List[Dict[str, Any]]:
    """Run the baseline agent on all 3 tasks and return per-task results."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise HTTPException(status_code=500, detail="openai package not installed.")

    client = AsyncOpenAI(api_key=api_key)
    results = []

    for task_id in TASKS:
        result = await _run_baseline_task(client, task_id, seed=42)
        results.append(result)

    return results


async def _run_baseline_task(client: Any, task_id: str, seed: int) -> Dict[str, Any]:
    """Run a single task with the GPT-4o-mini baseline agent."""
    env = EmailTriageEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    steps_taken = 0

    while obs is not None:
        action = await _query_llm(client, obs, task_id)
        obs, reward, done, info = env.step(action)
        steps_taken += 1
        if done:
            break

    score, details = env.grade()
    return {
        "task_id": task_id,
        "seed": seed,
        "score": score,
        "steps": steps_taken,
        "cumulative_reward": env._cumulative_reward,
        "details": details,
    }


async def _query_llm(client: Any, obs: EmailObservation, task_id: str) -> EmailAction:
    """Ask GPT-4o-mini to classify a single email. Falls back to heuristics on error."""
    system_prompt = (
        "You are an expert email triage assistant. "
        "Given an email, you must output a JSON object with these exact fields:\n"
        '  "email_id": string (copy from input),\n'
        '  "label": one of [work, personal, spam, newsletter, urgent, support],\n'
        '  "priority": one of [low, medium, high, critical],\n'
        '  "route_to": one of [engineering, sales, support, hr, finance, ignore],\n'
        '  "draft_reply": string or null (required if label is urgent or support and '
        'task is inbox_zero_with_sla, otherwise null),\n'
        '  "archive": boolean (true if spam or newsletter)\n'
        "Output ONLY the JSON object, no explanation."
    )
    user_prompt = (
        f"Task: {task_id}\n"
        f"Email ID: {obs.email_id}\n"
        f"From: {obs.sender} <{obs.sender_domain}>\n"
        f"Subject: {obs.subject}\n"
        f"Body:\n{obs.body[:400]}\n"
        f"Has attachments: {obs.has_attachments}\n"
        f"Thread length: {obs.thread_length}"
    )

    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        return EmailAction(
            email_id=obs.email_id,  # override to ensure correctness
            label=data.get("label", "work"),
            priority=data.get("priority", "medium"),
            route_to=data.get("route_to", "ignore"),
            draft_reply=data.get("draft_reply"),
            archive=bool(data.get("archive", False)),
        )
    except Exception:
        # Safe fallback
        return EmailAction(
            email_id=obs.email_id,
            label="work",
            priority="medium",
            route_to="ignore",
            draft_reply=None,
            archive=False,
        )


def main() -> None:
    """Entry point for multi-mode deployment."""
    import uvicorn
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
    )


if __name__ == "__main__":
    main()
