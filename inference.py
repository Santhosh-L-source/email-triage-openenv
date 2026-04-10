#!/usr/bin/env python3
"""
inference.py -- Mandatory submission inference script for Email Triage OpenEnv.

Environment Variables Required:
    API_BASE_URL   The API endpoint base URL for the LLM provider (OpenAI-compatible).
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.

Structured stdout log format (strictly required for evaluation):
    [START] {"task_id": ..., "model": ..., "seed": ...}
    [STEP]  {"task_id": ..., "step": ..., "email_id": ..., "reward": ..., "done": ...}
    [END]   {"task_id": ..., "score": ..., "steps": ..., "elapsed_seconds": ...}

Usage:
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=sk-...
    python inference.py [--url http://localhost:7860]
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests

# ---------------------------------------------------------------------------
# Configuration — read from environment variables (mandatory)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME: str   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN: str     = os.environ.get("HF_TOKEN",     "")

ENV_BASE_URL: str = "http://localhost:7860"
SEED: int         = 42

TASK_IDS = [
    "single_label_classification",
    "priority_triage_with_routing",
    "inbox_zero_with_sla",
]

SYSTEM_PROMPT = (
    "You are an expert email triage assistant. "
    "Given an email, you must output a JSON object with these exact fields:\n"
    '  "label": one of [work, personal, spam, newsletter, urgent, support],\n'
    '  "priority": one of [low, medium, high, critical],\n'
    '  "route_to": one of [engineering, sales, support, hr, finance, ignore],\n'
    '  "draft_reply": string or null (required if label is urgent or support and '
    'task is inbox_zero_with_sla, otherwise null),\n'
    '  "archive": boolean (true if spam or newsletter)\n'
    "Output ONLY the JSON object, no explanation."
)

VALID_LABELS   = {"work", "personal", "spam", "newsletter", "urgent", "support"}
VALID_PRIORITY = {"low", "medium", "high", "critical"}
VALID_ROUTES   = {"engineering", "sales", "support", "hr", "finance", "ignore"}


# ---------------------------------------------------------------------------
# Structured logging helpers — MUST follow [START] / [STEP] / [END] format
# ---------------------------------------------------------------------------

def log_start(task_id: str, model: str, seed: int) -> None:
    """Emit [START] log line to stdout."""
    payload = json.dumps({"task_id": task_id, "model": model, "seed": seed})
    print(f"[START] {payload}", flush=True)


def log_step(
    task_id: str,
    step: int,
    email_id: str,
    reward: float,
    cumulative_reward: float,
    done: bool,
) -> None:
    """Emit [STEP] log line to stdout."""
    payload = json.dumps({
        "task_id": task_id,
        "step": step,
        "email_id": email_id,
        "reward": round(reward, 4),
        "cumulative_reward": round(cumulative_reward, 4),
        "done": done,
    })
    print(f"[STEP] {payload}", flush=True)


def log_end(task_id: str, score: float, steps: int, elapsed_seconds: float) -> None:
    """Emit [END] log line to stdout."""
    payload = json.dumps({
        "task_id": task_id,
        "score": round(score, 4),
        "steps": steps,
        "elapsed_seconds": round(elapsed_seconds, 2),
    })
    print(f"[END] {payload}", flush=True)


# ---------------------------------------------------------------------------
# LLM interaction (OpenAI-compatible client)
# ---------------------------------------------------------------------------

def build_user_prompt(obs: Dict[str, Any], task_id: str) -> str:
    return (
        f"Task: {task_id}\n"
        f"Email ID: {obs['email_id']}\n"
        f"From: {obs['sender']} <{obs['sender_domain']}>\n"
        f"Subject: {obs['subject']}\n"
        f"Body:\n{obs['body'][:400]}\n"
        f"Has attachments: {obs['has_attachments']}\n"
        f"Thread length: {obs['thread_length']}"
    )


def query_llm(client: Any, obs: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """Ask the LLM to triage a single email. Returns action dict."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(obs, task_id)},
        ],
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)

    return {
        "email_id":    obs["email_id"],
        "label":       data.get("label",    "work")   if data.get("label")    in VALID_LABELS   else "work",
        "priority":    data.get("priority", "medium") if data.get("priority") in VALID_PRIORITY else "medium",
        "route_to":    data.get("route_to", "ignore") if data.get("route_to") in VALID_ROUTES   else "ignore",
        "draft_reply": data.get("draft_reply"),
        "archive":     bool(data.get("archive", False)),
    }


# ---------------------------------------------------------------------------
# Task runner
# ---------------------------------------------------------------------------

def run_task(task_id: str, base_url: str, client: Any) -> Dict[str, Any]:
    """Run a full episode for one task; emit structured logs; return results."""

    log_start(task_id=task_id, model=MODEL_NAME, seed=SEED)
    t0 = time.time()

    # Reset environment
    r = requests.post(
        f"{base_url}/reset",
        json={"task_id": task_id, "seed": SEED},
        timeout=30,
    )
    r.raise_for_status()
    obs = r.json()

    step = 0

    while obs is not None:
        # Build action
        try:
            action = query_llm(client, obs, task_id)
        except Exception as exc:
            print(f"[WARN] LLM error at step {step}: {exc} — using fallback action", flush=True)
            action = {
                "email_id":    obs["email_id"],
                "label":       "work",
                "priority":    "medium",
                "route_to":    "ignore",
                "draft_reply": None,
                "archive":     False,
            }

        # Step environment
        r = requests.post(f"{base_url}/step", json=action, timeout=30)
        r.raise_for_status()
        resp    = r.json()

        next_obs    = resp["observation"]
        reward_info = resp["reward"]
        done        = resp["done"]
        step       += 1

        # Emit [STEP] log
        log_step(
            task_id=task_id,
            step=step,
            email_id=action["email_id"],
            reward=reward_info["step_reward"],
            cumulative_reward=reward_info["cumulative_reward"],
            done=done,
        )

        obs = next_obs
        if done:
            break

    # Grade episode
    r = requests.post(f"{base_url}/grader", timeout=30)
    r.raise_for_status()
    grade_resp = r.json()

    elapsed = time.time() - t0
    score   = grade_resp["score"]

    # Emit [END] log
    log_end(task_id=task_id, score=score, steps=step, elapsed_seconds=elapsed)

    return {
        "task_id":         task_id,
        "score":           score,
        "steps":           step,
        "elapsed_seconds": round(elapsed, 2),
        "details":         grade_resp.get("details", {}),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv inference script")
    parser.add_argument(
        "--url",
        default=ENV_BASE_URL,
        help="Base URL of the OpenEnv server (default: http://localhost:7860)",
    )
    args = parser.parse_args()

    # Validate required env vars
    missing = []
    if not HF_TOKEN:
        missing.append("HF_TOKEN")
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if missing:
        print(
            f"ERROR: Missing required environment variables: {', '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Initialise OpenAI-compatible client
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(
        api_key=HF_TOKEN,          # HF_TOKEN is the API key
        base_url=API_BASE_URL,     # API_BASE_URL is the provider base URL
    )

    base_url = args.url.rstrip("/")

    # Verify server is reachable
    try:
        r = requests.get(f"{base_url}/", timeout=10)
        r.raise_for_status()
        print(f"[INFO] Server reachable at {base_url}", flush=True)
    except Exception as exc:
        print(f"ERROR: Cannot reach OpenEnv server at {base_url}: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] model={MODEL_NAME}  api_base={API_BASE_URL}", flush=True)
    print(f"[INFO] Running {len(TASK_IDS)} tasks with seed={SEED}", flush=True)

    results = []
    for task_id in TASK_IDS:
        result = run_task(task_id, base_url, client)
        results.append(result)

    # Summary table
    print("\n" + "━" * 62, flush=True)
    print(f"  {'Task':<38} {'Score':>7}  {'Steps':>5}", flush=True)
    print("  " + "-" * 58, flush=True)
    for res in results:
        print(f"  {res['task_id']:<38} {res['score']:>7.4f}  {res['steps']:>5}", flush=True)
    avg = sum(r["score"] for r in results) / len(results)
    print("  " + "-" * 58, flush=True)
    print(f"  {'AVERAGE':<38} {avg:>7.4f}", flush=True)
    print("━" * 62 + "\n", flush=True)

    # Persist machine-readable results
    out_path = "inference_results.json"
    with open(out_path, "w") as f:
        json.dump({"results": results, "average_score": round(avg, 4)}, f, indent=2)
    print(f"[INFO] Results saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
