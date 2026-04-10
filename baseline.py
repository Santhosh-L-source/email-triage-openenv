#!/usr/bin/env python3
"""
Baseline inference script for the Email Triage OpenEnv environment.

Uses the OpenAI API (GPT-4o-mini) to run a zero-shot agent against all 3 tasks.
Reads API credentials from the OPENAI_API_KEY environment variable.
Uses fixed seed=42 for full reproducibility.

Usage:
    export OPENAI_API_KEY=sk-...
    python baseline.py [--url http://localhost:7860]

Output:
    JSON table with score for each task printed to stdout.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import requests


BASE_URL = "http://localhost:7860"
SEED = 42
MODEL = "gpt-4o-mini"

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
    """Ask GPT to triage a single email. Returns action dict."""
    response = client.chat.completions.create(
        model=MODEL,
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

    # Build valid action
    VALID_LABELS    = {"work", "personal", "spam", "newsletter", "urgent", "support"}
    VALID_PRIORITY  = {"low", "medium", "high", "critical"}
    VALID_ROUTES    = {"engineering", "sales", "support", "hr", "finance", "ignore"}

    return {
        "email_id":    obs["email_id"],
        "label":       data.get("label",    "work")   if data.get("label")    in VALID_LABELS   else "work",
        "priority":    data.get("priority", "medium") if data.get("priority") in VALID_PRIORITY else "medium",
        "route_to":    data.get("route_to", "ignore") if data.get("route_to") in VALID_ROUTES   else "ignore",
        "draft_reply": data.get("draft_reply"),
        "archive":     bool(data.get("archive", False)),
    }


def run_task(task_id: str, base_url: str, client: Any) -> Dict[str, Any]:
    """Run a full episode for one task; return score and stats."""
    print(f"\n  Running task: {task_id}")

    # Reset
    r = requests.post(f"{base_url}/reset", json={"task_id": task_id, "seed": SEED}, timeout=30)
    r.raise_for_status()
    obs = r.json()

    steps = 0
    t0 = time.time()

    while obs is not None:
        try:
            action = query_llm(client, obs, task_id)
        except Exception as e:
            # Safe fallback on LLM error
            print(f"    [LLM error at step {steps}]: {e} — using fallback action")
            action = {
                "email_id": obs["email_id"],
                "label": "work",
                "priority": "medium",
                "route_to": "ignore",
                "draft_reply": None,
                "archive": False,
            }

        r = requests.post(f"{base_url}/step", json=action, timeout=30)
        r.raise_for_status()
        resp = r.json()

        obs = resp["observation"]
        done = resp["done"]
        steps += 1

        if steps % 5 == 0:
            print(f"    Step {steps:02d} | cumulative_reward={resp['reward']['cumulative_reward']:.3f}")

        if done:
            break

    # Grade
    r = requests.post(f"{base_url}/grader", timeout=30)
    r.raise_for_status()
    grade_resp = r.json()

    elapsed = round(time.time() - t0, 2)
    score = grade_resp["score"]
    print(f"  ✓ Score: {score:.4f} | Steps: {steps} | Time: {elapsed}s")
    return {
        "task_id": task_id,
        "score": score,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "details": grade_resp["details"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv baseline")
    parser.add_argument("--url", default=BASE_URL, help="Base URL of the env server")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    base_url = args.url.rstrip("/")

    # Verify server is up
    try:
        r = requests.get(f"{base_url}/", timeout=10)
        r.raise_for_status()
        print(f"✓ Server is up at {base_url}")
    except Exception as e:
        print(f"ERROR: Cannot reach server at {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    task_ids = ["single_label_classification", "priority_triage_with_routing", "inbox_zero_with_sla"]
    results = []

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Email Triage OpenEnv — Baseline Inference (GPT-4o-mini)")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    for task_id in task_ids:
        result = run_task(task_id, base_url, client)
        results.append(result)

    # Print summary table
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  {'Task':<38} {'Score':>7}  {'Difficulty'}")
    print("  " + "-" * 56)
    difficulties = {"single_label_classification": "easy", "priority_triage_with_routing": "medium", "inbox_zero_with_sla": "hard"}
    for r in results:
        diff = difficulties.get(r["task_id"], "?")
        print(f"  {r['task_id']:<38} {r['score']:>7.4f}  {diff}")
    avg = sum(r["score"] for r in results) / len(results)
    print("  " + "-" * 56)
    print(f"  {'AVERAGE':<38} {avg:>7.4f}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

    # Write machine-readable results
    with open("baseline_results.json", "w") as f:
        json.dump({"results": results, "average_score": round(avg, 4)}, f, indent=2)
    print("  Results saved to baseline_results.json")


if __name__ == "__main__":
    main()
