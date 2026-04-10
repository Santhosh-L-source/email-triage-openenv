#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate.py -- Pre-submission validation script for Email Triage OpenEnv.

Runs all checklist items defined in the submission spec and prints
a pass/fail report. Exit code 0 = all checks passed.

Usage:
    # With server already running on port 7860:
    python validate.py

    # Or point at a remote HF Space URL:
    python validate.py --url https://<your-space>.hf.space

    # Offline only (no live server required):
    python validate.py --offline
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import requests

PASS_ = "[PASS]"
FAIL_ = "[FAIL]"
WARN_ = "[WARN]"
SEP   = "=" * 64


def check(label: str, ok: bool, detail: str = "") -> Tuple[bool, str]:
    icon = PASS_ if ok else FAIL_
    msg  = f"  {icon}  {label}"
    if detail:
        msg += f"\n         {detail}"
    return ok, msg


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_env_vars() -> List[Tuple[bool, str]]:
    """1. Mandatory environment variables defined."""
    results = []
    for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        val = os.environ.get(var, "")
        results.append(check(
            f"Env var {var} is set",
            bool(val),
            detail=f"value={'<SET>' if val else '<MISSING>'}",
        ))
    return results


def check_files() -> List[Tuple[bool, str]]:
    """2. Required files present in root directory."""
    required = {
        "inference.py":   "Mandatory inference script",
        "validate.py":    "Pre-submission validation script",
        "openenv.yaml":   "OpenEnv spec file",
        "Dockerfile":     "Container build file",
        "requirements.txt": "Python dependencies",
    }
    results = []
    for fname, desc in required.items():
        exists = os.path.isfile(fname)
        results.append(check(f"File exists: {fname}", exists, detail=desc))
    return results


def check_openenv_yaml() -> List[Tuple[bool, str]]:
    """3. openenv.yaml is valid and has required fields."""
    results = []
    try:
        import yaml  # type: ignore
        with open("openenv.yaml", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

        results.append(check("openenv.yaml: parseable YAML", True))

        for field in ("name", "version", "tasks", "observation_space", "action_space", "reward_range"):
            present = field in spec
            results.append(check(f"openenv.yaml: field '{field}'", present))

        tasks = spec.get("tasks", [])
        results.append(check(
            "openenv.yaml: has 3+ tasks",
            len(tasks) >= 3,
            detail=f"found {len(tasks)} task(s)",
        ))

        for t in tasks:
            tid = t.get("id", "?")
            sr  = t.get("score_range", [])
            ok  = isinstance(sr, list) and len(sr) == 2 and sr[0] == 0.0 and sr[1] == 1.0
            results.append(check(
                f"Task '{tid}': score_range=[0.0, 1.0]",
                ok,
                detail=str(sr),
            ))

    except FileNotFoundError:
        results.append(check("openenv.yaml: parseable YAML", False, detail="file not found"))
    except Exception as exc:
        results.append(check("openenv.yaml: parseable YAML", False, detail=str(exc)))

    return results


def check_inference_script() -> List[Tuple[bool, str]]:
    """4. inference.py uses required env variables and OpenAI client."""
    results = []
    try:
        with open("inference.py", encoding="utf-8") as f:
            src = f.read()

        for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
            results.append(check(f"inference.py references {var}", var in src))

        results.append(check(
            "inference.py uses OpenAI client",
            "from openai import OpenAI" in src or "OpenAI(" in src,
        ))

        for tag in ("[START]", "[STEP]", "[END]"):
            results.append(check(f"inference.py emits {tag} log", tag in src))

    except FileNotFoundError:
        results.append(check("inference.py: readable", False, detail="file not found"))

    return results


def check_dockerfile() -> List[Tuple[bool, str]]:
    """5. Dockerfile has required instructions."""
    results = []
    try:
        with open("Dockerfile", encoding="utf-8") as f:
            df = f.read()

        results.append(check("Dockerfile: EXPOSE 7860",            "7860" in df))
        results.append(check("Dockerfile: HEALTHCHECK",            "HEALTHCHECK" in df))
        results.append(check("Dockerfile: CMD/ENTRYPOINT",         "CMD" in df or "ENTRYPOINT" in df))
        results.append(check(
            "Dockerfile: copies inference.py",
            "inference.py" in df or "COPY . ." in df,
        ))
    except FileNotFoundError:
        results.append(check("Dockerfile: readable", False, detail="file not found"))
    return results


def check_server_health(base_url: str) -> List[Tuple[bool, str]]:
    """6. Server responds 200 at root."""
    try:
        r = requests.get(f"{base_url}/", timeout=10)
        ok = r.status_code == 200
        return [check("GET / returns 200", ok, detail=f"status={r.status_code} url={base_url}")]
    except Exception as exc:
        return [check("GET / returns 200", False, detail=str(exc))]


def check_reset_endpoint(base_url: str) -> List[Tuple[bool, str]]:
    """7. POST /reset works and returns a valid observation."""
    results = []
    try:
        r = requests.post(
            f"{base_url}/reset",
            json={"task_id": "single_label_classification", "seed": 42},
            timeout=30,
        )
        ok = r.status_code == 200
        results.append(check("POST /reset returns 200", ok, detail=f"status={r.status_code}"))
        if ok:
            obs = r.json()
            for field in ("email_id", "subject", "sender", "body", "task_id", "step", "emails_remaining"):
                results.append(check(f"reset() obs has field '{field}'", field in obs))
    except Exception as exc:
        results.append(check("POST /reset returns 200", False, detail=str(exc)))
    return results


def check_step_endpoint(base_url: str) -> List[Tuple[bool, str]]:
    """8. POST /step works and returns valid structured response."""
    results = []
    try:
        r = requests.post(
            f"{base_url}/reset",
            json={"task_id": "single_label_classification", "seed": 42},
            timeout=30,
        )
        r.raise_for_status()
        obs = r.json()

        action = {
            "email_id":    obs["email_id"],
            "label":       "work",
            "priority":    "medium",
            "route_to":    "ignore",
            "draft_reply": None,
            "archive":     False,
        }
        r = requests.post(f"{base_url}/step", json=action, timeout=30)
        ok = r.status_code == 200
        results.append(check("POST /step returns 200", ok, detail=f"status={r.status_code}"))
        if ok:
            resp = r.json()
            for field in ("observation", "reward", "done", "info"):
                results.append(check(f"step() response has field '{field}'", field in resp))
            reward = resp.get("reward", {})
            score  = reward.get("step_reward", -99)
            in_range = -1.0 <= score <= 1.0
            results.append(check(
                "step() reward in [-1.0, 1.0]",
                in_range,
                detail=f"step_reward={score}",
            ))
    except Exception as exc:
        results.append(check("POST /step returns 200", False, detail=str(exc)))
    return results


def check_state_endpoint(base_url: str) -> List[Tuple[bool, str]]:
    """9. GET /state works."""
    results = []
    try:
        r = requests.get(f"{base_url}/state", timeout=10)
        ok = r.status_code == 200
        results.append(check("GET /state returns 200", ok, detail=f"status={r.status_code}"))
        if ok:
            state = r.json()
            for field in ("task_id", "step", "done", "cumulative_reward"):
                results.append(check(f"state() has field '{field}'", field in state))
    except Exception as exc:
        results.append(check("GET /state returns 200", False, detail=str(exc)))
    return results


def check_tasks_endpoint(base_url: str) -> List[Tuple[bool, str]]:
    """10. GET /tasks returns 3+ tasks."""
    results = []
    try:
        r = requests.get(f"{base_url}/tasks", timeout=10)
        ok = r.status_code == 200
        results.append(check("GET /tasks returns 200", ok, detail=f"status={r.status_code}"))
        if ok:
            tasks_resp = r.json()
            tasks = tasks_resp.get("tasks", [])
            ok3 = len(tasks) >= 3
            results.append(check("Tasks endpoint returns 3+ tasks", ok3, detail=f"found {len(tasks)}"))
    except Exception as exc:
        results.append(check("GET /tasks returns 200", False, detail=str(exc)))
    return results


def check_grader_scores(base_url: str) -> List[Tuple[bool, str]]:
    """11. Run each task's grader and verify scores in [0.0, 1.0]."""
    results = []
    task_ids = [
        "single_label_classification",
        "priority_triage_with_routing",
        "inbox_zero_with_sla",
    ]
    for task_id in task_ids:
        try:
            r = requests.post(
                f"{base_url}/reset",
                json={"task_id": task_id, "seed": 42},
                timeout=30,
            )
            r.raise_for_status()
            obs = r.json()

            action = {
                "email_id":    obs["email_id"],
                "label":       "work",
                "priority":    "medium",
                "route_to":    "ignore",
                "draft_reply": None,
                "archive":     False,
            }
            r = requests.post(f"{base_url}/step", json=action, timeout=30)
            r.raise_for_status()

            r = requests.post(f"{base_url}/grader", timeout=30)
            ok_status = r.status_code == 200
            results.append(check(f"Task '{task_id}': grader returns 200", ok_status))
            if ok_status:
                grade = r.json()
                score = grade.get("score", -1)
                in_range = 0.0 <= score <= 1.0
                results.append(check(
                    f"Task '{task_id}': score in [0.0, 1.0]",
                    in_range,
                    detail=f"score={score}",
                ))
        except Exception as exc:
            results.append(check(f"Task '{task_id}': grader", False, detail=str(exc)))

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-submission validation for Email Triage OpenEnv",
    )
    parser.add_argument(
        "--url",
        default="http://localhost:7860",
        help="Base URL of the running OpenEnv server (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Only run offline checks (no live server required)",
    )
    args = parser.parse_args()

    base_url = args.url.rstrip("/")

    print()
    print(SEP)
    print("  Email Triage OpenEnv -- Pre-Submission Validation")
    print(SEP)

    all_checks: List[Tuple[bool, str]] = []

    # -- Offline checks --
    print("\n[1/5] Environment Variables")
    for item in check_env_vars():
        all_checks.append(item)
        print(item[1])

    print("\n[2/5] Required Files")
    for item in check_files():
        all_checks.append(item)
        print(item[1])

    print("\n[3/5] openenv.yaml Spec")
    for item in check_openenv_yaml():
        all_checks.append(item)
        print(item[1])

    print("\n[4/5] inference.py Script")
    for item in check_inference_script():
        all_checks.append(item)
        print(item[1])

    print("\n[4b] Dockerfile")
    for item in check_dockerfile():
        all_checks.append(item)
        print(item[1])

    if args.offline:
        print("\n  [INFO] --offline flag set, skipping live server checks.")
    else:
        print(f"\n[5/5] Live Server Checks  ({base_url})")
        server_reachable = False

        h_checks = check_server_health(base_url)
        for item in h_checks:
            all_checks.append(item)
            print(item[1])
            if item[0]:
                server_reachable = True

        if server_reachable:
            for item in check_reset_endpoint(base_url):
                all_checks.append(item)
                print(item[1])
            for item in check_step_endpoint(base_url):
                all_checks.append(item)
                print(item[1])
            for item in check_state_endpoint(base_url):
                all_checks.append(item)
                print(item[1])
            for item in check_tasks_endpoint(base_url):
                all_checks.append(item)
                print(item[1])
            print("\nGrader Scores -- All 3 Tasks")
            for item in check_grader_scores(base_url):
                all_checks.append(item)
                print(item[1])
        else:
            print(f"\n  {WARN_} Server not reachable. Start it with:")
            print(f"       uvicorn server.app:app --host 0.0.0.0 --port 7860")
            print(f"  Then re-run: python validate.py --url {base_url}")

    # -- Summary --
    total  = len(all_checks)
    passed = sum(1 for ok, _ in all_checks if ok)
    failed = total - passed

    print()
    print(SEP)
    if failed == 0:
        print(f"  Summary: {passed}/{total} checks passed  {PASS_}  ALL CHECKS PASSED -- ready to submit!")
    else:
        print(f"  Summary: {passed}/{total} checks passed  {FAIL_}  {failed} check(s) failed -- fix before submitting.")
    print(SEP)
    print()

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
