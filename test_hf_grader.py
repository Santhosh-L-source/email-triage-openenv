"""Test if the live HF Space grader returns valid scores."""
import requests
import json

HF_URL = "https://santhoshl-email-triage-openenv.hf.space"

TASK_IDS = [
    "single_label_classification",
    "priority_triage_with_routing",
    "inbox_zero_with_sla",
]

for task_id in TASK_IDS:
    print(f"\n--- Task: {task_id} ---")
    try:
        r = requests.post(f"{HF_URL}/reset", json={"task_id": task_id, "seed": 42}, timeout=30)
        r.raise_for_status()
        obs = r.json()
        print(f"Reset OK, email_id={obs.get('email_id')}")

        action = {
            "email_id": obs["email_id"],
            "label": "work",
            "priority": "medium",
            "route_to": "ignore",
            "draft_reply": None,
            "archive": False,
        }
        r2 = requests.post(f"{HF_URL}/step", json=action, timeout=30)
        r2.raise_for_status()
        print(f"Step OK, status={r2.status_code}")

        r3 = requests.post(f"{HF_URL}/grader", timeout=30)
        r3.raise_for_status()
        result = r3.json()
        score = result.get("score", "MISSING")
        print(f"Grader score = {score}")
        valid = 0.0 < float(score) < 1.0 if score != "MISSING" else False
        print(f"Is strictly in (0,1): {valid}")
        if not valid:
            print(f"  !! INVALID SCORE — this is why it keeps failing !!")
    except Exception as e:
        print(f"Error: {e}")

print("\nDone.")
