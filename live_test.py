import requests, json, sys

BASE = "http://localhost:7860"
PASS = "[PASS]"
FAIL = "[FAIL]"
all_ok = True

def chk(label, ok, detail=""):
    global all_ok
    if not ok:
        all_ok = False
    icon = PASS if ok else FAIL
    print(f"  {icon}  {label}" + (f"  ({detail})" if detail else ""))

print()
print("=" * 55)
print("  Email Triage OpenEnv -- Live Test")
print("=" * 55)

# 1. Health check
try:
    r = requests.get(f"{BASE}/", timeout=5)
    chk("GET /  health", r.status_code == 200, f"status={r.status_code}")
except Exception as e:
    chk("GET /  health", False, str(e)); sys.exit(1)

# 2. Tasks list
r = requests.get(f"{BASE}/tasks", timeout=5)
tasks = r.json().get("tasks", [])
chk("GET /tasks", r.status_code == 200 and len(tasks) >= 3, f"{len(tasks)} tasks")

# 3. State
r = requests.get(f"{BASE}/state", timeout=5)
chk("GET /state", r.status_code == 200, f"status={r.status_code}")

# 4-6. Reset + Step + Grader for each task
print()
for tid in ["single_label_classification", "priority_triage_with_routing", "inbox_zero_with_sla"]:
    print(f"  --- Task: {tid} ---")

    r = requests.post(f"{BASE}/reset", json={"task_id": tid, "seed": 42}, timeout=10)
    chk("  POST /reset", r.status_code == 200, f"status={r.status_code}")
    obs = r.json()
    has_fields = all(f in obs for f in ["email_id", "subject", "sender", "body", "task_id"])
    chk("  obs fields present", has_fields)

    action = {
        "email_id": obs["email_id"],
        "label": "work",
        "priority": "medium",
        "route_to": "ignore",
        "draft_reply": None,
        "archive": False,
    }
    r = requests.post(f"{BASE}/step", json=action, timeout=10)
    chk("  POST /step", r.status_code == 200, f"status={r.status_code}")
    resp = r.json()
    reward = resp["reward"]["step_reward"]
    chk("  step_reward in [-1,1]", -1.0 <= reward <= 1.0, f"reward={reward}")

    r = requests.post(f"{BASE}/grader", timeout=10)
    chk("  POST /grader", r.status_code == 200, f"status={r.status_code}")
    score = r.json().get("score", -1)
    chk("  score in [0,1]", 0.0 <= score <= 1.0, f"score={score:.4f}")
    print()

print("=" * 55)
if all_ok:
    print(f"  {PASS}  ALL TESTS PASSED -- project is working perfectly!")
else:
    print(f"  {FAIL}  Some tests failed.")
print("=" * 55)
print()
sys.exit(0 if all_ok else 1)
