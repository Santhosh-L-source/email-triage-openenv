import requests

url = "https://santhoshl-email-triage-openenv.hf.space"

# 1. Health check
r = requests.get(url, timeout=30)
print(f"GET /  status={r.status_code}  length={len(r.text)} bytes")
if r.status_code == 200:
    print("[PASS] Space is LIVE!")

# 2. Reset
r = requests.post(f"{url}/reset", json={"task_id": "single_label_classification", "seed": 42}, timeout=30)
print(f"POST /reset  status={r.status_code}")
if r.status_code == 200:
    obs = r.json()
    print(f"  email_id={obs.get('email_id')}")
    print("[PASS] reset() works!")
