import time
import requests

url = "https://santhoshl-email-triage-openenv.hf.space"
print("Waiting 45s for Space to rebuild...")
time.sleep(45)

# Check health
r = requests.get(url, timeout=20)
print("GET /  status:", r.status_code)

# Check reset with no body
r2 = requests.post(url + "/reset", timeout=20)
print("POST /reset (no body)  status:", r2.status_code)

if r.status_code == 200 and r2.status_code == 200:
    print("[PASS] Space healthy - click Update Submission now!")
else:
    print("[FAIL] Space not responding correctly")
    print(r2.text[:200])
