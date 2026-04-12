"""Check GitHub Actions status and upload corrected file to HF Space via web API."""
import requests, json, sys

# Check GitHub Actions status
r = requests.get(
    "https://api.github.com/repos/Santhosh-L-source/email-triage-openenv/actions/runs",
    headers={"Accept": "application/vnd.github+json"},
)
data = r.json()
if "workflow_runs" in data:
    for run in data["workflow_runs"][:3]:
        name       = run.get("name", "?")
        status     = run.get("status", "?")
        conclusion = run.get("conclusion", "?")
        created    = run.get("created_at", "?")
        url        = run.get("html_url", "?")
        print(f"Run:       {name}")
        print(f"Status:    {status} / {conclusion}")
        print(f"Created:   {created}")
        print(f"URL:       {url}")
        print()
else:
    print("No runs:", json.dumps(data, indent=2)[:200])
