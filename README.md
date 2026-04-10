---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - email-triage
  - agent-evaluation
short_description: OpenEnv environment for AI email triage agents
---

# 📧 Email Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.dev)
[![Python](https://img.shields.io/badge/python-3.11-brightgreen)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **real-world OpenEnv environment** where AI agents learn to manage email inboxes — classifying, prioritizing, routing, and replying to realistic emails under SLA constraints.

> Email triage is one of the highest-frequency knowledge-work tasks. This environment trains and evaluates agents on real productivity workflows, not toy problems.

---

## 🌍 Environment Description

Each episode presents an agent with a stream of realistic synthetic emails. The agent must:

1. **Classify** each email into one of 6 categories
2. **Prioritize** correctly (low / medium / high / critical)
3. **Route** to the right team (engineering, sales, support, hr, finance, ignore)
4. **Draft replies** for customer-facing and urgent emails
5. **Meet SLA deadlines** — critical emails flagged with countdown budgets

The environment generates diverse, realistic emails with varied language patterns, sender types, and urgency signals. No two episodes are identical (seeded for reproducibility).

---

## 📦 Action Space

```json
{
  "email_id":    "string   — ID of the email being actioned",
  "label":       "enum     — work | personal | spam | newsletter | urgent | support",
  "priority":    "enum     — low | medium | high | critical",
  "route_to":    "enum     — engineering | sales | support | hr | finance | ignore",
  "draft_reply": "string | null — required for urgent/support in hard task",
  "archive":     "boolean  — set true to archive spam/newsletter"
}
```

## 👁️ Observation Space

```json
{
  "email_id":        "string",
  "subject":         "string",
  "sender":          "string",
  "sender_domain":   "string",
  "body":            "string (≤ 800 chars)",
  "timestamp":       "ISO 8601",
  "thread_length":   "integer",
  "has_attachments": "boolean",
  "inbox_position":  "integer",
  "task_id":         "string",
  "step":            "integer",
  "emails_remaining":"integer"
}
```

---

## 🎯 Tasks

| Task | Difficulty | Emails | Description |
|------|-----------|--------|-------------|
| `single_label_classification` | 🟢 Easy | 10 | Classify each email into one of 6 labels |
| `priority_triage_with_routing` | 🟡 Medium | 20 | Label (40%) + Priority (30%) + Routing (30%) |
| `inbox_zero_with_sla` | 🔴 Hard | 30 | Full triage + reply drafting + SLA deadlines |

### Expected Baseline Scores (GPT-4o-mini, seed=42)

| Task | Score |
|------|-------|
| `single_label_classification` | ~0.84 |
| `priority_triage_with_routing` | ~0.63 |
| `inbox_zero_with_sla` | ~0.45 |

---

## 🏆 Reward Function

Shaped reward at every step (range: −1.0 to +1.0):

| Signal | Value |
|--------|-------|
| Correct label | +0.30 |
| Correct priority | +0.20 |
| Correct routing | +0.20 |
| Spam correctly archived | +0.10 |
| Reply quality ROUGE-L ≥ 0.5 | +0.20 |
| Reply quality 0.2–0.5 | +0.10 |
| Urgent email deprioritized | −0.30 |
| SLA deadline missed | −0.50 |
| Poor reply quality | −0.10 |
| Invalid action (wrong email_id) | −0.20 |

---

## 🚀 Setup & Usage

### Option 1 — Run with Docker

```bash
git clone https://huggingface.co/spaces/<your-user>/email-triage-openenv
cd email-triage-openenv

docker build -t email-triage-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... email-triage-env
```

### Option 2 — Run locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/`       | Health check |
| `POST` | `/reset`  | Start a new episode |
| `POST` | `/step`   | Submit an action |
| `GET`  | `/state`  | Get current state |
| `GET`  | `/tasks`  | List tasks + action schema |
| `POST` | `/grader` | Grade completed episode |
| `POST` | `/baseline` | Run baseline inference |

### Quick start example

```python
import requests

BASE = "http://localhost:7860"

# 1. Reset for task
obs = requests.post(f"{BASE}/reset", json={"task_id": "single_label_classification", "seed": 42}).json()
print(f"First email: {obs['subject']} from {obs['sender']}")

# 2. Step with an action
action = {
    "email_id": obs["email_id"],
    "label": "work",
    "priority": "medium",
    "route_to": "engineering",
    "draft_reply": None,
    "archive": False
}
result = requests.post(f"{BASE}/step", json=action).json()
print(f"Reward: {result['reward']['step_reward']} | Done: {result['done']}")

# 3. Grade
score = requests.post(f"{BASE}/grader").json()
print(f"Episode score: {score['score']}")
```

---

## 🤖 Running the Baseline

```bash
export OPENAI_API_KEY=sk-...

# Make sure the server is running first
uvicorn server.app:app --host 0.0.0.0 --port 7860 &

# Run baseline on all 3 tasks
python baseline.py --url http://localhost:7860
```

Expected output:
```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Email Triage OpenEnv — Baseline Inference (GPT-4o-mini)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Task                                   Score  Difficulty
  --------------------------------------------------------
  single_label_classification            0.8400  easy
  priority_triage_with_routing           0.6300  medium
  inbox_zero_with_sla                    0.4500  hard
  --------------------------------------------------------
  AVERAGE                                0.6400
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

Results are also saved to `baseline_results.json`.

---

## 📁 Project Structure

```
├── env/
│   ├── __init__.py        # Package exports
│   ├── models.py          # Pydantic models: Observation, Action, Reward
│   ├── email_env.py       # EmailTriageEnv — step/reset/state/grade
│   ├── data_gen.py        # Synthetic email generator (seeded, reproducible)
│   ├── tasks.py           # Task definitions + deterministic graders
│   └── reward.py          # Shaped reward function with ROUGE-L
├── server/
│   ├── __init__.py
│   └── app.py             # FastAPI server (all endpoints)
├── baseline.py            # Standalone baseline inference script
├── openenv.yaml           # OpenEnv spec metadata
├── Dockerfile             # Container definition (port 7860)
├── requirements.txt
└── README.md
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
