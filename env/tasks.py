"""
Task definitions and graders for the Email Triage environment.

Task 1 (easy):   single_label_classification   — classify 10 emails by label
Task 2 (medium): priority_triage_with_routing  — label + priority + route 20 emails
Task 3 (hard):   inbox_zero_with_sla           — full triage + replies + SLA on 30 emails
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from .models import EmailRecord, EmailAction, LabelType, PriorityType, RouteType


# ---------------------------------------------------------------------------
# Task metadata
# ---------------------------------------------------------------------------

@dataclass
class TaskDefinition:
    task_id: str
    description: str
    difficulty: str          # easy / medium / hard
    n_emails: int
    label_distribution: Dict[str, int]   # label → count (must sum to n_emails)
    requires_reply: bool
    requires_sla: bool


TASKS: Dict[str, TaskDefinition] = {
    "single_label_classification": TaskDefinition(
        task_id="single_label_classification",
        description=(
            "Classify 10 emails into one of 6 categories: work, personal, spam, "
            "newsletter, urgent, support. Priority and routing are ignored. "
            "Score = fraction of emails correctly labelled."
        ),
        difficulty="easy",
        n_emails=10,
        label_distribution={
            "work": 2, "personal": 2, "spam": 2,
            "newsletter": 2, "urgent": 1, "support": 1,
        },
        requires_reply=False,
        requires_sla=False,
    ),
    "priority_triage_with_routing": TaskDefinition(
        task_id="priority_triage_with_routing",
        description=(
            "Process 20 emails: assign label, priority (low/medium/high/critical), "
            "and route to the correct team (engineering/sales/support/hr/finance/ignore). "
            "Score is weighted: label 40%, priority 30%, routing 30%."
        ),
        difficulty="medium",
        n_emails=20,
        label_distribution={
            "work": 5, "personal": 3, "spam": 3,
            "newsletter": 3, "urgent": 3, "support": 3,
        },
        requires_reply=False,
        requires_sla=False,
    ),
    "inbox_zero_with_sla": TaskDefinition(
        task_id="inbox_zero_with_sla",
        description=(
            "Process 30 emails under SLA constraints. Critical/urgent emails must be "
            "handled within their sla_steps budget. Support and urgent emails require "
            "a draft_reply. Score: label 20%, priority 20%, routing 20%, reply ROUGE-L 20%, "
            "SLA compliance 20%."
        ),
        difficulty="hard",
        n_emails=30,
        label_distribution={
            "work": 7, "personal": 3, "spam": 5,
            "newsletter": 3, "urgent": 6, "support": 6,
        },
        requires_reply=True,
        requires_sla=True,
    ),
}


def list_tasks() -> List[Dict[str, Any]]:
    """Return task listing with action schema for /tasks endpoint."""
    action_schema = {
        "email_id": "string — ID of the email to act on",
        "label": "enum[work|personal|spam|newsletter|urgent|support]",
        "priority": "enum[low|medium|high|critical]",
        "route_to": "enum[engineering|sales|support|hr|finance|ignore]",
        "draft_reply": "string|null — reply draft (required in hard task for urgent/support)",
        "archive": "boolean — set true to archive the email",
    }
    return [
        {
            "task_id": task.task_id,
            "description": task.description,
            "difficulty": task.difficulty,
            "n_emails": task.n_emails,
            "requires_reply": task.requires_reply,
            "requires_sla": task.requires_sla,
            "action_schema": action_schema,
        }
        for task in TASKS.values()
    ]


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _clamp(score: float) -> float:
    """Clamp score to strictly (0, 1) exclusive as required by the validator."""
    return max(0.001, min(0.999, score))


def _rouge_l(prediction: str, reference: str) -> float:
    """LCS-based ROUGE-L (no external deps)."""
    if not prediction or not reference:
        return 0.0
    pred = prediction.lower().split()
    ref = reference.lower().split()
    m, n = len(ref), len(pred)
    if m == 0 or n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if ref[i - 1] == pred[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / n
    r = lcs / m
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def grade_episode(
    task_id: str,
    records: List[EmailRecord],
    actions: List[EmailAction],
    sla_violations: int,
) -> Tuple[float, Dict[str, Any]]:
    """
    Grade a completed episode and return (score 0.0–1.0, details dict).

    Args:
        task_id: Which task was run.
        records: Ground-truth email records in order.
        actions: Agent actions in order (must match records by index).
        sla_violations: Number of SLA deadlines missed.

    Returns:
        (score, details): score in [0.0, 1.0], details dict with sub-scores.
    """
    if task_id not in TASKS:
        raise ValueError(f"Unknown task_id: {task_id}")

    n = len(records)
    if n == 0 or len(actions) == 0:
        return 0.0, {"error": "No actions recorded"}

    task = TASKS[task_id]

    label_hits     = sum(1 for r, a in zip(records, actions) if a.label    == r.true_label)
    priority_hits  = sum(1 for r, a in zip(records, actions) if a.priority == r.true_priority)
    routing_hits   = sum(1 for r, a in zip(records, actions) if a.route_to == r.true_route)

    label_acc    = label_hits    / n
    priority_acc = priority_hits / n
    routing_acc  = routing_hits  / n

    details: Dict[str, Any] = {
        "task_id": task_id,
        "n_emails": n,
        "label_accuracy": round(label_acc, 4),
        "priority_accuracy": round(priority_acc, 4),
        "routing_accuracy": round(routing_acc, 4),
    }

    # ── Task 1: label only ─────────────────────────────────────────────────
    if task_id == "single_label_classification":
        score = _clamp(label_acc)
        details["score"] = round(score, 4)
        return round(score, 4), details

    # ── Task 2: label 40% + priority 30% + routing 30% ────────────────────
    if task_id == "priority_triage_with_routing":
        score = _clamp(0.4 * label_acc + 0.3 * priority_acc + 0.3 * routing_acc)
        details["score"] = round(score, 4)
        return round(score, 4), details

    # ── Task 3: label 20% + priority 20% + routing 20% + reply 20% + SLA 20%
    if task_id == "inbox_zero_with_sla":
        # Reply quality
        total_reply_score = 0.0
        reply_needed = 0
        for r, a in zip(records, actions):
            if r.gold_reply:
                reply_needed += 1
                rl = _rouge_l(a.draft_reply or "", r.gold_reply)
                total_reply_score += rl
        reply_avg = (total_reply_score / reply_needed) if reply_needed > 0 else 1.0

        # SLA compliance: penalise per violation
        max_sla_emails = sum(1 for r in records if r.sla_steps is not None)
        sla_score = max(0.0, 1.0 - (sla_violations / max_sla_emails)) if max_sla_emails > 0 else 1.0

        score = _clamp(
            0.20 * label_acc
            + 0.20 * priority_acc
            + 0.20 * routing_acc
            + 0.20 * reply_avg
            + 0.20 * sla_score
        )
        details["reply_avg_rouge_l"] = round(reply_avg, 4)
        details["sla_violations"] = sla_violations
        details["sla_score"] = round(sla_score, 4)
        details["score"] = round(score, 4)
        return round(score, 4), details

    return _clamp(0.0), details
