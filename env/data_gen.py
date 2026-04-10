"""
Synthetic email data generator for the Email Triage environment.
Produces realistic, diverse email records with ground-truth labels.
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Optional

from .models import EmailRecord, LabelType, PriorityType, RouteType


# ---------------------------------------------------------------------------
# Template pools
# ---------------------------------------------------------------------------

_TEMPLATES: list[dict] = [
    # ── WORK emails ────────────────────────────────────────────────────────
    {
        "label": "work", "priority": "medium", "route": "engineering",
        "sender": "Alice Chen", "domain": "techcorp.com",
        "subjects": [
            "Q3 sprint planning notes",
            "Updated architecture diagram for review",
            "Team standup recap — Tuesday",
            "Code review request: PR #412",
        ],
        "bodies": [
            "Hi team,\n\nPlease find the sprint planning notes attached. We'll be focusing on the auth refactor and API rate-limiting this cycle. Let me know if you have questions.\n\nThanks,\nAlice",
            "Sharing the updated architecture diagram. Please review before Thursday's design sync. Comments in Confluence.\n\nAlice",
            "Quick recap: we covered deployment blockers, Q3 OKRs, and the new onboarding flow. Full notes in the shared doc.\n\nAlice",
            "Could you review PR #412 when you get a chance? It addresses the memory leak in the worker pool. No rush, EOD is fine.\n\nAlice",
        ],
        "has_attachments": True,
    },
    {
        "label": "work", "priority": "high", "route": "engineering",
        "sender": "DevOps Bot", "domain": "alerts.techcorp.com",
        "subjects": [
            "🚨 CPU spike on prod-web-02",
            "Disk usage at 89% — prod-db-01",
            "Deploy pipeline failed: main branch",
        ],
        "bodies": [
            "Alert triggered at 14:32 UTC. prod-web-02 CPU > 95% for 5 minutes. Auto-scaling initiated. Manual review recommended.\n\nDevOps Bot",
            "Disk usage on prod-db-01 has exceeded 89%. Please investigate within 1 hour to avoid service disruption.\n\nDevOps Bot",
            "The CI/CD pipeline failed on commit a3f92b. Stage: integration tests. See attached logs for details.\n\nDevOps Bot",
        ],
        "has_attachments": False,
    },
    # ── URGENT emails ──────────────────────────────────────────────────────
    {
        "label": "urgent", "priority": "critical", "route": "engineering",
        "sender": "CTO Office", "domain": "techcorp.com",
        "subjects": [
            "URGENT: Production outage — all hands",
            "CRITICAL: Data breach suspected on auth service",
            "IMMEDIATELY: Board demo failing — need fix NOW",
        ],
        "bodies": [
            "Production is down. All engineering leads please join war room immediately: meet.techcorp.com/warroom\n\nThis is a P0 incident.",
            "Security team has flagged anomalous login patterns on auth-service. Treat as data breach. Incident response team please escalate immediately.",
            "The board demo environment is returning 500 errors. We have 45 minutes before the presentation. All hands.",
        ],
        "has_attachments": False,
        "gold_reply": "Acknowledged. Joining immediately and escalating to the relevant team.",
        "sla_steps": 3,
    },
    {
        "label": "urgent", "priority": "critical", "route": "support",
        "sender": "Enterprise Customer", "domain": "bigclient.com",
        "subjects": [
            "URGENT: Your API is down — we're losing revenue",
            "SLA breach — 4 hours of downtime unacceptable",
        ],
        "bodies": [
            "Your API has been returning 503 errors for the past 2 hours. Our orders system is completely down. We need a fix immediately or will invoke our SLA penalty clause.\n\nJohn Smith, VP Engineering, BigClient Inc.",
            "We have now had 4 hours of downtime this month. Per our contract this triggers an SLA review and potential penalty. Please escalate to your leadership.\n\nJohn Smith",
        ],
        "has_attachments": False,
        "gold_reply": "We sincerely apologize for the disruption. Our team is actively investigating and we will provide an update within 30 minutes.",
        "sla_steps": 3,
    },
    # ── SUPPORT emails ─────────────────────────────────────────────────────
    {
        "label": "support", "priority": "medium", "route": "support",
        "sender": "Customer User", "domain": "gmail.com",
        "subjects": [
            "Can't log into my account",
            "How do I export my data?",
            "Billing question — charged twice",
            "Feature request: dark mode",
            "Reset password not working",
        ],
        "bodies": [
            "Hi,\n\nI've been trying to log in for the past hour and keep getting 'Invalid credentials' even though I just reset my password. Please help!\n\nThanks",
            "Hello, I need to export all my project data as CSV for a compliance audit. I can't find the option in the settings. Could you point me in the right direction?\n\nBest,",
            "I was charged twice on March 3rd for my Pro subscription. Please refund the duplicate charge. My account email is user@example.com.\n\nRegards,",
            "Love the product! One thing that would really help is a dark mode option. Many of us work late at night. Any plans for this?\n\nCheers,",
            "The reset password link in the email I received gives me a 404 error. I've tried 3 times. Please help as I'm locked out.\n\nThanks",
        ],
        "has_attachments": False,
        "gold_reply": "Thank you for reaching out. Our support team will review your request and respond within 24 hours.",
    },
    # ── SPAM emails ────────────────────────────────────────────────────────
    {
        "label": "spam", "priority": "low", "route": "ignore",
        "sender": "WINNER NOTIFICATION", "domain": "promo-deals-now.xyz",
        "subjects": [
            "You've been selected — claim your $500 gift card!",
            "Congratulations! iPhone 15 Pro is YOURS",
            "Your package is waiting — verify to claim",
        ],
        "bodies": [
            "CONGRATULATIONS! You have been randomly selected to receive a $500 gift card. Click here to claim NOW before it expires: http://totally-legit.xyz/claim\n\nThis offer expires in 24 hours!",
            "You are our lucky winner! To receive your free iPhone 15 Pro, simply click the link and verify your details. Limited time offer!",
            "A package is being held for you. To claim, please verify your shipping details using the link below. Act within 48 hours.",
        ],
        "has_attachments": False,
    },
    {
        "label": "spam", "priority": "low", "route": "ignore",
        "sender": "Dr. James Williams", "domain": "inheritance-help.ng",
        "subjects": [
            "Strictly Confidential Business Proposal",
            "I need your trusted assistance",
        ],
        "bodies": [
            "Dear Friend,\n\nI am the late Dr. Williams' attorney. He left behind $12.5 million USD and named you as beneficiary. Please reply with your bank details to process the transfer.\n\nConfidentially,\nDr. James",
            "I am reaching out regarding a very personal matter that requires absolute discretion. A sum of $8 million in inheritance requires a trustworthy foreign partner...",
        ],
        "has_attachments": False,
    },
    # ── NEWSLETTER emails ──────────────────────────────────────────────────
    {
        "label": "newsletter", "priority": "low", "route": "ignore",
        "sender": "TechCrunch", "domain": "techcrunch.com",
        "subjects": [
            "This week in AI: GPT-5 rumors, Mistral raises $1B",
            "Your weekly startup digest — March edition",
            "Breaking: OpenAI launches new model",
        ],
        "bodies": [
            "This week's top stories:\n• GPT-5 rumored to launch Q2\n• Mistral raises $1B Series B\n• EU AI Act enforcement begins\n\nRead more at techcrunch.com",
            "Startup news this week: 23 new unicorns, VC funding up 18% YoY, and the top 10 AI tools of 2025.\n\nUnsubscribe | View in browser",
            "OpenAI has launched its latest model with 2M token context length and improved reasoning. Full breakdown inside.",
        ],
        "has_attachments": False,
    },
    # ── PERSONAL emails ────────────────────────────────────────────────────
    {
        "label": "personal", "priority": "low", "route": "ignore",
        "sender": "Mom", "domain": "gmail.com",
        "subjects": [
            "Dinner on Sunday?",
            "Did you see the news?",
            "Happy Birthday to your cousin!",
        ],
        "bodies": [
            "Hi honey,\n\nAre you free for dinner on Sunday? Dad and I are thinking of making your favourite pasta. Let me know!\n\nLove, Mom",
            "Did you see what happened on the news today? Give me a call when you get a chance.\n\nLove, Mom",
            "Just wanted to remind you — it's your cousin Jake's birthday tomorrow. Maybe send him a message!\n\nXOXO, Mom",
        ],
        "has_attachments": False,
    },
    {
        "label": "personal", "priority": "low", "route": "ignore",
        "sender": "LinkedIn", "domain": "linkedin.com",
        "subjects": [
            "You have 3 new connection requests",
            "Sarah commented on your post",
            "5 people viewed your profile this week",
        ],
        "bodies": [
            "You have 3 new connection requests waiting. See who wants to connect.\n\nLinkedIn Team",
            "Sarah Johnson commented on your post: 'Great insights! Totally agree with your take on async patterns.'",
            "Your profile is getting noticed! 5 people viewed your profile this week. See who's been looking.",
        ],
        "has_attachments": False,
    },
    # ── SALES emails ───────────────────────────────────────────────────────
    {
        "label": "work", "priority": "high", "route": "sales",
        "sender": "Sales Lead", "domain": "techcorp.com",
        "subjects": [
            "Inbound: Fortune 500 company interested in Enterprise plan",
            "Hot lead: $200K deal — needs response today",
            "Follow-up: Q3 contract renewal — Acme Corp",
        ],
        "bodies": [
            "We've got a hot inbound from MegaCorp (12,000 employees). They want a demo of the Enterprise plan ASAP. Revenue potential: $180K ARR. Who can own this?\n\nSales Team",
            "A lead from yesterday's conference is very interested. They have a Q1 budget to spend ($200K) and need a response today to move forward. Please prioritize.\n\nSales",
            "Acme Corp's contract is up for renewal in 30 days. They need a proposal with updated pricing. Let's not lose this one — $95K ARR.\n\nSales",
        ],
        "has_attachments": False,
    },
]


def _pick(lst: list, rng: random.Random) -> str:
    return rng.choice(lst)


def generate_email(
    email_id: str,
    rng: random.Random,
    base_time: datetime,
    position: int,
    task_id: str,
    force_label: Optional[LabelType] = None,
) -> EmailRecord:
    """Generate a single synthetic email record."""
    pool = _TEMPLATES
    if force_label:
        pool = [t for t in _TEMPLATES if t["label"] == force_label] or _TEMPLATES

    tpl = _pick(pool, rng)
    timestamp = (base_time + timedelta(minutes=position * rng.randint(3, 30))).isoformat()

    return EmailRecord(
        email_id=email_id,
        subject=_pick(tpl["subjects"], rng),
        sender=tpl["sender"],
        sender_domain=tpl["domain"],
        body=_pick(tpl["bodies"], rng)[:800],
        timestamp=timestamp,
        thread_length=rng.randint(1, tpl.get("thread_length_max", 4)),
        has_attachments=tpl["has_attachments"],
        true_label=tpl["label"],
        true_priority=tpl["priority"],
        true_route=tpl["route"],
        gold_reply=tpl.get("gold_reply"),
        sla_steps=tpl.get("sla_steps"),
    )


def generate_inbox(
    task_id: str,
    seed: int,
    n_emails: int,
    label_distribution: Optional[dict] = None,
) -> List[EmailRecord]:
    """
    Generate a complete inbox of n_emails for the given task.

    Args:
        task_id: The task identifier.
        seed: Random seed for reproducibility.
        n_emails: Number of emails to generate.
        label_distribution: Optional dict mapping label → count to enforce distribution.

    Returns:
        List of EmailRecord objects in inbox order (oldest first).
    """
    rng = random.Random(seed)
    base_time = datetime(2025, 3, 1, 8, 0, 0)

    records: List[EmailRecord] = []

    if label_distribution:
        forced: List[Optional[LabelType]] = []
        for lbl, cnt in label_distribution.items():
            forced.extend([lbl] * cnt)  # type: ignore
        # pad with None (random)
        while len(forced) < n_emails:
            forced.append(None)
        rng.shuffle(forced)
        labels = forced[:n_emails]
    else:
        labels = [None] * n_emails

    for i in range(n_emails):
        eid = f"email_{i:03d}"
        rec = generate_email(eid, rng, base_time, i, task_id, force_label=labels[i])
        records.append(rec)

    return records
