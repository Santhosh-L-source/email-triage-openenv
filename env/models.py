"""
Pydantic models for the Email Triage OpenEnv environment.
Defines the typed observation, action, and reward spaces.
"""

from __future__ import annotations
from typing import Optional, Literal, List, Dict, Any
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class EmailObservation(BaseModel):
    """What the agent sees at each step."""
    email_id: str = Field(..., description="Unique identifier for the email")
    subject: str = Field(..., description="Email subject line")
    sender: str = Field(..., description="Sender display name")
    sender_domain: str = Field(..., description="Domain of sender's email address")
    body: str = Field(..., description="Email body text (may be truncated to 500 chars)")
    timestamp: str = Field(..., description="ISO 8601 timestamp the email was received")
    thread_length: int = Field(..., description="Number of messages in thread (1 = new)")
    has_attachments: bool = Field(..., description="Whether the email has file attachments")
    inbox_position: int = Field(..., description="Position in queue (0 = oldest unread)")

    # Episode context
    task_id: str = Field(..., description="Current task identifier")
    step: int = Field(..., description="Current step number within the episode")
    emails_remaining: int = Field(..., description="Number of emails left to process")


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

LabelType = Literal["work", "personal", "spam", "newsletter", "urgent", "support"]
PriorityType = Literal["low", "medium", "high", "critical"]
RouteType = Literal["engineering", "sales", "support", "hr", "finance", "ignore"]


class EmailAction(BaseModel):
    """Action the agent takes on a given email."""
    email_id: str = Field(..., description="ID of the email being acted upon")
    label: LabelType = Field(..., description="Classification label for the email")
    priority: PriorityType = Field(..., description="Priority level assigned to the email")
    route_to: RouteType = Field(
        default="ignore",
        description="Team/department to route the email to, or 'ignore' if not applicable"
    )
    draft_reply: Optional[str] = Field(
        default=None,
        description="Optional reply draft (required in hard task for support/urgent emails)"
    )
    archive: bool = Field(
        default=False,
        description="Whether to archive (remove from inbox) after handling"
    )


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class EmailReward(BaseModel):
    """Reward signal returned after each step."""
    step_reward: float = Field(..., description="Reward earned at this step (range: -1.0 to 1.0)")
    cumulative_reward: float = Field(..., description="Total reward accumulated in episode so far")
    label_correct: bool = Field(..., description="Whether the label was correct")
    priority_correct: bool = Field(..., description="Whether the priority was correct")
    routing_correct: bool = Field(..., description="Whether the routing was correct")
    reply_quality: float = Field(
        default=0.0,
        description="ROUGE-L score of draft_reply vs gold template (0.0 if no reply needed)"
    )
    penalties: float = Field(
        default=0.0,
        description="Total penalties applied at this step (negative value)"
    )
    sla_violated: bool = Field(
        default=False,
        description="Whether a SLA deadline was missed at this step (hard task only)"
    )


# ---------------------------------------------------------------------------
# Internal Email Record
# ---------------------------------------------------------------------------

class EmailRecord(BaseModel):
    """Internal email record used by the environment (not exposed to agent)."""
    email_id: str
    subject: str
    sender: str
    sender_domain: str
    body: str
    timestamp: str
    thread_length: int
    has_attachments: bool

    # Ground truth labels
    true_label: LabelType
    true_priority: PriorityType
    true_route: RouteType
    gold_reply: Optional[str] = None   # gold reply template for hard task
    sla_steps: Optional[int] = None    # if set, must be actioned within N steps


# ---------------------------------------------------------------------------
# Environment State
# ---------------------------------------------------------------------------

class EnvState(BaseModel):
    """Full environment state (returned by /state endpoint)."""
    task_id: str
    seed: int
    step: int
    done: bool
    cumulative_reward: float
    emails_total: int
    emails_processed: int
    emails_remaining: int
    current_email_id: Optional[str]
    sla_violations: int
    episode_history: List[Dict[str, Any]] = Field(default_factory=list)
