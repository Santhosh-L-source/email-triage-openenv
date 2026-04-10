"""
Reward shaping logic for the Email Triage environment.
Provides per-step partial rewards so the agent gets training signal throughout.
"""

from __future__ import annotations
from .models import EmailRecord, EmailAction, EmailReward


# Reward weights (sum of positives = 1.0 max per step)
_LABEL_CORRECT       = +0.30
_PRIORITY_CORRECT    = +0.20
_ROUTING_CORRECT     = +0.20
_REPLY_GOOD          = +0.20   # ROUGE-L >= 0.5
_REPLY_OK            = +0.10   # ROUGE-L in [0.2, 0.5)
_SPAM_ARCHIVED       = +0.10   # bonus for correctly archiving spam

# Penalty weights
_PRIORITY_DOWNGRADE  = -0.30   # urgent email given low/medium priority
_SLA_VIOLATED        = -0.50   # critical email not handled within sla_steps
_REPLY_BAD           = -0.10   # reply given but very poor quality (< 0.2)
_INVALID_ACTION      = -0.20   # empty/wrong email_id


def _rouge_l(prediction: str, reference: str) -> float:
    """Simplified ROUGE-L (LCS ratio) — no extra dependencies needed."""
    if not prediction or not reference:
        return 0.0
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # Build LCS table
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == pred_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    precision = lcs_len / n if n else 0.0
    recall = lcs_len / m if m else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_reward(
    record: EmailRecord,
    action: EmailAction,
    cumulative_reward: float,
    sla_violated: bool = False,
    task_id: str = "",
) -> EmailReward:
    """
    Compute the shaped reward for a single step.

    Args:
        record: Ground-truth email record.
        action: Agent's action on this email.
        cumulative_reward: Reward accumulated before this step.
        sla_violated: Whether an SLA deadline was missed.
        task_id: Current task (reply quality only in hard task).

    Returns:
        EmailReward with step_reward and breakdown fields.
    """
    # --- Correctness checks ---
    label_correct    = action.label    == record.true_label
    priority_correct = action.priority == record.true_priority
    routing_correct  = action.route_to == record.true_route

    step_reward = 0.0
    penalties   = 0.0
    reply_quality = 0.0

    # Label
    if label_correct:
        step_reward += _LABEL_CORRECT

    # Priority
    if priority_correct:
        step_reward += _PRIORITY_CORRECT
    elif record.true_priority in ("critical", "high") and action.priority in ("low", "medium"):
        # Downgrading a genuinely urgent email is especially bad
        penalties += _PRIORITY_DOWNGRADE

    # Routing
    if routing_correct:
        step_reward += _ROUTING_CORRECT

    # Archiving spam correctly
    if record.true_label == "spam" and action.archive:
        step_reward += _SPAM_ARCHIVED

    # Reply quality (only relevant for hard task or if reply is provided)
    if task_id == "inbox_zero_with_sla" and record.gold_reply:
        if action.draft_reply:
            reply_quality = _rouge_l(action.draft_reply, record.gold_reply)
            if reply_quality >= 0.5:
                step_reward += _REPLY_GOOD
            elif reply_quality >= 0.2:
                step_reward += _REPLY_OK
            else:
                penalties += _REPLY_BAD
        # No penalty for missing reply on non-support emails

    # SLA violation penalty
    if sla_violated:
        penalties += _SLA_VIOLATED

    step_reward += penalties
    # Clamp to [-1.0, 1.0]
    step_reward = max(-1.0, min(1.0, step_reward))

    return EmailReward(
        step_reward=round(step_reward, 4),
        cumulative_reward=round(cumulative_reward + step_reward, 4),
        label_correct=label_correct,
        priority_correct=priority_correct,
        routing_correct=routing_correct,
        reply_quality=round(reply_quality, 4),
        penalties=round(penalties, 4),
        sla_violated=sla_violated,
    )
