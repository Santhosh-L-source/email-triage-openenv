"""
Core Email Triage environment — implements the OpenEnv step/reset/state interface.
"""

from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    EmailAction, EmailObservation, EmailRecord, EmailReward, EnvState
)
from .data_gen import generate_inbox
from .reward import compute_reward
from .tasks import TASKS, grade_episode


class EmailTriageEnv:
    """
    Email Triage OpenEnv environment.

    The agent processes one email at a time, taking an EmailAction for each.
    The environment tracks SLA deadlines and accumulates a shaped reward.

    Usage:
        env = EmailTriageEnv()
        obs = env.reset(task_id="single_label_classification", seed=42)
        while True:
            action = EmailAction(email_id=obs.email_id, label="work", ...)
            obs, reward, done, info = env.step(action)
            if done:
                break
        score, details = env.grade()
    """

    def __init__(self) -> None:
        self._task_id: str = ""
        self._seed: int = 0
        self._records: List[EmailRecord] = []
        self._actions: List[EmailAction] = []
        self._step: int = 0
        self._cumulative_reward: float = 0.0
        self._sla_violations: int = 0
        self._done: bool = False
        # SLA tracking: email_id → latest step it must be handled by
        self._sla_deadlines: Dict[str, int] = {}
        # History for /state
        self._episode_history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "single_label_classification", seed: int = 42) -> EmailObservation:
        """
        Reset the environment for the given task and seed.

        Args:
            task_id: One of the three task identifiers.
            seed: Random seed for reproducible inbox generation.

        Returns:
            Initial EmailObservation (first email in inbox).
        """
        if task_id not in TASKS:
            raise ValueError(f"Unknown task_id '{task_id}'. Valid: {list(TASKS.keys())}")

        task = TASKS[task_id]
        self._task_id = task_id
        self._seed = seed
        self._step = 0
        self._cumulative_reward = 0.0
        self._sla_violations = 0
        self._done = False
        self._actions = []
        self._episode_history = []

        self._records = generate_inbox(
            task_id=task_id,
            seed=seed,
            n_emails=task.n_emails,
            label_distribution=copy.deepcopy(task.label_distribution),
        )

        # Pre-compute SLA deadlines
        self._sla_deadlines = {}
        for i, rec in enumerate(self._records):
            if rec.sla_steps is not None:
                # Must be handled by step i + sla_steps
                deadline = i + rec.sla_steps
                self._sla_deadlines[rec.email_id] = deadline

        return self._make_observation()

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self, action: EmailAction
    ) -> Tuple[Optional[EmailObservation], EmailReward, bool, Dict[str, Any]]:
        """
        Process an action for the current email.

        Args:
            action: EmailAction with label, priority, routing, optional reply.

        Returns:
            (next_observation, reward, done, info)
            next_observation is None when done=True (no more emails).
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping again.")

        current_record = self._records[self._step]

        # Validate email_id matches current email
        if action.email_id != current_record.email_id:
            # Penalise wrong email_id but still advance
            reward = EmailReward(
                step_reward=-0.20,
                cumulative_reward=round(self._cumulative_reward - 0.20, 4),
                label_correct=False,
                priority_correct=False,
                routing_correct=False,
                reply_quality=0.0,
                penalties=-0.20,
                sla_violated=False,
            )
            self._cumulative_reward += -0.20
            self._actions.append(action)
            self._step += 1
            self._check_done()
            obs = self._make_observation() if not self._done else None
            return obs, reward, self._done, {"error": "email_id mismatch"}

        # Check if any past SLA deadlines have been violated at this step
        sla_violated = False
        for eid, deadline in list(self._sla_deadlines.items()):
            if self._step >= deadline and eid != current_record.email_id:
                # This email passed its deadline without being actioned
                sla_violated = True
                self._sla_violations += 1
                del self._sla_deadlines[eid]

        # Remove current email from SLA tracking (it's being handled now)
        self._sla_deadlines.pop(current_record.email_id, None)

        reward = compute_reward(
            record=current_record,
            action=action,
            cumulative_reward=self._cumulative_reward,
            sla_violated=sla_violated,
            task_id=self._task_id,
        )
        self._cumulative_reward = reward.cumulative_reward
        self._actions.append(action)

        # Log history entry
        self._episode_history.append({
            "step": self._step,
            "email_id": current_record.email_id,
            "true_label": current_record.true_label,
            "agent_label": action.label,
            "step_reward": reward.step_reward,
        })

        self._step += 1
        self._check_done()

        # Check any remaining SLA violations after final step
        if self._done:
            for eid in list(self._sla_deadlines.keys()):
                self._sla_violations += 1

        obs = self._make_observation() if not self._done else None
        info = {
            "step": self._step,
            "sla_violations": self._sla_violations,
            "cumulative_reward": self._cumulative_reward,
        }
        return obs, reward, self._done, info

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> EnvState:
        """Return the full current environment state."""
        current_id = (
            self._records[self._step].email_id
            if not self._done and self._step < len(self._records)
            else None
        )
        return EnvState(
            task_id=self._task_id,
            seed=self._seed,
            step=self._step,
            done=self._done,
            cumulative_reward=self._cumulative_reward,
            emails_total=len(self._records),
            emails_processed=self._step,
            emails_remaining=max(0, len(self._records) - self._step),
            current_email_id=current_id,
            sla_violations=self._sla_violations,
            episode_history=self._episode_history,
        )

    # ------------------------------------------------------------------
    # grade
    # ------------------------------------------------------------------

    def grade(self) -> Tuple[float, Dict[str, Any]]:
        """
        Run the task grader and return (score, details).
        Can be called mid-episode (grades actions so far) or at end.
        """
        records_so_far = self._records[: len(self._actions)]
        return grade_episode(
            task_id=self._task_id,
            records=records_so_far,
            actions=self._actions,
            sla_violations=self._sla_violations,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> Optional[EmailObservation]:
        if self._done or self._step >= len(self._records):
            return None
        rec = self._records[self._step]
        return EmailObservation(
            email_id=rec.email_id,
            subject=rec.subject,
            sender=rec.sender,
            sender_domain=rec.sender_domain,
            body=rec.body,
            timestamp=rec.timestamp,
            thread_length=rec.thread_length,
            has_attachments=rec.has_attachments,
            inbox_position=self._step,
            task_id=self._task_id,
            step=self._step,
            emails_remaining=max(0, len(self._records) - self._step),
        )

    def _check_done(self) -> None:
        if self._step >= len(self._records):
            self._done = True
