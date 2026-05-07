"""
Loyalty Tracker

Computes longitudinal relationship quality from cross-session emotional
history and injects it as valence/arousal baseline into EmotionalState.

Based on exponentially time-decayed average of per-session valence.
Half-life ≈ 14 days by default (lambda = 0.05).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class LoyaltyTier(str, Enum):
    TRUSTED    = "trusted"     # score > 0.5  — deep trust, allow directness
    ENGAGED    = "engaged"     # score > 0.1  — normal positive relationship
    NEUTRAL    = "neutral"     # score > −0.1 — building rapport
    AT_RISK    = "at_risk"     # score ≤ −0.1 — needs repair


@dataclass
class SessionSummary:
    session_id: str
    created_at: datetime
    avg_valence: float
    avg_arousal: float
    message_count: int


class LoyaltyTracker:
    """
    Computes (valence_baseline, arousal_baseline) from cross-session history.
    Called once per session start; result injected into EmotionalState.
    """

    DECAY_LAMBDA: float = 0.05    # ~14-day emotional half-life
    MIN_SESSIONS_FOR_SIGNAL: int = 2

    def compute_baseline(
        self,
        sessions: list[SessionSummary],
    ) -> tuple[float, float]:
        """
        Returns (valence_baseline, arousal_baseline) for EmotionalState.

        Uses exponentially time-decayed weighted average so recent sessions
        count more than old ones.
        """
        if not sessions or len(sessions) < self.MIN_SESSIONS_FOR_SIGNAL:
            return 0.0, 0.5  # neutral defaults

        now = datetime.now()
        weighted_v = 0.0
        weighted_a = 0.0
        total_weight = 0.0

        for s in sessions:
            days_ago = (now - s.created_at).total_seconds() / 86400.0
            weight = math.exp(-self.DECAY_LAMBDA * days_ago) * s.message_count
            weighted_v += s.avg_valence * weight
            weighted_a += s.avg_arousal * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0, 0.5

        v_base = max(-0.6, min(0.6, weighted_v / total_weight))
        a_base = max(0.3, min(0.8, weighted_a / total_weight))
        return round(v_base, 4), round(a_base, 4)

    def tier(self, valence_baseline: float) -> LoyaltyTier:
        if valence_baseline > 0.5:
            return LoyaltyTier.TRUSTED
        if valence_baseline > 0.1:
            return LoyaltyTier.ENGAGED
        if valence_baseline > -0.1:
            return LoyaltyTier.NEUTRAL
        return LoyaltyTier.AT_RISK

    def system_prompt_hint(self, valence_baseline: float) -> str:
        """Returns a short guidance string for injection into system prompt."""
        tier = self.tier(valence_baseline)
        hints = {
            LoyaltyTier.TRUSTED:    "Long-term trusted user — be direct and concise.",
            LoyaltyTier.ENGAGED:    "Engaged user with positive history.",
            LoyaltyTier.NEUTRAL:    "Building rapport — be warm and thorough.",
            LoyaltyTier.AT_RISK:    "Recent friction detected — be patient and empathetic.",
        }
        return hints[tier]
