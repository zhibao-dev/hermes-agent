"""
Affective State Engine — Russell (1980) two-dimensional circumplex model.

Valence  (V): −1.0 (negative) → +1.0 (positive)
Arousal  (A):  0.0 (calm)     →  1.0 (high activation)

Emotional state modulates the precision matrix used across all cognitive
processes: perception, memory encoding, belief updating, tool selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class EmotionalQuadrant(str, Enum):
    EXCITED    = "excited"     # +V +A  兴奋/投入
    CONTENT    = "content"     # +V −A  平静/满足
    FRUSTRATED = "frustrated"  # −V +A  挫败/紧张
    DISENGAGED = "disengaged"  # −V −A  冷漠/疲惫


class AgentMode(str, Enum):
    NORMAL     = "normal"      # 默认模式
    SIMPLIFY   = "simplify"    # 简化输出，直击要点（frustrated）
    EXPLORE    = "explore"     # 深入探索，丰富输出（excited）
    REACTIVATE = "reactivate"  # 换角度重新激活（disengaged）


@dataclass
class EmotionalState:
    """
    Runtime emotional state, updated each conversation turn.

    Uses exponential moving average with separate time constants for
    valence (slow, relational) and arousal (fast, reactive).
    """

    valence: float = 0.0   # −1.0 → +1.0
    arousal: float = 0.5   # 0.0  → 1.0

    # Longitudinal baseline injected by LoyaltyTracker
    valence_baseline: float = 0.0
    arousal_baseline: float = 0.5

    # Inertia time constants (turns)
    tau_valence: float = 5.0
    tau_arousal: float = 2.0

    # Behaviour thresholds
    frustrated_threshold: float = -0.4
    excited_threshold: float = 0.7

    # History for trend analysis (last N turns)
    _history: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _max_history: int = field(default=20, repr=False)

    # ── Core update ──────────────────────────────────────────────────────

    def update(self, delta_v: float, delta_a: float) -> None:
        """
        Exponential moving average update toward (baseline + delta),
        keeping both coordinates within legal bounds.
        """
        alpha_v = 1.0 / max(1.0, self.tau_valence)
        alpha_a = 1.0 / max(1.0, self.tau_arousal)

        target_v = self.valence_baseline + delta_v
        target_a = self.arousal_baseline + delta_a

        self.valence = (1.0 - alpha_v) * self.valence + alpha_v * target_v
        self.arousal = (1.0 - alpha_a) * self.arousal + alpha_a * target_a

        self.valence = max(-1.0, min(1.0, self.valence))
        self.arousal = max(0.0, min(1.0, self.arousal))

        self._history.append((self.valence, self.arousal))
        if len(self._history) > self._max_history:
            self._history.pop(0)

    def set_baseline(self, valence_baseline: float, arousal_baseline: float) -> None:
        self.valence_baseline = max(-1.0, min(1.0, valence_baseline))
        self.arousal_baseline = max(0.0, min(1.0, arousal_baseline))

    # ── Derived properties ───────────────────────────────────────────────

    @property
    def quadrant(self) -> EmotionalQuadrant:
        if self.valence >= 0 and self.arousal >= 0.5:
            return EmotionalQuadrant.EXCITED
        if self.valence >= 0 and self.arousal < 0.5:
            return EmotionalQuadrant.CONTENT
        if self.valence < 0 and self.arousal >= 0.5:
            return EmotionalQuadrant.FRUSTRATED
        return EmotionalQuadrant.DISENGAGED

    @property
    def precision(self) -> float:
        """
        Scalar precision value used to weight epistemic free energy.

        High arousal → focused, high-precision processing.
        Negative valence → slight precision boost for threat detection.
        Range: [0.1, 1.0]
        """
        base = 0.5 + 0.5 * self.arousal
        valence_bias = 0.1 * self.valence
        return max(0.1, min(1.0, base + valence_bias))

    @property
    def suggested_mode(self) -> AgentMode:
        if self.valence <= self.frustrated_threshold:
            return AgentMode.SIMPLIFY
        if self.valence >= self.excited_threshold and self.arousal >= 0.6:
            return AgentMode.EXPLORE
        if self.valence < 0 and self.arousal < 0.4:
            return AgentMode.REACTIVATE
        return AgentMode.NORMAL

    @property
    def emotional_significance(self) -> float:
        """
        Memory encoding weight: |valence| × arousal.
        Emotionally intense moments get deeper encoding (Craik & Lockhart).
        Range: [0.0, 1.0]
        """
        return abs(self.valence) * self.arousal

    @property
    def trend(self) -> Optional[str]:
        """Valence trend over last 5 turns: 'improving', 'declining', 'stable'."""
        if len(self._history) < 5:
            return None
        recent = [v for v, _ in self._history[-5:]]
        delta = recent[-1] - recent[0]
        if delta > 0.1:
            return "improving"
        if delta < -0.1:
            return "declining"
        return "stable"

    # ── Serialisation ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "valence": round(self.valence, 4),
            "arousal": round(self.arousal, 4),
            "valence_baseline": round(self.valence_baseline, 4),
            "arousal_baseline": round(self.arousal_baseline, 4),
            "quadrant": self.quadrant.value,
            "precision": round(self.precision, 4),
            "significance": round(self.emotional_significance, 4),
            "mode": self.suggested_mode.value,
            "trend": self.trend,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EmotionalState":
        state = cls(
            valence=d.get("valence", 0.0),
            arousal=d.get("arousal", 0.5),
            valence_baseline=d.get("valence_baseline", 0.0),
            arousal_baseline=d.get("arousal_baseline", 0.5),
        )
        return state

    def to_context_summary(self) -> str:
        """Compact injection for system prompt (<40 tokens)."""
        q = self.quadrant.value
        v = f"{self.valence:+.2f}"
        a = f"{self.arousal:.2f}"
        return f"[Affective: {q} V={v} A={a} precision={self.precision:.2f}]"

    def __repr__(self) -> str:
        return (
            f"EmotionalState(V={self.valence:+.3f}, A={self.arousal:.3f}, "
            f"quadrant={self.quadrant.value}, mode={self.suggested_mode.value})"
        )
