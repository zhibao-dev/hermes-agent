"""
Bayesian Belief State

Maintains an explicit probability distribution over task-relevant hypotheses,
replacing the implicit "everything is in the context window" model.

Key operations:
  - bayesian_update(): update hypothesis probabilities given a new observation
  - shannon_entropy(): measure current uncertainty (bits)
  - to_context_injection(): compact string for system prompt (<150 tokens)

The LLM acts as the likelihood estimator during bayesian_update(), called
with a structured prompt that returns JSON probabilities.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Optional


@dataclass
class Hypothesis:
    """A candidate explanation for the current task state."""
    description: str
    probability: float                          # 0.0 → 1.0
    evidence_for: list[str] = field(default_factory=list)
    evidence_against: list[str] = field(default_factory=list)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    def __post_init__(self):
        self.probability = max(0.0, min(1.0, self.probability))


@dataclass
class KnownFact:
    """A fact believed with a confidence level, attributed to a source tool."""
    key: str
    value: str
    confidence: float   # 0.0 → 1.0
    source: str         # tool name that produced this fact
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )


class BeliefState:
    """
    Task-level Bayesian belief state.

    Lives alongside the conversation context but is managed separately to
    avoid inflating the token budget.  Injected into the system prompt as
    a compact summary only when uncertainty is above a configurable threshold.
    """

    def __init__(self, task_description: str = ""):
        self.task: str = task_description
        self.hypotheses: list[Hypothesis] = []
        self.known_facts: dict[str, KnownFact] = {}
        self.open_questions: list[str] = []
        self._update_count: int = 0

    # ── Entropy ───────────────────────────────────────────────────────────

    def shannon_entropy(self) -> float:
        """Current uncertainty over hypotheses (bits). 0 = certain, ∞ = maximal."""
        probs = [h.probability for h in self.hypotheses if h.probability > 0]
        if not probs:
            return 1.0  # unknown → treat as high uncertainty
        total = sum(probs)
        normed = [p / total for p in probs]
        return -sum(p * math.log2(p) for p in normed if p > 0)

    # ── Hypothesis management ─────────────────────────────────────────────

    def add_hypothesis(self, description: str, probability: float) -> None:
        self.hypotheses.append(Hypothesis(description=description, probability=probability))
        self._normalize()

    def set_hypotheses(self, pairs: list[tuple[str, float]]) -> None:
        self.hypotheses = [
            Hypothesis(description=d, probability=p) for d, p in pairs
        ]
        self._normalize()

    def _normalize(self) -> None:
        total = sum(h.probability for h in self.hypotheses)
        if total > 0:
            for h in self.hypotheses:
                h.probability /= total

    @property
    def most_likely(self) -> Optional[Hypothesis]:
        return max(self.hypotheses, key=lambda h: h.probability) if self.hypotheses else None

    # ── Bayesian update ───────────────────────────────────────────────────

    def bayesian_update(
        self,
        observation: str,
        tool_name: str,
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Update hypothesis probabilities given a new observation.

        If llm_caller is provided, uses LLM to estimate likelihoods.
        Otherwise falls back to heuristic keyword matching.
        """
        if not self.hypotheses:
            return

        self._update_count += 1

        if llm_caller is not None:
            self._llm_update(observation, tool_name, llm_caller)
        else:
            self._heuristic_update(observation)

        # Update known facts from high-confidence observations
        self._extract_facts(observation, tool_name)

    def _llm_update(
        self,
        observation: str,
        tool_name: str,
        llm_caller: Callable[[str], str],
    ) -> None:
        prompt = self._build_update_prompt(observation, tool_name)
        try:
            raw = llm_caller(prompt)
            data = json.loads(raw)
            new_probs = data.get("probabilities", [])
            if len(new_probs) == len(self.hypotheses):
                for h, p in zip(self.hypotheses, new_probs):
                    h.probability = max(1e-6, float(p))
                self._normalize()
                # Record evidence
                for h in self.hypotheses:
                    if h.probability > 0.4:
                        h.evidence_for.append(f"[{tool_name}] {observation[:80]}")
                    elif h.probability < 0.1:
                        h.evidence_against.append(f"[{tool_name}] {observation[:80]}")
        except (json.JSONDecodeError, ValueError, KeyError):
            # Fallback gracefully
            self._heuristic_update(observation)

    def _heuristic_update(self, observation: str) -> None:
        """
        Simple keyword-overlap likelihood heuristic when no LLM is available.
        Each hypothesis probability is boosted proportionally to keyword overlap
        between the observation and the hypothesis description.
        """
        obs_words = set(observation.lower().split())
        for h in self.hypotheses:
            hyp_words = set(h.description.lower().split())
            overlap = len(obs_words & hyp_words) / max(len(hyp_words), 1)
            # Small multiplicative update (likelihood ratio approximation)
            h.probability *= (1.0 + overlap * 0.5)
        self._normalize()

    def _build_update_prompt(self, observation: str, tool_name: str) -> str:
        current = [
            {"description": h.description, "current_probability": round(h.probability, 4)}
            for h in self.hypotheses
        ]
        return f"""You are a Bayesian reasoning assistant. Given the current belief state and a new observation, return updated probabilities.

Current hypotheses (must sum to 1.0):
{json.dumps(current, indent=2, ensure_ascii=False)}

New observation from tool '{tool_name}':
{observation[:500]}

Return ONLY a JSON object: {{"probabilities": [<float>, ...]}}
The array must have exactly {len(self.hypotheses)} values that sum to 1.0.
Reason: if the observation supports a hypothesis, increase its probability; if it contradicts it, decrease it."""

    def _extract_facts(self, observation: str, tool_name: str) -> None:
        """Heuristically mark short, factual observations as known facts."""
        if len(observation) < 200 and "\n" not in observation:
            key = f"{tool_name}:{self._update_count}"
            self.known_facts[key] = KnownFact(
                key=key,
                value=observation.strip(),
                confidence=0.9,
                source=tool_name,
            )

    # ── Context injection ─────────────────────────────────────────────────

    def to_context_injection(self, max_hypotheses: int = 3) -> str:
        """
        Compact summary for system prompt injection.
        Only called when entropy > threshold (controlled by BorgeAgent).
        """
        if not self.hypotheses and not self.open_questions:
            return ""

        lines = ["[Belief State]"]
        lines.append(f"Uncertainty: {self.shannon_entropy():.2f} bits")

        top = sorted(self.hypotheses, key=lambda h: h.probability, reverse=True)
        for h in top[:max_hypotheses]:
            lines.append(f"  • {h.description} ({h.probability * 100:.0f}%)")

        if self.open_questions:
            qs = "; ".join(self.open_questions[:2])
            lines.append(f"Open: {qs}")

        return "\n".join(lines)

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "entropy": round(self.shannon_entropy(), 4),
            "hypotheses": [
                {
                    "description": h.description,
                    "probability": round(h.probability, 4),
                    "evidence_for": h.evidence_for[-3:],
                    "evidence_against": h.evidence_against[-3:],
                }
                for h in self.hypotheses
            ],
            "known_facts": {
                k: {"value": v.value, "confidence": v.confidence, "source": v.source}
                for k, v in self.known_facts.items()
            },
            "open_questions": self.open_questions,
            "update_count": self._update_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeliefState":
        bs = cls(task_description=d.get("task", ""))
        for h in d.get("hypotheses", []):
            bs.hypotheses.append(
                Hypothesis(
                    description=h["description"],
                    probability=h["probability"],
                    evidence_for=h.get("evidence_for", []),
                    evidence_against=h.get("evidence_against", []),
                )
            )
        for k, v in d.get("known_facts", {}).items():
            bs.known_facts[k] = KnownFact(
                key=k, value=v["value"],
                confidence=v["confidence"], source=v["source"]
            )
        bs.open_questions = d.get("open_questions", [])
        bs._update_count = d.get("update_count", 0)
        return bs

    def __repr__(self) -> str:
        return (
            f"BeliefState(entropy={self.shannon_entropy():.2f}bits, "
            f"hypotheses={len(self.hypotheses)}, facts={len(self.known_facts)})"
        )
