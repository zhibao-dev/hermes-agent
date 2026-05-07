"""
Active Inference Engine

Scores candidate tool calls by Expected Free Energy (EFE):

  G(tool) = -(Epistemic Value + Pragmatic Value)

  Epistemic Value = expected entropy reduction (information gain)
  Pragmatic Value = expected progress toward goal / value alignment

The agent selects tools that minimise G (maximise expected free energy
reduction), naturally balancing exploration (epistemic) and exploitation
(pragmatic).  Arousal modulates the exploration/exploitation weight:
high arousal → favour epistemic (explore); low arousal → favour pragmatic.

LLM is used as the oracle for outcome prediction and alignment scoring.
Falls back to heuristics when no LLM caller is provided.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..affective.emotional_state import EmotionalState
from ..beliefs.belief_state import BeliefState


@dataclass
class ToolScore:
    tool_name: str
    tool_args: dict
    epistemic_value: float    # expected entropy reduction (0 → 1)
    pragmatic_value: float    # expected goal progress (0 → 1)
    efe: float                # -(ep_weight*ep + pr_weight*pr), lower = preferred
    reasoning: str = ""


class ActiveInferenceEngine:
    """
    Wraps LLM tool-call candidates with EFE scoring.

    Does NOT replace the LLM's tool selection — it re-ranks the LLM's
    proposals using principled expected free energy scores.
    """

    def __init__(
        self,
        belief_state: BeliefState,
        emotional_state: EmotionalState,
    ):
        self.beliefs = belief_state
        self.emotion = emotional_state

    # ── Public API ────────────────────────────────────────────────────────

    def score_and_rank(
        self,
        candidates: list[dict],
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> list[ToolScore]:
        """
        Given a list of LLM-proposed tool calls, return them ranked by EFE.

        candidates: [{"name": "tool_name", "arguments": {...}}, ...]
        """
        current_entropy = self.beliefs.shannon_entropy()
        ep_weight, pr_weight = self._exploration_weights()

        scores: list[ToolScore] = []
        for tool in candidates:
            ep = self._estimate_epistemic(tool, current_entropy, llm_caller)
            pr = self._estimate_pragmatic(tool, llm_caller)
            efe = -(ep_weight * ep + pr_weight * pr)
            scores.append(ToolScore(
                tool_name=tool.get("name", "unknown"),
                tool_args=tool.get("arguments", {}),
                epistemic_value=ep,
                pragmatic_value=pr,
                efe=efe,
                reasoning=(
                    f"ep={ep:.2f}×{ep_weight:.2f} "
                    f"pr={pr:.2f}×{pr_weight:.2f} "
                    f"arousal={self.emotion.arousal:.2f}"
                ),
            ))

        return sorted(scores, key=lambda s: s.efe)

    def top_tool(
        self,
        candidates: list[dict],
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> Optional[dict]:
        """Return the single best tool call by EFE."""
        ranked = self.score_and_rank(candidates, llm_caller)
        if not ranked:
            return None
        best = ranked[0]
        return {"name": best.tool_name, "arguments": best.tool_args}

    # ── Weight computation ────────────────────────────────────────────────

    def _exploration_weights(self) -> tuple[float, float]:
        """
        High arousal → lean epistemic (explore, reduce uncertainty).
        Low arousal  → lean pragmatic (exploit, make progress).
        Returns (ep_weight, pr_weight) summing to 1.0.
        """
        arousal = self.emotion.arousal
        ep_weight = 0.3 + 0.4 * arousal   # [0.30, 0.70]
        pr_weight = 1.0 - ep_weight
        return ep_weight, pr_weight

    # ── Value estimation ──────────────────────────────────────────────────

    def _estimate_epistemic(
        self,
        tool: dict,
        current_entropy: float,
        llm_caller: Optional[Callable[[str], str]],
    ) -> float:
        """
        Estimate expected information gain for this tool call.
        Returns value in [0, 1] where 1 = maximum entropy reduction.
        """
        if not self.beliefs.hypotheses:
            return 0.5  # no beliefs to update → neutral

        if llm_caller is not None:
            return self._llm_epistemic(tool, current_entropy, llm_caller)
        return self._heuristic_epistemic(tool, current_entropy)

    def _estimate_pragmatic(
        self,
        tool: dict,
        llm_caller: Optional[Callable[[str], str]],
    ) -> float:
        """
        Estimate expected progress toward the task goal.
        Returns value in [0, 1].
        """
        if llm_caller is not None:
            return self._llm_pragmatic(tool, llm_caller)
        return self._heuristic_pragmatic(tool)

    # ── LLM-based estimation ──────────────────────────────────────────────

    def _llm_epistemic(
        self,
        tool: dict,
        current_entropy: float,
        llm_caller: Callable[[str], str],
    ) -> float:
        beliefs_summary = [
            {"desc": h.description, "prob": round(h.probability, 3)}
            for h in self.beliefs.hypotheses
        ]
        prompt = f"""Rate the epistemic value (information gain) of calling this tool.

Current beliefs (entropy = {current_entropy:.2f} bits):
{json.dumps(beliefs_summary, ensure_ascii=False)}

Proposed tool call:
{json.dumps(tool, ensure_ascii=False)}

Return ONLY: {{"epistemic_value": <float 0.0-1.0>}}
1.0 = this tool will almost certainly resolve all uncertainty.
0.0 = this tool provides no new information about the hypotheses."""

        try:
            raw = llm_caller(prompt)
            data = json.loads(raw)
            return max(0.0, min(1.0, float(data["epistemic_value"])))
        except Exception:
            return self._heuristic_epistemic(tool, current_entropy)

    def _llm_pragmatic(
        self,
        tool: dict,
        llm_caller: Callable[[str], str],
    ) -> float:
        prompt = f"""Rate how much this tool call advances the current task goal.

Task: {self.beliefs.task or 'not specified'}
Open questions: {self.beliefs.open_questions[:3]}

Proposed tool call:
{json.dumps(tool, ensure_ascii=False)}

Return ONLY: {{"pragmatic_value": <float 0.0-1.0>}}
1.0 = directly solves the task.  0.0 = no progress toward the goal."""

        try:
            raw = llm_caller(prompt)
            data = json.loads(raw)
            return max(0.0, min(1.0, float(data["pragmatic_value"])))
        except Exception:
            return self._heuristic_pragmatic(tool)

    # ── Heuristic fallbacks ───────────────────────────────────────────────

    # Tools that tend to produce high epistemic value (resolve uncertainty)
    HIGH_EPISTEMIC_TOOLS = {
        "read_file", "bash", "search", "grep", "list_directory",
        "web_search", "fetch_url", "sql_query",
    }
    # Tools that tend to produce high pragmatic value (make progress)
    HIGH_PRAGMATIC_TOOLS = {
        "write_file", "edit_file", "bash", "send_message",
        "create", "update", "delete", "execute",
    }

    def _heuristic_epistemic(self, tool: dict, current_entropy: float) -> float:
        name = tool.get("name", "").lower()
        base = 0.6 if any(t in name for t in self.HIGH_EPISTEMIC_TOOLS) else 0.3
        # Scale with current entropy — more uncertainty → reads matter more
        entropy_factor = min(1.0, current_entropy / 3.0)
        return round(base * (0.5 + 0.5 * entropy_factor), 3)

    def _heuristic_pragmatic(self, tool: dict) -> float:
        name = tool.get("name", "").lower()
        return 0.65 if any(t in name for t in self.HIGH_PRAGMATIC_TOOLS) else 0.35
