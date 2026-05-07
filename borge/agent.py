"""
BorgeAgent — Next-Generation Agent Core

Extends Hermes AIAgent with:
  - Affective state tracking (Russell circumplex)
  - Bayesian belief state
  - Active inference tool scoring (EFE)
  - Cognitive memory pipeline hooks
  - Extended free energy monitoring (MetaAgent)
  - Value system integration

Integration strategy: MINIMAL INVASION.
All new logic runs through pre/post hooks added to the existing
AIAgent callback architecture.  No core Hermes files are modified.

Usage:
    from borge.agent import BorgeAgent
    agent = BorgeAgent(config, session_id=sid, source="cli")
    agent.run("help me debug this")
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable, Optional

from .affective.emotional_state import AgentMode, EmotionalState
from .affective.loyalty_tracker import LoyaltyTracker, SessionSummary
from .affective.signal_extractor import EmotionalSignalExtractor
from .beliefs.belief_state import BeliefState
from .inference.active_inference import ActiveInferenceEngine
from .memory.consolidation import MemoryConsolidationPipeline
from .memory.forgetting import ForgettingEngine
from .memory.knowledge_graph import KnowledgeGraph
from .meta.free_energy import ExtendedFreeEnergy
from .meta.meta_agent import MetaAgent
from .skill_evolution import SkillEvolutionEngine
from .values.soul_parser import parse_soul_frontmatter
from .values.value_system import ValueSystem

log = logging.getLogger(__name__)


class BorgeAgent:
    """
    Borge cognitive layer — wraps a Hermes AIAgent instance and injects
    all Borge capabilities via its public callback/hook interface.

    Designed as a decorator / wrapper rather than a subclass so it stays
    decoupled from Hermes version changes.
    """

    def __init__(
        self,
        hermes_agent,                          # AIAgent instance from run_agent.py
        db_path: Optional[str] = None,
        soul_path: Optional[str] = None,
        config: Optional[dict] = None,
    ):
        self._agent = hermes_agent
        self._config = config or {}
        self._db_path = db_path or self._resolve_db_path()

        # ── Core cognitive state ──────────────────────────────────────────
        self.values: ValueSystem = parse_soul_frontmatter(
            soul_path or self._resolve_soul_path()
        )
        self.emotion: EmotionalState = EmotionalState(
            valence_baseline=self.values.emotional_defaults.valence_baseline,
            arousal_baseline=self.values.emotional_defaults.arousal_baseline,
            tau_valence=self.values.emotional_defaults.tau_valence,
            tau_arousal=self.values.emotional_defaults.tau_arousal,
            frustrated_threshold=self.values.emotional_defaults.frustrated_threshold,
            excited_threshold=self.values.emotional_defaults.excited_threshold,
        )
        self.beliefs: BeliefState = BeliefState()

        # ── Engines ──────────────────────────────────────────────────────
        self._signal_extractor = EmotionalSignalExtractor()
        self._loyalty_tracker  = LoyaltyTracker()
        self._meta             = MetaAgent(
            free_energy_fn=ExtendedFreeEnergy(),
            entropy_injection_threshold=self._cfg("beliefs.entropy_injection_threshold", 0.5),
        )
        self._afe = ActiveInferenceEngine(self.beliefs, self.emotion)

        # ── Memory infrastructure ─────────────────────────────────────────
        self._kg = KnowledgeGraph(self._db_path) if self._cfg("memory.knowledge_graph.enabled", True) else None
        self._forgetting = ForgettingEngine(
            prune_threshold=self._cfg("memory.forgetting.prune_threshold", 2.0),
        )
        self._consolidation = MemoryConsolidationPipeline(
            db_path=self._db_path,
            knowledge_graph=self._kg,
            llm_caller=None,       # set after agent initialises
            forgetting_engine=self._forgetting,
        )
        self._skill_evolution = SkillEvolutionEngine(self._db_path)

        # ── Session state ─────────────────────────────────────────────────
        self._emotional_history: list[tuple[float, float]] = []
        self._turn_count: int = 0

        log.info("[BorgeAgent] Initialised")

    # ── Session start ─────────────────────────────────────────────────────

    def on_session_start(self, user_id: Optional[str] = None) -> None:
        """
        Call at the beginning of a new session.
        Sets up loyalty baseline and resets per-session state.
        """
        if user_id and self._cfg("affective.loyalty.enabled", True):
            sessions = self._fetch_past_sessions(user_id)
            v_base, a_base = self._loyalty_tracker.compute_baseline(sessions)
            self.emotion.set_baseline(v_base, a_base)
            tier = self._loyalty_tracker.tier(v_base)
            log.info(f"[BorgeAgent] Loyalty: {tier.value} (V_base={v_base:.2f})")

        self._emotional_history.clear()
        self._turn_count = 0
        self.beliefs = BeliefState()
        self._meta.reset()

    # ── Pre-turn hook ─────────────────────────────────────────────────────

    def pre_turn(
        self,
        user_message: str,
        conversation_history: list[dict],
    ) -> str:
        """
        Called before LLM inference each turn.
        Returns a context injection string to prepend to system prompt.
        """
        self._turn_count += 1

        # 1. Extract emotional signal
        if self._cfg("affective.enabled", True):
            dv, da = self._signal_extractor.extract(user_message, conversation_history)
            self.emotion.update(dv, da)
            self._emotional_history.append((self.emotion.valence, self.emotion.arousal))
            log.debug(f"[Emotion] {self.emotion}")

        # 2. Update belief task description from first user message
        if self._turn_count == 1:
            self.beliefs.task = user_message[:200]

        # 3. MetaAgent tick
        loyalty_hint = ""
        if self._cfg("affective.loyalty.enabled", True):
            tier = self._loyalty_tracker.tier(self.emotion.valence_baseline)
            loyalty_hint = self._loyalty_tracker.system_prompt_hint(
                self.emotion.valence_baseline
            )

        signal = self._meta.tick(
            self.beliefs, self.emotion, self.values,
            loyalty_hint=loyalty_hint,
        )

        # 4. Log mode change
        mode = signal.suggested_mode
        if mode != AgentMode.NORMAL:
            log.info(f"[BorgeAgent] Mode: {mode.value}")

        return signal.context_injection

    # ── Post-tool hook ────────────────────────────────────────────────────

    def post_tool(
        self,
        tool_name: str,
        tool_result: str,
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> None:
        """
        Called after each tool execution.
        Updates belief state via Bayesian update.
        """
        if self._cfg("beliefs.enabled", True) and self.beliefs.hypotheses:
            self.beliefs.bayesian_update(tool_result, tool_name, llm_caller)
            log.debug(f"[Belief] entropy={self.beliefs.shannon_entropy():.2f}bits")

        # Update value satisfaction
        self.values.update_satisfaction(tool_result)

    # ── Tool scoring ──────────────────────────────────────────────────────

    def score_tool_candidates(
        self,
        candidates: list[dict],
        llm_caller: Optional[Callable[[str], str]] = None,
    ) -> list[dict]:
        """
        Re-rank LLM tool candidates by Expected Free Energy.
        Returns re-ordered list (best first).
        """
        if not self._cfg("active_inference.enabled", True) or not candidates:
            return candidates

        self._afe.beliefs = self.beliefs
        self._afe.emotion = self.emotion
        scored = self._afe.score_and_rank(candidates, llm_caller)

        log.debug(
            f"[AIF] Tool scores: "
            + ", ".join(f"{s.tool_name}={s.efe:.3f}" for s in scored[:3])
        )
        # Return in EFE order, keeping original dict structure
        scored_names = [s.tool_name for s in scored]
        reordered = sorted(
            candidates,
            key=lambda c: scored_names.index(c.get("name", ""))
                          if c.get("name") in scored_names else 999,
        )
        return reordered

    # ── Session end ───────────────────────────────────────────────────────

    def on_session_end(
        self,
        session_id: str,
        messages: list[dict],
    ) -> None:
        """
        Called when a session closes.
        Triggers memory consolidation pipeline.
        """
        if not self._cfg("memory.consolidation.enabled", True):
            return

        log.info(f"[BorgeAgent] Running consolidation for session {session_id}")
        report = self._consolidation.run(
            session_id=session_id,
            messages=messages,
            emotional_history=self._emotional_history,
        )
        log.info(
            f"[Consolidation] entities={report.entities_extracted} "
            f"relations={report.relations_added} "
            f"skills={report.skill_candidates} "
            f"forgotten={report.entries_forgotten}"
        )

    # ── Skill tracking ────────────────────────────────────────────────────

    def record_skill(
        self,
        skill_name: str,
        success: bool,
        f_before: float = 0.5,
        f_after: float = 0.5,
    ) -> None:
        f_reduction = max(0.0, f_before - f_after)
        self._skill_evolution.record_invocation(skill_name, success, f_reduction)

    def skill_health_report(self) -> dict:
        return {
            "prune_candidates": self._skill_evolution.prune_candidates(),
            "generalise_candidates": self._skill_evolution.generalise_candidates(),
        }

    # ── System prompt injection ───────────────────────────────────────────

    def build_system_prompt_suffix(self) -> str:
        """
        Returns a string to append to the Hermes system prompt.
        Kept minimal to avoid token waste.
        """
        signal = self._meta.tick(self.beliefs, self.emotion, self.values)
        return signal.context_injection

    # ── Internal helpers ──────────────────────────────────────────────────

    def _cfg(self, key: str, default: Any = None) -> Any:
        """Dot-notation config lookup."""
        parts = key.split(".")
        node = self._config
        for p in parts:
            if not isinstance(node, dict):
                return default
            node = node.get(p, default)
        return node if node is not None else default

    def _resolve_db_path(self) -> str:
        home = os.path.expanduser("~/.hermes")
        os.makedirs(home, exist_ok=True)
        return os.path.join(home, "hermes.db")

    def _resolve_soul_path(self) -> str:
        candidates = [
            os.path.join(os.getcwd(), "SOUL.md"),
            os.path.expanduser("~/.hermes/SOUL.md"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[-1]

    def _fetch_past_sessions(self, user_id: str) -> list[SessionSummary]:
        """Fetch session summaries from Hermes SQLite for loyalty computation."""
        import sqlite3
        summaries = []
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT id, created_at,
                              COALESCE(emotional_valence, 0.0) as avg_valence,
                              COALESCE(emotional_arousal, 0.5) as avg_arousal,
                              COALESCE(message_count, 1) as message_count
                       FROM sessions
                       WHERE source_user_id = ?
                       ORDER BY created_at DESC LIMIT 50""",
                    (user_id,),
                ).fetchall()
                from datetime import datetime
                for r in rows:
                    try:
                        summaries.append(SessionSummary(
                            session_id=r["id"],
                            created_at=datetime.fromisoformat(r["created_at"]),
                            avg_valence=r["avg_valence"],
                            avg_arousal=r["avg_arousal"],
                            message_count=r["message_count"],
                        ))
                    except Exception:
                        pass
        except Exception as e:
            log.debug(f"[BorgeAgent] Could not fetch sessions: {e}")
        return summaries
