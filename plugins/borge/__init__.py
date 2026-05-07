"""Borge cognitive layer — Hermes plugin entry point.

Registers four lifecycle hooks that wire BorgeAgent's cognitive capabilities
into the Hermes agent loop without modifying any core files:

  on_session_start  — initialise per-session cognitive state + loyalty baseline
  pre_llm_call      — inject affective/belief context into each user turn
  post_llm_call     — Bayesian belief update from assistant response
  on_session_end    — run memory consolidation pipeline
"""
from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# session_id → BorgeAgent instance
_sessions: dict[str, Any] = {}

# session_id → last conversation history snapshot (for on_session_end)
_histories: dict[str, list] = {}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        from hermes_constants import get_hermes_home
        import yaml
        config_path = get_hermes_home() / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                all_config = yaml.safe_load(f) or {}
            return all_config.get("borge", {}) or {}
    except Exception:
        pass
    return {}


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------

def _get_or_create(session_id: str) -> Any:
    if session_id not in _sessions:
        try:
            from borge.agent import BorgeAgent
            config = _load_config()
            _sessions[session_id] = BorgeAgent(hermes_agent=None, config=config)
        except Exception as exc:
            logger.warning("[Borge] Failed to create BorgeAgent: %s", exc)
            return None
    return _sessions[session_id]


# ---------------------------------------------------------------------------
# Hook callbacks
# ---------------------------------------------------------------------------

def _on_session_start(
    session_id: str = "",
    model: str = "",
    platform: str = "",
    **kwargs,
) -> None:
    agent = _get_or_create(session_id)
    if agent is None:
        return
    # Use platform as a proxy for user identity (e.g. "cli", "slack:user123")
    user_id = platform if platform and platform not in ("", "cli") else None
    try:
        agent.on_session_start(user_id=user_id)
        logger.debug("[Borge] Session started: %s", session_id)
    except Exception as exc:
        logger.warning("[Borge] on_session_start error: %s", exc)


def _pre_llm_call(
    session_id: str = "",
    user_message: str = "",
    conversation_history: list | None = None,
    is_first_turn: bool = False,
    **kwargs,
) -> str:
    """Return context string to prepend to the current user message."""
    agent = _get_or_create(session_id)
    if agent is None:
        return ""
    try:
        context = agent.pre_turn(
            user_message=user_message or "",
            conversation_history=conversation_history or [],
        )
        return context or ""
    except Exception as exc:
        logger.warning("[Borge] pre_llm_call error: %s", exc)
        return ""


def _post_llm_call(
    session_id: str = "",
    user_message: str = "",
    assistant_response: str = "",
    conversation_history: list | None = None,
    **kwargs,
) -> None:
    agent = _sessions.get(session_id)
    if agent is None:
        return

    # Cache history for on_session_end (which doesn't receive it)
    if conversation_history:
        _histories[session_id] = list(conversation_history)

    try:
        agent.post_tool("assistant_turn", assistant_response or "")
    except Exception as exc:
        logger.warning("[Borge] post_llm_call error: %s", exc)


def _on_session_end(
    session_id: str = "",
    completed: bool = True,
    interrupted: bool = False,
    **kwargs,
) -> None:
    agent = _sessions.pop(session_id, None)
    messages = _histories.pop(session_id, [])

    if agent is None:
        return
    try:
        agent.on_session_end(session_id=session_id, messages=messages)
        logger.debug("[Borge] Session ended: %s (completed=%s)", session_id, completed)
    except Exception as exc:
        logger.warning("[Borge] on_session_end error: %s", exc)


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register Borge cognitive layer with the Hermes plugin system."""
    ctx.register_hook("on_session_start", _on_session_start)
    ctx.register_hook("pre_llm_call", _pre_llm_call)
    ctx.register_hook("post_llm_call", _post_llm_call)
    ctx.register_hook("on_session_end", _on_session_end)
    logger.info("[Borge] Plugin registered — affective/Bayesian/memory layer active")
