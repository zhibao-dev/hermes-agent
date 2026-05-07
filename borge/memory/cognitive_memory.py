"""
Cognitive Memory Architecture

Three-layer memory system based on Tulving (1972, 1983):

  Episodic Memory   — What happened (conversation history, SQLite sessions)
  Semantic Memory   — What we know (knowledge graph, entities, relations)
  Procedural Memory — How to do things (skills with fitness scores)

Each memory entry carries:
  - Emotional tags (V, A at encoding time) for mood-congruent retrieval
  - Encoding depth (Craik & Lockhart, 1972) for consolidation priority
  - Forget score for active forgetting (Ebbinghaus decay)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Optional


class EncodingDepth(IntEnum):
    """
    Craik & Lockhart levels of processing.
    Higher depth → more durable memory trace → priority consolidation.
    """
    SHALLOW    = 1   # surface form, verbatim storage (default)
    SEMANTIC   = 2   # entities and relations extracted
    SCHEMATIC  = 3   # integrated into existing knowledge graph schema
    META       = 4   # abstracted into reusable patterns / skill candidates


class MemoryType(str):
    EPISODIC   = "episodic"
    SEMANTIC   = "semantic"
    PROCEDURAL = "procedural"


@dataclass
class MemoryEntry:
    """
    Extended memory entry adding emotional, depth, and forgetting fields
    on top of Hermes's existing message storage.
    """

    # ── Core (mirrors Hermes messages table) ─────────────────────────────
    id: str
    session_id: str
    content: str
    role: str                      # user / assistant / tool
    timestamp: datetime = field(default_factory=datetime.now)

    # ── Emotional encoding (Borge) ────────────────────────────────────────
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.5
    emotional_significance: float = 0.0  # |V| × A → determines encoding depth

    # ── Encoding depth ────────────────────────────────────────────────────
    encoding_depth: EncodingDepth = EncodingDepth.SHALLOW

    # ── Knowledge graph links ─────────────────────────────────────────────
    entity_tags: list[str] = field(default_factory=list)
    graph_node_ids: list[str] = field(default_factory=list)

    # ── Active forgetting ─────────────────────────────────────────────────
    retrieval_count: int = 0
    last_retrieved: Optional[datetime] = None
    importance_score: float = 0.5   # updated by consolidation pipeline
    forget_score: float = 0.0       # updated periodically; > threshold → decay

    # ── Memory type ───────────────────────────────────────────────────────
    memory_type: str = MemoryType.EPISODIC

    # ── Computed properties ───────────────────────────────────────────────

    def compute_forget_score(self, now: Optional[datetime] = None) -> float:
        """
        Ebbinghaus-inspired forgetting score.  Higher = candidate for decay.

        Score = recency_decay × usage_penalty × importance_penalty × graph_isolation

        Entries with HIGH importance, MANY retrievals, or MANY graph connections
        are resistant to forgetting.
        """
        if now is None:
            now = datetime.now()

        ref_time = self.last_retrieved or self.timestamp
        days_since = max(0.0, (now - ref_time).total_seconds() / 86400.0)

        recency_decay    = days_since ** 0.7
        usage_penalty    = 1.0 / (1.0 + self.retrieval_count)
        importance_res   = 1.0 / (1.0 + self.importance_score)
        graph_resistance = 1.0 / (1.0 + len(self.graph_node_ids))

        score = recency_decay * usage_penalty * importance_res * graph_resistance
        self.forget_score = round(score, 4)
        return self.forget_score

    def compute_encoding_depth(self) -> EncodingDepth:
        """
        Determine encoding depth from emotional significance.

        Significance  Depth
        ≥ 0.7         META       (highly emotional, pattern-worthy)
        ≥ 0.4         SCHEMATIC  (notable, integrate into KG schema)
        ≥ 0.2         SEMANTIC   (extract entities and relations)
        < 0.2         SHALLOW    (verbatim storage)
        """
        sig = self.emotional_significance
        if sig >= 0.7:
            return EncodingDepth.META
        if sig >= 0.4:
            return EncodingDepth.SCHEMATIC
        if sig >= 0.2:
            return EncodingDepth.SEMANTIC
        return EncodingDepth.SHALLOW

    def record_retrieval(self) -> None:
        self.retrieval_count += 1
        self.last_retrieved = datetime.now()

    # ── Serialisation ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "content": self.content,
            "role": self.role,
            "timestamp": self.timestamp.isoformat(),
            "emotional_valence": self.emotional_valence,
            "emotional_arousal": self.emotional_arousal,
            "emotional_significance": self.emotional_significance,
            "encoding_depth": int(self.encoding_depth),
            "entity_tags": self.entity_tags,
            "graph_node_ids": self.graph_node_ids,
            "retrieval_count": self.retrieval_count,
            "last_retrieved": self.last_retrieved.isoformat() if self.last_retrieved else None,
            "importance_score": self.importance_score,
            "forget_score": self.forget_score,
            "memory_type": self.memory_type,
        }
