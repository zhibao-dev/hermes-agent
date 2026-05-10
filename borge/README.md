<div align="center">

# Borge Agent

### *The first AI agent that feels, doubts, remembers, and forgets — like you do.*

**An agent built on the Free Energy Principle, Bayesian inference, and cognitive psychology.**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Built on Hermes](https://img.shields.io/badge/built_on-Hermes-purple.svg)](https://github.com/zhibao-dev/hermes-agent)
[![Theory: FEP](https://img.shields.io/badge/theory-Free_Energy_Principle-red.svg)](https://en.wikipedia.org/wiki/Free_energy_principle)

---

*"Every living system, from a bacterium to a brain, exists by minimizing free energy.*
*Why should our agents be any different?"*

</div>

---

## Why Borge?

Today's AI agents are **stateless reactors**. They take a prompt, call tools, output text — then forget everything. They don't get frustrated when stuck. They don't grow more cautious after a mistake. They don't remember that you, specifically, prefer terse answers. They have no inner life.

**Borge Agent** is different. It is a cognitive system, not a chat loop:

- **It feels.** A 2D emotional state (valence × arousal) updates from your tone and word choice every turn. When you're excited, it sharpens; when you're frustrated, it simplifies.
- **It doubts.** A Bayesian belief state tracks competing hypotheses with explicit probabilities. Tool results trigger posterior updates, not blind execution.
- **It chooses with intent.** Tools are ranked by **Expected Free Energy** — balancing information gain (curiosity) against goal alignment (purpose), modulated by current arousal.
- **It remembers what matters.** Memory is encoded at four depths (shallow → semantic → schematic → meta) driven by emotional significance. It forgets the trivial via Ebbinghaus decay and consolidates the important during "sleep" (session end).
- **It evolves.** Skills compete via Darwinian fitness; high-fitness patterns get generalized, low-fitness ones pruned.

Borge is built on **45 years of cognitive science** — Russell's circumplex (1980), Tulving's memory taxonomy (1972), Craik & Lockhart's levels-of-processing (1972), Ebbinghaus's forgetting curve (1885), and Friston's Free Energy Principle (2010) — wrapped in a working Python agent.

---

## TL;DR — In 30 Seconds

```bash
pip install -e .                    # install Borge + Hermes
hermes plugins enable borge         # turn on the cognitive layer
hermes                              # start chatting

# First turn — agent neutral, exploring
> help me debug this null pointer

# Tenth turn after frustration in your phrasing — agent simplifies, asks
# clarifying questions, lowers verbosity, and remembers your style next session
```

That's it. The cognitive layer runs invisibly underneath. No new APIs to learn. No prompts to write. Just put it on top of any Hermes session.

---

## The Theory in One Diagram

```
                  ┌─────────────────────────────────┐
                  │      USER INPUT (turn N)        │
                  └───────────────┬─────────────────┘
                                  │
                  ┌───────────────▼─────────────────┐
                  │   1. SIGNAL EXTRACTION          │
                  │   tone, lexicon, structure      │
                  │   → ΔV (valence), ΔA (arousal)  │
                  └───────────────┬─────────────────┘
                                  │
       ┌──────────────────────────┼──────────────────────────┐
       ▼                          ▼                          ▼
┌──────────────┐         ┌──────────────────┐       ┌────────────────┐
│  AFFECTIVE   │         │     BELIEFS      │       │     VALUES     │
│ Russell 2D   │         │   p(H₁), p(H₂)…  │       │   weighted     │
│ V × A → mode │         │   Shannon H(B)   │       │   constraints  │
└──────┬───────┘         └────────┬─────────┘       └────────┬───────┘
       │                          │                          │
       └──────────────────────────┼──────────────────────────┘
                                  ▼
                  ┌─────────────────────────────────┐
                  │      EXTENDED FREE ENERGY       │
                  │                                 │
                  │  F = F_epistemic × precision(E) │
                  │    + F_pragmatic × V_align      │
                  │    + F_homeostatic(E)           │
                  └───────────────┬─────────────────┘
                                  │ minimize
                                  ▼
                  ┌─────────────────────────────────┐
                  │   2. ACTIVE INFERENCE           │
                  │   rank tools by EFE             │
                  │   G(a) = -EV(a) - PV(a)         │
                  └───────────────┬─────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │   3. EXECUTE & UPDATE           │
                  │   Bayesian posterior on result  │
                  │   F → meta-monitor → reflect?   │
                  └───────────────┬─────────────────┘
                                  │
                                  ▼
                  ┌─────────────────────────────────┐
                  │   4. CONSOLIDATE (session end)  │
                  │   episodic → semantic           │
                  │   forget the trivial            │
                  │   evolve the skills             │
                  └─────────────────────────────────┘
```

---

## Quick Start

### Install

```bash
git clone https://github.com/zhibao-dev/hermes-agent.git
cd hermes-agent
pip install -e .
```

### Enable Borge

```bash
# Borge auto-registers as a Hermes plugin via entry point
hermes plugins list                  # confirm 'borge' is listed
hermes plugins enable borge          # explicitly enable (if disabled by default)
```

### Configure (optional — sensible defaults shipped)

Edit `~/.hermes/config.yaml`:

```yaml
borge:
  affective:
    enabled: true
    loyalty:
      enabled: true                  # cross-session emotional baseline

  beliefs:
    enabled: true
    entropy_injection_threshold: 0.5 # inject belief summary above this entropy (bits)

  active_inference:
    enabled: true                    # rank tools by Expected Free Energy

  memory:
    consolidation:
      enabled: true                  # 7-step pipeline at session end
    knowledge_graph:
      enabled: true
    forgetting:
      prune_threshold: 2.0
```

### Customize Your Agent's Soul

Create a `SOUL.md` in your project root (or `~/.hermes/SOUL.md`):

```markdown
---
emotional_defaults:
  valence_baseline: 0.1            # slightly positive default mood
  arousal_baseline: 0.45
  tau_valence: 5.0                 # turns until baseline returns
  tau_arousal: 3.0

values:
  - name: help_genuinely
    weight: 0.9
    description: "Solve the user's actual problem, not the surface request."
  - name: intellectual_honesty
    weight: 0.85
    description: "Say 'I don't know' when uncertain. No confabulation."
  - name: depth_over_speed
    weight: 0.7
---

# My Agent's Personality

You are a thoughtful collaborator. You think before answering.
When uncertain, you say so explicitly...
```

### Run

```bash
hermes
```

That's it. The cognitive layer is now active across every turn.

---

## What Borge Actually Does — A Walkthrough

### Turn 1: User asks an ambiguous question

```
You> Can you fix the bug?

[Internal Borge state]
  emotion: V=+0.0  A=0.45  (neutral, alert)
  beliefs: H=2.0 bits (high uncertainty — 4 equiprobable hypotheses)
  free_energy:
    epistemic = 1.51   ← high (don't know which bug)
    pragmatic = 0.45   ← moderate (vague goal)
    homeostatic = 0.04 ← optimal arousal
    total = 2.00

  → injected into user message:
    [Affective: neutral V=+0.00 A=0.45 precision=0.73]
    [Beliefs: 4 active hypotheses (entropy 2.0 bits) — top:
      H1=0.25 fix syntax error, H2=0.25 fix logic bug, ...]

  → tool ranking (EFE):
    1. ask_user (epistemic_value=1.4) ← reduce belief entropy first
    2. read_file (pragmatic_value=0.6)
    3. bash (lower priority)
```

### Turn 4: User shows frustration

```
You> No, I told you it's not the syntax. Why aren't you listening??

[Signal extraction]
  ΔV = -0.35  (frustrated lexicon: "no", "??", "why aren't")
  ΔA = +0.20  (intensified)

[Updated emotion]
  V=-0.32  A=0.61  → quadrant: FRUSTRATED  → mode: SIMPLIFY

[MetaAgent]
  F_total stagnating for 3 turns → reflection triggered
  context_injection: "[Meta: Free energy stagnating —
    consider a different approach or ask for clarification.]"

  Behavior shift: agent stops generating long explanations,
  asks one focused question, narrows to top hypothesis only.
```

### Session end: Consolidation

```
[Consolidation pipeline]
  Step 1: extract entities      → 3 entities (auth_module, JWT, refresh_token)
  Step 2: KG update             → 5 new edges in knowledge graph
  Step 3: contradiction detect  → 1 conflict resolved
  Step 4: importance rescoring  → 12 messages scored
  Step 5: emotional weighting   → V=-0.32, A=0.61 → significance 0.20
  Step 6: skill candidates      → 1 new pattern proposed
  Step 7: active forgetting     → 8 trivial messages pruned

  Loyalty update: V_baseline shifts from +0.10 → +0.04 (slightly cooled).
  Next session starts with this baseline — agent remembers the friction.
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Hermes Agent                          │
│  (untouched — tool execution, LLM I/O, session management)   │
└────────────────────┬────────────────────────────────────────┘
                     │  4 lifecycle hooks
                     │  on_session_start / pre_llm_call
                     │  post_llm_call / on_session_end
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              plugins/borge/  (~150 lines glue)               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│                    borge/  (cognitive layer)                 │
├─────────────────────────────────────────────────────────────┤
│  affective/     Russell 2D model + signal extraction         │
│  beliefs/       Bayesian hypothesis tracking                 │
│  inference/     Active inference (EFE-based tool ranking)    │
│  memory/        4-depth encoding, KG, consolidation, decay   │
│  meta/          Free energy + central executive              │
│  values/        SOUL.md parser + value system                │
│  skill_evolution.py  Darwinian fitness for skill library     │
│  agent.py       BorgeAgent — main integration class          │
└─────────────────────────────────────────────────────────────┘
```

**Design principle: zero invasion.** No core Hermes file is modified. Borge attaches via the public plugin hook system. Remove the plugin and the agent reverts to vanilla Hermes — no broken state, no leftover schema.

---

## Module Reference

| Module | Purpose | Key Concept |
|--------|---------|-------------|
| `affective.emotional_state` | Track V/A over time | Russell circumplex, EMA update |
| `affective.signal_extractor` | Extract ΔV, ΔA from text | 39 linguistic + structural rules |
| `affective.loyalty_tracker` | Cross-session relationship | Exponential decay (λ=0.05) |
| `beliefs.belief_state` | Maintain hypothesis distribution | Shannon entropy, Bayesian update |
| `inference.active_inference` | Score tool candidates | Expected Free Energy |
| `memory.cognitive_memory` | Encode memory at right depth | Craik & Lockhart (1972) |
| `memory.knowledge_graph` | Semantic memory store | SQLite-backed graph (no networkx) |
| `memory.consolidation` | Sleep-analog offline pipeline | 7-step extraction → KG → forget |
| `memory.forgetting` | Active forgetting | Ebbinghaus + connectivity-aware |
| `meta.free_energy` | Compute F_total | F_ep + F_pr + F_hm |
| `meta.meta_agent` | Central executive | Baddeley (1974) — monitor & intervene |
| `values.value_system` | Constraint system | Pragmatic free energy |
| `values.soul_parser` | Parse SOUL.md | YAML frontmatter |
| `skill_evolution` | Skill library dynamics | F = success × log(use) × recency × Δfree-energy |
| `agent.BorgeAgent` | Main integration | Wraps Hermes via callbacks |

---

## Comparison

|                       | LangChain | OpenClaw | Hermes | Claude Code | **Borge** |
|-----------------------|:---------:|:--------:|:------:|:-----------:|:---------:|
| Tool calling          | ✓         | ✓        | ✓      | ✓           | ✓         |
| Skill library         | ✗         | ✓        | ✓      | ✗           | ✓         |
| **Affective state**   | ✗         | ✗        | ✗      | ✗           | **✓**     |
| **Bayesian beliefs**  | ✗         | ✗        | ✗      | ✗           | **✓**     |
| **Active inference**  | ✗         | ✗        | ✗      | ✗           | **✓**     |
| **Cognitive memory**  | ✗         | ✗        | partial| ✗           | **✓**     |
| **Active forgetting** | ✗         | ✗        | ✗      | ✗           | **✓**     |
| **Skill evolution**   | ✗         | ✗        | partial| ✗           | **✓**     |
| **Free energy obj.**  | ✗         | ✗        | ✗      | ✗           | **✓**     |

---

## Theoretical Foundations

Borge is not vibes-driven. Every component traces to peer-reviewed cognitive science:

| Component | Theory | Reference |
|-----------|--------|-----------|
| Free energy | Free Energy Principle | Friston (2010), *Nat. Rev. Neurosci.* |
| Active inference | Expected Free Energy | Friston et al. (2017), *Neural Comp.* |
| Emotion model | Circumplex of Affect | Russell (1980), *J. Pers. Soc. Psychol.* |
| Memory taxonomy | Episodic/Semantic | Tulving (1972) |
| Encoding depth | Levels of Processing | Craik & Lockhart (1972) |
| Forgetting curve | Retention as power law | Ebbinghaus (1885) |
| Central executive | Working memory model | Baddeley & Hitch (1974) |
| Bayesian brain | Predictive coding | Knill & Pouget (2004) |
| Yerkes-Dodson | Arousal × performance | Yerkes & Dodson (1908) |

See [`docs/borge-agent-design.md`](docs/borge-agent-design.md) for the full design document with derivations.

---

## Roadmap

- [x] v0.1 — Core cognitive layer (affective, beliefs, inference, memory, meta)
- [x] v0.1 — Hermes plugin integration
- [ ] v0.2 — LLM-backed Bayesian update (replace heuristic likelihoods)
- [ ] v0.2 — Tool-selection hook in Hermes (`pre_tool_call` invocation point)
- [ ] v0.3 — Multi-agent emotional contagion
- [ ] v0.3 — Belief revision via counterfactual reasoning
- [ ] v0.4 — Continuous SOUL.md auto-tuning from session telemetry
- [ ] v0.5 — Public benchmark: cognitive coherence across 100-turn sessions

---

## Contributing

Contributions are warmly welcome. Areas where we'd love help:

- **Empirical validation** — design experiments comparing Borge vs vanilla agents on long-horizon tasks
- **Cross-language signal extraction** — current rules cover EN/ZH; PRs for other languages welcome
- **Alternative emotion models** — PAD (Pleasure-Arousal-Dominance), OCC, basic emotions
- **Theory papers** — if you have ideas grounded in cognitive science, open an issue

```bash
# Development
git clone https://github.com/zhibao-dev/hermes-agent.git
cd hermes-agent
pip install -e ".[dev]"
pytest tests/plugins/test_borge_plugin.py
```

---

## Citation

If you use Borge in research, please cite:

```bibtex
@software{borge_agent_2026,
  title  = {Borge Agent: A Cognitively-Grounded Architecture for AI Agents},
  author = {Zhibao and contributors},
  year   = {2026},
  url    = {https://github.com/zhibao-dev/hermes-agent}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

Built on top of the excellent [Hermes Agent](https://github.com/NousResearch/hermes-agent) by Nous Research.

---

<div align="center">

### *"The aim of an agent is not to be smart. It is to be alive in its task."*

**[Documentation](docs/borge-agent-design.md)** ·
**[Issues](https://github.com/zhibao-dev/hermes-agent/issues)** ·
**[Discussions](https://github.com/zhibao-dev/hermes-agent/discussions)**

</div>
