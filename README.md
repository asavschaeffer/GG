# Game Grammar

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Status: Design Phase](https://img.shields.io/badge/Status-Design%20Phase-yellow)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)

**A transformer learns the grammar of videogames by predicting what happens next.**

Gameplay is atomized into discrete events — movements, collisions, collections, deaths — and tokenized into a sequence. A causal transformer trained on next-token prediction learns the rules, physics, and behavioral patterns of the game from this sequence alone. Player archetypes emerge as statistical regularities in the learned grammar, without labels.

```
Game (any) → Event Stream → Token Sequence → Transformer → Grammar
```

The game never touches the model. The tokenization layer is where game-agnosticism lives.

---

## Motivation

> *The meaning of a word is its use in the language.* — Wittgenstein, *Philosophical Investigations* &#167;43

A videogame is a language-game. Consider a 2D Pac-Man-like game.

**Inputs** are discrete:

```
joystick ∈ {L, R, U, D}
```

**State variables** are discrete:

```
player.pos(x, y)
enemy.pos(x, y)
coin.pos(x, y)
berry.pos(x, y)
```

But state has no intrinsic meaning. Meaning arises only through **events**.

**Game events are collision-defined:**

```
player.pos touches wall        → movement blocked
player.pos touches coin.pos   → score increment, coin removed
player.pos touches enemy.pos  → game over
player.pos touches berry.pos  → buff gained
```

A coin *is* the thing that increments your score when you touch it. An enemy *is* the thing that kills you. These are not labels — they are behavioral definitions. And rule semantics are conditional:

```
if buff active:
    player.pos touches enemy.pos → enemy removed
```

The same collision, different context, opposite outcome. Entities are defined solely by their behavior under collision. The structure of what can follow what — which events are valid continuations of which histories — is the game's grammar.

A causal transformer trained to predict the next event token given prior context learns that grammar the same way a language model learns syntax. From this single objective it learns:

- physical regularities (movement, blocking)
- rule mappings (collision → outcome)
- temporal dependencies (buff duration)
- long-horizon behavioral patterns

Multi-head attention supports simultaneous modeling of mechanics, rules, and strategy without privileging any single description.

---

## How It Works

### 1. Any game emits events

A game implements a minimal protocol — `reset()`, `step(action)`, `legal_actions(state)` — and emits structured events tagged with salience levels. The event stream layer converts state transitions into discrete, named events without knowing anything about the specific game.

> [Event Stream](../../wiki/event-stream) — Game interface protocol, salience levels, entity identity, tick bundling.

### 2. Events become tokens

Gameplay is represented as a sequence of **state transitions**, not static facts. Tokens encode changes:

- inputs
- movements
- collisions
- rule effects

The resulting sequence forms a symbolic trace of meaningful events. This is where the hard design decisions live — five approaches span the spectrum from full state snapshots to fully learned representations. The recommended starting point is a hybrid — periodic keyframe snapshots + high-frequency delta events, like video codecs (I-frames + P-frames) and like event sourcing with periodic snapshots:

```
[SNAPSHOT player@3,4 enemy@7,2 coins:3 buff:off]
[INPUT_R] [MOVE player 4,4] [COLLECT coin@4,4] [SCORE 4]
[INPUT_R] [MOVE player 5,4]
[INPUT_U] [MOVE player 5,3] [COLLIDE enemy@5,3] [DEATH]
```

Snapshot frequency is a hyperparameter — the model's working memory refresh rate. Invariance is represented by the absence of tokens. Frequently co-occurring transitions may be merged into compound tokens.

> [Tokenization](../../wiki/tokens) — Full approach spectrum, vocabulary design, encoding examples, hyperparameters.

### 3. A transformer learns the grammar

Adapted from Karpathy's [microgpt](https://github.com/karpathy/microgpt) — a complete GPT in pure Python with autograd. Replace the data pipeline (character names → game events), scale the architecture (larger vocab, longer context, more layers), keep the core intact.

> [Transformer](../../wiki/transformer) — Architecture decisions, training curriculum, scaling plan.

### 4. Archetypes emerge

The trained model predicts valid event continuations and recurrent gameplay patterns. Distinct player behaviors emerge as stable statistical regularities:

- Trajectories biased toward power-ups followed by enemy engagement
- Trajectories biased toward scoring while avoiding enemies

These behaviors are not explicitly labeled; they arise as predictive abstractions over event sequences.

Three evaluation tiers confirm this progressively:

1. **Valid behavior** — Can the model generate sequences that obey game rules?
2. **Conditional rules** — Does it learn that a buff changes what collision means?
3. **Play styles** — Do distinct behavioral patterns appear in the generated traces?

> [Analysis](../../wiki/analysis) — Evaluation tiers, stance-shifting detection, archetype discovery, progression systems.

---

## Why Not MuZero?

MuZero asks *"what should I do?"* This asks *"what is happening?"*

Both learn world models, but they're almost duals of each other:

|  | MuZero / Dreamer | Game Grammar |
|---|---|---|
| State | Learned continuous latent | Explicit symbolic tokens |
| World model | Neural dynamics function | Next-token prediction |
| Interpretability | Opaque | Readable by construction |
| Objective | Maximize reward | Predict what happens |
| Output | Optimal policy | Behavioral grammar |

The critical difference: a MuZero latent state is unreadable. A token sequence is a sentence. You can inspect it, cluster it, ask "does this obey the rules?" That readability is what makes archetype discovery possible — you can actually read the play styles.

The connection to [event sourcing](https://martinfowler.com/eaaDev/EventSourcing.html) is nearly 1:1. The token sequence is an event log. The transformer learns the projection function. Snapshots are snapshots. The vocabulary is the event schema.

---

## Game Progression

Each game tests a specific capability. Ordered by complexity — each adds something the previous couldn't test.

| | Game | Tests | Key Question |
|---|---|---|---|
| 1 | **Snake** | Movement, growth, conditional self-collision | Can it learn that self-collision depends on length? |
| 2 | **Pac-Man** | Multi-entity, enemy AI, buff-conditional rules | Can it learn that a power pellet flips ghost interactions? |
| 3 | **Survivor** | Massive entity scale, build paths, wave structure | Can it handle 100+ enemies and learn that upgrades predict behavior? |
| 4 | **Chess** | No physics, turn-based, two-player | Can the system that learns Snake physics also learn opening theory? |

If chess works, the system is genuinely game-agnostic — not just "action-game-agnostic."

> Game docs: [Snake](../../wiki/snake) | [Pac-Man](../../wiki/pacman) | [Survivor](../../wiki/survivor) | [Chess](../../wiki/chess)

---

## The Endgame

Identify how someone plays and give them progression that fits.

```
1. Player plays normally
2. Event stream records gameplay
3. Trained model classifies trace → archetype
4. Game offers skills that complement their natural style

   Farmer    → "Harvest": auto-collect nearby coins
   Hunter    → "Predator": buff duration +50%
   Berserker → "Fury": speed boost near enemies
   Survivor  → "Phantom": invulnerability on near-miss
```

Not because anyone defined "berserker" — because the model found the pattern.

---

## Getting Started

> **Note:** Game Grammar is in the design phase. The pipeline is not yet implemented — we're documenting approaches and trade-offs before building. Contributions to the design discussion are welcome.

### Prerequisites

- Python 3.10+

### Explore the Design

The best way to get started is to read the [wiki](../../wiki):

1. **[Event Stream](../../wiki/event-stream)** — How any game becomes a stream of events
2. **[Tokenization](../../wiki/tokens)** — The core design problem: events to integers
3. **[Transformer](../../wiki/transformer)** — Model architecture and training plan
4. **[Analysis](../../wiki/analysis)** — How we know it works, archetype discovery

The reference transformer is available to inspect:

```bash
# Read through Karpathy's microgpt — our starting point
python microgpt.py
```

---

## Project Structure

```
microgpt.py             Reference transformer (Karpathy)
game_grammar/           Source code (coming soon)
scripts/                Data generation and training scripts (coming soon)
tests/                  Test suite (coming soon)
```

Design documentation lives in the [wiki](../../wiki).

---

## Roadmap

- [x] Theoretical framing and premise
- [x] Tokenization approach analysis (5 strategies documented)
- [x] Transformer architecture decisions
- [x] Evaluation tier design
- [x] Game specifications (Snake, Pac-Man, Survivor, Chess)
- [ ] Implement Snake game + event stream
- [ ] Build tokenizer (hybrid keyframes + deltas)
- [ ] Adapt microgpt for game event training
- [ ] Train on Snake traces, evaluate Tier 1 (valid behavior)
- [ ] Evaluate Tier 2 (conditional rule emergence)
- [ ] Evaluate Tier 3 (play style / archetype emergence)
- [ ] Pac-Man implementation and training
- [ ] Port to PyTorch for scale
- [ ] Survivor implementation
- [ ] Chess (Lichess data integration)
- [ ] Archetype-based progression system prototype

---

## Contributing

This project is in early design. The most valuable contributions right now are:

- Feedback on the [tokenization approaches](../../wiki/tokens) — this is the core unsolved problem
- Alternative games that would test capabilities we haven't considered
- Connections to related work (world models, player modeling, event sourcing)
- Experience reports from applying similar ideas

Please open an issue to discuss before submitting a PR.

---

## License

[GPL-3.0](LICENSE) — Free for research, educational, and personal use. Commercial use requires permission. See LICENSE for details.

---

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — microgpt, the reference transformer implementation
- Ludwig Wittgenstein — *Philosophical Investigations*, the theoretical foundation
