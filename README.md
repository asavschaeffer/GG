# Game Grammar

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Status: Snake](https://img.shields.io/badge/Status-Snake%20PoC-green)]()
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg)](https://www.python.org/)
[![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen)]()

**A transformer learns the grammar of videogames by predicting what happens next.** Built in pure Python with zero dependencies. Yes, I waited 36 hours while this thing farted itself into existence on my CPU.

Gameplay is atomized into discrete events — movements, collisions, collections, deaths — and tokenized into a sequence. A causal transformer trained on next-token prediction learns the rules, physics, and behavioral patterns of the game from this sequence alone. Player archetypes emerge as statistical regularities in the learned grammar, without labels.

```
Game (any) → Event Stream → Token Sequence → Transformer → Grammar
```

For now, the game never touches the model. The tokenization layer is where game-agnosticism lives. In the long run we will feed the snake it's own tail.

---

## Quick Start

```bash
# Generate 200 episodes from a mix of agents → episodes.json
python scripts/generate.py

# Train the model for 5000 steps → weights.txt
python scripts/train.py

# Sample 20 novel gameplay sequences and validate them
python scripts/sample.py
```

> [`episodes.json`](episodes.json) — tokenized gameplay traces from Random (40%), Greedy (40%), and WallFollower (20%) agents.
> [`weights.txt`](weights.txt) — 31K trained parameters for a 2-layer, 32-dim, 4-head transformer with a 64-token context window.

### Results

First pass — 31K parameters, 200 episodes, 5000 training steps:

| Metric              | Result                                                  |
| ------------------- | ------------------------------------------------------- |
| Loss                | 4.47 → 0.25 (random baseline: ln(74) ≈ 4.3)             |
| Physical validity   | **95%** — moves are adjacent cells, positions in bounds |
| Rule validity       | **100%** — EAT→GROW+FOOD_SPAWN, DIE→EOS                 |
| Structural validity | 45%\*                                                   |

_\*Structural validity is low because the model often hits the 64-token context limit mid-sequence without generating EOS. The test expects complete BOS→EOS episodes — the model generates valid gameplay that simply runs longer than the context window allows._

Sampled sequences read like real Snake games. The model learned that movement is one cell per tick, that eating triggers growth and food respawn, and that death ends the episode.

**Sample token sequence:**

```
BOS SNAP PLAYER X5 Y5 DIR_R LEN1 FOOD X8 Y6 SCORE V0
  TICK INPUT_L MOVE X4 Y5
  TICK INPUT_D MOVE X4 Y6
  TICK INPUT_D MOVE X4 Y7
  TICK INPUT_R MOVE X5 Y7
  TICK INPUT_R MOVE X6 Y7
  ...
```

74 tokens cover the full Snake vocabulary — structural markers, entities, positions, directions, event types, and values.

---

## Motivation

> _The meaning of a word is its use in the language._ — Wittgenstein, _Philosophical Investigations_ §43

State has no intrinsic meaning. Meaning arises only through **events** — a coin is the thing that increments your score when you touch it. Entities are defined solely by their behavior under collision. The structure of what can follow what is the game's **grammar**.

A causal transformer learns this grammar by predicting the next event token, the same way a language model learns syntax. From this single objective it learns physical regularities, rule mappings, temporal dependencies, and long-horizon behavioral patterns.

> [Theory](../../wiki/theory) — Wittgensteinian framing, collision-defined semantics, conditional rules, event sourcing.

---

## How It Works

Unlike MuZero, which learns what a player _should_ do, this learns what players _actually_ do — and the output is readable by construction.

### 1. Any game emits events

A game implements a minimal protocol — `reset()`, `step(action)`, `legal_actions(state)` — and emits structured events tagged with salience levels. The event stream layer converts state transitions into discrete, named events without knowing anything about the specific game.

> [Event Stream](../../wiki/event-stream) — Game interface protocol, salience levels, entity identity, tick bundling.

### 2. Events become tokens

Gameplay is represented as a sequence of **state transitions**, not static facts. Tokens encode changes — inputs, movements, collisions, rule effects. The recommended approach is a hybrid: periodic keyframe snapshots + high-frequency delta events, like video codecs (I-frames + P-frames).

> [Tokenization](../../wiki/tokens) — Five approaches, vocabulary design, encoding examples, hyperparameters.

### 3. A transformer learns the grammar

Adapted from Karpathy's [microgpt](https://github.com/karpathy/microgpt) — a complete GPT in pure Python with autograd. No frameworks, no dependencies.

| Parameter      | microgpt         | GameGPT          |
| -------------- | ---------------- | ---------------- |
| Vocabulary     | ~27 (characters) | 74 (game events) |
| Embedding dim  | 16               | 32               |
| Layers         | 1                | 2                |
| Context window | 16               | 64               |
| Heads          | 4                | 4                |
| Parameters     | ~1.2K            | ~31K             |

Same architecture: token + position embeddings, multi-head causal attention, RMSNorm, ReLU MLPs, Adam optimizer.

> [Transformer](../../wiki/transformer) — Architecture decisions, training curriculum, scaling plan.

### 4. Archetypes emerge

The trained model predicts valid event continuations and recurrent gameplay patterns. Distinct player behaviors emerge as stable statistical regularities — trajectories biased toward power-ups vs. scoring while avoiding enemies. These behaviors are not explicitly labeled; they arise as predictive abstractions over event sequences.

> [Analysis](../../wiki/analysis) — Three evaluation tiers, stance-shifting detection, archetype discovery, progression systems.

---

## Game Progression

The system is game-agnostic. These games test it at increasing complexity:

|     | Game         | Status           | Tests                                             |
| --- | ------------ | ---------------- | ------------------------------------------------- |
| 1   | **Snake**    | Proof of Concept | Movement, growth, conditional self-collision      |
| 2   | **Pac-Man**  | Designed         | Multi-entity, enemy AI, buff-conditional rules    |
| 3   | **Survivor** | Fantasizing      | Massive entity scale, build paths, wave structure |
| 4   | **Chess**    | AlphaGone        | No physics, turn-based, two-player                |

If chess works, the system is genuinely game-agnostic — not just "action-game-agnostic."

> Game docs: [Snake](../../wiki/snake) | [Pac-Man](../../wiki/pacman) | [Survivor](../../wiki/survivor) | [Chess](../../wiki/chess)

---

## The Endgame

In the long run, I'd like to identify player archetypes from learned event grammar and use them to drive progression systems — give players unique passives and skills that complement their natural play style. Imagine CoD Zombies that gives you piercing bullets if you're a trainer or longer mags if you're camping behind claymores. Of course at that point we could buff the mobs to break the train or jump claymores, too.

```
1. Player plays normally
2. Event stream records gameplay
3. Trained model classifies trace → archetype
4. Game offers skills that complement their natural style

   Wall Hugger  → "Wall Slide": speed +1 when moving parallel to a wall
   Food Chaser  → "Scent": directional arrow toward food from anywhere
   Coiler       → "Phase": pass through your own tail once per life
   Space Filler → "Momentum": every 5th move without turning is free
```

Not because anyone defined "berserker" — because the model found the pattern.

---

## Project Structure

```
game_grammar/
  core.py          Event, SnakeState, Action, Salience
  snake.py         Snake game (10x10 grid, configurable)
  vocab.py         74-token vocabulary
  codec.py         Hybrid snapshot+delta encoder/decoder
  agents.py        RandomAgent, GreedyAgent, WallFollowerAgent
  data.py          Episode collection and tokenization
  model.py         Value autograd + GameGPT transformer
  validate.py      Three-tier validity checker

scripts/
  generate.py      Run agents, produce tokenized episodes
  train.py         Train model on episodes
  sample.py        Sample sequences, validate, report

tests/
  test_snake.py    Game logic
  test_codec.py    Tokenizer round-trips
  test_validate.py Validity checker
```

Design documentation lives in the [wiki](../../wiki).

---

## Roadmap

- [x] Theoretical framing and premise
- [x] Tokenization approach analysis (5 strategies documented)
- [x] Transformer architecture decisions
- [x] Evaluation tier design
- [x] Game specifications (Snake, Pac-Man, Survivor, Chess)
- [x] Snake game + event stream
- [x] Hybrid keyframe+delta tokenizer (74 tokens)
- [x] GameGPT transformer (31K params, pure Python)
- [x] Train on Snake traces — Tier 1 (valid behavior): **95% physical, 100% rule**
- [ ] Evaluate Tier 2 (conditional rule emergence)
- [ ] Evaluate Tier 3 (play style / archetype emergence)
- [ ] Visualizers
- [ ] Performance
- [ ] Ouroboros
- [ ] Pac-Man implementation and training
- [ ] Archetype-based progression system prototype
- [ ] Survivor implementation
- [ ] Chess (Lichess data integration)

---

## Contributing

The most valuable contributions right now are:

- Feedback on the [tokenization approaches](../../wiki/tokens) — this is the core unsolved problem
- Alternative games that would test capabilities we haven't considered
- Connections to related work (world models, player modeling, event sourcing)
- Experience reports from applying similar ideas
- Hate and F.U.D.

Please open an issue or submit a PR.

---

## License

[GPL-3.0](LICENSE) — Free for research, educational, and personal use. Commercial use requires permission. See LICENSE for details.

---

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — microgpt, the reference transformer implementation
- Ludwig Wittgenstein — _Philosophical Investigations_, the theoretical foundation
