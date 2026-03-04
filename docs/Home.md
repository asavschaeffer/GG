# Game Grammar Wiki

Detailed design documentation for the Game Grammar project. For an overview, see the [README](../README.md).

---

## Pipeline

| Page                             | Contents                                                                                                         |
| -------------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **[Theory](theory)**             | Wittgenstein foundation, collision-defined semantics, why events are the unit of meaning                         |
| **[Event Stream](event-stream)** | Game interface protocol, salience levels, entity identity, tick bundling, event sourcing connection              |
| **[Tokenization](tokens)**       | The core design problem — 5 approaches analyzed, vocabulary design, encoding examples, hyperparameters           |
| **[Transformer](transformer)**   | Architecture decisions, positional encoding, loss functions, training curriculum, scaling plan                   |
| **[Analysis](analysis)**         | Evaluation tiers, stance-shifting detection, archetype discovery, MuZero/Dreamer comparison, progression systems |

---

## Games

Each game tests a specific capability. Ordered by complexity.

| Page                     | What It Tests                                                                   | Status      |
| ------------------------ | ------------------------------------------------------------------------------- | ----------- |
| **[Snake](snake)**       | Movement, growth, conditional self-collision, simple play styles                | Complete    |
| **[Pac-Man](pacman)**    | Multi-entity, enemy AI, buff-conditional rules, spatial reasoning               | Designed    |
| **[Survivor](survivor)** | Massive entity scale, build paths, wave structure, extreme archetype divergence | Fantasizing |
| **[Chess](chess)**       | No physics, turn-based, two-player — proves game-agnosticism                    | AlphaGONE   |
