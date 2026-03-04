"""Train a GameGPT model on tokenized Snake episodes."""

import json
import os
import sys
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_grammar.model import GameGPT
from game_grammar.vocab import VOCAB_SIZE


def main():
    # Load episodes
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "episodes.json")
    data_path = os.path.normpath(data_path)
    if not os.path.exists(data_path):
        print(f"No data at {data_path}. Run generate.py first.")
        return

    with open(data_path) as f:
        episodes = json.load(f)
    print(f"Loaded {len(episodes)} episodes")

    # Init model
    model = GameGPT(
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_embd=32,
        block_size=64,
        n_head=4,
        seed=42,
    )
    print(f"Model params: {len(model.params)}")

    # Training
    num_steps = 5000
    rng = random.Random(42)

    for step in range(num_steps):
        ep = rng.choice(episodes)
        # Random offset within episode
        if len(ep) > model.block_size + 1:
            start = rng.randint(0, len(ep) - model.block_size - 1)
            tokens = ep[start:start + model.block_size + 1]
        else:
            tokens = ep

        loss = model.train_step(tokens, lr=0.01)

        if (step + 1) % 100 == 0 or step == 0:
            print(f"step {step+1:5d} / {num_steps} | loss {loss:.4f}")

    # Save weights
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights.txt")
    weights_path = os.path.normpath(weights_path)
    model.save_weights(weights_path)
    print(f"\nWeights saved to {weights_path}")


if __name__ == "__main__":
    main()
