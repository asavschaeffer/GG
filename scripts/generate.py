"""Generate tokenized Snake episodes and save as JSON."""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_grammar.agents import RandomAgent, GreedyAgent, WallFollowerAgent
from game_grammar.codec import EventCodec
from game_grammar.data import collect_episodes
from game_grammar.vocab import ID_TO_TOKEN


def main():
    n_episodes = 200
    print(f"Generating {n_episodes} episodes...")

    agent_mix = [
        (RandomAgent(seed=1), 0.4),
        (GreedyAgent(seed=2), 0.4),
        (WallFollowerAgent(10, 10, seed=3), 0.2),
    ]

    codec = EventCodec(snapshot_interval=16)
    episodes = collect_episodes(
        n=n_episodes,
        agent_mix=agent_mix,
        codec=codec,
        seed=42,
    )

    # Stats
    lengths = [len(ep) for ep in episodes]
    print(f"Episodes: {len(episodes)}")
    print(f"Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

    # Show a sample decoded
    print("\n--- Sample episode (first 80 tokens) ---")
    sample = episodes[0][:80]
    names = [ID_TO_TOKEN[t] for t in sample]
    print(" ".join(names))

    # Save
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "episodes.json")
    out_path = os.path.normpath(out_path)
    with open(out_path, "w") as f:
        json.dump(episodes, f)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
