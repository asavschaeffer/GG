"""Sample sequences from a trained GameGPT and validate them."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_grammar.model import GameGPT
from game_grammar.vocab import VOCAB, VOCAB_SIZE, ID_TO_TOKEN
from game_grammar.validate import validity_rate, check_structural, check_physical, check_rules


def main():
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "weights.txt")
    weights_path = os.path.normpath(weights_path)
    if not os.path.exists(weights_path):
        print(f"No weights at {weights_path}. Run train.py first.")
        return

    model = GameGPT(
        vocab_size=VOCAB_SIZE,
        n_layer=2,
        n_embd=32,
        block_size=64,
        n_head=4,
        seed=42,
    )
    model.load_weights(weights_path)
    print("Model loaded.\n")

    bos_id = VOCAB["BOS"]
    eos_id = VOCAB["EOS"]

    samples = []
    for i in range(20):
        tokens = model.sample(bos_id, eos_id, temperature=0.5)
        samples.append(tokens)
        names = [ID_TO_TOKEN[t] for t in tokens]
        label = " ".join(names[:40])
        if len(names) > 40:
            label += " ..."
        s_pass = "S" if check_structural(tokens)["structural_pass"] else "-"
        p_pass = "P" if check_physical(tokens)["physical_pass"] else "-"
        r_pass = "R" if check_rules(tokens)["rule_pass"] else "-"
        print(f"[{s_pass}{p_pass}{r_pass}] sample {i+1:2d} ({len(tokens):3d} tok): {label}")

    print("\n--- Validity rates ---")
    rates = validity_rate(samples)
    for tier, rate in rates.items():
        print(f"  {tier:12s}: {rate:.0%}")


if __name__ == "__main__":
    main()
