"""Tier-based validity checker for sampled token sequences."""

from .vocab import VOCAB, ID_TO_TOKEN, VOCAB_SIZE
from .codec import EventCodec


def check_structural(tokens: list[int]) -> dict[str, bool]:
    """Tier 1: BOS/EOS present, TICK separates events, SNAP has correct format."""
    names = [ID_TO_TOKEN.get(t, "?") for t in tokens]

    has_bos = len(names) > 0 and names[0] == "BOS"
    has_eos = len(names) > 0 and names[-1] == "EOS"

    # Check SNAP format: SNAP PLAYER X_ Y_ DIR_ LEN_ FOOD X_ Y_ SCORE V_
    snap_ok = True
    for i, tok in enumerate(names):
        if tok == "SNAP":
            if i + 10 >= len(names):
                snap_ok = False
                break
            if names[i + 1] != "PLAYER":
                snap_ok = False
                break
            if not names[i + 2].startswith("X"):
                snap_ok = False
                break
            if not names[i + 3].startswith("Y"):
                snap_ok = False
                break
            if not names[i + 4].startswith("DIR_"):
                snap_ok = False
                break
            if not names[i + 5].startswith("LEN"):
                snap_ok = False
                break
            if names[i + 6] != "FOOD":
                snap_ok = False
                break
            if not names[i + 7].startswith("X"):
                snap_ok = False
                break
            if not names[i + 8].startswith("Y"):
                snap_ok = False
                break
            if names[i + 9] != "SCORE":
                snap_ok = False
                break
            if not names[i + 10].startswith("V"):
                snap_ok = False
                break

    # TICK should appear and events should follow
    has_tick = "TICK" in names

    return {
        "has_bos": has_bos,
        "has_eos": has_eos,
        "snap_format": snap_ok,
        "has_tick": has_tick,
        "structural_pass": has_bos and has_eos and snap_ok,
    }


def check_physical(tokens: list[int]) -> dict[str, bool]:
    """Tier 2: positions in bounds, consecutive MOVEs differ by 1 cell."""
    names = [ID_TO_TOKEN.get(t, "?") for t in tokens]

    positions_ok = True
    for tok in names:
        if tok.startswith("X") and tok != "X?" and len(tok) == 2:
            val = int(tok[1])
            if val < 0 or val > 9:
                positions_ok = False
        if tok.startswith("Y") and tok != "Y?" and len(tok) == 2:
            val = int(tok[1])
            if val < 0 or val > 9:
                positions_ok = False

    # Check consecutive MOVEs differ by 1 cell (Manhattan distance)
    moves_ok = True
    last_move_pos = None
    for i, tok in enumerate(names):
        if tok == "MOVE" and i + 2 < len(names):
            x_tok, y_tok = names[i + 1], names[i + 2]
            if x_tok.startswith("X") and y_tok.startswith("Y") and len(x_tok) == 2 and len(y_tok) == 2:
                try:
                    x, y = int(x_tok[1]), int(y_tok[1])
                    if last_move_pos is not None:
                        lx, ly = last_move_pos
                        dist = abs(x - lx) + abs(y - ly)
                        if dist != 1:
                            moves_ok = False
                    last_move_pos = (x, y)
                except ValueError:
                    pass

    return {
        "positions_in_bounds": positions_ok,
        "moves_adjacent": moves_ok,
        "physical_pass": positions_ok and moves_ok,
    }


def check_rules(tokens: list[int]) -> dict[str, bool]:
    """Tier 3: EAT→GROW+FOOD_SPAWN, DIE→EOS (game ends), LEN increments by 1."""
    names = [ID_TO_TOKEN.get(t, "?") for t in tokens]

    # EAT should be followed (eventually, before next TICK) by GROW and FOOD_SPAWN
    eat_grow_ok = True
    i = 0
    while i < len(names):
        if names[i] == "EAT":
            # Scan until next TICK or EOS
            rest = names[i+1:]
            tick_idx = len(rest)
            for j, t in enumerate(rest):
                if t in ("TICK", "EOS"):
                    tick_idx = j
                    break
            segment = rest[:tick_idx]
            if "GROW" not in segment or "FOOD_SPAWN" not in segment:
                eat_grow_ok = False
                break
        i += 1

    # DIE → should be followed by EOS (no more game events after death)
    die_eos_ok = True
    for i, tok in enumerate(names):
        if tok in ("DIE_WALL", "DIE_SELF"):
            rest_after = [t for t in names[i+1:] if t not in ("EOS", "SCORE")]
            # After death, only SCORE and EOS should remain
            for t in rest_after:
                if t not in ("V0", "V1", "V2", "V3", "V4", "V5",
                             "V6", "V7", "V8", "V9", "V10"):
                    die_eos_ok = False
                    break

    return {
        "eat_triggers_grow": eat_grow_ok,
        "die_ends_game": die_eos_ok,
        "rule_pass": eat_grow_ok and die_eos_ok,
    }


def validity_rate(samples: list[list[int]]) -> dict[str, float]:
    """Compute pass rates per tier across a batch of samples."""
    n = len(samples)
    if n == 0:
        return {}

    structural = sum(1 for s in samples if check_structural(s)["structural_pass"]) / n
    physical = sum(1 for s in samples if check_physical(s)["physical_pass"]) / n
    rules = sum(1 for s in samples if check_rules(s)["rule_pass"]) / n
    full = sum(
        1 for s in samples
        if check_structural(s)["structural_pass"]
        and check_physical(s)["physical_pass"]
        and check_rules(s)["rule_pass"]
    ) / n

    return {
        "structural": structural,
        "physical": physical,
        "rules": rules,
        "full": full,
    }
