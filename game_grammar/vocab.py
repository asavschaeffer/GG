"""Token vocabulary for Snake game-grammar (74 tokens)."""

_tokens = []

# Structural (4)
_tokens += ["BOS", "EOS", "TICK", "SNAP"]

# Entity (3)
_tokens += ["PLAYER", "FOOD", "WALL"]

# Direction (4)
_tokens += ["DIR_U", "DIR_D", "DIR_L", "DIR_R"]

# Input (4)
_tokens += ["INPUT_U", "INPUT_D", "INPUT_L", "INPUT_R"]

# Position X (10)
_tokens += [f"X{i}" for i in range(10)]

# Position Y (10)
_tokens += [f"Y{i}" for i in range(10)]

# Event types (7)
_tokens += ["MOVE", "EAT", "GROW", "DIE_WALL", "DIE_SELF", "FOOD_SPAWN", "SCORE"]

# Value (11): V0-V10
_tokens += [f"V{i}" for i in range(11)]

# Length (21): LEN1-LEN20, LEN_LONG
_tokens += [f"LEN{i}" for i in range(1, 21)]
_tokens += ["LEN_LONG"]

VOCAB: dict[str, int] = {tok: i for i, tok in enumerate(_tokens)}
ID_TO_TOKEN: dict[int, str] = {i: tok for tok, i in VOCAB.items()}
VOCAB_SIZE: int = len(VOCAB)

assert VOCAB_SIZE == 74, f"Expected 74 tokens, got {VOCAB_SIZE}"
