"""EventCodec: encode/decode events and snapshots to/from token sequences."""

from .core import Action, Event, Salience, SnakeState
from .vocab import VOCAB, ID_TO_TOKEN

# Lookup maps
_DIR_TOKEN = {
    Action.UP: "DIR_U", Action.DOWN: "DIR_D",
    Action.LEFT: "DIR_L", Action.RIGHT: "DIR_R",
}

_INPUT_MAP = {
    "INPUT_U": Action.UP, "INPUT_D": Action.DOWN,
    "INPUT_L": Action.LEFT, "INPUT_R": Action.RIGHT,
}

_EVENT_TOKENS = {"MOVE", "EAT", "GROW", "DIE_WALL", "DIE_SELF", "FOOD_SPAWN", "SCORE"}


class EventCodec:
    def __init__(self, snapshot_interval=16, salience_threshold=Salience.TICK):
        self.snapshot_interval = snapshot_interval
        self.salience_threshold = salience_threshold

    def encode_snapshot(self, state: SnakeState) -> list[int]:
        length = len(state.body)
        len_tok = f"LEN{length}" if length <= 20 else "LEN_LONG"
        score_tok = f"V{min(state.score, 10)}"
        tokens = [
            "SNAP", "PLAYER",
            f"X{state.head[0]}", f"Y{state.head[1]}",
            _DIR_TOKEN[state.direction],
            len_tok,
            "FOOD",
            f"X{state.food[0]}", f"Y{state.food[1]}",
            "SCORE", score_tok,
        ]
        return [VOCAB[t] for t in tokens]

    def encode_event(self, event: Event) -> list[int]:
        t = event.type
        tokens: list[str] = []

        if t.startswith("INPUT_"):
            tokens.append(t)
        elif t == "MOVE":
            x, y = event.payload["pos"]
            tokens.extend(["MOVE", f"X{x}", f"Y{y}"])
        elif t == "EAT":
            tokens.append("EAT")
        elif t == "GROW":
            length = event.payload["length"]
            len_tok = f"LEN{length}" if length <= 20 else "LEN_LONG"
            tokens.extend(["GROW", len_tok])
        elif t == "FOOD_SPAWN":
            x, y = event.payload["pos"]
            tokens.extend(["FOOD_SPAWN", f"X{x}", f"Y{y}"])
        elif t == "DIE_WALL":
            tokens.append("DIE_WALL")
        elif t == "DIE_SELF":
            tokens.append("DIE_SELF")
        elif t == "SCORE":
            score = event.payload["score"]
            score_tok = f"V{min(score, 10)}"
            tokens.extend(["SCORE", score_tok])
        else:
            raise ValueError(f"Unknown event type: {t}")

        return [VOCAB[t] for t in tokens]

    def encode_tick_events(self, events: list[Event]) -> list[int]:
        filtered = [e for e in events if e.salience >= self.salience_threshold]
        if not filtered:
            return []
        tokens = [VOCAB["TICK"]]
        for event in filtered:
            tokens.extend(self.encode_event(event))
        return tokens

    def encode_episode(
        self,
        events_by_tick: dict[int, list[Event]],
        states_by_tick: dict[int, SnakeState],
    ) -> list[int]:
        tokens = [VOCAB["BOS"]]

        max_tick = max(states_by_tick.keys()) if states_by_tick else 0

        for tick in range(max_tick + 1):
            # Snapshot at tick 0, every snapshot_interval, or on rule-effect events
            need_snapshot = (
                tick == 0
                or (tick % self.snapshot_interval == 0)
                or any(
                    e.salience >= Salience.RULE_EFFECT
                    for e in events_by_tick.get(tick, [])
                )
            )
            if need_snapshot and tick in states_by_tick:
                tokens.extend(self.encode_snapshot(states_by_tick[tick]))

            if tick in events_by_tick:
                tick_tokens = self.encode_tick_events(events_by_tick[tick])
                tokens.extend(tick_tokens)

        tokens.append(VOCAB["EOS"])
        return tokens

    def decode(self, tokens: list[int]) -> list[dict]:
        """Decode token sequence into a list of parsed records for validation."""
        names = [ID_TO_TOKEN[t] for t in tokens]
        records: list[dict] = []
        i = 0
        while i < len(names):
            tok = names[i]
            if tok == "BOS":
                records.append({"type": "BOS"})
                i += 1
            elif tok == "EOS":
                records.append({"type": "EOS"})
                i += 1
            elif tok == "TICK":
                records.append({"type": "TICK"})
                i += 1
            elif tok == "SNAP":
                # SNAP PLAYER X_ Y_ DIR_ LEN_ FOOD X_ Y_ SCORE V_
                rec = {"type": "SNAP"}
                if i + 10 < len(names):
                    rec["player_x"] = names[i + 2]
                    rec["player_y"] = names[i + 3]
                    rec["direction"] = names[i + 4]
                    rec["length"] = names[i + 5]
                    rec["food_x"] = names[i + 7]
                    rec["food_y"] = names[i + 8]
                    rec["score_tok"] = names[i + 10]
                records.append(rec)
                i += 11
            elif tok.startswith("INPUT_"):
                records.append({"type": "INPUT", "direction": tok})
                i += 1
            elif tok == "MOVE":
                rec = {"type": "MOVE"}
                if i + 2 < len(names):
                    rec["x"] = names[i + 1]
                    rec["y"] = names[i + 2]
                records.append(rec)
                i += 3
            elif tok == "EAT":
                records.append({"type": "EAT"})
                i += 1
            elif tok == "GROW":
                rec = {"type": "GROW"}
                if i + 1 < len(names):
                    rec["length"] = names[i + 1]
                records.append(rec)
                i += 2
            elif tok == "FOOD_SPAWN":
                rec = {"type": "FOOD_SPAWN"}
                if i + 2 < len(names):
                    rec["x"] = names[i + 1]
                    rec["y"] = names[i + 2]
                records.append(rec)
                i += 3
            elif tok == "DIE_WALL":
                records.append({"type": "DIE_WALL"})
                i += 1
            elif tok == "DIE_SELF":
                records.append({"type": "DIE_SELF"})
                i += 1
            elif tok == "SCORE":
                rec = {"type": "SCORE"}
                if i + 1 < len(names):
                    rec["value"] = names[i + 1]
                records.append(rec)
                i += 2
            else:
                records.append({"type": "UNKNOWN", "token": tok})
                i += 1
        return records
