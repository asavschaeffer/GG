"""Tests for event codec."""

from game_grammar.core import Action, Event, Salience, SnakeState
from game_grammar.codec import EventCodec
from game_grammar.vocab import VOCAB, ID_TO_TOKEN


def make_state(**overrides):
    defaults = dict(
        head=(5, 5), body=[(5, 5)], direction=Action.RIGHT,
        food=(3, 3), score=0, alive=True, tick=0,
    )
    defaults.update(overrides)
    return SnakeState(**defaults)


def test_snapshot_encoding():
    codec = EventCodec()
    state = make_state()
    tokens = codec.encode_snapshot(state)
    names = [ID_TO_TOKEN[t] for t in tokens]
    assert names == [
        "SNAP", "PLAYER", "X5", "Y5", "DIR_R", "LEN1",
        "FOOD", "X3", "Y3", "SCORE", "V0",
    ]


def test_snapshot_length_11():
    codec = EventCodec()
    state = make_state()
    tokens = codec.encode_snapshot(state)
    assert len(tokens) == 11


def test_event_encoding_move():
    codec = EventCodec()
    event = Event("MOVE", "player", {"pos": (6, 5)}, tick=1, salience=Salience.MOVEMENT)
    tokens = codec.encode_event(event)
    names = [ID_TO_TOKEN[t] for t in tokens]
    assert names == ["MOVE", "X6", "Y5"]


def test_event_encoding_input():
    codec = EventCodec()
    event = Event("INPUT_R", "player", {"action": "RIGHT"}, tick=1, salience=Salience.MOVEMENT)
    tokens = codec.encode_event(event)
    names = [ID_TO_TOKEN[t] for t in tokens]
    assert names == ["INPUT_R"]


def test_event_encoding_eat():
    codec = EventCodec()
    event = Event("EAT", "player", {"pos": (3, 3)}, tick=1, salience=Salience.RULE_EFFECT)
    tokens = codec.encode_event(event)
    names = [ID_TO_TOKEN[t] for t in tokens]
    assert names == ["EAT"]


def test_event_encoding_grow():
    codec = EventCodec()
    event = Event("GROW", "player", {"length": 2}, tick=1, salience=Salience.RULE_EFFECT)
    tokens = codec.encode_event(event)
    names = [ID_TO_TOKEN[t] for t in tokens]
    assert names == ["GROW", "LEN2"]


def test_event_encoding_food_spawn():
    codec = EventCodec()
    event = Event("FOOD_SPAWN", "food", {"pos": (7, 8)}, tick=1, salience=Salience.RULE_EFFECT)
    tokens = codec.encode_event(event)
    names = [ID_TO_TOKEN[t] for t in tokens]
    assert names == ["FOOD_SPAWN", "X7", "Y8"]


def test_episode_bos_eos():
    codec = EventCodec()
    state = make_state()
    tokens = codec.encode_episode({}, {0: state})
    names = [ID_TO_TOKEN[t] for t in tokens]
    assert names[0] == "BOS"
    assert names[-1] == "EOS"


def test_decode_roundtrip():
    codec = EventCodec()
    state = make_state()
    tokens = codec.encode_snapshot(state)
    # Wrap in BOS/EOS
    full = [VOCAB["BOS"]] + tokens + [VOCAB["EOS"]]
    records = codec.decode(full)
    assert records[0]["type"] == "BOS"
    assert records[1]["type"] == "SNAP"
    assert records[1]["player_x"] == "X5"
    assert records[-1]["type"] == "EOS"


def test_position_bounds():
    codec = EventCodec()
    for x in range(10):
        for y in range(10):
            event = Event("MOVE", "player", {"pos": (x, y)}, tick=1, salience=Salience.MOVEMENT)
            tokens = codec.encode_event(event)
            names = [ID_TO_TOKEN[t] for t in tokens]
            assert names[1] == f"X{x}"
            assert names[2] == f"Y{y}"
