"""Tests for the validity checker."""

from game_grammar.vocab import VOCAB
from game_grammar.validate import check_structural, check_physical, check_rules


def tok(*names):
    return [VOCAB[n] for n in names]


def test_structural_pass():
    tokens = tok(
        "BOS",
        "SNAP", "PLAYER", "X5", "Y5", "DIR_R", "LEN1", "FOOD", "X3", "Y3", "SCORE", "V0",
        "TICK", "INPUT_R", "MOVE", "X6", "Y5",
        "EOS",
    )
    result = check_structural(tokens)
    assert result["structural_pass"]


def test_structural_no_bos():
    tokens = tok("TICK", "INPUT_R", "MOVE", "X6", "Y5", "EOS")
    result = check_structural(tokens)
    assert not result["has_bos"]
    assert not result["structural_pass"]


def test_structural_no_eos():
    tokens = tok("BOS", "TICK", "INPUT_R", "MOVE", "X6", "Y5")
    result = check_structural(tokens)
    assert not result["has_eos"]
    assert not result["structural_pass"]


def test_physical_pass():
    tokens = tok(
        "BOS",
        "TICK", "INPUT_R", "MOVE", "X5", "Y5",
        "TICK", "INPUT_R", "MOVE", "X6", "Y5",
        "TICK", "INPUT_R", "MOVE", "X7", "Y5",
        "EOS",
    )
    result = check_physical(tokens)
    assert result["positions_in_bounds"]
    assert result["moves_adjacent"]
    assert result["physical_pass"]


def test_physical_non_adjacent_moves():
    tokens = tok(
        "BOS",
        "TICK", "MOVE", "X1", "Y1",
        "TICK", "MOVE", "X5", "Y5",
        "EOS",
    )
    result = check_physical(tokens)
    assert not result["moves_adjacent"]
    assert not result["physical_pass"]


def test_rules_eat_grow():
    tokens = tok(
        "BOS",
        "TICK", "INPUT_R", "EAT", "MOVE", "X3", "Y3", "GROW", "LEN2", "FOOD_SPAWN", "X7", "Y8", "SCORE", "V1",
        "EOS",
    )
    result = check_rules(tokens)
    assert result["eat_triggers_grow"]


def test_rules_die_ends_game():
    tokens = tok(
        "BOS",
        "TICK", "INPUT_R", "DIE_WALL",
        "EOS",
    )
    result = check_rules(tokens)
    assert result["die_ends_game"]
    assert result["rule_pass"]
