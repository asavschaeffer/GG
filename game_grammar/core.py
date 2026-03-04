"""Foundation types for the game-grammar pipeline."""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any


class Action(Enum):
    UP = "UP"
    DOWN = "DOWN"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


class Salience(IntEnum):
    TICK = 0
    MOVEMENT = 1
    COLLISION = 2
    RULE_EFFECT = 3
    PHASE = 4


@dataclass(frozen=True)
class Event:
    type: str
    entity: str
    payload: dict
    tick: int
    salience: Salience


@dataclass
class SnakeState:
    head: tuple[int, int]
    body: list[tuple[int, int]]
    direction: Action
    food: tuple[int, int]
    score: int
    alive: bool
    tick: int


DIR_DELTA: dict[Action, tuple[int, int]] = {
    Action.UP:    ( 0, -1),
    Action.DOWN:  ( 0,  1),
    Action.LEFT:  (-1,  0),
    Action.RIGHT: ( 1,  0),
}

OPPOSITE: dict[Action, Action] = {
    Action.UP:    Action.DOWN,
    Action.DOWN:  Action.UP,
    Action.LEFT:  Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}
