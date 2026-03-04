"""Agents for generating Snake gameplay data."""

import random as _random
from .core import Action, SnakeState, DIR_DELTA


class RandomAgent:
    def __init__(self, seed=None):
        self.rng = _random.Random(seed)

    def act(self, state: SnakeState, legal: list[Action]) -> Action:
        return self.rng.choice(legal)


class GreedyAgent:
    """Minimize Manhattan distance to food."""

    def __init__(self, seed=None):
        self.rng = _random.Random(seed)

    def act(self, state: SnakeState, legal: list[Action]) -> Action:
        fx, fy = state.food
        best_dist = float("inf")
        best_actions: list[Action] = []
        for a in legal:
            dx, dy = DIR_DELTA[a]
            nx, ny = state.head[0] + dx, state.head[1] + dy
            dist = abs(nx - fx) + abs(ny - fy)
            if dist < best_dist:
                best_dist = dist
                best_actions = [a]
            elif dist == best_dist:
                best_actions.append(a)
        return self.rng.choice(best_actions)


class WallFollowerAgent:
    """Prefer moves that keep a wall adjacent."""

    def __init__(self, width=10, height=10, seed=None):
        self.width = width
        self.height = height
        self.rng = _random.Random(seed)

    def _near_wall(self, x: int, y: int) -> bool:
        return x <= 0 or x >= self.width - 1 or y <= 0 or y >= self.height - 1

    def act(self, state: SnakeState, legal: list[Action]) -> Action:
        wall_moves: list[Action] = []
        safe_moves: list[Action] = []
        for a in legal:
            dx, dy = DIR_DELTA[a]
            nx, ny = state.head[0] + dx, state.head[1] + dy
            # Skip moves that hit walls
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                continue
            safe_moves.append(a)
            if self._near_wall(nx, ny):
                wall_moves.append(a)
        choices = wall_moves if wall_moves else (safe_moves if safe_moves else legal)
        return self.rng.choice(choices)
