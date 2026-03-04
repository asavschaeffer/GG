"""Snake game logic."""

import random as _random
from .core import Action, Event, Salience, SnakeState, DIR_DELTA, OPPOSITE


class SnakeGame:
    def __init__(self, width=10, height=10, seed=None):
        self.width = width
        self.height = height
        self.rng = _random.Random(seed)
        self._state: SnakeState | None = None

    def reset(self) -> SnakeState:
        cx, cy = self.width // 2, self.height // 2
        direction = self.rng.choice(list(Action))
        head = (cx, cy)
        body = [head]
        food = self._spawn_food(body)
        self._state = SnakeState(
            head=head, body=body, direction=direction,
            food=food, score=0, alive=True, tick=0,
        )
        return self._state

    def step(self, action: Action) -> tuple[SnakeState, list[Event], bool]:
        s = self._state
        assert s is not None and s.alive
        events = []
        tick = s.tick + 1

        # Resolve direction: ignore reversal
        if action != OPPOSITE.get(s.direction):
            direction = action
        else:
            direction = s.direction

        # Input event
        events.append(Event(
            type=f"INPUT_{action.value[0]}",
            entity="player",
            payload={"action": action.value},
            tick=tick,
            salience=Salience.MOVEMENT,
        ))

        # Compute new head
        dx, dy = DIR_DELTA[direction]
        nx, ny = s.head[0] + dx, s.head[1] + dy

        # Wall collision
        if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
            events.append(Event(
                type="DIE_WALL", entity="player",
                payload={"pos": (nx, ny)},
                tick=tick, salience=Salience.COLLISION,
            ))
            new_state = SnakeState(
                head=s.head, body=list(s.body), direction=direction,
                food=s.food, score=s.score, alive=False, tick=tick,
            )
            self._state = new_state
            return new_state, events, True

        new_head = (nx, ny)
        ate = new_head == s.food

        if ate:
            events.append(Event(
                type="EAT", entity="player",
                payload={"pos": new_head},
                tick=tick, salience=Salience.RULE_EFFECT,
            ))

        # Build new body
        new_body = [new_head] + list(s.body)
        if not ate:
            new_body.pop()  # retract tail

        # Self-collision (check against body AFTER building, excluding head)
        if new_head in new_body[1:]:
            events.append(Event(
                type="DIE_SELF", entity="player",
                payload={"pos": new_head},
                tick=tick, salience=Salience.COLLISION,
            ))
            new_state = SnakeState(
                head=new_head, body=new_body, direction=direction,
                food=s.food, score=s.score, alive=False, tick=tick,
            )
            self._state = new_state
            return new_state, events, True

        # Successful move
        events.append(Event(
            type="MOVE", entity="player",
            payload={"pos": new_head},
            tick=tick, salience=Salience.MOVEMENT,
        ))

        new_score = s.score
        new_food = s.food

        if ate:
            new_score += 1
            events.append(Event(
                type="GROW", entity="player",
                payload={"length": len(new_body)},
                tick=tick, salience=Salience.RULE_EFFECT,
            ))
            new_food = self._spawn_food(new_body)
            events.append(Event(
                type="FOOD_SPAWN", entity="food",
                payload={"pos": new_food},
                tick=tick, salience=Salience.RULE_EFFECT,
            ))
            events.append(Event(
                type="SCORE", entity="player",
                payload={"score": new_score},
                tick=tick, salience=Salience.RULE_EFFECT,
            ))

        new_state = SnakeState(
            head=new_head, body=new_body, direction=direction,
            food=new_food, score=new_score, alive=True, tick=tick,
        )
        self._state = new_state
        return new_state, events, False

    def legal_actions(self, state: SnakeState) -> list[Action]:
        return [a for a in Action if a != OPPOSITE.get(state.direction)]

    def _spawn_food(self, occupied: list[tuple[int, int]]) -> tuple[int, int]:
        occupied_set = set(occupied)
        empty = [
            (x, y)
            for x in range(self.width)
            for y in range(self.height)
            if (x, y) not in occupied_set
        ]
        return self.rng.choice(empty)
