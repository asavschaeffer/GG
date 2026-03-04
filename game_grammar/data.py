"""Episode collection and tokenization for training."""

import random as _random
from .core import Action, Event, SnakeState
from .snake import SnakeGame
from .codec import EventCodec


def play_episode(
    game: SnakeGame,
    agent,
    codec: EventCodec | None = None,
    max_ticks: int = 200,
) -> tuple[dict[int, list[Event]], dict[int, SnakeState]]:
    """Play one episode, returning events and states indexed by tick."""
    state = game.reset()
    events_by_tick: dict[int, list[Event]] = {}
    states_by_tick: dict[int, SnakeState] = {0: state}

    for _ in range(max_ticks):
        if not state.alive:
            break
        legal = game.legal_actions(state)
        action = agent.act(state, legal)
        state, events, done = game.step(action)
        events_by_tick[state.tick] = events
        states_by_tick[state.tick] = state
        if done:
            break

    return events_by_tick, states_by_tick


def collect_episodes(
    n: int,
    agent_mix: list[tuple[object, float]],
    width: int = 10,
    height: int = 10,
    codec: EventCodec | None = None,
    max_ticks: int = 200,
    seed: int = 42,
) -> list[list[int]]:
    """Collect n tokenized episodes from a weighted agent mix.

    agent_mix: list of (agent, weight) pairs.
    Returns list of token sequences.
    """
    if codec is None:
        codec = EventCodec()

    rng = _random.Random(seed)
    agents = [a for a, _ in agent_mix]
    weights = [w for _, w in agent_mix]
    episodes: list[list[int]] = []

    for i in range(n):
        agent = rng.choices(agents, weights=weights)[0]
        game = SnakeGame(width=width, height=height, seed=rng.randint(0, 2**31))
        events_by_tick, states_by_tick = play_episode(game, agent, max_ticks=max_ticks)
        tokens = codec.encode_episode(events_by_tick, states_by_tick)
        episodes.append(tokens)

    return episodes
