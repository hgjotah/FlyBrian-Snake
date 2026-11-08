"""Turn-synchronized competition against a human player."""
from __future__ import annotations

import numpy as np

from .game import Direction, Snake
from .training import Trainer


DEFAULT_APPLE_GOAL = 50


def apple_capacity(size: int) -> int:
    return size**2


class Duel:
    def __init__(self, brian: Trainer, seed: int, apple_goal: int):
        cfg = brian.config
        if not 1 <= apple_goal <= apple_capacity(cfg.size):
            raise ValueError("La meta de manzanas debe caber en el tablero")
        self.brian = brian
        self.apple_goal = apple_goal
        brian.begin(seed)
        brian.game.max_steps = 0
        self.human = Snake(cfg.size, seed, cfg.stage, 0, cfg.rewards, cfg.food_partition)
        self.turn = 0
        self.activity = ()
        self.winner = self.result = None

    def advance(self, direction: Direction, on_player_move=None):
        if self.winner is not None:
            return
        self.human.step(direction)
        if on_player_move is not None:
            on_player_move(self.human.board_state())
        self.brian.step(detail=False)
        self.activity = (np.flatnonzero(self.brian.decision_counts),
                         np.flatnonzero(self.brian.outcome_counts))
        self.turn += 1
        brian, human = self.brian.game, self.human
        brian_crash, human_crash = brian.end == "pared", human.end == "pared"
        brian_goal, human_goal = brian.apples >= self.apple_goal, human.apples >= self.apple_goal
        if brian_crash or human_crash:
            self.winner = "Empate" if brian_crash and human_crash else "Tú" if brian_crash else "Brian"
            self.result = "choque"
        elif brian_goal or human_goal:
            self.winner = "Empate" if brian_goal and human_goal else "Brian" if brian_goal else "Tú"
            self.result = "meta"

    def snapshot(self):
        return {"brian": self.brian.game.board_state(), "human": self.human.board_state(),
                "turn": self.turn, "winner": self.winner, "result": self.result,
                "seed": self.brian.game.seed,
                "apple_goal": self.apple_goal, "activity": self.activity,
                "reinforcement_hz": self.brian.last_step.get("reinforcement_hz", {}),
                "synaptic_signal": self.brian.config.synaptic_signal}
