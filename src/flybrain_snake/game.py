"""Deterministic Snake rules with intentional pass-through body contact."""
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum
import random

from .config import Rewards

Point = tuple[int, int]
VECTORS = ((0, -1), (1, 0), (0, 1), (-1, 0))


class Direction(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    def turn(self, relative: int):
        return Direction((int(self) + relative) % 4)


@dataclass(frozen=True)
class Reward:
    apple: float = 0.0
    collision: float = 0.0
    distance: float = 0.0
    wall: float = 0.0
    idle: float = 0.0
    route: float = 0.0
    completion: float = 0.0
    avoidance: float = 0.0
    practice_failure: float = 0.0
    edge_contact: float = 0.0

    def __add__(self, other: "Reward") -> "Reward":
        return Reward(**{name: value + getattr(other, name) for name, value in vars(self).items()})

    @property
    def total(self):
        return sum(vars(self).values())


@dataclass(frozen=True)
class Observation:
    head: Point
    apple: Point | None
    direction: Direction
    size: int
    walls: bool

    def wall_distances(self, point: Point) -> tuple[int, int, int, int]:
        x, y = point
        return y, self.size - 1 - x, self.size - 1 - y, x

    def wall_distance(self, direction: Direction) -> int:
        return self.wall_distances(self.head)[direction]

    def edge_distance(self) -> int:
        x, y = self.head
        return min(x, y, self.size - 1 - x, self.size - 1 - y)

    def food_distance(self) -> int:
        if self.apple is None:
            return 0
        dx, dy = (abs(a - b) for a, b in zip(self.head, self.apple, strict=True))
        return min(dx, self.size - dx) + min(dy, self.size - dy) if not self.walls else dx + dy


    def shortest_food_path(self) -> int:
        if self.apple is None:
            return 0
        dx, dy = self.apple[0] - self.head[0], self.apple[1] - self.head[1]
        fx, fy = VECTORS[self.direction]
        forward = dx * fx + dy * fy
        lateral = dx * fy - dy * fx
        if lateral or not forward:
            return self.food_distance()
        # A first move directly backward is illegal. A one-cell flank adds two moves.
        if self.walls:
            return forward if forward > 0 else -forward + 2
        forward %= self.size
        return min(forward, self.size - forward + 2)


class Snake:
    def __init__(self, size: int = 7, seed: int = 1, stage: int = 4,
                 max_steps: int = 100, rewards: Rewards | None = None, food_partition: str = "all",
                 *, allow_edge_food: bool = True):
        if size < 4 or stage not in (2, 3, 4) or max_steps < 0:
            raise ValueError("Tablero, etapa o límite inválidos")
        self.size, self.seed, self.stage, self.max_steps = size, seed, stage, max_steps
        self.rewards = rewards or Rewards()
        self.food_partition = food_partition
        self.allow_edge_food = allow_edge_food
        self.rng = random.Random(seed)
        x = y = size // 2
        self.body = [(x, y), (x - 1, y)] if stage == 4 else [(x, y)]
        self.direction = Direction.RIGHT
        self.apples = self.steps = 0
        self.end = ""
        self.route_moves = self.route_start = self.route_deviations = 0
        self.best_food_distance = None
        self.edge_exit_moves = 0
        self.apple = self._food()

    def _food(self):
        used = set(self.body)
        partition = {"all": None, "train": 0, "heldout": 1}[self.food_partition]
        start, stop = (0, self.size) if self.allow_edge_food else (1, self.size - 1)
        free = [(x, y) for y in range(start, stop) for x in range(start, stop)
                if (x, y) not in used and (partition is None or (x + y) % 2 == partition)]
        if free:
            return self.rng.choice(free)
        # Until the apple goal is reached, release the first eligible tail cell
        # if growth has occupied every possible food tile.
        if self.stage == 4:
            occupied = Counter(self.body)
            while len(self.body) > 1:
                x, y = self.body.pop()
                occupied[x, y] -= 1
                interior = self.allow_edge_food or (0 < x < self.size - 1 and 0 < y < self.size - 1)
                if interior and not occupied[x, y] and (partition is None or (x + y) % 2 == partition):
                    return self.rng.choice([(x, y)])
        return None

    def restrict_food_to_interior(self):
        self.allow_edge_food = False
        if self.apple is not None and min(self.observe().wall_distances(self.apple)) == 0:
            self.apple = self._food()
            self.route_moves = self.route_start = self.route_deviations = 0
            self.best_food_distance = None
            self.edge_exit_moves = 0
            return True
        return False

    def observe(self):
        return Observation(self.body[0], self.apple, self.direction, self.size, self.stage >= 3)

    def step(self, action: Direction) -> Reward:
        if self.end:
            raise ValueError("La partida ya ha terminado")
        previous = self.observe()
        action = Direction(action)
        if action != self.direction.turn(2):
            self.direction = action
        x, y = self.body[0]
        dx, dy = VECTORS[self.direction]
        x, y = x + dx, y + dy
        self.steps += 1
        if self.stage == 2:
            x, y = x % self.size, y % self.size
        elif not (0 <= x < self.size and 0 <= y < self.size):
            self.end = "pared"
            return Reward(collision=self.rewards.collision)

        ate = (x, y) == self.apple
        self.body.insert(0, (x, y))
        if not (ate and self.stage == 4):
            self.body.pop()
        current = self.observe()
        shortest_before = previous.shortest_food_path()
        remaining = current.shortest_food_path()
        best = shortest_before if self.best_food_distance is None else self.best_food_distance
        improved = remaining < best
        self.best_food_distance = min(best, remaining)
        distance = 0.0 if ate else self.rewards.distance * (best - self.best_food_distance)
        follows_shortest_path = remaining + 1 == shortest_before
        apple_on_edge = (previous.walls and previous.apple is not None
                         and 0 in previous.wall_distances(previous.apple))
        wall = 0.0
        if current.walls and not apple_on_edge and not self.edge_exit_moves:
            wall = self._wall_penalty(previous, current, follows_shortest_path)
        self.edge_exit_moves = max(0, self.edge_exit_moves - 1)
        route = self._route_reward(shortest_before, remaining, ate, improved)
        idle = -self.rewards.idle if not ate and distance <= 0 else 0.0
        if wall < 0:
            distance = idle = 0.0
            route = min(0.0, route)
        completion = 0.0
        if ate:
            if apple_on_edge:
                self.edge_exit_moves = 2
            self.route_deviations = 0
            self.best_food_distance = None
            self.apples += 1
            if self.apples >= self.size**2:
                self.apple = None
                self.end = "tablero completo"
                completion = self.rewards.completion
            else:
                self.apple = self._food()
                if self.apple is None:
                    self.end = "reserva agotada"
        if not self.end and self.max_steps and self.steps >= self.max_steps:
            self.end = "límite"
        return Reward(self.rewards.apple if ate else 0.0, distance=distance, wall=wall,
                      idle=idle, route=route, completion=completion)

    def _route_reward(self, before: int, remaining: int, ate: bool, improved: bool):
        if not self.rewards.route or not before:
            return 0.0
        penalty = 0.0
        if remaining + 1 != before and self.route_deviations < 2:
            self.route_deviations += 1
            if self.route_deviations == 2:
                penalty = -self.rewards.route
        if not self.route_moves:
            self.route_start = before
        self.route_moves += 1
        if self.route_moves < 3 and not ate:
            return penalty
        extra = self.route_moves + remaining - self.route_start
        progress = improved and remaining < self.route_start
        self.route_moves = 0
        if penalty:
            return penalty
        if progress and extra == 0:
            return self.rewards.route
        if progress and extra <= 2:
            return self.rewards.route / 2
        return 0.0

    def _wall_penalty(self, previous: Observation, current: Observation, follows_shortest_path: bool) -> float:
        distance = current.edge_distance()
        if distance == 0:
            return -self.rewards.wall

        # Leave at least the central strip outside the warning zone on small boards.
        warning_range = min(3, (current.size - 3) // 2)
        if distance > warning_range:
            return 0.0

        fruit_clearances = (current.wall_distances(current.apple) if follows_shortest_path and current.apple is not None
                            else (current.size,) * 4)
        clearances = zip(previous.wall_distances(previous.head), current.wall_distances(current.head),
                         fruit_clearances, strict=True)
        nearest = min((after for before, after, fruit in clearances if after <= before and after < fruit),
                      default=warning_range + 1)
        proximity = max(0.0, 1 - nearest / (warning_range + 1))
        return -self.rewards.wall * proximity

    def board_state(self):
        return {"size": self.size, "body": tuple(self.body), "apple": self.apple,
                "direction": int(self.direction), "apples": self.apples, "steps": self.steps, "end": self.end}

    def state(self):
        return {"size": self.size, "seed": self.seed, "stage": self.stage, "max_steps": self.max_steps,
                "food_partition": self.food_partition, "allow_edge_food": self.allow_edge_food,
                "body": self.body, "apple": self.apple, "direction": int(self.direction),
                "apples": self.apples, "steps": self.steps, "end": self.end, "rng": self.rng.getstate(),
                "route_moves": self.route_moves, "route_start": self.route_start,
                "route_deviations": self.route_deviations, "best_food_distance": self.best_food_distance,
                "edge_exit_moves": self.edge_exit_moves}

    @classmethod
    def restore(cls, state: dict, rewards: Rewards):
        game = cls(state["size"], state["seed"], state["stage"], state["max_steps"], rewards,
                   state.get("food_partition", "all"), allow_edge_food=state.get("allow_edge_food", True))
        game.body = [tuple(cell) for cell in state["body"]]
        game.apple = tuple(state["apple"]) if state["apple"] is not None else None
        game.direction = Direction(state["direction"])
        game.apples, game.steps, game.end = state["apples"], state["steps"], state["end"]
        game.route_moves = state.get("route_moves", 0)
        game.route_start = state.get("route_start", 0)
        game.route_deviations = state.get("route_deviations", 0)
        game.best_food_distance = state.get("best_food_distance")
        game.edge_exit_moves = state.get("edge_exit_moves", 0)
        version, internal, gaussian = state["rng"]
        game.rng.setstate((version, tuple(internal), gaussian))
        return game
