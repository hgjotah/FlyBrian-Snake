from itertools import product

import pytest

from flybrain_snake.config import Rewards
from flybrain_snake.game import Direction, Snake, VECTORS


@pytest.mark.parametrize("direction", list(Direction))
def test_cardinal_motion(direction):
    g = Snake()
    g.direction = direction
    x, y = g.body[0]
    dx, dy = VECTORS[direction]
    g.step(direction)
    assert g.body[0] == (x + dx, y + dy)


def test_reverse_and_wall():
    g = Snake(7)
    g.apple = (0, 0)
    assert g.step(Direction.LEFT).collision == 0
    assert g.body[0] == (4, 3)
    for _ in range(3):
        result = g.step(Direction.RIGHT)
    assert g.end == "pared"
    assert result.collision == -60


def test_own_body_is_pass_through():
    g = Snake()
    g.body = [(3, 3), (3, 4), (4, 4), (4, 3), (4, 2)]
    g.apple = (6, 6)
    reward = g.step(Direction.RIGHT)
    assert g.body.count((4, 3)) == 2
    assert not g.end
    assert not reward.collision


def test_food_growth_and_filled_board():
    g = Snake(4)
    g.body = [(2, 2)] + [p for p in product(range(4), repeat=2) if p not in [(2, 2), (3, 2)]]
    g.apple = (3, 2)
    g.apples = 15
    reward = g.step(Direction.RIGHT)
    assert len(g.body) == 16
    assert reward.apple == 30
    assert reward.completion == 150
    assert g.apples == 16
    assert g.end == "tablero completo"
    assert g.apple is None


def test_near_wall_apple_has_no_wall_deduction():
    g = Snake()
    g.body = [(5, 3), (4, 3)]
    g.apple = (6, 3)
    r = g.step(Direction.RIGHT)
    assert r.apple == 30 and r.wall == 0
    assert r.total >= 30
    assert g.apple not in g.body


def test_reproducible_rng_and_torus():
    g, h = Snake(seed=71, stage=2), Snake(seed=71, stage=2)
    for d in [Direction.RIGHT] * 20:
        assert g.apple == h.apple
        assert g.step(d) == h.step(d)
        assert g.body == h.body
    assert g.end == ""
    assert g.steps == 20


@pytest.mark.parametrize("profile", ("edge_target", "legacy"))
def test_wall_penalty_ramps_up_across_three_cells(profile):
    game = Snake(11, seed=1, rewards=Rewards(profile=profile))
    game.apple = (1, 5)
    assert [game.step(Direction.RIGHT).wall for _ in range(5)] == [0, -3, -6, -9, -12]
    assert game.step(Direction.RIGHT).total == -60


def test_direct_approach_to_an_edge_apple_is_exempt():
    game = Snake()
    game.apple = (6, 3)
    rewards = [game.step(Direction.RIGHT) for _ in range(3)]
    assert all(reward.wall == 0 and reward.total > 0 for reward in rewards)
    assert rewards[-1].apple == 30


def test_an_edge_apple_excuses_touching_any_wall():
    game = Snake(11)
    game.body = [(0, 1), (0, 2)]
    game.direction = Direction.UP
    game.apple = (6, 0)
    assert game.step(Direction.UP).wall == 0
    assert game.step(Direction.RIGHT).wall == 0


def test_edge_apple_grants_two_exit_moves_after_it_is_collected(monkeypatch):
    game = Snake(11)
    game.body = [(3, 1), (3, 2)]
    game.direction = Direction.UP
    game.apple = (3, 0)
    monkeypatch.setattr(game, "_food", lambda: (5, 5))

    collected = game.step(Direction.UP)
    assert collected.apple == 30 and collected.wall == 0
    assert game.step(Direction.RIGHT).wall == 0
    game = Snake.restore(game.state(), game.rewards)
    assert game.step(Direction.RIGHT).wall == 0
    assert game.step(Direction.RIGHT).wall == -12


def test_wall_collision_applies_during_edge_exit(monkeypatch):
    game = Snake(11)
    game.body = [(1, 1), (2, 1)]
    game.direction = Direction.LEFT
    game.apple = (0, 1)
    monkeypatch.setattr(game, "_food", lambda: (5, 5))

    assert game.step(Direction.LEFT).apple == 30
    assert game.step(Direction.UP).wall == 0
    assert game.step(Direction.UP).collision == -60


def test_short_step_that_overshoots_apple_clearance_is_penalized():
    game = Snake(11)
    game.body = [(1, 5), (1, 6)]
    game.direction = Direction.DOWN
    game.apple = (1, 2)
    assert game.step(Direction.LEFT).wall == -12


def test_board_stays_playable_until_eating_one_apple_per_tile():
    g = Snake(4)
    g.body = [(2, 2)] + [p for p in product(range(4), repeat=2) if p not in [(2, 2), (3, 2)]]
    g.apple = (3, 2)
    g.apples = 13
    assert not g.step(Direction.RIGHT).completion
    assert not g.end and g.apple is not None and g.apple not in g.body


def test_short_legal_routes_and_detours():
    direct = Snake(15)
    direct.apple = (11, 7)
    assert direct.observe().shortest_food_path() == 4
    assert [direct.step(Direction.RIGHT).route for _ in range(3)] == [0, 0, 2]

    behind = Snake(15)
    behind.apple = (5, 7)
    assert behind.observe().shortest_food_path() == 4
    assert [behind.step(d).route for d in (Direction.UP, Direction.LEFT, Direction.LEFT)] == [0, 0, 2]

    longer = Snake(15)
    longer.apple = (10, 7)
    assert [longer.step(d).route for d in (Direction.UP, Direction.RIGHT, Direction.DOWN)] == [0, 0, 1]

    detour = Snake(15)
    detour.apple = (10, 7)
    assert [detour.step(d).route for d in (Direction.UP, Direction.LEFT, Direction.LEFT)] == [0, -2, 0]
    restored = Snake.restore(longer.state(), longer.rewards)
    assert restored.state() == longer.state()


def test_route_penalty_is_saved_and_only_rearms_after_an_apple():
    game = Snake(15)
    game.apple = (10, 7)
    assert game.step(Direction.UP).route == 0
    game = Snake.restore(game.state(), game.rewards)
    assert game.step(Direction.LEFT).route == -2
    game = Snake.restore(game.state(), game.rewards)
    detours = (Direction.LEFT, Direction.DOWN, Direction.LEFT, Direction.DOWN)
    assert all(game.step(direction).route >= 0 for direction in detours)
    for _ in range(6):
        assert game.step(Direction.RIGHT).route >= 0
    assert game.step(Direction.UP).apple == 30
    game.apple = (10, 4)
    assert game.step(Direction.RIGHT).route == 0
    assert game.step(Direction.DOWN).route == -2
