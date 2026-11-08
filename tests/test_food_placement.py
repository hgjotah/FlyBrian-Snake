from dataclasses import replace

import numpy as np
import pytest

from flybrain_snake.arena import Duel
from flybrain_snake.game import Snake
from flybrain_snake.storage import checkpoint, restore


@pytest.mark.parametrize("practice", ("snake", "routes", "walls"))
@pytest.mark.parametrize("size", (4, 7, 10))
def test_all_brian_training_setups_keep_apples_off_the_perimeter(laboratory, practice, size):
    t = laboratory
    t.config = replace(t.config, practice=practice, size=size, max_steps=20)
    for episode in range(24):
        t.episode = episode
        t.begin(101 + episode)
        assert not t.game.allow_edge_food
        x, y = t.game.apple
        assert 0 < x < size - 1 and 0 < y < size - 1
        assert t.game.apple not in t.game.body
        for _ in range(10):
            x, y = t.game._food()
            assert 0 < x < size - 1 and 0 < y < size - 1


@pytest.mark.parametrize("partition", ("all", "train", "heldout"))
def test_full_interior_releases_an_eligible_interior_tail_cell(partition):
    game = Snake(4, food_partition=partition, allow_edge_food=False)
    game.body = [(1, 1), (0, 0), (2, 1), (2, 2), (1, 2), (0, 1)]
    apple = game._food()
    assert apple is not None and apple not in game.body
    assert all(0 < coordinate < 3 for coordinate in apple)
    if partition != "all":
        assert sum(apple) % 2 == (partition == "heldout")


def test_arena_keeps_edge_apples_available_only_to_the_human(laboratory, monkeypatch):
    duel = Duel(laboratory, 53, 30)
    assert not duel.brian.game.allow_edge_food and duel.human.allow_edge_food
    x, y = duel.brian.game.apple
    assert 0 < x < 6 and 0 < y < 6
    monkeypatch.setattr(duel.human.rng, "choice", lambda cells:
                        next(cell for cell in cells if 0 in duel.human.observe().wall_distances(cell)))
    assert min(duel.human.observe().wall_distances(duel.human._food())) == 0
    restored = Snake.restore(duel.human.state(), duel.human.rewards)
    assert restored.allow_edge_food


def test_loading_a_brian_session_replaces_an_existing_edge_apple_without_losing_weights(laboratory):
    t = laboratory
    t.begin(11)
    t.game.allow_edge_food = True  # Historical checkpoint made before the restriction.
    t.game.apple = (0, 0)
    t.sim.scales[:] = [.75, .875]
    loaded = restore(checkpoint(t), t.sim.graph, t.sim.circuit)
    assert not loaded.game.allow_edge_food
    assert min(loaded.game.observe().wall_distances(loaded.game.apple)) > 0
    np.testing.assert_array_equal(loaded.sim.scales, t.sim.scales)
    copy = restore(checkpoint(loaded), t.sim.graph, t.sim.circuit)
    assert copy.game.state() == loaded.game.state()
