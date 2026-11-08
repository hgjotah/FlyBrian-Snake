from dataclasses import replace

import numpy as np
import pytest

from flybrain_snake.bridge import reinforcement
from flybrain_snake.config import RunConfig
from flybrain_snake.experiments import trainer_for
from flybrain_snake.game import Direction, Reward
from flybrain_snake.storage import checkpoint, restore


@pytest.mark.parametrize("head,direction,edge", (
    ((1, 3), Direction.LEFT, (0, 3)), ((5, 3), Direction.RIGHT, (6, 3)),
    ((3, 1), Direction.UP, (3, 0)), ((3, 5), Direction.DOWN, (3, 6)),
))
def test_touching_any_outer_row_or_column_ends_before_the_collision(laboratory, head, direction, edge):
    t = laboratory
    t.config = replace(t.config, practice="edge_reset", max_steps=0)
    t.begin(11)
    t.game.body = [head]
    t.game.direction = direction
    t.game.apple = (2, 2)
    t.game.edge_exit_moves = 2  # This mode has no perimeter exemption.
    trace = t.step(direction)
    assert trace["head"] == edge and trace["end"] == "borde alcanzado"
    assert trace["total"] == trace["delivered"]["edge_contact"] == -120
    assert trace["delivered"]["collision"] == trace["delivered"]["wall"] == 0
    assert set(trace["reinforcement_hz"]) == {"aversive"}
    ordinary, = reinforcement(Reward(collision=-60), t.sim.circuit.groups, t.config.reinforcement_hz)
    assert trace["reinforcement_hz"]["aversive"] > 1.8 * ordinary.rate
    assert t.finish()["edge_termination"] == 1
    t.begin(12)
    assert t.game.steps == t.game.apples == 0 and t.game.observe().edge_distance() > 0


def test_one_cell_before_the_outermost_column_still_plays_normally(laboratory):
    t = laboratory
    t.config = replace(t.config, practice="edge_reset", max_steps=0)
    t.begin(11)
    t.game.apple = (1, 1)
    assert t.step(Direction.RIGHT)["end"] == ""
    assert t.step(Direction.RIGHT)["end"] == ""
    assert t.game.observe().edge_distance() == 1
    resumed = restore(checkpoint(t), t.sim.graph, t.sim.circuit)
    first, copy = t.step(Direction.RIGHT), resumed.step(Direction.RIGHT)
    assert first["end"] == copy["end"] == "borde alcanzado"
    assert first["delivered"] == copy["delivered"]
    np.testing.assert_array_equal(t.sim.voltage, resumed.sim.voltage)


def test_regular_games_keep_their_original_wall_rule(laboratory):
    t = laboratory
    t.begin(11)
    t.game.body = [(5, 3), (4, 3)]
    t.game.apple = (1, 1)
    trace = t.step(Direction.RIGHT)
    assert trace["head"] == (6, 3) and trace["end"] == ""
    assert trace["delivered"]["edge_contact"] == 0
    assert t.step(Direction.RIGHT)["end"] == "pared"


def test_new_mode_starts_like_a_normal_game_and_keeps_interior_food(laboratory):
    source = laboratory
    cfg = replace(source.config, practice="edge_reset", max_steps=0)
    strict = trainer_for(source.sim.graph, source.sim.circuit, cfg)
    normal = trainer_for(source.sim.graph, source.sim.circuit, replace(cfg, practice="snake"))
    strict.begin(12)
    normal.begin(12)
    assert strict.game.state() == normal.game.state()
    assert strict.game.observe().edge_distance() > 0
    assert min(strict.game.observe().wall_distances(strict.game.apple)) > 0
    with pytest.raises(ValueError):
        RunConfig(practice="edge_reset", stage=3)


def test_training_automatically_restarts_after_delivering_the_border_punishment(laboratory, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from flybrain_snake import desktop_worker
    t = laboratory
    monkeypatch.setattr(desktop_worker, "gate", lambda *_: None)
    w = desktop_worker.Worker(tmp_path)
    w.graph, w.circuit = t.sim.graph, t.sim.circuit
    w.active_scales = t.sim.scales.copy()
    w.active_config = t.config
    w.saved_scales, w.saved_config = w.active_scales.copy(), t.config
    try:
        w._command("new", (replace(t.config, practice="edge_reset", max_steps=0), 2, True))
        for _ in range(6):
            w._advance(None, False)
        rows = list(w.journal.rows("episodes"))
        assert len(rows) == 2
        assert all(row["steps"] == 3 and row["edge_termination"] == 1 and not row["collision"] for row in rows)
        endings = [step for step in w.journal.rows("steps") if step["end"]]
        assert len(endings) == 2 and all(step["delivered"]["edge_contact"] == -120 for step in endings)
        assert not w.running and w.trainer.game is None
        assert w.trainer.sim.scales is w.active_scales
    finally:
        w._close_session()
