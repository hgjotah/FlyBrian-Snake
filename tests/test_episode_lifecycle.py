from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from flybrain_snake.experiments import trainer_for
from flybrain_snake.game import Reward
from flybrain_snake.simulator import Stimulus
from flybrain_snake.storage import checkpoint, restore


def test_new_game_matches_a_fresh_brain_after_heavy_activity(laboratory):
    t = laboratory
    t.config = replace(t.config, sensory_hz=90, plasticity=False)
    t.sim.scales[:] = [.75, .875]
    t.begin(90)
    t.step()
    t.sim.advance(500, (Stimulus("old cue", t.sim.circuit.groups["KC"], 200),
                        Stimulus("old reward", t.sim.circuit.groups["appetitive"], 180)))
    t.delayed = [Reward(collision=-60)]
    assert t.decision_counts.any() or t.outcome_counts.any()
    before = t.sim.scales.copy()
    t.begin(171)
    fresh = trainer_for(t.sim.graph, t.sim.circuit, t.config)
    fresh.sim.scales[:] = before
    fresh.begin(171)
    assert not t.decision_counts.any() and not t.outcome_counts.any()
    assert not t.delayed and not t.last_step and not t.interventions
    assert t.game.state() == fresh.game.state()
    for _ in range(2):
        actual, expected = t.step(shuffled=True), fresh.step(shuffled=True)
        for key in ("decision", "head", "apple", "reward", "delivered", "groups"):
            assert actual[key] == expected[key]
        np.testing.assert_array_equal(t.sim.voltage, fresh.sim.voltage)
        np.testing.assert_array_equal(t.sim.drive, fresh.sim.drive)
    np.testing.assert_array_equal(t.sim.scales, before)


def test_frozen_games_do_not_depend_on_the_order_of_previous_games(laboratory):
    source = laboratory
    cfg = replace(source.config, plasticity=False, sensory_hz=90)
    def play(seeds):
        t = trainer_for(source.sim.graph, source.sim.circuit, cfg)
        output = {}
        for seed in seeds:
            t.begin(seed)
            steps = []
            while not t.game.end:
                row = t.step(detail=False)
                steps.append((row["head"], row["apple"], row["decision"], row["reward"]))
            t.finish()
            output[seed] = steps
        return output
    assert play((91, 13, 207)) == play((207, 13, 91))


def test_checkpoint_keeps_pending_rewards_within_the_current_game(laboratory):
    t = laboratory
    t.begin(101)
    t.delayed = [Reward(apple=30), Reward(collision=-60)]
    resumed = restore(checkpoint(t), t.sim.graph, t.sim.circuit)
    assert resumed.delayed == t.delayed
    for _ in range(2):
        actual, expected = resumed.step(shuffled=True), t.step(shuffled=True)
        assert actual["delivered"] == expected["delivered"]
        assert actual["decision"] == expected["decision"]
        np.testing.assert_array_equal(resumed.sim.scales, t.sim.scales)
        np.testing.assert_array_equal(resumed.sim.voltage, t.sim.voltage)


@pytest.fixture
def worker(laboratory, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from flybrain_snake import desktop_worker
    t = laboratory
    monkeypatch.setattr(desktop_worker, "gate", lambda *_: None)
    w = desktop_worker.Worker(tmp_path)
    w.graph, w.circuit = t.sim.graph, t.sim.circuit
    w.active_scales = t.sim.scales.copy()
    w.active_config = t.config
    w.saved_scales, w.saved_config = w.active_scales.copy(), w.active_config
    w._command("new", (t.config, 3, True))
    yield w
    w._close_session()


def test_resume_applies_learning_switches_without_restarting_the_board(worker):
    w = worker
    t = w.trainer
    t.begin(12)
    game = t.game
    before = t.sim.scales.copy()
    frozen = replace(t.config, plasticity=False, dopamine=False)
    w._command("resume", (3, frozen))
    assert t.game is game and game.steps == 0
    assert not t.config.plasticity and not t.config.dopamine
    assert w.active_config == t.config
    game.apple = (4, 3)
    trace = t.step()
    assert trace["reward"]["apple"] == 30
    assert not trace["plasticity_enabled"] and not trace["reinforcement_hz"]
    np.testing.assert_array_equal(t.sim.scales, before)
    w._command("resume", (3, replace(frozen, dopamine=True)))
    game.apple = (5, 3)
    trace = t.step()
    assert not trace["plasticity_enabled"] and trace["reinforcement_hz"]["appetitive"] > 0
    np.testing.assert_array_equal(t.sim.scales, before)


def test_failed_load_does_not_leave_a_dead_session_running(worker, tmp_path):
    w = worker
    cfg = w.trainer.config
    with pytest.raises(FileNotFoundError):
        w._command("load", (tmp_path / "missing.sqlite", 3))
    assert w.trainer is None and w.journal is None and not w.running
    w._command("resume", (3, cfg))
    assert w.running and w.trainer is not None and w.journal is not None


def test_learning_switches_apply_to_single_steps_immediately(worker):
    w = worker
    w.trainer.begin(12)
    game = w.trainer.game
    game.apple = (4, 3)
    before = w.active_scales.copy()
    w._command("learning_switches", (False, False))
    w._command("step", ())
    assert game.steps == 1 and w.trainer.last_step["reward"]["apple"] == 30
    assert not w.trainer.last_step["plasticity_enabled"]
    assert not w.trainer.last_step["reinforcement_hz"]
    np.testing.assert_array_equal(w.active_scales, before)


def test_return_from_arena_preserves_board_and_weights_but_clears_old_activity(worker):
    w = worker
    t = w.trainer
    t.config = replace(t.config, sensory_hz=90)
    t.begin(121)
    t.step()
    original_board = t.game.state()
    t.delayed = [Reward(collision=-60)]
    w.extra = Stimulus("pending", t.sim.circuit.groups["KC"], 90)
    w.extra_reward = Reward(apple=30)
    w.atlas = SimpleNamespace(active_contacts=lambda _: ())
    w._duel_command("duel_start", (7, t.config, 7, 10, t.config.rewards))
    assert w.duel is not None and w.extra is None and w.extra_reward is None
    w._command("learning_switches", (False, False))
    assert not w.duel.brian.config.plasticity and not w.duel.brian.config.dopamine
    w.active_scales[:] = [.5, .6]
    w._duel_command("duel_cancel", (7,))
    assert t.game.state() == original_board
    assert t.sim.scales is w.active_scales
    np.testing.assert_array_equal(t.sim.scales, np.array([.5, .6], np.float32))
    assert not t.sim.drive.any() and not t.sim.ring_counts.any()
    assert (t.sim.voltage == t.sim.lif.rest).all() and (t.sim.last == -1).all()
    assert not t.delayed and not t.last_step
    assert not t.decision_counts.any() and not t.outcome_counts.any()
    assert t.interventions
    assert list(w.journal.rows("events"))[-1]["kind"] == "arena_return"
