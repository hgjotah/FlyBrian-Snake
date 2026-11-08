from dataclasses import replace
import json

import numpy as np
import pytest

from flybrain_snake.bridge import reinforcement, sensory
from flybrain_snake.game import Direction, Reward, Snake, VECTORS
from flybrain_snake.storage import checkpoint, restore
from flybrain_snake.wall_study import run


@pytest.mark.parametrize("size", (4, 7, 10, 100))
def test_wall_curriculum_balances_sides_and_keeps_valid_starts(laboratory, size):
    t = laboratory
    t.config = replace(t.config, size=size, practice="walls", max_steps=20)
    seen = []
    for index in range(24):
        t.episode = index
        t.begin(101 + index)
        seen.append(t.stats["wall_case"])
        assert t.game.apple not in t.game.body
        assert all(0 <= x < size and 0 <= y < size for x, y in (*t.game.body, t.game.apple))
        assert t.stats["wall_protocol"] == 2
        assert t.game.observe().wall_distance(t.game.direction) == 0
        dx, dy = VECTORS[t.game.direction]
        x, y = t.game.body[0]
        assert not (0 <= x + dx < size and 0 <= y + dy < size)
        assert not t.stats["wall_escaped"]
    assert np.bincount(seen).tolist() == [4] * 6


def test_wall_memory_cue_is_graded_and_uses_neuronal_input(laboratory):
    game = Snake(11)
    game.body[0] = (7, 5)
    groups = laboratory.sim.circuit.groups
    cues = {cue.name: cue for cue in sensory(game.observe(), groups, 90, wall_memory_hz=180)}
    assert cues["wall_memory"].rate == 45
    np.testing.assert_array_equal(cues["wall_memory"].indices, groups["conditioning_A"])
    assert "wall_memory" not in {cue.name for cue in sensory(game.observe(), groups, 90)}
    game.body[0] = (6, 5)
    assert "wall_memory" not in {cue.name for cue in sensory(game.observe(), groups, 90, wall_memory_hz=180)}


@pytest.mark.parametrize("heading", tuple(Direction))
@pytest.mark.parametrize("profile", ("edge_target", "edge_aware", "graded", "legacy"))
def test_boundary_preview_is_a_neuronal_input_before_touching_the_edge(laboratory, heading, profile):
    game = Snake(11)
    head, apple, direction = (9, 5), (4, 3), Direction.RIGHT
    for _ in range((int(heading) - int(Direction.RIGHT)) % 4):
        head = 10 - head[1], head[0]
        apple = 10 - apple[1], apple[0]
        direction = direction.turn(1)
    game.body[0], game.apple, game.direction = head, apple, direction
    groups = laboratory.sim.circuit.groups
    cues = {cue.name: cue for cue in sensory(game.observe(), groups, 90, profile=profile, wall_preview_cells=1)}
    assert game.observe().wall_distance(heading) == 1
    for side in ("left", "right"):
        cue = cues["boundary_" + side]
        assert cue.rate > 0 and cue.stream == "boundary"
        np.testing.assert_array_equal(cue.indices, groups["loom_" + side])
    assert "wall_memory" not in cues
    assert any(name.startswith("food_") for name in cues)
    game.body[0] = (5, 5)
    assert not any(cue.name.startswith("boundary_") for cue in
                   sensory(game.observe(), groups, 90, profile=profile, wall_preview_cells=1))


def test_boundary_preview_does_not_shift_the_random_stream_for_food(laboratory):
    from flybrain_snake.simulator import Stimulus
    t = laboratory
    t.begin(13)
    copy = restore(checkpoint(t), t.sim.graph, t.sim.circuit)
    food = Stimulus("food", t.sim.circuit.groups["KC"], 90)
    boundary = Stimulus("boundary", t.sim.circuit.groups["loom_left"], 50, stream="boundary")
    t.sim._events(600, (food, boundary))
    copy.sim._events(600, (food,))
    assert t.sim.rng.bit_generator.state == copy.sim.rng.bit_generator.state
    assert t.sim.boundary_rng.bit_generator.state != copy.sim.boundary_rng.bit_generator.state
    resumed = restore(checkpoint(t), t.sim.graph, t.sim.circuit)
    first = t.sim.advance(60, (boundary,))
    second = resumed.sim.advance(60, (boundary,))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(t.sim.voltage, resumed.sim.voltage)


def test_autonomous_failed_trials_are_punished_and_restart_until_block_finishes(laboratory, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from flybrain_snake import desktop_worker
    t = laboratory
    monkeypatch.setattr(desktop_worker, "gate", lambda *_: None)
    w = desktop_worker.Worker(tmp_path)
    w.graph, w.circuit = t.sim.graph, t.sim.circuit
    w.active_scales = t.sim.scales.copy()
    w.active_config = t.config
    w.saved_scales, w.saved_config = w.active_scales.copy(), t.config
    starts = []
    w.episode_started.connect(starts.append)
    try:
        w._command("new", (replace(t.config, practice="walls", max_steps=20), 3, True))
        for _ in range(3):
            w._advance(None, False)
        rows = list(w.journal.rows("episodes"))
        assert len(starts) == len(rows) == 3
        assert all(row["steps"] == 1 and row["collision"] == 1 and row["reward"] == -60 for row in rows)
        assert all(not row["manual"] and row["wall_success"] == 0 for row in rows)
        assert w.trainer.game is None and not w.running
    finally:
        w._close_session()


def test_wall_escape_reward_and_trial_state_survive_checkpoint(laboratory):
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=20)
    t.begin(11)
    t.game.body = [(5, 3), (4, 3)]
    t.game.direction = Direction.RIGHT
    t.game.apple = (3, 2)
    assert t.step(Direction.UP)["end"] == ""
    assert t.last_step["delivered"]["avoidance"] == 0
    other = restore(checkpoint(t), t.sim.graph, t.sim.circuit)
    for direction in (Direction.LEFT, Direction.LEFT):
        first, resumed = t.step(direction), other.step(direction)
        assert first["delivered"] == resumed["delivered"]
        np.testing.assert_array_equal(t.sim.scales, other.sim.scales)
    assert first["end"] == "pared evitada"
    assert first["apple"] is None and t.game.apple is None
    assert first["delivered"]["avoidance"] == t.config.rewards.avoidance
    assert first["reinforcement_hz"]["appetitive"] > 0
    ordinary, = reinforcement(Reward(apple=30), t.sim.circuit.groups, t.config.reinforcement_hz)
    assert first["reinforcement_hz"]["appetitive"] > 1.4 * ordinary.rate
    assert t.finish()["wall_success"] == 1


def test_wall_timeout_is_punished_and_does_not_count_as_escape(laboratory):
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=1)
    t.begin(11)
    t.game.apple = (1, 4)
    trace = t.step(Direction.UP)
    assert trace["end"] == "manzana no alcanzada"
    assert trace["delivered"]["practice_failure"] == -60
    assert trace["delivered"]["avoidance"] == 0
    assert trace["total"] < 0 and set(trace["reinforcement_hz"]) == {"aversive"}
    assert t.finish()["wall_success"] == 0
    t.begin(12)
    assert t.game.steps == 0 and not t.game.end and not t.stats["wall_escaped"]


def test_apple_after_escaping_on_last_allowed_step_gets_reward_and_new_trial(laboratory):
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=1)
    t.begin(11)
    t.game.apple = (t.game.body[0][0], t.game.body[0][1] - 1)
    trace = t.step(Direction.UP)
    assert trace["end"] == "pared evitada"
    assert trace["delivered"]["apple"] == 30 and trace["delivered"]["avoidance"] == 18
    assert trace["delivered"]["practice_failure"] == 0
    assert set(trace["reinforcement_hz"]) == {"appetitive"}
    assert t.finish()["wall_success"] == 1
    t.begin(12)
    assert t.game.apples == 0 and t.game.steps == 0 and not t.stats["wall_escaped"]


def test_getting_a_fruit_without_evading_does_not_receive_the_escape_prize(laboratory):
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=20)
    t.begin(11)
    t.game.body = [(5, 3), (4, 3)]
    x, y = t.game.body[0]
    t.game.apple = (x + 1, y)
    trace = t.step(Direction.RIGHT)
    assert trace["end"] == "evasión no realizada"
    assert trace["delivered"]["avoidance"] == 0 and trace["total"] < 0
    assert set(trace["reinforcement_hz"]) == {"aversive"}


def test_wall_collision_never_receives_escape_reward(laboratory):
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=20)
    t.begin(11)
    t.game.body = [(6, 3), (5, 3)]
    t.game.direction = Direction.RIGHT
    t.game.apple = (3, 3)
    t.stats.update(wall_sides=[int(Direction.RIGHT)], wall_target=3)
    trace = t.step(Direction.RIGHT)
    assert trace["delivered"]["collision"] == -60 and trace["delivered"]["avoidance"] == 0
    assert set(trace["reinforcement_hz"]) == {"aversive"}
    assert t.finish()["wall_success"] == 0


def test_every_generated_trial_collides_on_the_next_straight_move(laboratory):
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=20)
    for index in range(24):
        t.begin(21 + index)
        trace = t.step(t.game.direction)
        assert trace["end"] == "pared" and trace["step"] == 1
        assert trace["delivered"]["collision"] == -60
        assert trace["delivered"]["avoidance"] == trace["delivered"]["practice_failure"] == 0
        assert t.finish()["wall_success"] == 0


def test_wall_study_freezes_controls_and_uses_reserved_seeds(laboratory, tmp_path):
    t = laboratory
    report = run(t.sim.graph, t.sim.circuit, replace(t.config, max_steps=6),
                 t.sim.scales.copy(), tmp_path / "study", episodes=2, trials=24, progress=lambda _: None)
    for name in ("frozen", "no_dopamine"):
        condition = report["conditions"][name]
        assert condition["weights_changed"] == 0
        assert condition["trials"] == report["baseline"]["trials"]
    assert len(set(report["test_seeds"])) == 24
    assert not set(report["test_seeds"]) & set(report["validation_seeds"])
    chosen = report["conditions"]["trained"]
    assert chosen["validation_success_rate"] >= chosen["validation_history"][0]["success_rate"]
    assert len(report["conditions"]["trained"]["trials"]) == 24
    saved = json.loads((tmp_path / "study" / "report.json").read_text())
    assert saved["improvement_demonstrated"] == report["improvement_demonstrated"]


def test_validation_rejects_regression_and_keeps_shared_weights(laboratory):
    from flybrain_snake.wall_study import WallSelection
    t = laboratory
    results = iter((
        {"success_rate": .8, "collision_rate": .1},
        {"success_rate": .7, "collision_rate": .2},
        {"success_rate": .9, "collision_rate": .1},
        {"success_rate": .9, "collision_rate": .2},
    ))
    calls = []
    def measure(graph, circuit, config, scales, seeds):
        calls.append(tuple(seeds))
        return next(results)
    scales = t.sim.scales
    before = scales.copy()
    selection = WallSelection(t.sim.graph, t.sim.circuit, t.config, scales, measure=measure,
                              game_measure=lambda *_: {"collisions": 0, "apples": 3,
                                                       "survival_moves": 60, "trials": []})
    scales[:] = .5
    assert not selection.consider(scales, 24)["accepted"]
    np.testing.assert_array_equal(scales, before)
    scales[:] = .7
    assert selection.consider(scales, 48)["accepted"]
    scales[:] = .3
    assert not selection.consider(scales, 72)["accepted"]
    np.testing.assert_array_equal(scales, np.full_like(scales, .7))
    assert scales is t.sim.scales
    assert all(seeds == calls[0] for seeds in calls)


def test_leaving_wall_practice_preserves_validated_memory_and_full_games(laboratory, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from flybrain_snake import desktop_worker
    from flybrain_snake.storage import Journal
    from flybrain_snake.wall_study import WallSelection
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=20, wall_memory_hz=180)
    worker = desktop_worker.Worker(tmp_path)
    worker.graph, worker.circuit = t.sim.graph, t.sim.circuit
    worker.trainer, worker.active_config = t, t.config
    worker.active_scales = t.sim.scales
    worker.saved_scales, worker.saved_config = t.sim.scales.copy(), t.config
    measurements = iter(({"success_rate": .9, "collision_rate": .1},
                         {"success_rate": .8, "collision_rate": .2}))
    monkeypatch.setattr(desktop_worker, "WallSelection", lambda *args, **kwargs:
                        WallSelection(*args, measure=lambda *_: next(measurements)))
    with Journal(tmp_path / "practice.sqlite") as journal:
        worker.journal = journal
        worker._wall_check(initialize=True)
        before = t.sim.scales.copy()
        t.begin(91)
        t.sim.scales[:] = .3
        worker._set_practice("snake", 0)
        np.testing.assert_array_equal(worker.active_scales, before)
        assert t.config.wall_memory_hz == 0 and t.config.max_steps == 0
        assert t.sim.learning == t.config.learning
        assert t.game is None and worker.wall_selection is None
        assert list(journal.rows("events"))[0]["kind"] == "wall_candidate_held"
        t.begin(91)
        assert t.game.max_steps == 0


@pytest.mark.parametrize("clearance", (0, 1, 2))
def test_escape_to_another_edge_is_not_rewarded(laboratory, clearance):
    t = laboratory
    t.config = replace(t.config, size=10, practice="walls", max_steps=20)
    t.begin(91)
    t.game.body = [(6, clearance), (7, clearance)]
    t.game.direction = Direction.LEFT
    t.game.apple = (2, 5)
    t.game.edge_exit_moves = 0
    t.stats.update(wall_sides=[int(Direction.RIGHT)], wall_target=3)
    t.step(Direction.LEFT)
    trace = t.step(Direction.LEFT)
    assert trace["reward"]["wall"] < 0
    assert trace["reward"]["avoidance"] == 0
    assert trace["total"] < 0 and trace["end"] == ""
    assert set(trace["reinforcement_hz"]) == {"aversive"}


def test_forced_initial_escape_is_not_punished_for_starting_on_the_edge(laboratory):
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=20)
    t.begin(11)
    t.game.body = [(6, 3), (5, 3)]
    t.game.direction = Direction.RIGHT
    t.game.apple = (3, 3)
    first = t.step(Direction.UP)
    second = t.step(Direction.LEFT)
    assert first["reward"]["wall"] == second["reward"]["wall"] == 0
    assert first["reward"]["avoidance"] == second["reward"]["avoidance"] == 0
    assert t.game.edge_exit_moves == 0 and t.game.end == ""


def test_more_exercise_successes_cannot_compensate_for_more_collisions(laboratory):
    from flybrain_snake.wall_study import WallSelection
    t = laboratory
    measurements = iter(({"success_rate": .8, "collision_rate": .1},
                         {"success_rate": .9, "collision_rate": .2}))
    selector = WallSelection(t.sim.graph, t.sim.circuit, t.config, t.sim.scales,
                             measure=lambda *_: next(measurements),
                             game_measure=lambda *_: pytest.fail("Unsafe candidate reached transfer check"))
    original = t.sim.scales.copy()
    t.sim.scales[:] = .5
    result = selector.consider(t.sim.scales, 24)
    assert not result["accepted"]
    np.testing.assert_array_equal(t.sim.scales, original)


def test_improvement_on_exercises_is_rejected_if_full_games_regress(laboratory):
    from flybrain_snake.wall_study import WallSelection
    t = laboratory
    measurements = iter(({"success_rate": .8, "collision_rate": .1},
                         {"success_rate": .9, "collision_rate": 0.0}))
    games = iter(({"collisions": 0, "apples": 8, "survival_moves": 64,
                   "trials": [{"collision": 0}]},
                  {"collisions": 1, "apples": 2, "survival_moves": 10,
                   "trials": [{"collision": 1}]}))
    selector = WallSelection(t.sim.graph, t.sim.circuit, t.config, t.sim.scales,
                             measure=lambda *_: next(measurements), game_measure=lambda *_: next(games))
    original = t.sim.scales.copy()
    t.sim.scales[:] = .5
    result = selector.consider(t.sim.scales, 24)
    assert not result["accepted"] and result["reason"] == "regresión en partidas completas"
    np.testing.assert_array_equal(t.sim.scales, original)
    assert not set(selector.seeds) & set(selector.game_seeds)


def test_a_mastered_exercise_can_still_consolidate_improvements_in_full_games(laboratory):
    from flybrain_snake.wall_study import WallSelection
    t = laboratory
    games = iter(({"collisions": 0, "apples": 8, "survival_moves": 64, "trials": []},
                  {"collisions": 0, "apples": 9, "survival_moves": 64, "trials": []}))
    selector = WallSelection(t.sim.graph, t.sim.circuit, t.config, t.sim.scales,
                             measure=lambda *_: {"success_rate": 1.0, "collision_rate": 0.0},
                             game_measure=lambda *_: next(games))
    t.sim.scales[:] = .7
    result = selector.consider(t.sim.scales, 24)
    assert result["accepted"] and selector.best["success_rate"] == 1.0
    np.testing.assert_array_equal(selector.scales, np.full_like(selector.scales, .7))


def test_an_extra_collision_on_another_seed_is_not_hidden_by_average():
    from flybrain_snake.wall_study import improves_walls
    before = {"success_rate": .8, "collision_rate": .1,
              "trials": [{"collision": 1}, {"collision": 0}]}
    after = {"success_rate": .9, "collision_rate": .1,
             "trials": [{"collision": 0}, {"collision": 1}]}
    assert not improves_walls(after, before)


def test_unvalidated_weights_stay_private_across_save_load_and_mode_change(laboratory, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from flybrain_snake import desktop_worker
    from flybrain_snake.storage import Journal, load_active_memory
    from flybrain_snake.wall_study import WallSelection
    t = laboratory
    monkeypatch.setattr(desktop_worker, "gate", lambda *_: None)
    monkeypatch.setattr(desktop_worker, "WallSelection", lambda *args, **kwargs:
                        WallSelection(*args, measure=lambda *_: {"success_rate": .8, "collision_rate": .1}))
    def worker():
        w = desktop_worker.Worker(tmp_path)
        w.graph, w.circuit = t.sim.graph, t.sim.circuit
        w.active_scales = np.ones_like(t.sim.scales)
        w.active_config = t.config
        w.saved_scales, w.saved_config = w.active_scales.copy(), w.active_config
        return w
    w = worker()
    cfg = replace(t.config, practice="walls", max_steps=20, wall_memory_hz=180)
    w._command("new", (cfg, 30, True))
    assert w.trainer.config.wall_memory_hz == 0
    assert not w.trainer.config.learning.post_required
    assert w.trainer.sim.learning == w.trainer.config.learning
    assert not np.shares_memory(w.trainer.sim.scales, w.active_scales)
    w._wall_check(initialize=True)
    w.trainer.sim.scales[:] = .3
    selector = w.wall_selection
    w._set_practice("walls", 40)
    assert w.wall_selection is selector
    w._command("save", ())
    saved, _ = load_active_memory(w.memory_path, w.graph, w.circuit)
    np.testing.assert_array_equal(saved, np.ones_like(saved))
    session = w.journal.path
    w._close_session()
    w = worker()
    w._command("load", (session, 30))
    np.testing.assert_array_equal(w.active_scales, np.ones_like(saved))
    np.testing.assert_array_equal(w.trainer.sim.scales, np.full_like(saved, .3))
    w._wall_check(initialize=True)
    selected = restore(w.journal.get("paredes-mejor"), w.graph, w.circuit)
    np.testing.assert_array_equal(selected.sim.scales, w.active_scales)
    w._command("new", (replace(t.config, max_steps=0), 30, True))
    assert w.trainer.sim.scales is w.active_scales
    np.testing.assert_array_equal(w.active_scales, np.ones_like(saved))
    w._close_session()


def test_startup_removes_the_old_wall_cue_with_a_lossless_backup(laboratory, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from flybrain_snake import desktop_worker
    from flybrain_snake.storage import load_active_memory, save_active_memory
    t = laboratory
    monkeypatch.setattr(desktop_worker.Graph, "load", lambda *_: t.sim.graph)
    monkeypatch.setattr(desktop_worker.Circuit, "from_graph", lambda *_: t.sim.circuit)
    w = desktop_worker.Worker(tmp_path)
    config = replace(t.config, practice="walls", max_steps=20, wall_memory_hz=180)
    t.sim.scales[:] = .6
    save_active_memory(w.memory_path, t.sim.graph, t.sim.circuit, t.sim.scales, config)
    original = w.memory_path.read_bytes()
    w.closing = True
    w.run()
    scales, settings = load_active_memory(w.memory_path, t.sim.graph, t.sim.circuit)
    assert settings.wall_memory_hz == 0
    assert not settings.learning.post_required
    np.testing.assert_array_equal(scales, t.sim.scales)
    backups = list(w.memory_path.parent.glob("memoria-antes-retirar-senal-pared-*.npz"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def test_boundary_vision_upgrade_preserves_the_exact_learned_weights(laboratory, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from flybrain_snake import desktop_worker
    from flybrain_snake.storage import load_active_memory, save_active_memory
    t = laboratory
    monkeypatch.setattr(desktop_worker.Graph, "load", lambda *_: t.sim.graph)
    monkeypatch.setattr(desktop_worker.Circuit, "from_graph", lambda *_: t.sim.circuit)
    w = desktop_worker.Worker(tmp_path)
    cfg = replace(t.config, wall_preview_cells=0)
    t.sim.scales[:] = .6
    save_active_memory(w.memory_path, t.sim.graph, t.sim.circuit, t.sim.scales, cfg)
    original = w.memory_path.read_bytes()
    w.closing = True
    w.run()
    scales, settings = load_active_memory(w.memory_path, t.sim.graph, t.sim.circuit)
    assert settings.wall_preview_cells == 1
    np.testing.assert_array_equal(scales, t.sim.scales)
    backups = list(w.memory_path.parent.glob("memoria-antes-vision-limites-*.npz"))
    assert len(backups) == 1 and backups[0].read_bytes() == original


def test_arena_uses_accepted_memory_and_returns_its_learning_to_practice(laboratory, tmp_path):
    pytest.importorskip("PySide6")
    from types import SimpleNamespace
    from flybrain_snake import desktop_worker
    from flybrain_snake.storage import Journal
    t = laboratory
    t.config = replace(t.config, practice="walls", max_steps=20)
    w = desktop_worker.Worker(tmp_path)
    w.graph, w.circuit = t.sim.graph, t.sim.circuit
    w.trainer = t
    w.active_config = t.config
    w.active_scales = np.ones_like(t.sim.scales)
    w.saved_scales, w.saved_config = w.active_scales.copy(), t.config
    w.atlas = SimpleNamespace(active_contacts=lambda _: ())
    w.wall_selection = object()  # Starting Arena must not invoke a slow validation.
    t.sim.scales[:] = .3
    with Journal(tmp_path / "practice.sqlite") as journal:
        w.journal = journal
        w._duel_command("duel_start", (7, t.config, 7, 10, t.config.rewards))
        assert w.duel is not None
        assert w.duel.brian.config.wall_memory_hz == 0
        assert w.duel.brian.config.max_steps == 0
        assert w.duel.brian.sim.scales is w.active_scales
        np.testing.assert_array_equal(w.active_scales, np.ones_like(t.sim.scales))
        assert w.wall_selection is None
        w.duel.brian.sim.scales[0] = .8
        w._duel_command("duel_cancel", (7,))
        np.testing.assert_array_equal(t.sim.scales, w.active_scales)
        assert not np.shares_memory(t.sim.scales, w.active_scales)


@pytest.mark.parametrize("assessor", ("assess", "assess_games"))
def test_validation_can_be_cancelled_without_changing_memory(laboratory, assessor):
    from flybrain_snake import wall_study
    t = laboratory
    config = replace(t.config, practice="walls", max_steps=20)
    before = t.sim.scales.copy()
    with pytest.raises(InterruptedError):
        getattr(wall_study, assessor)(t.sim.graph, t.sim.circuit, config, t.sim.scales, [91], stop=lambda: True)
    np.testing.assert_array_equal(t.sim.scales, before)
