from pathlib import Path

import numpy as np
import pytest

from flybrain_snake.bridge import motor, reinforcement, sensory
from flybrain_snake.circuits import Circuit
from flybrain_snake.config import RunConfig
from flybrain_snake.connectome import Graph
from flybrain_snake.experiments import shuffled_graph, summarize, trainer_for
from flybrain_snake.game import Direction, Reward, Snake
from flybrain_snake.storage import Journal, checkpoint, restore
from flybrain_snake.validation import association, validate_brain


def test_sensory_and_motor_provenance(laboratory):
    t = laboratory
    g = Snake()
    g.apple = (4, 2)
    cues = sensory(g.observe(), t.sim.circuit.groups, 90)
    assert {cue.name for cue in cues} >= {"food_left", "food_front"}
    reward = reinforcement(Reward(apple=10), t.sim.circuit.groups, 120)
    assert len(reward) == 1 and reward[0].name == "appetitive"
    counts = np.array([0, 0, 400, 200, 0, 3, 0])
    assert motor(counts, t.sim.circuit.groups).absolute(Direction.RIGHT) == Direction.UP
    t.begin()
    assert t.step()["source"] == "Brian"
    manual = t.step(Direction.UP)
    assert manual["source"] == "teclado" and manual["manual"] and not manual["plasticity_enabled"]


def test_checkpoint_exact_resume(laboratory, tmp_path):
    t = laboratory
    t.begin()
    t.step()
    t.sim.scales[:] = [.875, .9375]
    t.sim.group_updates[:] = [2, 3]
    filename = tmp_path / "session.sqlite"
    with Journal(filename) as journal:
        journal.append("steps", t.last_step)
        journal.save(t, "fixture")
        journal.append("events", {"unsaved": True})
    with Journal(filename, create=False) as journal:
        resumed = restore(journal.get("fixture"), t.sim.graph, t.sim.circuit)
        assert list(journal.rows("events")) == []
        assert len(list(journal.rows("steps"))) == 1
    np.testing.assert_array_equal(t.sim.scales, resumed.sim.scales)
    assert t.game.state() == resumed.game.state()
    assert t.sim.rng.bit_generator.state == resumed.sim.rng.bit_generator.state
    result = t.step()
    actual = resumed.step()
    for k in ("source", "direction", "decision", "reward", "groups", "head", "apple", "memory"):
        assert actual[k] == result[k]
    np.testing.assert_array_equal(t.sim.voltage, resumed.sim.voltage)


@pytest.fixture(scope="module")
def real_graph():
    path = Path("data/processed")
    if not path.is_dir():
        pytest.skip("Ejecuta prepare para probar el conectoma oficial completo")
    return Graph.load(path, verify=True)


@pytest.mark.flywire
def test_full_v783(real_graph):
    graph = real_graph
    assert graph.n == 139255
    assert len(graph.targets) == 16847997
    assert graph.manifest["unique_pairs"] == 15091983
    assert graph.manifest["contacts"] == 54492922
    assert graph.targets.max() < graph.n
    assert np.array_equal(graph.ids, np.unique(graph.ids))
    c = Circuit.from_graph(graph)
    assert len(c.edges) == 4157
    assert len(np.unique(c.edges)) == len(c.edges)
    t = trainer_for(graph, c, RunConfig())
    assert validate_brain(t.sim, 1, duration_ms=250)["passed"]
    assert association(t.sim, t.config, trials=3)["passed"]
    t.begin()
    t.step()
    t2 = restore(checkpoint(t), graph, c)
    first, resumed = t.step(), t2.step()
    assert first["decision"] == resumed["decision"]
    assert first["groups"] == resumed["groups"]
    np.testing.assert_array_equal(t.sim.scales, t2.sim.scales)
    assert first["rss_mib"] > 0 and first["wall_ms"] > 0


def test_journal_readers_and_single_writer(laboratory, tmp_path):
    filename = tmp_path / "single.sqlite"
    with Journal(filename) as first:
        first.save(laboratory)
        with Journal(filename, read_only=True) as reader:
            assert reader.get() == first.get()
        with pytest.raises(ValueError, match="escritor"):
            Journal(filename)
    with Journal(filename, create=False) as next_writer:
        assert next_writer.labels() == ["latest"]


def test_graph_control_preserves_degree_and_plastic_rows(laboratory):
    g, c = laboratory.sim.graph, laboratory.sim.circuit
    shuffled = shuffled_graph(g, c, 1)
    assert np.array_equal(g.offsets, shuffled.offsets)
    for i in range(len(g.region_names)):
        group = g.regions == i
        assert np.array_equal(np.sort(g.targets[group]), np.sort(shuffled.targets[group]))
    assert np.array_equal(g.targets[c.edges], shuffled.targets[c.edges])
    assert shuffled.fingerprint != g.fingerprint


def test_holdout_is_disjoint_and_reproducible():
    for i in range(50):
        train = Snake(seed=i, food_partition="train")
        holdout = Snake(seed=i, food_partition="heldout")
        assert sum(train.apple) % 2 == 0
        assert sum(holdout.apple) % 2 == 1
        assert Snake.restore(train.state(), train.rewards).apple == train.apple


def test_edge_aware_reinforcement_keeps_food_and_collision_strong(laboratory):
    groups = laboratory.sim.circuit.groups
    approach = reinforcement(Reward(wall=-.4), groups, 120)[0].rate
    edge = reinforcement(Reward(wall=-1.5), groups, 120)[0].rate
    collision = reinforcement(Reward(collision=-60), groups, 120)[0].rate
    apple_at_edge, = reinforcement(Reward(apple=30, wall=-1.5), groups, 120)
    conflicting_step, = reinforcement(Reward(distance=.25, wall=-.39), groups, 120)
    assert conflicting_step.name == "aversive"
    assert apple_at_edge.name == "appetitive"
    assert 0 < approach < edge < 12
    assert min(collision, apple_at_edge.rate) > 110
    complete, = reinforcement(Reward(apple=30, completion=150), groups, 120)
    assert complete.name == "appetitive" and complete.rate > 2.5 * apple_at_edge.rate


def test_compact_trace_preserves_decision_and_neural_update(laboratory):
    t = laboratory
    t.begin()
    clone = restore(checkpoint(t), t.sim.graph, t.sim.circuit)
    full = t.step()
    compact = clone.step(detail=False)
    assert t.game.state() == clone.game.state()
    for key in ("decision", "reward", "delivered", "memory"):
        assert compact[key] == full[key]
    assert "input" in full and "input" not in compact
    np.testing.assert_array_equal(t.sim.scales, clone.sim.scales)
    np.testing.assert_array_equal(t.sim.voltage, clone.sim.voltage)


def test_frontal_wall_alert_activates_before_collision(laboratory):
    groups = laboratory.sim.circuit.groups
    game = Snake()
    game.body[0] = (6, 0)
    cues = {cue.name: cue.rate for cue in sensory(game.observe(), groups, 90)}
    assert cues["loom_left"] > cues["loom_right"] > 0
    game.body[0] = (6, 3)
    balanced = {cue.name: cue.rate for cue in sensory(game.observe(), groups, 90)}
    assert balanced["loom_left"] == pytest.approx(1.5 * balanced["loom_right"])
    assert "loom_left" not in {cue.name for cue in sensory(Snake().observe(), groups, 90)}
    assert "loom_left" not in {cue.name for cue in sensory(game.observe(), groups, 90, profile="graded")}


@pytest.mark.parametrize("head,heading,apple,side", [
    ((5, 0), Direction.UP, (6, 0), "right"),
    ((1, 0), Direction.UP, (0, 0), "left"),
])
def test_food_in_open_flank_is_not_encoded_as_a_wall(laboratory, head, heading, apple, side):
    game = Snake()
    game.body[0], game.direction, game.apple = head, heading, apple
    groups = laboratory.sim.circuit.groups

    def rates(profile):
        return {cue.name: cue.rate for cue in sensory(game.observe(), groups, 90, profile=profile)}

    before, after = rates("edge_aware"), rates("edge_target")
    other = "left" if side == "right" else "right"
    assert after[f"food_{side}"] == before[f"food_{side}"]
    assert after[f"loom_{side}"] < before[f"loom_{side}"]
    assert after[f"wall_{side}"] < before[f"wall_{side}"]
    assert after[f"loom_{other}"] == before[f"loom_{other}"]
    assert game.step(heading.turn(1 if side == "right" else -1)).apple == 30


def test_manual_pulse_does_not_replace_the_collision(laboratory):
    t = laboratory
    t.begin()
    t.game.body = [(6, 3), (5, 3)]
    t.game.apple = (0, 0)
    trace = t.step(Direction.RIGHT, manual_reward=Reward(apple=10))
    assert trace["reward"]["collision"] == trace["delivered"]["collision"] == -60
    assert trace["delivered"]["apple"] == 10
    assert trace["reinforcement_hz"]["aversive"] > 110
    assert trace["reinforcement_hz"]["appetitive"] > 0


def test_terminal_reward_precedes_the_next_episode(laboratory):
    t = laboratory
    t.begin()
    game = t.game
    game.apples = game.size**2 - 1
    game.apple = (4, 3)
    trace = t.step()
    assert trace["end"] == "tablero completo"
    assert trace["delivered"]["completion"] == 150
    assert trace["reinforcement_hz"]["appetitive"] > 300
    assert t.finish()["apples"] == game.size**2
    t.begin()
    assert t.game.apples == 0 and t.game.end == ""


@pytest.mark.parametrize("profile", ("edge_target", "edge_aware", "graded", "legacy"))
def test_wall_approach_strengthens_aversive_signals(laboratory, profile):
    from flybrain_snake.config import Rewards

    game = Snake(11, rewards=Rewards(profile=profile, distance=0, route=0))
    game.apple = (1, 5)
    groups = laboratory.sim.circuit.groups
    assert not reinforcement(game.step(Direction.RIGHT), groups, 120, profile=profile)
    rates = []
    for _ in range(5):
        pulse, = reinforcement(game.step(Direction.RIGHT), groups, 120, profile=profile)
        assert pulse.name == "aversive"
        rates.append(pulse.rate)
    assert 0 < rates[0] and all(first < second for first, second in zip(rates, rates[1:]))


def test_avoidable_route_along_edge_cannot_cancel_wall_penalty(laboratory):
    game = Snake(11)
    game.body[0] = (1, 0)
    game.apple = (8, 4)
    for _ in range(3):
        reward = game.step(Direction.RIGHT)
    assert reward.total == reward.wall < 0
    signal, = reinforcement(reward, laboratory.sim.circuit.groups, 120)
    assert signal.name == "aversive"


def test_needed_edge_route_only_sends_food_reinforcement(laboratory):
    game = Snake(11)
    game.body = [(1, 1), (0, 1)]
    game.apple = (8, 1)
    for _ in range(7):
        reward = game.step(Direction.RIGHT)
        signal, = reinforcement(reward, laboratory.sim.circuit.groups, 120)
        assert reward.wall == 0 and signal.name == "appetitive"
    assert reward.apple == 30


def test_second_route_deviation_is_delivered_even_on_the_perimeter(laboratory):
    game = Snake(11)
    game.body = [(1, 1), (0, 1)]
    game.apple = (7, 5)
    first = game.step(Direction.UP)
    second = game.step(Direction.LEFT)
    assert first.wall == second.wall == -12
    assert first.route == 0 and second.route == -2
    groups = laboratory.sim.circuit.groups
    edge, = reinforcement(first, groups, 120)
    combined, = reinforcement(second, groups, 120)
    assert edge.name == combined.name == "aversive"
    assert combined.rate > edge.rate


def test_detours_only_send_one_aversive_signal_for_the_same_apple(laboratory):
    game = Snake(15)
    game.apple = (10, 7)
    groups = laboratory.sim.circuit.groups
    directions = (Direction.RIGHT, Direction.UP, Direction.LEFT, Direction.DOWN, Direction.RIGHT)
    signals = [reinforcement(game.step(direction), groups, 120) for direction in directions]
    assert [tuple(signal.name for signal in step) for step in signals] == [
        ("appetitive",), (), ("aversive",), (), (),
    ]



def test_side_wall_does_not_loom_when_moving_parallel(laboratory):
    game = Snake(11)
    game.body[0] = (2, 0)
    game.apple = (8, 0)
    groups = laboratory.sim.circuit.groups
    cues = {cue.name: cue.rate for cue in sensory(game.observe(), groups, 90)}
    assert cues["wall_left"] == pytest.approx(7.2)
    assert cues["food_front"] == pytest.approx(27)
    assert cues["tag_forward"] == pytest.approx(43.2)
    assert cues["tag_wall_left"] == pytest.approx(72)
    game.body[0] = (10, 0)
    cues = {cue.name: cue.rate for cue in sensory(game.observe(), groups, 90)}
    assert "loom_left" in cues and "loom_right" in cues


def test_route_practice_uses_the_brain_and_closes_on_the_first_apple(laboratory):
    from dataclasses import replace

    t = laboratory
    t.config = replace(t.config, practice="routes", max_steps=20)
    t.begin(32)
    optimal = t.game.observe().shortest_food_path()
    assert t.game.body[0][1] == t.game.apple[1] == 1
    while not t.game.end:
        trace = t.step()
        assert trace["source"] == "Brian" and not trace["manual"]
    row = t.finish()
    assert row["apples"] == 1 and row["steps"] == optimal
    assert row["route_efficiency"] == 1
    assert row["end"] == "trayecto completado"


def test_route_practice_can_continue_after_the_deviation_penalty(laboratory):
    from dataclasses import replace

    t = laboratory
    t.config = replace(t.config, practice="routes", max_steps=20)
    t.begin(32)
    t.game.body = [(3, 3), (2, 3)]
    t.game.direction = Direction.RIGHT
    t.game.apple = (6, 3)
    assert t.step(Direction.UP)["end"] == ""
    failed = t.step(Direction.LEFT)
    assert failed["end"] == "" and failed["delivered"]["route"] == -2
    assert failed["reinforcement_hz"]["aversive"] > 0
    assert t.step(Direction.DOWN)["step"] == 3
    assert t.game.end == ""
