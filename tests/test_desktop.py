from dataclasses import replace
from pathlib import Path
import os
import time

import pytest


@pytest.mark.desktop
def test_desktop_background_worker(tmp_path, monkeypatch):
    if not Path("data/anatomy/manifest.json").is_file():
        pytest.skip("Se requiere el atlas oficial")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from flybrain_snake.config import RunConfig
    from flybrain_snake.connectome import Graph
    from flybrain_snake import desktop_worker
    from flybrain_snake.gui import Window

    app = QApplication.instance() or QApplication([])
    real = Graph.load(Path("data/processed"))
    monkeypatch.setattr(desktop_worker.Graph, "load", lambda *_: real)
    round_seeds = iter((1, 42, 43))
    monkeypatch.setattr(desktop_worker, "randbits", lambda _: next(round_seeds))
    receipt = Path("data/runs/validation.json")
    if receipt.is_file():
        (tmp_path / "runs").mkdir()
        (tmp_path / "runs" / "validation.json").write_bytes(receipt.read_bytes())
    win = Window(tmp_path, RunConfig(), smoke=True)
    win.show()
    pulses = []
    timer = QTimer()
    timer.timeout.connect(lambda: pulses.append(time.monotonic()))
    timer.start(20)
    try:
        deadline = time.monotonic() + 20
        while not win.arena.play.isEnabled() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert win.start.isEnabled()
        assert len(pulses) >= 2
        assert win.arena.grid_size.value() == 11
        assert win.arena.apple_goal.value() == 50
        assert len(win.arena.brain.atlas_positions) == 6000
        win.arena.grid_size.setValue(4)
        assert win.arena.apple_goal.maximum() == 16
        win.arena.grid_size.setValue(100)
        assert win.arena.human.board.state["size"] == 100
        win.arena.grid_size.setValue(13)
        assert win.arena.brian.board.state["size"] == win.arena.human.board.state["size"] == 13
        win.arena.apple_goal.setValue(42)
        win.activateWindow()
        win.arena.play.click()
        deadline = time.monotonic() + 20
        while not win.arena.live and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert win.arena.live and win.arena.awaiting_first
        assert win.worker.duel.apple_goal == 42
        assert win.worker.duel.brian.game.max_steps == 0
        assert win.worker.duel.brian.config.synaptic_signal == "contact_probabilities"
        assert win.worker.duel.brian.sim.scales is win.worker.active_scales
        assert win.worker.duel.brian.config.plasticity and win.worker.duel.brian.config.dopamine
        assert len(win.worker.duel.brian.sim.edge_gain) == len(real.targets)
        assert not win.arena.timer.isActive()
        ax, ay = win.arena.brian.board.state["apple"]
        assert 0 < ax < 12 and 0 < ay < 12
        assert not win.worker.duel.brian.game.allow_edge_food
        assert win.worker.duel.human.allow_edge_food
        win.arena.setFocus()
        app.processEvents()
        moves = []
        win.worker.duel_player.connect(lambda _, board: moves.append(board["body"][0]))
        QTest.keyClick(win.arena, Qt.Key_Down)
        assert win.arena.waiting and not win.arena.awaiting_first
        assert win.arena.human.shown_key.name == "DOWN"
        QTest.keyClick(win.arena, Qt.Key_Right)
        QTest.keyClick(win.arena, Qt.Key_Up)
        assert [direction.name for direction in win.arena.queued] == ["RIGHT", "UP"]
        deadline = time.monotonic() + 20
        while win.arena.brian.board.state["steps"] < 3 and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert moves[:3] == [(6, 7), (7, 7), (7, 6)]
        assert win.arena.brian.board.state["steps"] == win.arena.human.board.state["steps"] == 3
        assert any(len(ids) for ids in win.arena.brain.pulses)
        assert win.arena.brain.pulse_timer.isActive()
        assert not win.arena.brain.grab().isNull()
        win.tabs.setCurrentWidget(win.brain_tab)
        assert win.arena.paused
        app.processEvents()
        assert not win.brain.grab().isNull()
        assert len(win.brain.indices) == 3200
        assert len(win.brain.atlas_positions) == 6000
        win._select("720575940630909714")
        deadline = time.monotonic() + 10
        while not win.brain.selected_contacts and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert any(len(points) for points in win.brain.selected_contacts)
        win.worker.duel.brian.sim.scales[0] = .5
        win.moves.setValue(2)
        win.size.setValue(10)
        win.synaptic_signal.setCurrentIndex(win.synaptic_signal.findData("curated"))
        win.reward_fields["wall"].setValue(4)
        win.n.setValue(1)
        win._new()
        before = len(pulses)
        deadline = time.monotonic() + 60
        while not win.rows and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert win.rows and win.rows[0]["steps"] == 2
        assert win.worker.trainer.sim.scales is win.worker.active_scales
        assert win.worker.active_scales[0] <= .5
        assert win.last["step"]["source"] == "Brian"
        assert win.board_stack.currentWidget() is win.board
        win.tabs.setCurrentWidget(win.telemetry_tab)
        assert win.telemetry_stack.currentIndex() == 1
        assert win.telemetry.topLevelItemCount() >= 2
        assert len(pulses) > before + 2
        for i in range(win.chart_metric.count()):
            win.chart_metric.setCurrentIndex(i)
            app.processEvents()
            assert not win.plot.grab().isNull()
        profile = win.worker.trainer.config
        assert win.arena.grid_size.value() == profile.size == 10
        win.reward_fields["wall"].setValue(7)
        win.tabs.setCurrentWidget(win.arena)
        win.arena.play.click()
        deadline = time.monotonic() + 20
        while not win.arena.awaiting_first and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert win.arena.awaiting_first
        assert win.worker.duel.brian.game.seed == win.worker.duel.human.seed == 42
        assert win.worker.duel.brian.config == replace(profile, max_steps=0)
        assert win.worker.duel.brian.config.rewards.wall == 4
        assert win.worker.duel.brian.sim.scales is win.worker.trainer.sim.scales
        win.arena.grid_size.setValue(12)
        win.arena.play.click()
        deadline = time.monotonic() + 20
        while not win.arena.awaiting_first and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        assert win.arena.awaiting_first
        assert win.worker.duel.brian.game.seed == win.worker.duel.human.seed == 43
        assert win.worker.duel.brian.config.size == 12
        assert win.worker.active_config == profile
        win.reset_learning.click()
        deadline = time.monotonic() + 10
        while win.last is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.01)
        from flybrain_snake.storage import load_active_memory
        assert not win.rows and win.last is None
        scales, settings = load_active_memory(win.worker.memory_path, real, win.worker.circuit)
        assert (scales == 1).all() and settings is None
        assert (win.worker.active_scales == 1).all()
    finally:
        timer.stop()
        win.worker.send("exit")
        assert win.worker.wait(30000)
        win.close()


@pytest.mark.desktop
def test_strict_perimeter_mode_in_the_desktop(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from flybrain_snake.config import RunConfig
    from flybrain_snake.connectome import Graph
    from flybrain_snake import desktop_worker
    from flybrain_snake.gui import Window

    if not Path("data/processed").is_dir():
        pytest.skip("Se requiere el conectoma oficial")
    app = QApplication.instance() or QApplication([])
    graph = Graph.load(Path("data/processed"))
    monkeypatch.setattr(desktop_worker.Graph, "load", lambda *_: graph)
    monkeypatch.setattr(desktop_worker, "gate", lambda *_: None)
    # Silence inputs to make the UI-only trajectory deterministic and straight.
    cfg = RunConfig(practice="edge_reset", max_steps=0, sensory_hz=0, plasticity=False, dopamine=False)
    win = Window(tmp_path, cfg, smoke=True)
    win.show()
    def until(condition):
        deadline = time.monotonic() + 20
        while not condition() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(.005)
        assert condition(), win.status.text()
    try:
        until(lambda: win.start.isEnabled())
        assert win.practice.currentData() == "edge_reset" and win.moves.value() == 0
        win.n.setValue(2)
        win._new()
        until(lambda: len(win.rows) == 2 and not win.worker.running)
        assert all(row["end"] == "borde alcanzado" and row["steps"] == 3 for row in win.rows)
        assert win.chart_metric.currentData() == "edge_termination"
        assert win.worker.trainer.config.max_steps == 0
        win.practice.setCurrentIndex(win.practice.findData("snake"))
        until(lambda: win.worker.trainer.config.practice == "snake")
        assert win.moves.value() == 0
    finally:
        win.worker.send("exit")
        assert win.worker.wait(30000)
        win.close()


@pytest.mark.desktop
def test_switch_practice_and_resume_updates_the_running_session(tmp_path, monkeypatch):
    if not Path("data/anatomy/manifest.json").is_file():
        pytest.skip("Se requiere el atlas oficial")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication
    from flybrain_snake.config import RunConfig
    from flybrain_snake.connectome import Graph
    from flybrain_snake.experiments import trainer_for
    from flybrain_snake.storage import Journal, save_active_memory
    from flybrain_snake.circuits import Circuit
    import numpy as np
    from flybrain_snake import desktop_worker
    from flybrain_snake.gui import Window

    app = QApplication.instance() or QApplication([])
    graph = Graph.load(Path("data/processed"))
    monkeypatch.setattr(desktop_worker.Graph, "load", lambda *_: graph)
    monkeypatch.setattr(desktop_worker, "gate", lambda *_: None)
    circuit = Circuit.from_graph(graph)
    initial_memory = np.ones(len(circuit.edges), dtype=np.float32)
    initial_memory[0] = .75
    save_active_memory(tmp_path / "sessions" / "memoria-activa.npz", graph, circuit, initial_memory,
                       RunConfig(size=10, max_steps=2))
    win = Window(tmp_path, RunConfig(size=10, max_steps=0), smoke=True)
    failures = []
    win.worker.message.connect(failures.append)
    win.show()

    def until(condition):
        deadline = time.monotonic() + 20
        while not condition() and time.monotonic() < deadline:
            app.processEvents()
            QTest.qWait(5)
        assert condition(), failures or win.status.text()

    try:
        until(lambda: win.arena.play.isEnabled())
        assert win.moves.value() == win.worker.active_config.max_steps == 0
        np.testing.assert_array_equal(win.worker.active_scales, initial_memory)
        win.tabs.setCurrentWidget(win.training_tab)
        win.n.setValue(10000)
        win.practice.setCurrentIndex(win.practice.findData("routes"))
        assert win.moves.value() == 40
        win._resume()
        until(lambda: win.worker.trainer is not None and bool(win.worker.trainer.last_step))
        win.worker.send("pause")
        until(lambda: not win.worker.running)
        trainer = win.worker.trainer
        memory = trainer.sim.scales
        memory[0] = .75
        episode = trainer.episode

        win.practice.setCurrentIndex(win.practice.findData("snake"))
        assert win.moves.value() == 0
        until(lambda: trainer.config.practice == "snake")
        assert trainer.game is None and trainer.config.max_steps == 0
        assert trainer.sim.scales is memory and memory[0] == .75
        assert trainer.episode == episode
        win._resume()
        until(lambda: trainer.game is not None and trainer.game.steps >= 3)
        assert trainer.config.practice == "snake"
        win.worker.send("pause")
        until(lambda: not win.worker.running)

        current_game = trainer.game
        win.plastic.setChecked(False)
        win.dopamine.setChecked(False)
        until(lambda: not trainer.config.plasticity and not trainer.config.dopamine)
        assert trainer.game is current_game
        assert not win.worker.active_config.plasticity
        win.plastic.setChecked(True)
        win.dopamine.setChecked(True)
        until(lambda: trainer.config.plasticity and trainer.config.dopamine)
        win.moves.setValue(2)
        until(lambda: trainer.config.max_steps == 2)
        assert trainer.game is current_game and current_game.max_steps == 2
        from PySide6.QtCore import QSignalBlocker
        with QSignalBlocker(win.moves):
            win.moves.setValue(0)
        win._resume()
        until(lambda: current_game.steps >= 4)
        assert trainer.game is current_game and current_game.max_steps == 0
        assert trainer.config.max_steps == 0
        win.worker.send("pause")
        until(lambda: not win.worker.running)

        full_source = tmp_path / "source_full_two_steps.sqlite"
        short = trainer_for(graph, win.worker.circuit, replace(trainer.config, max_steps=2))
        short.sim.scales[:] = memory
        short.begin(53)
        short.step(detail=False)
        with Journal(full_source) as journal:
            journal.save(short)
        win.worker.send("load", full_source, 20)
        until(lambda: win.worker.journal is not None and win.worker.journal.path == full_source)
        until(lambda: win.moves.value() == 0)
        loaded = win.worker.trainer
        assert loaded.config.max_steps == loaded.game.max_steps == 0
        np.testing.assert_array_equal(loaded.sim.scales, short.sim.scales)
        win._resume()
        until(lambda: loaded.game is not None and loaded.game.steps >= 4)
        win.worker.send("pause")
        until(lambda: not win.worker.running)

        source = tmp_path / "source_routes.sqlite"
        cfg = replace(trainer.config, practice="routes", max_steps=25)
        fresh = trainer_for(graph, win.worker.circuit, cfg)
        fresh.sim.scales[:] = memory
        with Journal(source) as journal:
            journal.save(fresh)
        win.worker.send("load", source, 20)
        until(lambda: win.practice.currentData() == "routes")
        assert win.moves.value() == 25
        win.practice.setCurrentIndex(win.practice.findData("snake"))
        until(lambda: win.worker.trainer.config.practice == "snake")
        assert win.moves.value() == 0
        win._resume()
        until(lambda: win.worker.trainer.game is not None and win.worker.trainer.game.steps >= 3)
        assert win.worker.trainer.config.practice == "snake"
        assert not failures
    finally:
        win.worker.send("exit")
        assert win.worker.wait(30000)
        win.close()
