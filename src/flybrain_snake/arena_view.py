"""Qt presentation and keyboard input for the two-board arena."""
from __future__ import annotations

from collections import deque
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QAbstractSpinBox, QFrame, QHBoxLayout, QLabel, QPushButton, QSpinBox, QVBoxLayout, QWidget

from .arena import DEFAULT_APPLE_GOAL, apple_capacity
from .game import Direction, Snake
from .widgets import BRIAN, HUMAN, Board, BrainView

PACE_MS = 135
DEFAULT_SIZE = 11
MIN_SIZE, MAX_SIZE = 4, 100
KEYS = (("Up", "↑", Direction.UP), ("Left", "←", Direction.LEFT),
        ("Down", "↓", Direction.DOWN), ("Right", "→", Direction.RIGHT))


def caption(text: str, style: str):
    label = QLabel(text)
    label.setObjectName(style)
    return label


class PlayerCard(QFrame):
    def __init__(self, name, color, size):
        super().__init__()
        self.setObjectName("playerCard")
        self.setProperty("player", "brian" if color == BRIAN else "human")
        box = QVBoxLayout(self)
        box.setContentsMargins(19, 17, 19, 17)
        header = QHBoxLayout()
        header.addWidget(caption(name, "playerTitle"), 1)
        self.score = caption("0", "score")
        self.score.setToolTip("Manzanas")
        header.addWidget(self.score)
        box.addLayout(header)
        self.board = Board(size, color, dense_grid=True)
        box.addWidget(self.board, 1)
        controls = QWidget()
        controls.setObjectName("keys")
        keys = QHBoxLayout(controls)
        keys.setContentsMargins(0, 0, 0, 0)
        keys.setSpacing(4)
        keys.addStretch()
        self.keycaps = {}
        self.shown_key = None
        if color == HUMAN:
            title = caption("Flechas", "keysLabel")
            keys.addWidget(title)
            keys.addSpacing(4)
            for _, arrow, direction in KEYS:
                key = caption(arrow, "keycap")
                key.setAlignment(Qt.AlignCenter)
                key.setFixedSize(23, 23)
                keys.addWidget(key)
                self.keycaps[direction] = key
        controls.setFixedHeight(23)
        box.addWidget(controls)

    def show_key(self, direction):
        if direction == self.shown_key:
            return
        for key, name in ((self.shown_key, "keycap"), (direction, "keycapActive")):
            if key in self.keycaps:
                cap = self.keycaps[key]
                cap.setObjectName(name)
                cap.style().unpolish(cap)
                cap.style().polish(cap)
        self.shown_key = direction

    def refresh(self, board, apple_goal):
        self.board.state = board
        self.board.update()
        self.score.setText(f"{board['apples']} / {apple_goal}")


class Arena(QWidget):
    def __init__(self, worker, cfg, load_session, rewards):
        super().__init__()
        self.worker, self.cfg, self.rewards = worker, cfg, rewards
        self.training_profile = None
        self.round_id = 0
        self.ready = False
        self.waiting = self.live = self.paused = False
        self.awaiting_first = False
        self.turn_started = 0.0
        self.heading = Direction.RIGHT
        self.queued = deque(maxlen=2)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(14)
        field_panel = QWidget()
        field_panel.setMaximumHeight(560)
        field = QHBoxLayout(field_panel)
        field.setContentsMargins(0, 0, 0, 0)
        field.setSpacing(10)
        size = DEFAULT_SIZE if cfg.size == 7 else cfg.size
        self.brian = PlayerCard("Brian", BRIAN, size)
        self.human = PlayerCard("Tú", HUMAN, size)
        field.addWidget(self.brian, 5)
        scope = QFrame()
        scope.setObjectName("arenaScope")
        scope_layout = QVBoxLayout(scope)
        scope_layout.setContentsMargins(0, 12, 0, 12)
        scope_layout.setSpacing(0)
        scope_title = self.scope_title = caption("Cerebro", "scopeTitle")
        scope_title.setAlignment(Qt.AlignCenter)
        scope_layout.addWidget(scope_title)
        self.brain = BrainView(compact=True)
        scope_layout.addWidget(self.brain, 1)
        self.status = caption("Cargando…", "arenaStatus")
        self.status.setAlignment(Qt.AlignCenter)
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(28)
        scope_layout.addWidget(self.status)
        field.addWidget(scope, 4)
        field.addWidget(self.human, 5)
        layout.addStretch(1)
        layout.addWidget(field_panel, 10)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        load = QPushButton("Sesión…")
        load.setToolTip("Cargar una sesión de Brian")
        load.clicked.connect(load_session)
        controls.addWidget(load)
        controls.addStretch()
        self.pause = QPushButton("Pausa")
        self.pause.setEnabled(False)
        self.pause.clicked.connect(self.toggle_pause)
        controls.addWidget(self.pause)
        self.play = QPushButton("Jugar")
        self.play.setObjectName("accent")
        self.play.setEnabled(False)
        self.play.clicked.connect(self.start_round)
        controls.addWidget(self.play)
        controls.addStretch()
        controls.addWidget(caption("Tablero", "detail"))
        self.grid_size = QSpinBox()
        self.grid_size.setRange(MIN_SIZE, MAX_SIZE)
        self.grid_size.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.grid_size.setAlignment(Qt.AlignCenter)
        self.grid_size.setMinimumWidth(72)
        self.grid_size.setValue(size)
        self.grid_size.setSuffix(" × " + str(size))
        self.grid_size.setAccessibleName("Casillas por lado")
        self.grid_size.setToolTip("Tamaño de entrenamiento por defecto. Puedes ajustarlo para el próximo duelo.")
        self.grid_size.setKeyboardTracking(False)
        self.grid_size.valueChanged.connect(self._size_changed)
        minus = QPushButton("−")
        plus = QPushButton("+")
        for control, name, callback in ((minus, "Reducir tablero", self.grid_size.stepDown),
                                        (plus, "Ampliar tablero", self.grid_size.stepUp)):
            control.setObjectName("sizeStep")
            control.setFixedWidth(29)
            control.setAccessibleName(name)
            control.clicked.connect(callback)
        controls.addWidget(minus)
        controls.addWidget(self.grid_size)
        controls.addWidget(plus)
        controls.addSpacing(8)
        controls.addWidget(caption("Meta", "detail"))
        self.apple_goal = QSpinBox()
        self.apple_goal.setRange(1, apple_capacity(size))
        self.apple_goal.setValue(DEFAULT_APPLE_GOAL)
        self.apple_goal.setAlignment(Qt.AlignCenter)
        self.apple_goal.setKeyboardTracking(False)
        self.apple_goal.setAccessibleName("Meta de manzanas")
        self.apple_goal.setToolTip("Manzanas para ganar en el próximo duelo. El máximo depende del tablero.")
        self.apple_goal.valueChanged.connect(self._preview)
        controls.addWidget(self.apple_goal)
        layout.addLayout(controls)
        layout.addStretch(1)
        self.brian.setToolTip("Brian juega y aprende con su memoria activa.")
        self._size_changed(size)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self._tick)
        self.rematch_timer = QTimer(self)
        self.rematch_timer.setSingleShot(True)
        self.rematch_timer.timeout.connect(self._restart_completed_board)
        self.shortcuts = []
        for key, _, direction in KEYS:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.setAutoRepeat(False)
            shortcut.setEnabled(False)
            shortcut.activated.connect(lambda d=direction: self.steer(d))
            self.shortcuts.append(shortcut)
        worker.duel_frame.connect(self.receive)
        worker.duel_player.connect(self.receive_player)
        worker.duel_failed.connect(self.failed)

    def _size_changed(self, size):
        self.grid_size.setSuffix(f" × {size}")
        self.apple_goal.setMaximum(apple_capacity(size))
        self._preview()

    def _preview(self):
        if not self.live and not self.waiting:
            size = self.grid_size.value()
            self.brian.refresh(Snake(size, self.cfg.seed, allow_edge_food=False).board_state(),
                               self.apple_goal.value())
            self.human.refresh(Snake(size, self.cfg.seed).board_state(), self.apple_goal.value())

    def connect_brain(self, graph, circuit):
        self.brain.load(graph, circuit)
        self.status.setText("Cargando atlas…")

    def connect_atlas(self, regions, view):
        if regions is None:
            self.status.setText("Atlas pendiente · anatomy-prepare")
            self.status.setToolTip(view)
            return
        self.brain.set_atlas(regions, view)
        self.ready = True
        self.play.setEnabled(True)
        self.scope_title.setText("Cerebro · FlyWire v783")
        self.scope_title.setToolTip("Contactos de FlyWire y pulsos de las neuronas que disparan en cada turno.")
        self.status.clear()

    def set_training_profile(self, profile):
        self.training_profile = profile
        if profile is not None:
            self.grid_size.setValue(profile.size)

    def start_round(self):
        try:
            rewards = self.rewards() if self.training_profile is None else self.training_profile.rewards
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.cancel()
        self.round_id += 1
        self.waiting = True
        self.heading = Direction.RIGHT
        self.queued.clear()
        self.human_finished = False
        self.play.setEnabled(False)
        self.status.setText("Preparando…")
        self.setFocus()
        size = self.grid_size.value()
        self.round_goal = self.apple_goal.value()
        self.worker.send("duel_start", self.round_id, self.cfg, size, self.round_goal, rewards)

    def reset_preview(self):
        self.play.setText("Jugar")
        if self.ready:
            self.status.clear()
        self._preview()

    def cancel(self):
        self.rematch_timer.stop()
        if self.live or self.waiting:
            self.worker.send("duel_cancel", self.round_id)
            self.round_id += 1
            self.status.clear()
        self._stop()
        self.brain.flash(())
        self.play.setEnabled(self.ready)

    def _stop(self):
        self.timer.stop()
        self.live = self.paused = self.waiting = False
        self.awaiting_first = False
        self.queued.clear()
        self.human.show_key(None)
        self.pause.setEnabled(False)
        self.pause.setText("Pausa")
        for shortcut in self.shortcuts:
            shortcut.setEnabled(False)

    def steer(self, direction: Direction):
        if not self.live or self.paused or self.human_finished:
            return
        heading = self.queued[-1] if self.queued else self.heading
        if direction == heading.turn(2):
            return
        if direction != heading:
            if len(self.queued) == self.queued.maxlen:
                return
            self.queued.append(direction)
        self.human.show_key(direction)
        if self.awaiting_first:
            self.awaiting_first = False
            self.status.clear()
            self._tick()

    def _tick(self):
        if not self.live or self.paused or self.waiting or self.awaiting_first:
            return
        if self.queued:
            self.heading = self.queued.popleft()
        self.waiting = True
        self.turn_started = time.monotonic()
        self.worker.send("duel_tick", self.round_id, self.heading)

    def _schedule_turn(self):
        if not self.live or self.paused or self.waiting or self.awaiting_first:
            return
        elapsed_ms = (time.monotonic() - self.turn_started) * 1000
        self.timer.start(max(1, round(PACE_MS - elapsed_ms)))

    def toggle_pause(self):
        if not self.live:
            return
        self.paused = not self.paused
        self.pause.setText("Continuar" if self.paused else "Pausa")
        if self.paused:
            self.timer.stop()
            self.queued.clear()
            self.human.show_key(None)
            self.status.setText("Pausa")
        else:
            self.status.setText("↑  ↓  → para empezar" if self.awaiting_first else "")
            self.turn_started = time.monotonic()
            self._schedule_turn()

    def receive_player(self, round_id, board):
        if round_id != self.round_id:
            return
        self.human.refresh(board, self.round_goal)
        self.human_finished = bool(board["end"])
        if self.human_finished:
            self.queued.clear()
            self.human.show_key(None)

    def receive(self, round_id, snapshot):
        if round_id != self.round_id:
            return
        self.waiting = False
        self.brian.refresh(snapshot["brian"], self.round_goal)
        self.brain.flash(snapshot["activity"], snapshot["contact_activity"])
        if snapshot["turn"] == 0:
            self.play.setToolTip(f"Semilla de esta partida: {snapshot['seed']}")
            self.receive_player(round_id, snapshot["human"])
            self.live = self.awaiting_first = True
            self.play.setText("Reiniciar")
            self.play.setEnabled(True)
            self.pause.setEnabled(True)
            self.status.setText("Pulsa una flecha")
            for shortcut in self.shortcuts:
                shortcut.setEnabled(True)
            if not self.isVisible():
                self.toggle_pause()
        elif snapshot["winner"] is not None:
            self.worker.send("duel_cancel", self.round_id)
            self._stop()
            winner = snapshot["winner"]
            title = {"Brian": "Gana Brian", "Tú": "Has ganado", "Empate": "Empate"}[winner]
            reason = "Meta alcanzada" if snapshot["result"] == "meta" else "Choque"
            self.status.setText(f"{title} · {reason}")
            self.play.setText("Revancha")
            if snapshot["brian"]["end"] == "tablero completo":
                self.rematch_timer.start(1800)
        else:
            self._schedule_turn()

    def _restart_completed_board(self):
        if self.isVisible():
            self.start_round()

    def failed(self, round_id, error):
        if round_id == self.round_id:
            self._stop()
            self.status.setText(f"Duelo detenido · {error}")
            self.play.setEnabled(self.ready)
