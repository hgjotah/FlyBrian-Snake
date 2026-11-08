"""FlyBrain Snake desktop laboratory."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import numpy as np
from PySide6.QtCore import QProcess, QSignalBlocker, QSize, Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGridLayout, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem,
    QTreeWidget, QTreeWidgetItem,
    QVBoxLayout, QWidget,
)

from .anatomy_view import AnatomyPanel
from .arena_view import Arena
from .desktop_worker import Worker
from .game import Direction
from .widgets import Board, BrainView, EmptyState, LearningPlot, WingMark

STYLE = """
QWidget { background: #101119; color: #eeece6; font-family: "Segoe UI"; font-size: 13px; }
QLabel#title { font-size: 23px; font-weight: 700; }
QLabel#badge { color: #a7a9b7; font-size: 11px; font-weight: 650; }
QLabel#scopeTitle { color: #d4c9bd; font-size: 13px; font-weight: 650; }
QLabel#sub, QLabel#detail { color: #a7a9b7; }
QLabel#section { color: #f0e4d6; font-size: 17px; font-weight: 700; }
QLabel#formSection { color: #dfc8ae; font-size: 14px; font-weight: 650; }
QLabel#emptyTitle { color: #e9dfd4; font-size: 15px; font-weight: 650; }
QLabel#arenaStatus { color: #e1e4ce; font-size: 18px; font-weight: 700; }
QLabel#value { color: #eeece6; font-family: Consolas; font-size: 23px; font-weight: 600; }
QLabel#keycap { background: #1c1e29; color: #e6b17e; border: 1px solid #343645; border-radius: 4px; }
QFrame#playerCard QLabel#keycapActive { background: #ffb66e; color: #211914; border-radius: 4px; }
QFrame#card { border: 1px solid #2d2f3c; border-radius: 5px; background: #171822; }
QFrame#playerCard { border: 1px solid #2d2f3c; border-radius: 13px; background: #171822; }
QFrame#formCard { border: 1px solid #2d2f3c; border-radius: 7px; background: #151620; }
QFrame#formCard QLabel, QFrame#playerCard QWidget#keys, QFrame#playerCard QLabel#keysLabel { background: transparent; }
QFrame#arenaScope { border: 1px solid #353849; border-radius: 8px; background: #171822; }
QFrame#card QLabel, QFrame#playerCard QLabel, QFrame#arenaScope QLabel { background: transparent; }
QLabel#playerTitle { font-size: 21px; font-weight: 700; color: #eeece6; }
QLabel#score { color: #d5e8ad; font-family: Consolas; font-size: 33px; font-weight: 700; }
QFrame#playerCard[player="human"] QLabel#score { color: #ffb66e; }
QPushButton { background: #1c1e29; color: #eeece6; border: 1px solid #373a4a; padding: 7px 12px; border-radius: 5px; font-weight: 600; }
QPushButton:hover { background: #282b38; border-color: #74798c; }
QPushButton:focus { border-color: #9bce78; }
QPushButton#sizeStep { padding: 4px 0; font-size: 16px; }
QPushButton#accent { background: #b7ee85; color: #192314; border-color: #b7ee85; min-width: 80px; }
QPushButton#accent:hover { background: #ccf5a5; }
QPushButton:disabled, QPushButton#accent:disabled { color: #9295a3; background: #1b1d27; border-color: #30323d; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTableWidget, QTreeWidget {
    background: #191b26; border: 1px solid #353847; padding: 4px; border-radius: 4px;
    selection-background-color: #43593e; selection-color: #f7fbf3;
}
QSpinBox, QDoubleSpinBox, QLineEdit#numeric { font-family: Consolas; }
QTreeWidget, QTableWidget { alternate-background-color: #1e202c; }
QTabWidget::pane { border: 0; border-top: 1px solid #2d2f3c; }
QTabBar::tab { color: #acafbd; padding: 8px 15px; border-bottom: 2px solid transparent; }
QTabBar::tab:hover { background: #1c1e29; }
QTabBar::tab:selected { color: #b7ee85; border-bottom: 2px solid #b7ee85; font-weight: 700; }
QHeaderView::section { background: #212330; color: #dddde5; padding: 5px; border: 0; }
QSplitter::handle { background: #2d2f3c; width: 3px; }
QScrollArea { border: 0; }
QScrollBar:vertical { background: #151620; width: 9px; margin: 0; }
QScrollBar::handle:vertical { background: #383a4a; min-height: 26px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QFrame#formCard QCheckBox { background: transparent; }
QCheckBox::indicator:unchecked { width: 13px; height: 13px; border: 1px solid #646879; border-radius: 3px; background: #191b26; }
QCheckBox::indicator:unchecked:hover { border-color: #c5ad93; }
"""


def label(text, kind=None):
    widget = QLabel(text)
    if kind:
        widget.setObjectName(kind)
    return widget


def page(layout=None):
    widget = QWidget()
    widget.setLayout(layout or QVBoxLayout())
    return widget


def button(title, callback, *, primary=False):
    b = QPushButton(title)
    b.clicked.connect(callback)
    if primary:
        b.setObjectName("accent")
    return b


def form_card(title, fields, switches=()):
    card = QFrame()
    card.setObjectName("formCard")
    box = QVBoxLayout(card)
    box.setContentsMargins(13, 10, 13, 12)
    box.setSpacing(8)
    box.addWidget(label(title, "formSection"))
    form = QFormLayout()
    form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
    form.setRowWrapPolicy(QFormLayout.WrapLongRows)
    form.setHorizontalSpacing(9)
    form.setVerticalSpacing(5)
    for name, field in fields:
        form.addRow(name, field)
    box.addLayout(form)
    for switch in switches:
        box.addWidget(switch)
    box.addStretch()
    return card


def meter(title):
    card = QFrame()
    card.setObjectName("card")
    box = QVBoxLayout(card)
    box.setContentsMargins(12, 6, 12, 6)
    box.setSpacing(2)
    box.addWidget(label(title, "badge"))
    value = label("—", "value")
    box.addWidget(value)
    return card, value


def int_field(value, low, high):
    control = QSpinBox()
    control.setRange(low, high)
    control.setValue(value)
    return control


def float_field(value, low, high, digits=2):
    control = QDoubleSpinBox()
    control.setDecimals(digits)
    control.setRange(low, high)
    control.setValue(value)
    return control


def text_box():
    edit = QPlainTextEdit()
    edit.setReadOnly(True)
    return edit


class Window(QMainWindow):
    def __init__(self, data: Path, cfg, *, smoke=False):
        super().__init__()
        self.data, self.cfg, self.smoke = data, cfg, smoke
        self.setWindowTitle("FlyBrain Snake · Laboratorio de Brian")
        self.resize(1320, 840)
        self.setMinimumSize(QSize(940, 650))
        self.closing = False
        self.last = None
        self.rows = []
        self.autonomous = self.collisions = 0
        self.worker = Worker(data)
        self.worker.loaded.connect(self._loaded)
        self.worker.atlas_ready.connect(self._atlas_loaded)
        self.worker.status.connect(self._status)
        self.worker.message.connect(self._message)
        self.worker.frame.connect(self._frame)
        self.worker.memory_reset.connect(self._clear_learning)
        self.worker.episode_done.connect(self._episode)
        self.worker.episode_started.connect(self._episode_started)
        self.worker.history.connect(self._history)
        self.worker.checkpoints.connect(self._checkpoints)
        self.worker.comparison.connect(self._comparison)
        self.worker.finished.connect(self._stopped)
        self.study_stop = None
        self.study_process = QProcess(self)
        self.study_process.setProcessChannelMode(QProcess.MergedChannels)
        self.study_process.readyReadStandardOutput.connect(self._study_output)
        self.study_process.finished.connect(self._study_done)
        self.study_process.errorOccurred.connect(lambda _: self._status(self.study_process.errorString()))
        self._build()
        self.worker.start()

    def _build(self):
        outer = QVBoxLayout()
        outer.setContentsMargins(24, 14, 24, 14)
        outer.setSpacing(9)
        self.setCentralWidget(page(outer))
        header = QHBoxLayout()
        header.setSpacing(10)
        header.addWidget(WingMark())
        header.addWidget(label("FlyBrain Snake", "title"))
        header.addStretch()
        self.status = label("Cargando FlyWire…", "sub")
        self.status.setMaximumWidth(430)
        self.status.setWordWrap(False)
        header.addWidget(self.status)
        self.status.hide()
        outer.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(False)
        self.arena = Arena(self.worker, self.cfg, self._load, self._rewards)
        self.worker.training_profile.connect(self.arena.set_training_profile)
        self.worker.training_profile.connect(self._training_profile)
        self.tabs.addTab(self.arena, "Arena")
        self.training_tab = self._training()
        self.tabs.addTab(self.training_tab, "Entrenamiento")
        self.brain_tab = self._brain()
        self.tabs.addTab(self.brain_tab, "Cerebro 3D")
        self.anatomy_tab = AnatomyPanel(self.data)
        self.tabs.addTab(self.anatomy_tab, "Anatomía")
        self.telemetry_tab = self._telemetry()
        self.tabs.addTab(self.telemetry_tab, "Telemetría")
        self.memory_tab = self._memory()
        self.tabs.addTab(self.memory_tab, "Memoria")
        self.tabs.addTab(self._experiments(), "Experimentos")
        self.tabs.currentChanged.connect(self._view_changed)
        outer.addWidget(self.tabs, 1)

    def _training(self):
        tab = page()
        layout = tab.layout()
        layout.setContentsMargins(1, 12, 1, 0)
        meters = QHBoxLayout()
        self.meters = {}
        for key, name in (("episodes", "EPISODIOS"), ("apples", "MANZANAS"),
                          ("active", "NEURONAS ACTIVAS"), ("changes", "PESOS MODIFICADOS"),
                          ("speed", "VELOCIDAD")):
            card, self.meters[key] = meter(name)
            meters.addWidget(card, 1)
        self.meters["speed"].setToolTip("Milisegundos neuronales por segundo de reloj")
        layout.addLayout(meters)

        split = QSplitter(Qt.Horizontal)
        layout.addWidget(split, 1)
        left = page()
        left.layout().addWidget(label("Brian", "section"))
        self.board = Board(self.cfg.size)
        self.board_stack = QStackedWidget()
        self.board_stack.addWidget(EmptyState("Sin partida", "Nuevo bloque inicia una sesión.", "board"))
        self.board_stack.addWidget(self.board)
        left.layout().addWidget(self.board_stack, 5)
        self.board_status = label("", "sub")
        self.board_status.setWordWrap(True)
        left.layout().addWidget(self.board_status)
        chart_header = QHBoxLayout()
        chart_header.addWidget(label("Aprendizaje", "section"), 1)
        self.chart_metric = QComboBox()
        for title, key in (("Manzanas", "apples"), ("Supervivencia", "survival_moves"),
                           ("Retorno", "reward"), ("Conexiones modificadas", "memory"),
                           ("Actualizaciones por episodio", "plastic_updates"), ("Evasiones correctas", "wall_success"),
                           ("Finales por borde", "edge_termination")):
            self.chart_metric.addItem(title, key)
        self.chart_metric.currentIndexChanged.connect(self._metric)
        chart_header.addWidget(self.chart_metric)
        left.layout().addLayout(chart_header)
        self.plot = LearningPlot()
        left.layout().addWidget(self.plot, 2)
        self.episodes_detail = label("", "detail")
        self.episodes_detail.setWordWrap(True)
        left.layout().addWidget(self.episodes_detail)
        split.addWidget(left)
        split.addWidget(self._controls())
        split.setSizes([520, 740])
        split.setStretchFactor(1, 1)

        actions = QHBoxLayout()
        self.start = button("Nuevo bloque", self._new, primary=True)
        self.start.setEnabled(False)
        actions.addWidget(self.start)
        actions.addSpacing(14)
        actions.addWidget(button("Pausa", lambda: self.worker.send("pause")))
        actions.addWidget(button("Un paso", lambda: self.worker.send("step")))
        actions.addWidget(button("Reanudar", self._resume))
        actions.addSpacing(14)
        actions.addWidget(button("Detener", lambda: self.worker.send("stop")))
        actions.addWidget(button("Guardar", lambda: self.worker.send("save")))
        actions.addWidget(button("Cargar…", self._load))
        self.reset_learning = button("Reiniciar aprendizaje", self._reset_learning)
        self.reset_learning.setEnabled(False)
        self.reset_learning.setToolTip("Devuelve a Brian a sus pesos iniciales. Las sesiones guardadas quedan disponibles.")
        actions.addWidget(self.reset_learning)
        actions.addStretch()
        self.manual = QCheckBox("Manual")
        self.manual.setToolTip("Controlar a Brian con las flechas")
        self.manual.toggled.connect(lambda value: self.worker.send("manual", value))
        self.manual.toggled.connect(self._view_changed)
        actions.addWidget(self.manual)
        layout.addLayout(actions)
        self.manual_shortcuts = []
        for key, direction in (("Up", Direction.UP), ("Right", Direction.RIGHT),
                               ("Down", Direction.DOWN), ("Left", Direction.LEFT)):
            shortcut = QShortcut(QKeySequence(key), tab)
            shortcut.setContext(Qt.WidgetWithChildrenShortcut)
            shortcut.setEnabled(False)
            shortcut.activated.connect(lambda d=direction: self.worker.send("direction", d))
            self.manual_shortcuts.append(shortcut)
        return tab

    def _view_changed(self):
        selected = self.tabs.currentWidget()
        self.status.setVisible(selected is not self.arena)
        if selected is self.anatomy_tab:
            self.anatomy_tab.activate()
        if selected is not self.arena and self.arena.live and not self.arena.paused:
            self.arena.toggle_pause()
        for shortcut in self.manual_shortcuts:
            shortcut.setEnabled(selected is self.training_tab and self.manual.isChecked())
        if self.last is not None:
            self._show_trace(self.last)

    def _controls(self):
        content = page()
        box = content.layout()
        box.addWidget(label("Protocolo", "section"))
        sections = QGridLayout()
        sections.setHorizontalSpacing(12)
        sections.setVerticalSpacing(10)
        sections.setColumnStretch(0, 1)
        sections.setColumnStretch(1, 1)
        box.addLayout(sections)
        self.stage = QComboBox()
        self.stage.addItems(["0 · Validación", "1 · Condicionamiento", "2 · Toroide",
                             "3 · Paredes", "4 · Snake"])
        self.stage.setCurrentIndex(self.cfg.stage)
        self.valence = QComboBox()
        self.valence.addItem("Apetitivo (PAM01)", "appetitive")
        self.valence.addItem("Aversivo (PPL101)", "aversive")
        self.valence.setCurrentIndex(self.valence.findData(self.cfg.association_valence))
        self.practice = QComboBox()
        self.practice.addItem("Partidas completas", "snake")
        self.practice.addItem("Rectas y giros", "routes")
        self.practice.addItem("Evitar paredes", "walls")
        self.practice.addItem("Perímetro estricto", "edge_reset")
        self.practice.setCurrentIndex(self.practice.findData(self.cfg.practice))
        self._practice_mode = self.cfg.practice
        self._practice_limits = {"snake": self.cfg.max_steps if self.cfg.practice == "snake" else 0,
                                 "routes": self.cfg.max_steps if self.cfg.practice == "routes" else self.cfg.size * 4,
                                 "walls": self.cfg.max_steps if self.cfg.practice == "walls" else max(12, self.cfg.size * 2),
                                 "edge_reset": self.cfg.max_steps if self.cfg.practice == "edge_reset" else 0}
        self._profile_loaded = False
        self._session_practice = None
        self.practice.setToolTip("Perímetro estricto termina al tocar la fila o columna exterior. Evitar paredes practica esquivar y comer.")
        self.seed = QLineEdit(str(self.cfg.seed))
        self.seed.setObjectName("numeric")
        self.size = int_field(self.cfg.size, 4, 100)
        self.n = int_field(20, 1, 100000)
        self.moves = int_field(self.cfg.max_steps, 0, 100000)
        self.moves.setSpecialValueText("Sin límite")
        self.moves.setKeyboardTracking(False)
        self.moves.valueChanged.connect(lambda value: self.worker.send("step_limit", value))
        self.practice.currentIndexChanged.connect(self._practice_changed)
        self.save_every = int_field(self.cfg.save_every, 1, 100000)
        self.eval_every = int_field(self.cfg.eval_every, 0, 100000)
        self.pace = int_field(150, 0, 2000)
        self.pace.valueChanged.connect(lambda v: self.worker.send("pace", v / 1000))
        self.refresh = int_field(8, 1, 30)
        self.refresh.valueChanged.connect(lambda v: self.worker.send("refresh_hz", v))
        self.intensity = float_field(self.cfg.sensory_hz, 0, 1000)
        self.reinforcement = float_field(self.cfg.reinforcement_hz, 0, 1000)
        self.synaptic_signal = QComboBox()
        self.synaptic_signal.addItem("Por neurona", "curated")
        self.synaptic_signal.addItem("Por conexión", "contact_probabilities")
        self.synaptic_signal.setCurrentIndex(self.synaptic_signal.findData(self.cfg.synaptic_signal))
        self.synaptic_signal.setToolTip("Ganancia sináptica rápida. El modelo por conexión requiere el atlas v783.")
        self.reward_fields = {}
        rewards = []
        for key, title in (("apple", "Manzana"), ("collision", "Colisión"), ("distance", "Progreso"),
                           ("wall", "Borde"), ("idle", "Inactividad"), ("route", "Ruta corta"),
                           ("completion", "Completar tablero"), ("avoidance", "Evasión")):
            field = float_field(getattr(self.cfg.rewards, key), -1000 if key == "collision" else 0, 1000, 3)
            self.reward_fields[key] = field
            rewards.append((title, field))
        self.plastic = QCheckBox("Plasticidad habilitada")
        self.plastic.setChecked(self.cfg.plasticity)
        self.dopamine = QCheckBox("Refuerzo dopaminérgico")
        self.dopamine.setChecked(self.cfg.dopamine)
        for control in (self.plastic, self.dopamine):
            control.toggled.connect(lambda _: self.worker.send("learning_switches",
                                    self.plastic.isChecked(), self.dopamine.isChecked()))
        self.carry = QCheckBox("Conservar pesos")
        self.carry.setChecked(True)
        self.carry.setToolTip("Conservar los pesos al iniciar otro bloque")
        sections.addWidget(form_card("Escenario", (("Etapa", self.stage), ("Práctica", self.practice), ("Semilla", self.seed),
                           ("Tablero", self.size), ("Episodios", self.n), ("Límite por partida", self.moves)),
                           (self.carry,)), 0, 0)
        self.moves.setToolTip("Límite por partida. Sin límite permite continuar hasta un choque o completar el tablero.")
        sections.addWidget(form_card("Estimulación", (("Entrada (Hz)", self.intensity),
                           ("Refuerzo (Hz)", self.reinforcement), ("Valencia · etapa 1", self.valence),
                           ("Transmisión", self.synaptic_signal)),
                           (self.plastic, self.dopamine)), 0, 1)
        sections.addWidget(form_card("Recompensas", rewards), 1, 0)
        sections.addWidget(form_card("Rendimiento", (("Guardar cada N", self.save_every),
                           ("Evaluar cada N", self.eval_every), ("Espera (ms)", self.pace),
                           ("Vista (Hz)", self.refresh))), 1, 1)
        self.eval_every.setToolTip("0 desactiva la evaluación periódica")
        box.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(content)
        return scroll

    def _brain(self):
        tab = page()
        layout = tab.layout()
        self.brain_title = label("Cerebro de Brian", "section")
        layout.addWidget(self.brain_title)
        filters = QHBoxLayout()
        self.filters = {}
        for key, title in (("cell_class", "Clase"), ("region", "Neuropilo"), ("nt", "Transmisor predicho")):
            control = QComboBox()
            control.addItem(title, "")
            control.currentIndexChanged.connect(self._filter)
            self.filters[key] = control
            filters.addWidget(control, 1)
        self.lod = int_field(3200, 100, 10000)
        self.lod.setToolTip("Máximo de neuronas visibles")
        self.lod.valueChanged.connect(self._filter)
        filters.addWidget(self.lod)
        layout.addLayout(filters)
        self.brain = BrainView()
        self.brain.neuron_selected.connect(self._select)
        self.worker.brain_contacts.connect(self.brain.set_neuron_contacts)
        layout.addWidget(self.brain, 1)
        self.neuron = QLineEdit()
        self.neuron.setPlaceholderText("Root ID de FlyWire")
        self.neuron.returnPressed.connect(lambda: self._select(self.neuron.text()))
        bar = QHBoxLayout()
        bar.addWidget(self.neuron, 1)
        bar.addWidget(button("Localizar", lambda: self._select(self.neuron.text())))
        bar.addWidget(button("Ver contactos", self._open_contacts))
        layout.addLayout(bar)
        actions = QHBoxLayout()
        for name, cmd in (("Pulso", "pulse"), ("Silenciar", "silence"), ("Restaurar", "unsilence")):
            actions.addWidget(button(name, lambda checked=False, c=cmd: self.worker.send(
                "intervention", self.neuron.text(), c)))
        hz = int_field(100, 0, 1000)
        hz.setToolTip("Intensidad del pulso manual (Hz)")
        hz.valueChanged.connect(lambda v: self.worker.send("hz", v))
        actions.addWidget(hz)
        actions.addStretch()
        actions.addWidget(button("Actividad en vivo", self._live))
        layout.addLayout(actions)
        self.neuron_detail = label("Azul · KC   Verde · MBON   Violeta · DAN   Amarillo · actividad", "sub")
        self.neuron_detail.setWordWrap(True)
        layout.addWidget(self.neuron_detail)
        return tab

    def _open_contacts(self):
        self.tabs.setCurrentWidget(self.anatomy_tab)
        root = self.neuron.text().strip()
        if root:
            self.anatomy_tab.root.setText(root)
            self.anatomy_tab.partner.clear()
            self.anatomy_tab.query()

    def _telemetry(self):
        tab = page()
        h = QHBoxLayout()
        h.addWidget(label("Último movimiento", "section"), 1)
        for title, kind in (("Pulso apetitivo", "positive"), ("Pulso aversivo", "negative")):
            h.addWidget(button(title, lambda checked=False, k=kind: self.worker.send("intervention", "", k)))
        tab.layout().addLayout(h)
        self.telemetry_stack = QStackedWidget()
        self.telemetry_stack.addWidget(EmptyState("Sin señal", "Aquí aparecerá el primer movimiento.", "signal"))
        dashboard = page()
        dashboard.layout().setContentsMargins(0, 4, 0, 0)
        strip = QHBoxLayout()
        self.trace_values = {}
        for key, name in (("reward", "RECOMPENSA"), ("motor", "DNa IZQ. / DER."),
                          ("window", "VENTANA · ms"), ("cpu", "CPU"), ("ram", "RAM · MiB")):
            card, self.trace_values[key] = meter(name)
            strip.addWidget(card, 1)
        dashboard.layout().addLayout(strip)
        self.trace_heading = label("", "sub")
        dashboard.layout().addWidget(self.trace_heading)
        self.telemetry = QTreeWidget()
        self.telemetry.setHeaderLabels(["Señal", "Valor", "Detalle"])
        self.telemetry.setAlternatingRowColors(True)
        self.telemetry.setUniformRowHeights(True)
        self.telemetry.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.telemetry.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.telemetry.header().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        dashboard.layout().addWidget(self.telemetry, 1)
        self.telemetry_stack.addWidget(dashboard)
        tab.layout().addWidget(self.telemetry_stack, 1)
        return tab

    def _memory(self):
        tab = page()
        tab.layout().addWidget(label("Conexiones KC→MBON · 30 cambios mayores", "section"))
        self.weights = QTableWidget(0, 6)
        self.weights.setHorizontalHeaderLabels(["KC (root ID)", "MBON (root ID)", "Neuropilo", "Base (mV)", "Actual (mV)", "Δ %"])
        self.weights.setEditTriggers(QTableWidget.NoEditTriggers)
        self.weights.setAlternatingRowColors(True)
        self.weights.setColumnWidth(0, 147)
        self.weights.setColumnWidth(1, 147)
        tab.layout().addWidget(self.weights)
        comparison = QHBoxLayout()
        self.checkpoint_picker = QComboBox()
        self.checkpoint_picker.addItem("actual")
        comparison.addWidget(self.checkpoint_picker, 1)
        comparison.addWidget(button("Comparar actividad inicial", lambda: self.worker.send(
            "compare", self.checkpoint_picker.currentText())))
        tab.layout().addLayout(comparison)
        tab.layout().addWidget(button("Exportar sesión y pesos…", self._export))
        return tab

    def _experiments(self):
        tab = page()
        tab.layout().addWidget(label("Comparación antes / después y controles", "section"))
        explanation = label("A Inicial · B Entrenada · C Congelada · D Sin refuerzo DA · E Grafo reordenado · "
                            "F Refuerzos desordenados · G Retención · H Generalización · I Ablación MBON", "sub")
        explanation.setWordWrap(True)
        tab.layout().addWidget(explanation)
        form = QFormLayout()
        self.study_labels = QLineEdit("ABCDEFGHI")
        self.study_train = int_field(12, 1, 100000)
        self.study_test = int_field(10, 1, 100000)
        form.addRow("Condiciones", self.study_labels)
        form.addRow("Episodios de entrenamiento", self.study_train)
        form.addRow("Semillas nuevas de evaluación", self.study_test)
        tab.layout().addLayout(form)
        controls = QHBoxLayout()
        controls.addWidget(button("Ejecutar estudio", self._study))
        controls.addWidget(button("Detener estudio", self._stop_study))
        tab.layout().addLayout(controls)
        self.study_log = text_box()
        tab.layout().addWidget(self.study_log)
        return tab

    def _loaded(self, graph, circuit):
        self.graph = graph
        self.brain.load(graph, circuit)
        self.anatomy_tab.set_graph(graph, circuit)
        for key, choices in (("cell_class", np.unique(graph.annotations["cell_class"])),
                             ("region", sorted(graph.region_names)), ("nt", np.unique(graph.annotations["top_nt"]))):
            self.filters[key].addItems([str(x) for x in choices if x])
        self.start.setEnabled(True)
        self.reset_learning.setEnabled(True)
        self.arena.connect_brain(graph, circuit)

    def _atlas_loaded(self, regions, view):
        self.arena.connect_atlas(regions, view)
        if regions is not None:
            if not self._profile_loaded:
                self.synaptic_signal.setCurrentIndex(self.synaptic_signal.findData("contact_probabilities"))
            self.brain_title.setText("Cerebro de Brian · FlyWire v783")
            self.brain.set_atlas(regions, view)
            if self.brain.selected is not None:
                self.worker.send("brain_contacts", str(self.graph.ids[self.brain.selected]))

    def _practice_changed(self):
        mode = self.practice.currentData()
        self._practice_limits[self._practice_mode] = self.moves.value()
        self._practice_mode = mode
        with QSignalBlocker(self.moves):
            self.moves.setValue(0 if mode == "edge_reset" or (mode == "snake" and self.stage.currentIndex() == 4)
                                else self._practice_limits[mode])
            if mode != "snake":
                self.stage.setCurrentIndex(4)
                if mode in ("routes", "walls") and self.moves.value() == 0:
                    self.moves.setValue(self.size.value() * 4)
        self.arena.cancel()
        self.worker.send("practice", mode, self.moves.value())
        if mode == "walls":
            self.chart_metric.setCurrentIndex(self.chart_metric.findData("wall_success"))
        elif mode == "edge_reset":
            self.chart_metric.setCurrentIndex(self.chart_metric.findData("edge_termination"))
        elif self.chart_metric.currentData() in ("wall_success", "edge_termination"):
            self.chart_metric.setCurrentIndex(self.chart_metric.findData("apples"))

    def _training_profile(self, profile):
        if profile is None:
            self._session_practice = None
            return
        changed = self._session_practice is not None and self._session_practice != profile.practice
        self._session_practice = profile.practice
        if not self._profile_loaded:
            self._practice_limits["routes"] = profile.size * 4
            self._practice_limits["walls"] = max(12, profile.size * 2)
            self._practice_limits["edge_reset"] = 0
            if profile.practice != "snake":
                self._practice_limits["snake"] = 0
        self._profile_loaded = True
        if profile.practice == "walls":
            self.chart_metric.setCurrentIndex(self.chart_metric.findData("wall_success"))
        elif profile.practice == "edge_reset":
            self.chart_metric.setCurrentIndex(self.chart_metric.findData("edge_termination"))
        elif self.chart_metric.currentData() in ("wall_success", "edge_termination"):
            self.chart_metric.setCurrentIndex(self.chart_metric.findData("apples"))
        self.cfg = profile
        self._practice_mode = profile.practice
        self._practice_limits[profile.practice] = profile.max_steps
        with QSignalBlocker(self.practice):
            self.practice.setCurrentIndex(self.practice.findData(profile.practice))
        self.stage.setCurrentIndex(profile.stage)
        self.size.setValue(profile.size)
        self.seed.setText(str(profile.seed))
        with QSignalBlocker(self.moves):
            self.moves.setValue(profile.max_steps)
        self.synaptic_signal.setCurrentIndex(self.synaptic_signal.findData(profile.synaptic_signal))
        self.valence.setCurrentIndex(self.valence.findData(profile.association_valence))
        self.intensity.setValue(profile.sensory_hz)
        self.reinforcement.setValue(profile.reinforcement_hz)
        with QSignalBlocker(self.plastic), QSignalBlocker(self.dopamine):
            self.plastic.setChecked(profile.plasticity)
            self.dopamine.setChecked(profile.dopamine)
        self.save_every.setValue(profile.save_every)
        self.eval_every.setValue(profile.eval_every)
        for name, field in self.reward_fields.items():
            field.setValue(getattr(profile.rewards, name))
        if changed:
            self.last = None
            self.board_stack.setCurrentIndex(0)
            self.board_status.clear()
            self.telemetry_stack.setCurrentIndex(0)

    def _rewards(self):
        return replace(self.cfg.rewards, **{key: field.value() for key, field in self.reward_fields.items()})

    def _configuration(self):
        return replace(self.cfg, stage=self.stage.currentIndex(), practice=self.practice.currentData(),
                       seed=int(self.seed.text()), size=self.size.value(),
                       max_steps=self.moves.value(), save_every=self.save_every.value(),
                       association_valence=self.valence.currentData(), synaptic_signal=self.synaptic_signal.currentData(),
                       eval_every=self.eval_every.value(), sensory_hz=self.intensity.value(),
                       reinforcement_hz=self.reinforcement.value(), plasticity=self.plastic.isChecked(),
                       dopamine=self.dopamine.isChecked(), rewards=self._rewards(),
                       wall_memory_hz=0.0, wall_preview_cells=1,
                       learning=replace(self.cfg.learning, post_required=False)
                       if self.practice.currentData() != "snake" else self.cfg.learning)

    def _new(self):
        try:
            cfg = self._configuration()
            self.arena.cancel()
            self.manual.setChecked(False)
            self.worker.send("new", cfg, self.n.value(), self.carry.isChecked())
        except ValueError as error:
            self._message(str(error))

    def _reset_learning(self):
        self.arena.cancel()
        self.manual.setChecked(False)
        self.worker.send("reset_memory")

    def _clear_learning(self):
        self.last = None
        self.arena.reset_preview()
        self.board_stack.setCurrentIndex(0)
        self.telemetry_stack.setCurrentIndex(0)
        self.weights.setRowCount(0)
        self.brain.delta = None
        self.brain.active = np.empty(0, dtype=np.int32)
        self.brain.update()
        self.board_status.clear()
        self.episodes_detail.clear()
        for meter in self.meters.values():
            meter.setText("0")

    def _resume(self):
        try:
            cfg = self._configuration()
        except ValueError as error:
            self._message(str(error))
            return
        self.arena.cancel()
        self.manual.setChecked(False)
        self.worker.send("resume", self.n.value(), cfg)

    def _load(self):
        filename, _ = QFileDialog.getOpenFileName(self, "Cargar sesión de Brian", str(self.data / "sessions"),
                                                  "Sesión SQLite (*.sqlite)")
        if filename:
            self.arena.cancel()
            self.worker.send("load", Path(filename), self.n.value())

    def _export(self):
        directory = QFileDialog.getExistingDirectory(self, "Carpeta de exportación")
        if directory:
            self.worker.send("export", Path(directory))

    def _study(self):
        from dataclasses import asdict
        from datetime import datetime
        from .__main__ import dump

        if self.study_process.state() != QProcess.NotRunning:
            return
        try:
            cfg = self._configuration()
        except ValueError as error:
            self._message(str(error))
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        root = self.data / "runs" / f"study_{stamp}"
        config = root.with_suffix(".json")
        self.study_stop = root.with_suffix(".stop")
        dump(config, asdict(cfg))
        self.study_log.setPlainText(str(root))
        executable = Path(sys.executable)
        if executable.name.lower() == "pythonw.exe":
            executable = executable.with_name("python.exe")
        self.study_process.start(str(executable), ["-u", "-m", "flybrain_snake", "--data", str(self.data),
                                                   "--config", str(config), "study", str(root), "--train",
                                                   str(self.study_train.value()), "--test", str(self.study_test.value()),
                                                   "--conditions", self.study_labels.text().strip(), "--stop-file", str(self.study_stop)])

    def _stop_study(self):
        if self.study_process.state() != QProcess.NotRunning and self.study_stop is not None:
            self.study_stop.touch()
            self.study_log.appendPlainText("Guardando al finalizar la ventana actual…")

    def _study_output(self):
        self.study_log.appendPlainText(bytes(self.study_process.readAllStandardOutput()).decode("utf-8", errors="replace").strip())

    def _study_done(self, status, _):
        self._study_output()
        self.study_log.appendPlainText("Estudio guardado." if status == 0 else f"Estudio finalizado con código {status}.")
        if self.study_stop is not None:
            self.study_stop.unlink(missing_ok=True)
            self.study_stop = None
        if self.closing:
            QTimer.singleShot(0, self.close)

    def _filter(self):
        self.brain.limit = self.lod.value()
        self.brain.apply_filters({key: ctrl.currentText() if ctrl.currentIndex() else None
                                  for key, ctrl in self.filters.items()})

    def _checkpoints(self, labels):
        current = self.checkpoint_picker.currentText()
        self.checkpoint_picker.clear()
        self.checkpoint_picker.addItems(["actual", *labels])
        if current in labels:
            self.checkpoint_picker.setCurrentText(current)

    def _comparison(self, result):
        self.brain.delta = result["trained"] - result["baseline"]
        self.brain.update()
        self.tabs.setCurrentWidget(self.brain_tab)
        self.neuron_detail.setText(f"{result['label']} · misma señal KC-A y semilla · "
                                  f"{result['baseline'].sum():,} → {result['trained'].sum():,} spikes · "
                                  f"{result['memory']['changed']:,} conexiones modificadas")
        self._inspect()

    def _live(self):
        self.brain.delta = None
        self.brain.update()
        self._inspect()

    def _select(self, root):
        if self.brain.graph is None:
            return
        try:
            self.brain.select(root)
            self.neuron.setText(root)
            self.worker.send("brain_contacts", root)
            self._inspect()
        except (ValueError, KeyError, OverflowError) as error:
            self._status(str(error))

    def _inspect(self):
        index = self.brain.selected
        if index is None:
            return
        a = self.graph.annotations
        value = self.last["counts"][index] if self.last else "—"
        if self.brain.delta is not None:
            value = f"{int(self.brain.delta[index]):+d} de diferencia en"
        self.neuron_detail.setText(f"{a['cell_type'][index] or 'Tipo sin anotar'} · "
                                  f"{a['top_nt'][index] or 'Transmisor sin predicción'} · {value} spikes en la última ventana")

    def _status(self, message):
        self.status.setText(message)

    def _message(self, message):
        if self.smoke:
            self.status.setText(message)
        else:
            QMessageBox.information(self, "Laboratorio de Brian", message)

    def _episode_started(self, state):
        self.last = None
        self.board.state = state["board"]
        self.board_stack.setCurrentWidget(self.board)
        self.board.update()
        self.board_status.setText(f"Esquivar la pared y recoger la manzana · intento {state['episode'] + 1}")
        self.meters["apples"].setText("0")
        self.meters["active"].setText("0")
        self.meters["changes"].setText(f"{state['memory']['changed']:,}")
        self.brain.active = np.empty(0, np.int32)
        self.brain.update()

    def _frame(self, state):
        self.last = state
        step = state["step"]
        if state["board"]:
            board = state["board"]
            self.board_stack.setCurrentWidget(self.board)
            self.board.state = board
            self.board.update()
            self.meters["apples"].setText(str(board["apples"]))
            title = step.get("objective", "Partida")
            self.board_status.setText(f"{title} {step['episode'] + 1} · movimiento {step['step']} · "
                                      f"{step['direction']} · recompensa {step['total']:+.3f} · {board['end'] or 'en curso'}"
                                      f" · cálculo {state['episode_time']:.1f} s")
        self.meters["active"].setText(f"{step['active']:,}")
        self.meters["changes"].setText(f"{state.get('memory_state', step['memory'])['changed']:,}")
        self.meters["speed"].setText(f"{step['neural_ms'] * 1000 / max(.01, step['wall_ms']):.0f}")
        self.brain.active = np.flatnonzero(state["counts"])
        if self.tabs.currentWidget() is self.brain_tab and self.brain.delta is None:
            self.brain.update()
        self._inspect()
        self._show_trace(state)

    def _show_trace(self, state):
        if self.tabs.currentWidget() is self.telemetry_tab:
            self._show_telemetry(state["step"])
        elif self.tabs.currentWidget() is self.memory_tab:
            self._show_weights(state["memory"])

    def _show_telemetry(self, step):
        decision = step["decision"]
        cpu = step.get("cpu_pct_one_core")
        values = {"reward": f"{step['total']:+.2f}",
                  "motor": f"{decision['left_spikes']} / {decision['right_spikes']}",
                  "window": f"{step['neural_ms']:.0f} / {step['wall_ms']:.0f}",
                  "cpu": "—" if cpu is None else f"{cpu:.0f}%",
                  "ram": f"{step['rss_mib']:.0f}"}
        for key, value in values.items():
            self.trace_values[key].setText(value)
        self.trace_heading.setText(f"{step['source']} · {decision['provenance']} · neural / real")
        self.telemetry_stack.setCurrentIndex(1)
        mono = QFont("Consolas")
        expanded = {self.telemetry.topLevelItem(i).text(0): self.telemetry.topLevelItem(i).isExpanded()
                    for i in range(self.telemetry.topLevelItemCount())}
        scroll_position = self.telemetry.verticalScrollBar().value()
        self.telemetry.setUpdatesEnabled(False)
        self.telemetry.clear()
        def section(title, rows):
            if not rows:
                return
            group = QTreeWidgetItem(self.telemetry, [title])
            group.setExpanded(expanded.get(title, True))
            for name, value, detail in rows:
                item = QTreeWidgetItem(group, [name, value, detail])
                for col in (1, 2):
                    item.setFont(col, mono)
        labels = {"apple": "Manzana", "collision": "Choque", "distance": "Progreso",
                  "wall": "Borde", "idle": "Inactividad", "route": "Ruta corta", "completion": "Completar tablero",
                  "avoidance": "Evasión", "practice_failure": "Ejercicio fallido", "edge_contact": "Perímetro estricto"}
        section("Recompensa", [(labels.get(key, key), f"{value:+.3f}", "")
                               for key, value in step["reward"].items()])
        section("Poblaciones", [(name, f"{row['spikes']} spikes", f"{row['hz']:.1f} Hz")
                                for name, row in step.get("groups", {}).items()
                                if name in ("KC", "MBON", "PAM", "PPL1", "motor_left", "motor_right")])
        directions = {"front": "Frente", "left": "Izquierda", "right": "Derecha"}
        section("Límites percibidos", [(directions[name], str(distance), "casillas libres al decidir")
                                       for name, distance in step.get("wall_distances", {}).items()])
        section("Estímulos", [(row["name"], f"{row['rate_hz']:.1f} Hz",
                               f"{len(row['ids'])} neuronas · {row['duration_ms']:.0f} ms")
                              for row in step.get("input", []) + step.get("modulators", [])])
        self.telemetry.verticalScrollBar().setValue(scroll_position)
        self.telemetry.setUpdatesEnabled(True)

    def _show_weights(self, rows):
        self.weights.setRowCount(len(rows))
        mono = QFont("Consolas")
        for i, row in enumerate(rows):
            for j, value in enumerate((row["pre_id"], row["post_id"], row["region"], f"{row['base_mv']:.3f}",
                                       f"{row['current_mv']:.3f}", f"{row['change_pct']:+.2f}")):
                item = QTableWidgetItem(value)
                if j != 2:
                    item.setFont(mono)
                if j >= 3:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                age = row["last_pre_age_ms"]
                detail = f"{row['pre_type']} → {row['post_type']} · {row['compartment']}"
                if age is not None:
                    detail += f" · pre-spike {age:.1f} ms antes del ajuste"
                item.setToolTip(detail)
                self.weights.setItem(i, j, item)

    def _metric(self):
        self.plot.series_key = self.chart_metric.currentData()
        self.plot.update()

    def _history(self, rows):
        self.rows = list(rows)
        eligible = [r for r in self.rows if "apples" in r and not r["manual"] and r.get("split") != "evaluation"]
        self.autonomous = len(eligible)
        self.collisions = sum(r["collision"] for r in eligible)
        self.plot.rows = self.rows
        self.plot.update()
        self.meters["episodes"].setText(str(len(rows)))

    def _episode(self, row):
        self.rows.append(row)
        self.plot.update()
        self.meters["episodes"].setText(str(len(self.rows)))
        if "apples" in row:
            if not row["manual"]:
                self.autonomous += 1
                self.collisions += row["collision"]
            collisions = self.collisions / self.autonomous if self.autonomous else 0
            self.episodes_detail.setText(f"Último episodio · {row['steps']} pasos · "
                                         f"retorno {row['reward']:+.2f} · "
                                         f"colisiones autónomas {collisions:.0%} · {row['end']}")
        else:
            self.episodes_detail.setText(f"Condicionamiento {row['cue']} · MBON {row['mbon_spikes']} spikes · "
                                         f"PAM {row['pam_spikes']} · PPL1 {row['ppl1_spikes']} spikes · {row['changed']} cambios")
            self.meters["changes"].setText(str(row["changed"]))

    def closeEvent(self, event):
        if self.study_process.state() != QProcess.NotRunning:
            self.closing = True
            self._stop_study()
            self._status("Guardando estudio…")
            event.ignore()
        elif self.worker.isRunning():
            self.arena.cancel()
            self.closing = True
            self.worker.send("exit")
            self._status("Guardando sesión…")
            event.ignore()
        else:
            self.anatomy_tab.close_tasks()
            event.accept()

    def _stopped(self):
        if self.closing:
            QTimer.singleShot(0, self.close)


def launch(data: Path, cfg, smoke: bool = False, session: Path | None = None):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    w = Window(data, cfg, smoke=smoke)
    if session is not None:
        w.worker.send("load", session.resolve(), 20)
    w.show()
    if smoke:
        def loaded(*_):
            def verify():
                target = data / "runs" / "desktop.png"
                target.parent.mkdir(parents=True, exist_ok=True)
                selected = w.tabs.currentWidget()
                w.tabs.setCurrentWidget(w.brain_tab)
                timings = []
                for _ in range(6):
                    started = time.perf_counter()
                    w.brain.grab()
                    timings.append((time.perf_counter() - started) * 1000)
                from .__main__ import dump
                dump(target.with_name("desktop-benchmark.json"),
                     {"points": len(w.brain.indices), "viewport_dip": [w.brain.width(), w.brain.height()],
                      "paint_ms": timings, "median_ms": float(np.median(timings)),
                      "platform": app.platformName()})
                w.tabs.setCurrentWidget(selected)
                app.setProperty("smoke_ok", w.grab().save(str(target)))
                w.close()
            QTimer.singleShot(450, verify)
        w.worker.loaded.connect(loaded)
        QTimer.singleShot(20000, w.close)
    code = app.exec()
    return code if not smoke or app.property("smoke_ok") else 2
