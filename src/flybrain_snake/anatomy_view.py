"""Interactive 3D view of measured contact coordinates with asynchronous queries."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QComboBox, QFormLayout, QFrame, QHeaderView, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QScrollArea, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from .anatomy import ContactAtlas, ContactView, MISSING_COORD, NT
from .widgets import BASE, INK, BrainView

TRANSMITTERS = ("Acetilcolina", "GABA", "Glutamato", "Dopamina", "Octopamina", "Serotonina", "Sin predicción")
NT_COLORS = ("#9be3a1", "#a5a1ff", "#ffd17f", "#fa9cc8", "#f58a70", "#68dae9", "#8c90a0")
REVIEW_COLORS = ("#a6a7af", "#e49a6b", "#c296ef", "#79d4c7")


def number(n):
    return f"{n:,}".replace(",", ".")


class ContactCanvas(BrainView):
    contact_picked = Signal(int)

    def __init__(self):
        super().__init__(compact=True)
        self.view = None
        self.color_mode = "transmitter"
        self.screen = np.empty((0, 3))
        self.current = None
        self.focused = False
        self.selection = None
        self.atlas_center = None

    def load(self, graph, circuit):
        super().load(graph, circuit)
        self.atlas_center, self.atlas_radius = self.center.copy(), self.radius

    def set_view(self, view: ContactView):
        valid = (view.geometry["xyz"][:, 1] != MISSING_COORD).all(axis=1)
        self.view = ContactView(view.rows[valid], view.geometry[valid], view.total)
        self.positions = self.view.geometry["xyz"][:, 1].astype(np.float64) / 1000
        self.current = None
        self.selection = None
        self.screen = np.empty((0, 3))
        self.update()

    def reset_camera(self):
        if self.atlas_center is None:
            return
        self.focused = False
        self.center, self.radius = self.atlas_center, self.atlas_radius
        self.pan = QPointF()
        self.zoom = 1.0
        self.update()

    def fit_contacts(self):
        if self.view is None or not len(self.positions):
            return
        self.focused = False
        self.center = (self.positions.max(axis=0) + self.positions.min(axis=0)) / 2
        self.radius = max(1.0, float(np.linalg.norm(self.positions - self.center, axis=1).max()))
        self.pan = QPointF()
        self.zoom = 1.0
        self.update()

    def show_selected(self, detail):
        coords = [[detail[f"{side}_pt_position_{axis}"] for axis in "xyz"] for side in ("pre", "post")]
        positions = np.asarray(coords, dtype=np.float64) / 1000
        self.selection = positions[np.isfinite(positions).all(axis=1)]
        if len(self.selection):
            self.focused = True
            self.screen = np.empty((0, 3))
            self.center = self.selection.mean(axis=0)
            self.radius = 1.0
            separation = float(np.linalg.norm(self.selection[-1] - self.selection[0]))
            self.zoom = min(12.0, .44 / max(separation, .04))
            self.pan = QPointF()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BASE)
        if self.graph is None:
            return
        if self.radius == self.atlas_radius:
            context = self._project(self.graph.xyz[self.indices])
            p.setPen(QPen(QColor("#424055"), 2.0, Qt.SolidLine, Qt.RoundCap))
            p.drawPoints(QPolygonF([QPointF(x, y) for x, y, _ in context]))
        if self.view is not None and not self.focused:
            self.screen = self._project(self.positions)
            colors = NT_COLORS if self.color_mode == "transmitter" else REVIEW_COLORS
            classes = self.view.geometry[self.color_mode]
            for i, color in enumerate(colors):
                points = self.screen[classes == i]
                p.setPen(QPen(QColor(color), 2.6, Qt.SolidLine, Qt.RoundCap))
                p.drawPoints(QPolygonF([QPointF(x, y) for x, y, _ in points]))
        if self.selection is not None and len(self.selection):
            pts = [QPointF(x, y) for x, y, _ in self._project(self.selection)]
            p.setPen(QPen(INK, 1.5))
            if len(pts) == 2:
                p.drawLine(pts[0], pts[1])
            for i, point in enumerate(pts):
                p.setPen(QPen(QColor(("#ffd17f", "#a5a1ff")[i]), 1.7))
                p.drawEllipse(point, 5, 5)
                if self.focused and len(pts) == 2:
                    p.drawText(point + QPointF(10, -8 if i == 0 else 16), ("PRE", "POST")[i])
        p.setPen(QColor("#b5b0be"))
        p.drawText(QPointF(14, self.height() - 13), "Girar · arrastrar    /    Zoom · rueda    /    Contacto · clic")

    def mouseReleaseEvent(self, e):
        if (e.button() != Qt.LeftButton or self.view is None or not len(self.screen)
                or (e.position() - self.mouse_start).manhattanLength() >= 5):
            return
        delta = self.screen[:, :2] - (e.position().x(), e.position().y())
        distances = np.einsum("ij,ij->i", delta, delta)
        i = int(distances.argmin())
        if distances[i] < 100:
            self.current = int(self.view.rows[i])
            self.contact_picked.emit(self.current)


class AnatomyPanel(QWidget):
    completed = Signal(int, object, str)
    page_size = 6000

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.graph = None
        self.atlas = None
        self.started = False
        self.closed = False
        self.request_id = 0
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="flywire-atlas")
        self.completed.connect(self.receive)
        self.contact = None
        self.query_options = None
        self.pending = None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 12, 0, 0)
        layout.setSpacing(14)
        rail = QFrame()
        rail.setObjectName("formCard")
        rail.setFixedWidth(320)
        controls = QVBoxLayout(rail)
        controls.setContentsMargins(15, 12, 15, 12)
        title = QLabel("Atlas FlyWire v783")
        title.setObjectName("section")
        controls.addWidget(title)
        self.summary = QLabel("Esperando al conectoma…")
        self.summary.setWordWrap(True)
        controls.addWidget(self.summary)
        self.region = QComboBox()
        self.region.addItem("Todos")
        self.nt = QComboBox()
        self.nt.addItem("Todos", None)
        for code, name in zip((*NT, "?"), TRANSMITTERS, strict=True):
            self.nt.addItem(name, code)
        self.review = QComboBox()
        for label, value in (("Todos", None), ("Ambos revisados", True), ("Algún extremo sin revisar", False)):
            self.review.addItem(label, value)
        self.root = QLineEdit()
        self.root.setPlaceholderText("ID neuronal")
        self.partner = QLineEdit()
        self.partner.setPlaceholderText("ID de la otra neurona")
        self.mode = QComboBox()
        for label, value in (("Ambos sentidos", "both"), ("Entradas", "post"), ("Salidas", "pre")):
            self.mode.addItem(label, value)
        self.color = QComboBox()
        self.color.addItem("Neurotransmisor", "transmitter")
        self.color.addItem("Revisión", "review")
        self.color.currentIndexChanged.connect(self.color_changed)
        form = QFormLayout()
        for label, widget in (("Neuropilo", self.region), ("Predicción", self.nt), ("Revisión", self.review),
                              ("Neurona", self.root), ("Pareja", self.partner), ("Sentido", self.mode),
                              ("Colorear por", self.color)):
            form.addRow(label, widget)
        controls.addLayout(form)
        self.show_contacts = QPushButton("Mostrar contactos")
        self.show_contacts.setObjectName("accent")
        self.show_contacts.setEnabled(False)
        self.show_contacts.clicked.connect(self.query)
        self.root.returnPressed.connect(self.query)
        self.partner.returnPressed.connect(self.query)
        controls.addWidget(self.show_contacts)
        paging = QHBoxLayout()
        self.previous = QPushButton("‹")
        self.previous.setToolTip("Página anterior")
        self.overview = QPushButton("Muestra")
        self.overview.setToolTip("Volver a la vista de conjunto")
        self.next = QPushButton("Páginas ›")
        for button in (self.previous, self.overview, self.next):
            paging.addWidget(button)
            button.setEnabled(False)
        self.previous.clicked.connect(lambda: self.fetch(max(0, self.offset - self.page_size)))
        self.overview.clicked.connect(lambda: self.fetch(None))
        self.next.clicked.connect(lambda: self.fetch(0 if self.offset is None else self.offset + self.page_size))
        controls.addLayout(paging)
        lookup = QHBoxLayout()
        self.synapse = QLineEdit()
        self.synapse.setPlaceholderText("ID de contacto")
        self.synapse.returnPressed.connect(self.lookup)
        self.jump = QPushButton("Localizar")
        self.jump.setEnabled(False)
        self.jump.clicked.connect(self.lookup)
        lookup.addWidget(self.synapse, 1)
        lookup.addWidget(self.jump)
        controls.addLayout(lookup)
        self.legend = QLabel()
        self.legend.setWordWrap(True)
        controls.addWidget(self.legend)
        self.details = QTreeWidget()
        self.details.setHeaderLabels(["Dato", "Valor"])
        self.details.setAlternatingRowColors(True)
        self.details.setMinimumHeight(145)
        self.details.header().setSectionResizeMode(0, QHeaderView.Fixed)
        self.details.header().resizeSection(0, 112)
        self.details.header().setSectionResizeMode(1, QHeaderView.Stretch)
        endpoints = QHBoxLayout()
        self.follow = {}
        for side in ("pre", "post"):
            button = self.follow[side] = QPushButton(f"Explorar {side}")
            button.hide()
            button.clicked.connect(lambda checked=False, key=side: self.follow_root(key))
            endpoints.addWidget(button)
        controls.addLayout(endpoints)
        controls.addWidget(self.details, 1)
        scroll = QScrollArea()
        scroll.setWidget(rail)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(336)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        layout.addWidget(scroll)
        right = QVBoxLayout()
        bar = QHBoxLayout()
        self.status = QLabel("Contactos individuales")
        self.status.setWordWrap(True)
        bar.addWidget(self.status, 1)
        self.canvas = ContactCanvas()
        for title, callback in (("Encuadrar contactos", self.canvas.fit_contacts), ("Cerebro entero", self.canvas.reset_camera)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            bar.addWidget(button)
        right.addLayout(bar)
        self.canvas.contact_picked.connect(self.inspect)
        right.addWidget(self.canvas, 1)
        layout.addLayout(right, 1)
        self.color_changed()

    def set_graph(self, graph, circuit):
        self.graph = graph
        self.circuit = circuit
        if self.isVisible():
            self.activate()

    def activate(self):
        if self.started or self.graph is None:
            return
        try:
            self.atlas = ContactAtlas(self.data, self.graph)
        except (OSError, ValueError) as error:
            self.summary.setText("Atlas pendiente. Ejecuta anatomy-prepare en la carpeta del proyecto.")
            self.summary.setToolTip(str(error))
            return
        self.started = True
        self.canvas.load(self.graph, self.circuit)
        self.region.addItems(sorted(self.atlas.names))
        reviewed = self.atlas.manifest["review"][3]
        self.summary.setText(f"{number(self.atlas.n)} contactos · {number(reviewed)} con ambos extremos revisados")
        self.show_contacts.setEnabled(True)
        self.jump.setEnabled(True)
        self.query()

    def close_tasks(self):
        self.closed = True
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _submit(self, job):
        self.request_id += 1
        request = self.request_id
        self.status.setText("Consultando FlyWire…")
        if self.pending is not None:
            self.pending.cancel()
        future = self.pending = self.pool.submit(job)
        def done(result):
            if self.closed or result.cancelled():
                return
            try:
                answer = result.result()
            except Exception as error:
                self.completed.emit(request, None, str(error))
            else:
                self.completed.emit(request, answer, "")
        future.add_done_callback(done)

    @staticmethod
    def _id(field):
        text = field.text().strip()
        if not text:
            return None
        if not text.isdecimal() or int(text) >= 2**63:
            raise ValueError("Introduce un ID numérico de FlyWire")
        return int(text)

    def query(self):
        if self.atlas is None:
            return
        try:
            root, partner = self._id(self.root), self._id(self.partner)
            if partner is not None and root is None:
                raise ValueError("Indica también el ID de la primera neurona")
        except ValueError as error:
            self.status.setText(str(error))
            return
        region = self.region.currentText() if self.region.currentIndex() else None
        self.query_options = {"region": region, "root": root, "partner": partner, "mode": self.mode.currentData(),
                              "transmitter": self.nt.currentData(), "review": self.review.currentData()}
        self.fetch(None)

    def fetch(self, offset):
        options = self.query_options
        self._submit(lambda: ("view", self.atlas.query(**options, limit=self.page_size, offset=offset),
                              (offset, self.atlas.profile(options["root"]) if options["root"] is not None else None)))

    def inspect(self, row):
        self._submit(lambda: self.read_detail(row))

    def lookup(self):
        if self.atlas is None:
            return
        try:
            synapse = self._id(self.synapse)
            if synapse is None:
                raise ValueError("Escribe el ID de un contacto")
        except ValueError as error:
            self.status.setText(str(error))
            return
        self._submit(lambda: self.read_detail(self.atlas.ordinal(synapse)))

    def read_detail(self, row):
        detail = self.atlas.detail(row)
        return "detail", detail, self.atlas.connection(detail)

    def follow_root(self, side):
        if self.contact:
            self.root.setText(str(self.contact[f"{side}_pt_root_id"]))
            self.partner.clear()
            self.query()

    def color_changed(self):
        self.canvas.color_mode = self.color.currentData()
        pairs = (zip(TRANSMITTERS, NT_COLORS, strict=True) if self.canvas.color_mode == "transmitter"
                 else zip(("Sin revisar", "Pre", "Post", "Ambos"), REVIEW_COLORS, strict=True))
        self.legend.setText(" · ".join(f'<span style="color:{c}">● {name}</span>' for name, c in pairs))
        self.canvas.update()

    def section(self, title, items):
        root = QTreeWidgetItem(self.details, [title])
        for key, value in items:
            item = QTreeWidgetItem(root, [str(key), str(value)])
            item.setToolTip(0, str(key))
            item.setToolTip(1, str(value))
        root.setExpanded(True)

    def receive(self, request, response, error):
        if request != self.request_id:
            return
        if error:
            self.status.setText(error)
            return
        kind, first, second = response
        self.details.clear()
        if kind == "view":
            self.contact = None
            for button in self.follow.values():
                button.hide()
            was_focused = self.canvas.focused
            self.canvas.set_view(first)
            self.offset, profile = second
            shown = len(first.rows)
            self.previous.setEnabled(self.offset is not None and self.offset > 0)
            self.overview.setEnabled(self.offset is not None)
            self.next.setText("Páginas ›" if self.offset is None else "Siguiente ›")
            self.next.setEnabled(first.total > self.page_size and
                                 (self.offset is None or self.offset + self.page_size < first.total))
            total = number(first.total)
            if self.offset is None or was_focused:
                if self.query_options["region"] is not None or self.query_options["root"] is not None:
                    self.canvas.fit_contacts()
                else:
                    self.canvas.reset_camera()
            if self.offset is None:
                self.status.setText(f"{number(len(self.canvas.positions))} puntos visibles / {total} contactos")
            else:
                self.status.setText(f"Contactos {number(self.offset + 1)}–{number(self.offset + shown)} de {total}")
            if profile is not None:
                for side, title in (("pre", "Salidas por neuropilo"), ("post", "Entradas por neuropilo")):
                    rows = sorted(profile[side].items(), key=lambda row: -row[1])
                    self.section(title, [(name, number(count)) for name, count in rows])
            return
        detail = self.contact = first
        connection = second
        for side, button in self.follow.items():
            button.setVisible(bool(detail[f"{side}_pt_root_id"]))
        self.canvas.show_selected(detail)
        self.synapse.setText(str(detail["id"]))
        self.status.setText(f"Contacto {detail['id']} · {detail['neuropil'] or 'sin región'}")
        for side in ("pre", "post"):
            root = detail[f"{side}_pt_root_id"]
            try:
                index = self.graph.index(root)
                name = self.graph.annotations["cell_type"][index] or "Sin tipo anotado"
                review = "Revisado"
            except (KeyError, TypeError):
                name, review = "Segmento del archivo", "Sin revisar"
            coords = [detail[f"{side}_pt_position_{axis}"] for axis in "xyz"]
            position = ", ".join("—" if value is None else f"{value / 1000:.2f}" for value in coords)
            self.section(side.upper(), [("Root ID", root), ("Tipo", name), ("Estado", review), ("XYZ (µm)", position)])
        self.section("Predicciones", [(name, "—" if detail[key] is None else f"{detail[key]:.2%}")
                                     for key, name in zip(NT, TRANSMITTERS[:-1], strict=True)])
        self.section("Detección", [("Conexión", detail["connection_score"]), ("Hendidura", detail["cleft_score"])])
        if connection is not None:
            count, baseline, expected = connection
            self.section("Transmisión", [("Contactos", number(count)),
                                                   ("Por neurona", f"{baseline:+d}"),
                                                   ("Por conexión", f"{expected:+.3f}")])
