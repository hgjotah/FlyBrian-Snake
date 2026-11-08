"""Native Qt board, scientific plots and a bounded CPU 3D point renderer."""
from __future__ import annotations

import math
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QWidget

from .anatomy import MISSING_COORD, NT

INK = QColor("#f2eee3")
MUTED = QColor("#a2a4b3")
ACCENT = QColor("#aabecf")
GRID = QColor("#383a4a")
BASE = QColor("#171822")
BRIAN = "#b7ee85"
HUMAN = "#ffb66e"


class WingMark(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedSize(38, 38)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor("#393846"), 1))
        p.setBrush(QColor("#1d1e29"))
        p.drawRoundedRect(QRectF(.5, .5, 37, 37), 9, 9)
        p.setPen(QPen(QColor("#e8d3b5"), 1.35, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        for side in (-1, 1):
            p.save()
            p.translate(19, 18)
            p.scale(side, 1)
            wing = QPainterPath(QPointF(2, -3))
            wing.cubicTo(QPointF(6, -9), QPointF(13, -9), QPointF(13, -4))
            wing.cubicTo(QPointF(14, 1), QPointF(7, 6), QPointF(2, 3))
            p.drawPath(wing)
            p.drawLine(QPointF(3, -2), QPointF(9, -4))
            p.drawLine(QPointF(2, -6), QPointF(5, -9))
            p.restore()
        p.drawEllipse(QRectF(16, 12, 6, 7))
        p.drawEllipse(QRectF(16.4, 18, 5.2, 10))


def draw_empty_icon(p, center, kind):
    p.save()
    p.translate(center)
    p.setPen(QPen(QColor("#393745"), 1))
    p.setBrush(QColor("#1d1e28"))
    p.drawRoundedRect(QRectF(-19, -19, 38, 38), 10, 10)
    p.setPen(QPen(QColor("#d6bfa2"), 1.45, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    if kind == "board":
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(QRectF(-8, -8, 16, 16), 2, 2)
        p.drawLine(QPointF(-8, 0), QPointF(8, 0))
        p.drawLine(QPointF(0, -8), QPointF(0, 8))
    elif kind == "signal":
        p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in
                                  ((-11, 0), (-5, 0), (-2, -7), (2, 7), (5, 0), (11, 0))]))
    else:
        p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in
                                  ((-8, -7), (-8, 8), (9, 8))]))
        p.drawPolyline(QPolygonF([QPointF(x, y) for x, y in
                                  ((-5, 4), (-1, 0), (3, 2), (8, -5))]))
    p.restore()


class EmptyState(QWidget):
    def __init__(self, title, detail, kind):
        super().__init__()
        self.title, self.detail, self.kind = title, detail, kind

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        center = self.rect().center()
        draw_empty_icon(p, QPointF(center.x(), center.y() - 23), self.kind)
        p.setPen(INK)
        p.setFont(QFont("Segoe UI", 12, QFont.DemiBold))
        p.drawText(QRectF(0, center.y() + 3, self.width(), 24), Qt.AlignCenter, self.title)
        p.setPen(MUTED)
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(QRectF(0, center.y() + 28, self.width(), 22), Qt.AlignCenter, self.detail)


class Board(QWidget):
    def __init__(self, size, color=BRIAN, *, dense_grid=False):
        super().__init__()
        self.state = {"size": size, "body": [(size // 2, size // 2), (size // 2 - 1, size // 2)]}
        self.accent = QColor(color)
        self.dense_grid = dense_grid
        self.setMinimumSize(228, 225)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        size = self.state["size"]
        side = min(self.width(), self.height()) - 36
        cell = side / size
        ox, oy = (self.width() - side) / 2, (self.height() - side) / 2
        board = QRectF(ox, oy, side, side)
        grid = QColor("#333647")
        p.setPen(QPen(grid, 1))
        p.setBrush(BASE)
        p.drawRoundedRect(board, 5, 5)
        if self.dense_grid or cell >= 7:
            if cell < 7:
                p.setPen(QPen(grid, .65))
            for i in range(1, size):
                p.drawLine(QPointF(ox + i * cell, oy), QPointF(ox + i * cell, oy + side))
                p.drawLine(QPointF(ox, oy + i * cell), QPointF(ox + side, oy + i * cell))
        edge = QColor("#f2847d") if self.state.get("end") == "pared" else self.accent.darker(144)
        p.setPen(QPen(edge, 1.5))
        for x, y, dx, dy in ((ox, oy, 1, 1), (board.right(), oy, -1, 1),
                             (ox, board.bottom(), 1, -1), (board.right(), board.bottom(), -1, -1)):
            p.drawLine(QPointF(x + 2 * dx, y), QPointF(x + 11 * dx, y))
            p.drawLine(QPointF(x, y + 2 * dy), QPointF(x, y + 11 * dy))

        def center(point):
            x, y = point
            return QPointF(ox + (x + .5) * cell, oy + (y + .5) * cell)

        apple = self.state.get("apple")
        if apple is not None:
            origin = center(apple)
            fruit = QColor("#f58978")
            p.setPen(Qt.NoPen)
            p.setBrush(fruit)
            p.drawEllipse(origin, cell * .22, cell * .22)
            if cell >= 12:
                p.setPen(QPen(QColor("#fac5a3"), max(1, cell * .048), Qt.SolidLine, Qt.RoundCap))
                p.drawLine(origin + QPointF(0, -cell * .17), origin + QPointF(cell * .09, -cell * .31))
                p.setPen(Qt.NoPen)
                p.setBrush(QColor("#ffc4a5"))
                p.drawEllipse(origin + QPointF(-cell * .07, -cell * .06), cell * .046, cell * .046)

        body = self.state["body"]
        body_color = self.accent.darker(132)
        p.setPen(QPen(body_color, cell * .52, Qt.SolidLine, Qt.RoundCap))
        for a, b in zip(body, body[1:]):
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                p.drawLine(center(a), center(b))
        p.setPen(Qt.NoPen)
        p.setBrush(body_color)
        for point in body[1:]:
            p.drawEllipse(center(point), cell * .26, cell * .26)
        head = center(body[0])
        p.setBrush(self.accent)
        p.drawRoundedRect(QRectF(head.x() - cell * .34, head.y() - cell * .34, cell * .68, cell * .68),
                          cell * .20, cell * .20)
        if cell >= 14:
            p.setBrush(BASE)
            dx, dy = ((0, -1), (1, 0), (0, 1), (-1, 0))[self.state.get("direction", 1)]
            for offset in (-.13, .13):
                eye = QPointF((dx * .12 - dy * offset) * cell, (dy * .12 + dx * offset) * cell)
                p.drawEllipse(head + eye, .047 * cell, .047 * cell)


class LearningPlot(QWidget):
    def __init__(self):
        super().__init__()
        self.rows = []
        self.series_key = "apples"
        self.setMinimumHeight(160)

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setFont(QFont("Consolas", 9))
        p.setPen(MUTED)
        rows = [row for row in self.rows[-300:] if "apples" in row and self.series_key in row and not row.get("manual")
                and row.get("split", "training") == "training"][-180:]
        if not rows:
            draw_empty_icon(p, QPointF(self.width() / 2, self.height() / 2 - 12), "plot")
            p.drawText(QRectF(0, self.height() / 2 + 13, self.width(), 22), Qt.AlignCenter, "Sin episodios")
            return
        vals = np.asarray([r[self.series_key] for r in rows], dtype=float)
        average = [vals[max(0, i - 9):i + 1].mean() for i in range(len(vals))]
        y_max, y_min = max(1.0, float(vals.max())), min(0.0, float(vals.min()))
        area = QRectF(34, 14, self.width() - 53, self.height() - 44)
        for i in range(3):
            y = area.bottom() - area.height() * i / 2
            p.setPen(GRID)
            p.drawLine(QPointF(area.left(), y), QPointF(area.right(), y))
            p.setPen(MUTED)
            p.drawText(QPointF(2, y + 4), f"{y_min + (y_max - y_min) * i / 2:.1f}")
        for data, color in ((vals, QColor("#b78368")), (average, ACCENT)):
            curve = QPainterPath()
            for i, value in enumerate(data):
                point = QPointF(area.left() + area.width() * i / max(1, len(data) - 1),
                                area.bottom() - area.height() * (value - y_min) / (y_max - y_min))
                if i == 0:
                    curve.moveTo(point)
                else:
                    curve.lineTo(point)
                p.setPen(QPen(color, 4))
                p.drawPoint(point)
            p.setPen(QPen(color, 1.5))
            p.drawPath(curve)
        p.setPen(MUTED)
        p.drawText(QPointF(36, self.height() - 7), "Por episodio   ·   Media móvil de 10 episodios")


class BrainView(QWidget):
    neuron_selected = Signal(str)
    MAX_LINKS = 250
    CONTACT_COLORS = ("#68816c", "#7374a4", "#94785e", "#9b688f", "#91665c", "#5c858d", "#515366")

    def __init__(self, *, compact=False):
        super().__init__()
        self.compact = compact
        self.setMinimumHeight(225 if compact else 300)
        self.pulses = self.contact_pulses = ()
        self.pulse_started = 0.0
        self.pulse_timer = QTimer(self)
        self.pulse_timer.setInterval(33)
        self.pulse_timer.timeout.connect(self._animate)
        self.graph = None
        self.circuit = None
        self.indices = np.empty(0, np.int32)
        self.active = np.empty(0, np.int32)
        self.delta = None
        self.rotation = [-.30, .25]
        self.pan = QPointF()
        self.zoom = 1.0
        self.limit = 1250 if compact else 3200
        self.selected = None
        self.atlas_geometry = None
        self.contact_layers = ()
        self.contact_raster = None
        self.selected_contacts = ()
        self.links = []
        self.projections = None
        self.filters = {}
        self.center = np.zeros(3)
        self.radius = 1.0

    def load(self, graph, circuit):
        self.graph, self.circuit = graph, circuit
        positions = graph.xyz[np.isfinite(graph.xyz).all(axis=1)]
        self.center = np.median(positions, axis=0)
        self.radius = float(np.quantile(np.linalg.norm(positions - self.center, axis=1), .98))
        self.apply_filters({})

    def apply_filters(self, filters: dict):
        self.filters = {k: v for k, v in filters.items() if v}
        if self.graph is None:
            return
        candidates = self.graph.select(**self.filters)
        candidates = candidates[np.isfinite(self.graph.xyz[candidates]).all(axis=1)]
        self.available = len(candidates)
        if len(candidates) > self.limit:
            candidates = candidates[np.linspace(0, len(candidates) - 1, self.limit, dtype=int)]
        self.indices = candidates
        self._filter_contacts()
        self.update()

    def set_atlas(self, regions, view):
        self.atlas_regions = regions
        geometry = view.geometry
        valid = (geometry["xyz"][:, 1] != MISSING_COORD).all(axis=1)
        self.atlas_geometry = geometry[valid]
        self.atlas_positions = self.atlas_geometry["xyz"][:, 1].astype(np.float32) / 1000
        self._filter_contacts()
        self.update()

    def _filter_contacts(self):
        if self.atlas_geometry is None:
            return
        geometry = self.atlas_geometry
        mask = np.ones(len(geometry), dtype=bool)
        region = self.filters.get("region")
        if region in self.atlas_regions:
            mask &= geometry["region"] == self.atlas_regions.index(region)
        nt = self.filters.get("nt", "").lower()
        if nt in NT:
            mask &= geometry["transmitter"] == NT.index(nt)
        self.contact_layers = tuple(self.atlas_positions[mask & (geometry["transmitter"] == i)]
                                    for i in range(len(self.CONTACT_COLORS)))
        self.contact_projection = None

    def set_neuron_contacts(self, root, view):
        if self.selected is None or int(root) != self.graph.ids[self.selected]:
            return
        positions = view.geometry["xyz"]
        self.selected_contacts = tuple(side[(side != MISSING_COORD).all(axis=1)].astype(np.float32) / 1000
                                       for side in (positions[:, 0], positions[:, 1]))
        self.update()

    def select(self, root_id: str):
        i = self.graph.index(int(root_id))
        self.selected = i
        self.selected_contacts = ()
        start, stop = self.graph.offsets[i:i + 2]
        targets = self.graph.targets[start:stop]
        if len(targets) and self.atlas_geometry is None:
            count = self.graph.counts[start:stop]
            top = np.argsort(count, kind="stable")[-self.MAX_LINKS:]
            self.links = np.unique(targets[top])
        else:
            self.links = []
        self.update()

    def _project(self, coords):
        x, y = self.rotation
        rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        r = (coords - self.center) @ (rx @ ry).T
        scale = min(self.width(), self.height()) * (.44 if self.compact else .42) * self.zoom / max(self.radius, 1)
        r[:, 0] = r[:, 0] * scale + self.width() / 2 + self.pan.x()
        r[:, 1] = r[:, 1] * scale + self.height() / 2 + self.pan.y()
        return r

    def _draw_contact_map(self, painter):
        key = (self.width(), self.height(), *self.rotation, self.zoom, self.pan.x(), self.pan.y())
        dpr = self.devicePixelRatioF()
        key = (*key, dpr)
        if self.contact_raster is None or key != self.contact_raster[0]:
            image = QPixmap(round(self.width() * dpr), round(self.height() * dpr))
            image.setDevicePixelRatio(dpr)
            image.fill(Qt.transparent)
            layer_painter = QPainter(image)
            layer_painter.setRenderHint(QPainter.Antialiasing)
            for color, layer in zip(self.CONTACT_COLORS, self.contact_layers):
                points = QPolygonF([QPointF(x, y) for x, y, _ in self._project(layer)])
                layer_painter.setPen(QPen(QColor(color), 1.8 if self.compact else 2.2, Qt.SolidLine, Qt.RoundCap))
                layer_painter.drawPoints(points)
            layer_painter.end()
            self.contact_raster = key, image
        painter.drawPixmap(0, 0, self.contact_raster[1])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), BASE)
        p.setPen(QColor("#d8c8d8"))
        if self.graph is None:
            if not self.compact:
                p.drawText(self.rect(), Qt.AlignCenter, "Cargando coordenadas de FlyWire…")
            return
        if self.compact:
            p.setPen(QPen(QColor("#2b2e40"), 1))
            center = QPointF(self.width() / 2, self.height() / 2)
            radius = min(self.width(), self.height()) * .41
            for fraction in (.55, 1):
                p.drawEllipse(center, radius * fraction, radius * fraction)
        if self.atlas_geometry is not None:
            self._draw_contact_map(p)
            if self.compact:
                self._draw_pulses(p)
                return
        points = self._project(self.graph.xyz[self.indices])
        self.projections = points
        palette = [
            (QColor("#61506b"), np.ones(len(points), dtype=bool), 1.6),
            (QColor("#74aafa"), np.isin(self.indices, self.circuit.groups["KC"]), 2.2),
            (QColor("#83edcb"), np.isin(self.indices, self.circuit.groups["MBON"]), 3.3),
            (QColor("#db91ff"), np.isin(self.indices, self.circuit.groups["DAN"]), 3.0),
            (QColor("#71b5ff"), np.isin(self.indices, self.circuit.groups["motor_left"]), 4.5),
            (QColor("#ffb09f"), np.isin(self.indices, self.circuit.groups["motor_right"]), 4.5),
            (QColor("#ffe097"), np.isin(self.indices, self.active), 3.4),
        ]
        if self.delta is not None:
            palette[-1:] = [(QColor("#f58978"), self.delta[self.indices] < 0, 4.0),
                            (QColor(BRIAN), self.delta[self.indices] > 0, 4.0)]
        for color, mask, radius in palette:
            p.setPen(QPen(color, radius, Qt.SolidLine, Qt.RoundCap))
            p.drawPoints(QPolygonF([QPointF(px, py) for px, py in points[mask, :2]]))
        for color, contacts in zip(("#ffd17f", "#b9a8ff"), self.selected_contacts):
            p.setPen(QPen(QColor(color), 3.2, Qt.SolidLine, Qt.RoundCap))
            p.drawPoints(QPolygonF([QPointF(x, y) for x, y, _ in self._project(contacts)]))
        if self.selected is not None and np.isfinite(self.graph.xyz[self.selected]).all():
            start = self._project(self.graph.xyz[[self.selected]])[0]
            origin = QPointF(*start[:2])
            p.setPen(QPen(QColor("#4477a5"), 1))
            links = self.graph.xyz[self.links] if self.atlas_geometry is None and len(self.links) else np.empty((0, 3))
            for px, py, _ in self._project(links[np.isfinite(links).all(axis=1)]):
                p.drawLine(origin, QPointF(px, py))
            p.setPen(QPen(INK, 2))
            p.drawEllipse(origin, 6, 6)
        if self.compact:
            self._draw_pulses(p)
            return
        p.setPen(QColor("#d8c8d8"))
        contacts = sum(map(len, self.contact_layers))
        caption = f"{contacts:,} contactos · {len(points):,} neuronas · µm" if self.atlas_geometry is not None else f"Puntos FAFB · {len(points):,} / {self.available:,} · µm"
        p.drawText(QPointF(12, 22), caption)
        if self.delta is not None:
            p.drawText(QPointF(12, 42), "Cambio de spikes · rojo disminuye · lima aumenta")
        p.drawText(QPointF(12, self.height() - 12), "Arrastra para rotar · botón derecho para mover · rueda para zoom")

    def flash(self, activity, contact_activity=()):
        # The two bands are the neurons that fired in the decision and reward windows.
        self.pulses = tuple(indices[::max(1, math.ceil(len(indices) / 70))] for indices in activity)
        self.contact_pulses = contact_activity
        if any(len(indices) for indices in self.pulses):
            self.pulse_started = time.monotonic()
            self.pulse_timer.start()
        else:
            self.pulse_timer.stop()
        self.update()

    def _animate(self):
        if time.monotonic() - self.pulse_started >= .24:
            self.pulse_timer.stop()
            self.pulses = self.contact_pulses = ()
        self.update()

    def _draw_pulses(self, painter):
        age = time.monotonic() - self.pulse_started
        for phase, indices in enumerate(self.pulses):
            elapsed = age - phase * .045
            if not 0 <= elapsed < .195:
                continue
            progress = elapsed / .195
            points = self.graph.xyz[indices]
            points = self._project(points[np.isfinite(points).all(axis=1)])
            color = QColor("#c5fdab" if phase == 0 else "#f5a7ff")
            color.setAlphaF((1 - progress)**1.25)
            painter.setPen(QPen(color, 1.15))
            painter.setBrush(Qt.NoBrush)
            radius = 2 + 7 * progress
            for x, y, _ in points:
                painter.drawEllipse(QPointF(x, y), radius, radius)
            if phase < len(self.contact_pulses):
                contacts = self._project(self.contact_pulses[phase])
                painter.setPen(QPen(color, 1.9, Qt.SolidLine, Qt.RoundCap))
                painter.drawPoints(QPolygonF([QPointF(x, y) for x, y, _ in contacts]))
            painter.setPen(QPen(color, 2, Qt.SolidLine, Qt.RoundCap))
            painter.drawPoints(QPolygonF([QPointF(x, y) for x, y, _ in points]))

    def mousePressEvent(self, e):
        self.mouse_start = self.mouse_last = e.position()

    def mouseMoveEvent(self, e):
        delta = e.position() - self.mouse_last
        if e.buttons() & Qt.RightButton:
            self.pan += delta
        elif e.buttons() & Qt.LeftButton:
            self.rotation[0] += delta.y() * .009
            self.rotation[1] += delta.x() * .009
        self.mouse_last = e.position()
        self.update()

    def mouseReleaseEvent(self, e):
        if (e.position() - self.mouse_start).manhattanLength() >= 5 or self.projections is None or not len(self.indices):
            return
        point = np.array([e.position().x(), e.position().y()])
        distances = np.linalg.norm(self.projections[:, :2] - point, axis=1)
        j = int(np.argmin(distances))
        if distances[j] < 12:
            self.neuron_selected.emit(str(self.graph.ids[self.indices[j]]))

    def wheelEvent(self, e):
        self.zoom = min(12, max(.2, self.zoom * math.exp(e.angleDelta().y() / 850)))
        self.update()
