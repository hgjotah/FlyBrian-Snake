"""SQLite journals and atomic checkpoints with compact synaptic deltas."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import csv
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import zipfile

import numpy as np

from . import __version__
from .circuits import Circuit
from .config import LIF, Plasticity, Rewards, RunConfig
from .connectome import Graph
from .game import Reward, Snake
from .simulator import Simulator
from .training import Trainer, episode_seed


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_config(data: dict) -> RunConfig:
    payload = dict(data)
    for key, cls in (("lif", LIF), ("learning", Plasticity), ("rewards", Rewards)):
        if key in payload:
            payload[key] = cls(**payload[key])
    return RunConfig(**payload)


def checkpoint(trainer: Trainer) -> bytes:
    sim = trainer.sim
    modified = np.flatnonzero(sim.scales != 1).astype(np.int32)
    metadata = {
        "format": 2, "software": __version__, "date_utc": timestamp(),
        "connectome": sim.graph.fingerprint, "dataset": sim.graph.manifest["dataset"],
        "gain_fingerprint": sim.gain_fingerprint,
        "config": asdict(trainer.config), "episode": trainer.episode,
        "game": None if trainer.game is None else trainer.game.state(),
        "stats": trainer.stats, "interventions": trainer.interventions,
        "last_step": trainer.last_step, "tick": sim.tick, "lifetime_ticks": sim.lifetime_ticks,
        "rng": sim.rng.bit_generator.state,
        "boundary_rng": sim.boundary_rng.bit_generator.state,
        "delayed": [asdict(reward) for reward in trainer.delayed],
    }
    pending = np.concatenate([row[:count] for row, count in zip(sim.ring, sim.ring_counts, strict=True)])
    buf = io.BytesIO()
    np.savez_compressed(buf, metadata=json.dumps(metadata, ensure_ascii=False), modified=modified,
                        edges=sim.circuit.edges[modified], scales=sim.scales[modified],
                        voltage=sim.voltage, drive=sim.drive, ready=sim.ready, last=sim.last,
                        silence=sim.silence, ring_counts=sim.ring_counts, pending=pending,
                        updates=sim.group_updates, last_update=sim.last_update[modified],
                        update_age=sim.update_age[modified], decision_counts=trainer.decision_counts,
                        outcome_counts=trainer.outcome_counts)
    return buf.getvalue()


def restore(data: bytes, graph: Graph, circuit: Circuit) -> Trainer:
    with np.load(io.BytesIO(data), allow_pickle=False) as z:
        meta = json.loads(z["metadata"].item())
        if meta["format"] != 2 or meta["connectome"] != graph.fingerprint:
            raise ValueError("Checkpoint incompatible con este conectoma o versión de archivo")
        stored_config = meta["config"]
        # Keep historical protocols reproducible; the desktop upgrades explicitly.
        stored_config.setdefault("wall_preview_cells", 0)
        if "profile" not in stored_config.get("rewards", {}):
            stored_config.setdefault("rewards", {})["profile"] = "legacy"
        config = load_config(stored_config)
        sim = Simulator(graph, circuit, config.lif, config.learning, config.seed,
                        synaptic_signal=config.synaptic_signal)
        if sim.gain_fingerprint != meta.get("gain_fingerprint"):
            raise ValueError("Las ganancias sinápticas del checkpoint no coinciden")
        modified = z["modified"]
        if not np.array_equal(circuit.edges[modified], z["edges"]):
            raise ValueError("El mapa de sinapsis plásticas del checkpoint no coincide")
        sim.scales[modified] = z["scales"]
        sim.last_update[modified] = z["last_update"]
        sim.update_age[modified] = z["update_age"]
        for name in ("voltage", "drive", "ready", "last", "silence", "ring_counts"):
            getattr(sim, name)[:] = z[name]
        sim.group_updates[:] = z["updates"]
        cursor = 0
        pending = z["pending"]
        for row, count in zip(sim.ring, sim.ring_counts, strict=True):
            row[:count] = pending[cursor:cursor + count]
            cursor += count
        sim.tick, sim.lifetime_ticks = meta["tick"], meta["lifetime_ticks"]
        sim.rng.bit_generator.state = meta["rng"]
        if "boundary_rng" in meta:
            sim.boundary_rng.bit_generator.state = meta["boundary_rng"]
        elif meta["game"] is not None:
            sim.boundary_rng = np.random.default_rng(episode_seed(meta["game"]["seed"], 0, 13))
        trainer = Trainer(sim, config)
        trainer.episode = meta["episode"]
        trainer.stats = meta["stats"]
        trainer.interventions = meta["interventions"]
        trainer.last_step = meta["last_step"]
        trainer.delayed = [Reward(**reward) for reward in meta.get("delayed", [])]
        trainer.decision_counts = z["decision_counts"]
        trainer.outcome_counts = z["outcome_counts"]
        if meta["game"] is not None:
            trainer.game = Snake.restore(meta["game"], config.rewards)
            if trainer.game.restrict_food_to_interior():
                trainer.reset_activity()
        return trainer


class Journal:
    def __init__(self, path: Path, *, create: bool = True, read_only: bool = False):
        if (not create or read_only) and not path.is_file():
            raise FileNotFoundError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = None
        if read_only:
            self.connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
            return
        self.lock = self._claim_writer(path)
        try:
            self.connection = sqlite3.connect(path)
            self.connection.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS steps (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS episodes (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS checkpoints (label TEXT PRIMARY KEY, data BLOB NOT NULL, updated TEXT NOT NULL);
            """)
        except Exception:
            if hasattr(self, "connection"):
                self.connection.close()
            self.lock.close()
            raise

    @staticmethod
    def _claim_writer(path: Path):
        lock = path.with_suffix(path.suffix + ".lock").open("a+b")
        try:
            lock.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            lock.close()
            raise ValueError(f"La sesión ya tiene un escritor activo: {path}") from error
        return lock

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def append(self, table: str, payload: dict):
        if table not in ("steps", "episodes", "events"):
            raise ValueError(table)
        self.connection.execute(f"INSERT INTO {table} (payload) VALUES (?)",
                                (json.dumps(payload, ensure_ascii=False),))

    def rows(self, table: str):
        if table not in ("steps", "episodes", "events"):
            raise ValueError(table)
        return (json.loads(row[0]) for row in self.connection.execute(f"SELECT payload FROM {table} ORDER BY id"))

    def save(self, trainer: Trainer, label: str = "latest"):
        data = checkpoint(trainer)
        self.connection.execute("INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)",
                                ("latest", data, timestamp()))
        if label != "latest":
            self.connection.execute("INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?)",
                                    (label, data, timestamp()))
        self.connection.commit()

    def get(self, label: str = "latest") -> bytes:
        result = self.connection.execute("SELECT data FROM checkpoints WHERE label=?", (label,)).fetchone()
        if result is None:
            raise ValueError(f"No existe checkpoint {label}")
        return result[0]

    def labels(self):
        return [r[0] for r in self.connection.execute("SELECT label FROM checkpoints ORDER BY updated")]

    def close(self):
        try:
            self.connection.close()
        finally:
            if self.lock is not None:
                self.lock.close()

    def export(self, destination: Path):
        destination.mkdir(parents=True, exist_ok=True)
        for table in ("steps", "episodes", "events"):
            with (destination / (table + ".jsonl")).open("w", encoding="utf-8") as handle:
                for row in self.rows(table):
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        rows = list(self.rows("episodes"))
        if rows:
            write_csv(destination / "episodes.csv", rows)
        with zipfile.ZipFile(destination / "checkpoints.zip", "w", zipfile.ZIP_STORED) as archive:
            for i, label in enumerate(self.labels()):
                archive.writestr(f"{i:03d}_{label.replace('/', '_')}.npz", self.get(label))


def write_csv(path: Path, rows: list[dict]):
    fields = sorted(set().union(*(row.keys() for row in rows)))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def atomic_bytes(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def load_active_memory(path: Path, graph: Graph, circuit: Circuit) -> tuple[np.ndarray, RunConfig | None]:
    if not path.is_file():
        return np.ones(len(circuit.edges), dtype=np.float32), None
    with np.load(path, allow_pickle=False) as data:
        scales = data["scales"]
        if (data["graph"].item() != graph.fingerprint or not np.array_equal(data["edges"], circuit.edges)
                or scales.shape != circuit.edges.shape or scales.dtype != np.float32
                or not np.isfinite(scales).all() or np.any((scales < 0) | (scales > 1))):
            raise ValueError("Memoria activa incompatible con el conectoma")
        config = data["config"].item() if "config" in data else ""
        settings = json.loads(config) if config else None
        if settings is not None:
            settings.setdefault("wall_preview_cells", 0)
        return scales, load_config(settings) if settings is not None else None


def save_active_memory(path: Path, graph: Graph, circuit: Circuit, scales: np.ndarray, config: RunConfig | None):
    buffer = io.BytesIO()
    settings = json.dumps(asdict(config)) if config is not None else ""
    np.savez_compressed(buffer, graph=graph.fingerprint, edges=circuit.edges, scales=scales, config=settings)
    atomic_bytes(path, buffer.getvalue())
