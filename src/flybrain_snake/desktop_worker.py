"""Run scientific work on a background Qt thread."""
from dataclasses import asdict, replace
from datetime import datetime
import queue
from secrets import randbits
import time
import traceback

from PySide6.QtCore import QThread, Signal

from .__main__ import dump, gate, validate, validation_path
from .anatomy import ContactAtlas
from .arena import Duel
from .circuits import Circuit
from .connectome import Graph
from .experiments import evaluate, trainer_for
from .game import Direction, Reward
from .simulator import Stimulus
from .storage import Journal, atomic_bytes, load_active_memory, restore, save_active_memory
from .training import episode_seed
from .wall_study import WallSelection

import numpy as np


class Worker(QThread):
    loaded = Signal(object, object)
    atlas_ready = Signal(object, object)
    brain_contacts = Signal(str, object)
    status = Signal(str)
    frame = Signal(object)
    memory_reset = Signal()
    training_profile = Signal(object)
    episode_done = Signal(object)
    episode_started = Signal(object)
    history = Signal(object)
    message = Signal(str)
    checkpoints = Signal(object)
    comparison = Signal(object)
    duel_frame = Signal(int, object)
    duel_player = Signal(int, object)
    duel_failed = Signal(int, str)

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.commands = queue.Queue()
        self.trainer = self.journal = None
        self.duel = None
        self.duel_id = None
        self.duel_primed = set()
        self.atlas = None
        self.memory_path = self.data / "sessions" / "memoria-activa.npz"
        self.running = False
        self.closing = False
        self.manual = False
        self.delay = .15
        self.limit = 0
        self.last_frame = 0.0
        self.frame_interval = .12
        self.rate = 100
        self.extra = None
        self.extra_reward = None
        self.wall_selection = None

    def send(self, command, *args):
        self.commands.put((command, args))

    def run(self):
        try:
            graph = Graph.load(self.data / "processed")
            self.graph, self.circuit = graph, Circuit.from_graph(graph)
            self.loaded.emit(graph, self.circuit)
            self.status.emit(f"{graph.n:,}".replace(",", ".") + " neuronas · listo")
        except Exception as error:
            self.message.emit(f"No se pudo cargar el conectoma. Ejecuta prepare.\n{error}")
            return
        try:
            self.active_scales, self.active_config = load_active_memory(self.memory_path, self.graph, self.circuit)
        except (OSError, ValueError, KeyError) as error:
            self.active_scales = np.ones(len(self.circuit.edges), dtype=np.float32)
            self.active_config = None
            self.message.emit(f"No se pudo cargar la memoria activa: {error}")
        self.saved_scales = self.active_scales.copy()
        self.saved_config = self.active_config
        if self.active_config is not None and self.active_config.wall_preview_cells != 1:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = self.memory_path.with_name(f"memoria-antes-vision-limites-{stamp}.npz")
            atomic_bytes(backup, self.memory_path.read_bytes())
            self.active_config = replace(self.active_config, wall_preview_cells=1)
            self._save_memory()
        if self.active_config is not None and (self.active_config.wall_memory_hz
                or (self.active_config.practice in ("walls", "edge_reset") and self.active_config.learning.post_required)):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            backup = self.memory_path.with_name(f"memoria-antes-retirar-senal-pared-{stamp}.npz")
            atomic_bytes(backup, self.memory_path.read_bytes())
            self.active_config = replace(self.active_config, wall_memory_hz=0.0,
                                         learning=replace(self.active_config.learning, post_required=False)
                                         if self.active_config.practice in ("walls", "edge_reset") else self.active_config.learning)
            self._save_memory()
        if (self.active_config is not None and self.active_config.stage == 4
                and self.active_config.practice == "snake" and self.active_config.max_steps):
            self.active_config = replace(self.active_config, max_steps=0)
            self._save_memory()
        self.training_profile.emit(self.active_config)
        try:
            atlas = ContactAtlas(graph.data_path or self.data, graph)
            overview = atlas.query()
            _ = atlas.outgoing_preview
        except (OSError, ValueError) as error:
            self.atlas_ready.emit(None, str(error))
        else:
            self.atlas = atlas
            self.atlas_ready.emit(atlas.names, overview)
        while not self.closing:
            try:
                try:
                    delay = 0 if self.running and self.trainer.config.compact_trace else self.delay
                    command, args = self.commands.get(timeout=delay if self.running else .15)
                except queue.Empty:
                    if self.running:
                        self._step()
                else:
                    self._command(command, args)
            except Exception as error:
                self.running = False
                self.status.emit("Ejecución detenida")
                self.message.emit(f"{type(error).__name__}: {error}")
                path = self.data / "runs" / "desktop-error.log"
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    traceback.print_exc(file=handle)
        try:
            self._save_memory()
        finally:
            self._close_session()

    def _save_memory(self, *, force=False):
        if (force or self.saved_config != self.active_config
                or not np.array_equal(self.saved_scales, self.active_scales)):
            save_active_memory(self.memory_path, self.graph, self.circuit, self.active_scales, self.active_config)
            self.saved_scales[:] = self.active_scales
            self.saved_config = self.active_config

    def _wall_check(self, *, initialize=False):
        t, j = self.trainer, self.journal
        if t is None or j is None or t.config.practice != "walls":
            return
        self.status.emit("Validando evasión · pesos congelados…")
        if self.wall_selection is None or initialize:
            if t.sim.scales is self.active_scales:
                t.sim.scales = self.active_scales.copy()
            try:
                self.wall_selection = WallSelection(self.graph, self.circuit, t.config, self.active_scales,
                                                    stop=lambda: not self.commands.empty())
            except InterruptedError:
                self.status.emit("Validación pendiente")
                return False
            self.wall_selection.episode = t.episode
            self._save_wall_best()
            return True
        j.save(t, f"paredes-candidato-{t.episode}")
        try:
            result = self.wall_selection.consider(t.sim.scales, t.episode)
        except InterruptedError:
            self.status.emit("Validación pendiente · memoria activa conservada")
            return False
        self.active_scales[:] = self.wall_selection.scales
        j.append("events", {"kind": "wall_validation", "episode": t.episode, **result})
        t.sim.reset()
        j.save(t, "paredes-mejor")
        self._save_memory()
        self.checkpoints.emit(j.labels())
        self._snapshot(force=True)
        self.status.emit(f"Evasión validada · {result['selected_success']:.0%} · "
                         + result["reason"])
        return True

    def _save_wall_best(self):
        t = self.trainer
        candidate = t.sim.scales
        try:
            t.sim.scales = self.active_scales
            self.journal.save(t, "paredes-mejor")
        finally:
            t.sim.scales = candidate

    def _suspend_wall_candidate(self):
        """Keep the pending model on disk and export only accepted memory."""
        t, j = self.trainer, self.journal
        if t is None or j is None or t.config.practice != "walls":
            return
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        j.save(t, f"paredes-pendiente-{stamp}")
        j.append("events", {"kind": "wall_candidate_held", "episode": t.episode,
                            "pending_changes": int(np.count_nonzero(t.sim.scales != self.active_scales))})
        t.sim.scales[:] = self.active_scales
        t.game, t.stats, t.last_step = None, {}, {}
        t.reset_activity()
        self.wall_selection = None

    def _close_session(self):
        if self.journal:
            try:
                self.journal.save(self.trainer)
            finally:
                self.journal.close()
                self.journal = None
        self.wall_selection = None
        self.trainer = None

    def _set_learning_switches(self, plasticity, dopamine):
        t = self.trainer
        cfg = t.config if t is not None else self.active_config
        if cfg is None and self.duel is not None:
            cfg = self.duel.brian.config
        if self.duel is not None:
            self.duel.brian.config = replace(self.duel.brian.config, plasticity=plasticity, dopamine=dopamine)
        if cfg is None or (cfg.plasticity, cfg.dopamine) == (plasticity, dopamine):
            return
        cfg = replace(cfg, plasticity=plasticity, dopamine=dopamine)
        if t is not None:
            t.config = cfg
        self.active_config = cfg
        self.wall_selection = None
        if self.journal is not None:
            self.journal.append("events", {"kind": "learning_switches", "plasticity": plasticity,
                                           "dopamine": dopamine, "episode": t.episode})
            self.journal.save(t)
        self._save_memory()
        self.training_profile.emit(cfg)

    def _set_practice(self, mode, max_steps):
        t, journal = self.trainer, self.journal
        if t is None or journal is None:
            return
        mode_changed = mode != t.config.practice
        if (not mode_changed and max_steps == t.config.max_steps
                and t.config.wall_preview_cells == 1
                and not (mode in ("walls", "edge_reset") and t.config.learning.post_required)
                and (t.game is None or t.game.max_steps == max_steps)):
            return
        if mode_changed and t.config.practice == "walls":
            self._suspend_wall_candidate()
        if mode_changed:
            self.wall_selection = None
        cfg = replace(t.config, practice=mode, max_steps=max_steps,
                      stage=4 if mode != "snake" else t.config.stage,
                      learning=replace(t.config.learning, post_required=False)
                               if mode != "snake" else t.config.learning,
                      wall_memory_hz=0.0, wall_preview_cells=1)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        journal.save(t, f"antes-cambio-practica-{stamp}")
        journal.append("events", {"kind": "practice_change" if mode_changed else "step_limit_change",
                                   "before": t.config.practice, "after": mode,
                                   "previous_max_steps": t.config.max_steps, "max_steps": max_steps,
                                   "interrupted_steps": t.game.steps if t.game else 0})
        t.config = cfg
        t.sim.learning = cfg.learning
        if mode_changed:
            t.sim.scales = self.active_scales.copy() if mode == "walls" else self.active_scales
            t.game = None
            t.stats = {}
            t.last_step = {}
            t.reset_activity()
            self.extra = self.extra_reward = None
        elif t.game is not None:
            t.game.max_steps = max_steps
            if t.game.end == "límite" and (max_steps == 0 or t.game.steps < max_steps):
                t.game.end = ""
        self.active_config = cfg
        journal.save(t)
        self._save_memory()
        self.checkpoints.emit(journal.labels())
        self.training_profile.emit(cfg)
        title = {"routes": "Rectas y giros", "walls": "Evitar paredes", "snake": "Partidas completas",
                 "edge_reset": "Perímetro estricto"}[mode]
        self.status.emit(title + (" · en marcha" if self.running else " · en pausa"))

    def _command(self, cmd, args):
        if cmd == "exit":
            self.closing = True
            self.running = False
        elif cmd == "brain_contacts":
            if self.atlas is not None:
                root = args[0]
                self.brain_contacts.emit(root, self.atlas.query(root=int(root), limit=500))
        elif cmd.startswith("duel_"):
            self._duel_command(cmd, args)
        elif cmd == "reset_memory":
            self.running = self.manual = False
            self.duel = self.duel_id = None
            self.extra = self.extra_reward = None
            self._close_session()
            self.trainer = None
            self.active_scales.fill(1)
            self.active_config = None
            self._save_memory(force=True)
            self.history.emit([])
            self.checkpoints.emit([])
            self.training_profile.emit(None)
            self.memory_reset.emit()
            self.status.emit("Aprendizaje reiniciado")
        elif cmd == "practice":
            self._set_practice(*args)
        elif cmd == "step_limit" and self.trainer is not None:
            self._set_practice(self.trainer.config.practice, args[0])
        elif cmd == "learning_switches":
            self._set_learning_switches(*args)
        elif cmd == "new":
            cfg, count, keep = args
            cfg = replace(cfg, wall_memory_hz=0.0, wall_preview_cells=1,
                          learning=replace(cfg.learning, post_required=False)
                          if cfg.practice in ("walls", "edge_reset") else cfg.learning)
            self.running = False
            self.extra = self.extra_reward = None
            if cfg.stage == 0:
                self.status.emit("Ejecutando validación del cerebro completo…")
                report = validate(self.graph, self.circuit, cfg, 8)
                dump(validation_path(self.data, cfg), report)
                self.status.emit("Etapa 0 superada" if report["passed"] else "Etapa 0 no superada")
                self.message.emit(f"Spikes no estimulados: {report['propagation']['sugar_non_target_spikes']:,}\n"
                                  f"Conexiones con cambios por asociación: {report['association']['conditions']['paired']['changed']:,}")
                return
            self.status.emit("Verificando el protocolo…")
            gate(self.graph, self.circuit, cfg, validation_path(self.data, cfg))
            next_trainer = trainer_for(self.graph, self.circuit, cfg)
            self._close_session()
            if not keep:
                self.active_scales.fill(1)
            next_trainer.sim.scales = self.active_scales.copy() if cfg.practice == "walls" else self.active_scales
            self.trainer = next_trainer
            session = self.data / "sessions" / datetime.now().strftime("Brian_%Y%m%d_%H%M%S_%f.sqlite")
            self.journal = Journal(session)
            self.journal.append("events", {"kind": "start", "config": asdict(cfg),
                                           "memory_carried": bool(np.any(self.active_scales != 1))})
            self.journal.save(self.trainer, f"etapa{cfg.stage}-inicio")
            self.active_config = cfg
            self._save_memory()
            self.training_profile.emit(cfg)
            self.limit = count
            self.history.emit([])
            self.checkpoints.emit(self.journal.labels())
            self.running = True
            self.status.emit(f"En marcha · {session.name}")
        elif cmd == "load":
            self.running = False
            self._close_session()
            journal = Journal(args[0], create=False)
            try:
                loaded = restore(journal.get(), self.graph, self.circuit)
                accepted = (restore(journal.get("paredes-mejor"), self.graph, self.circuit)
                            if loaded.config.practice == "walls" and "paredes-mejor" in journal.labels()
                            else loaded)
            except Exception:
                journal.close()
                raise
            if (loaded.config.stage == 4 and loaded.config.practice == "snake"
                    and (loaded.config.max_steps or (loaded.game is not None and loaded.game.max_steps))):
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                journal.save(loaded, f"antes-quitar-limite-{stamp}")
                journal.append("events", {"kind": "full_game_without_step_limit",
                                           "previous_max_steps": loaded.config.max_steps})
                loaded.config = replace(loaded.config, max_steps=0)
                if loaded.game is not None:
                    loaded.game.max_steps = 0
                    if loaded.game.end == "límite":
                        loaded.game.end = ""
            if loaded.config.wall_memory_hz:
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                journal.save(loaded, f"antes-retirar-senal-pared-{stamp}")
                journal.append("events", {"kind": "remove_auxiliary_wall_cue",
                                           "previous_hz": loaded.config.wall_memory_hz})
                loaded.config = replace(loaded.config, wall_memory_hz=0.0)
                loaded.sim.reset()
            if loaded.config.wall_preview_cells != 1:
                journal.save(loaded, "antes-vision-limites")
                journal.append("events", {"kind": "boundary_vision", "preview_cells": 1})
                loaded.config = replace(loaded.config, wall_preview_cells=1)
                loaded.sim.reset()
            if loaded.config.practice in ("walls", "edge_reset") and loaded.config.learning.post_required:
                journal.save(loaded, "antes-corregir-coincidencia-paredes")
                journal.append("events", {"kind": "wall_learning_rule_changed", "post_required": False})
                loaded.config = replace(loaded.config, learning=replace(loaded.config.learning, post_required=False))
                loaded.sim.learning = loaded.config.learning
            self.active_scales[:] = accepted.sim.scales
            if loaded.config.practice == "walls":
                loaded.sim.scales = loaded.sim.scales.copy()
                loaded.game = None
                loaded.last_step = {}
                loaded.stats = {}
                loaded.sim.reset()
            else:
                loaded.sim.scales = self.active_scales
            self.trainer, self.journal = loaded, journal
            self.active_config = loaded.config
            self._save_memory()
            self.training_profile.emit(loaded.config)
            self.extra = self.extra_reward = None
            self.limit = self.trainer.episode + args[1]
            self.history.emit(list(self.journal.rows("episodes")))
            self.checkpoints.emit(self.journal.labels())
            self._snapshot(force=True)
            self.status.emit("Sesión cargada · en pausa")
        elif cmd == "pace":
            self.delay = args[0]
        elif cmd == "refresh_hz":
            self.frame_interval = 1 / args[0]
        elif cmd == "hz":
            self.rate = args[0]
        elif cmd == "manual":
            self.manual = args[0]
            if self.manual:
                self.running = False
                self.status.emit("Control manual · flechas del teclado")
        elif cmd == "pause":
            self.running = False
            self.status.emit("En pausa")
        elif cmd == "resume":
            count = args[0]
            cfg = args[1] if len(args) > 1 else None
            if self.trainer is None or self.journal is None:
                if cfg is not None:
                    self._command("new", (cfg, count, True))
                return
            if cfg is not None:
                self._set_practice(cfg.practice, cfg.max_steps)
                self._set_learning_switches(cfg.plasticity, cfg.dopamine)
            if self.trainer.episode >= self.limit:
                self.limit = self.trainer.episode + count
            self.manual = False
            self.running = True
            self.status.emit("Brian en marcha")
        elif cmd == "stop":
            self.running = False
            self._save_memory()
            if self.journal:
                self.journal.save(self.trainer)
            self.status.emit("Detenido · progreso guardado")
        elif cmd == "save" and self.journal:
            self._save_memory()
            self.journal.save(self.trainer)
            self.checkpoints.emit(self.journal.labels())
            self.status.emit(f"Guardado · {self.journal.path.name}")
        elif cmd == "export" and self.journal:
            self.journal.save(self.trainer)
            self.journal.export(args[0])
            from .storage import write_csv
            write_csv(args[0] / "memory.csv", self.trainer.sim.memory_rows())
            self.status.emit(f"Exportado · {args[0]}")
        elif cmd == "compare" and self.journal:
            label = args[0]
            self.running = False
            if label == "actual":
                self.journal.save(self.trainer)
                label = "latest"
            trained = restore(self.journal.get(label), self.graph, self.circuit)
            sim = trained.sim
            sim.silence.fill(False)
            cue = Stimulus("conditioning_A", self.circuit.groups["conditioning_A"], trained.config.sensory_hz)
            learned = sim.scales.copy()
            output = []
            for scales in (np.ones_like(learned), learned):
                sim.reset()
                sim.rng = np.random.default_rng(91)
                sim.scales[:] = scales
                output.append(sim.advance(trained.config.decision_ms, (cue,)))
            self.comparison.emit({"label": label, "baseline": output[0], "trained": output[1],
                                  "memory": sim.memory()})
            self.status.emit("Comparación neuronal calculada · en pausa")
        elif cmd == "step" and self.trainer:
            self.running = False
            self._step(force=True)
        elif cmd == "direction" and self.trainer and self.manual:
            if self.trainer.config.stage >= 2:
                self._step(Direction(args[0]), force=True)
        elif cmd == "intervention" and self.trainer:
            root, kind = args
            self.running = False
            if kind == "positive":
                self.extra_reward = Reward(apple=self.trainer.config.rewards.apple)
            elif kind == "negative":
                self.extra_reward = Reward(collision=self.trainer.config.rewards.collision)
            else:
                index = self.graph.index(int(root))
                if kind == "pulse":
                    self.extra = Stimulus("pulso manual", np.array([index], np.int32), self.rate)
                else:
                    self.trainer.sim.set_silence(np.array([index]), kind == "silence")
            self.journal.append("events", {"kind": kind, "root": root, "manual": True,
                                           "episode": self.trainer.episode})
            self.status.emit("Intervención registrada · avanza un paso")

    def _duel_command(self, cmd, args):
        round_id = args[0]
        try:
            if cmd == "duel_start":
                if self.atlas is None:
                    raise ValueError("Prepara el atlas FlyWire antes de jugar")
                _, cfg, size, apple_goal, rewards = args
                self.running = False
                self.extra = self.extra_reward = None
                self._suspend_wall_candidate()
                selected = self.active_config
                if selected is None:
                    selected = replace(cfg, rewards=rewards, synaptic_signal="contact_probabilities")
                config = replace(selected, stage=4, size=size, max_steps=0, practice="snake",
                                 wall_memory_hz=0.0, wall_preview_cells=1)
                opponent = trainer_for(self.graph, self.circuit, config)
                opponent.sim.scales = self.active_scales
                if config.synaptic_signal not in self.duel_primed:
                    # Load the compiled kernel before the player starts the clock.
                    opponent.sim.advance(config.lif.dt)
                    self.duel_primed.add(config.synaptic_signal)
                self.duel = Duel(opponent, randbits(64), apple_goal)
                self.duel_id = round_id
            elif round_id != self.duel_id:
                return
            elif cmd == "duel_cancel":
                completed = self.duel
                self.duel = None
                self.duel_id = None
                if self.trainer is not None:
                    t = self.trainer
                    if t.config.practice == "walls":
                        t.sim.scales[:] = self.active_scales
                        self.wall_selection = None
                    t.reset_activity()
                    t.interventions |= t.game is not None
                    if self.journal is not None:
                        self.journal.append("events", {"kind": "arena_return", "episode": t.episode,
                                                       "steps": t.game.steps if t.game else 0,
                                                       "turns": completed.turn if completed else 0,
                                                       "neural_state_reset": True})
                self._save_memory()
                if self.journal is not None:
                    self.journal.save(self.trainer)
                return
            else:
                self.duel.advance(Direction(args[1]),
                                  lambda board: self.duel_player.emit(round_id, board))
            snapshot = self.duel.snapshot()
            snapshot["contact_activity"] = tuple(self.atlas.active_contacts(nodes) for nodes in snapshot["activity"])
            self.duel_frame.emit(round_id, snapshot)
            if snapshot["winner"] or (snapshot["turn"] and snapshot["turn"] % 20 == 0):
                self._save_memory()
        except Exception as error:
            self.duel = None
            self.duel_id = None
            self.duel_failed.emit(round_id, str(error))

    def _step(self, direction=None, force=False):
        try:
            self._advance(direction, force)
        except Exception:
            if self.journal is not None:
                self.journal.close()
                self.journal = self.trainer = None
            raise

    def _advance(self, direction, force):
        t, j = self.trainer, self.journal
        if t is None or j is None:
            return
        if t.config.practice == "walls" and self.wall_selection is None:
            if not self._wall_check(initialize=True):
                return
        if t.config.stage == 1:
            row = t.association_episode()
        else:
            if t.game is None:
                t.begin()
                if t.config.practice == "walls" and not t.config.compact_trace:
                    self.episode_started.emit({"board": t.game.board_state(), "episode": t.episode,
                                               "memory": t.sim.memory()})
            extra = (self.extra,) if self.extra is not None else ()
            t.interventions |= bool(t.sim.silence.any())
            detailed = not t.config.compact_trace or force or bool(extra or self.extra_reward)
            step = t.step(direction, extra=extra, manual_reward=self.extra_reward, detail=detailed,
                          force_frozen=bool(t.sim.silence.any()),
                          learn_from_manual=bool(extra or self.extra_reward))
            self.extra = self.extra_reward = None
            if detailed:
                j.append("steps", {"split": "interactive" if t.interventions else "training", **step})
            self._snapshot(force=force or (bool(t.game.end) and (not t.config.compact_trace
                                                                or t.episode + 1 >= self.limit)))
            if not t.game.end:
                return
            row = t.finish()
        j.append("episodes", {"split": "interactive" if row["manual"] else "training", **row})
        self.episode_done.emit(row)
        if (t.config.practice == "walls" and not row["manual"]
                and (t.episode - self.wall_selection.episode >= 24 or t.episode >= self.limit)):
            self._wall_check()
        if t.episode in (100, 1000) or t.episode >= self.limit:
            self._save_memory()
            j.save(t, f"etapa{t.config.stage}-episodio{t.episode}")
            self.checkpoints.emit(j.labels())
        elif t.episode % t.config.save_every == 0:
            self._save_memory()
            j.save(t)
        if t.config.stage >= 2 and t.config.eval_every and t.episode % t.config.eval_every == 0:
            results = evaluate(t, [episode_seed(t.config.seed, i, 3) for i in range(5)])
            j.append("events", {"kind": "periodic-evaluation", "trained_episodes": t.episode, "results": results})
        if t.episode >= self.limit and self.running:
            self.running = False
            self.status.emit(f"Bloque completado · {t.episode} episodios")

    def _snapshot(self, force=False):
        now = time.monotonic()
        interval = max(1.0, self.frame_interval) if self.trainer.config.compact_trace else self.frame_interval
        if not force and now - self.last_frame < interval:
            return
        self.last_frame = now
        t = self.trainer
        if not t.last_step:
            return
        counts = t.decision_counts + t.outcome_counts
        memory = t.sim.memory_rows(30)
        self.frame.emit({"board": t.game.board_state() if t.game else None, "step": t.last_step,
                         "counts": counts, "memory": memory, "memory_state": t.sim.memory(),
                         "episode_time": t.stats.get("wall_seconds", 0.0)})
