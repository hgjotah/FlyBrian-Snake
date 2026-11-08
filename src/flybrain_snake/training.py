"""Agent loop, curriculum and telemetry shared by the CLI and desktop."""
from __future__ import annotations

from dataclasses import asdict, replace
import time

import numpy as np
import psutil

from .bridge import motor, reinforcement, sensory
from .config import RunConfig
from .game import Direction, Reward, Snake, VECTORS
from .simulator import Simulator, Stimulus
from .validation import conditioning_trial


def episode_seed(seed: int, index: int, domain: int = 0) -> int:
    return int(np.random.SeedSequence([seed, index, domain]).generate_state(1, dtype=np.uint64)[0])


class Trainer:
    def __init__(self, simulator: Simulator, config: RunConfig):
        self.sim = simulator
        self.config = config
        self.episode = 0
        self.game: Snake | None = None
        self.interventions = False
        self.decision_counts = np.zeros(simulator.graph.n, np.int32)
        self.outcome_counts = self.decision_counts.copy()
        self.last_step: dict = {}
        self.stats: dict = {}
        self.delayed: list[Reward] = []
        self.process = psutil.Process()
        self.process.cpu_percent()

    def reset_activity(self):
        """Clear short-lived signals while retaining learned weights and counters."""
        self.sim.reset()
        self.decision_counts.fill(0)
        self.outcome_counts.fill(0)
        self.delayed.clear()
        self.last_step = {}

    def begin(self, seed: int | None = None):
        cfg = self.config
        seed = episode_seed(cfg.seed, self.episode) if seed is None else seed
        self.reset_activity()
        self.sim.rng = np.random.default_rng(episode_seed(seed, 0, 1))
        self.sim.boundary_rng = np.random.default_rng(episode_seed(seed, 0, 13))
        self.game = Snake(cfg.size, seed, cfg.stage, cfg.max_steps, cfg.rewards, cfg.food_partition,
                          allow_edge_food=False)
        if cfg.practice == "routes":
            self._route_task(seed)
        self.interventions = False
        self.stats = {"seed": seed, "reward": 0.0, "food_distance": 0.0, "wall_distance": 0.0,
                      "edge_apples": 0, "motor_turns": 0, "motor_left_spikes": 0,
                      "motor_right_spikes": 0, "dan_spikes": 0, "kc_spikes": 0, "mbon_spikes": 0,
                      "neural_ms": 0.0, "wall_seconds": 0.0, "active_peak": 0,
                      "rss_peak": self.process.memory_info().rss, "plastic_updates": 0}
        if cfg.practice == "routes":
            self.stats["route_optimal_steps"] = self.game.observe().shortest_food_path()
        elif cfg.practice == "walls":
            self._wall_task(seed)

    def _route_task(self, seed):
        game = self.game
        rng = np.random.default_rng(episode_seed(seed, 0, 12))
        task = self.episode % 12
        if task < 8:
            gap = int(rng.integers(1, game.size - 2))
            x = int(rng.integers(1, game.size - gap - 1))
            y = 1 if task < 4 else int(rng.integers(1, game.size - 1))
            head, apple = (x, y), (x + gap, y)
        else:
            y = int(rng.integers(1, game.size - 1))
            target = int(rng.integers(1, game.size - 2))
            if target >= y:
                target += 1
            head, apple = (game.size - 1, y), (game.size - 2, target)
        direction = Direction.RIGHT
        for _ in range(task % 4):
            head = game.size - 1 - head[1], head[0]
            apple = game.size - 1 - apple[1], apple[0]
            direction = direction.turn(1)
        dx, dy = VECTORS[direction]
        game.body = [head, (head[0] - dx, head[1] - dy)]
        game.direction, game.apple = direction, apple

    def _wall_task(self, seed):
        game = self.game
        rng = np.random.default_rng(episode_seed(seed, 0, 14))
        kind = (self.episode // 4) % 6
        turn = self.episode % 4
        direction = Direction.RIGHT
        side = -1 if kind % 2 == 0 else 1
        y = int(rng.integers(2, game.size - 1) if side < 0 else rng.integers(1, game.size - 2))
        head = (game.size - 1, y)
        dx, dy = VECTORS[direction]
        body = [head, (head[0] - dx, head[1] - dy)]
        inset = min(1 if kind < 2 else 2 if kind < 4 else 3, game.size - 3)
        available = y - 1 if side < 0 else game.size - 2 - y
        gap = int(rng.integers(1, min(3, available) + 1))
        apple = (head[0] - inset, y if kind >= 4 else y + side * gap)
        if apple in body:
            apple = (apple[0], y + side * gap)
        for _ in range(turn):
            body = [(game.size - 1 - y, x) for x, y in body]
            apple = game.size - 1 - apple[1], apple[0]
            direction = direction.turn(1)
        game.body, game.apple, game.direction = body, apple, direction
        # Turning is the only legal escape: allow the turn and the inward step.
        # Actual collisions are terminal even during this setup grace period.
        game.edge_exit_moves = 2
        self.stats.update({"wall_protocol": 2, "wall_heading": int(direction),
                           "wall_escaped": False, "wall_case": kind})

    def _wall_task_complete(self, reward):
        if self.game.end not in ("", "límite"):
            return False
        if self.stats.get("wall_protocol") == 2:
            return bool(reward.apple > 0 and self.stats.get("wall_escaped") and reward.wall == 0)
        # Finish a historical in-flight exercise when explicitly resumed by CLI.
        if not self.stats["wall_sides"]:
            return reward.apple > 0
        obs = self.game.observe()
        return (self.game.steps >= 2 and reward.wall == 0
                and obs.edge_distance() >= self.stats["wall_target"]
                and all(obs.wall_distance(Direction(side)) >= self.stats["wall_target"]
                        for side in self.stats["wall_sides"])
                and obs.wall_distance(obs.direction) >= min(2, self.stats["wall_target"]))

    def _stimuli(self, events: tuple[Stimulus, ...], ms: float):
        return [{"name": e.name, "rate_hz": e.rate, "duration_ms": ms, "stream": e.stream,
                 "ids": [str(root) for root in self.sim.graph.ids[e.indices]]} for e in events]

    def step(self, action: Direction | None = None, *, extra=(), manual_reward=None,
             shuffled=False, force_frozen=False, learn_from_manual=False, detail=True):
        if self.game is None or self.game.end:
            raise ValueError("Inicia un episodio antes de simular un movimiento")
        start = time.perf_counter()
        sim, cfg, game = self.sim, self.config, self.game
        updates = int(sim.group_updates.sum())
        obs = game.observe()
        inputs = sensory(obs, sim.circuit.groups, cfg.sensory_hz, profile=cfg.rewards.profile,
                         wall_memory_hz=cfg.wall_memory_hz, wall_preview_cells=cfg.wall_preview_cells) + tuple(extra)
        counts = sim.advance(cfg.decision_ms, inputs)
        readout = motor(counts, sim.circuit.groups)
        chosen = readout.absolute(obs.direction) if action is None else action
        manual = bool(action is not None or extra or manual_reward is not None)
        self.interventions |= manual
        reward = game.step(chosen)
        if cfg.practice == "edge_reset" and (game.end == "pared" or game.observe().edge_distance() == 0):
            if game.end != "pared":
                game.end = "borde alcanzado"
            reward = Reward(edge_contact=2 * cfg.rewards.collision)
        elif cfg.practice == "routes" and reward.apple:
            game.end = "trayecto completado"
        elif cfg.practice == "walls":
            if self.stats.get("wall_protocol") == 2:
                self.stats["wall_escaped"] |= (game.direction != Direction(self.stats["wall_heading"])
                                               and game.end != "pared")
            if self._wall_task_complete(reward):
                game.end = "pared evitada"
                game.apple = None
                reward = reward + Reward(avoidance=cfg.rewards.avoidance)
            elif self.stats.get("wall_protocol") == 2 and (game.end == "límite" or reward.apple > 0):
                game.end = "manzana no alcanzada" if not reward.apple else "evasión no realizada"
                if reward.apple:
                    game.apple = None
                reward = reward + Reward(practice_failure=cfg.rewards.collision)
        delivered = reward if manual_reward is None else reward + manual_reward
        if shuffled:
            delivered = self.delayed.pop(0) if self.delayed else Reward()
        gates = ()
        if cfg.dopamine:
            signals = (reward, manual_reward) if manual_reward is not None and not shuffled else (delivered,)
            gates = tuple(gate for signal in signals
                          for gate in reinforcement(signal, sim.circuit.groups, cfg.reinforcement_hz,
                                                    profile=cfg.rewards.profile))
        gate_rates = {}
        for gate in gates:
            gate_rates[gate.name] = gate_rates.get(gate.name, 0.0) + gate.rate
        learning = cfg.plasticity and cfg.dopamine and not force_frozen and (not manual or learn_from_manual)
        before = sim.scales.copy() if detail and learning and gates else None
        outcome = sim.advance(cfg.outcome_ms, gates, learn=learning and bool(gates))
        changed = np.flatnonzero(sim.scales != before) if before is not None else np.empty(0, np.int32)
        self.decision_counts, self.outcome_counts = counts, outcome
        after = game.observe()
        active = counts + outcome
        s = self.stats
        s["reward"] += reward.total
        s["food_distance"] += after.food_distance()
        wall_distance = after.edge_distance()
        s["wall_distance"] += wall_distance
        s["edge_apples"] += int(wall_distance == 0 and reward.apple > 0)
        s["motor_turns"] += int(readout.turn != 0 and action is None)
        s["motor_left_spikes"] += readout.left_spikes
        s["motor_right_spikes"] += readout.right_spikes
        for name in ("DAN", "KC", "MBON"):
            s[name.lower() + "_spikes"] += int(active[sim.circuit.groups[name]].sum())
        neural_ms = cfg.decision_ms + cfg.outcome_ms
        s["neural_ms"] += neural_ms
        rss = self.process.memory_info().rss
        active_count = int(np.count_nonzero(active))
        elapsed = time.perf_counter() - start
        s["wall_seconds"] += elapsed
        s["active_peak"] = max(s["active_peak"], active_count)
        s["rss_peak"] = max(s["rss_peak"], rss)
        s["plastic_updates"] += int(sim.group_updates.sum()) - updates
        self.last_step = {
            "episode": self.episode, "practice": cfg.practice, "step": game.steps, "source": "teclado" if action is not None else "Brian",
            "manual": manual, "decision": asdict(readout), "requested": chosen.name,
            "direction": game.direction.name, "reward": asdict(reward), "total": reward.total,
            "delivered": asdict(delivered), "memory": sim.memory(),
            "reinforcement_hz": gate_rates,
            "active": active_count, "neural_ms": neural_ms, "wall_ms": elapsed * 1000,
            "rss_mib": rss / 2**20, "plasticity_enabled": learning,
            "end": game.end, "head": after.head, "apple": after.apple,
            "wall_distances": {name: obs.wall_distance(obs.direction.turn(relative))
                               for relative, name in ((-1, "left"), (0, "front"), (1, "right"))}
                              if obs.walls else {},
        }
        if cfg.practice == "walls":
            self.last_step["objective"] = ("Esquivar la pared y recoger la manzana"
                                           if self.stats.get("wall_protocol") == 2
                                           else "Salir de la zona de riesgo")
        elif cfg.practice == "edge_reset":
            self.last_step["objective"] = "Perímetro estricto"
        if detail:
            self.last_step.update({
                "input": self._stimuli(inputs, cfg.decision_ms), "modulators": self._stimuli(gates, cfg.outcome_ms),
                "groups": sim.activity(active, neural_ms),
                "plastic_edge_updates": [{"edge": int(sim.circuit.edges[i]), "scale": float(sim.scales[i]),
                                          "last_pre_age_ms": float(sim.update_age[i]) * cfg.lif.dt}
                                         for i in changed],
                "cpu_pct_one_core": self.process.cpu_percent(),
            })
        return self.last_step

    def finish(self):
        if self.game is None or not self.game.end:
            raise ValueError("El episodio todavía está activo")
        g, s = self.game, self.stats
        row = {**s, "episode": self.episode, "stage": self.config.stage, "steps": g.steps,
               "survival_moves": g.steps - int(g.end == "pared"), "end": g.end,
               "collision": int(g.end == "pared"), "score": g.apples, "apples": g.apples,
               "food_distance": s["food_distance"] / g.steps, "wall_distance": s["wall_distance"] / g.steps,
               "manual": self.interventions, "memory": self.sim.memory()["changed"]}
        row["practice"] = self.config.practice
        if self.config.practice == "routes":
            row["route_efficiency"] = s["route_optimal_steps"] / g.steps if g.apples else 0.0
        elif self.config.practice == "walls":
            row["wall_success"] = int(g.end == "pared evitada")
        elif self.config.practice == "edge_reset":
            row["edge_termination"] = int(g.end in ("borde alcanzado", "pared"))
        self.episode += 1
        self.game = None
        return row

    def association_episode(self, *, learn=True, pattern="A"):
        seed = episode_seed(self.config.seed, self.episode)
        report = conditioning_trial(self.sim, self.config, pattern, reinforced=self.config.dopamine,
                                    learn=learn and self.config.plasticity and self.config.dopamine, seed=seed,
                                    valence=self.config.association_valence)
        report.update({"stage": 1, "episode": self.episode, "seed": seed, "manual": False})
        self.episode += 1
        return report

    def set_stage(self, stage: int, *, keep_memory: bool):
        if self.game is not None:
            raise ValueError("Termina la partida antes de cambiar de etapa")
        self.config = replace(self.config, stage=stage)
        self.reset_activity()
        if not keep_memory:
            self.sim.reset_memory()
