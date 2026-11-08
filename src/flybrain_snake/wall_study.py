"""Paired wall-avoidance evaluation with frozen and dopamine-free controls."""
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .experiments import train, trainer_for
from .storage import Journal, atomic_bytes, restore
from .training import episode_seed


def assess(graph, circuit, config, scales, seeds, *, stop=lambda: False):
    probe = trainer_for(graph, circuit, replace(config, plasticity=False, dopamine=True))
    probe.sim.scales[:] = scales
    rows = []
    for index, seed in enumerate(seeds):
        probe.episode = index
        probe.begin(seed)
        while not probe.game.end:
            if stop():
                raise InterruptedError("Validación de paredes interrumpida")
            probe.step(force_frozen=True, detail=False)
        row = probe.finish()
        rows.append({key: row[key] for key in ("seed", "wall_case", "wall_success", "collision", "steps", "end")})
    return {"success_rate": float(np.mean([row["wall_success"] for row in rows])),
            "collision_rate": float(np.mean([row["collision"] for row in rows])),
            "mean_steps": float(np.mean([row["steps"] for row in rows])), "trials": rows}


def assess_games(graph, circuit, config, scales, seeds, *, stop=lambda: False):
    """Check transfer to ordinary Snake; the cap applies to this probe only."""
    config = replace(config, practice="snake", stage=4, plasticity=False, max_steps=64)
    probe = trainer_for(graph, circuit, config)
    probe.sim.scales[:] = scales
    rows = []
    for seed in seeds:
        probe.begin(seed)
        while not probe.game.end:
            if stop():
                raise InterruptedError("Validación de partidas interrumpida")
            probe.step(force_frozen=True, detail=False)
        row = probe.finish()
        rows.append({key: row[key] for key in ("seed", "collision", "apples", "survival_moves")})
    return {"trials": rows, "collisions": sum(row["collision"] for row in rows),
            "apples": sum(row["apples"] for row in rows),
            "survival_moves": sum(row["survival_moves"] for row in rows)}


def preserves_walls(candidate, reference):
    # A completed extra exercise must never compensate for an extra collision.
    if (candidate["collision_rate"] > reference["collision_rate"]
            or candidate["success_rate"] < reference["success_rate"]):
        return False
    if any(a["collision"] > b["collision"] for a, b in
           zip(candidate.get("trials", ()), reference.get("trials", ()), strict=True)):
        return False
    return True


def improves_walls(candidate, reference):
    return (preserves_walls(candidate, reference)
            and (candidate["success_rate"] > reference["success_rate"]
                 or candidate["collision_rate"] < reference["collision_rate"]))


def preserves_games(candidate, reference):
    return (candidate["collisions"] <= reference["collisions"]
            and candidate["apples"] >= reference["apples"]
            and candidate["survival_moves"] >= reference["survival_moves"]
            and all(a["collision"] <= b["collision"] for a, b in
                    zip(candidate["trials"], reference["trials"], strict=True)))


def improves_games(candidate, reference):
    return (preserves_games(candidate, reference)
            and (candidate["collisions"] < reference["collisions"]
                 or candidate["apples"] > reference["apples"]
                 or candidate["survival_moves"] > reference["survival_moves"]))


class WallSelection:
    """Keep a candidate only when independent, frozen validation improves."""

    def __init__(self, graph, circuit, config, scales, *, measure=assess, game_measure=assess_games, stop=None):
        self.graph, self.circuit = graph, circuit
        self.config = replace(config, practice="walls", stage=4, max_steps=max(12, config.size * 2))
        self.measure = measure if stop is None else lambda *args: measure(*args, stop=stop)
        self.game_measure = game_measure if stop is None else lambda *args: game_measure(*args, stop=stop)
        self.seeds = [episode_seed(config.seed, i, 25) for i in range(24)]
        self.game_seeds = [episode_seed(config.seed, i, 26) for i in range(6)]
        self.scales = scales.copy()
        self.best = self.measure(graph, circuit, self.config, self.scales, self.seeds)
        self.best_games = None
        self.episode = 0

    def consider(self, scales, episode):
        candidate = self.measure(self.graph, self.circuit, self.config, scales, self.seeds)
        accepted = preserves_walls(candidate, self.best) and not np.array_equal(scales, self.scales)
        better_exercises = improves_walls(candidate, self.best)
        reason = "sin mejora segura en ejercicios"
        games = None
        if accepted:
            if self.best_games is None:
                self.best_games = self.game_measure(self.graph, self.circuit, self.config,
                                                     self.scales, self.game_seeds)
            games = self.game_measure(self.graph, self.circuit, self.config, scales, self.game_seeds)
            safe_games = preserves_games(games, self.best_games)
            accepted = (safe_games
                        and (better_exercises or improves_games(games, self.best_games)))
            reason = ("mejora conservada" if accepted else "sin mejora validada"
                      if safe_games else "regresión en partidas completas")
        if accepted:
            self.scales[:] = scales
            self.best = candidate
            self.best_games = games
        scales[:] = self.scales
        self.episode = episode
        return {"candidate_success": candidate["success_rate"],
                "candidate_collision": candidate["collision_rate"],
                "selected_success": self.best["success_rate"],
                "selected_collision": self.best["collision_rate"], "accepted": accepted,
                "reason": reason, "games": games, "selected_games": self.best_games}


def run(graph, circuit, config, scales, destination: Path, episodes=96, trials=48, progress=print):
    if episodes < 1 or trials < 24:
        raise ValueError("Se necesita al menos un episodio y 24 ensayos para cubrir todos los escenarios")
    config = replace(config, stage=4, practice="walls", compact_trace=True, eval_every=0, save_every=12,
                     max_steps=config.max_steps or max(12, config.size * 2))
    destination.mkdir(parents=True, exist_ok=False)
    seeds = [episode_seed(config.seed, index, 24) for index in range(trials)]
    baseline = assess(graph, circuit, config, scales, seeds)
    validation_seeds = [episode_seed(config.seed, index, 25) for index in range(24)]
    selector = WallSelection(graph, circuit, config, scales)
    initial_validation = selector.best
    report = {"validation_seeds": validation_seeds, "config": asdict(config), "graph": graph.fingerprint, "train_episodes": episodes,
              "test_seeds": seeds, "baseline": baseline, "conditions": {},
              "primary_metric": "wall_success", "weights_frozen_during_evaluation": True,
              "selection_requires_no_extra_collisions": True, "transfer_seeds": selector.game_seeds}
    atomic_bytes(destination / "report.json", json.dumps(report, indent=2).encode())
    progress(f"Antes del entrenamiento: {baseline['success_rate']:.1%} de evasiones")
    for name, plasticity, dopamine in (("trained", True, True), ("frozen", False, True),
                                       ("no_dopamine", True, False)):
        learner = trainer_for(graph, circuit, replace(config, plasticity=plasticity, dopamine=dopamine))
        learner.sim.scales[:] = scales
        file = destination / f"{name}.sqlite"
        best_label = "antes"
        best_rate = initial_validation["success_rate"]
        selection = [{"episode": 0, "success_rate": best_rate}]
        with Journal(file) as journal:
            journal.save(learner, "antes")

            def completed(row):
                nonlocal best_label, best_rate
                episode = row["episode"] + 1
                if episode % 12 and episode != episodes:
                    return
                progress(f"{name}: {episode}/{episodes}")
                if name != "trained":
                    return
                label = f"validacion-{episode}"
                journal.save(learner, label)
                # Preserve the candidate training trajectory; only selection uses a copy.
                result = selector.consider(learner.sim.scales.copy(), episode)
                selection.append({"episode": episode, "success_rate": result["candidate_success"], **result})
                if result["accepted"]:
                    best_rate, best_label = result["selected_success"], label
                progress(f"Validación: {result['candidate_success']:.1%}; {result['reason']}")

            train(learner, episodes, journal, callback=completed)
            journal.save(learner, "ultimo_entrenado")
            last_changed = int(np.count_nonzero(learner.sim.scales != scales))
            if name == "trained":
                learner = restore(journal.get(best_label), graph, circuit)
            journal.save(learner, "seleccionado")
        changed = int(np.count_nonzero(learner.sim.scales != scales))
        if name != "trained" and changed:
            raise RuntimeError(f"El control {name} modificó los pesos")
        measured = assess(graph, circuit, config, learner.sim.scales, seeds)
        delta = np.array([after["wall_success"] - before["wall_success"]
                          for before, after in zip(baseline["trials"], measured["trials"], strict=True)])
        rng = np.random.default_rng(79)
        bootstrap = delta[rng.integers(0, trials, (2000, trials))].mean(axis=1)
        result = {**measured, "weights_changed": changed, "last_trained_weights_changed": last_changed,
                  "selected_checkpoint": best_label, "validation_history": selection,
                  "validation_success_rate": best_rate, "session": str(file.resolve()),
                  "paired_success_delta": float(delta.mean()),
                  "paired_bootstrap95": np.quantile(bootstrap, [.025, .975]).tolist()}
        report["conditions"][name] = result
        atomic_bytes(destination / "report.json", json.dumps(report, indent=2).encode())
        progress(f"{name}: {measured['success_rate']:.1%} de evasiones, {changed} pesos modificados")
    learned = report["conditions"]["trained"]
    report["improvement_demonstrated"] = bool(learned["weights_changed"]
                                              and learned["paired_bootstrap95"][0] > 0)
    report["interpretation"] = (
        "Mejora en los ensayos reservados, con intervalo de la diferencia por encima de cero."
        if report["improvement_demonstrated"] else
        "La evaluación no demuestra una mejora fiable atribuible al aprendizaje.")
    atomic_bytes(destination / "report.json", json.dumps(report, indent=2).encode())
    return report
