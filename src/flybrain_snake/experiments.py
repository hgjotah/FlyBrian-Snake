"""Reproducible, separately seeded experiments and paired uncertainty."""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .circuits import Circuit
from .config import RunConfig
from .connectome import Graph
from .game import Reward
from .simulator import Simulator, Stimulus
from .storage import Journal, checkpoint, restore, write_csv
from .training import Trainer, episode_seed


METRICS = ("apples", "survival_moves", "collision", "reward", "food_distance", "wall_distance", "edge_apples")
CONDITIONS = ("A", "B", "C", "D", "E", "F", "G", "H", "I")


def trainer_for(graph: Graph, circuit: Circuit, cfg: RunConfig):
    return Trainer(Simulator(graph, circuit, cfg.lif, cfg.learning, cfg.seed,
                             synaptic_signal=cfg.synaptic_signal), cfg)


def shuffled_graph(graph: Graph, circuit: Circuit, seed: int):
    rng = np.random.default_rng(seed)
    targets = graph.targets.copy()
    # Retain explicitly modeled KC→MBON routes, preserving the tested plastic substrate.
    # The other target stubs are permuted within each neuropil.
    for region in range(len(graph.region_names)):
        edges = np.flatnonzero((graph.regions == region) & (circuit.slots < 0))
        targets[edges] = rng.permutation(targets[edges])
    manifest = dict(graph.manifest)
    manifest["source_graph"] = graph.fingerprint
    manifest["control"] = {"seed": seed, "method": "within-neuropil target stubs, plastic rows fixed",
                           "rewired_rows": int(np.count_nonzero(targets != graph.targets))}
    manifest["fingerprint"] = hashlib.sha256(
        (graph.fingerprint + json.dumps(manifest["control"], sort_keys=True)).encode()
        + targets.tobytes()).hexdigest()
    return replace(graph, targets=targets, manifest=manifest)


def evaluate(t: Trainer, seeds: list[int], sink: Journal | None = None, *, stop=lambda: False, **overrides):
    config = replace(t.config, plasticity=False, **overrides)
    probe = trainer_for(t.sim.graph, t.sim.circuit, config)
    probe.sim.scales[:] = t.sim.scales
    probe.sim.silence[:] = t.sim.silence
    results = []
    for seed in seeds:
        probe.begin(seed)
        while not probe.game.end:
            if stop():
                raise InterruptedError("Evaluación interrumpida por el usuario")
            record = probe.step(force_frozen=True)
            if sink:
                sink.append("steps", {"split": "evaluation", "seed": seed, **record})
        row = {"split": "evaluation", **probe.finish()}
        results.append(row)
        if sink:
            sink.append("episodes", row)
    return results


def train(t: Trainer, episodes: int, journal: Journal, *, shuffled_rewards=None, callback=None, stop=lambda: False):
    target = t.episode + episodes
    while t.episode < target and not stop():
        if t.config.stage == 1:
            row = t.association_episode()
        else:
            if t.game is None:
                t.begin()
            while not t.game.end:
                if shuffled_rewards:
                    t.delayed = [shuffled_rewards[(t.episode * max(1, t.config.max_steps) + t.game.steps)
                                                  % len(shuffled_rewards)]]
                record = t.step(shuffled=shuffled_rewards is not None, detail=not t.config.compact_trace)
                if not t.config.compact_trace:
                    journal.append("steps", {"split": "training", **record})
                if stop() and not t.game.end:
                    journal.save(t)
                    return
            row = t.finish()
        journal.append("episodes", {"split": "training", **row})
        label = f"etapa{t.config.stage}-episodio{t.episode}"
        if t.episode in (100, 1000) or t.episode == target:
            journal.save(t, label)
        elif t.episode % t.config.save_every == 0:
            journal.save(t)
        if t.config.stage >= 2 and t.config.eval_every and t.episode % t.config.eval_every == 0:
            seeds = [episode_seed(t.config.seed, i, 3) for i in range(5)]
            journal.append("events", {"kind": "periodic-evaluation", "trained_episodes": t.episode,
                                      "results": evaluate(t, seeds)})
        if callback:
            callback(row)
    journal.save(t)


def summarize(rows: list[dict], baseline: list[dict] | None = None, seed: int = 9):
    rng = np.random.default_rng(seed)
    if baseline is not None:
        refs = {r["seed"]: r for r in baseline}
        if set(refs) != {r["seed"] for r in rows}:
            raise ValueError("La comparación requiere las mismas semillas")
    result = {"n": len(rows), "paired": baseline is not None, "metrics": {}}
    for name in METRICS:
        x = np.asarray([r[name] if baseline is None else r[name] - refs[r["seed"]][name] for r in rows])
        ci = None
        if len(x) >= 5:
            # Resample independent evaluation seeds, never individual steps.
            boot = np.concatenate([x[rng.integers(0, len(x), (100, len(x)))].mean(axis=1)
                                   for _ in range(20)])
            ci = np.quantile(boot, [0.025, 0.975]).tolist()
        result["metrics"][name] = {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None,
                                    "bootstrap95": ci}
    return result


def run_study(graph: Graph, circuit: Circuit, cfg: RunConfig, destination: Path,
              n_train: int, n_test: int, conditions: str, progress=print, stop=lambda: False):
    def check_stop():
        if stop():
            raise InterruptedError("Estudio detenido. Se guardaron los resultados confirmados.")

    check_stop()
    destination.mkdir(parents=True, exist_ok=False)
    seeds = [episode_seed(cfg.seed, i, 9) for i in range(n_test)]
    generalization = [episode_seed(cfg.seed, i, 10) for i in range(n_test)]
    report = {"config": asdict(cfg), "train_episodes": n_train, "test_seeds": seeds,
              "generalization_seeds": generalization, "connectome": graph.fingerprint,
              "conditions": {}, "primary_metric": "apples"}
    needs_trained = any(label in conditions for label in "BG")
    requested = set(conditions) | {"A"} | ({"B"} if needs_trained else set())
    baseline = {}
    condition_scores = {}
    if "F" in requested:
        donor = trainer_for(graph, circuit, replace(cfg, plasticity=False))
        reward_stream = []
        for i in range(max(4, n_train)):
            donor.begin(episode_seed(cfg.seed, i, 11))
            while not donor.game.end:
                check_stop()
                reward_stream.append(Reward(**donor.step()["reward"]))
        rng = np.random.default_rng(episode_seed(cfg.seed, 0, 12))
        order = rng.permutation(len(reward_stream))
        scrambled = [reward_stream[i] for i in order]
        (destination / "shuffled_rewards.json").write_text(
            json.dumps([asdict(r) for r in scrambled], indent=2), encoding="utf-8")
    else:
        scrambled = None

    all_rows = []
    for label in CONDITIONS:
        if label not in requested:
            continue
        check_stop()
        progress(f"Experimento {label}")
        selected = generalization if label == "H" else seeds
        config = replace(cfg, plasticity=label != "C", dopamine=label != "D", eval_every=0)
        alt = shuffled_graph(graph, circuit, episode_seed(cfg.seed, 0, 13)) if label == "E" else graph
        # The control preserves all original plastic edges and their slot indices.
        t = trainer_for(alt, circuit, config)
        if label == "I":
            t.sim.set_silence(circuit.groups["MBON"], True)
        if label == "G":
            with Journal(destination / "B.sqlite", read_only=True) as source:
                t = restore(source.get(), graph, circuit)
            t.sim.reset()
        if label == "H":
            t.config = replace(t.config, food_partition="train")
        with Journal(destination / f"{label}.sqlite") as journal:
            journal.append("events", {"kind": "protocol", "condition": label, "graph": alt.manifest})
            if label in "BCDEFHI":
                train(t, n_train, journal, shuffled_rewards=scrambled if label == "F" else None, stop=stop)
            check_stop()
            if label == "H":
                journal.append("events", {"kind": "holdout", "train": "(x+y)%2=0", "test": "(x+y)%2=1"})
            try:
                scores = evaluate(t, selected, journal, stop=stop, dopamine=cfg.dopamine,
                                  food_partition="heldout" if label == "H" else cfg.food_partition)
            except InterruptedError:
                journal.append("events", {"kind": "evaluation-interrupted"})
                journal.save(t)
                raise
            journal.save(t)
        if label == "A":
            baseline["standard"] = scores
            if "H" in requested:
                baseline["held_out"] = evaluate(t, generalization, stop=stop, food_partition="heldout")
        reference = baseline["held_out"] if label == "H" else baseline["standard"]
        report["conditions"][label] = {"summary": summarize(scores), "delta_vs_A": summarize(scores, reference),
                                        "plasticity": t.sim.memory(), "graph": alt.fingerprint}
        condition_scores[label] = scores
        all_rows.extend({"condition": label, **row} for row in scores)
        (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        write_csv(destination / "evaluation.csv", all_rows)
    if "B" in condition_scores:
        report["B_vs_controls"] = {label: summarize(condition_scores["B"], condition_scores[label])
                                  for label in "CDEFGI" if label in condition_scores}
    stats = report["conditions"].get("B", {}).get("delta_vs_A", {}).get("metrics", {}).get("apples")
    report["interpretation"] = "Inconcluso con este protocolo. Consulta los controles y los intervalos."
    if stats and stats["bootstrap95"] and stats["bootstrap95"][0] > 0:
        report["interpretation"] = ("La media de manzanas en B supera A en este conjunto de semillas. "
                                    "La atribución requiere la comparación con C, D, F e I.")
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def benchmark(graph: Graph, circuit: Circuit, cfg: RunConfig, repetitions: int):
    import psutil
    from .storage import atomic_bytes
    import tempfile

    process = psutil.Process()
    t0 = time.perf_counter()
    trainer = trainer_for(graph, circuit, cfg)
    initialized = time.perf_counter() - t0
    sim = trainer.sim
    cue = Stimulus("KC", circuit.groups["conditioning_A"], cfg.sensory_hz)
    pam = Stimulus("PAM", circuit.groups["appetitive"], cfg.reinforcement_hz)

    def measure(learn: bool):
        times = []
        for i in range(repetitions):
            sim.reset()
            sim.rng = np.random.default_rng(i)
            sim.advance(cfg.decision_ms, (cue,))
            start = time.perf_counter()
            sim.advance(cfg.outcome_ms, (pam,), learn=learn)
            times.append((time.perf_counter() - start) * 1000)
        return times

    started = time.perf_counter()
    sim.advance(cfg.decision_ms, (cue,))
    first = time.perf_counter() - started
    process.cpu_percent()
    frozen = measure(False)
    plastic = measure(True)
    cpu = process.cpu_percent()
    t0 = time.perf_counter()
    data = checkpoint(trainer)
    with tempfile.TemporaryDirectory() as folder:
        atomic_bytes(Path(folder) / "benchmark.npz", data)
    save_ms = (time.perf_counter() - t0) * 1000
    trainer.begin()
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)
        with Journal(path / "sample.sqlite") as journal:
            for _ in range(3):
                if trainer.game.end:
                    break
                journal.append("steps", trainer.step())
            journal.save(trainer)
            t0 = time.perf_counter()
            journal.export(path / "export")
            export_ms = (time.perf_counter() - t0) * 1000
    mem = process.memory_info()
    return {"neurons": graph.n, "rows": len(graph.targets), "plastic_rows": len(circuit.edges),
            "init_s": initialized, "first_window_s": first, "window_neural_ms": cfg.outcome_ms,
            "frozen_median_ms": float(np.median(frozen)), "plastic_median_ms": float(np.median(plastic)),
            "frozen_ms": frozen, "plastic_ms": plastic, "save_ms": save_ms,
            "checkpoint_bytes": len(data), "rss_mib": mem.rss / 2**20,
            "cpu_pct_one_core": cpu, "export_three_steps_ms": export_ms,
            "peak_rss_mib": getattr(mem, "peak_wset", mem.rss) / 2**20}
