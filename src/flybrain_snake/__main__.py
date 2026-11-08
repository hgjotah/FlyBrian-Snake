"""FlyBrain Snake command line."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import os

import numpy as np
from pathlib import Path
import platform
import signal
import sys
import time

import psutil

from . import __version__
from .circuits import Circuit
from .config import burst_config
from .connectome import Graph, acquire, build
from .experiments import benchmark, evaluate, run_study, summarize, train, trainer_for
from .storage import Journal, atomic_bytes, checkpoint, load_config, restore, write_csv
from .training import episode_seed
from .validation import association, conditioning_trial, validate_brain


def print_json(value):
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)


def dump(path: Path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2).encode())


def doctor():
    modules = {}
    for name in ("numpy", "numba", "pyarrow", "psutil", "PySide6-Essentials", "brian2"):
        try:
            modules[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            modules[name] = None
    vm = psutil.virtual_memory()
    processor = os.environ.get("PROCESSOR_IDENTIFIER", platform.processor())
    if os.name == "nt":
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
                processor = winreg.QueryValueEx(key, "ProcessorNameString")[0]
        except OSError:
            pass
    return {"platform": platform.platform(), "processor": processor,
            "python": platform.python_version(), "logical_cpus": psutil.cpu_count(),
            "physical_cpus": psutil.cpu_count(logical=False), "ram_total_gib": vm.total / 2**30,
            "ram_available_gib": vm.available / 2**30, "dependencies": modules, "backend": "CPU / Numba"}


def validate(graph, circuit, config, trials):
    t = trainer_for(graph, circuit, config)
    propagation = validate_brain(t.sim, config.seed)
    conditioning = association(t.sim, config, trials)
    for i in range(trials):
        conditioning_trial(t.sim, config, seed=config.seed + i)
    loaded = restore(checkpoint(t), graph, circuit)
    retention = bool(np.array_equal(t.sim.scales, loaded.sim.scales) and t.sim.memory()["changed"])
    return {"software": __version__, "graph": graph.fingerprint, "lif": asdict(config.lif), "learning": asdict(config.learning), "protocol": validation_protocol(config),
            "propagation": propagation, "association": conditioning, "exact_checkpoint_retention": retention,
            "passed": bool(propagation["passed"] and conditioning["passed"] and retention)}


def validation_protocol(config):
    protocol = {name: getattr(config, name) for name in ("seed", "decision_ms", "outcome_ms",
                                                        "sensory_hz", "reinforcement_hz")}
    if config.synaptic_signal != "curated":
        protocol["synaptic_signal"] = config.synaptic_signal
    return protocol


def validation_path(data: Path, config):
    suffix = "-burst" if config.compact_trace else ""
    if config.synaptic_signal == "contact_probabilities":
        suffix += "-contacts"
    filename = f"validation{suffix}.json"
    return data / "runs" / filename


def gate(graph, circuit, config, file: Path):
    if file.is_file():
        report = json.loads(file.read_text(encoding="utf-8"))
        if (report.get("passed") and report.get("exact_checkpoint_retention") and report.get("graph") == graph.fingerprint
                and report.get("lif") == asdict(config.lif)
                and report.get("learning") == asdict(config.learning)
                and report.get("protocol") == validation_protocol(config)
                and report.get("software") == __version__):
            return
    report = validate(graph, circuit, config, 8)
    dump(file, report)
    if not report["passed"]:
        raise ValueError(f"Etapa 0 sin superar. Resultados en {file}")


def parser():
    p = argparse.ArgumentParser(prog="flybrain", description="FlyBrain Snake · laboratorio de Brian")
    p.add_argument("--data", type=Path, default=Path("data"), help="Directorio base de datos")
    p.add_argument("--config", type=Path, help="JSON con parámetros de RunConfig")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    walls = sub.add_parser("wall-study", help="Entrenar y evaluar evasión de paredes con controles")
    walls.add_argument("out", type=Path)
    walls.add_argument("--train", type=int, default=96)
    walls.add_argument("--test", type=int, default=48)
    walls.add_argument("--session", type=Path)
    walls.add_argument("--label", default="latest")
    prep = sub.add_parser("prepare")
    prep.add_argument("--local-only", action="store_true")
    atlas = sub.add_parser("anatomy-prepare", help="Indexar el archivo oficial de contactos v783")
    atlas.add_argument("--source", type=Path, help="Carpeta que contiene los archivos oficiales")
    info = sub.add_parser("anatomy-show", help="Consultar el atlas local")
    pick = info.add_mutually_exclusive_group()
    pick.add_argument("--id", type=int, help="ID oficial de un contacto")
    pick.add_argument("--root", type=int, help="ID neuronal para el recuento por neuropilo")
    check = sub.add_parser("validate")
    check.add_argument("--trials", type=int, default=12)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--repetitions", type=int, default=5)
    desktop = sub.add_parser("gui")
    desktop.add_argument("--smoke", action="store_true")
    desktop.add_argument("--session", type=Path)
    runner = sub.add_parser("train")
    runner.add_argument("session", type=Path)
    runner.add_argument("--episodes", type=int)
    runner.add_argument("--burst", action="store_true", help="Práctica compacta con 12 movimientos por episodio")
    runner.add_argument("--resume", action="store_true")
    runner.add_argument("--stage", type=int, choices=range(1, 5))
    runner.add_argument("--reset-memory", action="store_true")
    runner.add_argument("--frozen", action="store_true")
    evalp = sub.add_parser("evaluate")
    evalp.add_argument("--session", type=Path)
    evalp.add_argument("--label", default="latest")
    evalp.add_argument("--episodes", type=int, default=10)
    evalp.add_argument("--out", type=Path, default=Path("data/runs/evaluation.csv"))
    study = sub.add_parser("study")
    study.add_argument("out", type=Path)
    study.add_argument("--train", type=int, default=20)
    study.add_argument("--test", type=int, default=10)
    study.add_argument("--conditions", default="ABCDEFGHI")
    study.add_argument("--stop-file", type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("session", type=Path)
    compare.add_argument("out", type=Path)
    compare.add_argument("--episodes", type=int, default=10)
    compare.add_argument("--labels", nargs="+", default=["latest"])
    exp = sub.add_parser("export")
    exp.add_argument("session", type=Path)
    exp.add_argument("out", type=Path)
    diff = sub.add_parser("memory")
    diff.add_argument("session", type=Path)
    diff.add_argument("out", type=Path)
    diff.add_argument("--label", default="latest")
    return p


def dispatch(a):
    data = a.data.resolve()
    processed = data / "processed"
    cfg = load_config(json.loads(a.config.read_text(encoding="utf-8-sig")) if a.config else {})
    if a.cmd == "doctor":
        print_json(doctor())
        return
    if a.cmd == "prepare":
        if processed.is_dir():
            print_json(Graph.load(processed, verify=True).manifest)
        else:
            raw = data / "raw"
            print_json(build(raw, processed, acquire(raw, a.local_only)))
        return
    if a.cmd == "gui":
        from .gui import launch
        if a.config is None and cfg.stage == 4 and cfg.practice == "snake":
            cfg = replace(cfg, max_steps=0)
        return launch(data, cfg, a.smoke, a.session)
    if a.cmd == "export":
        with Journal(a.session, read_only=True) as j:
            j.export(a.out)
        return
    start = time.perf_counter()
    graph = Graph.load(processed)
    if a.cmd in ("anatomy-prepare", "anatomy-show"):
        from .anatomy import ContactAtlas, prepare
        if a.cmd == "anatomy-prepare":
            manifest = prepare(data, graph, a.source)
            print_json({k: v for k, v in manifest.items() if k != "batch_offsets"})
        else:
            atlas = ContactAtlas(data, graph)
            if a.id is not None:
                try:
                    row = atlas.ordinal(a.id)
                except KeyError as error:
                    raise ValueError(error.args[0]) from error
                print_json(atlas.detail(row))
            elif a.root is not None:
                print_json(atlas.profile(a.root))
            else:
                print_json({k: v for k, v in atlas.manifest.items() if k != "batch_offsets"})
        return
    circuit = Circuit.from_graph(graph)
    initialized = time.perf_counter() - start
    report_path = validation_path(data, cfg)
    if a.cmd == "wall-study":
        from .wall_study import run
        initial = trainer_for(graph, circuit, cfg)
        if a.session is not None:
            with Journal(a.session, read_only=True) as journal:
                initial = restore(journal.get(a.label), graph, circuit)
            settings = asdict(initial.config)
            overrides = json.loads(a.config.read_text(encoding="utf-8-sig")) if a.config else {}
            for key, value in overrides.items():
                settings[key] = {**settings[key], **value} if key in ("lif", "learning", "rewards") else value
            cfg = load_config(settings)
        report = run(graph, circuit, cfg, initial.sim.scales.copy(), a.out, a.train, a.test)
        print(report["interpretation"])
        return
    if a.cmd == "validate":
        if a.trials < 1:
            raise ValueError("El número de ensayos debe ser positivo")
        report = validate(graph, circuit, cfg, a.trials)
        dump(report_path, report)
        print_json(report)
        return 0 if report["passed"] else 2
    if a.cmd == "benchmark":
        if a.repetitions < 1:
            raise ValueError("Se necesita al menos una repetición")
        result = {"machine": doctor(), "open_graph_and_circuits_s": initialized,
                  **benchmark(graph, circuit, cfg, a.repetitions)}
        dump(data / "runs" / "benchmark.json", result)
        print_json(result)
    if a.cmd == "train":
        episodes = a.episodes if a.episodes is not None else (1000 if a.burst else 20)
        if a.burst:
            if a.resume or a.stage not in (None, 4):
                raise ValueError("La ráfaga inicia una sesión nueva de etapa 4")
            cfg = burst_config(cfg)
        if episodes < 1:
            raise ValueError("El número de episodios debe ser positivo")
        if a.session.exists() != a.resume:
            raise ValueError("Para una sesión existente utiliza --resume. Para una nueva, indica un archivo libre.")
        with Journal(a.session, create=not a.resume) as j:
            t = restore(j.get(), graph, circuit) if a.resume else trainer_for(graph, circuit, cfg)
            stage = t.config.stage if a.stage is None else a.stage
            if stage == 0:
                raise ValueError("Usa validate para la etapa 0")
            if stage != t.config.stage or a.reset_memory:
                t.set_stage(stage, keep_memory=not a.reset_memory)
            if a.frozen:
                t.config = replace(t.config, plasticity=False)
            gate(graph, circuit, t.config, validation_path(data, t.config))
            j.append("events", {"kind": "resume" if a.resume else "start", "config": asdict(t.config)})
            j.save(t)
            stopping = False

            def interrupt(*_):
                nonlocal stopping
                stopping = True

            old = signal.signal(signal.SIGINT, interrupt)
            try:
                train(t, episodes, j, callback=lambda row: print(
                    json.dumps({k: row[k] for k in ("episode", "stage", "seed", "apples", "changed", "end") if k in row},
                               ensure_ascii=False), flush=True), stop=lambda: stopping)
            finally:
                signal.signal(signal.SIGINT, old)
        return 130 if stopping else 0
    if a.cmd in ("evaluate", "memory"):
        if a.session:
            with Journal(a.session, create=False) as j:
                t = restore(j.get(a.label), graph, circuit)
        else:
            t = trainer_for(graph, circuit, cfg)
        if a.cmd == "memory":
            rows = t.sim.memory_rows()
            a.out.parent.mkdir(parents=True, exist_ok=True)
            write_csv(a.out, rows)
            print(a.out.resolve())
        else:
            if a.episodes < 1 or t.config.stage < 2:
                raise ValueError("Evaluar Snake requiere una etapa de juego y al menos un episodio")
            seeds = [episode_seed(t.config.seed, i, 8) for i in range(a.episodes)]
            rows = evaluate(t, seeds)
            a.out.parent.mkdir(parents=True, exist_ok=True)
            write_csv(a.out, rows)
            print_json(summarize(rows))
    if a.cmd == "compare":
        if a.episodes < 1:
            raise ValueError("La comparación requiere al menos una semilla")
        with Journal(a.session, read_only=True) as j:
            first = restore(j.get(a.labels[0]), graph, circuit)
            if first.config.stage < 2:
                raise ValueError("La comparación de partidas requiere una etapa de Snake")
            seeds = [episode_seed(first.config.seed, i, 8) for i in range(a.episodes)]
            reference = trainer_for(graph, circuit, first.config)
            baseline = evaluate(reference, seeds)
            rows = [{"checkpoint": "inicial", **row} for row in baseline]
            summary = {"config": asdict(first.config), "graph": graph.fingerprint, "seeds": seeds,
                       "inicial": summarize(baseline), "checkpoints": {}}
            a.out.mkdir(parents=True, exist_ok=True)
            for label in a.labels:
                t = restore(j.get(label), graph, circuit)
                if t.config != first.config:
                    raise ValueError("Los checkpoints deben compartir el mismo protocolo")
                scores = evaluate(t, seeds)
                rows.extend({"checkpoint": label, **row} for row in scores)
                summary["checkpoints"][label] = {"memory": t.sim.memory(), "summary": summarize(scores),
                                                  "delta_vs_initial": summarize(scores, baseline)}
                weights = t.sim.memory_rows()
                write_csv(a.out / f"pesos-{len(summary['checkpoints']):02d}.csv", weights)
        write_csv(a.out / "evaluation.csv", rows)
        dump(a.out / "report.json", summary)
        print_json(summary)
    if a.cmd == "study":
        conditions = a.conditions.upper()
        if a.train < 1 or a.test < 1 or cfg.stage < 2 or not conditions or set(conditions) - set("ABCDEFGHI"):
            raise ValueError("Estudio inválido. Usa una etapa de juego y condiciones de A a I")
        gate(graph, circuit, cfg, report_path)
        report = run_study(graph, circuit, cfg, a.out, a.train, a.test, conditions,
                           stop=a.stop_file.exists if a.stop_file else lambda: False)
        print(report["interpretation"])
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return dispatch(args) or 0
    except (OSError, ValueError, MemoryError, KeyboardInterrupt) as error:
        print(f"FlyBrain Snake · {type(error).__name__}: {error}", file=sys.stderr)
        return 130 if isinstance(error, (InterruptedError, KeyboardInterrupt)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
