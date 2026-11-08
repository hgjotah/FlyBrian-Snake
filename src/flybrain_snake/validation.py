"""Brain-only propagation and conditioning protocols."""
from __future__ import annotations

import numpy as np

from .config import RunConfig
from .simulator import Simulator, Stimulus

# SugarR protocol and MN9 readout from the Shiu et al. example notebook.
SUGAR_ROOTS = (
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
)
MN9_ROOT = 720575940660219265


def validate_brain(sim: Simulator, seed: int, duration_ms: float = 1000):
    graph = sim.graph
    available = set(graph.ids.tolist())
    missing = [str(root) for root in SUGAR_ROOTS if root not in available]
    sugar = np.array([graph.index(root) for root in SUGAR_ROOTS if root in available], dtype=np.int32)
    mn9 = graph.index(MN9_ROOT)

    def trial():
        sim.reset()
        sim.rng = np.random.default_rng(seed)
        return sim.advance(duration_ms, (Stimulus("Shiu SugarR", sugar, 150),))

    sim.reset()
    baseline = sim.advance(100)
    first = trial()
    repeated = trial()
    sim.reset()
    response = sim.advance(60, (Stimulus("LC10a izquierda", sim.circuit.groups["food_left"], 90),))
    after = sim.advance(250)
    novel_spikes = int(first.sum() - first[sugar].sum())
    sensory_spikes = int(response[sim.circuit.groups["food_left"]].sum())
    circuit_activity = sim.activity(response, 60)
    motor_spikes = sum(circuit_activity[name]["spikes"] for name in ("motor_left", "motor_right"))
    return {
        "neurons": graph.n, "rest_spikes": int(baseline.sum()), "sugar_spikes": int(first.sum()),
        "sugar_non_target_spikes": novel_spikes, "sugar_active": int(np.count_nonzero(first)),
        "mn9_hz": int(first[mn9]) * 1000 / duration_ms,
        "duration_ms": duration_ms, "sugar_ids_missing": missing, "identical_seed": bool(np.array_equal(first, repeated)),
        "sensory_spikes": sensory_spikes, "motor_spikes": motor_spikes, "sensory_response": circuit_activity,
        "post_stimulus_spikes_250ms": int(after.sum()),
        "peak_hz_after_stimulus": float(after.max()) * 4,
        "passed": bool(graph.n == 139255 and not baseline.any() and novel_spikes > 0
                       and np.array_equal(first, repeated) and sensory_spikes > 0 and motor_spikes > 0
                       and after.max() < 100),
    }


def conditioning_trial(sim: Simulator, cfg: RunConfig, pattern: str = "A", *,
                       reinforced=True, learn=True, seed: int | None = None, valence="appetitive"):
    sim.reset()
    if seed is not None:
        sim.rng = np.random.default_rng(seed)
    groups = sim.circuit.groups
    cue = (Stimulus("conditioning_" + pattern, groups["conditioning_" + pattern], cfg.sensory_hz),)
    response = sim.advance(cfg.decision_ms, cue)
    reward = (Stimulus(valence, groups[valence], cfg.reinforcement_hz),) if reinforced else ()
    outcome = sim.advance(cfg.outcome_ms, reward, learn=learn)
    return {
        "cue": pattern, "mbon_spikes": int(response[groups["MBON"]].sum()),
        "pam_spikes": int(outcome[groups["appetitive"]].sum()),
        "ppl1_spikes": int(outcome[groups["aversive"]].sum()), "valence": valence,
        "changed": sim.memory()["changed"],
    }


def association(sim: Simulator, cfg: RunConfig, trials: int = 12):
    initial = sim.scales.copy()

    def probe(pattern):
        return conditioning_trial(sim, cfg, pattern, reinforced=False, learn=False, seed=91)

    before = {p: probe(p) for p in ("A", "B")}
    cases = {}
    for label in ("paired", "frozen", "no_dopamine", "dopamine_alone"):
        sim.scales[:] = initial
        for seed in range(cfg.seed, cfg.seed + trials):
            if label == "dopamine_alone":
                sim.reset()
                sim.rng = np.random.default_rng(seed)
                sim.advance(cfg.outcome_ms, (Stimulus("PAM01", sim.circuit.groups["appetitive"],
                                                      cfg.reinforcement_hz),), learn=True)
            else:
                conditioning_trial(sim, cfg, reinforced=label != "no_dopamine",
                                   learn=label == "paired", seed=seed)
        delta = np.abs(sim.scales - initial)
        select = {p: np.isin(sim.circuit.pre, sim.circuit.groups["conditioning_" + p])
                  for p in ("A", "B")}
        cases[label] = {
            "changed": int(np.count_nonzero(delta)), "max_change": float(delta.max()),
            "mean_change_by_cue": {p: float(delta[mask].mean()) if mask.any() else None
                                   for p, mask in select.items()},
            "after": {p: probe(p) for p in ("A", "B")},
        }
    sim.scales[:] = initial
    sim.reset()
    return {"before": before, "conditions": cases, "trials": trials,
            "passed": cases["paired"]["changed"] > 0 and all(
                cases[k]["changed"] == 0 for k in ("frozen", "no_dopamine", "dopamine_alone"))}
