"""Compiled CPU LIF solver. CSR rows are traversed on delayed spike arrival."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from numba import njit

from .circuits import Circuit
from .config import LIF, Plasticity
from .connectome import Graph


@dataclass(frozen=True)
class Stimulus:
    name: str
    indices: np.ndarray
    rate: float
    stream: str = "main"


@njit(inline="always")
def _decay(i, t, start, ready, updated, voltage, drive, rest, av, ag, coupling):
    elapsed = t - max(updated[i], ready[i] - start - 1)
    if elapsed > 0:
        voltage[i] = rest + (voltage[i] - rest) * av[elapsed] + drive[i] * coupling[elapsed]
        drive[i] *= ag[elapsed]
    updated[i] = t


@njit(cache=True, nogil=True)
def _integrate(n_steps, start, av, ag, coupling, rest, reset, threshold,
               refractory_steps, delay_steps, jump, weight, voltage, drive, ready,
               last, silence, offsets, targets, contacts, sign, edge_gain, ring, ring_counts,
               input_times, input_nodes, counts, fired, scales, slots, dan_ptr,
               dans, dan_strengths, plastic_ptr, pre, post, learn, pre_decay, post_decay, rate,
               minimum, require_post, group_updates, last_update, update_age):
    # With no new inputs, a neuron below threshold and with drive <= threshold - rest
    # cannot cross threshold. Integrate those neurons lazily when an event arrives.
    gap = threshold - rest
    updated = np.full(voltage.size, -1, dtype=np.int32)
    in_hot = np.zeros(voltage.size, dtype=np.bool_)
    hot = np.empty(voltage.size, dtype=np.int32)
    n_hot = 0
    for i in range(voltage.size):
        if not silence[i] and (drive[i] > gap or voltage[i] > threshold):
            hot[n_hot] = i
            n_hot += 1
            in_hot[i] = True
    contact_mode = len(edge_gain) > 0
    p = 0
    for local in range(n_steps):
        t = start + local
        num_fired = 0
        remaining = 0
        for k in range(n_hot):
            neuron = hot[k]
            _decay(neuron, local, start, ready, updated, voltage, drive, rest, av, ag, coupling)
            if t >= ready[neuron] and voltage[neuron] > threshold:
                fired[num_fired] = neuron
                num_fired += 1
                counts[neuron] += 1
                last[neuron] = t
                in_hot[neuron] = False
            elif drive[neuron] > gap or voltage[neuron] > threshold:
                hot[remaining] = neuron
                remaining += 1
            else:
                in_hot[neuron] = False
        n_hot = remaining

        slot = t % len(ring_counts)
        for k in range(ring_counts[slot]):
            source = ring[slot, k]
            signed = weight * sign[source]
            if (contact_mode or signed != 0) and not silence[source]:
                for edge in range(offsets[source], offsets[source + 1]):
                    dest = targets[edge]
                    if not silence[dest] and last[dest] != t:
                        delta = (weight * edge_gain[edge] if contact_mode else signed) * contacts[edge]
                        if delta == 0:
                            continue
                        _decay(dest, local, start, ready, updated, voltage, drive, rest, av, ag, coupling)
                        plastic = slots[edge]
                        if plastic >= 0:
                            delta *= scales[plastic]
                        drive[dest] += delta
                        if drive[dest] > gap and not in_hot[dest]:
                            hot[n_hot] = dest
                            n_hot += 1
                            in_hot[dest] = True

        while p < len(input_times) and input_times[p] == local:
            neuron = input_nodes[p]
            if not silence[neuron] and t >= ready[neuron] and last[neuron] != t:
                _decay(neuron, local, start, ready, updated, voltage, drive, rest, av, ag, coupling)
                voltage[neuron] += jump
                if voltage[neuron] > threshold and not in_hot[neuron]:
                    hot[n_hot] = neuron
                    n_hot += 1
                    in_hot[neuron] = True
            p += 1

        ring_counts[slot] = 0
        outgoing = (t + delay_steps) % len(ring_counts)
        # Keep accumulation order reproducible across layouts.
        ordered = np.sort(fired[:num_fired]) if num_fired > 1 else fired[:num_fired]
        for i in range(num_fired):
            neuron = ordered[i]
            voltage[neuron] = reset
            drive[neuron] = 0
            ready[neuron] = t + refractory_steps
            ring[outgoing, i] = neuron
        ring_counts[outgoing] = num_fired

        if learn:
            for group in range(len(dan_ptr) - 1):
                gate = 0.0
                for k in range(dan_ptr[group], dan_ptr[group + 1]):
                    if last[dans[k]] == t:
                        gate += dan_strengths[k]
                if gate == 0:
                    continue
                for k in range(plastic_ptr[group], plastic_ptr[group + 1]):
                    pre_tick = last[pre[k]]
                    post_tick = last[post[k]]
                    if pre_tick < 0 or (require_post and post_tick < 0):
                        continue
                    eligibility = math.exp((pre_tick - t) * pre_decay)
                    if require_post:
                        eligibility *= math.exp((post_tick - t) * post_decay)
                    new_scale = max(minimum, scales[k] - rate * gate * eligibility * (scales[k] - minimum))
                    if np.float32(new_scale) != scales[k]:
                        scales[k] = new_scale
                        group_updates[group] += 1
                        last_update[k] = t
                        update_age[k] = t - pre_tick

    for neuron in range(voltage.size):
        if not silence[neuron] and (drive[neuron] != 0 or voltage[neuron] != rest):
            _decay(neuron, n_steps - 1, start, ready, updated, voltage, drive, rest, av, ag, coupling)


class Simulator:
    def __init__(self, graph: Graph, circuit: Circuit, lif: LIF, learning: Plasticity, seed: int,
                 *, synaptic_signal: str = "curated"):
        self.graph, self.circuit, self.lif, self.learning = graph, circuit, lif, learning
        self.edge_gain = np.empty(0, dtype=np.float32)
        self.gain_fingerprint = None
        if synaptic_signal == "contact_probabilities":
            from .anatomy import load_gain
            if graph.data_path is None:
                raise ValueError("El modo por contactos necesita el atlas v783 preparado")
            self.edge_gain, self.gain_fingerprint = load_gain(
                graph.data_path, graph.manifest.get("source_graph", graph.fingerprint), len(graph.targets))
        self.rng = np.random.default_rng(seed)
        self.boundary_rng = np.random.default_rng(np.random.SeedSequence([seed, 13]))
        self.delay_steps = round(lif.delay / lif.dt)
        self.refractory_steps = round(lif.refractory / lif.dt)
        self.voltage = np.empty(graph.n, dtype=np.float64)
        self.drive = np.empty(graph.n, dtype=np.float64)
        self.ready = np.empty(graph.n, dtype=np.int64)
        self.last = np.empty(graph.n, dtype=np.int64)
        self.silence = np.zeros(graph.n, dtype=np.bool_)
        self.ring = np.zeros((self.delay_steps + 1, graph.n), dtype=np.int32)
        self.ring_counts = np.zeros(self.delay_steps + 1, dtype=np.int32)
        self.scales = np.ones(len(circuit.edges), dtype=np.float32)
        self.group_updates = np.zeros(len(circuit.compartment_names), dtype=np.int64)
        self.last_update = np.full(len(circuit.edges), -1, dtype=np.int64)
        self.update_age = np.full(len(circuit.edges), -1, dtype=np.int64)
        self._fired = np.empty(graph.n, dtype=np.int32)
        self.lifetime_ticks = 0
        self.reset()

    def reset(self):
        self.voltage.fill(self.lif.rest)
        self.drive.fill(0)
        self.ready.fill(0)
        self.last.fill(-1)
        self.ring_counts.fill(0)
        self.tick = 0

    def reset_memory(self):
        self.scales.fill(1)
        self.group_updates.fill(0)
        self.last_update.fill(-1)
        self.update_age.fill(-1)

    def _events(self, n_steps: int, stimuli: tuple[Stimulus, ...]):
        times, nodes = [], []
        for item in stimuli:
            if item.stream not in ("main", "boundary"):
                raise ValueError(f"Flujo de estímulo desconocido: {item.stream}")
            if (not math.isfinite(item.rate) or item.rate < 0
                    or np.any(item.indices < 0) or np.any(item.indices >= self.graph.n)):
                raise ValueError(f"Estímulo inválido: {item.name}")
            if item.rate <= 0 or not len(item.indices):
                continue
            prob = -math.expm1(-item.rate * self.lif.dt / 1000)
            for start in range(0, n_steps, 1024):
                stop = min(start + 1024, n_steps)
                generator = self.boundary_rng if item.stream == "boundary" else self.rng
                hits, indices = np.nonzero(generator.random((stop - start, len(item.indices))) < prob)
                times.append(hits + start)
                nodes.append(item.indices[indices])
        if not times:
            return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
        t, n = np.concatenate(times), np.concatenate(nodes)
        order = np.argsort(t, kind="stable")
        return t[order].astype(np.int32), n[order].astype(np.int32)

    def advance(self, ms: float, stimuli: tuple[Stimulus, ...] = (), *, learn: bool = False):
        steps = round(ms / self.lif.dt)
        if steps <= 0 or not math.isclose(ms, steps * self.lif.dt, abs_tol=1e-8):
            raise ValueError("El intervalo debe ser un múltiplo positivo de dt")
        t, nodes = self._events(steps, stimuli)
        counts = np.zeros(self.graph.n, dtype=np.int32)
        lif, graph, circuit, rule = self.lif, self.graph, self.circuit, self.learning
        times = np.arange(steps + 1, dtype=np.float64) * lif.dt
        av, ag = np.exp(-times / lif.membrane_tau), np.exp(-times / lif.synapse_tau)
        if lif.membrane_tau == lif.synapse_tau:
            coupling = av * times / lif.membrane_tau
        else:
            coupling = (av - ag) * lif.synapse_tau / (lif.membrane_tau - lif.synapse_tau)
        _integrate(
            steps, self.tick, av, ag, coupling, lif.rest, lif.reset, lif.threshold,
            self.refractory_steps, self.delay_steps, lif.stimulus_jump, lif.unit_weight,
            self.voltage, self.drive, self.ready, self.last, self.silence,
            graph.offsets, graph.targets, graph.counts, graph.sign, self.edge_gain, self.ring, self.ring_counts,
            t, nodes, counts, self._fired, self.scales, circuit.slots, circuit.dan_ptr,
            circuit.dan_indices, circuit.dan_strengths, circuit.plastic_ptr, circuit.pre, circuit.post, learn,
            lif.dt / rule.tau_ms, lif.dt / rule.post_tau_ms, rule.learning_rate, rule.min_scale,
            rule.post_required, self.group_updates, self.last_update, self.update_age,
        )
        self.tick += steps
        self.lifetime_ticks += steps
        return counts

    def set_silence(self, indices: np.ndarray, enabled: bool):
        self.silence[indices] = enabled
        if enabled:
            self.voltage[indices] = self.lif.rest
            self.drive[indices] = 0
            self.last[indices] = -1

    def activity(self, counts: np.ndarray, ms: float):
        return {name: {"spikes": int(counts[indices].sum()),
                       "hz": float(counts[indices].mean()) * 1000 / ms}
                for name, indices in self.circuit.groups.items()}

    def memory_rows(self, limit: int | None = None):
        rows = list(self.circuit.summaries(self.graph, self.scales, self.lif.unit_weight, limit, self.edge_gain))
        ranked = np.argsort(self.scales, kind="stable")[:limit]
        for slot, row in zip(ranked, rows, strict=True):
            row["last_update_tick"] = int(self.last_update[slot])
            age = self.update_age[slot]
            row["last_pre_age_ms"] = float(age) * self.lif.dt if age >= 0 else None
        return rows

    def memory(self):
        changed = self.scales != 1.0
        return {"changed": int(changed.sum()),
                "mean_change": float((1 - self.scales[changed]).mean()) if changed.any() else 0.0,
                "compartment_updates": self.group_updates.tolist()}
