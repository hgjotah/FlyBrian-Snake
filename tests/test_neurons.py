import numpy as np
import pytest

from flybrain_snake.connectome import index_ids, transmitter_sign
from flybrain_snake.simulator import Stimulus


def test_ids_above_float_precision():
    roots = np.array([720575940620900445, 720575940620900446, 720575940620900447], np.uint64)
    assert index_ids(roots, roots.astype(np.int64)).tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        index_ids(roots, np.array([roots[-1] + 3], np.uint64))
    assert transmitter_sign("dopamine", "acetylcholine; sNPF; acetylcholine, sNPF") == 1
    assert transmitter_sign("gaba", "") == -1
    assert transmitter_sign("serotonin", "") == 0


def inject(sim, monkeypatch, events, ms, **kwargs):
    t = np.array([e[0] for e in events], np.int32)
    n = np.array([e[1] for e in events], np.int32)
    monkeypatch.setattr(sim, "_events", lambda *_: (t, n))
    return sim.advance(ms, **kwargs)


def test_propagation_delay_and_sign(laboratory, monkeypatch):
    s = laboratory.sim
    assert not s.advance(5).any()
    before = inject(s, monkeypatch, [(0, 0)], 1)
    assert before[0] == 1 and before[1] == 0
    after = inject(s, monkeypatch, [], 15)
    assert after[1] == 1 and after[5] == 1
    s.reset()
    assert inject(s, monkeypatch, [(0, 4)], 5)[4] == 1
    assert s.drive[5] < 0 and s.voltage[5] < s.lif.rest


def test_eligibility_freeze_locality_and_time(laboratory, monkeypatch):
    s = laboratory.sim
    inject(s, monkeypatch, [(0, 2)], 20, learn=True)
    np.testing.assert_array_equal(s.scales, [1, 1])
    s.reset()
    inject(s, monkeypatch, [(0, 0)], 14)
    inject(s, monkeypatch, [(0, 2)], 4, learn=False)
    np.testing.assert_array_equal(s.scales, [1, 1])
    inject(s, monkeypatch, [(0, 2)], 4, learn=True)
    near = float(s.scales[0])
    assert s.learning.min_scale < near < 1
    assert s.scales[1] == 1
    assert s.update_age[0] > 0
    s.reset()
    s.reset_memory()
    inject(s, monkeypatch, [(0, 0)], 150)
    inject(s, monkeypatch, [(0, 2)], 4, learn=True)
    assert near < s.scales[0] < 1
    for _ in range(100):
        s.reset()
        inject(s, monkeypatch, [(0, 0), (110, 1), (140, 2)], 20, learn=True)
    assert s.learning.min_scale <= s.scales[0] < .25
    assert s.scales[1] == 1


def test_brian2_cpu_reference(laboratory, monkeypatch):
    b = pytest.importorskip("brian2")
    b.start_scope()
    b.prefs.codegen.target = "numpy"
    b.defaultclock.dt = .1 * b.ms
    neurons = b.NeuronGroup(7, """dv/dt = (-52*mV-v+g)/(20*ms) : volt (unless refractory)
dg/dt = -g/(5*ms) : volt (unless refractory)""", threshold="v > -45*mV",
                            reset="v = -52*mV; g = 0*mV", refractory=2.2 * b.ms,
                            method="exact")
    neurons.v = -52 * b.mV
    links = b.Synapses(neurons, neurons, "w : volt", on_pre="g_post += w", delay=1.8 * b.ms)
    links.connect(i=[0, 0, 4], j=[1, 5, 5])
    links.w = np.array([220, 220, -220]) * .275 * b.mV
    generator = b.SpikeGeneratorGroup(2, [0, 1], [0., 12.] * b.ms)
    inputs = b.Synapses(generator, neurons, on_pre="v_post += 68.75*mV")
    inputs.connect(i=[0, 1], j=[0, 4])
    watch = b.SpikeMonitor(neurons)
    b.Network(neurons, links, generator, inputs, watch).run(25 * b.ms)
    s = laboratory.sim
    actual = inject(s, monkeypatch, [(0, 0), (120, 4)], 25)
    np.testing.assert_array_equal(actual, watch.count[:])
    np.testing.assert_allclose(s.voltage, neurons.v[:] / b.mV, atol=3e-4, rtol=0)
