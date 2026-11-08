from pathlib import Path

import numpy as np
import pytest

from flybrain_snake.anatomy import ContactAtlas, load_gain
from flybrain_snake.circuits import Circuit
from flybrain_snake.config import RunConfig
from flybrain_snake.connectome import Graph
from flybrain_snake.experiments import trainer_for
from flybrain_snake.storage import checkpoint, restore


@pytest.fixture(scope="module")
def atlas():
    data = Path("data")
    if not (data / "anatomy" / "manifest.json").is_file():
        pytest.skip("Ejecuta anatomy-prepare para probar los contactos oficiales")
    return ContactAtlas(data, Graph.load(data / "processed"))


@pytest.mark.flywire
def test_contact_id_and_anatomical_coordinates(atlas):
    assert sum(atlas.manifest["review"]) == atlas.n == 130_054_535
    assert atlas.manifest["review"][3] == 54_492_970
    assert atlas.manifest["graph_contacts"] == 54_492_922
    row = atlas.ordinal(2021)
    detail = atlas.detail(row)
    assert detail["id"] == 2021
    assert detail["neuropil"] == "AVLP_R"
    assert detail["ach"] == pytest.approx(0.9732, abs=0.0001)
    np.testing.assert_array_equal(atlas.geometry[row]["xyz"],
                                  [[672408, 244016, 74160], [672444, 244020, 74240]])
    with pytest.raises(KeyError):
        atlas.ordinal(2**32)


@pytest.mark.flywire
def test_filter_pages_and_regional_counts(atlas):
    graph = atlas.graph
    index = np.argmax(np.diff(atlas.array("pre_offsets")))
    root = int(graph.ids[index])
    complete = atlas.roots("pre", root)
    assert len(complete) > 11
    review = atlas.geometry[complete]["review"]
    reviewed = complete[review == 3]
    assert len(reviewed) > 11
    first = atlas.query(root=root, mode="pre", review=True, limit=7, offset=0)
    second = atlas.query(root=root, mode="pre", review=True, limit=7, offset=7)
    np.testing.assert_array_equal(np.concatenate((first.rows, second.rows)), reviewed[:14])
    assert first.total == second.total == len(reviewed)
    region_counts = atlas.profile(root)["pre"]
    assert sum(region_counts.values()) == len(complete)
    detail = atlas.detail(int(first.rows[0]))
    assert detail["pre_pt_root_id"] == root
    partner = int(detail["post_pt_root_id"])
    pair = atlas.query(root=root, partner=partner, mode="pre", limit=len(complete))
    assert first.rows[0] in pair.rows


@pytest.mark.flywire
def test_probability_mode_uses_atlas_and_resumes(atlas):
    graph = atlas.graph
    gain, fingerprint = load_gain(graph.data_path, graph.fingerprint, len(graph.targets))
    circuit = Circuit.from_graph(graph)
    cfg = RunConfig(synaptic_signal="contact_probabilities", decision_ms=36, outcome_ms=18)
    trainer = trainer_for(graph, circuit, cfg)
    assert trainer.sim.edge_gain is gain
    assert trainer.sim.gain_fingerprint == fingerprint
    trainer.begin()
    trainer.step()
    resumed = restore(checkpoint(trainer), graph, circuit)
    assert resumed.sim.gain_fingerprint == fingerprint
    assert trainer.step()["decision"] == resumed.step()["decision"]
    np.testing.assert_array_equal(trainer.sim.voltage, resumed.sim.voltage)
