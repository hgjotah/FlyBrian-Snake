import numpy as np
import pytest

from flybrain_snake.circuits import Circuit
from flybrain_snake.config import RunConfig
from flybrain_snake.connectome import Graph
from flybrain_snake.experiments import trainer_for


@pytest.fixture
def laboratory():
    # A seven-cell synthetic fixture for unit tests only.
    n = 7
    i32 = lambda a: np.array(a, dtype=np.int32)
    offsets = np.array([0, 2, 2, 2, 2, 3, 3, 3], dtype=np.int64)
    graph = Graph(np.arange(100, 100 + n, dtype=np.uint64), offsets, i32([1, 5, 5]),
                  np.array([220, 220, 220], dtype=np.uint32), np.array([0, 1, 1], np.uint16),
                  np.array([1, 1, 0, 0, -1, 1, 1], np.int8), np.zeros((n, 3), np.float32),
                  {"cell_type": np.array(["KC", "MBON", "PAM", "PPL1", "GABA", "DNa", "DNa"])},
                  ["MB_ML_L", "MB_PED_L"], {"dataset": "synthetic test", "fingerprint": "synthetic"})
    groups = {name: i32([0]) for name in (
        "food_left", "food_right", "food_front", "food_back", "wall_left", "wall_right", "loom_left", "loom_right",
        "tag_left", "tag_right", "tag_forward", "tag_back", "tag_wall_left", "tag_wall_front",
        "tag_wall_right", "heading_up", "heading_right", "heading_down", "heading_left",
        "conditioning_A", "conditioning_B", "KC")}
    groups.update({"MBON": i32([1]), "PAM": i32([2]), "PPL1": i32([3]), "DAN": i32([2, 3]),
                   "appetitive": i32([2]), "aversive": i32([3]),
                   "motor_left": i32([5]), "motor_right": i32([6])})
    circuit = Circuit(groups, ("a", "b"), i32([0, 1, 2]), i32([2, 3]),
                      np.array([1., 1.], dtype=np.float32), i32([0, 1, 2]), i32([0, 1]),
                      i32([0, 0]), i32([1, 5]), i32([0, 1, -1]))
    cfg = RunConfig(seed=4, sensory_hz=0.0, max_steps=6)
    return trainer_for(graph, circuit, cfg)
