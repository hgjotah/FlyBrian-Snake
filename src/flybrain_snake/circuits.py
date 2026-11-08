"""Circuit definitions resolved from the pinned FlyWire annotations."""
from dataclasses import dataclass
import numpy as np

from .connectome import Graph


@dataclass
class Circuit:
    groups: dict[str, np.ndarray]
    compartment_names: tuple[str, ...]
    dan_ptr: np.ndarray
    dan_indices: np.ndarray
    dan_strengths: np.ndarray
    plastic_ptr: np.ndarray
    edges: np.ndarray
    pre: np.ndarray
    post: np.ndarray
    slots: np.ndarray

    @classmethod
    def from_graph(cls, graph: Graph):
        a = graph.annotations
        groups = {}
        groups["KC"] = graph.select(cell_class="Kenyon_Cell")
        for name, prefix in (("MBON", "MBON"), ("PAM", "PAM"), ("PPL1", "PPL1")):
            groups[name] = np.flatnonzero(np.char.startswith(a["cell_type"], prefix)).astype(np.int32)
        groups["DAN"] = np.union1d(groups["PAM"], groups["PPL1"])
        for side in ("left", "right"):
            groups[f"motor_{side}"] = graph.select(cell_type=["DNa01", "DNa02"], side=side)
            groups[f"food_{side}"] = graph.select(cell_type="LC10a", side=side)
            groups[f"wall_{side}"] = graph.select(cell_type="LC4", side=side)
            groups[f"loom_{side}"] = graph.select(cell_type="LC6", side=side)
        groups["food_front"] = graph.select(cell_type="LC10d")
        groups["food_back"] = graph.select(cell_type="LPLC2")
        groups["conditioning_A"] = graph.select(cell_type="KCg-m", side="left")[::5]
        groups["conditioning_B"] = graph.select(cell_type="KCg-m", side="left")[1::5]
        # Sparse KC tags make the synthetic cues available to the memory circuit.
        tagged = graph.select(cell_type="KCg-m")
        cues = ("forward", "back", "left", "right", "wall_left", "wall_right", "wall_front")
        for i, cue in enumerate(cues):
            groups["tag_" + cue] = tagged[i::len(cues)][:64]
        for i, name in enumerate(("up", "right", "down", "left")):
            groups["heading_" + name] = tagged[512 + i * 24:536 + i * 24]

        group_names, dans, strengths, edge_sets, pres, posts = [], [], [], [], [], []
        dan_ptr, plastic_ptr = [0], [0]
        slots = np.full(len(graph.targets), -1, np.int32)
        mbon_by_kind = (("appetitive", "PAM01", "MBON01", "γ5 proxy", "MB_ML"),
                        ("aversive", "PPL101", "MBON11", "γ1/pedc proxy", "MB_PED"))
        gamma = np.isin(a["cell_type"], ["KCg-m", "KCg-d", "KCg-s1", "KCg-s2", "KCg-s3"])
        for kind, dan_type, mbon_type, compartment, neuropil in mbon_by_kind:
            dan = graph.select(cell_type=dan_type)
            mbon = graph.select(cell_type=mbon_type)
            candidates = np.flatnonzero(np.isin(graph.targets, mbon))
            sources = np.searchsorted(graph.offsets[1:], candidates, side="right").astype(np.int32)
            eligible = gamma[sources] & (graph.sign[sources] > 0)
            for side in ("L", "R"):
                region = graph.region_names.index(f"{neuropil}_{side}")
                matching = eligible & (graph.regions[candidates] == region)
                local = candidates[matching]
                contacts = np.array([
                    graph.counts[graph.offsets[d]:graph.offsets[d + 1]][
                        graph.regions[graph.offsets[d]:graph.offsets[d + 1]] == region].sum()
                    for d in dan
                ], dtype=np.float32)
                connected = contacts > 0
                if not connected.any() or not len(local):
                    raise ValueError(f"Circuito de memoria ausente: {compartment} {side}")
                group_names.append(f"{compartment} {side}")
                dans.append(dan[connected])
                strengths.append(contacts[connected] / contacts.sum())
                edge_sets.append(local)
                pres.append(sources[matching])
                posts.append(graph.targets[local])
                dan_ptr.append(dan_ptr[-1] + int(connected.sum()))
                plastic_ptr.append(plastic_ptr[-1] + len(local))
            groups[kind] = dan
        for name, indices in groups.items():
            if not len(indices):
                raise ValueError(f"Población vacía: {name}")
        edges = np.concatenate(edge_sets)
        slots[edges] = np.arange(len(edges), dtype=np.int32)
        return cls(groups, tuple(group_names), np.array(dan_ptr), np.concatenate(dans), np.concatenate(strengths),
                   np.array(plastic_ptr), edges, np.concatenate(pres), np.concatenate(posts), slots)

    def summaries(self, graph: Graph, scales: np.ndarray, weight: float, limit: int | None = None, gains=None):
        ranked = np.argsort(scales, kind="stable")
        for slot in ranked[:limit]:
            edge = self.edges[slot]
            group = np.searchsorted(self.plastic_ptr[1:], slot, side="right")
            baseline = float(graph.counts[edge]) * weight
            if gains is not None and len(gains):
                baseline *= float(gains[edge])
            yield {
                "pre_id": str(graph.ids[self.pre[slot]]), "post_id": str(graph.ids[self.post[slot]]),
                "pre_type": str(graph.annotations["cell_type"][self.pre[slot]]),
                "post_type": str(graph.annotations["cell_type"][self.post[slot]]),
                "region": graph.region_names[graph.regions[edge]], "compartment": self.compartment_names[group],
                "edge": int(edge), "base_mv": baseline, "current_mv": baseline * float(scales[slot]),
                "change_pct": (float(scales[slot]) - 1.0) * 100.0,
            }
