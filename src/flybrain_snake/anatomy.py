"""Disk-backed FlyWire contact atlas. The original Feather retains every field."""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import json
from pathlib import Path
import tempfile

import numpy as np
import pyarrow as pa

from .connectome import EDGE_NAME, ID_NAME, Graph, SOURCES, digest, index_ids, transmitter_sign

CONTACTS = "flywire_synapses_783.feather"
COUNTS = {side: f"per_neuron_neuropil_count_{side}_783.feather" for side in ("pre", "post")}
ARCHIVE = {
    CONTACTS: (9492998242, "f8f1b97c9d4b0ea9b4c8b287f6b99091"),
    COUNTS["pre"]: (16853770, "90fcdb42c1ba05ed92820840fa1e6ba0"),
    COUNTS["post"]: (233843050, "bb5999f10920ade803d9f37097a43a56"),
    **{name: (SOURCES[name][1], SOURCES[name][3]) for name in (EDGE_NAME, ID_NAME)},
}
MISSING_COORD = np.iinfo(np.int32).min
NT = ("ach", "gaba", "glut", "da", "oct", "ser")
XYZ = tuple(f"{end}_pt_position_{axis}" for end in ("pre", "post") for axis in "xyz")
GEOMETRY = np.dtype([("xyz", np.int32, (2, 3)), ("region", np.uint8),
                     ("review", np.uint8), ("transmitter", np.uint8)])


def locate(data: Path, name: str, source: Path | None = None) -> Path:
    folders = ([source] if source is not None else []) + [data.parent, data / "raw", data]
    for folder in folders:
        file = folder / name
        if file.is_file() and file.stat().st_size == ARCHIVE[name][0]:
            return file.resolve()
    raise FileNotFoundError(f"No se encuentra {name}. Indica la carpeta con --source.")


def columns(reader, names):
    return pa.ipc.IpcReadOptions(included_fields=sorted(reader.schema.get_field_index(n) for n in names))


def reviewed(ids: np.ndarray, values: np.ndarray):
    positions = np.searchsorted(ids, values.astype(ids.dtype, copy=False))
    positions = np.minimum(positions, len(ids) - 1)
    return positions, ids[positions] == values


def region_codes(array, names: list[str]):
    dictionary = array.dictionary_encode()
    mapping = []
    for value in dictionary.dictionary.to_pylist() + [None]:
        value = value or "sin región"
        if value not in names:
            if len(names) >= 256:
                raise ValueError("Demasiados neuropilos para el índice")
            names.append(value)
        mapping.append(names.index(value))
    return np.asarray(mapping, dtype=np.uint8)[dictionary.indices.fill_null(-1).to_numpy()]


def mmap_array(folder, name, dtype, shape):
    return np.lib.format.open_memmap(folder / (name + ".npy"), mode="w+", dtype=dtype, shape=(shape,))


def write_root_hits(staging, side, keys, size, neurons):
    view = keys[:size]
    view.sort()
    offsets = np.searchsorted(view, np.arange(neurons + 1, dtype=np.uint64) << 32)
    np.save(staging / f"{side}_offsets.npy", offsets)
    hits = mmap_array(staging, f"{side}_hits", np.uint32, size)
    for start in range(0, size, 2**20):
        hits[start:start + 2**20] = view[start:start + 2**20].astype(np.uint32)
    hits.flush()


def build_contacts(file, staging, graph):
    with pa.memory_map(str(file), "r") as source:
        header = pa.ipc.open_file(source)
        reader = pa.ipc.open_file(source, options=columns(header, (
            "id", "pre_pt_root_id", "post_pt_root_id", "neuropil", *XYZ, *NT,
        )))
        offsets = [0]
        # Standard Arrow batches in the pinned archive are uniformly sized except the last.
        last = reader.num_record_batches - 1
        capacity = reader.get_batch(0).num_rows * last + reader.get_batch(last).num_rows
        geometry = mmap_array(staging, "contacts", GEOMETRY, capacity)
        roots = [mmap_array(staging, side + "_tmp", np.uint64, capacity) for side in ("pre", "post")]
        synapses = mmap_array(staging, "id_index", np.uint64, capacity)
        id_sorted = True
        previous_id = -1
        cursor = [0, 0]
        names = list(graph.region_names)
        statuses = np.zeros(4, dtype=np.int64)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            start = offsets[-1]
            stop = start + batch.num_rows
            offsets.append(stop)
            ordinal = np.arange(start, stop, dtype=np.uint64)
            original = batch["id"].to_numpy()
            if original.min() < 0 or original.max() >= 2**32:
                raise ValueError("ID sináptico fuera del rango del archivo v783")
            if id_sorted:
                id_sorted = bool(previous_id < original[0] and np.all(original[1:] > original[:-1]))
                previous_id = int(original[-1])
            synapses[start:stop] = (original.astype(np.uint64) << 32) | ordinal
            rows = geometry[start:stop]
            rows["review"] = 0
            for j, side in enumerate(("pre", "post")):
                values = batch[f"{side}_pt_root_id"].fill_null(0).to_numpy()
                idx, valid = reviewed(graph.ids, values)
                rows["review"] |= valid.astype(np.uint8) << j
                end = cursor[j] + int(valid.sum())
                roots[j][cursor[j]:end] = (idx[valid].astype(np.uint64) << 32) | ordinal[valid]
                cursor[j] = end
                for dim, axis in enumerate("xyz"):
                    rows["xyz"][:, j, dim] = batch[f"{side}_pt_position_{axis}"].fill_null(MISSING_COORD).to_numpy()
            rows["region"] = region_codes(batch["neuropil"], names)
            probs = np.stack([batch[nt].to_numpy(zero_copy_only=False) for nt in NT])
            valid = np.isfinite(probs).all(axis=0) & (probs.max(axis=0) > 0)
            rows["transmitter"] = np.where(valid, probs.argmax(axis=0), len(NT))
            statuses += np.bincount(rows["review"], minlength=4)
            if (i + 1) % 250 == 0:
                print(f"Contactos procesados: {stop:,}", flush=True)
        geometry.flush()
        for side, keys, count in zip(("pre", "post"), roots, cursor, strict=True):
            print(f"Indexando extremos {side} revisados: {count:,}", flush=True)
            write_root_hits(staging, side, keys, count, graph.n)
        del keys, roots
        print("Indexando IDs de todos los contactos", flush=True)
        if not id_sorted:
            synapses.sort()
        synapses.flush()
        del synapses
    for side in ("pre", "post"):
        (staging / f"{side}_tmp.npy").unlink()
    return names, offsets, statuses.tolist()


def build_counts(file, side, staging, graph, names):
    counts = np.zeros((graph.n, 256), dtype=np.uint32)
    with pa.memory_map(str(file), "r") as source:
        reader = pa.ipc.open_file(source)
        limits = []
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            ids = batch[f"{side}_pt_root_id"].to_numpy()
            limits.append((int(ids.min()), int(ids.max())))
            idx, valid = reviewed(graph.ids, ids)
            if valid.any():
                regions = region_codes(batch["neuropil"], names)
                np.add.at(counts, (idx[valid], regions[valid]), batch["count"].to_numpy()[valid])
        np.save(staging / f"{side}_counts.npy", counts[:, :len(names)])
        np.save(staging / f"{side}_count_batches.npy", np.asarray(limits, np.int64))


def build_gains(file, staging, graph, names):
    """Mean expected fast sign, with the curated presynaptic assignment taking priority."""
    n = len(graph.targets)
    pre = np.empty(n, dtype=np.int32)
    post = np.empty(n, dtype=np.int32)
    region = np.empty(n, dtype=np.uint8)
    count = np.empty(n, dtype=np.uint32)
    gain = np.empty(n, dtype=np.float32)
    curated = np.array([transmitter_sign("", known) for known in graph.annotations["known_nt"]])
    cursor = 0
    with pa.memory_map(str(file), "r") as source:
        reader = pa.ipc.open_file(source)
        for i in range(reader.num_record_batches):
            batch = reader.get_batch(i)
            end = cursor + batch.num_rows
            a = index_ids(graph.ids, batch["pre_pt_root_id"].to_numpy())
            pre[cursor:end] = a
            post[cursor:end] = index_ids(graph.ids, batch["post_pt_root_id"].to_numpy())
            region[cursor:end] = region_codes(batch["neuropil"], names)
            count[cursor:end] = batch["syn_count"].to_numpy()
            expected = batch["ach_avg"].to_numpy(zero_copy_only=False) - batch["gaba_avg"].to_numpy(zero_copy_only=False) - batch["glut_avg"].to_numpy(zero_copy_only=False)
            gain[cursor:end] = np.where(curated[a] != 0, curated[a],
                                       np.where(np.isfinite(expected), np.clip(expected, -1, 1), graph.sign[a]))
            cursor = end
    order = np.lexsort((region, post, pre))
    if (cursor != n or not np.array_equal(np.bincount(pre, minlength=graph.n), np.diff(graph.offsets))
            or not np.array_equal(post[order], graph.targets)
            or not np.array_equal(region[order], graph.regions)
            or not np.array_equal(count[order], graph.counts)):
        raise ValueError("Los pesos anatómicos no coinciden con el orden del CSR")
    np.save(staging / "gain.npy", gain[order])


def prepare(data: Path, graph: Graph, source: Path | None = None):
    output = data / "anatomy"
    if output.exists():
        return ContactAtlas(data, graph).manifest
    files = {name: locate(data, name, source) for name in ARCHIVE}
    for name, file in files.items():
        print(f"Verificando {name}", flush=True)
        if digest(file, "md5") != ARCHIVE[name][1]:
            raise ValueError(f"MD5 incorrecto: {file}")
    if not np.array_equal(np.sort(np.load(files[ID_NAME], allow_pickle=False)), graph.ids):
        raise ValueError("Los IDs revisados no coinciden con el CSR")
    with tempfile.TemporaryDirectory(dir=data, prefix=".anatomy-", ignore_cleanup_errors=True) as folder:
        staging = Path(folder)
        build_gains(files[EDGE_NAME], staging, graph, list(graph.region_names))
        names, offsets, statuses = build_contacts(files[CONTACTS], staging, graph)
        for side, name in COUNTS.items():
            build_counts(files[name], side, staging, graph, names)
        manifest = {"version": 1, "graph": graph.fingerprint, "contacts": offsets[-1],
                    "review": statuses, "graph_contacts": graph.manifest["contacts"],
                    "regions": names, "batch_offsets": offsets,
                    "gain_sha256": digest(staging / "gain.npy"),
                    "source": {name: {"bytes": ARCHIVE[name][0], "md5": ARCHIVE[name][1],
                                      "path": str(file)} for name, file in files.items()}}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(output)
    return manifest


@lru_cache(maxsize=4)
def load_gain(data: Path, graph_id: str, n: int):
    folder = data / "anatomy"
    file = folder / "gain.npy"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if manifest["version"] != 1 or manifest["graph"] != graph_id:
        raise ValueError("Ganancias de contactos incompatibles con el grafo")
    fingerprint = digest(file)
    if fingerprint != manifest["gain_sha256"]:
        raise ValueError("Hash incorrecto en las ganancias de contactos")
    gain = np.load(file, mmap_mode="r", allow_pickle=False)
    if gain.shape != (n,) or gain.dtype != np.float32:
        raise ValueError("Forma o tipo de las ganancias de contactos incorrectos")
    return gain, fingerprint


@dataclass
class ContactView:
    rows: np.ndarray
    geometry: np.ndarray
    total: int


class ContactAtlas:
    def __init__(self, data: Path, graph: Graph):
        self.data, self.graph = data, graph
        self.folder = data / "anatomy"
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["version"] != 1 or self.manifest["graph"] != graph.fingerprint:
            raise ValueError("Atlas incompatible con el conectoma activo")
        self.n = self.manifest["contacts"]
        self.names = self.manifest["regions"]
        self.batch_offsets = np.array(self.manifest["batch_offsets"])
        self.geometry = self.array("contacts")[:self.n]

    def array(self, name):
        return np.load(self.folder / (name + ".npy"), mmap_mode="r", allow_pickle=False)

    def source(self, name):
        origin = Path(self.manifest["source"][name]["path"])
        if origin.is_file() and origin.stat().st_size == ARCHIVE[name][0]:
            return origin
        return locate(self.data, name)

    def roots(self, side, root):
        i = self.graph.index(root)
        offsets = self.array(f"{side}_offsets")
        return self.array(f"{side}_hits")[offsets[i]:offsets[i + 1]]

    def profile(self, root):
        try:
            index = self.graph.index(root)
        except KeyError:
            index = None
        result = {}
        for side in ("pre", "post"):
            if index is not None:
                values = self.array(f"{side}_counts")[index]
                result[side] = {self.names[i]: int(values[i]) for i in np.flatnonzero(values)}
            else:
                # Neuropil summaries also contain roots outside the reviewed graph.
                bounds = self.array(f"{side}_count_batches")
                candidates = np.flatnonzero((bounds[:, 0] <= root) & (root <= bounds[:, 1]))
                values = {}
                with pa.memory_map(str(self.source(COUNTS[side])), "r") as f:
                    reader = pa.ipc.open_file(f)
                    for i in candidates:
                        b = reader.get_batch(int(i))
                        matches = np.flatnonzero(b[f"{side}_pt_root_id"].to_numpy() == root)
                        for row in matches:
                            region = b["neuropil"][int(row)].as_py() or "sin región"
                            values[region] = values.get(region, 0) + b["count"][int(row)].as_py()
                result[side] = values
        return result

    def ordinal(self, synapse_id):
        if not 0 <= synapse_id < 2**32:
            raise KeyError(f"ID sináptico ausente: {synapse_id}")
        keys = self.array("id_index")
        pos = int(np.searchsorted(keys, np.uint64(synapse_id << 32)))
        if pos >= len(keys) or int(keys[pos]) >> 32 != synapse_id:
            raise KeyError(f"ID sináptico ausente: {synapse_id}")
        return int(keys[pos]) & 0xFFFFFFFF

    def detail(self, row):
        if not 0 <= row < self.n:
            raise IndexError(row)
        i = int(np.searchsorted(self.batch_offsets, row, side="right")) - 1
        j = row - int(self.batch_offsets[i])
        with pa.memory_map(str(self.source(CONTACTS)), "r") as f:
            batch = pa.ipc.open_file(f).get_batch(i)
            return {name: batch[name][j].as_py() for name in batch.schema.names}

    def connection(self, detail):
        graph = self.graph
        try:
            pre = graph.index(detail["pre_pt_root_id"])
            post = graph.index(detail["post_pt_root_id"])
            region = graph.region_names.index(detail["neuropil"] or "sin región")
        except (KeyError, TypeError, ValueError):
            return None
        start, stop = graph.offsets[pre:pre + 2]
        targets = graph.targets[start:stop]
        left, right = np.searchsorted(targets, post), np.searchsorted(targets, post, side="right")
        matches = np.flatnonzero(graph.regions[start + left:start + right] == region)
        if not len(matches):
            return None
        edge = start + left + matches[0]
        gain, _ = load_gain(self.data, graph.fingerprint, len(graph.targets))
        return int(graph.counts[edge]), int(graph.sign[pre]), float(gain[edge])

    @cached_property
    def outgoing_preview(self):
        """Small in-memory copy for animation, indexed by compact presynaptic neuron."""
        from .storage import atomic_bytes
        import io

        file = self.folder / "outgoing_preview.npy"
        shape = (self.graph.n, 8, 3)
        if file.is_file():
            points = np.load(file, allow_pickle=False)
            if points.shape == shape and points.dtype == np.float32:
                return points

        offsets, hits = self.array("pre_offsets"), self.array("pre_hits")
        count = np.diff(offsets)
        spans = np.minimum(count, shape[1])
        j = np.arange(shape[1])
        local = j * count[:, None] // np.maximum(spans[:, None], 1)
        nodes, slots = np.nonzero(j < spans[:, None])
        rows = hits[offsets[nodes] + local[nodes, slots]]
        order = np.argsort(rows)
        geometry = self.geometry[rows[order]]
        xyz = geometry["xyz"][:, 1]
        valid = (geometry["review"] == 3) & (xyz != MISSING_COORD).all(axis=1)
        selected = order[valid]
        points = np.full(shape, np.nan, dtype=np.float32)
        points[nodes[selected], slots[selected]] = xyz[valid].astype(np.float32) / 1000
        buffer = io.BytesIO()
        np.save(buffer, points, allow_pickle=False)
        atomic_bytes(file, buffer.getvalue())
        return points

    def active_contacts(self, nodes):
        """Bounded display sample of reviewed outgoing contacts from firing neurons."""
        sample = np.linspace(0, len(nodes) - 1, min(len(nodes), 48), dtype=np.intp)
        points = self.outgoing_preview[nodes[sample]].reshape(-1, 3)
        return points[np.isfinite(points).all(axis=1)]

    def scan_roots(self, root, partner, mode):
        """Fallback for roots outside the reviewed graph, without loading all columns."""
        a, b = "pre_pt_root_id", "post_pt_root_id"
        matches = []
        with pa.memory_map(str(self.source(CONTACTS)), "r") as source:
            header = pa.ipc.open_file(source)
            reader = pa.ipc.open_file(source, options=columns(header, (a, b)))
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                pre = batch[a].fill_null(0).to_numpy()
                post = batch[b].fill_null(0).to_numpy()
                incoming, outgoing = post == root, pre == root
                if partner is not None:
                    incoming &= pre == partner
                    outgoing &= post == partner
                selected = incoming if mode == "post" else outgoing if mode == "pre" else incoming | outgoing
                if selected.any():
                    matches.append((np.flatnonzero(selected) + self.batch_offsets[i]).astype(np.uint32))
        return np.concatenate(matches) if matches else np.empty(0, np.uint32)

    def query(self, *, region=None, root=None, partner=None, mode="both",
              transmitter=None, review=None, limit=6000, offset=None):
        if (limit < 1 or (offset is not None and offset < 0)
                or mode not in ("pre", "post", "both") or (partner is not None and root is None)):
            raise ValueError("Consulta de contactos inválida")
        subset = None
        if root is not None:
            try:
                incoming = self.roots("post", root) if mode != "pre" else np.empty(0, np.uint32)
                outgoing = self.roots("pre", root) if mode != "post" else np.empty(0, np.uint32)
                if partner is not None:
                    incoming = np.intersect1d(incoming, self.roots("pre", partner), assume_unique=True)
                    outgoing = np.intersect1d(outgoing, self.roots("post", partner), assume_unique=True)
                subset = np.union1d(incoming, outgoing)
            except KeyError:
                subset = self.scan_roots(root, partner, mode)
        region_id = None if region is None else self.names.index(region)
        nt_id = None if transmitter is None else (*NT, "?").index(transmitter)
        size = self.n if subset is None else len(subset)
        def blocks():
            for start in range(0, size, 2**20):
                stop = min(start + 2**20, size)
                rows = self.geometry[start:stop] if subset is None else self.geometry[subset[start:stop]]
                mask = np.ones(stop - start, dtype=bool)
                if region_id is not None:
                    mask &= rows["region"] == region_id
                if nt_id is not None:
                    mask &= rows["transmitter"] == nt_id
                if review is not None:
                    mask &= rows["review"] == 3 if review else rows["review"] != 3
                yield start, mask
        def requested_ranks(total):
            if offset is not None:
                return np.arange(offset, min(total, offset + limit), dtype=np.int64)
            return np.linspace(0, total - 1, min(total, limit), dtype=np.int64)

        if region_id is None and nt_id is None and review is None:
            total = size
            selected = requested_ranks(total).astype(np.uint32)
        else:
            total = sum(np.count_nonzero(mask) for _, mask in blocks())
            ranks = requested_ranks(total)
            selected = np.empty(len(ranks), dtype=np.uint32)
            seen = cursor = 0
            for start, mask in blocks():
                matches = np.flatnonzero(mask)
                end = int(np.searchsorted(ranks, seen + len(matches)))
                selected[cursor:end] = start + matches[ranks[cursor:end] - seen]
                cursor = end
                seen += len(matches)
                if cursor == len(ranks):
                    break
        if subset is not None:
            selected = subset[selected]
        return ContactView(selected, self.geometry[selected], int(total))
