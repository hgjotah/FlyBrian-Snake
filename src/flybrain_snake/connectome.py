"""Versioned FlyWire FAFB import and sparse, immutable graph access."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
from http.client import HTTPException
import json
import re
from pathlib import Path
import shutil
import tempfile
import time
from urllib.request import Request, urlopen

import numpy as np
import pyarrow as pa

RECORD = "https://zenodo.org/records/10676866/files/"
ANNOTATION_REVISION = "a83b2776d60d5764cef36b927f5f9679c16c47a2"
ANNOTATION_NAME = "Supplemental_file1_neuron_annotations.tsv"
EDGE_NAME = "proofread_connections_783.feather"
ID_NAME = "proofread_root_ids_783.npy"
SOURCES = {
    EDGE_NAME: (RECORD + EDGE_NAME + "?download=1", 852022274,
                "md5", "f48f972d262323a102aed49af1396b8a"),
    ID_NAME: (RECORD + ID_NAME + "?download=1", 1114168,
              "md5", "e0e6c19732fd8c7a4e39a2d170105421"),
    ANNOTATION_NAME: (
        "https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
        + ANNOTATION_REVISION + "/supplemental_files/" + ANNOTATION_NAME,
        31720298, "sha256", "b214970b55d2fbe0853bba536fdcb9e28730f4eb7ab06f600491df795da683cd"),
}
FIELDS = ("cell_type", "cell_class", "super_class", "side", "top_nt", "known_nt",
          "top_nt_conf", "flow", "nerve")
SIGN = {"acetylcholine": 1, "ach": 1, "gaba": -1, "glutamate": -1, "glut": -1}
ARRAYS = ("ids", "offsets", "targets", "counts", "regions", "sign", "xyz")


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, algorithm).hexdigest()


def download(url: str, partial: Path, expected_size: int) -> None:
    for attempt in range(6):
        offset = partial.stat().st_size
        if offset == expected_size:
            return
        headers = {"User-Agent": "FlyBrain-Snake/0.2"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=90) as response:
                if response.status == 206:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset}-") or not content_range.endswith(f"/{expected_size}"):
                        raise ValueError(f"Rango inesperado de descarga: {content_range}")
                elif response.status == 200:
                    offset = 0
                else:
                    raise ValueError(f"Respuesta inesperada: HTTP {response.status}")
                with partial.open("ab" if offset else "wb") as handle:
                    shutil.copyfileobj(response, handle, 2**20)
            if partial.stat().st_size == expected_size:
                return
        except (OSError, HTTPException) as error:
            if attempt == 5:
                raise OSError(f"No se pudo descargar {url}: {error}") from error
        time.sleep(min(2**attempt, 15))
    raise OSError(f"Descarga incompleta: {url}")


def acquire(directory: Path, local_only: bool = False) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for name, (url, size, algorithm, expected) in SOURCES.items():
        path = directory / name
        valid = path.is_file() and path.stat().st_size == size and digest(path, algorithm) == expected
        if not valid:
            if local_only:
                raise ValueError(f"Archivo ausente o corrupto: {path}")
            print(f"Descargando {name} ({size / 1e6:.1f} MB)", flush=True)
            with tempfile.NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as tmp:
                partial = Path(tmp.name)
            try:
                download(url, partial, size)
                if partial.stat().st_size != size or digest(partial, algorithm) != expected:
                    raise ValueError(f"Tamaño o hash incorrecto: {name}")
                partial.replace(path)
            finally:
                partial.unlink(missing_ok=True)
        provenance[name] = {"url": url, "bytes": size, "sha256": digest(path)}
    return provenance


@dataclass
class Graph:
    ids: np.ndarray
    offsets: np.ndarray
    targets: np.ndarray
    counts: np.ndarray
    regions: np.ndarray
    sign: np.ndarray
    xyz: np.ndarray
    annotations: dict[str, np.ndarray]
    region_names: list[str]
    manifest: dict
    data_path: Path | None = None

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def fingerprint(self) -> str:
        return self.manifest["fingerprint"]

    def index(self, root_id: int) -> int:
        index = int(np.searchsorted(self.ids, np.uint64(root_id)))
        if index >= self.n or int(self.ids[index]) != root_id:
            raise KeyError(f"ID ausente del conectoma: {root_id}")
        return index

    def select(self, *, cell_type=None, cell_class=None, nt=None, side=None, region=None):
        mask = np.ones(self.n, dtype=bool)
        for field, value in (("cell_type", cell_type), ("cell_class", cell_class),
                             ("top_nt", nt), ("side", side)):
            if value:
                mask &= np.isin(self.annotations[field], np.atleast_1d(value))
        if region:
            present = np.zeros(self.n, dtype=bool)
            matches = self.regions == self.region_names.index(region)
            edge = np.flatnonzero(matches)
            present[self.targets[edge]] = True
            present[np.searchsorted(self.offsets[1:], edge, side="right")] = True
            mask &= present
        return np.flatnonzero(mask).astype(np.int32)

    @classmethod
    def load(cls, path: Path, verify: bool = False) -> Graph:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if manifest["dataset"] != "FlyWire FAFB 783":
            raise ValueError("Se requiere FlyWire FAFB v783")
        if verify:
            claims = {k: v for k, v in manifest.items() if k != "fingerprint"}
            if hashlib.sha256(json.dumps(claims, sort_keys=True).encode()).hexdigest() != manifest["fingerprint"]:
                raise ValueError("El hash del manifiesto no coincide")
            expected_names = {"annotations.npz", *(name + ".npy" for name in ARRAYS)}
            if set(manifest["processed"]) != expected_names:
                raise ValueError("El manifiesto no describe los arrays esperados")
            for name, expected in manifest["processed"].items():
                if digest(path / name) != expected:
                    raise ValueError(f"Hash incorrecto: {name}")
        columns = {}
        with np.load(path / "annotations.npz", allow_pickle=False) as archive:
            columns.update(archive)
        arrays = {name: np.load(path / (name + ".npy"), mmap_mode="r") for name in ARRAYS}
        if (len(arrays["ids"]) != manifest["neurons"]
                or len(arrays["targets"]) != manifest["rows"]
                or int(arrays["offsets"][-1]) != manifest["rows"]):
            raise ValueError("El manifiesto y el CSR no coinciden")
        return cls(**arrays, annotations=columns, region_names=manifest["regions"], manifest=manifest, data_path=path.parent.resolve())


def index_ids(ids: np.ndarray, roots: np.ndarray) -> np.ndarray:
    roots = roots.astype(np.uint64, copy=False)
    index = np.searchsorted(ids, roots)
    if np.any(index >= len(ids)) or np.any(ids[index] != roots):
        raise ValueError("La tabla contiene un extremo que no figura en los IDs revisados")
    return index.astype(np.int32)


def read_annotations(path: Path, ids: np.ndarray):
    lookup = {int(root): index for index, root in enumerate(ids)}
    columns = {field: [""] * len(ids) for field in FIELDS}
    xyz = np.full((len(ids), 3), np.nan, dtype=np.float32)
    seen = set()
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            index = lookup.get(int(row["root_id"]))
            if index is None:
                continue
            if index in seen:
                raise ValueError(f"Anotación duplicada para {row['root_id']}")
            seen.add(index)
            for field in FIELDS:
                columns[field][index] = row.get(field, "").strip()
            for keys in (("soma_x", "soma_y", "soma_z"), ("pos_x", "pos_y", "pos_z")):
                try:
                    candidate = [float(row[key]) for key in keys]
                except (ValueError, KeyError):
                    continue
                if np.isfinite(candidate).all():
                    xyz[index] = np.asarray(candidate) * (0.004, 0.004, 0.040)
                    break
    return {key: np.asarray(value) for key, value in columns.items()}, xyz, len(seen)


def transmitter_sign(predicted: str, known: str) -> int:
    candidates = {SIGN[token] for token in re.split(r"\s*[;,]\s*", known.lower()) if token in SIGN}
    if len(candidates) == 1:
        return candidates.pop()
    return SIGN.get(predicted.lower(), 0)


def build(raw: Path, output: Path, provenance: dict) -> dict:
    ids = np.sort(np.load(raw / ID_NAME, allow_pickle=False).astype(np.uint64))
    if len(ids) != 139255 or np.any(ids[1:] <= ids[:-1]):
        raise ValueError("El snapshot 783 debe contener 139255 IDs únicos")
    annotations, xyz, annotated = read_annotations(raw / ANNOTATION_NAME, ids)
    sign = np.array([transmitter_sign(p, k) for p, k in zip(
        annotations["top_nt"], annotations["known_nt"], strict=True)], dtype=np.int8)
    region_names: list[str] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".build-") as tmp:
        staging = Path(tmp)
        with pa.memory_map(str(raw / EDGE_NAME), "r") as source:
            reader = pa.ipc.open_file(source)
            pre_key, post_key = "pre_pt_root_id", "post_pt_root_id"
            count_key = next(key for key in ("syn_count", "synapse_count", "Count", "count")
                             if key in reader.schema.names)
            region_key = "neuropil"
            n_edges = sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches))
            src = np.empty(n_edges, np.int32)
            dst = np.empty(n_edges, np.int32)
            counts = np.empty(n_edges, np.uint32)
            regions = np.empty(n_edges, np.uint16)
            cursor = 0
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                end = cursor + batch.num_rows
                for key, dest in ((pre_key, src), (post_key, dst)):
                    dest[cursor:end] = index_ids(ids, batch[key].to_numpy())
                count = batch[count_key].to_numpy()
                if np.any(count < 1) or np.any(count > np.iinfo(np.uint32).max):
                    raise ValueError("Recuento de contactos fuera de rango")
                counts[cursor:end] = count
                neuropil = batch[region_key]
                if not pa.types.is_dictionary(neuropil.type):
                    neuropil = neuropil.dictionary_encode()
                names = []
                for value in neuropil.dictionary.to_pylist():
                    value = value or "sin región"
                    if value not in region_names:
                        region_names.append(value)
                    names.append(region_names.index(value))
                indices = neuropil.indices.fill_null(-1).to_numpy()
                if np.any(indices < 0) and "sin región" not in region_names:
                    region_names.append("sin región")
                if np.any(indices < 0):
                    names.append(region_names.index("sin región"))
                regions[cursor:end] = np.asarray(names, dtype=np.uint16)[indices]
                cursor = end
        offsets = np.empty(len(ids) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(np.bincount(src, minlength=len(ids)), out=offsets[1:])
        order = np.lexsort((regions, dst, src))
        paired = np.uint64(len(ids)) * src[order].astype(np.uint64) + dst[order]
        pair_count = 1 + int(np.count_nonzero(paired[1:] != paired[:-1]))
        del paired, src
        arrays = (ids, offsets, dst[order], counts[order], regions[order], sign, xyz)
        for name, values in zip(ARRAYS, arrays, strict=True):
            np.save(staging / (name + ".npy"), values, allow_pickle=False)
        np.savez_compressed(staging / "annotations.npz", **annotations)
        hashes = {p.name: digest(p) for p in sorted(staging.iterdir())}
        manifest = {
            "format": 1, "dataset": "FlyWire FAFB 783", "neurons": len(ids), "annotated": annotated,
            "rows": n_edges, "unique_pairs": pair_count, "contacts": int(counts.sum(dtype=np.uint64)),
            "regions": region_names, "sources": provenance, "processed": hashes,
            "sign_policy": {"weights": SIGN, "curated_first": True}, "annotation_revision": ANNOTATION_REVISION,
        }
        manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if output.exists():
            raise FileExistsError(f"El destino {output} ya existe. Usa otra carpeta para reconstruir.")
        staging.rename(output)
    return manifest
