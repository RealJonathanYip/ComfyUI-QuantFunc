"""Cheap safetensors header reads.

All functions here open the file, read only the 8-byte length + JSON header,
and close the file. They never read tensor data, so they're safe to run on
multi-gigabyte models in a UI dropdown handler.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Iterator


def read_safetensors_header(path: str | Path) -> dict:
    """Return the full header dict including __metadata__ and tensor entries.

    Header layout (per safetensors spec):
        [8 bytes: little-endian uint64 = JSON length N]
        [N bytes: UTF-8 JSON]
        [tensor data...]
    """
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8"))


def read_safetensors_metadata(path: str | Path) -> dict:
    """Return only the __metadata__ map (string → string)."""
    h = read_safetensors_header(path)
    return h.get("__metadata__", {}) or {}


def read_safetensors_keys(path: str | Path) -> Iterator[str]:
    """Yield all tensor keys without loading any weight data."""
    h = read_safetensors_header(path)
    for k in h.keys():
        if k != "__metadata__":
            yield k


def read_first_tensor_dtype(path: str | Path) -> str:
    """Return the dtype string of the first tensor in the header.

    safetensors dtypes are strings like "F16", "BF16", "F8_E4M3", "F8_E5M2",
    "F4_E2M1", "I8", "U8", "F32".  Returns empty string if no tensors.
    """
    h = read_safetensors_header(path)
    for k, info in h.items():
        if k == "__metadata__":
            continue
        return info.get("dtype", "")
    return ""


def count_tensors_by_dtype(path: str | Path) -> dict:
    """Return a {dtype: count} histogram over all tensors in the file."""
    h = read_safetensors_header(path)
    counts: dict[str, int] = {}
    for k, info in h.items():
        if k == "__metadata__":
            continue
        dt = info.get("dtype", "")
        counts[dt] = counts.get(dt, 0) + 1
    return counts


def has_keys_starting_with(path: str | Path, prefixes: list[str]) -> set[str]:
    """Return the subset of `prefixes` that have at least one matching key.

    Useful for detecting bundled-checkpoint structure (e.g. testing whether
    `model.diffusion_model.`, `text_encoders.`, `vae.` all coexist).
    """
    found: set[str] = set()
    h = read_safetensors_header(path)
    for k in h.keys():
        if k == "__metadata__":
            continue
        for p in prefixes:
            if p in found:
                continue
            if k.startswith(p):
                found.add(p)
        if len(found) == len(prefixes):
            break
    return found
