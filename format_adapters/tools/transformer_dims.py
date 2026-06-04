"""Probe transformer architecture dims from a safetensors file header.

Used by HFLayoutAdapter when staging a standalone-file model: bundled
config.json carries reasonable defaults but variants of the same arch
(e.g. Klein 4B vs 9B, both `Flux2Transformer2DModel`) differ in block
counts and hidden dim. The probe reads tensor shapes/keys from the
source file (and any sibling shards) and emits a dict of overrides to
merge into the bundled config so the C engine allocates correctly-sized
blocks at load.

Returns None when probing can't be done cleanly — the caller falls back
to the bundled defaults in that case.
"""
from __future__ import annotations

import json
import os
import re
from typing import Optional

from .safetensors_io import read_safetensors_header


def _all_related_safetensors(path: str) -> list[str]:
    """Return [path] for a single-file model, or all shard paths when
    `path` is one shard of a sharded model. Detects shards via either
    `<path>.index.json` or `<prefix>-NNNNN-of-NNNNN.safetensors` siblings.
    """
    if not path or not os.path.isfile(path):
        return []
    d = os.path.dirname(path)
    idx = path + ".index.json"
    if os.path.exists(idx):
        try:
            with open(idx) as f:
                meta = json.load(f)
            shards = sorted(set(meta.get("weight_map", {}).values()))
            return [os.path.join(d, s) for s in shards
                    if os.path.isfile(os.path.join(d, s))]
        except Exception:
            pass
    name = os.path.basename(path)
    m = re.match(r"^(.+?)-(\d{5})-of-(\d{5})\.safetensors$", name)
    if m:
        prefix = m.group(1)
        try:
            shards = sorted(
                f for f in os.listdir(d)
                if re.match(rf"^{re.escape(prefix)}-\d{{5}}-of-\d{{5}}\.safetensors$", f)
            )
            return [os.path.join(d, f) for f in shards]
        except OSError:
            pass
    return [path]


def probe_transformer_dims(path: str, arch: str) -> Optional[dict]:
    """Return a dict of architecture-config overrides probed from `path`.

    Currently handles `Flux2Klein` (Klein 4B vs 9B). Returns None for
    archs without size variants worth probing or when the file lacks
    enough information to derive the dims cleanly.

    For Klein, derives:
      - num_layers           ← max(transformer_blocks.N)+1 across all shards
      - num_single_layers    ← max(single_transformer_blocks.N)+1
      - num_attention_heads  ← context_embedder out-dim / 128
      - joint_attention_dim  ← context_embedder in-dim

    Reads BF16 weight (`.weight`) when present, else QF SVDQ packed
    (`._qweight`) since QF SVDQ keeps `[out, joint_attn]` shape.
    """
    if arch != "Flux2Klein":
        return None

    paths = _all_related_safetensors(path)
    if not paths:
        return None

    keys: dict = {}
    for p in paths:
        try:
            h = read_safetensors_header(p)
        except Exception:
            continue
        for k, v in h.items():
            if not k.startswith("__"):
                keys[k] = v
    if not keys:
        return None

    dbl = sgl = -1
    for k in keys:
        m = re.match(r"^transformer_blocks\.(\d+)\.", k)
        if m:
            dbl = max(dbl, int(m.group(1)))
        m2 = re.match(r"^single_transformer_blocks\.(\d+)\.", k)
        if m2:
            sgl = max(sgl, int(m2.group(1)))

    dim = joint_attn_dim = None
    for k in ("context_embedder.weight", "context_embedder._weight",
              "context_embedder._qweight"):
        info = keys.get(k)
        if info and "shape" in info and len(info["shape"]) == 2:
            dim, joint_attn_dim = info["shape"]
            break

    head_dim = 128
    if dim is None or dbl < 0 or sgl < 0:
        return None
    if dim % head_dim != 0:
        return None

    return {
        "num_attention_heads":  dim // head_dim,
        "num_layers":           dbl + 1,
        "num_single_layers":    sgl + 1,
        "joint_attention_dim":  joint_attn_dim,
    }
