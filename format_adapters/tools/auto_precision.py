"""Auto-derive a precision_config from a transformer file.

Lookup priority (each step is a no-op if the previous one already produced
a non-empty map):

  1. `transformer.precision_map` in safetensors metadata (qf_flat_bundle —
     authoritative, written at export time by quantfunc's bundle exporter).
  2. `quantization_config.precision_map` in safetensors metadata
     (prequant_lighting_separate / prequant_svdq_separate exports — same
     authoritative source, packaged inside a JSON-string field).
  3. Per-tensor dtype scan (current fallback): F8/I8/F16/BF16 weights →
     target_quant ("i4" / "f4" / ...), 1-D weights and norm/embedding
     patterns → "fp16". Used for raw FP16/BF16 AIO checkpoints where the
     producer didn't stamp a precision_map.

The output mirrors configs/qwenimage-all-i4-baseline.json layout:
  { "<layer-key>": "i4" | "fp16" | ... }

Layer keys are normalized to the form our C++ precision_config matcher
expects (e.g. `transformer_blocks.attn.to_qkv` for layer-pattern matching).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

from .safetensors_io import read_safetensors_header, read_safetensors_metadata

logger = logging.getLogger(__name__)


def _precision_map_from_metadata(file_path: str | Path) -> Optional[dict]:
    """Return precision_map from safetensors metadata if present.

    Tries two field layouts (in order):
      - `transformer.precision_map` (qf_flat_bundle's flat-prefixed metadata)
      - `quantization_config` (separated exports — a JSON string whose
        decoded value has a `precision_map` key, which itself may be a
        JSON-encoded string for backward compat).

    Returns None if neither is present or parsing fails — caller falls
    back to the per-tensor dtype scan.
    """
    try:
        meta = read_safetensors_metadata(file_path)
    except Exception:
        return None

    def _coerce_to_dict(v):
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            try:
                d = json.loads(v)
                return d if isinstance(d, dict) else None
            except Exception:
                return None
        return None

    # (1) qf_flat_bundle stamp
    raw = meta.get("transformer.precision_map")
    pm = _coerce_to_dict(raw)
    if pm:
        return pm

    # (2) separated prequant: `quantization_config` JSON-string with nested
    # `precision_map` field (which may itself be a JSON string).
    qc_raw = meta.get("quantization_config") or meta.get("transformer.quantization_config")
    qc = _coerce_to_dict(qc_raw)
    if qc:
        pm = _coerce_to_dict(qc.get("precision_map"))
        if pm:
            return pm

    return None


# Tensor dtype categories
_QUANTIZABLE_DTYPES = {"F8_E4M3", "F8_E5M2", "I8", "F16", "BF16"}
_PRESERVE_AS_FP16 = {"F32"}  # too large to quantize as 4-bit cleanly


# Layer-name patterns that should NEVER be quantized (norms, embeddings).
# Matched against the stripped-of-prefix key.
_KEEP_FP16_PATTERNS = [
    re.compile(r"\.norm[0-9]?\."),
    re.compile(r"\.norm$"),
    re.compile(r"_norm\."),
    re.compile(r"_norm$"),
    re.compile(r"\.norm\d+\."),
    re.compile(r"\.embedding\."),
    re.compile(r"\.embeddings\."),
    re.compile(r"^embedding\."),
    re.compile(r"^embed_tokens\."),
    re.compile(r"\.bias$"),                 # biases stay fp16
    re.compile(r"\.gamma$"),                 # adaln gamma
    re.compile(r"_modulation\."),            # modulation (small)
    re.compile(r"_mod\."),                   # qwen mod weights
    re.compile(r"\.mod\.weight$"),
    re.compile(r"\.add_q_proj\.weight$"),    # alternative naming
    re.compile(r"\.proj_out\.weight$"),
]


def _layer_key_from_tensor_key(tensor_key: str) -> Optional[str]:
    """Convert a tensor key like `transformer_blocks.0.attn.to_q.weight`
    into a layer key `transformer_blocks.0.attn.to_q` (drop final `.weight`).

    Returns None for keys that aren't .weight (we don't generate config
    entries for biases / scales / etc.).
    """
    if not tensor_key.endswith(".weight"):
        return None
    return tensor_key[: -len(".weight")]


def _should_keep_fp16(layer_key: str, shape: list[int]) -> bool:
    """Heuristic: 1-D weights (norm scales) and explicit norm/embedding
    layer-name patterns stay FP16."""
    if len(shape) <= 1:
        return True
    for pat in _KEEP_FP16_PATTERNS:
        if pat.search(layer_key):
            return True
    return False


def auto_derive_precision_map(
    transformer_path: str | Path,
    target_quant: str = "i4",
    key_strip_prefix: str = "",
) -> dict:
    """Build a precision_config dict by inspecting per-tensor dtypes.

    Args:
        transformer_path:    Path to the transformer .safetensors file.
        target_quant:        "i4" / "f4" / "i8" / "fp8" — the quant level for
                              quantizable layers.
        key_strip_prefix:    Optional ComfyUI-style prefix to strip off before
                              matching against patterns (e.g.
                              "model.diffusion_model.").

    Returns:
        dict keyed by layer pattern → precision string.

    Memory: only reads safetensors header (~few MB JSON). No weight data.
    """
    # Priority 1+2: authoritative precision_map stamped by quantfunc's
    # exporter. Prequant bundles (separated or qf_flat) write this; using
    # it directly avoids the per-tensor scan misclassifying obfuscated
    # UUID keys (cf. img_mod.1 being silently dropped because the bundle
    # transformer keys are `model.diffusion_model.<UUID>`, not the layer
    # path the scanner is built for).
    pm = _precision_map_from_metadata(transformer_path)
    if pm:
        logger.info("[auto-precision] %s: %d entries from safetensors metadata",
                      Path(transformer_path).name, len(pm))
        return pm

    h = read_safetensors_header(transformer_path)
    h.pop("__metadata__", None)
    config: dict[str, str] = {}
    quantized_count = 0
    fp16_count = 0
    skipped_count = 0
    for tensor_key, info in h.items():
        layer_key = _layer_key_from_tensor_key(tensor_key)
        if layer_key is None:
            continue                           # not a .weight
        if key_strip_prefix and layer_key.startswith(key_strip_prefix):
            layer_key = layer_key[len(key_strip_prefix):]
        dtype = info.get("dtype", "")
        shape = info.get("shape", [])
        if dtype not in _QUANTIZABLE_DTYPES and dtype not in _PRESERVE_AS_FP16:
            skipped_count += 1
            continue
        if _should_keep_fp16(layer_key, shape):
            config[layer_key] = "fp16"
            fp16_count += 1
        else:
            config[layer_key] = target_quant
            quantized_count += 1
    logger.info("[auto-precision] %s: %d quant + %d fp16 + %d skipped",
                  Path(transformer_path).name,
                  quantized_count, fp16_count, skipped_count)
    return config
