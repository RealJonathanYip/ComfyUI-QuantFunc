"""Helper for ComfyUI standalone Load CLIP / Load TextEncoder file format.

Detection of the right prefix to strip — depends on which TE was loaded.
This adapter doesn't have its own detect(); it's used cooperatively by
ComfyUIDiffusionModelAdapter when text_encoder is provided alongside.

Common prefixes we strip:
  - text_encoders.qwen25_7b.transformer.    → Qwen2.5-VL bundle (TE only)
  - text_encoders.qwen25_7b.                → some variants keep `transformer.`
  - cond_stage_model.transformer.           → SD-style (legacy)
  - text_model.                             → CLIP wrapper
  - ""                                       → already HF-native
"""

from __future__ import annotations

from .tools import read_safetensors_keys


CANDIDATE_TE_PREFIXES = [
    "text_encoders.qwen25_7b.transformer.",
    "text_encoders.qwen2vl.transformer.",
    "text_encoders.qwen3.transformer.",
    "text_encoders.qwen25_7b.",
    "text_encoders.qwen2vl.",
    "cond_stage_model.transformer.",
    "cond_stage_model.",
]


def _detect_te_prefix(file_path: str) -> str:
    # Materialize all keys once (bundled checkpoints have ~3000 keys) —
    # first-50 sampling missed TE entirely on bundles since
    # `model.diffusion_model.*` keys come first.
    # For each candidate prefix in declared (longest-first) order, return
    # the first one ANY key matches. This avoids picking a too-short prefix
    # just because a stray key (e.g. `text_encoders.qwen25_7b.logit_scale`,
    # which sits OUTSIDE the .transformer. subtree) matched it before the
    # transformer keys were inspected.
    all_keys = list(read_safetensors_keys(file_path))
    for px in CANDIDATE_TE_PREFIXES:
        if any(k.startswith(px) for k in all_keys):
            return px
    return ""
