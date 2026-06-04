"""Helper for ComfyUI standalone Load VAE file format.

Common prefixes:
  - first_stage_model.   → SD-style legacy
  - vae.                  → bundled-checkpoint slice (rarely standalone but seen)
  - ""                    → already HF-native (Flux/Qwen standalone VAE)

Used cooperatively by other adapters when sources.vae is provided.
"""

from __future__ import annotations

from .tools import read_safetensors_keys


CANDIDATE_VAE_PREFIXES = [
    "first_stage_model.",
    "vae.",
]


def _detect_vae_prefix(file_path: str) -> str:
    sample: list[str] = []
    for k in read_safetensors_keys(file_path):
        sample.append(k)
        if len(sample) >= 50:
            break
    for px in CANDIDATE_VAE_PREFIXES:
        if any(k.startswith(px) for k in sample):
            return px
    return ""
