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


# Diffusers-native VAE markers — when ANY is present the file is already
# diffusers-shaped (exactly the naming the engine's VAE loader asks for), so it
# must NOT be wrapped in the engine's ComfyUIVAEAliasProvider. That provider
# remaps diffusers → ComfyUI flat (post_quant_conv.bias → conv2.bias, etc.);
# applying it to an already-diffusers VAE makes the lookup miss →
# "Tensor post_quant_conv.bias not found" (#408, SVDQ staging with a
# diffusion_pytorch_model.safetensors VAE).
_DIFFUSERS_VAE_MARKERS = (
    "post_quant_conv.", "quant_conv.",
    "encoder.conv_in.", "decoder.conv_in.",
    "encoder.down_blocks.", "decoder.up_blocks.",
)
# ComfyUI FLAT-named VAE markers (root conv1/conv2, flat up/down samples, head.*).
# Such a file NEEDS the alias because the engine asks diffusers names it lacks.
_COMFYUI_FLAT_VAE_MARKERS = (
    "conv1.", "conv2.",
    "decoder.upsamples.", "encoder.downsamples.",
    "decoder.middle.", "encoder.middle.",
)


def _is_comfyui_flat_vae(file_path: str, prefix: str = "") -> bool:
    """True only when the VAE uses ComfyUI FLAT naming and therefore NEEDS the
    engine's ComfyUIVAEAliasProvider. False for a diffusers-native VAE
    (post_quant_conv.*/encoder.conv_in.*), which the engine reads directly —
    wrapping that one in the alias provider is the #408 root cause.

    `prefix` is an optional key prefix (e.g. "vae.") to strip before matching,
    so a bundle-embedded VAE slice is classified by its inner key shape.
    """
    sample: list[str] = []
    for k in read_safetensors_keys(file_path):
        kk = k[len(prefix):] if (prefix and k.startswith(prefix)) else k
        sample.append(kk)
        if len(sample) >= 256:
            break
    # A single diffusers marker is decisive — it's already diffusers-shaped.
    if any(any(k.startswith(p) for p in _DIFFUSERS_VAE_MARKERS) for k in sample):
        return False
    return any(any(k.startswith(p) for p in _COMFYUI_FLAT_VAE_MARKERS) for k in sample)
