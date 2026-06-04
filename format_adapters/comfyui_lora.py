"""LoRA list pass-through.

LoRAs from chained QuantFuncLoadLoRA nodes are accumulated into a list of
LoRARef in the SourceBundle. We don't write them to staging — we forward the
list to quantfunc_create() via config_json["lora"], which our existing
LoRAMerge consumes.

This file is a placeholder for any LoRA-specific normalization (e.g.
detecting kohya vs diffusers naming and rewriting on the fly). For MVP, our
LoRAMerge already handles the common formats so it's pass-through.
"""

from __future__ import annotations

from .base import LoRARef


def loras_to_config_list(loras: list[LoRARef]) -> list[dict]:
    """Convert SourceBundle.loras → config_json["lora"] expected by C API.

    The C API expects a list of {"path": str, "strength": float} dicts. Our
    LoRAMerge supports separate model / clip strengths but reads them as a
    single scalar — the higher of the two for now. (Sprint 2 may add
    separate strength channels.)
    """
    result: list[dict] = []
    for lora in loras:
        result.append({
            "path": lora.path,
            "strength": max(lora.strength_model, lora.strength_clip),
        })
    return result
