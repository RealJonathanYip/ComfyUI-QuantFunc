"""Abstract base classes for the format adapter factory.

Every adapter consumes a SourceBundle (one or more file references from the
ComfyUI loader nodes) and produces a StagingResult — a small HF-style directory
that the C++ engine consumes through its existing Pipeline::from_pretrained
entry point. quantfunc_config.json in the staging dir carries hints (key_strip,
key_filter, method) so the C++ side knows how to view the on-disk safetensors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


class UnsupportedFormatError(RuntimeError):
    """Raised when no adapter matches the given sources."""


# Canonical plugin-internal method_hint values. Describe BOTH the producer
# and the packaging so downstream dispatch (auto-derive skip, backend
# selection) can be table-driven instead of scattered string-matches.
#
#   prequant_lighting_separate  — quantfunc CLI `--export-format separated`
#                                  (HF-diffusers multi-subdir prequant)
#   prequant_lighting_bundle    — quantfunc CLI `--export-format bundle`
#                                  (qf_flat_bundle single .safetensors)
#   prequant_svdq_separate      — third-party SVDQuant (Nunchaku/MIT)
#                                  single transformer file
#   online_quant                — clean BF16/FP16 checkpoint, engine
#                                  quantizes at load via precision_config
#   nvfp4_disk                  — third-party NVFP4 disk-load format
#
# Adapters MUST return one of these in StagingResult.method_hint.
METHOD_HINTS = (
    "prequant_lighting_separate",
    "prequant_lighting_bundle",
    "prequant_svdq_separate",
    "online_quant",
    "nvfp4_disk",
)

# `nodes_format_adapters.py` dispatches on these. Bundles whose metadata
# already carries an authoritative precision_map MUST appear here — auto-
# derive must skip the per-tensor dtype scan and read the bundle's own map.
PREQUANT_METHOD_HINTS = (
    "prequant_lighting_separate",
    "prequant_lighting_bundle",
    "prequant_svdq_separate",
)

# Legacy → canonical rename map. Apply to anything that may come from disk
# caches, persisted markers, or old code paths so the rename is invisible
# to the dispatcher.
_LEGACY_METHOD_HINT_RENAMES = {
    "prequant_ours":        "prequant_lighting_separate",
    "lighting_precomputed": "prequant_lighting_bundle",
    "prequant_svdq":        "prequant_svdq_separate",
}


def canonicalize_method_hint(h: str) -> str:
    """Map any legacy method_hint string to its canonical form. Idempotent."""
    return _LEGACY_METHOD_HINT_RENAMES.get(h, h)


# ── Inputs ──────────────────────────────────────────────────────────────────

@dataclass
class LoRARef:
    """A single LoRA reference accumulated by a chained QuantFuncLoadLoRA chain."""
    path: str
    strength_model: float = 1.0
    strength_clip: float = 1.0


@dataclass
class FileRef:
    """A reference to a single safetensors file from a scanning loader node.

    `arch` and `kind` are populated at scan time by ModelScanner, so adapters
    can dispatch without re-fingerprinting.
    """
    path: str
    arch: str = ""              # "QwenImage" | "QwenImageEdit" | "QwenImageLayered" | "Flux2Klein" | "ZImage" | "Ideogram4" | ""
    kind: str = ""              # fingerprint_kind_from_metadata: "raw_highprec" | "prequant_lighting_separate" | "nvfp4_disk" | "raw_fp4" | "raw_fp8" | "raw_fp8_mixed" | "raw_int8" | "" ; or force_kind "bundled_checkpoint"
    mtime: float = 0.0


@dataclass
class SourceBundle:
    """Inputs collected from ComfyUI loader nodes for one BuildPipeline run.

    Either `checkpoint` is set (single-file all-in-one) OR the trio
    {transformer, text_encoder, vae} is set. Both populated → checkpoint wins
    and the trio is ignored.
    """
    transformer: Optional[FileRef] = None
    text_encoder: Optional[FileRef] = None
    vae: Optional[FileRef] = None
    checkpoint: Optional[FileRef] = None
    loras: list[LoRARef] = field(default_factory=list)
    scheduler_config: Optional[str] = None  # path to scheduler JSON


# ── Build context ───────────────────────────────────────────────────────────

@dataclass
class BuildContext:
    """Pipeline-wide configuration for the build, threaded through adapters."""
    precision_map_xfm: Optional[dict] = None     # {path, target, arch}
    precision_map_te: Optional[dict] = None
    vae_precision: str = "fp16"                  # "fp16" | "fp8"
    text_precision: str = "int4"                 # used when precision_map_te is None
    device_idx: int = 0
    backend: str = "lighting"                    # always lighting for online quant
    api_key: str = ""
    server_url: str = ""


# ── Outputs ─────────────────────────────────────────────────────────────────

@dataclass
class StagingResult:
    """What an adapter returns: the path to feed quantfunc_create + diagnostics."""
    model_dir: str                  # The staging dir (or original prequant dir)
    arch: str                       # Detected architecture
    method_hint: str                # One of METHOD_HINTS (see top of file)
    cleanup_dir: Optional[str] = None  # If non-None, BuildPipeline removes it on dtor


# ── Adapter interface ───────────────────────────────────────────────────────

class FormatAdapter(ABC):
    """Convert one or more source files into the standardized staging layout.

    Adapters self-register via `@adapter(priority=N)` on the class. The factory
    iterates registered adapters in priority order (high → low) and uses the
    first whose detect() returns True.

    Detection should be cheap (header-only). Adaptation is allowed to do
    O(model_size) symlink / config writes but MUST NOT copy weight data.
    """

    @classmethod
    @abstractmethod
    def detect(cls, sources: SourceBundle) -> bool:
        """Cheap predicate: does this adapter handle these sources?

        Implementations should only inspect metadata / file extensions / cheap
        header reads. Returning False means the factory tries the next adapter.
        """

    @abstractmethod
    def adapt(self, sources: SourceBundle, staging_dir: Path,
              context: BuildContext) -> StagingResult:
        """Materialize `staging_dir` (or repurpose original model_dir).

        Implementations populate transformer/, text_encoder/, vae/, tokenizer/
        subdirs of `staging_dir` using symlinks, write model_index.json and
        quantfunc_config.json, and return a StagingResult.
        """
