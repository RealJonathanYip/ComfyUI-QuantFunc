"""HF-style staging directory builder.

Adapters use HFLayout to populate a temp directory with the canonical layout
that quantfunc_create() expects:

    staging_dir/
    ├── model_index.json
    ├── transformer/
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors  → symlink to source
    ├── text_encoder/
    │   ├── config.json
    │   └── model.safetensors                    → symlink
    ├── vae/
    │   ├── config.json
    │   └── diffusion_pytorch_model.safetensors  → symlink
    ├── tokenizer/                               (copied from plugin bundle)
    ├── scheduler/
    │   └── scheduler_config.json                (optional)
    └── quantfunc_config.json                    (our hints)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Map our internal arch tag → diffusers _class_name (what detectPipelineKind expects)
ARCH_TO_PIPELINE_CLASS = {
    "QwenImage":      "QwenImagePipeline",
    "QwenImageEdit":  "QwenImageEditPipeline",
    "Flux2Klein":     "Flux2KleinPipeline",
    "ZImage":         "ZImagePipeline",
}

ARCH_TO_TRANSFORMER_CLASS = {
    "QwenImage":      "QwenImageTransformer2DModel",
    "QwenImageEdit":  "QwenImageTransformer2DModel",  # same class, edit_mode=true
    "Flux2Klein":     "Flux2Transformer2DModel",
    "ZImage":         "ZImageTransformer2DModel",
}


class HFLayout:
    """Builder for the HF-style staging directory.

    Adapters call methods on this object; nothing touches the filesystem until
    a method is invoked. All weight references are *symlinks*; only configs
    and tokenizer files are actually written.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._hints: dict = {}              # quantfunc_config.json contents
        self._pipeline_class: str = ""

    # ── Component placement ───────────────────────────────────────────────

    def add_transformer(self, source_path: str | Path,
                         config: Optional[dict] = None) -> Path:
        """Symlink the transformer weights under BOTH canonical filenames:
        - `diffusion_pytorch_model.safetensors` — diffusers convention used
          by ComfyUI's standard loader probes and runtime-quantize / SVDQ
          paths in the C engine.
        - `model.safetensors` — hard-coded filename the Lighting prequant
          loader (ComponentImpl.cpp) uses; without this alias prequant
          reload errors with "Pre-quantized Lighting model directory
          missing model.safetensors". One symlink solves both lookups."""
        sub = self._add_component(
            "transformer", source_path,
            "diffusion_pytorch_model.safetensors", config)
        # Add the alias as a sibling symlink (target the same source so
        # both names resolve to the same file).
        alias = sub / "model.safetensors"
        if alias.exists() or alias.is_symlink():
            alias.unlink()
        alias.symlink_to(os.path.abspath(str(source_path)))
        return sub

    def add_transformer_remapped(self, source_path: str | Path,
                                  remap_fn,
                                  config: Optional[dict] = None) -> Path:
        """Write a remapped transformer (not a symlink) using `remap_fn`.

        `remap_fn(src_path: Path, dst_path: Path) -> None` is called once to
        write the diffusers-keyed safetensors at the staging path. Used for
        BFL → diffusers ZImage and similar key-rename cases that can't be
        expressed via simple `set_key_strip` / `set_key_filter`.
        """
        sub = self.root / "transformer"
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / "diffusion_pytorch_model.safetensors"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        remap_fn(Path(source_path), dst)
        if config:
            (sub / "config.json").write_text(json.dumps(config, indent=2))
        return sub

    def add_text_encoder(self, source_path: str | Path,
                          config: Optional[dict] = None) -> Path:
        return self._add_component(
            "text_encoder", source_path, "model.safetensors", config)

    def add_vae(self, source_path: str | Path,
                config: Optional[dict] = None) -> Path:
        return self._add_component(
            "vae", source_path, "diffusion_pytorch_model.safetensors", config)

    def add_vae_remapped(self, source_path: str | Path,
                          remap_fn,
                          config: Optional[dict] = None) -> Path:
        """Write a remapped VAE (not a symlink) using `remap_fn`.

        `remap_fn(src_path: Path, dst_path: Path) -> None`. Used for
        BFL → HF AutoencoderKL key renames.
        """
        sub = self.root / "vae"
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / "diffusion_pytorch_model.safetensors"
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        remap_fn(Path(source_path), dst)
        if config:
            (sub / "config.json").write_text(json.dumps(config, indent=2))
        return sub

    def _add_component(self, subdir: str, source_path: str | Path,
                        weight_filename: str, config: Optional[dict]) -> Path:
        sub = self.root / subdir
        sub.mkdir(parents=True, exist_ok=True)
        weight_link = sub / weight_filename
        if weight_link.exists() or weight_link.is_symlink():
            weight_link.unlink()
        weight_link.symlink_to(os.path.abspath(str(source_path)))
        # Sharded model fan-out: when source is one shard of a multi-shard
        # safetensors, symlink ALL siblings + the index.json. Without this
        # the C engine's ShardedSafeTensors only sees one shard and silently
        # drops keys living in the other shards (with `partial=true` in
        # loadParams), producing empty weights at quant time
        # (e.g. Klein 4B TE down_proj for layers 20-35 lives in
        # `model-00002-of-00002.safetensors`; staging only the first shard
        # makes layer 20's down_proj.weight invalid → setQuantMode crashes).
        src_path = os.path.abspath(str(source_path))
        src_dir = os.path.dirname(src_path)
        src_name = os.path.basename(src_path)
        # Detect sharded layout by `<prefix>-NNNNN-of-NNNNN.safetensors`
        # naming, or by sibling `<source>.index.json`.
        import re as _re
        m = _re.match(r"^(.+?)-(\d{5})-of-(\d{5})\.safetensors$", src_name)
        if m:
            prefix = m.group(1)
            try:
                shards = sorted(
                    f for f in os.listdir(src_dir)
                    if _re.match(rf"^{_re.escape(prefix)}-\d{{5}}-of-\d{{5}}\.safetensors$", f)
                )
            except OSError:
                shards = []
            # Always link every shard under its ORIGINAL name (engine /
            # ShardedSafeTensors reads them via the index.json which
            # references original names, not the alias).
            for f in shards:
                link = sub / f
                if link.exists() or link.is_symlink():
                    link.unlink()
                link.symlink_to(os.path.join(src_dir, f))
            # Also link index.json under BOTH the source's natural name
            # and the alias the C side expects (it scans for any
            # `*.safetensors.index.json`).
            idx_src = os.path.join(src_dir, f"{prefix}.safetensors.index.json")
            if os.path.isfile(idx_src):
                # Source-named index (what the inner shards reference)
                idx_link_src = sub / f"{prefix}.safetensors.index.json"
                if idx_link_src.exists() or idx_link_src.is_symlink():
                    idx_link_src.unlink()
                idx_link_src.symlink_to(idx_src)
                # Engine-side filename based on `weight_filename`
                wf_stem = weight_filename[:-len(".safetensors")] \
                    if weight_filename.endswith(".safetensors") else weight_filename
                idx_link_dst = sub / f"{wf_stem}.safetensors.index.json"
                if (idx_link_dst != idx_link_src and
                        (idx_link_dst.exists() or idx_link_dst.is_symlink())):
                    idx_link_dst.unlink()
                if idx_link_dst != idx_link_src:
                    idx_link_dst.symlink_to(idx_src)
        if config:
            (sub / "config.json").write_text(json.dumps(config, indent=2))
        return sub

    # ── Hint accumulation ─────────────────────────────────────────────────

    def set_method(self, method: str) -> None:
        """e.g. "online_quant" / "nvfp4_disk" / "prequant_lighting_separate" / "prequant_lighting_bundle" / "prequant_svdq_separate" — or, when the qf_flat_bundle adapter is mirroring the bundle's own metadata, the raw C++-facing value ("lighting_precomputed" / "svdq")."""
        self._hints["method"] = method

    def set_key_strip(self, kind: str, prefix: str) -> None:
        """kind ∈ {"transformer", "te", "vae"}.  Additive lookup."""
        if prefix:
            self._hints[f"{kind}_key_strip"] = prefix

    def set_key_filter(self, kind: str, prefix: str) -> None:
        """kind ∈ {"transformer", "te", "vae"}.  Strict filter."""
        if prefix:
            self._hints[f"{kind}_key_filter"] = prefix

    def set_extra(self, key: str, value) -> None:
        """Additional hint (e.g. nvfp4_block_size)."""
        self._hints[key] = value

    def set_pipeline_class(self, arch: str) -> None:
        """Set _class_name for model_index.json (from arch tag)."""
        self._pipeline_class = ARCH_TO_PIPELINE_CLASS.get(arch, "")

    # ── Index files ───────────────────────────────────────────────────────

    def write_model_index(self, arch: str,
                           extra_components: Optional[dict] = None) -> None:
        """Write model_index.json so detectPipelineKind() can identify the arch."""
        cls = ARCH_TO_PIPELINE_CLASS.get(arch, "")
        if not cls:
            raise ValueError(f"Unknown arch: {arch!r}")
        idx: dict = {
            "_class_name": cls,
            "_diffusers_version": "0.30.0",
            "transformer": ["diffusers", ARCH_TO_TRANSFORMER_CLASS.get(arch, "")],
        }
        if (self.root / "vae").exists():
            idx["vae"] = ["diffusers", "AutoencoderKL"]
        if (self.root / "text_encoder").exists():
            idx["text_encoder"] = ["transformers", "Qwen2_5VLForConditionalGeneration"
                                    if arch == "QwenImageEdit"
                                    else "Qwen3ForCausalLM"]
        if extra_components:
            idx.update(extra_components)
        (self.root / "model_index.json").write_text(json.dumps(idx, indent=2))

    def apply_user_precisions(
            self,
            text_precision: Optional[str] = None,
            vae_precision: Optional[str] = None) -> None:
        """Propagate user-selected precisions from BuildContext into the
        nested per-component config blocks the C++ engine reads. Adapters
        whose source has no per-component metadata (raw HF layouts, separate
        UNet/CLIP/VAE files, upstream FP8-mixed Qwen bundles) MUST call this
        before write_quantfunc_config — otherwise the C++ TE/VAE silently
        default to FP16 and the export bloats by tens of GB."""
        if text_precision:
            te = dict(self._hints.get("text_encoder") or {})
            te.setdefault("text_precision", text_precision)
            self._hints["text_encoder"] = te
        if vae_precision:
            v = dict(self._hints.get("vae") or {})
            v.setdefault("vae_precision", vae_precision)
            self._hints["vae"] = v

    def write_quantfunc_config(self) -> None:
        """Flush accumulated hints to staging/quantfunc_config.json."""
        if not self._hints:
            return
        (self.root / "quantfunc_config.json").write_text(
            json.dumps(self._hints, indent=2))
        logger.debug("[staging] quantfunc_config.json: %s", self._hints)

    # ── Scheduler / tokenizer ────────────────────────────────────────────

    def add_scheduler(self, scheduler_config_path: Optional[str],
                       arch: Optional[str] = None) -> None:
        """Copy a scheduler_config.json into staging.

        Priority:
          1. Caller-provided `scheduler_config_path` (e.g. user's diffusers
             model dir scheduler/scheduler_config.json) — copy as-is.
          2. Bundled fallback at <plugin>/bin/scheduler_configs/<arch>.json —
             used when the caller doesn't have a scheduler config but we know
             the arch. This is required because the C++ engine fails to
             open the file if `scheduler/scheduler_config.json` is missing.

        If neither is available, the caller's pipeline load will fail at
        the C++ side with a clear "Cannot open file" error.
        """
        sub = self.root / "scheduler"
        sub.mkdir(parents=True, exist_ok=True)
        dst = sub / "scheduler_config.json"
        if scheduler_config_path:
            shutil.copy2(scheduler_config_path, dst)
            return
        if arch:
            plugin_root = Path(__file__).resolve().parent.parent.parent
            bundled = plugin_root / "bin" / "scheduler_configs" / f"{arch}.json"
            if bundled.is_file():
                shutil.copy2(bundled, dst)
                return
        logger.warning("[staging] no scheduler_config.json (caller=None, "
                        "arch=%r → bundled missing); pipeline will fail to "
                        "load if the engine demands one.", arch)

    def tokenizer_dir(self) -> Path:
        sub = self.root / "tokenizer"
        sub.mkdir(parents=True, exist_ok=True)
        return sub


def copy_tokenizer_bundle(arch: str, dst: Path) -> None:
    """Copy the bundled tokenizer for `arch` from <plugin>/bin/tokenizers/<arch>/."""
    plugin_root = Path(__file__).resolve().parent.parent.parent
    src = plugin_root / "bin" / "tokenizers" / arch
    if not src.is_dir():
        logger.warning("[staging] no bundled tokenizer for arch=%s at %s", arch, src)
        return
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file():
            shutil.copy2(f, dst / f.name)
    logger.debug("[staging] tokenizer bundle copied for arch=%s (%d files)",
                  arch, sum(1 for _ in src.iterdir()))


def bundled_transformer_config(arch: str) -> Optional[dict]:
    """Return bundled transformer config (architecture dims) for `arch`, or None.

    Used by adapters when the source file (NVFP4 disk-load, etc.) doesn't ship
    a per-component config.json. Without proper dims (num_attention_heads,
    num_layers, ...) the C++ side falls back to defaults that may not match
    the actual weight shapes (e.g. Klein 4B vs 9B).
    """
    plugin_root = Path(__file__).resolve().parent.parent.parent
    p = plugin_root / "bin" / "transformer_configs" / f"{arch}.json"
    if not p.is_file():
        logger.warning("[staging] no bundled transformer config for arch=%s at %s", arch, p)
        return None
    return json.loads(p.read_text())


def bundled_te_config(arch: str) -> Optional[dict]:
    """Return the bundled text-encoder config.json for `arch`, or None.

    Bundled at <plugin>/bin/text_encoder_configs/<arch>.json. Used by adapters
    when the source file (e.g. a bundled multi-component checkpoint)
    doesn't ship per-component config.json — without this, our C++ TE
    loader falls back to defaults (Qwen3 2.5B) and crashes when the actual
    weights are Qwen2.5-VL 7B.
    """
    plugin_root = Path(__file__).resolve().parent.parent.parent
    p = plugin_root / "bin" / "text_encoder_configs" / f"{arch}.json"
    if not p.is_file():
        logger.warning("[staging] no bundled TE config for arch=%s at %s", arch, p)
        return None
    return json.loads(p.read_text())


def bundled_vae_config(arch: str) -> Optional[dict]:
    """Same idea but for the VAE config.json."""
    plugin_root = Path(__file__).resolve().parent.parent.parent
    p = plugin_root / "bin" / "vae_configs" / f"{arch}.json"
    if not p.is_file():
        logger.warning("[staging] no bundled VAE config for arch=%s at %s", arch, p)
        return None
    return json.loads(p.read_text())
