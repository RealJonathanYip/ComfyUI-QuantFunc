"""Helpers used by the unified BuildPipeline node in nodes_format_adapters.py.

Two pieces:

1. `install_loader_path_patches()` — runs once at plugin import. Wraps the
   official `comfy.sd.load_*` functions and the `VAELoader` node so the
   returned MODEL / CLIP / VAE objects carry a `qf_source_path` attribute
   pointing at the safetensors file ComfyUI loaded from. ComfyUI itself
   does not retain that path on the loaded objects, so without this patch
   we would have to round-trip `state_dict()` through disk. Use
   `extract_qf_source_path(obj, label)` to read the path back.

2. Precision preset helpers (`build_precision_preset_options()` and
   `resolve_precision_preset()`) — populate inline precision-config
   dropdowns from three sources merged into one list:
     [builtin]  — JSON files from <plugin>/configs or QUANTFUNC_CONFIGS_DIR
     [series]   — QuantFunc model series presets cached via model_auto_loader
                  (downloaded on demand from ModelScope / HuggingFace)
     custom path — user-supplied absolute path to JSON
     [auto-derive] / [none] — sentinels.

This module exports no ComfyUI nodes itself — the actual node is
`QuantFuncBuildPipeline` in nodes_format_adapters.py.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("QuantFunc.PipelineBuilder")


# ============================================================================
# Loader path patches
# ============================================================================

_PATCHES_INSTALLED = False


def install_loader_path_patches() -> bool:
    """Install monkey-patches that attach qf_source_path to MODEL/CLIP/VAE.

    Idempotent. Returns True if patches are now in place (or were already).
    Failures are logged at DEBUG and return False — the FromComfy node will
    raise a clear error at build() time if a path can't be recovered.
    """
    global _PATCHES_INSTALLED
    if _PATCHES_INSTALLED:
        return True

    try:
        import comfy.sd as _csd
    except Exception as e:
        logger.debug("comfy.sd not importable, skipping patches: %s", e)
        return False

    # --- diffusion model (UNETLoader) ---------------------------------------
    try:
        _orig_load_diffusion = _csd.load_diffusion_model

        def _patched_load_diffusion(unet_path, *args, **kwargs):
            m = _orig_load_diffusion(unet_path, *args, **kwargs)
            try:
                setattr(m, "qf_source_path", unet_path)
            except Exception:
                pass
            return m

        _csd.load_diffusion_model = _patched_load_diffusion
    except Exception as e:
        logger.debug("load_diffusion_model patch failed: %s", e)

    # --- text encoder (CLIPLoader / DualCLIPLoader) -------------------------
    try:
        _orig_load_clip = _csd.load_clip

        def _patched_load_clip(ckpt_paths=None, *args, **kwargs):
            # Pass ckpt_paths POSITIONALLY: comfy.sd.load_clip takes it as the
            # first positional param, so a caller passing ANY extra positional
            # arg (whatever it currently is — don't name it, ComfyUI may change
            # the call-site) would otherwise hit
            # "got multiple values for argument 'ckpt_paths'".
            c = _orig_load_clip(ckpt_paths, *args, **kwargs)
            try:
                paths = list(ckpt_paths or [])
                setattr(c, "qf_source_paths", paths)
                if paths:
                    setattr(c, "qf_source_path", paths[0])
            except Exception:
                pass
            return c

        _csd.load_clip = _patched_load_clip
    except Exception as e:
        logger.debug("load_clip patch failed: %s", e)

    # --- checkpoint (CheckpointLoaderSimple) --------------------------------
    # Returns (model, clip, vae, clipvision). Tag all three with the same path.
    try:
        _orig_load_ckpt = _csd.load_checkpoint_guess_config

        def _patched_load_ckpt(ckpt_path, *args, **kwargs):
            result = _orig_load_ckpt(ckpt_path, *args, **kwargs)
            try:
                for obj in result[:3]:
                    if obj is not None:
                        setattr(obj, "qf_source_path", ckpt_path)
                        setattr(obj, "qf_is_checkpoint", True)
            except Exception:
                pass
            return result

        _csd.load_checkpoint_guess_config = _patched_load_ckpt
    except Exception as e:
        logger.debug("load_checkpoint_guess_config patch failed: %s", e)

    # --- VAE (VAELoader) ----------------------------------------------------
    # comfy.sd.VAE.__init__ takes sd= dict, not a path. Patch the node class
    # method directly so we can resolve the filename via folder_paths.
    try:
        import nodes as _nodes
        import folder_paths as _fp
        _orig_vae_load = _nodes.VAELoader.load_vae

        def _patched_vae_load(self, vae_name):
            result = _orig_vae_load(self, vae_name)
            try:
                full = _fp.get_full_path("vae", vae_name)
                if full and isinstance(result, tuple) and result and result[0] is not None:
                    setattr(result[0], "qf_source_path", full)
            except Exception:
                pass
            return result

        _nodes.VAELoader.load_vae = _patched_vae_load
    except Exception as e:
        logger.debug("VAELoader.load_vae patch failed: %s", e)

    _PATCHES_INSTALLED = True
    logger.info("[QuantFunc] ComfyUI loader path patches installed")
    return True


def _recover_path_from_cached_init(obj: Any) -> Optional[str]:
    """Best-effort source-path recovery when qf_source_path was stripped.

    Our `qf_source_path` tag is set by the monkey-patches above, but ComfyUI's
    `CLIP.clone()` / `ModelPatcher.clone()` only copy a fixed whitelist of
    attributes that does NOT include it (comfy/sd.py CLIP.clone copies
    patcher/cond_stage_model/tokenizer/layer_idx/...). So ANY node that clones
    the object between the loader and BuildPipeline — CLIPSetLastLayer, most
    LoRA loaders, the typical Qwen-Image-Edit graph — yields an untagged clone
    and the bare-`getattr` path above fails.

    ComfyUI's own clone() DOES preserve `cached_patcher_init`
    (comfy/model_patcher.py: `n.cached_patcher_init = self.cached_patcher_init`),
    a `(callable, args_tuple)` recording the original loader call. Its first
    positional arg is the source path(s):
      • diffusion model → `model.cached_patcher_init`        = (load_diffusion_model, (unet_path, ...))
      • CLIP            → `clip.patcher.cached_patcher_init`  = (load_clip_model_patcher, (ckpt_paths, ...))
      • checkpoint      → `model.cached_patcher_init` / `clip.patcher.cached_patcher_init` = (..., (ckpt_path, ...))
    `ckpt_paths` is a list (CLIP/DualCLIP), `unet_path`/`ckpt_path` a string —
    handle both. Returns the first arg that names an existing file, else None.

    COVERAGE = MODEL + CLIP (and checkpoint, whose MODEL+CLIP both record the
    same .ckpt — see nodes_format_adapters.py:668 which dedups te_path==xfm_path
    and routes a unified checkpoint through the `checkpoint=` field). It does NOT
    cover VAE: ComfyUI's `VAE` class sets no `cached_patcher_init`
    (load_checkpoint_guess_config tags only out[0]=MODEL and out[1].patcher=CLIP;
    load_clip tags the CLIP patcher; there is no VAE assignment site), so for a
    VAE this returns None — which is fine, because `VAE` has NO `clone()` method
    (only `CLIP.clone()` exists in comfy/sd.py), so a VAE's qf_source_path is
    never clone-stripped. The only VAE failure mode is a third-party node that
    rebuilds the VAE without the tag (rare); that still degrades to the clear
    RuntimeError below.
    """
    # Holder order matters only as a search order, not correctness: MODEL is
    # itself a ModelPatcher (attr on the object); CLIP holds it on `.patcher`.
    # For a unified checkpoint both holders record the SAME .ckpt path, so
    # whichever is found first yields the identical result.
    for holder in (obj, getattr(obj, "patcher", None)):
        if holder is None:
            continue
        cpi = getattr(holder, "cached_patcher_init", None)
        if not (isinstance(cpi, tuple) and len(cpi) == 2):
            continue
        # ComfyUI declares the 2nd element a tuple of loader args; require a
        # non-empty sequence so a malformed (fn, <dict/non-sequence>) record
        # from a third-party plugin degrades to None instead of raising.
        args = cpi[1]
        if not (isinstance(args, (list, tuple)) and args):
            continue
        # Position 0 is always the load TARGET — a path string (UNET/checkpoint)
        # or a list of paths (CLIP/DualCLIP); later args are embedding_dir / opts.
        first = args[0]
        cands = list(first) if isinstance(first, (list, tuple)) else [first]
        for q in cands:
            if isinstance(q, str) and os.path.isfile(q):
                return q
    return None


def _extract_path(obj: Any, label: str) -> str:
    """Pull qf_source_path off a MODEL/CLIP/VAE; raise with hint if missing.

    The qf_source_path/qf_source_paths attribute read covers MODEL, CLIP AND VAE.
    The cached_patcher_init clone-recovery fallback below covers MODEL + CLIP ONLY
    (VAE has no cached_patcher_init in ComfyUI) — so a VAE that is missing BOTH
    attributes raises (acceptable: VAE has no clone() method, so its tag is never
    clone-stripped; see _recover_path_from_cached_init).
    """
    p = getattr(obj, "qf_source_path", None)
    if p and os.path.isfile(p):
        return p
    paths = getattr(obj, "qf_source_paths", None)
    if paths:
        for q in paths:
            if q and os.path.isfile(q):
                return q
    # Clone-proof fallback: the tag is dropped by CLIP/ModelPatcher .clone(),
    # but cached_patcher_init survives the clone — recover the path from it.
    recovered = _recover_path_from_cached_init(obj)
    if recovered:
        logger.info(
            "[QuantFunc] %s had no qf_source_path (likely a cloned object); "
            "recovered source via cached_patcher_init: %s", label, recovered)
        return recovered
    raise RuntimeError(
        f"Cannot recover source file for {label}. The MODEL/CLIP/VAE object "
        f"has no qf_source_path attribute and the path could not be recovered "
        f"from cached_patcher_init. Causes:\n"
        f"  • The QuantFunc loader monkey-patch on comfy.sd failed to install "
        f"(see ComfyUI startup log for 'comfy loader path patches failed'),\n"
        f"  • The upstream node bypasses comfy.sd.load_* (custom loader, e.g. "
        f"GGUF / alternate text-encoder loaders),\n"
        f"  • An intermediate node returned a clone that dropped BOTH "
        f"qf_source_path and cached_patcher_init (the cached_patcher_init "
        f"fallback covers MODEL + CLIP only; VAE has none, but VAE is also "
        f"never clone-stripped — it has no clone() method — so a VAE failure "
        f"here means a third-party node rebuilt it without the tag),\n"
        f"  • The object was created by another plugin without the patch.\n"
        f"Workarounds:\n"
        f"  1. Use QuantFunc Model Loader / QuantFunc Model Auto Loader, or\n"
        f"  2. Use QuantFunc Pick Diffusion Model / Pick CLIP / Pick VAE, or\n"
        f"  3. Re-create the official ComfyUI loader node and re-run "
        f"(monkey-patches install at plugin import — comfy loaders created "
        f"before then won't be tagged)."
    )


# ============================================================================
# Precision config preset helpers (used inline by BuildPipelineFromComfy)
# ============================================================================

_BUILTIN_PREFIX = "[builtin] "
_SERIES_PREFIX = "[series] "
_NONE_LABEL = "[none]"
_AUTO_LABEL = "[auto-derive]"


def _scan_builtin_precision_presets() -> list[str]:
    """JSON files in <plugin>/configs/ + $QUANTFUNC_CONFIGS_DIR."""
    out: list[str] = []
    plugin_root = os.path.dirname(os.path.abspath(__file__))
    d = os.path.join(plugin_root, "configs")
    if os.path.isdir(d):
        for f in sorted(os.listdir(d)):
            if f.endswith(".json"):
                out.append(_BUILTIN_PREFIX + f)
    env_dir = os.environ.get("QUANTFUNC_CONFIGS_DIR", "")
    if env_dir and os.path.isdir(env_dir):
        for f in sorted(os.listdir(env_dir)):
            if f.endswith(".json"):
                out.append(_BUILTIN_PREFIX + os.path.join(env_dir, f))
    return out


def _scan_series_precision_presets() -> list[str]:
    """Presets cached by model_auto_loader from ModelScope precision-config/ dirs."""
    try:
        from .model_auto_loader import get_precision_config_options
    except Exception:
        return []
    out: list[str] = []
    for opt in get_precision_config_options():
        if opt and opt != "None":
            out.append(_SERIES_PREFIX + opt)
    return out


def _build_precision_preset_options() -> list[str]:
    """Full preset dropdown — sentinels first, then builtin, then series."""
    return ([_NONE_LABEL, _AUTO_LABEL]
            + _scan_builtin_precision_presets()
            + _scan_series_precision_presets())


def _resolve_precision_preset(preset: str, custom_path: str,
                               target: str, data_source: str
                               ) -> Optional[dict]:
    """Translate a (preset, custom_path) pair into a {path, target, preset} dict.

    Returns None for [none] (caller should not insert into options).
    Returns dict with empty `path` for [auto-derive]. The C++ engine treats
    empty path + preset=='auto' as "derive from per-tensor dtypes".
    """
    cp = (custom_path or "").strip()
    if cp:
        if not os.path.isfile(cp):
            raise RuntimeError(f"custom precision path not found: {cp}")
        return {"path": cp, "target": target, "preset": "custom"}

    if preset == _NONE_LABEL:
        return None
    if preset == _AUTO_LABEL:
        return {"path": "", "target": target, "preset": "auto"}

    if preset.startswith(_BUILTIN_PREFIX):
        name = preset[len(_BUILTIN_PREFIX):]
        if os.path.isabs(name):
            return {"path": name, "target": target,
                    "preset": os.path.basename(name)}
        plugin_root = os.path.dirname(os.path.abspath(__file__))
        full = os.path.join(plugin_root, "configs", name)
        return {"path": full, "target": target, "preset": name}

    if preset.startswith(_SERIES_PREFIX):
        sel = preset[len(_SERIES_PREFIX):]
        from .model_auto_loader import (
            resolve_selection_no_series, download_precision_config,
        )
        series, fname = resolve_selection_no_series(sel, "Precision config")
        if not series or not fname:
            return None
        local = download_precision_config(series, fname, data_source)
        return {"path": local, "target": target, "preset": fname}

    return None


# ============================================================================
# Auto-detect scheduler config for known model variants
# ============================================================================
#
# Source of truth for scheduler configs is THIS dict (Python literals);
# JSON files under <plugin>/configs/_auto/ are derived caches written on
# first use, so the engine (which expects a path) has something to read.
#
# Keys in this dict are matched by detect_scheduler_config() using filename
# heuristics; safetensors metadata 'method' field is also consulted.
# Add a new variant by adding an entry here — no JSON file needed.

_SCHEDULER_PRESETS: dict[str, dict] = {
    # Qwen-Image-Lightning few-step variants (4-8 steps).
    # Ground-truth: Qwen-Image-Lightning diffusers scheduler_config.json.
    "qwen_lightning": {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "base_image_seq_len": 256,
        "base_shift": 1.0986122886681098,
        "max_image_seq_len": 8192,
        "max_shift": 1.0986122886681098,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "time_shift_type": "exponential",
        "use_dynamic_shifting": True,
    },

    # Qwen-Image / Qwen-Image-Edit base (20-50 step inference).
    # Ground-truth: Qwen/Qwen-Image diffusers scheduler_config.json.
    "qwen_base": {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 8192,
        "max_shift": 0.9,
        "num_train_timesteps": 1000,
        "shift": 1.0,
        "shift_terminal": 0.02,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    },

    # FLUX.2 Klein 4B / 9B base. Ground-truth: black-forest-labs flux.2-klein.
    # Differs from qwen base: max_shift=1.15, shift=3.0, shift_terminal=null,
    # max_image_seq_len=4096 (Klein renders at lower max seq len).
    "klein_base": {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "base_image_seq_len": 256,
        "base_shift": 0.5,
        "invert_sigmas": False,
        "max_image_seq_len": 4096,
        "max_shift": 1.15,
        "num_train_timesteps": 1000,
        "shift": 3.0,
        "shift_terminal": None,
        "stochastic_sampling": False,
        "time_shift_type": "exponential",
        "use_beta_sigmas": False,
        "use_dynamic_shifting": True,
        "use_exponential_sigmas": False,
        "use_karras_sigmas": False,
    },

    # Z-Image-Turbo — Tongyi-MAI/Z-Image-Turbo. Distilled but uses
    # NON-dynamic shifting (different from Qwen-Image-Lightning), so it
    # has its own preset rather than reusing 'qwen_lightning'.
    "z_image_turbo": {
        "_class_name": "FlowMatchEulerDiscreteScheduler",
        "num_train_timesteps": 1000,
        "use_dynamic_shifting": False,
        "shift": 3.0,
    },
}


def _select_preset_key(transformer_path: str) -> Optional[str]:
    """Detect architecture from filename / metadata, then pick a preset.

    Architecture is detected first; distilled-variant keywords (lightning /
    distill / rapid) only flip Qwen → qwen_lightning. ZImage Turbo and
    Klein are matched by their own architecture keywords because their
    schedulers are NOT compatible with the Qwen lightning shift curve.
    """
    fname = os.path.basename(transformer_path).lower()

    def _arch(s: str) -> Optional[str]:
        if any(k in s for k in ("z-image", "zimage", "z_image")):
            return "z_image"
        if any(k in s for k in ("klein", "flux-2", "flux.2", "flux_2")):
            return "klein"
        if any(k in s for k in ("qwen-image", "qwenimage", "qwen_image")):
            return "qwen"
        return None

    def _is_distilled(s: str) -> bool:
        # Nunchaku-specific naming: their entire Qwen-Image lineup is
        # distilled few-step (`ultimate_speed`, `best_quality`, `balance`
        # — all are 4-8 step variants, just different rank/quality
        # tradeoffs). Treat any `nunchaku_*` as Lightning.
        return any(k in s for k in
                    ("lightning", "distill", "rapid", "lcm", "turbo",
                     "nunchaku", "ultimate_speed", "best_quality",
                     "_balance_", "_speed_"))

    arch = _arch(fname)
    distilled = _is_distilled(fname)

    # Metadata fallback for prequant_ours exports that lack hints in filename
    if arch is None or (arch == "qwen" and not distilled):
        try:
            from .format_adapters.tools import read_safetensors_metadata
            meta = read_safetensors_metadata(transformer_path) or {}
            blob = (str(meta.get("method", "")) + " "
                    + str(meta.get("_class_name", ""))
                    + " " + str(meta.get("model_id", ""))).lower()
            if arch is None:
                arch = _arch(blob)
            if not distilled:
                distilled = _is_distilled(blob)
        except Exception as e:
            logger.debug("scheduler metadata probe failed for %s: %s",
                          transformer_path, e)

    if arch == "z_image":
        # Currently only turbo is verified; non-turbo z-image gets the same
        # preset as a best-effort default.
        return "z_image_turbo"
    if arch == "klein":
        # No Klein Lightning released yet; if user labels one as lightning
        # we still return klein_base (safer than wrong shift).
        return "klein_base"
    if arch == "qwen":
        return "qwen_lightning" if distilled else "qwen_base"

    # Unknown architecture: only intervene if clearly distilled
    if distilled:
        return "qwen_lightning"
    return None


def _materialize_preset(preset_key: str) -> Optional[str]:
    """Write the preset dict to <plugin>/configs/_auto/<key>.json once and
    return its path. Returns None if the preset is intentionally null
    (TODO marker — engine default is preferable to a wrong config)."""
    preset = _SCHEDULER_PRESETS.get(preset_key)
    if not preset:
        return None
    plugin_root = os.path.dirname(os.path.abspath(__file__))
    auto_dir = os.path.join(plugin_root, "configs", "_auto")
    try:
        os.makedirs(auto_dir, exist_ok=True)
    except OSError:
        return None
    out_path = os.path.join(auto_dir, f"{preset_key}.json")
    if not os.path.isfile(out_path):
        try:
            import json as _json
            with open(out_path, "w", encoding="utf-8") as f:
                _json.dump(preset, f, indent=2)
        except OSError as e:
            logger.debug("failed to write %s: %s", out_path, e)
            return None
    return out_path


def detect_scheduler_config(transformer_path: str) -> Optional[str]:
    """Pick a scheduler config for the given transformer file.

    Returns absolute path to a materialized JSON, or None to let the
    engine use its built-in FlowMatchEulerDiscrete default (correct for
    Klein / ZImage today; the *base* qwen scheduler is materialized so
    Qwen base models get the diffusers-equivalent shift_terminal=0.02).

    Distilled / Lightning variants ALWAYS get a non-None result —
    without the right shift schedule, 4-8 step inference produces
    blurry / washed-out images.
    """
    key = _select_preset_key(transformer_path)
    if key is None:
        return None
    path = _materialize_preset(key)
    if path:
        logger.info("[scheduler] %s → preset=%s",
                     os.path.basename(transformer_path), key)
    return path


# Public API: imported by nodes_format_adapters.py for the unified BuildPipeline.
__all__ = [
    "install_loader_path_patches",
    "extract_qf_source_path",
    "build_precision_preset_options",
    "resolve_precision_preset",
    "detect_scheduler_config",
    "PRECISION_NONE_LABEL",
    "PRECISION_AUTO_LABEL",
]


# Re-export under stable public names
extract_qf_source_path = _extract_path
build_precision_preset_options = _build_precision_preset_options
resolve_precision_preset = _resolve_precision_preset
PRECISION_NONE_LABEL = _NONE_LABEL
PRECISION_AUTO_LABEL = _AUTO_LABEL
