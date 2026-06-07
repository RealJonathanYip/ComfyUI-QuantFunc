"""Adapter registry + dispatch entry point.

Adapters self-register via @adapter(priority=N). build_pipeline_inputs()
selects the first matching adapter and runs its adapt() to produce a staging
dir suitable for quantfunc_create().
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Type

from .base import (
    BuildContext,
    FormatAdapter,
    SourceBundle,
    StagingResult,
    UnsupportedFormatError,
    canonicalize_method_hint,
)

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Priority-sorted list of adapter classes.

    Higher priority is tried first. When two adapters match, the higher-
    priority one wins. Use these conventions:
      100  most-specific identification (our prequant via metadata)
       90  exact format markers (NVFP4 _quantization_metadata)
       80  multi-component bundles (bundled checkpoint)
       50  format-by-prefix (ComfyUI single-file UNETLoader)
       10  fallback
    """
    _entries: list[tuple[int, Type[FormatAdapter]]] = []

    @classmethod
    def register(cls, adapter_cls: Type[FormatAdapter], priority: int) -> None:
        cls._entries.append((priority, adapter_cls))
        cls._entries.sort(key=lambda x: -x[0])

    @classmethod
    def select(cls, sources: SourceBundle) -> FormatAdapter:
        for priority, adapter_cls in cls._entries:
            try:
                if adapter_cls.detect(sources):
                    logger.debug("[adapter] selected %s (priority=%d)",
                                  adapter_cls.__name__, priority)
                    return adapter_cls()
            except Exception as e:
                logger.warning("[adapter] %s.detect() raised %r — skipping",
                                adapter_cls.__name__, e)
        raise UnsupportedFormatError(
            "No format adapter matched these sources. Tried: "
            + ", ".join(c.__name__ for _, c in cls._entries))

    @classmethod
    def list_adapters(cls) -> list[tuple[int, str]]:
        """Return [(priority, name), ...] for diagnostics."""
        return [(p, c.__name__) for p, c in cls._entries]


def adapter(priority: int = 50):
    """Decorator: register a FormatAdapter subclass with the registry."""
    def deco(cls: Type[FormatAdapter]):
        AdapterRegistry.register(cls, priority)
        return cls
    return deco


def _stable_staging_id(sources: SourceBundle, context: BuildContext) -> str:
    """Hash (source files + relevant context) into a stable dir suffix.

    Re-runs of the same workflow with the same inputs land on the SAME
    staging dir, which keeps `model_dir` stable across invocations. The
    pipeline manager (nodes.py::_make_cache_key) keys on `model_dir`, so a
    drifting tmpdir name forces it to "Destroying orphan pipeline ... node
    changed config" and reload the model on every prompt — even though
    nothing actually changed. With a stable id, identical re-runs reuse
    the loaded pipeline.

    Hash inputs:
      - Each FileRef's (path, size, mtime) — content can be inferred from
        these and we avoid hashing GBs of weights.
      - BuildContext fields that semantically affect the staging output
        (text_precision, vae_precision, backend, precision_map_xfm path).
        Excludes api_key / device_idx / cosmetic fields that don't change
        the staging on disk.
    """
    h = hashlib.sha256()

    def _stamp(ref) -> dict:
        if ref is None:
            return {}
        try:
            st = os.stat(ref.path)
            return {"path": ref.path, "size": st.st_size,
                    "mtime": int(st.st_mtime)}
        except OSError:
            return {"path": ref.path, "size": 0, "mtime": 0}

    payload = {
        "transformer":  _stamp(getattr(sources, "transformer", None)),
        "text_encoder": _stamp(getattr(sources, "text_encoder", None)),
        "vae":          _stamp(getattr(sources, "vae", None)),
        "checkpoint":   _stamp(getattr(sources, "checkpoint", None)),
        "loras": [_stamp(l) for l in (getattr(sources, "loras", None) or [])],
        "scheduler_config": getattr(sources, "scheduler_config", None) or "",
        "ctx": {
            "text_precision":  getattr(context, "text_precision", ""),
            "vae_precision":   getattr(context, "vae_precision", ""),
            "backend":         getattr(context, "backend", ""),
            # precision_map_xfm may be a dict {path,target,preset} or None.
            # Only hash the `path` since that's what determines the layer
            # mix actually written to staging.
            "precision_map_xfm": (
                (getattr(context, "precision_map_xfm", None) or {}).get("path", "")
                if isinstance(getattr(context, "precision_map_xfm", None), dict)
                else str(getattr(context, "precision_map_xfm", "") or "")
            ),
        },
    }
    h.update(json.dumps(payload, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _choose_staging_root(sources: SourceBundle) -> str:
    """Pick where to build the staging tree.

    Priority:
      1. $QUANTFUNC_CACHE_DIR — explicit user override.
      2. On Windows, a folder on the SAME volume as the primary weight file, so
         the staging tree can be HARDLINKED into place. Windows hardlinks need
         no privilege (unlike symlinks, which raise WinError 1314 without
         Developer Mode / admin) and cost zero extra disk. Staging onto a
         different volume than the model (e.g. model on D:, system temp on C:)
         is exactly what forced symlinks and broke non-privileged users.
      3. System temp dir — the Linux default, where symlinks are unprivileged
         and work across filesystems.
    """
    env = os.environ.get("QUANTFUNC_CACHE_DIR")
    if env:
        return env
    ref = (sources.transformer or sources.checkpoint
           or sources.text_encoder or sources.vae)
    if ref is not None:
        try:
            drive = os.path.splitdrive(os.path.abspath(ref.path))[0]
        except Exception:
            drive = ""
        if drive:  # Windows-style drive letter (e.g. "D:") — stay on it.
            cand = os.path.join(drive + os.sep, "quantfunc_staging")
            try:
                os.makedirs(cand, exist_ok=True)
                return cand
            except OSError as e:
                logger.warning("[staging] cannot use %s (%s); falling back to "
                               "system temp", cand, e)
    return tempfile.gettempdir()


def build_pipeline_inputs(
    sources: SourceBundle,
    context: BuildContext,
    staging_root: str | None = None,
) -> StagingResult:
    """Top-level entry: pick an adapter and build a staging dir.

    Args:
        sources:        Inputs from BuildPipeline node (transformer/te/vae/lora).
        context:        Pipeline-wide config (precision maps, device, etc.).
        staging_root:   Where to create the staging dir. Defaults to
                        _choose_staging_root(sources): $QUANTFUNC_CACHE_DIR, else
                        (Windows) the model's drive so weights can be hardlinked
                        with no symlink privilege, else system tmpdir.

    Returns:
        StagingResult whose `model_dir` is suitable for quantfunc_create().
        Caller is responsible for cleaning up `cleanup_dir` if non-None.

    Caching: when the same (sources, context) hash to the same id and a
    completed staging dir already exists at `<staging_root>/quantfunc_staging_<id>/`
    (marker file `.staging_complete`), it is reused as-is and no adapter
    runs. This keeps `model_dir` stable across re-runs so the pipeline
    cache (keyed on `model_dir`) doesn't tear down on every prompt.
    """
    adapter_inst = AdapterRegistry.select(sources)

    if staging_root is None:
        staging_root = _choose_staging_root(sources)
    Path(staging_root).mkdir(parents=True, exist_ok=True)

    stable_id = _stable_staging_id(sources, context)
    staging_dir = Path(staging_root) / f"quantfunc_staging_{stable_id}"
    marker = staging_dir / ".staging_complete"

    # Reuse complete staging from a prior run on identical inputs.
    if marker.is_file():
        try:
            cached = json.loads(marker.read_text())
            res = StagingResult(
                model_dir=cached.get("model_dir", str(staging_dir)),
                arch=cached.get("arch", ""),
                method_hint=canonicalize_method_hint(cached.get("method_hint", "")),
                # cleanup_dir=None: we want to KEEP the cache around so
                # the next run can reuse it again. The pipeline manager
                # owns lifecycle of the loaded model; staging dirs are
                # disposable symlink trees, ~tens of KB.
                cleanup_dir=None,
            )
            logger.debug("[adapter] reused cached staging %s", staging_dir)
            return res
        except Exception as e:
            # Corrupted marker — fall through and rebuild.
            logger.warning("[adapter] marker unreadable, rebuilding: %s", e)

    # Stale partial staging from an interrupted run — clean it before
    # rebuilding so symlinks don't conflict.
    if staging_dir.exists():
        import shutil
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = adapter_inst.adapt(sources, staging_dir, context)
    except Exception:
        import shutil
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    # If adapter took the shortcut and didn't use staging_dir, remove the
    # empty placeholder we created — and skip writing the cache marker
    # since there's nothing here to reuse.
    if result.cleanup_dir is None and result.model_dir != str(staging_dir):
        import shutil
        shutil.rmtree(staging_dir, ignore_errors=True)
        return result

    # Mark staging as complete + persist the result so future runs hit the
    # cache. Always set cleanup_dir=None on the returned result so the
    # caller doesn't delete our cache after first use.
    try:
        marker.write_text(json.dumps({
            "model_dir":   result.model_dir,
            "arch":        result.arch,
            "method_hint": result.method_hint,
            "stable_id":   stable_id,
        }))
    except Exception as e:
        logger.warning("[adapter] couldn't write staging marker: %s", e)

    if result.cleanup_dir is not None:
        result = dataclasses.replace(result, cleanup_dir=None)
    return result
