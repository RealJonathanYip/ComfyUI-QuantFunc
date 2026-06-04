"""BFL → diffusers key remap for ZImage transformer + VAE.

ZERO-DATA-COPY design: we never rewrite the ~12 GB safetensors. The
staging dir contains a symlink to the original file plus a tiny JSON
manifest (`key_remap.json`) listing the alias rules:

  - "rename":      target_key → source_key                (zero-copy view)
  - "row_slice":   target_key → {source, slice, shape}     (stride view, dim 0)
  - "reshape":     target_key → {source, shape}            (no-op metadata)

The C++ engine's KeyAliasingProvider reads the JSON, wraps the
TensorsProvider, and on `getTensor(target)` returns the appropriate
sliced/reshaped view of the source mmap'd tensor — no bytes copied.

Detection signatures (BFL vs HF-diffusers):
  - Transformer: `*.attention.qkv.weight` exists (BFL fused QKV) +
                 `cap_embedder.0` + `(layers|noise_refiner).N.*`
  - VAE:         `*.mid.attn_1.q.weight` + `(up|down).N.block.N.*`
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


_PREFIX_TO_STRIP = "model.diffusion_model."
_BLOCK_PREFIXES = ("layers.", "noise_refiner.", "context_refiner.")


def _manifest_cache_dir() -> Path:
    """Stable per-user cache for built manifests, persisting across ComfyUI
    BuildPipeline invocations. Default: `<plugin>/cache/manifests/`. Override
    via env `QF_MANIFEST_CACHE_DIR`. Manifest filenames embed source mtime
    + size so a re-quantised / replaced source file invalidates the cache.
    """
    env = os.environ.get("QF_MANIFEST_CACHE_DIR")
    if env:
        d = Path(env)
    else:
        # Plugin root: this file lives at <plugin>/format_adapters/zimage_bfl_remap.py
        plugin_root = Path(__file__).resolve().parent.parent
        d = plugin_root / "cache" / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(src: Path, kind: str) -> Path:
    """Cache filename: `<kind>_<basename>_<size>_<mtime_ns>.json`."""
    st = src.stat()
    safe = src.name.replace("/", "_").replace(" ", "_")
    return _manifest_cache_dir() / f"{kind}_{safe}_{st.st_size}_{st.st_mtime_ns}.json"


def _read_header(src_path: Path) -> dict:
    """Return the safetensors header dict (without __metadata__)."""
    with open(src_path, "rb") as f:
        sz_bytes = f.read(8)
        header_size = struct.unpack("<Q", sz_bytes)[0]
        hdr = json.loads(f.read(header_size).decode("utf-8"))
    hdr.pop("__metadata__", None)
    return hdr


def is_bfl_zimage(src_path: Path | str) -> bool:
    """Detect ZImage BFL transformer layout (fused QKV + cap_embedder)."""
    try:
        hdr = _read_header(Path(src_path))
    except Exception as e:
        logger.debug("is_bfl_zimage: header read failed: %s", e)
        return False
    keys = list(hdr.keys())
    has_qkv = any(k.endswith("attention.qkv.weight") for k in keys)
    has_cap = any(".cap_embedder." in k or k.endswith(".cap_embedder.0.weight")
                   or "cap_embedder.0" in k for k in keys)
    has_layers = any(re.search(r"\.(?:layers|noise_refiner)\.\d+\.", k)
                     for k in keys)
    return has_qkv and has_cap and has_layers


def is_bfl_vae(src_path: Path | str) -> bool:
    """Detect BFL-style VAE (mid.attn_1.q + up.N.block.N)."""
    try:
        hdr = _read_header(Path(src_path))
    except Exception:
        return False
    keys = list(hdr.keys())
    has_bfl_mid = any(".mid.attn_1.q.weight" in k or k.endswith("mid.attn_1.q.weight")
                      for k in keys)
    has_bfl_block = any(re.search(r"\.(?:up|down)\.\d+\.block\.\d+\.", k)
                        for k in keys)
    return has_bfl_mid and has_bfl_block


# ============================================================================
# Transformer remap manifest builder
# ============================================================================

def build_zimage_xfm_manifest(src_path: Path) -> dict:
    """Generate a remap manifest for a BFL ZImage transformer file.

    The manifest is consumed by the C++ KeyAliasingProvider to translate
    `getTensor(diffusers_key)` requests into mmap views of source bytes.
    """
    hdr = _read_header(Path(src_path))
    rename: dict[str, str] = {}
    row_slice: dict[str, dict] = {}

    for src_key, info in hdr.items():
        # Strip `model.diffusion_model.` prefix for transformation logic;
        # but we always reference the FULL src_key in the manifest since
        # the C++ side reads the original safetensors file as-is.
        k = src_key
        if k.startswith(_PREFIX_TO_STRIP):
            k = k[len(_PREFIX_TO_STRIP):]
        shape = info["shape"]

        # Top-level renames
        if k.startswith("final_layer."):
            tgt = "all_final_layer.2-1." + k[len("final_layer."):]
            rename[tgt] = src_key
            continue
        if k.startswith("x_embedder."):
            tgt = "all_x_embedder.2-1." + k[len("x_embedder."):]
            rename[tgt] = src_key
            continue

        # Per-block attention rewrites (3 prefix groups: layers/noise_refiner/context_refiner)
        matched = False
        for px in _BLOCK_PREFIXES:
            if not k.startswith(px):
                continue
            rest = k[len(px):]
            dot = rest.find(".")
            if dot < 0:
                continue
            idx = rest[:dot]
            tail = rest[dot + 1:]
            matched = True

            if tail == "attention.qkv.weight":
                if len(shape) != 2 or shape[0] % 3 != 0:
                    raise RuntimeError(
                        f"qkv tensor {src_key!r}: expected [3D, D], got {shape}")
                D = shape[0] // 3
                D_in = shape[1]
                row_slice[f"{px}{idx}.attention.to_q.weight"] = {
                    "source": src_key, "slice": [0, D],     "shape": [D, D_in],
                }
                row_slice[f"{px}{idx}.attention.to_k.weight"] = {
                    "source": src_key, "slice": [D, 2 * D], "shape": [D, D_in],
                }
                row_slice[f"{px}{idx}.attention.to_v.weight"] = {
                    "source": src_key, "slice": [2 * D, 3 * D], "shape": [D, D_in],
                }
            elif tail == "attention.out.weight":
                rename[f"{px}{idx}.attention.to_out.0.weight"] = src_key
            elif tail == "attention.q_norm.weight":
                rename[f"{px}{idx}.attention.norm_q.weight"] = src_key
            elif tail == "attention.k_norm.weight":
                rename[f"{px}{idx}.attention.norm_k.weight"] = src_key
            else:
                # Pass-through: rename target == post-prefix-strip key.
                # Diffusers consumer expects key WITHOUT model.diffusion_model.
                # prefix; engine uses the staging-dir's `set_key_strip`
                # mechanism for that. So here we add an entry only if the
                # strip alone doesn't suffice (i.e. always — we record every
                # mapping for clarity).
                rename[k] = src_key
            break

        if not matched:
            # Top-level pass-through (cap_embedder, t_embedder, cap_pad_token,
            # x_pad_token, etc.). The engine wants HF-name (without
            # `model.diffusion_model.` prefix).
            rename[k] = src_key

    return {"rename": rename, "row_slice": row_slice, "reshape": {}}


# ============================================================================
# VAE remap manifest builder
# ============================================================================

_BFL_VAE_ATTN_LINEAR_TAILS = (
    "to_q.weight", "to_k.weight", "to_v.weight",
    "to_out.0.weight",
)


def _detect_num_up_blocks(keys: list[str]) -> int:
    indices: set[int] = set()
    for k in keys:
        m = re.match(r"^decoder\.up\.(\d+)\.", k)
        if m:
            indices.add(int(m.group(1)))
    return (max(indices) + 1) if indices else 0


def _remap_one_vae_key(k: str, num_up_blocks: int) -> Optional[str]:
    """BFL → HF AutoencoderKL key remap (string only — no shape/dtype here)."""
    new = k
    new = re.sub(r"^(encoder|decoder)\.norm_out\.", r"\1.conv_norm_out.", new)
    new = re.sub(r"^(encoder|decoder)\.mid\.attn_1\.norm\.",
                 r"\1.mid_block.attentions.0.group_norm.", new)
    new = re.sub(r"^(encoder|decoder)\.mid\.attn_1\.proj_out\.",
                 r"\1.mid_block.attentions.0.to_out.0.", new)
    new = re.sub(r"^(encoder|decoder)\.mid\.attn_1\.q\.",
                 r"\1.mid_block.attentions.0.to_q.", new)
    new = re.sub(r"^(encoder|decoder)\.mid\.attn_1\.k\.",
                 r"\1.mid_block.attentions.0.to_k.", new)
    new = re.sub(r"^(encoder|decoder)\.mid\.attn_1\.v\.",
                 r"\1.mid_block.attentions.0.to_v.", new)

    def _mid_block_sub(m):
        side, b_idx, tail = m.group(1), m.group(2), m.group(3)
        return f"{side}.mid_block.resnets.{int(b_idx) - 1}.{tail}"
    new = re.sub(r"^(encoder|decoder)\.mid\.block_(\d+)\.(.+)$", _mid_block_sub, new)

    def _up_block_sub(m):
        return (f"decoder.up_blocks.{num_up_blocks - 1 - int(m.group(1))}"
                f".resnets.{m.group(2)}.")
    def _up_sample_sub(m):
        return (f"decoder.up_blocks.{num_up_blocks - 1 - int(m.group(1))}"
                f".upsamplers.0.conv.")
    new = re.sub(r"^decoder\.up\.(\d+)\.block\.(\d+)\.", _up_block_sub, new)
    new = re.sub(r"^decoder\.up\.(\d+)\.upsample\.conv\.", _up_sample_sub, new)
    new = re.sub(r"^encoder\.down\.(\d+)\.block\.(\d+)\.",
                 r"encoder.down_blocks.\1.resnets.\2.", new)
    new = re.sub(r"^encoder\.down\.(\d+)\.downsample\.conv\.",
                 r"encoder.down_blocks.\1.downsamplers.0.conv.", new)
    new = new.replace(".nin_shortcut.", ".conv_shortcut.")
    return new


def build_zimage_vae_manifest(src_path: Path) -> dict:
    """Generate a remap manifest for a BFL ZImage VAE file."""
    hdr = _read_header(Path(src_path))
    num_up = _detect_num_up_blocks(list(hdr.keys()))
    if num_up == 0:
        raise RuntimeError(
            f"VAE {src_path}: no `decoder.up.N.*` keys; cannot build manifest")

    rename: dict[str, str] = {}
    reshape: dict[str, dict] = {}

    for src_key, info in hdr.items():
        new_key = _remap_one_vae_key(src_key, num_up)
        if new_key is None:
            continue
        shape = info["shape"]
        # Conv2d 1×1 attention weights → Linear: drop trailing [..., 1, 1]
        if (len(shape) == 4 and shape[-1] == 1 and shape[-2] == 1
                and any(new_key.endswith(t) for t in _BFL_VAE_ATTN_LINEAR_TAILS)):
            reshape[new_key] = {
                "source": src_key, "shape": [shape[0], shape[1]],
            }
        else:
            rename[new_key] = src_key

    return {"rename": rename, "row_slice": {}, "reshape": reshape}


# ============================================================================
# Staging helpers (zero-data-copy: symlink + manifest JSON)
# ============================================================================

def stage_bfl_zimage(src_path: str, staging_transformer_dir: Path,
                       *, force: bool = False) -> Path:
    """Stage a BFL ZImage transformer with zero data copy.

    Writes:
      <staging_transformer_dir>/diffusion_pytorch_model.safetensors  (symlink)
      <staging_transformer_dir>/key_remap.json                       (~10 KB)
    """
    src = Path(src_path).resolve()
    dst = staging_transformer_dir / "diffusion_pytorch_model.safetensors"
    manifest_path = staging_transformer_dir / "key_remap.json"

    if (dst.exists() or dst.is_symlink()) and force:
        dst.unlink()
    if not (dst.exists() or dst.is_symlink()):
        staging_transformer_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(str(src), str(dst))

    if manifest_path.exists() and not force:
        return dst

    cache_path = _cache_key(src, "zimage_xfm")
    if cache_path.exists() and not force:
        # Hardlink (or copy if cross-fs) the cached manifest.
        # Symlink to cache (zero bytes; works across filesystems unlike
        # hardlink). Hardlink would also be valid on same FS but can fail
        # silently on /tmp ↔ /media boundaries.
        try:
            os.symlink(str(cache_path), str(manifest_path))
        except OSError:
            import shutil
            shutil.copyfile(str(cache_path), str(manifest_path))
        logger.info("[zimage_bfl_remap] %s → manifest from cache", src.name)
        return dst

    manifest = build_zimage_xfm_manifest(src)
    cache_path.write_text(json.dumps(manifest, indent=2))
    try:
        os.link(str(cache_path), str(manifest_path))
    except OSError:
        import shutil
        shutil.copyfile(str(cache_path), str(manifest_path))
    n_rename = len(manifest["rename"])
    n_split = len(manifest["row_slice"])
    logger.info("[zimage_bfl_remap] %s → manifest built+cached "
                 "(%d renames, %d QKV splits, %.1f KB, cache=%s)",
                 src.name, n_rename, n_split,
                 cache_path.stat().st_size / 1024, cache_path.name)
    return dst


def stage_bfl_vae(src_path: str, staging_vae_dir: Path,
                    *, force: bool = False) -> Path:
    """Stage a BFL VAE with zero data copy."""
    src = Path(src_path).resolve()
    dst = staging_vae_dir / "diffusion_pytorch_model.safetensors"
    manifest_path = staging_vae_dir / "key_remap.json"

    if (dst.exists() or dst.is_symlink()) and force:
        dst.unlink()
    if not (dst.exists() or dst.is_symlink()):
        staging_vae_dir.mkdir(parents=True, exist_ok=True)
        os.symlink(str(src), str(dst))

    if manifest_path.exists() and not force:
        return dst

    cache_path = _cache_key(src, "zimage_vae")
    if cache_path.exists() and not force:
        # Symlink to cache (zero bytes; works across filesystems unlike
        # hardlink). Hardlink would also be valid on same FS but can fail
        # silently on /tmp ↔ /media boundaries.
        try:
            os.symlink(str(cache_path), str(manifest_path))
        except OSError:
            import shutil
            shutil.copyfile(str(cache_path), str(manifest_path))
        logger.info("[zimage_bfl_remap] %s → VAE manifest from cache", src.name)
        return dst

    manifest = build_zimage_vae_manifest(src)
    cache_path.write_text(json.dumps(manifest, indent=2))
    try:
        os.link(str(cache_path), str(manifest_path))
    except OSError:
        import shutil
        shutil.copyfile(str(cache_path), str(manifest_path))
    n_rename = len(manifest["rename"])
    n_reshape = len(manifest["reshape"])
    logger.info("[zimage_bfl_remap] %s → VAE manifest built+cached "
                 "(%d renames, %d reshapes, %.1f KB, cache=%s)",
                 src.name, n_rename, n_reshape,
                 cache_path.stat().st_size / 1024, cache_path.name)
    return dst
