"""Derive a COMPLETE component config from the weight tensors — the general,
layout/precision-agnostic synthesis layer.

Why this exists
---------------
The C++ engine reads architecture dims (``num_layers`` /
``num_attention_heads`` / ``joint_attention_dim`` / …) out of ``config.json``
and, when a key is ABSENT, falls back to a hard-coded **family-canonical
default** (e.g. Flux2Klein → the 9B shape: 32 heads, 8+24 blocks). A standalone
single-file model (ComfyUI "Load Diffusion Model" / "Load Checkpoint") ships NO
config.json, so the plugin synthesizes one — but historically it wrote only
``{"_class_name": …}``. A *size variant* (Klein **4B**: 24 heads, 5+20 blocks)
then inherits the wrong 9B default → the engine allocates 9B-sized blocks
against 4B weights → ``copy_`` source-buffer overflow / shape-mismatch crash at
load.

The weights are the ground truth. This module reads the tensor SHAPES + KEY
structure and derives the real dims, exactly like ComfyUI core's own
``model_detection.detect_unet_config`` derives everything from the state-dict.
It is:

  * **layout-agnostic** — handles BFL (``double_blocks`` / ``img_in`` /
    ``txt_in``) and diffusers (``transformer_blocks`` / ``x_embedder`` /
    ``context_embedder``) key layouts, with or without a
    ``model.diffusion_model.`` / ``diffusion_model.`` prefix.
  * **precision-agnostic** — a packed/quantized weight (FP8/INT4/FP4) loses its
    2-D ``.weight`` shape, so a 1-D ``.bias`` fallback is always tried.
  * **family-as-DATA** — every family is one entry in a descriptor table; a new
    model family is a data addition, NOT a code change. This is the
    "thoroughly general" mechanism: no per-family if-chains.

Used by ``HFLayout.add_transformer`` / ``add_text_encoder`` / ``add_vae`` /
``add_vision_encoder`` (+ the ``_remapped`` variants), so every adapter that
stages a component through those methods (comfyui_unet, nunchaku_svdq, hf_native,
bundled_checkpoint for transformer/TE/VAE) inherits weight-derived config
synthesis. ONE documented exception: ``bundled_checkpoint`` stages an embedded
edit-mode vision_encoder via ``set_key_filter('ve', …)`` (an in-bundle slice, not
a separate staging dir), so the VE-dim derive does NOT run on that path — the
bundled per-arch config supplies its dims there; the VE derive applies to the
standalone-VE path (hf_native). A field that is ALREADY correct in a real
diffusers/prequant config equals the derived value (same weights), so the
override is a no-op there; a wrong/absent/bundled-guess value gets corrected.
When a field cannot be derived with confidence it is OMITTED (the caller's
bundled config / the engine default then applies) — derivation never guesses.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from .safetensors_io import read_safetensors_header

logger = logging.getLogger(__name__)

# Attention head dim is 128 across all currently-supported QuantFunc transformer
# families (Klein / QwenImage / ZImage). Declared per-arch in the descriptor so
# a future family with a different head dim is a one-line DATA change.
_DEFAULT_HEAD_DIM = 128

# Sanity caps. A crafted/corrupt key (e.g. `transformer_blocks.2147483647.weight`)
# would otherwise size the engine to billions of blocks/heads → OOM / heap
# corruption. No real diffusion transformer approaches these (largest real stack
# ≈60 layers / ≈32 heads), so exceeding a cap means the file is untrusted →
# reject the WHOLE derivation and fall back to the safe bundled/engine default.
_MAX_SANE_LAYERS = 512
_MAX_SANE_HEADS = 256
# A single dimension (hidden/intermediate/vocab/in_channels/joint) above this is
# not a real model — it is an adversarial/corrupt header. Real values: vocab
# ≈256k, hidden ≈16k, intermediate ≈50k; 4M is far above any of them. A derived
# dim over the cap is DROPPED (falls back to the bundled/engine default) so a
# crafted shape like [4096, 2147483647] can't inject a giant int into config.json.
_MAX_SANE_DIM = 4 * 1024 * 1024
# Plausible attention head_dim range (Qwen/Klein use 128). Outside this, a
# crafted q_norm length must not drive the heads computation.
_MIN_HEAD_DIM = 8
_MAX_HEAD_DIM = 1024
# A `weight_map` (untrusted index.json) with more entries than this is rejected
# before materialization (DoS: millions of entries → huge alloc + isfile storm).
_MAX_SANE_SHARDS = 10000


def _sane_dim(v) -> bool:
    """True iff v is a plausible single architecture dimension (0 < v ≤ cap).
    Excludes bool: Python `True` is an int subclass, so a spec-violating header
    `"shape": [4096, true]` would otherwise write a JSON `true` into config.json."""
    return isinstance(v, int) and not isinstance(v, bool) and 0 < v <= _MAX_SANE_DIM


# ── shard fan-out ────────────────────────────────────────────────────────────
def _all_related_safetensors(path: str) -> list[str]:
    """Return ``[path]`` for a single-file model, or every shard path when
    ``path`` is one shard of a sharded model (detected via ``<path>.index.json``
    or ``<prefix>-NNNNN-of-NNNNN.safetensors`` siblings)."""
    if not path or not os.path.isfile(path):
        return [path] if path else []
    d = os.path.dirname(path)
    idx = path + ".index.json"
    if os.path.exists(idx):
        try:
            with open(idx) as f:
                meta = json.load(f)
            weight_map = meta.get("weight_map", {})
            # Reject an absurd entry count BEFORE materializing — a crafted
            # index.json with millions of entries would otherwise blow up memory
            # and trigger an isfile() storm (DoS) even though each path is later
            # traversal-checked.
            if len(weight_map) > _MAX_SANE_SHARDS:
                raise ValueError(
                    f"index.json weight_map has {len(weight_map)} entries "
                    f"(> cap {_MAX_SANE_SHARDS}); refusing (untrusted/corrupt)")
            shards = sorted(set(weight_map.values()))
            # `weight_map` shard names come from an untrusted file → a crafted
            # name like "../../etc/x.safetensors" must NOT escape the model dir
            # (path traversal). Keep only paths that resolve inside `d`.
            real_d = os.path.realpath(d)
            got = []
            for s in shards:
                joined = os.path.join(d, s)
                if not os.path.realpath(joined).startswith(real_d + os.sep):
                    continue
                if os.path.isfile(joined):
                    got.append(joined)
            if got:
                return got
        except Exception:
            pass
    name = os.path.basename(path)
    m = re.match(r"^(.+?)-(\d{5})-of-(\d{5})\.safetensors$", name)
    if m:
        prefix = m.group(1)
        try:
            shards = sorted(
                f for f in os.listdir(d)
                if re.match(rf"^{re.escape(prefix)}-\d{{5}}-of-\d{{5}}\.safetensors$", f)
            )
            if shards:
                return [os.path.join(d, f) for f in shards]
        except OSError:
            pass
    return [path]


def _merged_header(path: str) -> dict:
    """Union of the (non-metadata) safetensors header entries across all shards.
    Each value is the upstream header dict (carries ``shape`` / ``dtype``)."""
    keys: dict = {}
    for p in _all_related_safetensors(path):
        try:
            h = read_safetensors_header(p)
        except Exception:
            continue
        for k, v in h.items():
            if not k.startswith("__"):
                keys[k] = v
    return keys


def _max_block_index(keys, block_name: str) -> int:
    """Largest N such that some key contains ``<block_name>.N.`` as a dotted-path
    segment (any leading prefix allowed). Returns -1 when absent."""
    pat = re.compile(rf"(?:^|\.){re.escape(block_name)}\.(\d+)\.")
    best = -1
    for k in keys:
        m = pat.search(k)
        if m:
            best = max(best, int(m.group(1)))
    return best


def _first_nonempty_shape(keys, suffixes) -> Optional[list]:
    """Shape of the first tensor whose key == suffix or ends with ``.<suffix>``
    and has a NON-EMPTY shape (packed scale tensors have shape ``[]`` and are
    skipped, so a quantized ``.weight`` correctly falls through to its ``.bias``
    sibling). Candidate ``suffixes`` are tried in priority order."""
    for suf in suffixes:
        for k, info in keys.items():
            if k == suf or k.endswith("." + suf):
                sh = info.get("shape") or []
                # Require all-int dims: a spec-violating header could carry a
                # float (e.g. "shape": [2560.0, 4096]) which would flow through
                # the `hidden // head_dim` arithmetic into a FLOAT
                # num_attention_heads that the engine's get<int>() rejects at
                # load. Skip such a shape (→ next candidate / None). This is the
                # single source-level guard for every shape consumer.
                if sh and all(isinstance(x, int) and not isinstance(x, bool) for x in sh):
                    return list(sh)
    return None


# ── TRANSFORMER: per-arch DESCRIPTOR TABLE (families are DATA) ────────────────
#
#   layer_counts : {config_field: [block-name, ...]}  — value = max(index)+1
#                  over ANY listed block-name.
#   hidden_from  : ordered tensor-key SUFFIXES whose dim-0 = transformer hidden
#                  width; a 2-D ``.weight``/``._qweight`` additionally yields
#                  ``in_channels`` from dim-1, a 1-D ``.bias`` is a fallback.
#   hidden_extra_from : ordered SUFFIXES used as a HIDDEN-ONLY fallback (dim-0 =
#                  hidden) when NONE of ``hidden_from`` is present — e.g. a
#                  klein/qwen *lighting/FP4* export ships neither ``img_in`` nor
#                  ``x_embedder`` (only its ``context_embedder`` + final norm
#                  carry the hidden dim). These keys' dim-1 is NOT in_channels
#                  (context_embedder dim-1 = joint), so they NEVER set
#                  in_channels — only hidden → num_attention_heads.
#   joint_from   : ordered SUFFIXES whose 2-D dim-1 = ``joint_attention_dim``
#                  (text cross-condition width).
#   head_dim     : num_attention_heads = hidden // head_dim.
#
# NOTE the ``._qweight`` / ``._weight`` variants: nunchaku SVDQ INT4 checkpoints
# replace the named ``.weight`` with a UUID-keyed packed ``._qweight`` that KEEPS
# the logical 2-D [out, in] shape (e.g. Klein `context_embedder._qweight` =
# [hidden, joint_attn]; `x_embedder._qweight` = [hidden, in_channels]). Listing
# them keeps derivation working on SVDQ files (the old Klein-only probe handled
# `context_embedder._qweight`; the generalization must not regress it).
_TRANSFORMER_DESCRIPTORS: dict[str, dict] = {
    "Flux2Klein": {
        "layer_counts": {
            "num_layers":        ["double_blocks", "transformer_blocks"],
            "num_single_layers": ["single_blocks", "single_transformer_blocks"],
        },
        "hidden_from": ["img_in.weight", "img_in.bias", "img_in._qweight",
                        "x_embedder.weight", "x_embedder.bias", "x_embedder._qweight"],
        # klein-9b/4b LIGHTING/FP4 exports ship neither img_in nor x_embedder
        # (only x_embedder._wscales, a 1-D scale) → hidden falls through to the
        # bundled 4B template (24 heads → 3072) → norm_out [3072]!=[4096] load
        # crash on a real 9B. Derive hidden from dim-0 of the context embedder /
        # final norm instead (present in every klein export). HIDDEN-ONLY: dim-1
        # of context_embedder is JOINT, never in_channels.
        "hidden_extra_from": ["context_embedder.weight", "context_embedder._qweight",
                              "context_embedder._weight", "norm_out.norm.weight",
                              "norm_out.norm.bias"],
        "joint_from":  ["txt_in.weight", "txt_in._qweight",
                        "context_embedder.weight", "context_embedder._qweight",
                        "context_embedder._weight"],
        "head_dim":    _DEFAULT_HEAD_DIM,
    },
    "QwenImage": {
        "layer_counts": {"num_layers": ["transformer_blocks"]},
        "hidden_from": ["img_in.weight", "img_in.bias", "img_in._qweight",
                        "x_embedder.weight", "x_embedder.bias", "x_embedder._qweight"],
        # PREEMPTIVE / UNVERIFIED (验证契约 rule 3) — layout-symmetric with Flux2Klein
        # but NOT exercised by any deployed QwenImage export. MEASURED 2026-06-30:
        # QwenImage lighting/FP4 (50x-above) AND INT4 (30x-below) BOTH ship
        # img_in.weight [hidden,in_ch] → the PRIMARY hidden_from fires and this
        # fallback NEVER runs (staging A/B: OLD == FIXED, heads=24, no regression).
        # QwenImage also has no context_embedder/norm_out.norm key today, so these
        # would not match anyway. Kept for the (currently hypothetical) future
        # QwenImage export that ships neither img_in nor x_embedder; will VALIDATE
        # the dim-0==hidden assumption when such a checkpoint actually ships.
        "hidden_extra_from": ["context_embedder.weight", "context_embedder._qweight",
                              "context_embedder._weight", "norm_out.norm.weight",
                              "norm_out.norm.bias"],
        "joint_from":  ["txt_in.weight", "txt_in._qweight",
                        "context_embedder.weight", "context_embedder._qweight",
                        "context_embedder._weight"],
        "head_dim":    _DEFAULT_HEAD_DIM,
    },
    "ZImage": {
        "layer_counts": {"num_layers": ["layers", "transformer_blocks"]},
        # BFL ZImage single-file staging is zero-copy (a SYMLINK to the original
        # + a key_remap.json manifest — zimage_bfl_remap), so the staged file
        # keeps BFL `x_embedder.*` keys, matched here. The diffusers/remapped
        # form `all_x_embedder.2-1.*` (the manifest's rename target) is ALSO
        # listed so derivation works whichever key form the file presents.
        "hidden_from": ["x_embedder.weight", "x_embedder.bias",
                        "all_x_embedder.2-1.weight", "all_x_embedder.2-1.bias",
                        "img_in.weight", "img_in.bias"],
        "joint_from":  [],  # ZImage cap-cross width: not single-tensor derivable → engine default
        "head_dim":    _DEFAULT_HEAD_DIM,
        # ZImage x_embedder.weight dim-1 is the PATCHIFIED width
        # (frames*patch^2*in_channels), NOT in_channels — so DON'T derive
        # in_channels from it (it would emit e.g. 64 instead of 16). The engine
        # hardcodes in_channels=16 for ZImage; leaving it unset keeps that.
        "derive_in_channels": False,
    },
}
# QwenImageEdit shares QwenImage's transformer layout (same class, edit_mode).
_TRANSFORMER_DESCRIPTORS["QwenImageEdit"] = _TRANSFORMER_DESCRIPTORS["QwenImage"]
# QwenImageLayered also reuses QwenImageTransformer2DModel (same transformer_blocks
# layout; only use_additional_t_cond adds an extra time-cond embedding) → identical
# weight-derived dims (num_layers / num_attention_heads / head_dim).
_TRANSFORMER_DESCRIPTORS["QwenImageLayered"] = _TRANSFORMER_DESCRIPTORS["QwenImage"]


def derive_transformer_config(path: str, arch: Optional[str]) -> Optional[dict]:
    """Return the architecture-dim overrides derivable from the weight tensors
    for ``arch`` (one of the descriptor keys), or ``None`` when nothing could be
    derived. Only high-confidence fields are returned; absent ones are omitted
    so the caller's bundled config / engine default applies."""
    desc = _TRANSFORMER_DESCRIPTORS.get(arch or "")
    if not desc:
        return None
    keys = _merged_header(path)
    if not keys:
        return None

    out: dict = {}
    # 1. block counts → num_layers / num_single_layers (reject adversarial counts)
    for field, block_names in desc["layer_counts"].items():
        best = -1
        for b in block_names:
            best = max(best, _max_block_index(keys, b))
        if best >= 0:
            n = best + 1
            if n > _MAX_SANE_LAYERS:
                return None  # untrusted/corrupt key index → bail to safe default
            out[field] = n

    # 2. hidden width → num_attention_heads (+ in_channels from a 2-D embed)
    head_dim = desc.get("head_dim", _DEFAULT_HEAD_DIM)
    hidden = None
    # 2a. PRIMARY: img_in / x_embedder — dim-0 = hidden AND (2-D) dim-1 =
    # in_channels. This is the only source that may set in_channels.
    embed_shape = _first_nonempty_shape(keys, desc.get("hidden_from", []))
    if embed_shape:
        hidden = embed_shape[0]
        # in_channels = embed weight dim-1 ONLY where that equals the real
        # in_channels (Klein/QwenImage img_in.weight = [hidden, in_channels]).
        # ZImage opts out (derive_in_channels=False): its x_embedder dim-1 is the
        # patchified width, not in_channels — emitting it would be wrong.
        if (len(embed_shape) >= 2 and desc.get("derive_in_channels", True)
                and _sane_dim(embed_shape[1])):
            out["in_channels"] = embed_shape[1]
    # 2b. HIDDEN-ONLY FALLBACK: a lighting/FP4 export ships NEITHER img_in nor
    # x_embedder (klein-9b-*-lighting) — derive hidden from dim-0 of the context
    # embedder / final norm. NEVER touches in_channels (context_embedder dim-1 =
    # joint; a 1-D norm has no dim-1) → in_channels stays from the bundled config
    # (4B/9B share it), uncorrupted. Without this, 9B hidden falls through to the
    # bundled 4B default (24 heads → 3072) → norm_out [3072]!=[4096] crash.
    if hidden is None:
        hidden_shape = _first_nonempty_shape(keys, desc.get("hidden_extra_from", []))
        if hidden_shape:
            hidden = hidden_shape[0]
    if hidden and head_dim and hidden % head_dim == 0:
        heads = hidden // head_dim
        # heads must be a plausible positive count: a crafted hidden < head_dim
        # gives heads==0, and a giant hidden gives an implausible count.
        if heads < 1 or heads > _MAX_SANE_HEADS:
            return None  # implausible head count → untrusted weights → bail
        out["num_attention_heads"] = heads
        out["attention_head_dim"] = head_dim

    # 3. joint_attention_dim (text cross-condition width) from a 2-D text embed
    joint_shape = _first_nonempty_shape(keys, desc.get("joint_from", []))
    if joint_shape and len(joint_shape) >= 2 and _sane_dim(joint_shape[1]):
        out["joint_attention_dim"] = joint_shape[1]

    return out or None


# ── VAE: family signature table (generalizes engine #257 into the plugin) ─────
#
# A standalone "Load VAE" file ships no config.json, so ``_class_name`` defaults
# to the generic 2-D ``AutoencoderKL`` → the engine builds the wrong decoder
# against a 3-D AutoencoderKLQwenImage / Flux2 VAE → load crash (#257). The VAE
# family is unambiguous from a few decoder key fragments. ``all`` fragments must
# be present (avoids false-positives); first matching entry wins.
_VAE_SIGNATURES: list[tuple[str, list[str]]] = [
    # AutoencoderKLQwenImage (3-D temporal VAE) ships in TWO on-disk layouts:
    #   • native ComfyUI standalone "Load VAE" file (the #257 case):
    #       decoder.upsamples. + decoder.conv1.
    #   • diffusers model_dir layout:
    #       decoder.up_blocks. + the 3-D `time_conv` temporal marker. A 2-D
    #       AutoencoderKL / Flux2 VAE never has time_conv (verified absent from
    #       flux2-vae + z_ae), so up_blocks+time_conv is unambiguous and cannot
    #       false-positive on Flux2 (which has up_blocks but no time_conv).
    ("AutoencoderKLQwenImage", ["decoder.upsamples.", "decoder.conv1."]),
    ("AutoencoderKLQwenImage", ["decoder.up_blocks.", "time_conv"]),
    # AutoencoderKLFlux2 — Klein VAE carries a root-level `bn.` batchnorm sidecar
    # (bn.running_mean/var) on top of the diffusers 2-D decoder layout, which the
    # standard AutoencoderKL never has.
    ("AutoencoderKLFlux2", ["bn.running_", "decoder.mid_block."]),
]


def derive_vae_class(path: str) -> Optional[str]:
    """Return the VAE ``_class_name`` identified from the weight keys, or
    ``None`` when no specific family signature matches (caller keeps its
    declared class / the generic AutoencoderKL default)."""
    keys = _merged_header(path)
    if not keys:
        return None
    for class_name, fragments in _VAE_SIGNATURES:
        if all(any(frag in k for k in keys) for frag in fragments):
            return class_name
    return None


# ── TEXT ENCODER: derive config dims from the TE weights (#267) ───────────────
#
# The engine reads TE dims (hidden_size / num_hidden_layers / heads / …) from
# config.json. The plugin's bundled per-arch TE config
# (bin/text_encoder_configs/<arch>.json) is a SINGLE size, so a same-family size
# variant loaded via a bare "Load CLIP" (no model_dir config.json) inherits the
# WRONG hidden_size: Klein bundles Qwen3-2560 (4B), but the 9B TE is 4096 → the
# engine ALLOCATES TE buffers at 2560 then loads the 4096 weights → copy_ shape
# mismatch → noise (#267). Derive the size dims from the actual TE weights — the
# standard HF/Qwen layout (`*embed_tokens.weight` + `*layers.N.*` +
# `self_attn.{q,k,v}_proj` + `mlp.{gate,up}_proj` + optional `self_attn.q_norm`).
# Same derive-from-weights class as the transformer dims and the VAE family.
_TE_HEAD_DIM_DEFAULT = _DEFAULT_HEAD_DIM  # Qwen3 / Qwen2.5(-VL) head dim; overridden by q_norm size when present.


def derive_te_config(path: str) -> Optional[dict]:
    """Return the text-encoder config dims derivable from the TE weights at
    `path`, or None when nothing could be derived. Only high-confidence,
    size-varying fields are returned (the bundled/sibling config supplies the
    non-derivable fields: model_type, rope_theta, rms_norm_eps, …). For an
    already-correct config the derived values EQUAL it, so the merge is a
    no-op (verified on Qwen3-4B + Qwen2.5-VL-7B)."""
    keys = _merged_header(path)
    if not keys:
        return None

    # Anchor on the token embedding. If absent, this file is not a (Qwen-style)
    # text encoder we can derive from — return None rather than guess from
    # foreign keys. The embedding's prefix also SCOPES every other lookup to the
    # TE's own subtree, so a co-bundled transformer (which may reuse `layers.N` /
    # `mlp.gate_proj` / `self_attn.q_proj` key names — notably ZImage uses
    # `layers.N`) cannot contaminate ANY derived TE dim. (`._qweight` variants
    # cover SVDQ-packed TEs; `embed_tokens` itself is never quantized.)
    # Anchor on the MOST-SPECIFIC (longest) embed_tokens key — deterministic and
    # dict-order-independent. `next(...)` could otherwise pick a bare
    # `embed_tokens.weight` over a prefixed `text_encoders.X.…embed_tokens.weight`
    # if dict order put the bare one first → root="" → te_keys = ALL keys → a
    # co-bundled transformer's `layers.N`/`mlp.gate_proj` would contaminate the TE
    # dims. The longest match is the most-prefixed = narrowest subtree = safest.
    emb_key = max((k for k in keys
                   if k == "embed_tokens.weight" or k.endswith("embed_tokens.weight")),
                  key=len, default=None)
    if emb_key is None:
        return None
    root = emb_key[:emb_key.rfind("embed_tokens.weight")]
    te_keys = {k: v for k, v in keys.items() if k.startswith(root)}

    out: dict = {}
    # hidden_size + vocab_size from the token embedding [vocab, hidden]
    emb = te_keys[emb_key].get("shape") or []
    if len(emb) >= 2:
        if _sane_dim(emb[0]):
            out["vocab_size"] = emb[0]
        if _sane_dim(emb[1]):
            out["hidden_size"] = emb[1]

    # num_hidden_layers — max `<root>layers.N` index within the TE subtree.
    layer_prefix = root + "layers."
    nl = -1
    for k in te_keys:
        if k.startswith(layer_prefix):
            seg = k[len(layer_prefix):].split(".", 1)[0]
            if seg.isdigit():
                nl = max(nl, int(seg))
    if nl >= 0:
        if nl + 1 > _MAX_SANE_LAYERS:
            return None  # untrusted/corrupt key index → bail to safe default
        out["num_hidden_layers"] = nl + 1

    # head_dim from Qwen3 per-head RMSNorm (`q_norm.weight` = [head_dim]); else
    # the family default. Only EMITTED when measured AND in a plausible range
    # (a crafted huge q_norm must not drive the heads computation to 0).
    head_dim = _TE_HEAD_DIM_DEFAULT
    qn = _first_nonempty_shape(te_keys, ["self_attn.q_norm.weight", "q_norm.weight"])
    if qn and _MIN_HEAD_DIM <= qn[0] <= _MAX_HEAD_DIM:
        head_dim = qn[0]
        out["head_dim"] = head_dim

    # attention / kv heads from the q / k projection out-dims (// head_dim)
    q = _first_nonempty_shape(te_keys, ["self_attn.q_proj.weight", "self_attn.q_proj._qweight"])
    if q and head_dim and q[0] % head_dim == 0:
        heads = q[0] // head_dim
        if heads < 1 or heads > _MAX_SANE_HEADS:
            return None  # implausible head count → untrusted weights → bail
        out["num_attention_heads"] = heads
    kv = _first_nonempty_shape(te_keys, ["self_attn.k_proj.weight", "self_attn.k_proj._qweight"])
    if kv and head_dim and kv[0] % head_dim == 0:
        kvh = kv[0] // head_dim
        if 1 <= kvh <= _MAX_SANE_HEADS:
            out["num_key_value_heads"] = kvh

    # MLP intermediate width from the gate (or up) projection out-dim
    inter = _first_nonempty_shape(te_keys, ["mlp.gate_proj.weight", "mlp.up_proj.weight",
                                            "mlp.gate_proj._qweight", "mlp.up_proj._qweight"])
    if inter and _sane_dim(inter[0]):
        out["intermediate_size"] = inter[0]

    return out or None


def derive_vision_encoder_config(path: str) -> Optional[dict]:
    """Return the vision-encoder (Qwen2.5-VL `visual.*`) config dims derivable
    from the weights, or None. Mirrors derive_te_config for the edit-pipeline
    vision tower: a bare standalone vision encoder with no sibling config would
    otherwise inherit a wrong hidden_size → the SAME #267-class shape mismatch.

    Derives only the reliably-shape-determined fields: `hidden_size` (the vision
    transformer width) and `depth` (block count). `num_heads` is intentionally
    NOT derived — it needs head_dim, which a single weight shape cannot
    disambiguate (1280 = 16×80 or 10×128 …) — so it is left to the sibling
    config / engine default. `out_hidden_size`/`intermediate_size` likewise."""
    keys = _merged_header(path)
    if not keys:
        return None
    out: dict = {}
    # vision hidden width — patch_embed projection out-dim, else a block norm.
    # A SVDQ vision encoder packs `patch_embed.proj` into `._qweight_w4a4` (no
    # `.weight`) and drops the `visual.` prefix, so the bare `blocks.0.norm1.weight`
    # (unpacked, = [hidden]) is the reliable fallback there.
    hid = _first_nonempty_shape(
        keys, ["visual.patch_embed.proj.weight", "patch_embed.proj.weight",
               "visual.blocks.0.norm1.weight", "blocks.0.norm1.weight"])
    if hid and _sane_dim(hid[0]):
        out["hidden_size"] = hid[0]
    # depth — max `visual.blocks.N` index (vision uses `blocks`, the LLM uses
    # `layers`, so this never picks up the language stack).
    d = _max_block_index(keys, "blocks")
    if d >= 0:
        if d + 1 > _MAX_SANE_LAYERS:
            return None  # untrusted/corrupt key index → bail to safe default
        out["depth"] = d + 1
    return out or None
