"""Adapter for ComfyUI single-file UNETLoader / Load Diffusion Model output.

Two sub-cases, one adapter:

1. **ComfyUI-prefixed** (original path): the file carries
   ``model.diffusion_model.X`` or ``diffusion_model.X`` prefix keys.
   These are the standard UNETLoader / Load Diffusion Model outputs.

2. **Bare BFL / diffusers layout** (new, general fix): the file has NO
   ComfyUI prefix but the arch can be identified by
   ``fingerprint_arch_from_keys()`` (e.g. raw BFL Klein files with bare
   ``double_blocks.``/``single_blocks.`` keys, or bare diffusers QwenImage
   files). This covers any precision — BF16, FP8, INT4, FP4 — as long as
   the arch fingerprint recognises the key pattern.

In both sub-cases the file contains ONLY transformer weights; TE and VAE
come from separate files (LoadCLIP / LoadVAE nodes).

Detection priority is below PrequantOurs and NVFP4Disk: this is the
"generic" single-file transformer format applied when no more-specific
adapter matched.

Adaptation:
  staging/transformer/diffusion_pytorch_model.safetensors  → symlink to source
  staging/transformer/config.json                           (synthesized)
  staging/quantfunc_config.json:
    {"method": "online_quant",
     "transformer_key_strip": "<prefix>"}   # empty string for bare layout
  + symlink whatever TE / VAE was provided (handled by their own adapters
    via cooperative co-adaptation — see factory.build_pipeline_inputs).
"""

from __future__ import annotations

import logging
from pathlib import Path

from .base import BuildContext, FormatAdapter, SourceBundle, StagingResult
from .factory import adapter
from .tools import (
    fingerprint_arch_from_keys,
    read_safetensors_keys,
)
from .tools.hf_layout import (
    HFLayout,
    ARCH_TO_TRANSFORMER_CLASS,
    copy_tokenizer_bundle,
    bundled_te_config,
    bundled_vae_config,
    assert_vae_matches_arch,
)

logger = logging.getLogger(__name__)


# Ordered: first hit wins. We strip whichever prefix is present.
CANDIDATE_PREFIXES = [
    "model.diffusion_model.",   # Flux/SDXL/Qwen UNETLoader output
    "diffusion_model.",          # Some Qwen variants
]


def _detect_transformer_prefix(file_path: str) -> str:
    """Return the matching prefix, or "" if none."""
    sample = []
    for k in read_safetensors_keys(file_path):
        sample.append(k)
        if len(sample) >= 50:
            break
    for px in CANDIDATE_PREFIXES:
        if any(k.startswith(px) for k in sample):
            return px
    return ""


@adapter(priority=50)
class ComfyUIDiffusionModelAdapter(FormatAdapter):
    """Single-file transformer: ComfyUI UNETLoader prefix OR bare BFL/diffusers.

    Two acceptance paths (mutually exclusive, both land in this adapter):

    Path A — ComfyUI-prefixed: file has ``model.diffusion_model.`` or
      ``diffusion_model.`` prefix → ``_detect_transformer_prefix`` returns
      non-empty. Classic UNETLoader / Load Diffusion Model output.

    Path B — Bare arch: file has NO ComfyUI prefix but
      ``fingerprint_arch_from_keys()`` identifies a known architecture
      (Flux2Klein, QwenImage, ZImage, …) from the bare key patterns.
      Handles raw BFL-layout single-file models at any precision (BF16,
      FP8, FP4, INT4) that the user loaded via "QuantFunc Pick Diffusion
      Model (zero-load)".

    In both paths the file contains ONLY transformer weights; TE and VAE
    come from separate Load nodes (co-adapted by this adapter).
    """

    @classmethod
    def detect(cls, sources: SourceBundle) -> bool:
        if sources.checkpoint is not None:
            return False                 # bundled-checkpoint has its own adapter
        if sources.transformer is None:
            return False
        try:
            path = sources.transformer.path
            # Path A: standard ComfyUI prefix
            if _detect_transformer_prefix(path):
                return True
            # Path B: no prefix, but arch is identifiable from key patterns.
            # Use the pre-scanned arch from FileRef when available (avoids an
            # extra header read on the hot path); fall back to a fresh scan.
            arch = sources.transformer.arch or fingerprint_arch_from_keys(path)
            return bool(arch)
        except Exception:
            return False

    def adapt(self, sources: SourceBundle, staging_dir: Path,
              context: BuildContext) -> StagingResult:
        assert sources.transformer is not None
        xfm_path = sources.transformer.path
        prefix = _detect_transformer_prefix(xfm_path)
        # NOTE: for bare BFL/diffusers layout (Path B) prefix is "".
        # Do NOT fall back to "model.diffusion_model." — the engine's
        # Flux2Klein / QwenImage / ZImage load paths already understand the
        # bare key layout; adding a bogus prefix-strip would corrupt loading.
        # For Path A (ComfyUI-prefixed), prefix is the detected string.
        arch = (fingerprint_arch_from_keys(xfm_path)
                or sources.transformer.arch
                or "")
        if not arch:
            raise RuntimeError(
                f"Could not identify transformer architecture for "
                f"{xfm_path}. Inspect the safetensors header — file may "
                f"be from an unsupported family. (Refusing to fall back "
                f"to a hardcoded arch since that produces silent garbage "
                f"output when wrong.)")

        layout = HFLayout(staging_dir)

        # ZImage BFL-style files use `layers.N.attention.qkv` (fused) +
        # `final_layer.*` / `x_embedder.*` (no `all_` prefix). The C++
        # engine wants HF-diffusers naming (split QKV + `all_final_layer.2-1.*`
        # / `all_x_embedder.2-1.*`). Materialise a remapped safetensors at
        # staging time — one full read+write pass; subsequent runs reuse.
        remap_used = False
        if arch == "ZImage":
            try:
                from .zimage_bfl_remap import is_bfl_zimage, stage_bfl_zimage
            except Exception:
                is_bfl_zimage = lambda *_: False  # fall back if import fails
                stage_bfl_zimage = None
            if is_bfl_zimage(xfm_path):
                logger.info("[comfyui_unet] ZImage BFL layout detected; "
                             "remapping qkv/out/q_norm/k_norm and final_layer "
                             "/ x_embedder paths to diffusers naming")
                layout.add_transformer_remapped(
                    xfm_path,
                    remap_fn=stage_bfl_zimage and (lambda s, d:
                        stage_bfl_zimage(str(s), d.parent, force=False)),
                    config={"_class_name": ARCH_TO_TRANSFORMER_CLASS.get(arch, "")})
                # The remapped file is HF-diffusers native (no prefix to strip).
                remap_used = True

        if not remap_used:
            # Transformer (symlink + on-load prefix strip)
            layout.add_transformer(
                xfm_path,
                config={"_class_name": ARCH_TO_TRANSFORMER_CLASS.get(arch, "")})
            # Only write a key_strip when there is actually a prefix to strip.
            # For bare BFL / diffusers layout (prefix == "") we skip this —
            # a key_strip of "" in quantfunc_config.json would cause the engine
            # to match every key with an empty prefix (= every key) and remove
            # nothing, which is a no-op but may trigger unexpected code paths.
            if prefix:
                layout.set_key_strip("transformer", prefix)

        # Text encoder (if provided)
        if sources.text_encoder is not None:
            from .comfyui_clip import _detect_te_prefix
            te_path = sources.text_encoder.path
            te_prefix = _detect_te_prefix(te_path)
            te_class = "Qwen2_5VLForConditionalGeneration" \
                if arch == "QwenImageEdit" else "Qwen3ForCausalLM"
            # Use bundled full TE config when available — it carries
            # hidden_size / num_attention_heads / head_dim / etc. that the
            # C++ engine needs to allocate the right tensor shapes. Minimal
            # `{_class_name: ...}` triggers a fallback `head_dim = hidden /
            # num_heads` (= 80 for ZImage), producing wrong q_proj shape.
            te_cfg = bundled_te_config(arch) or {"_class_name": te_class}
            layout.add_text_encoder(te_path, config=te_cfg)
            if te_prefix:
                layout.set_key_strip("te", te_prefix)

        # VAE (if provided)
        if sources.vae is not None:
            from .comfyui_vae import _detect_vae_prefix
            vae_path = sources.vae.path
            # Fail LOUD if the wired VAE's channel count is wrong for this arch
            # (QwenImageLayered needs a 4-channel RGBA VAE) — a clear, actionable
            # error instead of the engine's cryptic deep conv_out [4]vs[3] crash.
            assert_vae_matches_arch(arch, vae_path)
            vae_prefix = _detect_vae_prefix(vae_path)
            # ZImage / SDXL-style BFL VAE files use `mid.attn_1.{q,k,v}` +
            # `up.N.block.M` etc.; HF AutoencoderKL uses
            # `mid_block.attentions.0.to_{q,k,v}` + `up_blocks.N.resnets.M`.
            # Detect and remap on the fly.
            try:
                from .zimage_bfl_remap import is_bfl_vae, stage_bfl_vae
            except Exception:
                is_bfl_vae = lambda *_: False
                stage_bfl_vae = None
            if stage_bfl_vae is not None and is_bfl_vae(Path(vae_path)):
                logger.info("[comfyui_unet] BFL-style VAE detected; "
                             "remapping mid/up/down/nin_shortcut paths to "
                             "HF AutoencoderKL naming")
                layout.add_vae_remapped(
                    vae_path,
                    remap_fn=lambda s, d: stage_bfl_vae(str(s), d.parent, force=False),
                    config={"_class_name": "AutoencoderKL"})
            else:
                # A standalone "Pick VAE" file carries NO config.json. Hardcoding
                # AutoencoderKL here built the wrong 2D decoder against a 3D
                # AutoencoderKLQwenImage VAE → engine copy_ overflow / "conv_in.bias
                # not found" (#257, customer production crash). Use the bundled
                # per-arch VAE config (with _class_name + latents_mean/std) so a
                # standalone qwen_image_vae paired with a QwenImage(Edit) transformer
                # is staged correctly — same pattern the bundled/SVDQ adapters use.
                # Falls back to AutoencoderKL only for archs with no bundled config.
                layout.add_vae(
                    vae_path,
                    config=bundled_vae_config(arch) or {"_class_name": "AutoencoderKL"})
                if vae_prefix:
                    layout.set_key_strip("vae", vae_prefix)

        # Tokenizer bundle (mandatory — ComfyUI files don't carry one)
        copy_tokenizer_bundle(arch, layout.tokenizer_dir())

        # Scheduler config (optional)
        layout.add_scheduler(sources.scheduler_config, arch=arch)

        # Hints + index
        layout.set_method("online_quant")
        # Separate UNet/CLIP/VAE files carry no per-component precision
        # metadata — propagate the user's choice so TE/VAE actually quantize.
        layout.apply_user_precisions(
            text_precision=context.text_precision,
            vae_precision=context.vae_precision)
        layout.write_quantfunc_config()
        layout.write_model_index(arch)

        path_label = "comfyui-prefix" if prefix else "bare-bfl/diffusers"
        logger.info("[comfyui_unet] arch=%s prefix=%r path=%s staging=%s",
                     arch, prefix, path_label, staging_dir)
        return StagingResult(
            model_dir=str(staging_dir),
            arch=arch,
            method_hint="online_quant",
            cleanup_dir=str(staging_dir),
        )
