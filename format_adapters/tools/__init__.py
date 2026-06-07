"""Shared utilities for format adapters."""
from .safetensors_io import (
    read_safetensors_metadata,
    read_safetensors_keys,
    read_safetensors_header,
    read_first_tensor_dtype,
)
from .arch_fingerprint import (
    fingerprint_arch_from_keys,
    fingerprint_kind_from_metadata,
)
from .hf_layout import (
    HFLayout, copy_tokenizer_bundle, copy_tokenizer,
    bundled_te_config, bundled_vae_config, bundled_transformer_config,
)
from .fs_util import link_or_copy

__all__ = [
    "read_safetensors_metadata",
    "read_safetensors_keys",
    "read_safetensors_header",
    "read_first_tensor_dtype",
    "fingerprint_arch_from_keys",
    "fingerprint_kind_from_metadata",
    "HFLayout",
    "copy_tokenizer_bundle",
    "copy_tokenizer",
    "bundled_te_config",
    "bundled_vae_config",
    "bundled_transformer_config",
    "link_or_copy",
]
