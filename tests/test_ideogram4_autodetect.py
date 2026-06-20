"""Tests for Ideogram-4 precision-config auto-detect in the Build Pipeline node.

Covers the three wiring points added for Ideogram-4 [auto-derive] support:
  1. arch fingerprint returns "Ideogram4" (synthetic keys + the REAL on-disk model)
  2. _ARCH_TO_SERIES / autopick filename -> QuantFunc/Ideogram-4-Series + ideogram4_a4w4.json
  3. eligibility rule: activate the auto-config for ANY non-pre-quantized Ideogram
     base (fp16/bf16 AND the official fp8 distribution); skip pre-quantized
     (QuantFunc-stamped / nvfp4) and unknown; OTHER families keep the strict
     full-precision-only gate (an fp8 Klein/Qwen is used as-is).

Run:  python3 tests/test_ideogram4_autodetect.py        (also pytest-compatible)
"""
import os
import sys
import types
import importlib

_PLUGIN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PARENT = os.path.dirname(_PLUGIN)
_PKG = os.path.basename(_PLUGIN)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
# Stub heavy optional deps so a logic-only import doesn't drag ComfyUI/torch in.
for _n in ("comfy", "torch", "folder_paths", "comfy.model_management"):
    sys.modules.setdefault(_n, types.ModuleType(_n))

af = importlib.import_module(f"{_PKG}.format_adapters.tools.arch_fingerprint")
nfa = importlib.import_module(f"{_PKG}.nodes_format_adapters")
mal = importlib.import_module(f"{_PKG}.model_auto_loader")

REAL_MODEL = ("/media/jonathan/Data/model_cache/modelscope/ideogram-ai/"
              "ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors")


# ----------------------------- detection -----------------------------
def test_detect_ideogram4_synthetic():
    keys = ["layers.0.attention.qkv.weight", "layers.0.feed_forward.w1.weight",
            "llm_cond_proj.weight", "input_proj.weight", "adaln_proj.weight",
            "t_embedding.mlp_in.weight", "final_layer.linear.weight"]
    assert af._detect_arch_by_keys(keys) == "Ideogram4"


def test_detect_ideogram4_dual_transformer_prefix():
    keys = ["unconditional_transformer.layers.3.attention.qkv.weight",
            "unconditional_transformer.llm_cond_norm.weight"]
    assert af._detect_arch_by_keys(keys) == "Ideogram4"


def test_detect_ideogram4_projection_trio_without_llm():
    keys = ["layers.0.attention.qkv.weight", "input_proj.weight",
            "adaln_proj.weight", "t_embedding.mlp_in.weight"]
    assert af._detect_arch_by_keys(keys) == "Ideogram4"


def test_class_name_map_ideogram4():
    assert af.CLASS_NAME_TO_ARCH["Ideogram4Pipeline"] == "Ideogram4"
    assert af.CLASS_NAME_TO_ARCH["Ideogram4Transformer2DModel"] == "Ideogram4"


def test_regression_zimage():
    assert af._detect_arch_by_keys(
        ["layers.0.attention.qkv.weight", "cap_embedder.0.weight",
         "context_refiner.0.attention.qkv.weight"]) == "ZImage"


def test_regression_klein_bfl():
    assert af._detect_arch_by_keys(
        ["single_blocks.0.x.weight", "double_blocks.0.y.weight"]) == "Flux2Klein"


def test_regression_qwen_image():
    keys = [f"transformer_blocks.{i}.attn.to_q.weight" for i in range(60)]
    assert af._detect_arch_by_keys(keys) == "QwenImage"


def test_detect_real_model_on_disk():
    if not os.path.isfile(REAL_MODEL):
        print("    (skipped: real model not present)")
        return
    assert af.fingerprint_arch_from_keys(REAL_MODEL) == "Ideogram4"
    # the official distribution is an fp8 base -> raw_fp8_mixed (NOT pre-quantized)
    assert af.fingerprint_kind_from_metadata(REAL_MODEL) == "raw_fp8_mixed"


# ------------------------- autopick eligibility -------------------------
class _Ref:
    def __init__(self, arch, kind, path="/x/model.safetensors"):
        self.arch, self.kind, self.path = arch, kind, path


def _run_autopick(arch, kind, preset="auto", path_in=""):
    captured = {}

    def fake_dl(series, fname, data_source):
        captured["series"] = series
        captured["fname"] = fname
        return "/fake/" + fname

    orig_dl, orig_sm = mal.download_precision_config, nfa._device_sm
    mal.download_precision_config = fake_dl
    nfa._device_sm = lambda idx: 89
    try:
        pm = {"preset": preset, "path": path_in, "target": "transformer"}
        res = nfa._autopick_precision_for_full_model(pm, _Ref(arch, kind), 0)
        return res, captured, pm
    finally:
        mal.download_precision_config, nfa._device_sm = orig_dl, orig_sm


def test_autopick_ideogram_fp8_activates():
    res, cap, _ = _run_autopick("Ideogram4", "raw_fp8_mixed")   # the REAL model's kind
    assert cap.get("series") == "QuantFunc/Ideogram-4-Series", cap
    assert cap.get("fname") == "ideogram4_a4w4.json", cap
    assert res["path"] == "/fake/ideogram4_a4w4.json"


def test_autopick_ideogram_fp8_nonmixed_activates():
    res, cap, _ = _run_autopick("Ideogram4", "raw_fp8")   # pure-fp8 variant
    assert cap.get("fname") == "ideogram4_a4w4.json", cap


def test_autopick_ideogram_fp16_activates():
    res, cap, _ = _run_autopick("Ideogram4", "raw_highprec")
    assert cap.get("fname") == "ideogram4_a4w4.json", cap


def test_autopick_ideogram_prequant_stamped_skips():
    res, cap, pm = _run_autopick("Ideogram4", "prequant_lighting_separate")
    assert res is pm and not cap


def test_autopick_ideogram_nvfp4_skips():
    res, cap, pm = _run_autopick("Ideogram4", "nvfp4_disk")
    assert res is pm and not cap


def test_autopick_ideogram_unknown_kind_skips():
    res, cap, pm = _run_autopick("Ideogram4", "")
    assert res is pm and not cap


def test_autopick_other_family_fp8_used_as_is():
    # regression guard: the Ideogram fp8 exception must NOT leak to other families
    res, cap, pm = _run_autopick("QwenImage", "raw_fp8_mixed")
    assert res is pm and not cap


def test_autopick_other_family_fullprecision_activates():
    res, cap, _ = _run_autopick("QwenImage", "raw_highprec")
    assert cap.get("series") == "QuantFunc/Qwen-Image-Series", cap


def test_autopick_non_auto_preset_untouched():
    # explicit/custom precision_config must never be overridden by autopick
    res, cap, pm = _run_autopick("Ideogram4", "raw_fp8_mixed",
                                 preset="custom", path_in="/some/explicit.json")
    assert res is pm and not cap


if __name__ == "__main__":
    _fns = [v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]
    _passed = 0
    for _fn in _fns:
        try:
            _fn()
            print(f"  PASS  {_fn.__name__}")
            _passed += 1
        except AssertionError as _e:
            print(f"  FAIL  {_fn.__name__}: {_e}")
        except Exception as _e:  # noqa: BLE001
            print(f"  ERROR {_fn.__name__}: {type(_e).__name__}: {_e}")
    print(f"\n{_passed}/{len(_fns)} passed")
    sys.exit(0 if _passed == len(_fns) else 1)
