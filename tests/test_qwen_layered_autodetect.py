"""Tests for Qwen-Image-Layered precision-config auto-detect in the Build Pipeline.

Qwen-Image-Layered reuses the base QwenImageTransformer2DModel class + the
`transformer_blocks.X` structure, so it must be distinguished from base Qwen-Image
by (a) the unique top-level key `time_text_embed.addition_t_embedding` (shard-1
only) and (b) a shard-independent sibling-`config.json` probe (`use_additional_t_cond`).
It then maps to QuantFunc/Qwen-Image-Layered-Series and the standard 50x-above
(NVFP4) / 50x-below (INT4) split by SM. It is full precision (BF16 -> raw_highprec).

Run:  python3 tests/test_qwen_layered_autodetect.py        (also pytest-compatible)
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
for _n in ("comfy", "torch", "folder_paths", "comfy.model_management"):
    sys.modules.setdefault(_n, types.ModuleType(_n))

af = importlib.import_module(f"{_PKG}.format_adapters.tools.arch_fingerprint")
nfa = importlib.import_module(f"{_PKG}.nodes_format_adapters")
mal = importlib.import_module(f"{_PKG}.model_auto_loader")

REAL_DIR = "/home/jonathan/model_cache/models/Qwen-Image_Latered/transformer"
SHARD1 = os.path.join(REAL_DIR, "diffusion_pytorch_model-00001-of-00005.safetensors")
SHARD3 = os.path.join(REAL_DIR, "diffusion_pytorch_model-00003-of-00005.safetensors")


# ----------------------------- detection -----------------------------
def test_detect_layered_by_key():
    keys = ["time_text_embed.addition_t_embedding.weight",
            "transformer_blocks.0.attn.to_qkv.weight",
            "transformer_blocks.59.img_mlp.net.0.proj.weight"]
    assert af._detect_arch_by_keys(keys) == "QwenImageLayered"


def test_class_name_map_layered():
    assert af.CLASS_NAME_TO_ARCH["QwenImageLayeredPipeline"] == "QwenImageLayered"
    # the shared transformer class must STILL map to base QwenImage
    assert af.CLASS_NAME_TO_ARCH["QwenImageTransformer2DModel"] == "QwenImage"


def test_regression_base_qwen_without_layered_key():
    # base Qwen-Image: 60 blocks, NO addition_t_embedding -> plain QwenImage
    keys = [f"transformer_blocks.{i}.attn.to_q.weight" for i in range(60)]
    assert af._detect_arch_by_keys(keys) == "QwenImage"


def test_detect_real_model_shard1_via_key():
    if not os.path.isfile(SHARD1):
        print("    (skipped: real model not present)"); return
    # shard-1 carries the distinctive addition_t_embedding key
    assert af.fingerprint_arch_from_keys(SHARD1) == "QwenImageLayered"
    assert af.fingerprint_kind_from_metadata(SHARD1) == "raw_highprec"   # BF16 base


def test_detect_real_model_middle_shard_via_sibling():
    # robustness: a middle shard has NO top-level key, but the sibling config.json
    # (use_additional_t_cond) must still resolve it to QwenImageLayered
    if not os.path.isfile(SHARD3):
        print("    (skipped: real model not present)"); return
    assert af.fingerprint_arch_from_keys(SHARD3) == "QwenImageLayered"


def test_sibling_probe_true_on_real_dir():
    if not os.path.isfile(SHARD3):
        print("    (skipped: real model not present)"); return
    assert af._is_qwen_layered_sibling(SHARD3) is True


def test_sibling_probe_false_on_standalone():
    # a standalone safetensors path (no diffusers config.json next to it) -> False
    assert af._is_qwen_layered_sibling("/tmp/no/such/model.safetensors") is False


# ------------------------- autopick wiring -------------------------
class _Ref:
    def __init__(self, arch, kind, path="/x/model.safetensors"):
        self.arch, self.kind, self.path = arch, kind, path


def _run_autopick(arch, kind, sm, preset="auto", path_in=""):
    captured = {}

    def fake_dl(series, fname, data_source):
        captured["series"] = series
        captured["fname"] = fname
        return "/fake/" + fname

    orig_dl, orig_sm = mal.download_precision_config, nfa._device_sm
    mal.download_precision_config = fake_dl
    nfa._device_sm = lambda idx: sm
    try:
        pm = {"preset": preset, "path": path_in, "target": "transformer"}
        res = nfa._autopick_precision_for_full_model(pm, _Ref(arch, kind), 0)
        return res, captured, pm
    finally:
        mal.download_precision_config, nfa._device_sm = orig_dl, orig_sm


def test_arch_to_series_mapping():
    assert nfa._ARCH_TO_SERIES["QwenImageLayered"] == "QuantFunc/Qwen-Image-Layered-Series"


def test_autopick_layered_sm120_picks_fp4():
    res, cap, _ = _run_autopick("QwenImageLayered", "raw_highprec", sm=120)
    assert cap.get("series") == "QuantFunc/Qwen-Image-Layered-Series", cap
    assert cap.get("fname") == "50x-above-fp4-sample.json", cap


def test_autopick_layered_sm89_picks_int4():
    res, cap, _ = _run_autopick("QwenImageLayered", "raw_highprec", sm=89)
    assert cap.get("fname") == "50x-below-int4-sample.json", cap


def test_autopick_layered_sm75_picks_int4():
    # Turing (pre-fp8) still routes to the INT4 map
    res, cap, _ = _run_autopick("QwenImageLayered", "raw_highprec", sm=75)
    assert cap.get("fname") == "50x-below-int4-sample.json", cap


def test_autopick_layered_quantized_skipped():
    # already-quantized layered (e.g. a stamped export) uses its on-disk precision
    res, cap, pm = _run_autopick("QwenImageLayered", "prequant_lighting_separate", sm=120)
    assert res is pm and not cap


def test_autopick_layered_nvfp4_skipped():
    res, cap, pm = _run_autopick("QwenImageLayered", "nvfp4_disk", sm=120)
    assert res is pm and not cap


def test_autopick_layered_raw_fp8_skipped():
    # unlike Ideogram-4, a non-ideogram fp8 model uses the strict full-precision gate
    res, cap, pm = _run_autopick("QwenImageLayered", "raw_fp8", sm=120)
    assert res is pm and not cap


def test_autopick_base_qwen_unaffected():
    # regression: base QwenImage still maps to its own series + 50x split
    res, cap, _ = _run_autopick("QwenImage", "raw_highprec", sm=120)
    assert cap.get("series") == "QuantFunc/Qwen-Image-Series", cap


if __name__ == "__main__":
    _fns = [v for k, v in sorted(globals().items())
            if k.startswith("test_") and callable(v)]
    _passed = 0
    for _fn in _fns:
        try:
            _fn(); print(f"  PASS  {_fn.__name__}"); _passed += 1
        except AssertionError as _e:
            print(f"  FAIL  {_fn.__name__}: {_e}")
        except Exception as _e:  # noqa: BLE001
            print(f"  ERROR {_fn.__name__}: {type(_e).__name__}: {_e}")
    print(f"\n{_passed}/{len(_fns)} passed")
    sys.exit(0 if _passed == len(_fns) else 1)
