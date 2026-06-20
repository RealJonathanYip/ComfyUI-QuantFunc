"""Tests for config-only series registration in the auto-load catalog.

The Ideogram-4 / Qwen-Image-Layered repos are config-only (precision-config/ only,
no transformer/ or base-model dir). They must:
  - be discovered + offered by the Precision Config Auto Loader dropdown, AND
  - resolve correctly back to their full series id when selected, BUT
  - NOT appear in the model-download `model_series` dropdown (nothing to download).

These are offline tests (no network) — the live ModelScope listing is exercised
separately; here we inject the resource cache to validate the wiring.

Run:  python3 tests/test_precision_config_autoload.py        (also pytest-compatible)
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

mal = importlib.import_module(f"{_PKG}.model_auto_loader")


def test_model_series_list_stays_full_models_only():
    # the model-download dropdown must NOT gain the config-only series
    assert len(mal.MODEL_SERIES_LIST) == 5
    assert "QuantFunc/Ideogram-4-Series" not in mal.MODEL_SERIES_LIST
    assert "QuantFunc/Qwen-Image-Layered-Series" not in mal.MODEL_SERIES_LIST


def test_all_resource_series_includes_config_only():
    assert "QuantFunc/Ideogram-4-Series" in mal._ALL_RESOURCE_SERIES
    assert "QuantFunc/Qwen-Image-Layered-Series" in mal._ALL_RESOURCE_SERIES
    # union == full + config-only, no duplicates
    assert mal._ALL_RESOURCE_SERIES == mal.MODEL_SERIES_LIST + mal._PRECISION_CONFIG_ONLY_SERIES


def test_config_only_series_not_in_prequant_set():
    # config-only series must never be probed for a prequant/ dir
    for s in mal._PRECISION_CONFIG_ONLY_SERIES:
        assert s not in mal._SERIES_WITH_PREQUANT


def test_resolve_ideogram_config_selection():
    s, n = mal.resolve_selection_no_series(
        "Ideogram-4-Series/ideogram4_a4w4.json", "Precision config")
    assert s == "QuantFunc/Ideogram-4-Series" and n == "ideogram4_a4w4.json"


def test_resolve_qwen_layered_config_selection():
    s, n = mal.resolve_selection_no_series(
        "Qwen-Image-Layered-Series/50x-above-fp4-sample.json", "Precision config")
    assert s == "QuantFunc/Qwen-Image-Layered-Series" and n == "50x-above-fp4-sample.json"


def test_dropdown_lists_config_only_after_cache_populated():
    # simulate a successful refresh and confirm get_precision_config_options() lists it
    with mal._cache_lock:
        mal._resource_cache.setdefault(
            "QuantFunc/Qwen-Image-Layered-Series", {})["precision-config"] = [
                "50x-above-fp4-sample.json", "50x-below-int4-sample.json"]
        mal._resource_cache.setdefault(
            "QuantFunc/Ideogram-4-Series", {})["precision-config"] = ["ideogram4_a4w4.json"]
    opts = mal.get_precision_config_options()
    assert "Ideogram-4-Series/ideogram4_a4w4.json" in opts
    assert "Qwen-Image-Layered-Series/50x-above-fp4-sample.json" in opts
    assert "Qwen-Image-Layered-Series/50x-below-int4-sample.json" in opts


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
