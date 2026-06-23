"""Unit tests for the clone-proof qf_source_path recovery in
nodes_pipeline_builder._extract_path / _recover_path_from_cached_init.

Root cause being guarded (customer report, Qwen-Image-Edit):
ComfyUI's CLIP.clone() / ModelPatcher.clone() copy a whitelist of attributes
that does NOT include our `qf_source_path` tag, so any node that clones the
CLIP between the loader and BuildPipeline (CLIPSetLastLayer, LoRA loaders,
typical edit graphs) drops the tag -> _extract_path used to raise. But ComfyUI
preserves `cached_patcher_init` across clone (model_patcher.py:440), whose first
arg is the source path(s); the fallback recovers from it.

Runs standalone: `python3 tests/test_extract_path_fallback.py` (exit 0 = pass),
and is pytest-discoverable.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nodes_pipeline_builder as npb  # noqa: E402


def _dummy_loader(*a, **k):  # stand-in for comfy.sd.load_* callables
    return None


class _FakePatcher:
    """Mimics comfy ModelPatcher: holds cached_patcher_init, NO qf_source_path."""
    def __init__(self, cached_patcher_init):
        self.cached_patcher_init = cached_patcher_init


class _FakeClipClone:
    """A CLIP clone: qf_source_path stripped, but .patcher.cached_patcher_init kept."""
    def __init__(self, ckpt_paths):
        self.patcher = _FakePatcher((_dummy_loader, (ckpt_paths, None, 0, {})))


class _FakeModelClone:
    """A diffusion MODEL clone (ModelPatcher): cached_patcher_init on itself, string path."""
    def __init__(self, unet_path):
        self.cached_patcher_init = (_dummy_loader, (unet_path, {}))


class _FakeCheckpointClipClone:
    """A CLIP from a UNIFIED checkpoint (CheckpointLoaderSimple): the patcher's
    cached_patcher_init records the .ckpt as a STRING (load_checkpoint_guess_config_clip_only),
    NOT a list — comfy/sd.py:1716. Recovery must return that .ckpt string."""
    def __init__(self, ckpt_path):
        self.patcher = _FakePatcher((_dummy_loader, (ckpt_path, None, {}, {})))


class _FakeTagged:
    def __init__(self, path):
        self.qf_source_path = path


class _FakeUntaggedNoCache:
    pass


def _mk_file():
    fd, path = tempfile.mkstemp(suffix=".safetensors")
    os.close(fd)
    return path


def test_clip_clone_recovers_from_cached_patcher_init():
    f = _mk_file()
    try:
        clip = _FakeClipClone(ckpt_paths=[f])  # list, on .patcher (CLIP shape)
        assert not hasattr(clip, "qf_source_path")
        assert npb._extract_path(clip, "CLIP (CLIPLoader)") == f
    finally:
        os.remove(f)


def test_model_clone_recovers_string_path():
    f = _mk_file()
    try:
        model = _FakeModelClone(unet_path=f)  # string, on object itself (MODEL shape)
        assert npb._extract_path(model, "diffusion model (UNETLoader)") == f
    finally:
        os.remove(f)


def test_checkpoint_clip_recovers_string_path():
    # Unified checkpoint: CLIP's .patcher.cached_patcher_init args[0] is the .ckpt STRING.
    # Recovery returns the .ckpt; BuildPipeline (nodes_format_adapters.py:668) then dedups
    # te_path==xfm_path under is_ckpt and routes it through the checkpoint= field.
    f = _mk_file()
    try:
        clip = _FakeCheckpointClipClone(ckpt_path=f)
        assert npb._extract_path(clip, "CLIP (CheckpointLoaderSimple)") == f
    finally:
        os.remove(f)


def test_tag_still_wins_when_present():
    f = _mk_file()
    try:
        assert npb._extract_path(_FakeTagged(f), "VAE") == f
    finally:
        os.remove(f)


def test_nonexistent_cached_path_is_ignored():
    # cached_patcher_init points at a path that doesn't exist -> must NOT return it.
    bogus = _FakeClipClone(ckpt_paths=["/no/such/file.safetensors"])
    try:
        npb._extract_path(bogus, "CLIP")
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError for a non-existent cached path")


def test_no_tag_no_cache_raises():
    try:
        npb._extract_path(_FakeUntaggedNoCache(), "CLIP")
    except RuntimeError as e:
        assert "cached_patcher_init" in str(e)  # message mentions the recovery attempt
    else:
        raise AssertionError("expected RuntimeError when nothing is recoverable")


def test_recover_helper_returns_none_on_garbage():
    assert npb._recover_path_from_cached_init(object()) is None
    class Bad:
        cached_patcher_init = ("not", "a", "valid", "tuple")  # len != 2
    assert npb._recover_path_from_cached_init(Bad()) is None

    # 2-tuple but args is a non-sequence (dict) — must degrade to None, NOT raise.
    class BadArgs:
        cached_patcher_init = (_dummy_loader, {"embedding_directory": "x"})
    assert npb._recover_path_from_cached_init(BadArgs()) is None

    # Explicit None cached_patcher_init (e.g. a not-yet-initialized ModelPatcher) — None, no raise.
    class NoneCache:
        cached_patcher_init = None
        patcher = None
    assert npb._recover_path_from_cached_init(NoneCache()) is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {e!r}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
