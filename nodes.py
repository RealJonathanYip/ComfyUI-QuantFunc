"""QuantFunc ComfyUI nodes: pipeline config, model loader, LoRA, LoRA config, and inference.

Data flow:
  (PipelineConfig) ──config──→ ModelLoader ──pipeline──→ (LoRA) ──→ (LoRA Config) ──→ Generate → IMAGE

PipelineConfig provides advanced init options (optional — without it, auto_optimize defaults apply).
ModelLoader outputs a pipeline config. LoRA nodes append lora paths (chainable).
LoRA Config sets merge strategy. Generate materializes the pipeline (cached) and runs inference.

The quantfunc engine runs in a separate worker process to isolate its CUDA runtime
from ComfyUI's PyTorch (avoids DLL version conflicts on Windows).
"""

import atexit
import ctypes
import datetime as _datetime
import hashlib
import json
import logging
import numpy as np
import os
import platform
import queue as _queue
import re as _re
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time

# Worker stderr noise filter: every line lands in the per-worker temp log
# (see _stderr_reader); only lines that match these patterns are echoed to
# the ComfyUI console. Keeps the user informed of per-stage timings + errors
# without the engine's verbose info dump (every block-quant warn, every
# auth handshake, every VAE-DIAG trace).
_WORKER_CONSOLE_KEEP = _re.compile(
    r'\[load\] '                          # component load timings
    r'|Progress: \d+/\d+'                 # per-step diffusion progress
    r'|\[total\]'                         # end-to-end timing
    r'|\[\d+/\d+\] [a-z_]+:'              # per-stage component time (e.g. [3/5] vae_encoder: 178 ms)
    r'|Saving image|Done!'                # image save
    r'|Bundle export complete'            # export
    r'|Pipeline type:'                    # pipeline kind
    r'|Strategy:'                         # auto-optimize summary
    r'|Error|Failed|fatal|exception|terminate|segfault'  # error keywords
    r'|out of memory|OOM'                 # OOM
    r'|Tensor .* not found'               # missing tensor
    r'|abort'                             # abort
)


class _WorkerLogWriter:
    """Singleton background writer for worker stderr logs.

    Why a global thread?  The stderr reader per-WorkerManager is on the
    critical path of every engine log line (info, warn, debug). Doing the
    file write + flush() inline meant a syscall per line, blocking the
    reader thread; if the kernel pipe buffer fills (~64 KB on Linux) the
    engine's own stderr write() blocks, stalling generation.

    With a dedicated writer:
      - readers only `put_nowait` into a bounded queue (microseconds)
      - one OS thread owns all file handles; ~64 KB block-buffered writes
        + flush every 2 s of idle/work and on error keywords
      - multi-pipeline isolation: each WorkerManager registers its own
        log path; the writer keys file handles by path and never mixes
        streams across workers
      - bounded queue (8192 entries): if the writer is briefly behind,
        readers drop the oldest entries rather than blocking the worker
        (we prefer "lost log lines" over "stalled generation")
    """
    _instance = None
    _init_lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is not None:
            return cls._instance
        with cls._init_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._q: "_queue.Queue[tuple[str, str|None, bool]]" = _queue.Queue(maxsize=8192)
        self._fhs: "dict[str, object]" = {}
        self._stop = threading.Event()
        self._t = threading.Thread(
            target=self._loop, daemon=True, name="QuantFuncLogWriter")
        self._t.start()
        atexit.register(self._shutdown)

    def open(self, path: str) -> None:
        """Pre-open log file. Idempotent."""
        if path in self._fhs:
            return
        try:
            self._fhs[path] = open(
                path, "w", buffering=64 * 1024,
                encoding="utf-8", errors="replace")
        except Exception as e:
            logging.warning("[QuantFunc-worker] log open failed for %s: %s", path, e)

    def write(self, path: str, text: str, *, flush: bool = False) -> None:
        """Enqueue a write. Non-blocking — drops oldest if queue full."""
        try:
            self._q.put_nowait((path, text, flush))
        except _queue.Full:
            # Backpressure: drop one to free a slot, then enqueue. Avoids
            # blocking the stderr reader on transient writer stalls.
            try:
                self._q.get_nowait()
            except _queue.Empty:
                pass
            try:
                self._q.put_nowait((path, text, flush))
            except _queue.Full:
                pass

    def close(self, path: str) -> None:
        """Schedule final flush + close of one path's handle."""
        try:
            # text=None signals "close this fd"
            self._q.put_nowait((path, None, True))
        except _queue.Full:
            pass

    def _loop(self) -> None:
        last_flush = time.monotonic()
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=1.0)
            except _queue.Empty:
                # Idle: periodic flush so `tail -f` sees recent progress
                now = time.monotonic()
                if now - last_flush > 2.0:
                    for fh in list(self._fhs.values()):
                        try: fh.flush()
                        except Exception: pass
                    last_flush = now
                continue
            path, text, do_flush = item
            fh = self._fhs.get(path)
            if fh is None:
                # Lazy open if writer race ahead of open()
                try:
                    fh = open(path, "a", buffering=64 * 1024,
                              encoding="utf-8", errors="replace")
                    self._fhs[path] = fh
                except Exception:
                    continue
            if text is None:
                # Close request
                try:
                    fh.flush(); fh.close()
                except Exception: pass
                self._fhs.pop(path, None)
                continue
            try:
                fh.write(text)
                fh.write("\n")
                if do_flush:
                    fh.flush()
                    last_flush = time.monotonic()
            except Exception:
                pass
            # Opportunistic periodic flush (cheap monotonic compare per item)
            now = time.monotonic()
            if now - last_flush > 2.0:
                for f in list(self._fhs.values()):
                    try: f.flush()
                    except Exception: pass
                last_flush = now

    def _shutdown(self) -> None:
        self._stop.set()
        try:
            self._t.join(timeout=2.0)
        except Exception:
            pass
        # Drain any remaining items + close all handles
        while True:
            try:
                path, text, _ = self._q.get_nowait()
            except _queue.Empty:
                break
            fh = self._fhs.get(path)
            if fh is None or text is None:
                continue
            try: fh.write(text + "\n")
            except Exception: pass
        for fh in self._fhs.values():
            try: fh.flush(); fh.close()
            except Exception: pass
        self._fhs.clear()

# ============================================================================
# Library path resolution
# ============================================================================

_IS_WINDOWS = platform.system() == "Windows"
_BIN_SUBDIR = "windows" if _IS_WINDOWS else "linux"

def _resolve_lib_path():
    """Find the quantfunc shared library.
    Uses lib_setup to detect CUDA version and select the correct DLL.
    """
    # Environment override takes priority
    env_path = os.environ.get("QUANTFUNC_LIB", "")
    if env_path and os.path.exists(env_path):
        return os.path.abspath(env_path)

    try:
        from .lib_setup import resolve_library
        return resolve_library()
    except RuntimeError:
        # Intentional fatal config errors (e.g. SM120 GPU without CUDA 13)
        # must surface, not silently fall back to a mismatched default name.
        raise
    except Exception as e:
        logging.getLogger("QuantFunc").warning("lib_setup failed: %s, using default", e)

    # Fallback: default name
    pkg_dir = os.path.dirname(__file__)
    lib_name = "quantfunc.dll" if _IS_WINDOWS else "libquantfunc.so"
    return os.path.join(pkg_dir, "bin", _BIN_SUBDIR, lib_name)

_LIB_PATH = _resolve_lib_path()
_WORKER_PY = os.path.join(os.path.dirname(__file__), "worker.py")


def _get_available_devices():
    """Detect available CUDA GPU devices. Returns list of string device IDs."""
    devices = []
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                devices.append("{}: {}".format(i, name))
    except Exception:
        pass
    if not devices:
        # Fallback: try nvidia-smi
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
                timeout=5, stderr=subprocess.DEVNULL,
            ).decode().strip()
            for line in out.split("\n"):
                line = line.strip()
                if line:
                    devices.append(line.replace(", ", ": "))
        except Exception:
            pass
    return devices if devices else ["0: GPU"]


_AVAILABLE_DEVICES = _get_available_devices()


def _detect_model_backend(transformer_path: str, model_dir: str) -> str:
    """Auto-detect svdq vs lighting from transformer safetensors (metadata first,
    then tensor-key fingerprint). Returns "svdq" or "lighting"; defaults to
    "lighting" when nothing matches the SVDQ signature (Lighting handles both
    FP16-base runtime-quant and `lighting_precomputed` reload).

    SVDQ signature (Nunchaku export):
      - metadata `quantization_config.method == "svdquant"`
      - or tensor keys like `transformer_blocks.0.attn.to_qkv.qweight`
        (BARE `qweight`, no underscore prefix) co-existing with
        `*.lora_down` / `*.smooth_orig` sidecars

    Lighting signature:
      - metadata method `lighting_precomputed` / `lighting` / `flux2klein_runtime`
      - or tensor keys with `_qweight_w4a4` / `_qweight` (underscore prefix
        → QuantFunc Lighting export's persistent buffers)
      - or only FP16 `*.weight` tensors (FP16 base, runtime-quantize)
    """
    # Resolve probe target: explicit transformer arg → model_dir/transformer/
    candidates = []
    if transformer_path:
        if os.path.isfile(transformer_path):
            candidates.append(transformer_path)
        elif os.path.isdir(transformer_path):
            try:
                candidates += sorted(
                    os.path.join(transformer_path, f)
                    for f in os.listdir(transformer_path)
                    if f.endswith(".safetensors")
                )[:1]
            except OSError:
                pass
    if not candidates and model_dir:
        xfm_dir = os.path.join(model_dir, "transformer")
        if os.path.isdir(xfm_dir):
            try:
                candidates += sorted(
                    os.path.join(xfm_dir, f) for f in os.listdir(xfm_dir)
                    if f.endswith(".safetensors")
                )[:1]
            except OSError:
                pass
    if not candidates:
        return "lighting"  # safe fallback — engine errors clearly if file missing

    probe = candidates[0]
    try:
        from .format_adapters.tools.safetensors_io import read_safetensors_header
    except Exception as e:
        logging.debug("[QuantFunc] backend detect: cannot import helpers: %s", e)
        return "lighting"

    # ONE header read — JSON-only, no tensor data. Typical < 1 MB even for
    # 17 GB transformers; ~5-20 ms on warm cache.
    try:
        header = read_safetensors_header(probe)
    except Exception as e:
        logging.debug("[QuantFunc] backend detect: header read failed: %s", e)
        return "lighting"

    # 1. Metadata `method` — ground truth when present.
    meta = header.get("__metadata__", {}) or {}
    qc_str = meta.get("quantization_config", "")
    if qc_str:
        try:
            method = json.loads(qc_str).get("method", "")
            if method == "svdquant":
                return "svdq"
            if method in ("lighting_precomputed", "lighting", "flux2klein_runtime"):
                return "lighting"
        except json.JSONDecodeError:
            pass
    if meta.get("model_class", "").find("Nunchaku") >= 0:
        return "svdq"

    # 2. Tensor-key fingerprint — first ~200 keys from the same header dict.
    seen_underscore_qweight = False
    seen_bare_qweight = False
    seen_lora_sidecar = False
    scanned = 0
    for k in header:
        if k == "__metadata__":
            continue
        scanned += 1
        if scanned > 200:
            break
        if "._qweight" in k:
            seen_underscore_qweight = True
        elif k.endswith(".qweight"):
            seen_bare_qweight = True
        if k.endswith(".lora_down") or ".lora_down." in k or k.endswith(".smooth_orig"):
            seen_lora_sidecar = True
    if seen_underscore_qweight:
        return "lighting"
    if seen_bare_qweight and seen_lora_sidecar:
        return "svdq"

    return "lighting"


def _load_lib_config():
    """Load config.json from the same directory as the quantfunc library binary.
    Returns dict with server_url and api_key (empty strings if not found).
    """
    config_path = os.path.join(os.path.dirname(_LIB_PATH), "config.json")
    try:
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                cfg = json.load(f)
            return {
                "server_url": cfg.get("server_url", ""),
                "api_key": cfg.get("api_key", ""),
            }
    except Exception as e:
        logging.debug("[QuantFunc] Failed to load %s: %s", config_path, e)
    return {"server_url": "", "api_key": ""}


def _make_cache_key(cfg):
    """Build a cache key from pipeline config.
    Excludes api_key and server_url — changing auth credentials should not
    force pipeline recreation (use set_api_key for hot-swap instead).
    """
    opts = dict(cfg.get("options", {}))
    opts.pop("api_key", None)
    opts.pop("server_url", None)
    parts = json.dumps({
        "model_dir": cfg.get("model_dir", ""),
        "transformer": cfg.get("transformer", ""),
        "backend": cfg.get("backend", "svdq"),
        "precision": cfg.get("precision", "int4"),
        "scheduler": cfg.get("scheduler", ""),
        "device": cfg.get("device", 0),
        "options": opts,
    }, sort_keys=True)
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


# ============================================================================
# Worker Manager — manages worker subprocess
# ============================================================================

_dep_download_lock = threading.Lock()
_dep_downloading = False  # True while download is in progress
_dep_downloaded = False   # True after dep download attempted (success or fail)


if not _IS_WINDOWS:
    try:
        _libc = ctypes.CDLL("libc.so.6", use_errno=True)
        _PR_SET_PDEATHSIG = 1

        def _linux_die_with_parent():
            """preexec_fn: kernel will send SIGTERM to this worker when its
            parent process (ComfyUI) dies, even from SIGKILL or crash."""
            _libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except Exception:
        _linux_die_with_parent = None
else:
    _linux_die_with_parent = None


def _kill_stale_workers(dll_path):
    """Find and kill leftover worker.py subprocesses from previous ComfyUI
    runs that didn't exit cleanly. Identified by matching the --dll-path arg
    in their cmdline. Must only match OUR workers, not other tools.
    """
    my_pid = os.getpid()
    worker_script = os.path.abspath(_WORKER_PY)
    dll_abs = os.path.abspath(dll_path)
    candidates = []

    if _IS_WINDOWS:
        # psutil would be cleanest but we can't assume it. Use WMIC.
        # /format:list output is blocks of Key=Value lines separated by blank
        # lines. Key order inside a block is not guaranteed.
        try:
            out = subprocess.check_output(
                ["wmic", "process", "where",
                 "name='python.exe' or name='pythonw.exe'",
                 "get", "ProcessId,CommandLine", "/format:list"],
                stderr=subprocess.DEVNULL, timeout=5).decode("utf-8", "replace")
            block = {}
            def flush_block():
                cmd = block.get("CommandLine", "")
                pid_str = block.get("ProcessId", "").strip()
                if pid_str.isdigit() and cmd:
                    pid = int(pid_str)
                    if (pid != my_pid and worker_script in cmd
                            and dll_abs in cmd):
                        candidates.append(pid)
            for line in out.splitlines():
                line = line.strip()
                if not line:
                    if block:
                        flush_block()
                        block = {}
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    block[k.strip()] = v
            if block:
                flush_block()
        except Exception as e:
            logging.debug("[QuantFunc] stale-worker scan (wmic) failed: %s", e)
    else:
        # Linux: walk /proc
        try:
            for pid_entry in os.listdir("/proc"):
                if not pid_entry.isdigit():
                    continue
                pid = int(pid_entry)
                if pid == my_pid:
                    continue
                try:
                    with open("/proc/%d/cmdline" % pid, "rb") as f:
                        cmdline = f.read().decode("utf-8", "replace")
                except (OSError, IOError):
                    continue
                # Args are NUL-separated; also match if joined with spaces
                if worker_script in cmdline and dll_abs in cmdline:
                    candidates.append(pid)
        except Exception as e:
            logging.debug("[QuantFunc] stale-worker scan (/proc) failed: %s", e)

    if not candidates:
        return
    logging.warning("[QuantFunc] Killing %d stale worker process(es) from previous run: %s",
                    len(candidates), candidates)
    for pid in candidates:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    # Give them 3 seconds to exit gracefully, then SIGKILL leftovers
    time.sleep(3)
    for pid in candidates:
        try:
            os.kill(pid, 0)  # still alive?
        except OSError:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            logging.warning("[QuantFunc] SIGKILL'd stale worker pid %d", pid)
        except OSError:
            pass


class WorkerManager:
    """Manages a QuantFunc worker subprocess with isolated CUDA libraries."""

    def __init__(self):
        self._process = None
        self._stdin = None
        self._stdout = None
        self._stderr_thread = None
        self._loaded_keys = set()          # keys currently alive in worker
        self._api_keys = {}                # cache_key -> api_key last set
        self._node_refs = {}               # generate_node_id -> cache_key (owner tracking)
        # cache_key -> unload_mode policy. Read by ComfyUI's free_memory hook
        # to decide how aggressively to respond to VRAM pressure from other
        # plugins. keep_on_gpu: never release (refuse ComfyUI's free_memory);
        # follow_comfy: offload GPU on normal requests, destroy on blanket
        # requests (Free Model and Node Cache); clean_up_every_time is
        # handled by the generate node itself (destroys right after gen).
        self._unload_modes = {}            # cache_key -> "keep_on_gpu" | "follow_comfy" | "clean_up_every_time"
        self._req_counter = 0
        self._lock = threading.Lock()

    # ── Worker lifecycle ──

    def _build_worker_env(self, dll_dir):
        """Build environment dict for the worker subprocess."""
        env = os.environ.copy()
        if _IS_WINDOWS:
            extra = [dll_dir]
            cuda_path = env.get("CUDA_PATH", "")
            if cuda_path:
                cuda_bin = os.path.join(cuda_path, "bin")
                if os.path.isdir(cuda_bin):
                    extra.insert(0, cuda_bin)
            env["PATH"] = os.pathsep.join(extra) + os.pathsep + env.get("PATH", "")
        else:
            ld_parts = [dll_dir]
            cuda_path = env.get("CUDA_PATH", "/usr/local/cuda")
            lib64 = os.path.join(cuda_path, "lib64")
            if os.path.isdir(lib64):
                ld_parts.append(lib64)
            existing = env.get("LD_LIBRARY_PATH", "")
            if existing:
                ld_parts.append(existing)
            env["LD_LIBRARY_PATH"] = os.pathsep.join(ld_parts)
        return env

    def _start_worker(self, dll_path, env):
        """Start worker subprocess and wait for ready signal.
        Returns (success, error_message).
        """
        python_exe = os.environ.get("QUANTFUNC_PYTHON", "") or sys.executable
        cmd = [python_exe, _WORKER_PY, "--dll-path", dll_path]

        # Kill leftover workers from a previous ComfyUI run that didn't exit
        # cleanly (SIGKILL / reboot / force-quit). They still hold VRAM.
        _kill_stale_workers(dll_path)

        creation_flags = 0
        preexec = None
        if _IS_WINDOWS:
            creation_flags = subprocess.CREATE_NO_WINDOW
        else:
            # Linux: ask the kernel to send SIGTERM to the worker whenever our
            # process dies (PR_SET_PDEATHSIG). Without this, if ComfyUI gets
            # SIGKILL'd or crashes, the worker is reparented to init and keeps
            # holding ~15 GB of VRAM until the user manually kills it.
            preexec = _linux_die_with_parent

        logging.info("[QuantFunc] Starting worker: %s (python=%s)",
                     " ".join(cmd[:4]), python_exe)

        try:
            self._process = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, env=env, creationflags=creation_flags,
                preexec_fn=preexec)
        except Exception as e:
            return False, (f"Failed to start worker process: {e}\n"
                           f"Python: {python_exe}\n"
                           f"Set QUANTFUNC_PYTHON env var to a working Python 3.8+ path.")

        self._stdin = self._process.stdin
        self._stdout = self._process.stdout

        # Allocate this worker's log path NOW (before the stderr reader
        # spawns) so we can print it to the ComfyUI console at the same
        # init point as "Starting worker:" / "Worker ready". Pipeline
        # isolation: per-pid path → each WorkerManager owns its own file.
        log_ts = _datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self._worker_log_path = os.path.join(
            tempfile.gettempdir(),
            f"quantfunc_worker_{self._process.pid}_{log_ts}.log")
        _WorkerLogWriter.get().open(self._worker_log_path)
        logging.info("[QuantFunc] Worker log: %s", self._worker_log_path)

        self._stderr_thread = threading.Thread(
            target=self._stderr_reader, daemon=True)
        self._stderr_thread.start()

        ready = self._read_response(timeout=60)
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            try:
                self._process.kill()
                _, stderr_out = self._process.communicate(timeout=5)
                stderr_msg = stderr_out.decode(errors="replace")[-500:] if stderr_out else ""
            except Exception:
                stderr_msg = ""
            self._process = None
            return False, (f"Worker failed to start (timeout or crash).\n"
                           f"Python: {python_exe}\n"
                           f"DLL: {dll_path}\n"
                           f"Worker stderr: {stderr_msg}\n"
                           f"Hint: Set QUANTFUNC_PYTHON env var to a Python with ctypes + numpy.")

        logging.info("[QuantFunc] Worker ready (version %s, pid %d)",
                     ready.get("version", "?"), self._process.pid)
        return True, ""

    @staticmethod
    def _try_download_deps(dll_path):
        """Download dependency DLLs if not already attempted. Thread-safe.
        Returns True if deps were newly downloaded.
        Raises RuntimeError if another thread is currently downloading.
        """
        global _dep_downloading, _dep_downloaded
        if _dep_downloaded:
            return False
        acquired = _dep_download_lock.acquire(blocking=False)
        if not acquired:
            # Another thread is downloading right now
            raise RuntimeError(
                "[QuantFunc] 依赖库正在下载中，请稍后再试。\n"
                "Dependency libraries are being downloaded. Please try again shortly.")
        try:
            if _dep_downloaded:
                return False
            _dep_downloading = True
            try:
                from .lib_setup import select_cuda_major, _download_dep_zip
                cuda_major = select_cuda_major()
                bin_dir = os.path.dirname(os.path.abspath(dll_path))
                logging.warning("[QuantFunc] Worker failed to load DLL, "
                                "downloading dependency libraries...")
                result = _download_dep_zip(cuda_major, bin_dir)
                _dep_downloaded = True
                return result
            except Exception as e:
                logging.error("[QuantFunc] Dependency download failed: %s", e)
                _dep_downloaded = True
                return False
            finally:
                _dep_downloading = False
        finally:
            _dep_download_lock.release()

    def _ensure_worker(self):
        """Start worker process if not running.
        On first load failure, downloads deps and retries once.
        """
        if self._process is not None and self._process.poll() is None:
            return

        # If deps are being downloaded by another thread, fail fast
        if _dep_downloading:
            raise RuntimeError(
                "[QuantFunc] 依赖库正在下载中，请稍后再试。\n"
                "Dependency libraries are being downloaded. Please try again shortly.")

        if self._process is not None:
            logging.warning("[QuantFunc] Worker process died, restarting...")
            self._loaded_keys.clear()
            self._api_keys.clear()
            self._unload_modes.clear()
            self._node_refs.clear()

        dll_path = _LIB_PATH
        if not os.path.exists(dll_path):
            raise RuntimeError(
                f"QuantFunc library not found: {dll_path}\n"
                f"The auto-download may still be in progress or may have failed.\n"
                f"Check the ComfyUI console for download status messages.")
        dll_dir = os.path.dirname(os.path.abspath(dll_path))
        env = self._build_worker_env(dll_dir)

        # First attempt
        ok, err = self._start_worker(dll_path, env)
        if ok:
            return

        # First failure may be missing dependency libs — download and retry
        if self._try_download_deps(dll_path):
            logging.info("[QuantFunc] Dependencies installed, retrying worker...")
            env = self._build_worker_env(dll_dir)  # rebuild (deps now in dll_dir)
            ok2, err2 = self._start_worker(dll_path, env)
            if ok2:
                return
            raise RuntimeError(err2)

        raise RuntimeError(err)

    def _stderr_reader(self):
        """Forward worker's stderr to console + per-worker temp log.

        ComfyUI console gets only the lines we'd want a user to see
        (per-stage timings, progress, errors, OOM). The full stream goes
        to /tmp/quantfunc_worker_<pid>_<ts>.log so engine debug output is
        still available when diagnosing.
        """
        self._recent_stderr = []
        # Log path + writer-side file handle are set up by _start_worker
        # so the path can be printed to the ComfyUI console at the same
        # `[QuantFunc] Worker log: ...` moment as `Starting worker:` /
        # `Worker ready`. Defensive default: if the field is missing (e.g.
        # legacy entry into this method), still write somewhere sane.
        log_path = getattr(self, "_worker_log_path", None) or os.path.join(
            tempfile.gettempdir(),
            f"quantfunc_worker_unknown_"
            f"{_datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        writer = _WorkerLogWriter.get()
        if not getattr(self, "_worker_log_path", None):
            writer.open(log_path)
            self._worker_log_path = log_path
            logging.info("[QuantFunc] Worker log: %s", log_path)
        _ERROR_FLUSH_KW = ("error", "fatal", "exception", "oom",
                            "abort", "segfault", "terminate")
        try:
            for line in self._process.stderr:
                text = line.decode("utf-8", errors="replace").rstrip()
                if not text:
                    continue
                # Detect error keywords (cheap substring scan) — used both
                # for crash-diagnostic cache and to request immediate flush
                # on the writer thread (so a subsequent crash leaves a
                # complete log on disk).
                is_error = False
                low = text.lower()
                for k in _ERROR_FLUSH_KW:
                    if k in low:
                        is_error = True
                        break
                if is_error:
                    self._recent_stderr.append(text)
                    if len(self._recent_stderr) > 10:
                        self._recent_stderr.pop(0)
                # All disk I/O is delegated to the global writer thread;
                # this hot path only does substring/regex + a non-blocking
                # queue put. No syscalls per line → stderr pipe stays
                # drained → engine never blocks on stderr write().
                writer.write(log_path, text, flush=is_error)
                # Console echo only for key lines (full stream in log_path)
                if _WORKER_CONSOLE_KEEP.search(text):
                    logging.info("[QuantFunc-worker] %s", text)
        except Exception:
            pass
        finally:
            writer.close(log_path)

    def _kill_worker(self):
        if self._process is not None:
            pid = self._process.pid
            # First try graceful SIGTERM
            try:
                self._process.terminate()
                self._process.wait(timeout=3)
            except Exception:
                pass
            # Then SIGKILL if still alive
            if self._process.poll() is None:
                try:
                    self._process.kill()
                    self._process.wait(timeout=5)
                except Exception:
                    pass
            # Last resort: os.kill (handles CUDA driver stuck in uninterruptible sleep)
            if self._process.poll() is None:
                try:
                    import signal
                    os.kill(pid, signal.SIGKILL)
                    self._process.wait(timeout=3)
                except Exception:
                    logging.error("[QuantFunc] Failed to kill worker pid %d — "
                                  "process may be stuck in CUDA driver (D state). "
                                  "GPU resources may remain occupied until reboot.", pid)
            self._process = None
            self._loaded_keys.clear()
            self._api_keys.clear()
            self._unload_modes.clear()
            self._node_refs.clear()

    # ── IPC ──

    def _next_req_id(self):
        self._req_counter += 1
        return self._req_counter

    def _send_command(self, cmd):
        """Send a JSON command to worker."""
        data = json.dumps(cmd, ensure_ascii=True).encode("utf-8") + b"\n"
        self._stdin.write(data)
        self._stdin.flush()

    _SENTINEL_TIMEOUT = "__timeout__"
    _SENTINEL_WORKER_DIED = "__worker_died__"

    def _read_response(self, timeout=600):
        """Read one JSON line from worker stdout."""
        # Simple blocking read with timeout via thread
        result = [self._SENTINEL_TIMEOUT]  # default = timeout
        def reader():
            try:
                line = self._stdout.readline()
                if line:
                    result[0] = json.loads(line.decode("utf-8").strip())
                else:
                    # Empty line = worker process exited (stdout closed)
                    result[0] = self._SENTINEL_WORKER_DIED
            except Exception as e:
                result[0] = {"type": "error", "error_message": str(e)}
        t = threading.Thread(target=reader, daemon=True)
        t.start()
        t.join(timeout=timeout)
        return result[0]

    def _read_binary(self, n_bytes):
        """Read exactly n_bytes from worker stdout."""
        data = b""
        while len(data) < n_bytes:
            chunk = self._stdout.read(n_bytes - len(data))
            if not chunk:
                raise RuntimeError("Worker stdout closed during binary read")
            data += chunk
        return data

    def _call(self, cmd, progress_cb=None, timeout=1800):
        """Send command and collect response, relaying progress."""
        self._send_command(cmd)

        while True:
            resp = self._read_response(timeout=timeout)
            if resp is self._SENTINEL_TIMEOUT:
                self._kill_worker()
                raise RuntimeError(f"Worker timeout after {timeout}s — the operation took too long. "
                                   "Try a smaller resolution or fewer steps.")
            if resp is self._SENTINEL_WORKER_DIED:
                # Worker crashed — read cached stderr error lines
                import time
                time.sleep(0.5)  # let stderr reader finish flushing
                stderr_lines = getattr(self, "_recent_stderr", [])
                error_detail = stderr_lines[-1] if stderr_lines else ""
                # Also check exit code
                exit_code = None
                try:
                    exit_code = self._process.poll()
                except Exception:
                    pass
                self._kill_worker()
                if not error_detail:
                    error_detail = f"Worker process crashed (exit code: {exit_code}, no error details captured)"
                else:
                    error_detail = f"Worker crashed (exit code: {exit_code}): {error_detail}"
                logging.error(f"[QuantFunc] {error_detail}")
                raise RuntimeError(error_detail)

            msg_type = resp.get("type", "")

            if msg_type == "progress":
                if progress_cb:
                    progress_cb(resp.get("step", 0), resp.get("total", 0))
                continue

            if msg_type == "result":
                status = resp.get("status", "")
                if status == "cancelled":
                    raise InterruptedError("Generation cancelled")
                if status == "error":
                    error_msg = resp.get("error_message", "Unknown worker error")
                    error_code = resp.get("error_code", -1)
                    # Kill worker on CUDA/OOM/internal errors — CUDA state may be
                    # corrupted and the process will hold GPU memory indefinitely.
                    # Auth errors (code 7) are recoverable — don't kill.
                    if error_code not in (7,):  # QUANTFUNC_ERROR_AUTH
                        logging.warning("[QuantFunc] C API error (code %d), killing worker "
                                        "to release GPU resources: %s", error_code, error_msg[:200])
                        self._kill_worker()
                    raise RuntimeError(error_msg)
                return resp

            # Unknown message type, skip
            continue

    # ── Public API ──

    def set_api_key(self, cache_key, api_key):
        """Hot-swap API key on the specified pipeline (no pipeline recreation)."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if cache_key not in self._loaded_keys:
                return
            self._set_api_key_locked(cache_key, api_key)

    def _unload_others_locked(self, keep_key):
        """Offload all pipelines except keep_key to CPU, to free VRAM before
        loading/running keep_key. Must be called with self._lock held.

        Uses sync=True so the worker's quantfunc_unload_sync blocks until
        VRAM is actually released — the caller is about to load or run
        keep_key and can't race with an async offload of the others."""
        others = [k for k in self._loaded_keys if k != keep_key]
        for k in others:
            try:
                self._call({"cmd": "unload", "req_id": self._next_req_id(),
                            "cache_key": k, "sync": True}, timeout=60)
                logging.info("[QuantFunc] Evicted pipeline %s to free VRAM for %s",
                             k[:8], keep_key[:8] if keep_key else "?")
            except Exception as e:
                logging.warning("[QuantFunc] Failed to evict %s: %s", k[:8], e)

    def ensure_pipeline(self, cfg, node_id=None, alive_node_ids=None):
        """Ensure pipeline matching cfg is loaded in worker. Returns its cache key.
        `node_id` is the caller's ComfyUI UNIQUE_ID. If provided, we track which
        node references which cache key and destroy pipelines that no node
        references anymore (e.g. when a loader node's transformer path changes).
        `alive_node_ids` is the set of Generate node ids present in the current
        workflow — any _node_refs entry not in this set is considered stale
        (node was deleted from the workflow) and its pipeline is destroyed if
        no one else references it.
        """
        with self._lock:
            self._ensure_worker()

            key = _make_cache_key(cfg)
            opts = dict(cfg.get("options", {}))
            # Route the transformer's coalesced backup through disk-file-
            # backed mmap whenever the caller declared they might later
            # unload. The backing store is a real file (/var/tmp/
            # quantfunc_backup_XXXXXX, or $QUANTFUNC_BACKUP_DIR / $TMPDIR
            # if set), released on clean process exit. When the Generate
            # node sets activate_unload=True and ComfyUI later requests a
            # free_memory, we trigger releaseRamPages() on that backup so
            # the kernel writes dirty pages to disk and reclaims ~13 GB of
            # physical RAM — visible to other plugins. Next gen page-
            # faults pages back in from disk.
            # When activate_unload=False (default in the Generate node),
            # the backup stays in pinned RAM and no disk file is created;
            # the pipeline refuses ComfyUI's free_memory requests.
            opts.setdefault("activate_unload", False)
            new_api_key = opts.get("api_key", "")

            # Evict stale refs from deleted/disconnected Generate nodes
            if alive_node_ids is not None:
                stale = [nid for nid in self._node_refs if nid not in alive_node_ids]
                for nid in stale:
                    dropped_key = self._node_refs.pop(nid)
                    logging.info("[QuantFunc] Dropping stale node ref %s -> %s",
                                 nid, dropped_key[:8] if dropped_key else None)
                    if (dropped_key and
                            dropped_key not in self._node_refs.values() and
                            dropped_key != key and
                            dropped_key in self._loaded_keys):
                        logging.info("[QuantFunc] Destroying orphan pipeline %s (no nodes reference it)",
                                     dropped_key[:8])
                        try:
                            self._call({"cmd": "destroy",
                                        "req_id": self._next_req_id(),
                                        "cache_key": dropped_key}, timeout=30)
                        except Exception as e:
                            logging.warning("[QuantFunc] destroy(%s) failed: %s", dropped_key[:8], e)
                        self._loaded_keys.discard(dropped_key)
                        self._api_keys.pop(dropped_key, None)
                        self._unload_modes.pop(dropped_key, None)

            # Update node→key ownership and release orphaned pipelines
            if node_id is not None:
                old_key = self._node_refs.get(node_id)
                self._node_refs[node_id] = key
                if old_key and old_key != key:
                    # If no other node still references the old key, destroy it
                    if old_key not in self._node_refs.values() and old_key in self._loaded_keys:
                        logging.info("[QuantFunc] Destroying orphan pipeline %s (node %s changed config)",
                                     old_key[:8], node_id)
                        try:
                            self._call({"cmd": "destroy",
                                        "req_id": self._next_req_id(),
                                        "cache_key": old_key}, timeout=30)
                        except Exception as e:
                            logging.warning("[QuantFunc] destroy(%s) failed: %s", old_key[:8], e)
                        self._loaded_keys.discard(old_key)
                        self._api_keys.pop(old_key, None)
                        self._unload_modes.pop(old_key, None)

            if key in self._loaded_keys:
                # Pipeline already loaded — check if API key changed
                if new_api_key and new_api_key != self._api_keys.get(key, ""):
                    self._set_api_key_locked(key, new_api_key)
                return key

            if self._loaded_keys:
                logging.info("[QuantFunc] New pipeline requested, offloading %d existing to CPU...",
                             len(self._loaded_keys))
                # Free VRAM before creating new pipeline
                self._unload_others_locked(keep_key=key)

            # Build create command
            create_cmd = {
                "cmd": "create",
                "req_id": self._next_req_id(),
                "cache_key": key,
                "model_dir": cfg.get("model_dir", ""),
                "transformer_path": cfg.get("transformer", ""),
                "scheduler_config": cfg.get("scheduler", "") or None,
                "model_backend": cfg.get("backend", "svdq"),
                "device_idx": cfg.get("device", 0),
                "config_json": json.dumps(opts),
            }

            logging.info(f"[QuantFunc] create_cmd: model_dir={create_cmd['model_dir']!r}, "
                         f"transformer={create_cmd['transformer_path']!r}, "
                         f"scheduler={create_cmd['scheduler_config']!r}, "
                         f"backend={create_cmd['model_backend']!r}, "
                         f"device={create_cmd['device_idx']!r}, "
                         f"config_json={create_cmd['config_json']!r}")

            self._call(create_cmd, timeout=1800)
            self._loaded_keys.add(key)
            self._api_keys[key] = new_api_key
            logging.info("[QuantFunc] Pipeline ready (%d loaded).", len(self._loaded_keys))
            return key

    def _set_api_key_locked(self, cache_key, api_key):
        """Internal: set API key while already holding self._lock."""
        cmd = {
            "cmd": "set_api_key",
            "req_id": self._next_req_id(),
            "cache_key": cache_key,
            "api_key": api_key,
        }
        self._call(cmd, timeout=30)
        self._api_keys[cache_key] = api_key
        logging.info("[QuantFunc] API key updated (hot-swap).")

    def text_to_image(self, cache_key, prompt, height, width, steps, seed,
                      guidance_scale, options_json=None, pbar=None):
        """Generate text-to-image on the specified pipeline. Returns [H, W, 3] float32 numpy array."""
        with self._lock:
            self._ensure_worker()

            # Before running, make sure only the target pipeline is GPU-resident
            self._unload_others_locked(keep_key=cache_key)

            def on_progress(step, total):
                if pbar is not None:
                    pbar.update(1)

            cmd = {
                "cmd": "text_to_image",
                "req_id": self._next_req_id(),
                "cache_key": cache_key,
                "prompt": prompt,
                "height": height,
                "width": width,
                "num_steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "options_json": options_json,
            }

            resp = self._call(cmd, progress_cb=on_progress, timeout=600)
            return self._read_image(resp)

    def image_to_image(self, cache_key, prompt, ref_paths, height, width, steps, seed,
                       true_cfg_scale=1.0, negative_prompt="",
                       options_json=None, pbar=None,
                       mask_path=None, mask_strength=1.0, mask_grow=6,
                       mask_blur=0.0, mask_no_snap=False):
        """Generate image-to-image on the specified pipeline. Returns [H, W, 3] float32 numpy array.
        Optional inpaint: pass `mask_path` to a pixel-space mask PNG (white=inpaint, black=preserve).
        Mirrors ComfyUI SetLatentNoiseMask + GrowMask + MaskBlur + VAEEncodeForInpaint.grow_mask_by.
        """
        with self._lock:
            self._ensure_worker()

            # Before running, make sure only the target pipeline is GPU-resident
            self._unload_others_locked(keep_key=cache_key)

            def on_progress(step, total):
                if pbar is not None:
                    pbar.update(1)

            cmd = {
                "cmd": "image_to_image",
                "req_id": self._next_req_id(),
                "cache_key": cache_key,
                "prompt": prompt,
                "ref_image_paths": ref_paths,
                "height": height,
                "width": width,
                "num_steps": steps,
                "true_cfg_scale": true_cfg_scale,
                "negative_prompt": negative_prompt,
                "seed": seed,
                "options_json": options_json,
            }
            if mask_path:
                cmd["mask_path"] = mask_path
                cmd["mask_strength"] = float(mask_strength)
                cmd["mask_grow"] = int(mask_grow)
                cmd["mask_blur"] = float(mask_blur)
                cmd["mask_no_snap"] = bool(mask_no_snap)

            resp = self._call(cmd, progress_cb=on_progress, timeout=600)
            return self._read_image(resp)

    def export_model(self, cfg, export_path):
        """Export model via worker."""
        with self._lock:
            self._ensure_worker()

            # Destroy all loaded pipelines first to free VRAM
            if self._loaded_keys:
                self._call({"cmd": "destroy", "req_id": self._next_req_id()})
                self._loaded_keys.clear()
                self._api_keys.clear()
                self._unload_modes.clear()

            opts = dict(cfg.get("options", {}))
            sched = cfg.get("scheduler", "")
            if sched:
                opts["scheduler_config"] = sched

            cmd = {
                "cmd": "export",
                "req_id": self._next_req_id(),
                "model_dir": cfg.get("model_dir", ""),
                "export_path": export_path,
                "transformer_path": cfg.get("transformer", ""),
                "model_backend": cfg.get("backend", "svdq"),
                "device_idx": cfg.get("device", 0),
                "config_json": json.dumps(opts),
            }

            self._call(cmd, timeout=1800)

    def cancel(self):
        """Send cancel signal to worker."""
        if self._process and self._process.poll() is None:
            try:
                cmd = json.dumps({"cmd": "cancel", "req_id": 0}).encode("utf-8") + b"\n"
                self._stdin.write(cmd)
                self._stdin.flush()
            except Exception:
                pass

    def unload_pipeline(self, cache_key=None, sync=False):
        """Offload models from GPU to CPU, freeing VRAM. Pipelines stay alive for fast reload.
        If cache_key given, unload that one; otherwise unload all.

        sync=True → blocks until VRAM is actually freed (skips the 3-second
        grace period in the worker). Use this when another component is
        about to allocate VRAM (cross-pipeline eviction, ComfyUI's
        free_memory hook for third-party models). Default sync=False is
        fire-and-forget so the caller can return immediately."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if not self._loaded_keys:
                return
            cmd = {"cmd": "unload", "req_id": self._next_req_id(), "sync": sync}
            if cache_key is not None:
                if cache_key not in self._loaded_keys:
                    return
                cmd["cache_key"] = cache_key
            try:
                # Sync unload can take up to ~5 s (grace skip + D2H + trim);
                # async returns almost instantly.
                self._call(cmd, timeout=60 if sync else 30)
                logging.info("[QuantFunc] Models offloaded to CPU — VRAM freed (%s%s)",
                             cache_key if cache_key else "all",
                             " sync" if sync else "")
            except Exception as e:
                logging.warning("[QuantFunc] Unload failed: %s", e)

    def destroy_all(self):
        """Destroy all loaded pipelines (keep worker alive)."""
        with self._lock:
            if self._process and self._process.poll() is None and self._loaded_keys:
                try:
                    self._call({"cmd": "destroy", "req_id": self._next_req_id()}, timeout=30)
                except Exception:
                    pass
                self._loaded_keys.clear()
                self._api_keys.clear()
                self._unload_modes.clear()

    def release_backup_pipeline(self, cache_key):
        """Release physical RAM pages of the mmap-backed CPU offload_backup
        while keeping the backing files (and the pipeline handle). Next
        generate() page-faults pages in (~2-5s via OS page cache / disk)
        instead of full re-init (~15-25s). Used by unload_mode='gpu+cpu'.

        Pair with unload_pipeline: first unload GPU → CPU (mmap), then
        release_backup to free the physical RAM while keeping content on
        disk for fast reload.

        No-op if the DLL predates quantfunc_release_backup, or if the
        pipeline wasn't created with activate_unload=True (e.g. export
        mode or the user unticked the widget)."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if cache_key not in self._loaded_keys:
                return
            try:
                self._call({"cmd": "release_backup",
                            "req_id": self._next_req_id(),
                            "cache_key": cache_key}, timeout=30)
                logging.info("[QuantFunc] Pipeline %s backup released — RAM pages returned to OS",
                             cache_key[:8] if cache_key else "None")
            except Exception as e:
                logging.warning("[QuantFunc] release_backup_pipeline(%s) failed: %s",
                                cache_key[:8] if cache_key else "None", e)

    def destroy_pipeline(self, cache_key):
        """Fully destroy a single pipeline (frees GPU + CPU/RAM).
        Next use will recreate from scratch — slower than unload_pipeline but
        reclaims the ~10 GB pinned coalesced backup held in system RAM.
        Used as a last resort / escape hatch."""
        with self._lock:
            if self._process is None or self._process.poll() is not None:
                return
            if cache_key not in self._loaded_keys:
                return
            try:
                self._call({"cmd": "destroy",
                            "req_id": self._next_req_id(),
                            "cache_key": cache_key}, timeout=60)
                self._loaded_keys.discard(cache_key)
                self._api_keys.pop(cache_key, None)
                self._unload_modes.pop(cache_key, None)
                logging.info("[QuantFunc] Pipeline %s destroyed — GPU + RAM released",
                             cache_key[:8] if cache_key else "None")
            except Exception as e:
                logging.warning("[QuantFunc] destroy_pipeline(%s) failed: %s",
                                cache_key[:8] if cache_key else "None", e)

    def shutdown(self):
        """Shutdown worker process."""
        with self._lock:
            if self._process and self._process.poll() is None:
                try:
                    # Try graceful shutdown via IPC first
                    cmd = json.dumps({"cmd": "shutdown", "req_id": 0}).encode("utf-8") + b"\n"
                    self._stdin.write(cmd)
                    self._stdin.flush()
                    self._process.wait(timeout=10)
                except Exception:
                    # IPC failed (broken pipe, etc.) — use signal-based kill
                    self._kill_worker()
            self._process = None
            self._loaded_keys.clear()
            self._api_keys.clear()
            self._unload_modes.clear()

    def _read_image(self, resp):
        """Read image data from worker response.

        Prefers /dev/shm path (zero-copy mmap-style read) over stdout pipe
        binary — the shm path avoids a ~3MB bytes() on worker side and
        multiple pipe syscalls on parent side."""
        n_bytes = resp.get("image_bytes", 0)
        w = resp.get("image_width", 0)
        h = resp.get("image_height", 0)
        if n_bytes == 0 or w == 0 or h == 0:
            raise RuntimeError("No image data in response")
        # Bound + consistency-check the worker-reported sizes BEFORE the read /
        # reshape: a desynced or compromised worker could otherwise force an
        # unbounded blocking read, or a reshape mismatch deep in numpy.
        _MAX_IMG_DIM = 16384
        if w > _MAX_IMG_DIM or h > _MAX_IMG_DIM:
            raise RuntimeError(f"Worker reported implausible image dims {w}x{h}")
        _bpp = 3 if resp.get("image_format", "rgb_float32") == "rgb_uint8" else 12
        if n_bytes != w * h * _bpp:
            raise RuntimeError(
                f"Worker image_bytes={n_bytes} != expected {w * h * _bpp} ({w}x{h})")
        shm_path = resp.get("image_shm_path")
        if shm_path:
            try:
                with open(shm_path, "rb") as f:
                    raw = f.read(n_bytes)
            finally:
                try:
                    os.unlink(shm_path)
                except OSError:
                    pass
        else:
            raw = self._read_binary(n_bytes)
        fmt = resp.get("image_format", "rgb_float32")
        if fmt == "rgb_uint8":
            # uint8 [0,255] → float32 [0,1], 4x less IPC data than float32
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).astype(np.float32) / 255.0
        else:
            # Legacy float32 path
            arr = np.frombuffer(raw, dtype=np.float32).reshape(h, w, 3).copy()
        return arr


_manager = WorkerManager()
atexit.register(_manager.shutdown)


# ============================================================================
# Hook into ComfyUI model management — auto-unload when other nodes need VRAM
# ============================================================================

try:
    import comfy.model_management as _mm

    _original_free_memory = _mm.free_memory

    # Threshold to distinguish real VRAM requests from ComfyUI's blanket
    # unload_all_models() which passes 1e30.  256 TB is far beyond any real
    # GPU memory request but still well below 1e30.
    _UNREALISTIC_VRAM_REQUEST = 256 * 1024 * 1024 * 1024 * 1024  # 256 TB

    def _hooked_free_memory(memory_required, device, keep_loaded=[], **kwargs):
        # Per-pipeline dispatch based on its configured unload_mode:
        #   none                — refuse to release under any request
        #   gpu                 — per-gen already offloaded; on blanket →
        #                         destroy; on normal → unload (no-op if done)
        #   gpu+cpu             — stay loaded after gen; on normal pressure →
        #                         offload GPU + madvise disk backup (return
        #                         ~10 GB RAM to OS); on blanket → destroy
        #   clean_up_every_time — per-gen already released; blanket → destroy
        if _manager._loaded_keys and memory_required > 0:
            blanket = memory_required >= _UNREALISTIC_VRAM_REQUEST

            if not blanket:
                # Check if we actually need to do anything (enough free VRAM?)
                try:
                    import torch
                    free_vram, _ = torch.cuda.mem_get_info(device)
                except Exception:
                    free_vram = 0
                need_release = free_vram < memory_required
            else:
                need_release = True

            if need_release:
                # Snapshot under lock so we iterate a stable set
                with _manager._lock:
                    keys = list(_manager._loaded_keys)
                    modes = dict(_manager._unload_modes)
                for k in keys:
                    mode = modes.get(k, "gpu+cpu")  # safe default
                    if mode == "none":
                        logging.debug(
                            "[QuantFunc] Pipeline %s is unload_mode=none — refusing free_memory request",
                            k[:8])
                        continue
                    # Third-party caller is about to allocate VRAM (or
                    # clicked Free Model) — every branch below uses sync
                    # paths so the caller doesn't race with our offload.
                    if blanket:
                        logging.info(
                            "[QuantFunc] Blanket free_memory — destroying pipeline %s (mode=%s)",
                            k[:8], mode)
                        _manager.destroy_pipeline(k)
                    elif mode == "gpu+cpu":
                        # Full disk-backed release: sync unload (VRAM truly
                        # freed before returning) + async release_backup
                        # (madvise on disk file — doesn't block caller).
                        logging.info(
                            "[QuantFunc] VRAM pressure (gpu+cpu) — sync offload + async release "
                            "for pipeline %s (need %d MB, free %d MB)",
                            k[:8], memory_required // 1024**2, free_vram // 1024**2)
                        _manager.unload_pipeline(k, sync=True)
                        _manager.release_backup_pipeline(k)
                    else:
                        # gpu / clean_up_every_time: sync offload so VRAM is
                        # actually freed for the caller. clean_up_every_time
                        # may already have released, but sync unload is
                        # idempotent (no-op if nothing on GPU).
                        logging.info(
                            "[QuantFunc] VRAM pressure (mode=%s) — sync offload pipeline %s "
                            "(need %d MB, free %d MB)",
                            mode, k[:8], memory_required // 1024**2, free_vram // 1024**2)
                        _manager.unload_pipeline(k, sync=True)
        return _original_free_memory(memory_required, device, keep_loaded=keep_loaded, **kwargs)

    _mm.free_memory = _hooked_free_memory
except Exception:
    pass


# ============================================================================
# Node: QuantFunc Pipeline Config
# ============================================================================

class QuantFuncPipelineConfig:
    """Advanced pipeline configuration for model initialization.

    VRAM/offload strategy is chosen automatically by libquantfunc based on
    your GPU's free VRAM and the loaded model size — no offload knobs are
    exposed here. Old workflows that still set cpu_offload / layer_offload /
    adaptive_offload / offload_compression will continue to load (libquantfunc
    silently ignores those config keys with a one-time deprecation warning),
    but the values have no effect.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tiled_vae": ("BOOLEAN", {"default": False, "tooltip": "Tile-based VAE decoding to reduce VRAM (auto-enabled at high resolution)"}),
                "attention_backend": (["auto", "sage", "flash", "sdpa"], {"default": "auto",
                                      "tooltip": "Attention implementation: auto picks best for your GPU"}),
                "precision": (["bf16", "fp16"], {"default": "bf16", "tooltip": "Compute precision for pipeline"}),
                "text_precision": (["int4", "int8", "fp4", "fp8", "fp16"], {"default": "int4",
                                    "tooltip": "Text encoder quantization precision (fp4 requires SM120+/Blackwell)"}),
                "vision_quant": (["int8", "int4", "fp8", "fp4", "fp16"], {"default": "int8",
                                  "tooltip": "Vision encoder quantization (int8 = INT8 weights + FP16 compute, best quality/size tradeoff)"}),
                "vae_precision": (["auto", "fp16", "fp8", "int8"], {"default": "auto",
                                  "tooltip": "VAE precision (auto picks fp8/int8 on SM120+ via cuDNN, falls back to fp16 elsewhere)"}),
                "act_quant_mode": (["auto", "absmax", "mse"], {"default": "auto",
                                  "tooltip": "Activation quantization scale algorithm (Lighting backend, INT4 only):\n"
                                             "• auto — engine picks: MSE-search when rotation>0 (best quality), else absmax\n"
                                             "• absmax — fast, scale = absmax/7\n"
                                             "• mse — search ~5 candidates for min MSE (+1dB quality, ~8% slower)\n"
                                             "FP4 / INT8 / FP8 ignore this setting (kernel uses its own scaling)."}),
            },
            "optional": {
                "vae_tile_size": ("INT", {"default": 0, "min": 0, "max": 2048, "step": 64,
                                  "tooltip": "VAE tile size in pixels (0 = auto)"}),
                "pinned_memory_limit": ("STRING", {"default": "", "tooltip": "Max pinned CPU memory: '60%', '48G', '48M', or empty for auto"}),
            }
        }

    RETURN_TYPES = ("QUANTFUNC_CONFIG",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = "QuantFunc"

    def build_config(self, tiled_vae, attention_backend, precision, text_precision,
                     vision_quant="int8", vae_precision="auto", act_quant_mode="absmax",
                     vae_tile_size=0, pinned_memory_limit=""):
        config = {
            "tiled_vae": tiled_vae,
            "attention_backend": attention_backend,
            "precision": precision,
            "text_precision": text_precision,
            "vision_quant": vision_quant,
            "vae_precision": vae_precision,
            "act_quant_mode": act_quant_mode,
        }

        if vae_tile_size > 0:
            config["vae_tile_size"] = vae_tile_size

        pinned = pinned_memory_limit if isinstance(pinned_memory_limit, str) and pinned_memory_limit else ""
        if pinned:
            config["pinned_memory_limit"] = pinned

        return (config,)


def _first_safetensors(dir_path: str) -> str:
    """First .safetensors file inside `dir_path`, alphabetical. Returns ""
    when the directory doesn't exist or has none.
    """
    if not dir_path or not os.path.isdir(dir_path):
        return ""
    try:
        for f in sorted(os.listdir(dir_path)):
            if f.endswith(".safetensors"):
                return os.path.join(dir_path, f)
    except OSError:
        pass
    return ""


def _resolve_to_safetensors(p: str) -> str:
    """Accepts a `.safetensors` file path or a directory and returns a
    concrete file path. Empty / missing → "".
    """
    if not p or not isinstance(p, str):
        return ""
    p = p.strip()
    if not p:
        return ""
    if os.path.isfile(p):
        return p
    if os.path.isdir(p):
        return _first_safetensors(p)
    return ""


def _build_model_refs(model_dir: str, transformer_path: str,
                       prequant_weights: str = "") -> tuple:
    """Common helper for ModelLoader / ModelAutoLoader: produce three
    `_QFPathStub` instances typed as comfy MODEL / CLIP / VAE so they plug
    directly into QuantFunc Build Pipeline (same socket types as ComfyUI's
    UNETLoader / CLIPLoader / VAELoader). Mirrors format_adapters' Pick*.
    """
    from .nodes_format_adapters import _QFPathStub

    prequant_weights = prequant_weights.strip() if isinstance(prequant_weights, str) else ""

    # Resolve concrete file paths inside the standard HF model_dir layout.
    # User-provided transformer_path may be either a file or a directory —
    # both are normalised to a concrete .safetensors here.
    xfm_path = _resolve_to_safetensors(transformer_path) or _first_safetensors(
        os.path.join(model_dir, "transformer"))
    te_path  = _first_safetensors(os.path.join(model_dir, "text_encoder"))
    vae_path = _first_safetensors(os.path.join(model_dir, "vae"))

    if not xfm_path:
        raise RuntimeError(
            "QuantFunc Model Loader: no transformer .safetensors found. "
            f"transformer_path={transformer_path!r}, "
            f"model_dir={model_dir!r}/transformer/ has no .safetensors. "
            "Provide an explicit transformer_path or check model_dir layout.")

    backend = _detect_model_backend(xfm_path, model_dir)
    logging.info("[QuantFunc] model_backend → %s (xfm=%s)",
                  backend, os.path.basename(xfm_path))

    model_stub = _QFPathStub(xfm_path, kind="transformer")
    clip_stub  = _QFPathStub(te_path,  kind="te")
    vae_stub   = _QFPathStub(vae_path, kind="vae")
    # Stash QuantFunc-specific hints onto the model stub so BuildPipeline
    # can recover model_dir context, backend, and prequant sidecar.
    model_stub.qf_model_dir = model_dir
    model_stub.qf_backend_hint = backend
    if prequant_weights:
        model_stub.qf_prequant_weights = prequant_weights
    return (model_stub, clip_stub, vae_stub)


class QuantFuncModelLoader:
    """Load a QuantFunc model — outputs three handles (transformer / text_encoder
    / vae), mirroring ComfyUI's Load Checkpoint shape. Wire all three into
    `QuantFunc Build Pipeline`, which carries device + advanced runtime config.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_dir": ("STRING", {"default": "", "tooltip": "Base model directory (contains model_index.json)"}),
                "transformer_path": ("STRING", {"default": "", "tooltip": "Transformer weights path (safetensors file or directory)"}),
            },
            "optional": {
                "prequant_weights": ("STRING", {
                    "default": "",
                    "tooltip": "Pre-quantized modulation weights safetensors path (Lighting backend only)",
                }),
            },
        }

    # Output comfy native MODEL/CLIP/VAE types so this loader interoperates
    # with QuantFunc Build Pipeline (and any other node accepting these
    # sockets — e.g. comfy CheckpointLoaderSimple / UNETLoader / CLIPLoader
    # / VAELoader output the same types). The values are `_QFPathStub`
    # objects carrying qf_source_path; non-QuantFunc consumers will crash
    # on these stubs by design (per format_adapters convention).
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "QuantFunc"

    def load_model(self, model_dir, transformer_path,
                   prequant_weights="", **kwargs):
        return _build_model_refs(model_dir, transformer_path, prequant_weights)


# ============================================================================
# Node: QuantFunc Model Auto Loader
# ============================================================================

def _get_auto_loader_dropdowns():
    """Get dropdown options from resource cache (loaded at import time)."""
    try:
        from .model_auto_loader import get_transformer_options
        return get_transformer_options()
    except Exception:
        return ["None"]


def _get_prequant_dropdowns():
    """Get prequant weight dropdown options from resource cache."""
    try:
        from .model_auto_loader import get_prequant_options
        return get_prequant_options()
    except Exception:
        return ["None"]


def _get_precision_config_dropdowns():
    """Get precision config dropdown options from resource cache."""
    try:
        from .model_auto_loader import get_precision_config_options
        return get_precision_config_options()
    except Exception:
        return ["None"]


class QuantFuncModelAutoLoader:
    """Auto-download and load QuantFunc models.

    Selects the correct GPU variant (50x-below/50x-above) automatically.
    Downloads base model, transformer, prequant weights, and precision config
    from HuggingFace or ModelScope on first use.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import MODEL_SERIES_LIST, _DATA_SOURCES
        transformer_opts = _get_auto_loader_dropdowns()
        return {
            "required": {
                "model_series": (MODEL_SERIES_LIST, {"tooltip": "Model series to download and load"}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
            "optional": {
                "transformer": (transformer_opts, {"default": "None", "tooltip": "Transformer model variant. Format: Series/name. Select None to use base model's default transformer."}),
            },
        }

    # Output comfy native MODEL/CLIP/VAE types — same rationale as
    # QuantFuncModelLoader above: interoperates with BuildPipeline and any
    # node accepting these sockets.
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_model"
    CATEGORY = "QuantFunc"

    def load_model(self, model_series, data_source,
                   transformer="None", **kwargs):
        from .model_auto_loader import (
            detect_gpu_variant, download_base_model,
            download_transformer, resolve_transformer_selection,
        )

        # ── GPU variant & base model ──
        gpu_variant = detect_gpu_variant(model_series)
        model_dir = download_base_model(model_series, gpu_variant, data_source)

        # ── Transformer (download if selected, otherwise use base model's) ──
        transformer_path = ""
        if transformer and transformer != "None":
            t_series, t_name = resolve_transformer_selection(transformer, model_series)
            if t_series and t_name:
                transformer_path = download_transformer(t_series, t_name, data_source)

        return _build_model_refs(model_dir, transformer_path)


# ============================================================================
# Node: QuantFunc Prequant Auto Loader
# ============================================================================

class QuantFuncPrequantAutoLoader:
    """Auto-download prequant weights from HuggingFace or ModelScope.

    Outputs a file path string that can be connected to ModelLoader's
    prequant_weights input. When not connected, ModelLoader falls back
    to its own text input field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import _DATA_SOURCES
        prequant_opts = _get_prequant_dropdowns()
        return {
            "required": {
                "prequant": (prequant_opts, {"default": "None", "tooltip": "Pre-quantized modulation weights. Format: Series/name. Select None to skip."}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prequant_weights",)
    FUNCTION = "load_prequant"
    CATEGORY = "QuantFunc"

    def load_prequant(self, prequant, data_source):
        if not prequant or prequant == "None":
            return ("",)

        from .model_auto_loader import resolve_selection_no_series, download_prequant

        pq_series, pq_name = resolve_selection_no_series(prequant, "Prequant")
        if not pq_series or not pq_name:
            return ("",)

        path = download_prequant(pq_series, pq_name, data_source)
        return (path,)


# ============================================================================
# Node: QuantFunc Precision Config Auto Loader
# ============================================================================

class QuantFuncPrecisionConfigAutoLoader:
    """Auto-download precision config from HuggingFace or ModelScope.

    Outputs a file path string that can be connected to ModelLoader's
    precision_config input. When not connected, ModelLoader falls back
    to its own text input field.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import _DATA_SOURCES
        pc_opts = _get_precision_config_dropdowns()
        return {
            "required": {
                "precision_config": (pc_opts, {"default": "None", "tooltip": "Per-layer precision config JSON. Format: Series/name. Select None to skip."}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("precision_config",)
    FUNCTION = "load_precision_config"
    CATEGORY = "QuantFunc"

    def load_precision_config(self, precision_config, data_source):
        if not precision_config or precision_config == "None":
            return ("",)

        from .model_auto_loader import resolve_selection_no_series, download_precision_config

        pc_series, pc_name = resolve_selection_no_series(precision_config, "Precision config")
        if not pc_series or not pc_name:
            return ("",)

        path = download_precision_config(pc_series, pc_name, data_source)
        return (path,)


# ============================================================================
# Node: QuantFunc Base Series Model Auto Loader
# ============================================================================

class QuantFuncBaseSeriesModelAutoLoader:
    """Auto-download base model from QuantFunc model series.

    Selects the correct GPU variant (50x-below/50x-above) automatically.
    Downloads the base model from QuantFunc series repos on first use.
    Outputs the local model directory path as a string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import MODEL_SERIES_LIST, _DATA_SOURCES
        return {
            "required": {
                "model_series": (MODEL_SERIES_LIST, {"tooltip": "QuantFunc model series"}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_dir",)
    FUNCTION = "load_base_model"
    CATEGORY = "QuantFunc"

    def load_base_model(self, model_series, data_source):
        from .model_auto_loader import detect_gpu_variant, download_base_model
        gpu_variant = detect_gpu_variant(model_series)
        model_dir = download_base_model(model_series, gpu_variant, data_source)
        return (model_dir,)


# ============================================================================
# Node: QuantFunc Base Model Auto Loader
# ============================================================================

def _get_diffusers_model_options():
    """Recursively scan ComfyUI/models/diffusers/ for model directories (containing model_index.json)."""
    try:
        dm_dir = os.path.join(_get_comfyui_dir(), "models", "diffusers")
        if os.path.isdir(dm_dir):
            dirs = []
            for root, subdirs, filenames in os.walk(dm_dir):
                if "model_index.json" in filenames:
                    rel = os.path.relpath(root, dm_dir)
                    dirs.append(rel.replace("\\", "/"))
            if dirs:
                return ["None"] + sorted(dirs)
    except Exception:
        pass
    return ["None"]


class QuantFuncBaseModelAutoLoader:
    """Load base models from ComfyUI/models/diffusers/ directory.

    Scans for subdirectories containing model_index.json.
    Outputs the local model directory path as a string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        model_opts = _get_diffusers_model_options()
        return {
            "required": {
                "model_dir": (model_opts, {"tooltip": "Base model from models/diffusers/"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_dir",)
    FUNCTION = "load_base_model"
    CATEGORY = "QuantFunc"

    def load_base_model(self, model_dir):
        if not model_dir or model_dir == "None":
            return ("",)

        full_path = os.path.join(_get_comfyui_dir(), "models", "diffusers", model_dir)
        if not os.path.isdir(full_path):
            raise RuntimeError("Model directory not found: {}".format(full_path))
        return (full_path,)


# ============================================================================
# Node: QuantFunc Base Model Auto Loader with Download
# ============================================================================

def _get_base_model_repo_dropdowns():
    """Get base model repo dropdown options from cache."""
    try:
        from .model_auto_loader import get_base_model_repo_options
        return get_base_model_repo_options()
    except Exception:
        return ["None"]


class QuantFuncBaseModelAutoLoaderWithDownload:
    """Auto-discover and download base models from ModelScope/HuggingFace.

    Searches upstream repos for available base models and downloads
    to ComfyUI/models/diffusers/ on first use.
    Outputs the local model directory path as a string.
    """

    @classmethod
    def INPUT_TYPES(cls):
        from .model_auto_loader import _DATA_SOURCES
        repo_opts = _get_base_model_repo_dropdowns()
        return {
            "required": {
                "base_model_repo": (repo_opts, {"tooltip": "Upstream base model repository. Auto-discovered from ModelScope."}),
                "data_source": (_DATA_SOURCES, {"default": "modelscope", "tooltip": "Download source: modelscope (China) or huggingface"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_dir",)
    FUNCTION = "load_base_model"
    CATEGORY = "QuantFunc"

    def load_base_model(self, base_model_repo, data_source):
        if not base_model_repo or base_model_repo == "None":
            return ("",)

        from .model_auto_loader import download_base_model_to_diffusers
        path = download_base_model_to_diffusers(base_model_repo, data_source)
        return (path,)


# ============================================================================
# Node: QuantFunc Transformer Auto Loader
# ============================================================================

def _get_local_transformer_file_options():
    """Recursively scan models/QuantFunc/transformer/ for .safetensors files."""
    try:
        from .model_auto_loader import get_models_dir
        tf_dir = os.path.join(get_models_dir(), "transformer")
        if os.path.isdir(tf_dir):
            files = []
            for root, _, filenames in os.walk(tf_dir):
                for f in filenames:
                    if f.endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root, f), tf_dir)
                        files.append(rel.replace("\\", "/"))
            if files:
                return ["None"] + sorted(files)
    except Exception:
        pass
    return ["None"]


class QuantFuncTransformerAutoLoader:
    """Auto-load transformer weights from models/QuantFunc/transformer/ directory.

    Recursively scans for .safetensors files and presents them as a dropdown.
    Outputs the file path as a string for connecting to ModelLoader's transformer input.
    """

    @classmethod
    def INPUT_TYPES(cls):
        tf_opts = _get_local_transformer_file_options()
        return {
            "required": {
                "transformer_file": (tf_opts, {"tooltip": "Transformer weights from models/QuantFunc/transformer/"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("transformer_path",)
    FUNCTION = "load_transformer"
    CATEGORY = "QuantFunc"

    def load_transformer(self, transformer_file):
        if not transformer_file or transformer_file == "None":
            return ("",)

        from .model_auto_loader import get_models_dir
        tf_path = os.path.join(get_models_dir(), "transformer", transformer_file)
        if not os.path.exists(tf_path):
            raise RuntimeError("Transformer file not found: {}".format(tf_path))
        return (tf_path,)


# ============================================================================
# Node: QuantFunc LoRA Auto Loader
# ============================================================================

def _get_comfyui_dir():
    """Return the ComfyUI root directory (parent of custom_nodes)."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_lora_file_options():
    """Recursively scan ComfyUI/models/loras/ for .safetensors files."""
    try:
        lora_dir = os.path.join(_get_comfyui_dir(), "models", "loras")
        if os.path.isdir(lora_dir):
            files = []
            for root, _, filenames in os.walk(lora_dir):
                for f in filenames:
                    if f.endswith(".safetensors"):
                        rel = os.path.relpath(os.path.join(root, f), lora_dir)
                        files.append(rel.replace("\\", "/"))
            if files:
                return ["None"] + sorted(files)
    except Exception:
        pass
    return ["None"]


class QuantFuncLoRAAutoLoader:
    """Auto-load LoRA weights from models/QuantFunc/lora/ directory.

    Scans the lora directory for .safetensors files and presents them
    as a dropdown. Appends the selected LoRA to the pipeline.
    """

    @classmethod
    def INPUT_TYPES(cls):
        lora_opts = _get_lora_file_options()
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "lora_file": (lora_opts, {"tooltip": "LoRA weights from models/loras/"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                           "tooltip": "LoRA weight scale (1.0 = full strength)"}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add_lora"
    CATEGORY = "QuantFunc"

    def add_lora(self, pipeline, lora_file, scale):
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))

        if lora_file and lora_file != "None":
            lora_path = os.path.join(_get_comfyui_dir(), "models", "loras", lora_file)
            if not os.path.exists(lora_path):
                raise RuntimeError("LoRA file not found: {}".format(lora_path))
            loras = list(cfg["options"].get("lora", []))
            if scale != 1.0:
                loras.append("{}:{}".format(lora_path, scale))
            else:
                loras.append(lora_path)
            cfg["options"]["lora"] = loras

        return (cfg,)


# ============================================================================
# Node: QuantFunc LoRA
# ============================================================================

class QuantFuncLoRALoader:
    """Append a LoRA to the pipeline. Chain multiple LoRA nodes together."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "lora_path": ("STRING", {"default": "", "tooltip": "Path to LoRA safetensors file"}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                           "tooltip": "LoRA weight scale (1.0 = full strength)"}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "add_lora"
    CATEGORY = "QuantFunc"

    def add_lora(self, pipeline, lora_path, scale):
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))

        if lora_path:
            loras = list(cfg["options"].get("lora", []))
            if scale != 1.0:
                loras.append(f"{lora_path}:{scale}")
            else:
                loras.append(lora_path)
            cfg["options"]["lora"] = loras

        return (cfg,)


# ============================================================================
# Node: QuantFunc LoRA Config
# ============================================================================

class QuantFuncLoRAConfig:
    """Configure LoRA merge strategy. Place after LoRA nodes, before Generate."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "max_rank": ("INT", {"default": 512, "min": 1, "max": 1024, "step": 1,
                              "tooltip": "Maximum LoRA rank for SVD merge (higher = more accurate, more VRAM)"}),
                "merge_method": (["auto", "itc", "awsvd", "rop", "concat"], {"default": "auto",
                                  "tooltip": "auto: best for model type; itc: IT+C; awsvd: activation-weighted SVD; rop: ROP+W; concat: concatenate weights"}),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_PIPELINE",)
    RETURN_NAMES = ("pipeline",)
    FUNCTION = "configure"
    CATEGORY = "QuantFunc"

    def configure(self, pipeline, max_rank, merge_method):
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))

        cfg["options"]["lora_max_rank"] = max_rank
        if merge_method == "concat":
            cfg["options"]["lora_concat"] = True
            cfg["options"]["lora_merge_method"] = "auto"
        else:
            cfg["options"]["lora_merge_method"] = merge_method
            cfg["options"]["lora_concat"] = False

        return (cfg,)


# ============================================================================
# Node: QuantFunc Generate
# ============================================================================

def _reinhard_color_match(target_hwc, reference_hwc, strength):
    """Plugin-side Reinhard color transfer in Lab space — mirrors the engine's
    apply_reinhard_color_match (src/ImageUtils.cpp). Matches `target`'s per-channel
    Lab mean/std to `reference`, then blends by `strength`. Done in Python (no C++
    change): the engine has the same algorithm but no i2i-options wiring for it.
    target_hwc / reference_hwc: float32 H×W×C RGB in [0,1]. Returns float32 HWC."""
    if strength <= 0.0:
        return target_hwc
    try:
        import cv2
    except Exception:
        logging.warning("[QuantFunc] color_match needs opencv-python; skipping")
        return target_hwc
    tgt = np.clip(target_hwc, 0.0, 1.0).astype(np.float32)
    ref = np.clip(reference_hwc, 0.0, 1.0).astype(np.float32)
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_RGB2Lab)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_RGB2Lab)
    # Per-channel mean/std via OpenCV's SIMD meanStdDev (the same call the engine
    # uses); then the affine match + blend in-place (no large temporaries). ~2x
    # faster than a per-channel numpy loop and avoids the engine's manual
    # per-pixel tensor<->cv::Mat scalar copies.
    t_mean, t_std = cv2.meanStdDev(tgt_lab)
    r_mean, r_std = cv2.meanStdDev(ref_lab)
    t_mean = t_mean.reshape(3).astype(np.float32); t_std = t_std.reshape(3).astype(np.float32)
    r_mean = r_mean.reshape(3).astype(np.float32); r_std = r_std.reshape(3).astype(np.float32)
    scale = np.where(t_std > 1e-6, r_std / t_std, 1.0).astype(np.float32)
    tgt_lab -= t_mean; tgt_lab *= scale; tgt_lab += r_mean    # out = (lab - tm)*scale + rm
    out = cv2.cvtColor(tgt_lab, cv2.COLOR_Lab2RGB)
    if strength < 1.0:                                         # out = (1-s)*tgt + s*out
        out = cv2.addWeighted(tgt, 1.0 - strength, out, strength, 0.0)
    return np.clip(out, 0.0, 1.0, out=out)


class QuantFuncGenerate:
    """Generate an image. Creates/reuses a cached pipeline from the config.
    Edit mode is auto-detected when ref_image is connected.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "prompt": ("STRING", {"default": "A cute cat", "multiline": True}),
                "width": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
                "height": ("INT", {"default": 1024, "min": 256, "max": 8192, "step": 64}),
                "steps": ("INT", {"default": 8, "min": 1, "max": 100}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
                "guidance_scale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1}),
            },
            "optional": {
                "ref_images": ("QUANTFUNC_IMAGE_LIST", {"tooltip": "Reference images for edit mode (from ImageList node)"}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
                "true_cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.1, "tooltip": "Classical CFG (needs a negative prompt). 1.0 = OFF (default) — correct for distilled / few-step models. Raise (e.g. 4.0) only for base / non-distilled models."}),
                "sampler_name": (["euler", "heun", "dpm++2m", "dpm++2m_sde", "euler_a", "ddim"], {
                    "default": "euler",
                    "tooltip": "Sampling algorithm:\n"
                               "• euler — 1st order, fast, deterministic\n"
                               "• heun — 2nd order, higher quality, 2x slower\n"
                               "• dpm++2m — 2nd order multistep, deterministic\n"
                               "• dpm++2m_sde — dpm++2m + noise (use sampler_eta)\n"
                               "• euler_a — euler + noise (use sampler_eta)\n"
                               "• ddim — classic DDIM, deterministic (eta=0) or stochastic (eta>0)",
                }),
                "sampler_eta": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Noise scale for stochastic samplers (dpm++2m_sde, euler_a, ddim).\n"
                               "0 = deterministic (no effect). Only used by stochastic samplers.\n"
                               "Recommended 0.2~0.5 for ≤20 steps. Higher eta needs more steps.",
                }),
                "activate_unload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Let ComfyUI unload this pipeline when it needs VRAM for "
                               "coexisting plugins.\n\n"
                               "False (default): never unload. Weights stay pinned in RAM "
                               "(~17 GB) after each generate; fastest subsequent runs (0 "
                               "overhead). ComfyUI's free_memory requests are refused.\n\n"
                               "True: listen to ComfyUI's memory-pressure signals. On "
                               "pressure — another plugin needs VRAM → offload GPU and "
                               "madvise the disk-backed transformer backup to return "
                               "~13 GB of RAM to the OS. On 'Free Model and Node Cache' → "
                               "full destroy. Next run page-faults the backup from disk "
                               "(~5-15 s on SSD). Enables the disk-backed coalesced backup "
                               "at pipeline creation, so a ~13 GB file will exist in "
                               "$QUANTFUNC_BACKUP_DIR / $TMPDIR / /var/tmp / /tmp while "
                               "the pipeline is alive (unlinked on clean exit).",
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID", "workflow_prompt": "PROMPT"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_NODE = True
    FUNCTION = "generate"
    CATEGORY = "QuantFunc"

    def generate(self, pipeline, prompt, width, height, steps, seed,
                 guidance_scale, ref_images=None,
                 negative_prompt="", true_cfg_scale=1.0,
                 sampler_name="euler", sampler_eta=0.0,
                 activate_unload=False, unload_mode=None, unload_every_time=None,
                 unique_id=None, workflow_prompt=None):
        # Backwards-compat with older saved workflows that used the
        # unload_mode dropdown or the unload_every_time bool: both are
        # collapsed onto a single activate_unload bool — true if the user
        # asked to release under pressure (any non-"none" legacy value),
        # false otherwise.  Silent; the new widget is the source of truth.
        if unload_mode is not None:
            activate_unload = activate_unload or (unload_mode != "none")
        if unload_every_time:
            activate_unload = True
        # Internal _unload_modes bookkeeping still uses the string "gpu+cpu"
        # / "none" keys the free_memory hook understands; map the bool.
        unload_mode_internal = "gpu+cpu" if activate_unload else "none"
        import torch

        # Auto-detect edit mode from ref_images
        cfg = dict(pipeline)
        cfg["options"] = dict(cfg.get("options", {}))
        # Propagate activate_unload into the pipeline's comp_opts so the
        # C++ side builds the transformer's coalesced backup on disk
        # (enabling future releaseRamPages) instead of in pinned RAM.
        # Must be set at create time — the coalesced backup's storage
        # medium is baked in during warmup and can't be switched later
        # without a 13 GB re-pack.
        cfg["options"]["activate_unload"] = bool(activate_unload)
        # Unpack ImageList dict format
        ref_img_resize = "720"
        ref_img_resize_others = "720"
        # Inpaint defaults (overridden if mask present in ref_images dict).
        inpaint_mask = None
        inpaint_strength = 1.0
        inpaint_grow = 6
        inpaint_blur = 0.0
        inpaint_no_snap = False
        edit_strength = 0.0
        color_match = 0.0
        if ref_images is not None and isinstance(ref_images, dict):
            # New: ref_img_resize ("720" / "1024" / "origin")
            # Backwards compat: old workflows may still send keep_ref_img_size (bool)
            if "ref_img_resize" in ref_images:
                ref_img_resize = ref_images["ref_img_resize"]
            elif ref_images.get("keep_ref_img_size"):
                ref_img_resize = "1024"
            ref_img_resize_others = ref_images.get("ref_img_resize_others", "720")
            # Inpaint payload (optional): mask tensor + 4 knobs.
            inpaint_mask = ref_images.get("mask")
            inpaint_strength = ref_images.get("mask_strength", 1.0)
            inpaint_grow = ref_images.get("mask_grow", 6)
            inpaint_blur = ref_images.get("mask_blur", 0.0)
            inpaint_no_snap = ref_images.get("mask_no_snap", False)
            edit_strength = float(ref_images.get("edit_strength", 0.0))
            color_match = float(ref_images.get("color_match", 0.0))
            ref_images = ref_images["images"]
        # edit_mode controls which Pipeline class the C++ engine instantiates:
        #   - QwenImage / QwenImageEdit are SEPARATE classes; QwenImageEditPipeline
        #     only supports generate_edit() and rejects generate() at runtime.
        #     So edit_mode MUST track ref_images presence for these.
        #   - Klein uses ONE pipeline class for both modes (edit_mode=true just
        #     pre-loads VAE encoder); toggling triggers recreate but t2i still
        #     works since the pipeline serves both.
        # Set conditionally — accept the recreate cost when toggling ref_images.
        cfg["options"]["edit_mode"] = ref_images is not None

        # Collect live QuantFuncGenerate node ids from the current workflow.
        # Exclude nodes whose required `pipeline` input isn't connected — those
        # never execute and should release their cached pipeline refs.
        alive_ids = None
        if isinstance(workflow_prompt, dict):
            alive_ids = set()
            for nid, node in workflow_prompt.items():
                if not (isinstance(node, dict) and node.get("class_type") == "QuantFuncGenerate"):
                    continue
                pipe_in = node.get("inputs", {}).get("pipeline")
                # Connected inputs are [source_node_id, output_slot]; anything
                # else (None / missing) means dangling.
                if not (isinstance(pipe_in, list) and len(pipe_in) == 2):
                    continue
                alive_ids.add(nid)
        cache_key = _manager.ensure_pipeline(cfg, node_id=unique_id,
                                             alive_node_ids=alive_ids)

        # Create ComfyUI progress bar
        pbar = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(steps)
        except Exception:
            pass

        # /dev/shm QFRAW staging files — RAM-backed (tmpfs); MUST be unlinked
        # once the engine has consumed them (see the `finally` below) or every
        # edit/inpaint generation leaks a raw RGB blob (+ mask) into RAM.
        tmp_paths = []
        mask_path = None
        try:
            if ref_images is not None:
                # Write each ref image as a "QFRAW01" raw uint8 RGB blob to
                # /dev/shm. Backend ImageUtils::load_image detects the magic
                # and skips cv::imread entirely — ~80 ms saved per 1728×2304
                # ref vs the prior BMP encode + cv::imread decode round-trip.
                # Format: 8-byte magic "QFRAW01\0" + uint32 H + uint32 W +
                # H*W*3 uint8 RGB bytes.
                tmp_paths = []
                try:
                    import cv2
                    _have_cv2 = True
                except ImportError:
                    _have_cv2 = False
                from PIL import Image  # used only for output preview path
                for img_tensor in ref_images:
                    for i in range(img_tensor.shape[0]):
                        fd, tmp_path = tempfile.mkstemp(suffix=".qfraw", dir="/dev/shm")
                        os.close(fd)
                        t = img_tensor[i]
                        # ComfyUI IMAGE is [B, H, W, C] FP32 in [0,1]. Convert
                        # to HWC uint8 [0,255]. cv2.convertScaleAbs is the
                        # fast SIMD path; numpy fallback is fine.
                        arr_f32 = t.numpy() if t.device.type == "cpu" else t.cpu().numpy()
                        if _have_cv2:
                            img_np = cv2.convertScaleAbs(arr_f32, alpha=255.0)
                        else:
                            img_np = (arr_f32 * 255).clip(0, 255).astype(np.uint8)
                        h, w = img_np.shape[0], img_np.shape[1]
                        header = b"QFRAW01\x00" + h.to_bytes(4, "little") + w.to_bytes(4, "little")
                        with open(tmp_path, "wb") as f:
                            f.write(header)
                            # tobytes() is C-contiguous HWC uint8 — RGB order
                            # matches ComfyUI's IMAGE convention, so no swap.
                            f.write(img_np.tobytes())
                        tmp_paths.append(tmp_path)
                neg = negative_prompt if isinstance(negative_prompt, str) and negative_prompt else ""
                i2i_opts = {}
                if sampler_name != "euler":
                    i2i_opts["sampler"] = sampler_name
                if sampler_eta > 0.0:
                    i2i_opts["eta"] = sampler_eta
                # Per-image resize: 主图 (refs[0]) uses main_image_resize,
                # 参考图 2~10 (refs[1..]) use ref_image_resize_others.
                # Build per-image array for C++ backend.
                num_refs = len(tmp_paths)
                if num_refs > 1 and ref_img_resize != ref_img_resize_others:
                    resize_arr = [ref_img_resize] + [ref_img_resize_others] * (num_refs - 1)
                    i2i_opts["ref_img_resize"] = resize_arr
                else:
                    i2i_opts["ref_img_resize"] = ref_img_resize
                # edit_strength → engine img2img strength (natively wired).
                if edit_strength > 0.0:
                    i2i_opts["edit_strength"] = edit_strength
                i2i_opts_json = json.dumps(i2i_opts) if i2i_opts else None

                # Inpaint mask: ComfyUI MASK is [B, H, W] float32 in [0,1]
                # (white=mask). Save first slice as raw uint8 L8 with
                # "QFRAWL1" magic header — backend skips PNG decode entirely.
                # Format: 8-byte magic + uint32 H + uint32 W + H*W uint8.
                mask_path = None
                if inpaint_mask is not None and inpaint_strength > 0.0:
                    fd, mask_path = tempfile.mkstemp(suffix=".qfraw", dir="/dev/shm")
                    os.close(fd)
                    m = inpaint_mask
                    # Accept [B,H,W] / [B,1,H,W] / [H,W]
                    if m.dim() == 4: m = m[0, 0]
                    elif m.dim() == 3: m = m[0]
                    m_np = (m.detach().cpu().clamp(0.0, 1.0).numpy() * 255).astype(np.uint8)
                    h, w = m_np.shape[0], m_np.shape[1]
                    header = b"QFRAWL1\x00" + h.to_bytes(4, "little") + w.to_bytes(4, "little")
                    with open(mask_path, "wb") as f:
                        f.write(header)
                        f.write(m_np.tobytes())

                arr = _manager.image_to_image(
                    cache_key=cache_key,
                    prompt=prompt, ref_paths=tmp_paths,
                    height=height, width=width, steps=steps, seed=seed,
                    true_cfg_scale=true_cfg_scale, negative_prompt=neg,
                    options_json=i2i_opts_json, pbar=pbar,
                    mask_path=mask_path,
                    mask_strength=inpaint_strength,
                    mask_grow=inpaint_grow,
                    mask_blur=inpaint_blur,
                    mask_no_snap=inpaint_no_snap)
                # color_match (潜在色彩匹配): plugin-side Reinhard post-decode against
                # the main reference image. Engine has the same algorithm but no
                # i2i-options wiring, so we do it here — no C++ change. Edit-mode only.
                if color_match > 0.0 and ref_images:
                    try:
                        ref0 = ref_images[0]
                        ref_np = (ref0[0] if ref0.dim() == 4 else ref0).detach().cpu().numpy()
                        arr = _reinhard_color_match(arr, ref_np, color_match)
                    except Exception as _cm_e:
                        logging.warning("[QuantFunc] color_match skipped: %s", _cm_e)
            else:
                t2i_opts = {}
                neg = negative_prompt if isinstance(negative_prompt, str) and negative_prompt else ""
                if neg and true_cfg_scale > 1.0:
                    t2i_opts["negative_prompt"] = neg
                    t2i_opts["true_cfg_scale"] = true_cfg_scale
                t2i_opts["sampler"] = sampler_name
                if sampler_eta > 0.0:
                    t2i_opts["eta"] = sampler_eta
                opts_json = json.dumps(t2i_opts) if t2i_opts else None
                logging.info("[QuantFunc] t2i sampler_name=%s, sampler_eta=%s, opts_json=%s",
                             sampler_name, sampler_eta, opts_json)

                arr = _manager.text_to_image(
                    cache_key=cache_key,
                    prompt=prompt, height=height, width=width,
                    steps=steps, seed=seed, guidance_scale=guidance_scale,
                    options_json=opts_json, pbar=pbar)

            # Persist the mode so the ComfyUI free_memory hook can look it up
            # for this pipeline. The hook inspects _unload_modes[k]:
            #   "gpu+cpu" (activate_unload=True)  → release on memory pressure
            #   "none"    (activate_unload=False) → refuse release requests
            # No per-gen action needed — the hook is the sole trigger.
            with _manager._lock:
                _manager._unload_modes[cache_key] = unload_mode_internal

            out = torch.from_numpy(arr).unsqueeze(0)
            return (out,)  # [1, H, W, 3]

        except InterruptedError:
            logging.info("[QuantFunc] Generation interrupted, returning blank image.")
            blank = torch.zeros(1, height, width, 3, dtype=torch.float32)
            return (blank,)
        finally:
            # Unlink the /dev/shm QFRAW staging files. By image_to_image's return
            # the engine has already read them; on interrupt they're unused. Either
            # way they must not accumulate in tmpfs (RAM). Output shm is freed
            # separately in the reader path.
            for _p in tmp_paths:
                try:
                    os.unlink(_p)
                except OSError:
                    pass
            if mask_path:
                try:
                    os.unlink(mask_path)
                except OSError:
                    pass


# ============================================================================
# Node: QuantFunc Image List
# ============================================================================

class QuantFuncImageList:
    """Reference images for edit mode. Single or multiple images supported.
    Optional MASK input enables inpaint (mirrors ComfyUI SetLatentNoiseMask):
    only the white region of the mask is regenerated; black region is preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        # main_image_mask = 主图遮罩(可选,触发 inpaint)
        optional["main_image_mask"] = ("MASK", {
            "tooltip": "主图的 inpaint 遮罩(白=重绘,黑=保留,等价 ComfyUI "
                       "SetLatentNoiseMask)。只有白色区域被模型重绘;"
                       "黑色区域通过后处理 snap 完整保留原图。",
        })
        # main_image_resize = 主图缩放
        optional["main_image_resize"] = (["720", "1024", "origin"], {
            "default": "720",
            "tooltip": "主图缩放模式:\n"
                       "  720  — 长边裁到 720 px(默认,最快)\n"
                       "  1024 — 长边裁到 1024 px(更高质量,稍慢)\n"
                       "  origin — 保留原尺寸,只把每边对齐到 16 的倍数",
        })
        # ref_image_2..ref_image_10 = 参考图 2-10
        for i in range(2, 11):
            optional[f"ref_image_{i}"] = ("IMAGE", {
                "tooltip": f"第 {i} 张参考图(可选)。最多 10 张。",
            })
        # ref_image_resize_others = 参考图 2~10 缩放
        optional["ref_image_resize_others"] = (["720", "1024", "origin"], {
            "default": "720",
            "tooltip": "参考图 2~10 的缩放模式:\n"
                       "  720  — 长边裁到 720 px(默认,省 VRAM)\n"
                       "  1024 — 长边裁到 1024 px\n"
                       "  origin — 保留原尺寸(图大可能 OOM)",
        })
        # mask_config = MASK 高级配置(可选,不连接走默认)。
        # 默认 = QuantFuncMaskConfig 的默认 = strength=1.0, grow=6, blur=0.0, no_snap=False。
        optional["mask_config"] = ("QUANTFUNC_MASK_CONFIG", {
            "tooltip": "可选:用 QuantFunc Mask Config 节点配置遮罩高级参数"
                       "(strength / grow / blur / no_snap)。不连接 = 全用默认值。",
        })
        # edit_strength = edit 模式 img2img 强度(引擎原生支持)
        optional["edit_strength"] = ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": "edit 模式 img2img 强度 (0~1,默认 0):\n"
                       "  0   — 标准 edit(纯按参考图重绘)\n"
                       "  >0  — 把主图当 img2img 起点,保留更多原图结构、降低色偏\n"
                       "        (值越大越接近原图)",
        })
        # color_match = 潜在色彩匹配(插件侧 Reinhard 后处理,镜像引擎算法)
        optional["color_match"] = ("FLOAT", {
            "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
            "tooltip": "潜在色彩匹配强度 (0~1,默认 0):解码后把输出色彩分布匹配到"
                       "主参考图(Reinhard,Lab 空间):\n"
                       "  0.0     — 不校正(最锐利,可能色偏)\n"
                       "  0.3~0.5 — 平衡(推荐)\n"
                       "  1.0     — 完全匹配(色彩最忠实,细节略软)",
        })
        return {
            "required": {
                # main_image = 主图
                "main_image": ("IMAGE", {"tooltip": "edit 模式的主图(被遮罩区域将被重绘)"}),
            },
            "optional": optional,
        }

    RETURN_TYPES = ("QUANTFUNC_IMAGE_LIST",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "QuantFunc"

    def combine(self, main_image, main_image_resize="720",
                ref_image_resize_others="720",
                main_image_mask=None, mask_config=None, **kwargs):
        images = [main_image]
        for i in range(2, 11):
            img = kwargs.get(f"ref_image_{i}")
            if img is not None:
                images.append(img)
        # Internal dict keys keep the legacy `ref_img_resize` / `mask` names so
        # QuantFuncGenerate.generate (which reads them) needs no parallel rename.
        out = {
            "images": images,
            "ref_img_resize": main_image_resize,
            "ref_img_resize_others": ref_image_resize_others,
            "edit_strength": float(kwargs.get("edit_strength", 0.0)),
            "color_match": float(kwargs.get("color_match", 0.0)),
        }
        if main_image_mask is not None:
            # Auto-align mask to main_image's pixel dims. Lets users wire any
            # ImageScale / Resize node into main_image without needing a
            # parallel resize for the mask (ComfyUI doesn't have a dedicated
            # MASK resize node — only Image-side scalers). main_image is
            # [B, H, W, C], main_image_mask is [B, H, W] or [B, 1, H, W].
            mask_t = main_image_mask
            img_h, img_w = main_image.shape[1], main_image.shape[2]
            mh = mask_t.shape[-2]
            mw = mask_t.shape[-1]
            if mh != img_h or mw != img_w:
                import torch.nn.functional as F
                m4 = mask_t.unsqueeze(1) if mask_t.dim() == 3 else mask_t
                m4 = F.interpolate(m4, size=(img_h, img_w),
                                    mode="bilinear", align_corners=False)
                mask_t = m4.squeeze(1) if mask_t.dim() == 3 else m4
            out["mask"] = mask_t
            # Defaults match QuantFuncMaskConfig defaults exactly.
            cfg = mask_config if isinstance(mask_config, dict) else {}
            out["mask_strength"] = float(cfg.get("mask_strength", 1.0))
            out["mask_grow"]     = int(cfg.get("mask_grow", 6))
            out["mask_blur"]     = float(cfg.get("mask_blur", 0.0))
            out["mask_no_snap"]  = bool(cfg.get("mask_no_snap", False))
        return (out,)


# ============================================================================
# Node: QuantFunc Mask Config (advanced inpaint knobs, optional)
# ============================================================================

class QuantFuncMaskConfig:
    """Inpaint MASK 高级配置(可选)。把 4 个边界参数打包成一个输出,接到
    QuantFunc Image List 的 `mask_config` 入口。不接就走默认值,跟 ComfyUI
    的 VAEEncodeForInpaint / SetLatentNoiseMask 保持一致。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "遮罩强度倍数 (0..1)。0 = 关闭 inpaint。",
                }),
                "mask_grow": ("INT", {
                    "default": 6, "min": 0, "max": 64, "step": 1,
                    "tooltip": "遮罩像素膨胀 N(对齐 ComfyUI VAEEncodeForInpaint "
                               "grow_mask_by 默认 6)。让接缝过渡更自然。",
                }),
                "mask_blur": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5,
                    "tooltip": "遮罩高斯模糊 sigma(像素;对齐 ComfyUI MaskBlur)。"
                               "0 = 边界硬切。",
                }),
                "mask_no_snap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "关闭最后一步对未遮罩区域的 snap 回原图。"
                               "默认关闭(snap 开,跟 ComfyUI 一致)。"
                               "开启 = 让模型决定整张图,边界过渡更柔和但"
                               "保留区会轻微飘移。",
                }),
            },
        }

    RETURN_TYPES = ("QUANTFUNC_MASK_CONFIG",)
    RETURN_NAMES = ("mask_config",)
    FUNCTION = "build"
    CATEGORY = "QuantFunc"

    def build(self, mask_strength, mask_grow, mask_blur, mask_no_snap):
        return ({
            "mask_strength": float(mask_strength),
            "mask_grow": int(mask_grow),
            "mask_blur": float(mask_blur),
            "mask_no_snap": bool(mask_no_snap),
        },)


# ============================================================================
# Node: QuantFunc Mask Scale By (mirrors ComfyUI ImageScaleBy, for MASK)
# ============================================================================

class QuantFuncMaskScaleBy:
    """按比例缩放 MASK,对称 ComfyUI 自带 ImageScaleBy(自带的不接 MASK 类型)。
    用法:LoadImageMask → QuantFunc Mask Scale By(同主图的 scale_by)→ main_image_mask"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK", {"tooltip": "要缩放的遮罩"}),
                "scale_by": ("FLOAT", {
                    "default": 1.0, "min": 0.01, "max": 8.0, "step": 0.01,
                    "tooltip": "缩放比例。要和主图的 ImageScaleBy 设成一样的值。",
                }),
                "method": (["bilinear", "nearest", "bicubic"], {
                    "default": "bilinear",
                    "tooltip": "插值方式。bilinear 默认平滑,nearest 保留硬边,"
                               "bicubic 最高质量但稍慢。",
                }),
            },
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "scale_by"
    CATEGORY = "QuantFunc"

    def scale_by(self, mask, scale_by, method):
        import torch.nn.functional as F
        # ComfyUI MASK is [B, H, W]. Add channel dim for interpolate.
        m4 = mask.unsqueeze(1) if mask.dim() == 3 else mask
        h, w = m4.shape[-2], m4.shape[-1]
        new_h = max(1, int(round(h * scale_by)))
        new_w = max(1, int(round(w * scale_by)))
        kwargs = {"size": (new_h, new_w), "mode": method}
        if method in ("bilinear", "bicubic"):
            kwargs["align_corners"] = False
        out = F.interpolate(m4, **kwargs)
        out = out.clamp(0.0, 1.0)
        if mask.dim() == 3:
            out = out.squeeze(1)
        return (out,)


# ============================================================================
# Node: QuantFunc Export
# ============================================================================

class QuantFuncExport:
    """Export a pre-quantized model directory.

    Two output formats:
      - diffusers (formerly "separated") — HF-style directory:
                     transformer/, text_encoder/, vae/, ... each component
                     its own safetensors. The traditional layout,
                     compatible with `--model-dir <out>` reload.
      - comfy_checkpoint (formerly "bundle", aka 全家桶) — single
                     safetensors with all components packed under
                     per-component prefixes (model.diffusion_model.*,
                     text_encoder.*, vae.*, vision_encoder.*). One file,
                     loadable directly as a ComfyUI checkpoint via the
                     QuantFunc plugin's bundled-checkpoint adapter.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipeline": ("QUANTFUNC_PIPELINE",),
                "export_path": ("STRING", {
                    "default": "",
                    "tooltip": "diffusers 模式: 输出到这个目录, 每个组件一个 safetensors\n"
                               "comfy_checkpoint 模式: 输出到这个目录下的单个 model.safetensors (全家桶)",
                }),
                "export_format": (["diffusers", "comfy_checkpoint"], {
                    "default": "diffusers",
                    "tooltip": "diffusers = HF 标准目录(每个组件独立文件)\n"
                               "comfy_checkpoint = 单文件 safetensors 打包所有组件 (全家桶)\n"
                               "  ↳ comfy_checkpoint 模式下 export_mode 强制为 'all'\n"
                               "    (transformer + text_encoder + vae + vision_encoder)",
                }),
                "export_mode": (["all", "custom"], {
                    "default": "all",
                    "tooltip": "(仅 diffusers) 'all' 复制整个模型(vae、tokenizer 等)用于独立使用; "
                               "'custom' 选择单个组件",
                }),
            },
            "optional": {
                "export_transformer": ("BOOLEAN", {"default": True}),
                "export_text_encoder": ("BOOLEAN", {"default": False}),
                "export_vision_encoder": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "export_model"
    CATEGORY = "QuantFunc"

    def export_model(self, pipeline, export_path,
                     export_format="diffusers", export_mode="all",
                     export_transformer=True, export_text_encoder=False,
                     export_vision_encoder=False):
        if not export_path:
            raise ValueError("export_path is required")

        if "options" not in pipeline:
            pipeline["options"] = {}

        # Accept both the new UI labels (diffusers / comfy_checkpoint) and the
        # legacy labels (separated / bundle) so old workflow JSON keeps working.
        # Engine-side string is unchanged: "bundle" or absent (= separated).
        is_bundle = export_format in ("comfy_checkpoint", "bundle")

        if is_bundle:
            # Bundle (= ComfyUI checkpoint) always packs the whole pipeline —
            # per-component selection doesn't apply. Force `all` so the worker
            # doesn't filter components and break the layout.
            pipeline["options"]["export_format"] = "bundle"
            pipeline["options"]["export_models"] = "all"
        else:
            pipeline["options"].pop("export_format", None)  # default = separated/diffusers
            if export_mode == "all":
                components = ["all"]
            else:
                components = []
                if export_transformer:
                    components.append("transformer")
                if export_text_encoder:
                    components.append("text_encoder")
                if export_vision_encoder:
                    components.append("vision_encoder")
                if not components:
                    raise ValueError("At least one component must be selected for export")
            pipeline["options"]["export_models"] = ",".join(components)

        _manager.export_model(pipeline, export_path)
        logging.info("[QuantFunc] Export complete (%s): %s", export_format, export_path)
        return {}


# ============================================================================
# Registration
# ============================================================================

NODE_CLASS_MAPPINGS = {
    "QuantFuncPipelineConfig": QuantFuncPipelineConfig,
    "QuantFuncModelLoader": QuantFuncModelLoader,
    "QuantFuncModelAutoLoader": QuantFuncModelAutoLoader,
    # QuantFuncBuildPipeline lives in nodes_format_adapters.py (one canonical
    # implementation; loaded after this map → __init__.py's update() lifts it
    # under the same registration key).
    "QuantFuncPrequantAutoLoader": QuantFuncPrequantAutoLoader,
    "QuantFuncPrecisionConfigAutoLoader": QuantFuncPrecisionConfigAutoLoader,
    "QuantFuncBaseSeriesModelAutoLoader": QuantFuncBaseSeriesModelAutoLoader,
    "QuantFuncBaseModelAutoLoader": QuantFuncBaseModelAutoLoader,
    "QuantFuncBaseModelAutoLoaderWithDownload": QuantFuncBaseModelAutoLoaderWithDownload,
    "QuantFuncTransformerAutoLoader": QuantFuncTransformerAutoLoader,
    "QuantFuncLoRAAutoLoader": QuantFuncLoRAAutoLoader,
    "QuantFuncLoRALoader": QuantFuncLoRALoader,
    "QuantFuncLoRAConfig": QuantFuncLoRAConfig,
    "QuantFuncGenerate": QuantFuncGenerate,
    "QuantFuncImageList": QuantFuncImageList,
    "QuantFuncMaskConfig": QuantFuncMaskConfig,
    "QuantFuncMaskScaleBy": QuantFuncMaskScaleBy,
    "QuantFuncExport": QuantFuncExport,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "QuantFuncPipelineConfig": "QuantFunc Pipeline Config",
    "QuantFuncModelLoader": "QuantFunc Model Loader",
    "QuantFuncModelAutoLoader": "QuantFunc Model Auto Loader",
    # QuantFuncBuildPipeline display name is set in nodes_format_adapters.py.
    "QuantFuncPrequantAutoLoader": "QuantFunc Prequant Auto Loader",
    "QuantFuncPrecisionConfigAutoLoader": "QuantFunc Precision Config Auto Loader",
    "QuantFuncBaseSeriesModelAutoLoader": "QuantFunc Base Series Model Auto Loader",
    "QuantFuncBaseModelAutoLoader": "QuantFunc Base Model Auto Loader",
    "QuantFuncBaseModelAutoLoaderWithDownload": "QuantFunc Base Model Auto Loader with Download",
    "QuantFuncTransformerAutoLoader": "QuantFunc Transformer Auto Loader",
    "QuantFuncLoRAAutoLoader": "QuantFunc LoRA Auto Loader",
    "QuantFuncLoRALoader": "QuantFunc LoRA",
    "QuantFuncLoRAConfig": "QuantFunc LoRA Config",
    "QuantFuncGenerate": "QuantFunc Generate",
    "QuantFuncImageList": "QuantFunc Image List",
    "QuantFuncMaskConfig": "QuantFunc Mask Config",
    "QuantFuncMaskScaleBy": "QuantFunc Mask Scale By",
    "QuantFuncExport": "QuantFunc Export",
}
