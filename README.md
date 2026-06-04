<div align="center" style="margin-top: 50px;">
  <img src="https://raw.githubusercontent.com/QuantFunc/ComfyUI-QuantFunc/main/assets/logo.webp" width="300" alt="QuantFunc Logo">
</div>

# ComfyUI-QuantFunc

[中文说明](README_zh.md)

## 1. Introduction

ComfyUI plugin for **QuantFunc** — the fastest diffusion model inference engine. Run quantized text-to-image and image editing models at 2x–11x speed with zero Python model dependencies.

**Key features:**
- Native C++/CUDA acceleration via `libquantfunc.so` / `quantfunc.dll`
- SVDQ (offline quantization) + Lighting (runtime quantization) dual engine
- Zero-cost LoRA stacking
- Image editing with reference images
- Export runtime-quantized models with LoRA fusion support
- Auto-update from ModelScope

## Version History

| Plugin (`comfy`) | Engine (`lib`) | Summary |
|:---:|:---:|---|
| **0.0.02** *(current)* | **0.0.07** | v2 loader architecture · inpainting · full GPU coverage · faster editing — details below |
| 0.0.01 | 0.0.01 – 0.0.06 | Base release: runtime/offline quantization · model & LoRA loaders · reference-image editing · export · auto-update |

### What's New in 0.0.02 (engine 0.0.07)

**🎯 Ease of Use**
- **v2 loaders** — separate `MODEL` / `CLIP` / `VAE` sockets feed a **Build Pipeline** node, so models wire up the ComfyUI-native way instead of one monolithic loader.
- **Universal format adapters** — load **diffusers / BFL (Flux) / nunchaku SVDQ / bundled-checkpoint (全家桶) / HF** layouts automatically, with no manual conversion.
- **Base Model Auto Loader** with one-click download; the plugin also auto-pulls the matching engine on first startup.

**🧩 Model Support**
- **SVDQ** (offline quantization) **+ Lighting** (runtime BF16/FP16 → 4-bit) dual engine.
- Pipelines: **Z-Image · QwenImage · QwenImage-Edit · Flux.2 Klein**.
- **Full GPU coverage** (engine 0.0.07): consumer **RTX 20 / 30 / 40 / 50-series**, datacenter **A100 / H100 / H200 / B100 / B200 / GB300**, workstation **RTX 6000 Ada / RTX PRO 6000 Blackwell** — across **CUDA 12 & 13**.

**⚡ Performance**
- **Consumer GPUs run native SASS** — *no first-run JIT compile stall* on 20/30/40/50-series (datacenter/workstation cards JIT once, then cache).
- Native **FP4 (NVFP4)** on Blackwell (SM120) — the fastest 4-bit path.
- **QFRAW raw staging** for reference images & masks skips the PNG/BMP encode (~80 ms saved per ref).
- **Multi-pipeline CPU↔GPU coexistence** — swap pipelines without a full reload; idle workers auto-free VRAM.

**✨ New Features**
- **Inpainting** — `MASK` input plus **Mask Config** and **Mask Scale By** nodes (white = regenerate, black = preserve), mirroring ComfyUI's SetLatentNoiseMask.
- **Build Pipeline** node (v2 assembly) with per-component precision control.
- Robust **worker-process architecture** — CPU↔GPU model swap + zombie-worker cleanup.

**🛡️ Stability & Security**
- Fixed a **`/dev/shm` RAM leak** — edit/inpaint staging files are now always cleaned up.
- **Zip-slip guard** on dependency-archive extraction.
- **IPC bound-check** on the worker → host image transfer.

> The plugin auto-pulls the matching engine on startup: bumping `comfy` to **0.0.02** lets the updater fetch engine **0.0.07** from ModelScope (older `comfy` stays capped at engine 0.0.06).

## 2. Installation

### 2.1 Method A: Clone from Git (Recommended)

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/QuantFunc/ComfyUI-QuantFunc.git
```

The plugin will **automatically download** the latest compatible `libquantfunc.so` (Linux) or `quantfunc.dll` (Windows) from ModelScope on first startup. No manual binary download needed.

### 2.2 Method B: Manual Installation

1. Download or clone this repository into `ComfyUI/custom_nodes/`:

```
ComfyUI/
└── custom_nodes/
    └── ComfyUI-QuantFunc/
        ├── __init__.py
        ├── nodes.py
        ├── worker.py
        ├── auto_update.py
        └── bin/
            ├── linux/
            │   └── version.json
            └── windows/
                └── version.json
```

2. Start ComfyUI — the plugin auto-downloads the library binary on first run.

3. (Optional) To skip auto-download, manually place the binary:
   - **Linux:** Download `libquantfunc.so` → `bin/linux/`
   - **Windows:** Download `quantfunc.dll` → `bin/windows/`

### 2.3 System Requirements

| Requirement | Minimum |
|-------------|---------|
| **GPU** | NVIDIA RTX 20 series or newer (CC 7.5+) |
| **VRAM** | 8 GB |
| **Driver** | NVIDIA ≥ 560 |
| **CUDA Runtime** | 13.0+ |
| **cuDNN** | 9.x |
| **OS** | Linux (glibc 2.31+) or Windows 10/11 |
| **Python** | 3.9+ (ComfyUI's embedded Python) |

### 2.4 Runtime Dependencies

#### Linux

```bash
# CUDA 12 runtime libraries
sudo apt install cuda-libraries-12-8
# or individual packages:
sudo apt install libcublas-12-8 libcurand-12-8 libcusolver-12-8 libcusparse-12-8 libnvjitlink-12-8

# cuDNN 9
sudo apt install libcudnn9-cuda-12

# --- OR ---

# CUDA 13 runtime libraries
sudo apt install cuda-libraries-13-0
# or individual packages:
sudo apt install libcublas-13-0 libcurand-13-0 libcusolver-13-0 libcusparse-13-0 libnvjitlink-13-0

# cuDNN 9
sudo apt install libcudnn9-cuda-13
```

#### Windows

- **NVIDIA Driver** ≥ 560 (provides CUDA runtime DLLs)
- **Visual C++ Redistributable** 2015-2022 ([download](https://aka.ms/vs/17/release/vc_redist.x64.exe))
- **cuDNN 9.x** ([download](https://developer.nvidia.com/cudnn))

### 2.5 ModelScope Dependency (for auto-update)

Auto-update requires `modelscope` Python package:

```bash
pip install modelscope
```

If `modelscope` is not installed, auto-update is silently skipped. You can manually download binaries from:
- https://www.modelscope.cn/models/QuantFunc/Plugin

### 2.6 Verify Installation

After starting ComfyUI, check the console for:

```
[QuantFunc] Checking for updates (plugin v0.0.01, lib v0.0.01)...
[QuantFunc] Library is up to date (v0.0.01)
```

If the library was not found:

```
[QuantFunc] No library found, checking ModelScope for download (plugin v0.0.01)...
[QuantFunc] Downloading libquantfunc.so v0.0.01 from ModelScope...
[QuantFunc] Updated libquantfunc.so to v0.0.01. Restart ComfyUI to use the new version.
```

## 3. Usage

See [doc/](doc/) for detailed tutorials and [workflow_sample/README.md](workflow_sample/README.md) for node reference.

### Must-Read for Beginners: Generate Your First Image in 3 Nodes

The easiest way to get started — import the Easy Gen workflow, pick a model from the dropdown, and the plugin auto-downloads everything. No manual model downloads or path configuration needed.

> **[Beginners: Easy Gen →](doc/tutorial-0-easy-gen.md)**

### 3.1 Runtime Quantization: Quantize BF16/FP16 Models to 4bit for Accelerated Inference

The **Lighting backend** provides **runtime quantization** — it uses the Lighting engine to quantize any diffusers-format BF16/FP16 model (e.g., [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)) to 4bit at load time for accelerated inference. Just set `model_backend` to `lighting` and leave `transformer_path` empty — no pre-quantized model download needed.

> **[Tutorial 1: Runtime Quantization →](doc/tutorial-1-use-without-quantfunc-models.md)**

### 3.2 Export Runtime-Quantized Models (with LoRA Fusion Support)

The Lighting export saves all runtime-quantized models to disk, so you don't need to re-quantize on every startup. If you've also stacked LoRAs, they are permanently fused into the exported weights — no LoRA nodes needed, no re-quantization, load and go.

> **[Tutorial 2: Export Runtime-Quantized Models →](doc/tutorial-2-export-quantized-models.md)**

### 3.3 Download and Use Pre-exported Quantized Models

QuantFunc has pre-exported commonly used models (runtime-quantized and ready to use). Download them directly from [ModelScope](https://www.modelscope.cn/models/QuantFunc) or [HuggingFace](https://huggingface.co/QuantFunc) — same 2x–11x inference speedup as runtime quantization, but with faster loading since the quantization step is skipped.

> **[Tutorial 3: Download & Use Pre-exported Models →](doc/tutorial-3-download-quantfunc-models.md)**

### 3.4 Example Workflows

Import from `workflow_sample/`:

| File | Use Case |
|------|----------|
| `QuantFunc-Easy-Gen.json` | **Beginners** — 3-node auto-download workflow |
| `QuantFunc-Text-to-Image-Workflow.json` | Text-to-image (SVDQ + Lighting side by side) |
| `QuantFunc-Image-to-Image-Workflow.json` | Image editing with reference images |
| `QuantFunc-Model-Export.json` | Export runtime-quantized models (supports LoRA fusion) |

## 4. Troubleshooting

| Issue | Solution |
|-------|----------|
| Worker failed to start | Check CUDA driver ≥ 560, ensure CUDA runtime libs installed |
| DLL/SO not found | Check `bin/linux/` or `bin/windows/` contains the library; restart ComfyUI to trigger auto-download |
| No log output | Update to latest library version (requires stderr log support) |
| cuDNN BAD_PARAM | Delete cuDNN algo cache and retry |
| Noisy output | Ensure model backend matches transformer weights (svdq vs lighting) |
| Auto-update fails | Install `modelscope` package, or manually download from ModelScope |

## 5. License

See [QuantFunc Plugin License](https://www.modelscope.cn/models/QuantFunc/Plugin).
