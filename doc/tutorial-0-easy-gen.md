# Must-Read for Beginners: Generate Your First Image with Model AutoLoader

[中文版本](tutorial-0-easy-gen_zh.md)

## Overview

This is the simplest way to get started. The **QuantFunc Model AutoLoader** node **automatically downloads** and configures models — just pick from dropdown menus, no manual downloads or paths needed.

The minimal auto-download generation chain is:

```
Model AutoLoader → Build Pipeline → Generate → Preview Image
```

> Model AutoLoader outputs ComfyUI-native `model` / `clip` / `vae` handles. Wire them into **QuantFunc Build Pipeline** to assemble the pipeline, then into **Generate** to produce an image (same as manual loading — it just saves you the download and the paths).

![Model AutoLoader node](../assets/node-model-auto-loader.png)

> **Sample workflow:** [`workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json) — its "Load with base model or prequanted model" group contains the Model AutoLoader; wire its `model`/`clip`/`vae` into Build Pipeline to auto-download and generate.

> **Want more control?** This tutorial is a simplified auto-download version of [Tutorial 1 (Runtime Quantization)](tutorial-1-use-without-quantfunc-models.md). For advanced features like local models, LoRA stacking, and pipeline configuration, see Tutorial 1, or the illustrated guide [Model Loading & API Key](model-loading-and-apikey_zh.md) (Chinese).

## Prerequisites

1. ComfyUI-QuantFunc plugin installed (see [README](../README.md))
2. CUDA 13.0+ runtime and cuDNN 9.x
3. Internet connection (models are downloaded automatically on first use)

## Steps

### Step 1: Import the Workflow

Import `workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json` in ComfyUI and find the **Model AutoLoader** in the "Load with base model or prequanted model" group. Wire its `model` / `clip` / `vae` outputs into **QuantFunc Build Pipeline** (then on to Generate → Preview).

### Step 2: Configure Model AutoLoader

In the **QuantFunc Model AutoLoader** node:

| Parameter | Description |
|-----------|-------------|
| `model_series` | Select a model series (e.g., `QuantFunc/Z-Image-Series`) — required |
| `data_source` | Download source: `modelscope` (China) or `huggingface` |
| `transformer` (optional) | Select specific transformer weights (format `Series/name`); leave as `None` to use the series' default transformer |

> **Tip:** The base model's GPU variant (`50x-below` for RTX 20/30/40, `50x-above` for RTX 50) is chosen **automatically** by the node based on your GPU — no manual selection needed. The `transformer` dropdown lists all weight variants of the series; leaving it `None` uses the default, or pick a specific variant manually.
>
> The backend (svdq / lighting) is also **auto-detected** by the engine from the weight metadata — neither Model Loader nor AutoLoader has a `model_backend` parameter.

### Step 3: Configure Generation Parameters

In the **QuantFunc Generate** node:

| Parameter | Suggested Value |
|-----------|-----------------|
| `prompt` | Your text prompt (e.g., "A cute cat") |
| `width` / `height` | `1024` x `1024` |
| `steps` | `8` (Lightning distilled) or `20` (full model) |
| `seed` | Any number, or select `randomize` for auto-random |
| `guidance_scale` | `0` (Lightning distilled) or `3.5` (full model) |

### Step 4: Run

Click **Queue Prompt**. On first run, the plugin automatically downloads the model (speed depends on your connection). Subsequent runs use the cached model.

The generated image appears in the **Preview Image** node.

## Next Steps

- Want to use your own local models? → [Tutorial 1: Runtime Quantization](tutorial-1-use-without-quantfunc-models.md)
- Want to export quantized models for faster loading? → [Tutorial 2: Export Runtime-Quantized Models](tutorial-2-export-quantized-models.md)
- Want to download and use pre-exported models? → [Tutorial 3: Download & Use Pre-exported Models](tutorial-3-download-quantfunc-models.md)
- Illustrated deep-dive on each loading method / parameter → [Model Loading & API Key](model-loading-and-apikey_zh.md) (Chinese)
