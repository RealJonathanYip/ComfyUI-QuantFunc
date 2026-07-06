# Tutorial 3: Download and Use Pre-exported Quantized Models

[中文版本](tutorial-3-download-quantfunc-models_zh.md)

## Overview

QuantFunc has pre-quantized and exported commonly used models. You can download these **pre-exported quantized models** directly and use them immediately — no need to perform runtime quantization yourself.

These models offer:

- **Instant loading**: No runtime quantization needed — loads pre-exported weights directly
- **Fast inference**: 2x-11x speedup
- **Ready to use**: Download, set path, and go

![Text-to-image workflow overview](../assets/t2i-overview.png)

> **Sample workflow:** [`workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json) — use the "Sample for text to image" group for t2i, or "Sample for edit Image" for editing.

## Step 1: Determine Your GPU Variant

QuantFunc provides different quantized model versions per GPU architecture:

| GPU Variant | GPUs | Notes |
|-------------|------|-------|
| `50x-below` | RTX 20/30/40 series | Optimized for Turing/Ampere/Ada |
| `50x-above` | RTX 50 series | Optimized for Blackwell |

> **Important:** The base model and transformer weights must use the **same GPU variant**. (Model AutoLoader auto-matches this when downloading — see [Tutorial 0](tutorial-0-easy-gen.md).)

## Step 2: Download the Model

Download pre-quantized models from:

- **ModelScope**: https://www.modelscope.cn/models/QuantFunc
- **HuggingFace**: https://huggingface.co/QuantFunc

QuantFunc provides two types of pre-exported models — **SVDQ** and **Lighting**, both usable directly. The backend is **auto-detected** by the engine from the weight metadata — no manual selection:

| Model Type | Auto-detected as | Notes |
|------------|------------------|-------|
| SVDQ | `svdq` | Offline SVD quantization; stacking LoRA needs a LoRA Config node |
| Lighting | `lighting` | Exported from runtime quantization, no low-rank compute overhead; LoRA stacks directly |

Each model repo typically contains:

```
QuantFunc/SomeModel/
├── model_index.json          # diffusers model index
├── transformer/              # pre-quantized transformer weights
│   └── *.safetensors
├── vae/                      # VAE weights
├── tokenizer/                # tokenizer
├── text_encoder/             # text encoder
└── scheduler/                # scheduler config
```

Download example:

```bash
# with modelscope (recommended in China)
pip install modelscope
modelscope download --model QuantFunc/YourModel-SVDQ --local_dir /path/to/QuantFunc-Model

# or with huggingface-cli
huggingface-cli download QuantFunc/YourModel-SVDQ --local-dir /path/to/QuantFunc-Model
```

![Model download page](../assets/t2-step2-download-model.png)

## Step 3: Import the Workflow and Configure

Import `workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json` in ComfyUI and use the "Sample for text to image" group. The chain:

```
QuantFunc Model Loader → QuantFunc Build Pipeline → QuantFunc Generate → Preview Image
```

In the **QuantFunc Model Loader** node, point at the model you downloaded (both SVDQ and Lighting use the same Model Loader — the engine auto-detects the backend):

| Parameter | Setting |
|-----------|---------|
| `model_dir` | QuantFunc model directory, e.g. `/path/to/QuantFunc-Model` |
| `transformer_path` | Transformer weights, e.g. `/path/to/QuantFunc-Model/transformer/model.safetensors` (SVDQ also accepts legacy nunchaku weights) |

![Configure Model Loader node](../assets/node-model-loader.png)

> Model Loader has **no** `model_backend` or `device` parameter: the backend is auto-detected, and `device`/`precision_config` live on **Build Pipeline** (see [Tutorial 1, Step 3](tutorial-1-use-without-quantfunc-models.md)). Since the model is already quantized, there's no runtime quantization — loading is fast.

## Step 4: Configure Generation Parameters

In the **QuantFunc Generate** node:

| Parameter | Suggested Value |
|-----------|-----------------|
| `prompt` | Your text prompt |
| `width` / `height` | `1024` x `1024` |
| `steps` | `20`-`30` (full model), `4` (Lightning distilled) |
| `guidance_scale` | `3.5` (distilled models use `0`–`1`) |
| `seed` | Any number |

![Configure Generate node parameters](../assets/node-generate.png)

## Step 5: Run

Click **Queue Prompt**. Pre-quantized models load fast, and the first inference needs no runtime quantization.

## Using LoRA (SVDQ Backend)

When using LoRA with the SVDQ backend, you **must** add a **QuantFunc LoRA Config** node to control the merge strategy:

```
Build Pipeline → QuantFunc LoRA (your LoRA) → QuantFunc LoRA Config (merge strategy) → QuantFunc Generate
```

**QuantFunc LoRA Config** parameters:

| Parameter | Description |
|-----------|-------------|
| `merge_method` | `auto` (recommended) — engine picks the best method |
| | `itc` — IT+C method |
| | `awsvd` — Activation-Weighted SVD |
| | `rop` — Rank-Orthogonal Projection (QuantFunc's algorithm) |
| | `concat` — direct concatenation (nunchaku's approach) |
| `max_rank` | Max merged LoRA rank (default is fine) |

> This is because the SVDQ model already has a fused pre-quantized low-rank structure, so a new LoRA must be merged with it. The Lighting backend does **not** need a LoRA Config node (see [Tutorial 1](tutorial-1-use-without-quantfunc-models.md)).

![SVDQ + LoRA + LoRA Config connection](../assets/node-lora-config.png)

## Image Editing Mode

Use the "Sample for edit Image" group instead:

1. Load a reference image with **LoadImage** → **QuantFunc Image List** → Generate's `ref_images`
2. Describe the edit in the prompt

Mask inpainting / color matching / size alignment are in the [Image Editing guide](image-edit-mask-colormatch_zh.md) (Chinese).

![Image editing workflow](../assets/edit-overview.png)

## SVDQ vs Lighting

| Dimension | SVDQ (offline quant) | Lighting (runtime quant) |
|-----------|----------------------|--------------------------|
| Model source | Must download QuantFunc pre-quantized models | Any diffusers FP16 model |
| First load | Fast (direct load) | Slow (first load runtime-quantizes; export via [Tutorial 2](tutorial-2-export-quantized-models.md) to speed up) |
| Inference speed | 2x-11x | 2x-11x (~20% faster than SVDQ on sub-RTX-50 machines) |
| Quant quality | Good (offline-optimized) | Good |
| LoRA usage | Needs a LoRA Config node | Stacks directly, zero cost |
| Flexibility | Limited to pre-quantized models | Any model |

> Both backends are auto-detected by the engine — you just point at the corresponding weights.

## FAQ

**Q: What's the difference between model_dir and transformer_path?**
A: `model_dir` points at the full diffusers model directory (with VAE, tokenizer, etc.); `transformer_path` points at the specific quantized transformer weights file (.safetensors).

**Q: Output is all noise?**
A: Make sure `transformer_path` points at **complete weights that match this model**, and that `precision_config` (on Build Pipeline) matches the model series. The backend is auto-detected — it can't (and needn't) be set manually; precision_config from different series cannot be mixed.

**Q: Can I mix 50x-below and 50x-above?**
A: No. Use the variant matching your GPU, or you may get errors or degraded performance.
