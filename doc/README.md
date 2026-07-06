# QuantFunc Documentation

[中文](#中文) | [English](#english)

---

<a id="english"></a>

## English

### Quick Start (Beginners)

Fastest path to your first image — the **Model AutoLoader** downloads everything for you:

1. Add a **QuantFunc Model Auto Loader**, pick a `model_series` (e.g. `QuantFunc/Z-Image-Series`). The GPU variant is auto-selected.
2. Wire its `model` / `clip` / `vae` into **QuantFunc Build Pipeline** (set `device`; leave `precision_config` at `[auto-derive]`).
3. Wire Build Pipeline into **QuantFunc Generate**, type a `prompt`, connect a **Preview Image** node.
4. Queue Prompt — the model auto-downloads on first run (cached afterward).

**No pre-quantized model needed:** point at a plain FP16 **diffusers** model and the **Lighting backend runtime-quantizes it to 4bit at load time** — no manual conversion. Full illustrated details (all three loading methods, `precision_config`, low-VRAM `prequant_weights`, API Key): [Model Loading & API Key](model-loading-and-apikey_zh.md) (Chinese).

### Illustrated Guides

These in-depth, screenshot-rich guides are in Chinese (`_zh`) except where an English version is noted.

| Guide | Description |
|----------|-------------|
| [Model Loading & API Key](model-loading-and-apikey_zh.md) (Chinese) | The three loading methods, downloading pre-quantized models, supported formats, `precision_config`, and the API Key |
| [Basic t2i, img2img & ControlNet](basic-t2i-img2img-controlnet_zh.md) (Chinese) | Basic text-to-image params, image-to-image (`init_img`), ControlNet — each with a full workflow example |
| [Image Edit: mask / color match / size align](image-edit-mask-colormatch_zh.md) (Chinese) | Reference-image editing, inpainting mask, color matching, pixel-align (`origin`) — per parameter |
| [Scheduler & Sampler Handbook](scheduler-and-sampler_zh.md) (Chinese) | Every Generate sampling knob: 22 samplers, 9 schedulers, CFG, FBCache acceleration |
| [Export Quantized Models](export-quantized-models.md) | Export runtime-quantized models to disk (with LoRA fusion), the Export node params, and distribution — English + [中文](export-quantized-models_zh.md) |
| [QwenImage-Layered: decompose to RGBA layers](qwen-image-layered_zh.md) (Chinese) | Split an image into N transparent layers (Layered Config / Layer Viewer) |
| [Ideogram 4: structured / regional prompts](ideogram4_zh.md) (Chinese) | Ideogram 4 text-to-image + the structured / regional Prompt Builder |

### Reference

| Document | Description |
|----------|-------------|
| [Node Reference (per-node params)](node-reference_zh.md) (Chinese) | Every QuantFunc node's INPUT_TYPES — each param (type / default / meaning) + cross-links |

### Installation

See the main [README.md](../README.md) for installation instructions.

---

<a id="中文"></a>

## 中文

### 快速上手（新手）

出第一张图的最快路径 —— **Model AutoLoader** 帮你自动下载一切：

1. 加一个 **QuantFunc Model Auto Loader**，选 `model_series`（如 `QuantFunc/Z-Image-Series`）。GPU 变体自动判定。
2. 把它的 `model` / `clip` / `vae` 接到 **QuantFunc Build Pipeline**（选 `device`；`precision_config` 保持默认 `[auto-derive]`）。
3. Build Pipeline 接到 **QuantFunc Generate**，填 `prompt`，再接一个 **Preview Image** 节点。
4. 点 Queue Prompt —— 首次运行自动下载模型（之后走缓存）。

**无需预量化模型：** 直接指向普通 FP16 **diffusers** 模型，**Lighting 后端会在加载时把它运行时量化为 4bit**，无需手动转换。完整图文详解（三大加载方式、`precision_config`、低显存 `prequant_weights`、API Key）见 [模型加载与 API Key](model-loading-and-apikey_zh.md)。

### 使用指南（图文）

| 指南 | 说明 |
|------|------|
| [模型加载与 API Key](model-loading-and-apikey_zh.md) | 三大类加载方式、下载 QuantFunc 预量化模型、支持的模型格式、precision_config，以及 API Key 的作用/获取/填写 |
| [基础文生图、img2img 与 ControlNet](basic-t2i-img2img-controlnet_zh.md) | 基础文生图参数、以图生图（init_img）、ControlNet 结构引导，各带完整 workflow 例子 |
| [图像编辑：mask / color match / 尺寸对齐](image-edit-mask-colormatch_zh.md) | 参考图编辑、局部重绘遮罩（Mask Config / Mask Scale By）、色彩匹配、像素级对齐（origin），逐参数详解 |
| [调度器与采样器手册](scheduler-and-sampler_zh.md) | Generate 上采样相关参数一次讲清：22 种采样器、9 种调度器、CFG、FBCache 加速，含快速推荐 |
| [导出量化模型（含 LoRA 融合）](export-quantized-models_zh.md) | 把运行时量化的模型导出到磁盘、Export 节点参数、多 LoRA 融合与分发 |
| [QwenImage-Layered：分解成 RGBA 图层](qwen-image-layered_zh.md) | 用 Qwen-Image-Layered 把图分解成 N 张透明图层（Layered Config / Layer Viewer） |
| [Ideogram 4：结构化/分区域提示词](ideogram4_zh.md) | Ideogram 4 文生图 + Prompt Builder 结构化/分区域提示词编辑器 |

### 参考文档

| 文档 | 说明 |
|------|------|
| [节点速查（逐节点参数）](node-reference_zh.md) | 每个 QuantFunc 节点的 INPUT_TYPES 每参数（类型/默认/含义）+ 交叉链接 |

### 安装

安装说明请参见 [README_zh.md](../README_zh.md)。
