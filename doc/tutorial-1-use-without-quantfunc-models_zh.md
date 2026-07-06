# 教程 1：运行时量化 —— 将 BF16/FP16 原模型量化为 4bit 加速推理

[English Version](tutorial-1-use-without-quantfunc-models.md)

## 概述

你**不需要**下载 QuantFunc 预量化模型也能使用本插件。**Lighting 后端**提供**运行时量化**能力 —— 在加载时将任意 **diffusers 格式**的 FP16 模型（例如 [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)）运行时量化并加速推理，无需预先转换。后端类型（lighting / svdq）由引擎从权重元数据**自动检测**，无需手动选择。

![文生图工作流全貌](../assets/t2i-overview.png)

> **示例工作流：** [`workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json)
> —— 文生图用其中「Sample for text to image」分组，图像编辑用「Sample for edit Image」分组。加载节点、Build Pipeline、Generate 的逐参数图文详解见 [模型加载与 API Key](model-loading-and-apikey_zh.md) 与 [基础文生图/img2img/ControlNet](basic-t2i-img2img-controlnet_zh.md)。

## 前置条件

1. 已安装 ComfyUI-QuantFunc 插件（参见 [README_zh.md](../README_zh.md)）
2. 已安装 CUDA 13.0+ 运行时及 cuDNN 9.x
3. 下载一个 diffusers 格式的模型到本地，例如：

```bash
# 使用 huggingface-cli 下载
huggingface-cli download Qwen/Qwen-Image-Edit-2511 --local-dir /path/to/Qwen-Image-Edit-2511

# 或使用 git lfs
git lfs install
git clone https://huggingface.co/Qwen/Qwen-Image-Edit-2511 /path/to/Qwen-Image-Edit-2511
```

> **什么是 diffusers 格式？** 目录下应包含 `model_index.json` 文件，以及 `transformer/`、`vae/`、`tokenizer/` 等子目录。

## 步骤

### 第一步：导入 Workflow

在 ComfyUI 中导入 `workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`，使用「Sample for text to image」分组。链路是：

```
QuantFunc Model Loader → QuantFunc Build Pipeline → QuantFunc Generate → Preview Image
```

> 加载节点只负责“指向权重文件”，输出 ComfyUI 原生的 `model`/`clip`/`vae`；**QuantFunc Build Pipeline** 才是组装并加载/量化管线的节点（承载 `device`、`precision_config` 等）。

### 第二步：配置 Model Loader（指向 FP16 原模型）

在 **QuantFunc Model Loader** 节点中：

| 参数 | 设置 |
|------|------|
| `model_dir` | 你的基础模型目录，例如 `/path/to/Qwen-Image-Edit-2511`（含 `model_index.json`） |
| `transformer_path` | **留空** —— Lighting 会从 `model_dir` 里的 FP16 权重运行时量化 |
| `prequant_weights`（可选） | 预量化调制权重路径，低显存 GPU 推荐（见下方说明） |

![配置 Model Loader 节点](../assets/node-model-loader.png)

> Model Loader 上**已不再有** `model_backend` / `device` / `precision_config` / `fused_mod` 参数：后端自动检测；`device` 与 `precision_config` 挪到了 **Build Pipeline**；调制融合由引擎按模型自动启用（无需手动）。

### 第三步：配置 Build Pipeline（device + precision_config）

把 Model Loader 的 `model`/`clip`/`vae` 接到 **QuantFunc Build Pipeline**：

| 参数 | 设置 |
|------|------|
| `device` | GPU 编号（通常为 `0`） |
| `precision_config` | 逐层精度：默认 `[auto-derive]`（引擎从模型自动推导）；也可选官方预设，或接 **QuantFunc Precision Config** 节点指定 JSON（见下方说明） |
| `pipeline_config`（可选） | 接 **QuantFunc Pipeline Config** 节点覆盖 attention 后端 / VAE 精度 / tiled VAE 等高级旋钮 |
| `api_key`（可选） | 部分受保护模型需要 |

![配置 Build Pipeline 节点](../assets/node-build-pipeline.png)

#### 关于 precision_config（逐层精度）

`precision_config` 定义 Transformer 每层的量化精度（INT4 / INT8 / FP8 等），在**速度和画质之间取平衡**。默认 `[auto-derive]` 即可让引擎自动推导；想用官方调优配置时，QuantFunc 为各系列提供了推荐配置：

> 示例下载地址（Qwen-Image-Edit 系列）：
> https://www.modelscope.cn/models/QuantFunc/Qwen-Image-Edit-Series/file/view/master/precision-config

```bash
# 示例：下载 Qwen-Image-Edit 系列的 precision config
modelscope download --model QuantFunc/Qwen-Image-Edit-Series --include "precision-config/*" --local_dir /path/to/configs
```

其他系列请在 [QuantFunc ModelScope 主页](https://www.modelscope.cn/models/QuantFunc) 找到对应仓库下的 `precision-config` 目录。下载后，用 **QuantFunc Precision Config (path)** 节点指向该 JSON 并接到 Build Pipeline 的 `precision_config`，或用 **Precision Config Auto Loader** 从下拉选择。

![Precision Config 节点](../assets/node-precision-config-path.png)

> **注意：** precision_config 与模型架构绑定，不同系列**不能混用**。

#### 低显存优化：prequant_weights（仅 QwenImage 系列）

QwenImage 系列在低显存机器上可用**预量化调制权重**替代运行时的调制融合：

| 你的显存 | 推荐 | 说明 |
|----------|------|------|
| **24 GB+**（RTX 4090 等） | 留空 `prequant_weights` | 引擎自动调制融合，画质更好，模型约 14 GB |
| **8–12 GB**（RTX 3060 等） | 设 `prequant_weights = 路径` | 模型约 11 GB，推理约 9 秒（无此优化需 20 秒+） |

从 [QuantFunc ModelScope](https://www.modelscope.cn/models/QuantFunc) 下载对应模型的 `mod_weights.safetensors`，在 Model Loader 的 `prequant_weights` 里填其路径即可。

### 第四步：配置生成参数

在 **QuantFunc Generate** 节点中：

| 参数 | 建议值 |
|------|--------|
| `prompt` | 你的文本提示词 |
| `width` / `height` | `1024` x `1024`（或模型支持的尺寸） |
| `steps` | `20`（完整模型），`4`（Lightning 蒸馏模型） |
| `guidance_scale` | `3.5`（蒸馏模型用 `0`~`1`） |
| `seed` | 任意数字 |

![配置 Generate 节点参数](../assets/node-generate.png)

> Generate 上采样相关的全部参数（22 种采样器、9 种调度器、CFG、FBCache）详见 [调度器与采样器手册](scheduler-and-sampler_zh.md)。

### 第五步：运行

点击 **Queue Prompt**。首次运行时 Lighting 引擎会对模型进行运行时量化（需要额外几十秒），后续运行会使用缓存加速。想彻底跳过每次的量化步骤，可用[教程 2](tutorial-2-export-quantized-models_zh.md) 把量化结果导出到磁盘。

## 可选：添加 LoRA

Lighting 后端支持**零成本 LoRA 叠加**。在 pipeline 和 Generate 之间插入 **QuantFunc LoRA Auto Loader**（或 **QuantFunc LoRA**）节点：

1. 选择 / 填写你的 LoRA `.safetensors`
2. 调整 `scale`（默认 `1.0`）
3. 可以串联多个 LoRA 节点

> Lighting 后端叠加 LoRA **不需要** QuantFunc LoRA Config 节点（那是 SVDQ 后端才需要的，见[教程 3](tutorial-3-download-quantfunc-models_zh.md)）。

![添加 LoRA 节点](../assets/node-lora-auto-loader.png)

## 可选：图像编辑模式

如果你使用的是图像编辑模型（如 Qwen-Image-Edit-2511），改用「Sample for edit Image」分组：

1. 用 **LoadImage** 节点加载参考图
2. 连接到 **QuantFunc Image List** 节点
3. 将 Image List 连接到 Generate 的 `ref_images` 输入
4. 在 prompt 中描述编辑内容（如 "把背景换成海滩"）

连接 `ref_images` 后，节点会自动切换为图像编辑模式。遮罩局部重绘 / 色彩匹配 / 尺寸对齐等逐参数详解见 [图像编辑图文指南](image-edit-mask-colormatch_zh.md)。

![图像编辑工作流](../assets/edit-overview.png)

## 可选：高级管线配置

如需调优，可添加 **QuantFunc Pipeline Config** 节点并接到 Build Pipeline 的 `pipeline_config`：

| 参数 | 说明 |
|------|------|
| `tiled_vae` | 生成高分辨率图像时开启（高分辨率也会自动开启） |
| `attention_backend` | 通常保持 `auto`（auto/sage/flash/sdpa） |
| `precision` | 计算精度 `bf16` / `fp16` |
| `text_precision` | 文本编码器精度，`int4` 最省显存 |
| `vae_precision` / `vision_quant` / `act_quant_mode` | VAE / 视觉编码器 / 激活量化的高级选项 |

> **显存/卸载策略由 libquantfunc 根据显卡自动决定，没有 `cpu_offload` / `layer_offload` 之类的手动开关**（旧工作流里若还留着这些键会被静默忽略）。大多数情况下无需此节点。

## 常见问题

**Q: 首次加载很慢？**
A: 正常现象。Lighting 首次运行需要进行运行时量化，后续运行会使用缓存的量化模型。你也可以用[教程 2](tutorial-2-export-quantized-models_zh.md) 将运行时量化的模型导出到磁盘，以后加载完全跳过量化步骤。

**Q: 哪些模型可以用？**
A: 任何 diffusers 格式的模型。目前支持的架构请查看 [QuantFunc 官方文档](https://www.modelscope.cn/models/QuantFunc)。

**Q: 和下载预量化模型比有什么区别？**
A: 已导出的量化模型（包括 SVDQ）加载更快，因为跳过了运行时量化步骤。推理速度上 SVDQ 与 Lighting 基本一致，在 RTX 50 以下的机器上 Lighting 没有 SVDQ 的 low-rank 计算开销，甚至快约 20%。详见[教程 3](tutorial-3-download-quantfunc-models_zh.md)。
