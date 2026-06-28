# QuantFunc ComfyUI 工作流

[English](README.md)

## 1. 概述

本目录提供一个**全家桶示例工作流**（演示所有 QuantFunc 节点），以及针对新增生成模式的**单功能工作流**。导入与你场景匹配的那个即可。

| 工作流 | 说明 |
|--------|------|
| `QuantFunc-Sample-WorkFlow-All-In-One.json` | 一个综合工作流：**3 种加载权重方式** × **文生图 / 图像编辑 / 模型导出**。画布内的便签对每个分组做了说明。 |
| `QuantFunc-Ideogram4.json` | **Ideogram4** 文生图 —— *Model Loader* → *Build Pipeline* → *Generate*，配合 Ideogram-4 提示词构建节点写排版/文字提示词。 |
| `QuantFunc-QwenImage-Layered.json` | **分层（透明 RGBA）** 生成 —— *Layered Config* 设定图层数；*Layer Viewer* + *Image List* 预览拆解出的每个 RGBA 图层。 |
| `QuantFunc-ControlNet.json` | **ControlNet** 结构引导生成 —— *ControlNet Auto Loader* + *Control Image* 将条件输入接入 *Generate*。 |

> 建议先用全家桶熟悉节点连线，再用单功能工作流做日常生成。

## 2. 工作流内容

拖动画布到你需要的分组。

### 2.1 加载模型 —— 三种方式

| 方式 | 节点 | 适用场景 |
|------|------|----------|
| **现有 UNet / CLIP / VAE** | *Pick Diffusion Model* + *Pick CLIP* + *Pick VAE* → **Build Pipeline** | 你已有拆分好的各组件文件（ComfyUI 原生连线方式）。 |
| **基础模型 / 预量化模型** | *Model Loader* 或 *Model Auto Loader*（一键下载） | diffusers 基础模型目录，或预量化模型。若加载原精度 diffusers 基础模型，还需接 *Precision Config (Auto) Loader*。 |
| **全家桶 checkpoint** | *Pick Checkpoint* → **Build Pipeline** | 单文件打包的 checkpoint（AIO）。 |

> 加载**原精度 diffusers 基础模型**时**必须**接 **Precision Config (Auto) Loader** —— 它提供逐层精度表；预量化 / checkpoint 模型则不需要。

### 2.2 用途 —— 三个示例

- **文生图** —— pipeline → **Generate** → **Preview Image**。
- **图像编辑** —— **Load Image** → **Image List** → **Generate**（参考图 / 局部重绘编辑）。在 Image List 上接 `MASK` 即可局部重绘（白色重绘、黑色保留）。
- **导出** —— **Export** 节点。选 **checkpoint** 格式导出全家桶（含所有组件），或选 **diffusers** 只导出 vae / clip / transformer 其中一部分。

**LoRA：** 在任意 pipeline 上接 **LoRA Auto Loader**；串联多个即可叠加 LoRA（零成本合并）。

## 3. 节点说明（v2）

| 节点 | 作用 |
|------|------|
| **Pick Diffusion Model / Pick CLIP / Pick VAE / Pick Checkpoint** | 选择各组件文件（UNet / CLIP / VAE）或单文件全家桶 checkpoint。 |
| **Build Pipeline** | 把选好的组件组装成可运行的管线，支持逐组件精度 / 后端控制。 |
| **Model Loader / Model Auto Loader** | 直接加载基础模型或预量化模型目录（Auto Loader 支持一键下载 + 按模型系列过滤下拉）。 |
| **Precision Config Loader / Precision Config Auto Loader** | 逐层精度表 —— 加载**原精度 diffusers 基础模型时必填**。 |
| **LoRA Auto Loader** | 向管线追加 LoRA 适配器（可链式叠加多个）。 |
| **Generate** | 推理 —— 文生图，或接入 Image List 后做参考图编辑。 |
| **Image List** | 打包 1~N 张参考图及可选的局部重绘蒙版用于编辑。 |
| **Export** | 导出运行时量化模型（checkpoint = 全家桶，diffusers = 分组件）。 |
| **Layered Config** | 为 **QwenImage Layered** 透明 RGBA 分层生成设定图层数 / 选项。 |
| **Layer Viewer** | 预览分层生成拆解出的每个 RGBA 图层。 |
| **ControlNet Auto Loader** | 一键下载并加载 ControlNet 模型（QwenImage InstantX）。 |
| **Control Image** | 预处理 / 传入控制图（边缘、深度、姿态等）到管线。 |

## 4. 模型下载

预量化模型下载地址：
- **ModelScope**: https://www.modelscope.cn/models/QuantFunc
- **HuggingFace**: https://huggingface.co/QuantFunc

> 基础模型与 Transformer 权重必须使用**相同的 GPU 变体**（`50x-below` 适用 RTX 30/40，`50x-above` 适用 RTX 50）。
