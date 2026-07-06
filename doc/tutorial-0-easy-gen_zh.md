# 新手入门必看：用 Model AutoLoader 生成你的第一张图

[English Version](tutorial-0-easy-gen.md)

## 概述

这是最简单的入门方式。**QuantFunc Model AutoLoader** 节点会**自动下载**并配置模型——你只需从下拉菜单中选择，无需手动下载模型或填写路径。

自动下载的最简出图链路是：

```
Model AutoLoader → Build Pipeline → Generate → Preview Image
```

> Model AutoLoader 输出 ComfyUI 原生的 `model` / `clip` / `vae` 三个接口，需要接 **QuantFunc Build Pipeline** 组装成管线，再接 **Generate** 出图（和手动加载法一样，只是省去了下载与填路径）。

![Model AutoLoader 节点](../assets/node-model-auto-loader.png)

> **示例工作流：** [`workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json) —— 其中「Load with base model or prequanted model」分组就是 Model AutoLoader，把它的 `model`/`clip`/`vae` 接到 Build Pipeline 即可自动下载出图。

> **想要更多控制？** 本教程是[教程 1（运行时量化）](tutorial-1-use-without-quantfunc-models_zh.md)的自动下载简化版。如果你需要手动指定本地模型、添加 LoRA、调整管线配置等高级操作，请参考教程 1，或图文指南 [模型加载与 API Key](model-loading-and-apikey_zh.md)。

## 前置条件

1. 已安装 ComfyUI-QuantFunc 插件（参见 [README](../README_zh.md)）
2. 已安装 CUDA 13.0+ 运行时及 cuDNN 9.x
3. 网络连接正常（首次使用需要自动下载模型）

## 步骤

### 第一步：导入工作流

在 ComfyUI 中导入 `workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`，找到「Load with base model or prequanted model」分组里的 **Model AutoLoader**。把它的 `model` / `clip` / `vae` 输出接到 **QuantFunc Build Pipeline**（再往下就是 Generate → Preview）。

### 第二步：配置 Model AutoLoader

在 **QuantFunc Model AutoLoader** 节点中：

| 参数 | 说明 |
|------|------|
| `model_series` | 选择模型系列（如 `QuantFunc/Z-Image-Series`）——必填 |
| `data_source` | 下载源：`modelscope`（国内推荐）或 `huggingface` |
| `transformer`（可选） | 选择具体的 Transformer 权重（格式 `系列/名称`）；保持 `None` 则使用该系列的默认 Transformer |

> **提示：** 基础模型的 GPU 变体（`50x-below` 适用 RTX 20/30/40、`50x-above` 适用 RTX 50）由节点根据你的显卡**自动选择**，无需手动指定。`transformer` 下拉列出的是该系列的全部权重变体，留 `None` 即可自动用默认权重；想指定某个变体再手动选。
>
> 后端类型（svdq / lighting）也由引擎从权重元数据**自动检测**，Model Loader / AutoLoader 上都没有 `model_backend` 参数。

### 第三步：配置生成参数

在 **QuantFunc Generate** 节点中：

| 参数 | 建议值 |
|------|--------|
| `prompt` | 你的文本提示词（如 "A cute cat"） |
| `width` / `height` | `1024` x `1024` |
| `steps` | `8`（Lightning 蒸馏模型）或 `20`（完整模型） |
| `seed` | 任意数字，或选择 `randomize` 自动随机 |
| `guidance_scale` | `0`（Lightning 蒸馏模型）或 `3.5`（完整模型） |

### 第四步：运行

点击 **Queue Prompt**。首次运行时插件会自动下载模型（取决于网速），后续运行直接使用缓存。

生成的图像会显示在 **Preview Image** 节点中。

## 下一步

- 想用自己的本地模型？→ [教程 1：运行时量化](tutorial-1-use-without-quantfunc-models_zh.md)
- 想导出量化模型加速加载？→ [教程 2：导出运行时量化模型](tutorial-2-export-quantized-models_zh.md)
- 想下载并使用已导出的量化模型？→ [教程 3：下载并使用已导出的量化模型](tutorial-3-download-quantfunc-models_zh.md)
- 各加载法/参数的图文详解 → [模型加载与 API Key](model-loading-and-apikey_zh.md)
