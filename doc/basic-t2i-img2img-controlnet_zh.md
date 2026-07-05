# 基础文生图、以图生图（img2img）与 ControlNet

本文覆盖三种最常用的生成方式，每种都给出可直接导入 ComfyUI 复现的**完整 workflow 例子**：

1. **基础文生图（t2i）** —— 从提示词生成图像；
2. **以图生图（img2img）** —— 从一张源图出发、保留构图按提示词改；
3. **ControlNet** —— 用边缘 / 深度 / 姿态等控制图引导构图。

> 本文对照插件当前代码（`nodes.py`）编写。模型加载部分见 [模型加载与 API Key](model-loading-and-apikey_zh.md)；采样器 / 调度器的完整枚举见 [调度器与采样器](scheduler-and-sampler_zh.md)（本文只讲基础项）。

---

## 一、基础文生图（t2i）

### 连线

```
[加载节点：model / clip / vae]  →  QuantFunc Build Pipeline  →  QuantFunc Generate  →  Preview Image
```

加载与 Build Pipeline 见文档《模型加载与 API Key》。核心生成节点是 **QuantFunc Generate**。

> **完整 workflow 例子：** [`workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json) 里的**“文生图”分组**（画布内便签有标注）。

![基础文生图连线](assets/basic-gen-t2i.png)
<!-- TODO-SCREENSHOT: 导入 QuantFunc-Sample-WorkFlow-All-In-One.json，截“文生图”分组：加载 → Build Pipeline → Generate → Preview Image 的完整连线 -->

### QuantFunc Generate —— 必填参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `pipeline` | QUANTFUNC_PIPELINE | — | 从 Build Pipeline（或 LoRA / ControlNet Loader）接入 |
| `prompt` | 文本（多行） | `A cute cat` | 正向提示词 |
| `width` / `height` | 整数 | `1024` / `1024` | 输出尺寸，范围 256–8192，步长 64 |
| `steps` | 整数 | `8` | 采样步数，范围 1–100 |
| `seed` | 整数 | `42` | 随机种子（设 `randomize` 每次随机） |
| `guidance_scale` | 浮点 | `0.0` | 引导强度，范围 0–30 |

### steps / guidance_scale 怎么设？

取决于你用的是**蒸馏 / 少步（lightning）模型**还是**完整（base）模型**：

| 模型类型 | `steps` | `guidance_scale` | `true_cfg_scale` |
|----------|---------|------------------|------------------|
| **蒸馏 / lightning（如 Z-Image-Turbo、Qwen-Lightning）** | `4`–`8` | `0`（蒸馏模型不吃 CFG） | 保持 `1.0`（关闭） |
| **完整 / base 模型** | `20`–`30` | `3.5` 左右 | 可升到 `4.0` 配合负向提示词 |

> 拿不准模型是不是蒸馏版？先用 `steps=8, guidance_scale=0` 试；如果图很糊 / 结构乱，说明是 base 模型，把 steps 提到 20+、guidance 提到 3.5。

### 常用可选参数（基础项）

| 参数 | 默认 | 说明 |
|------|------|------|
| `negative_prompt` | 空 | 负向提示词（配合 `true_cfg_scale > 1` 才起作用） |
| `true_cfg_scale` | `1.0` | 经典 CFG。`1.0` = 关闭（少步 / 蒸馏模型正确值）；base 模型可升到 `4.0`，需要负向提示词 |
| `sampler_name` | `euler` | 采样算法（共 23 种，详见文档《调度器与采样器》） |
| `scheduler` | `normal` | 噪声调度曲线（共 9 种，详见文档《调度器与采样器》） |
| `vram_budget` | `100%` | 限制本管线可用显存上限（按所选设备总显存百分比）；`100%`/`off` = 用满整卡 |
| `activate_unload` | `False` | 是否允许 ComfyUI 在需要显存时卸载本管线（默认 False = 权重常驻，后续更快） |

> `fbcache` / `fbcache_uncond`（首块缓存跳步加速）、`sampler_eta` 等进阶项见节点 tooltip 与《调度器与采样器》文档。

### 输出

Generate 有三个输出：`image`（图像）、`mask`（编辑 / inpaint 时的遮罩，纯 t2i 为空）、`latent_preview`（潜空间预览）。文生图只需把 `image` 接到 **Preview Image** / **Save Image**。

---

## 二、以图生图（img2img）

img2img 是**保留一张源图的构图 / 结构、按提示词改画**（不是局部重绘，也不是参考图编辑——那是[图像编辑](image-edit-mask-colormatch_zh.md)文档的内容）。

### 连线

```
Load Image  →  QuantFunc Image List (连 init_img)  →  QuantFunc Generate 的 ref_images
                                                          ↑ pipeline 仍来自 Build Pipeline
```

在 **QuantFunc Image List** 节点上：

1. 把 **Load Image** 的输出接到 Image List 的 **`init_img`** 输入（**不是** `main_image`）；
2. 调 **`init_img_strength`**（图生图强度）；
3. Image List 的 `images` 输出接到 **Generate** 的 `ref_images`。

连上 `init_img` 后，文生图就从这张源图出发：VAE 编码源图 → 按强度决定起始噪声 → 去噪。

### init_img_strength（图生图强度）

| 值 | 效果 |
|----|------|
| `0.2`–`0.4` | 强保留源图，只做微调 |
| `0.6`（默认） | 平衡（推荐） |
| `1.0` | 完全重画（等于纯文生图，忽略源图） |

> **`init_img` 与 `main_image` 互斥**：`init_img` 是 t2i 管线的“以图生图”（保留结构按 prompt 改）；`main_image` 是**编辑模式**（vision-edit，需要 Qwen-Image-Edit 这类编辑模型 + 可选遮罩）。Image List 只能二选一，编辑模式详见[图像编辑](image-edit-mask-colormatch_zh.md)文档。

> **完整 workflow 例子：** 以 [`QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json) 的“图像编辑”分组为基础，把源图接到 Image List 的 `init_img`（而非 `main_image`）即成 img2img。

![img2img 连线（init_img）](assets/basic-gen-img2img.png)
<!-- TODO-SCREENSHOT: All-In-One 里 Load Image → QuantFunc Image List（突出 init_img 与 init_img_strength 输入）→ Generate 的 ref_images 的连线 -->

---

## 三、ControlNet（结构引导，仅文生图）

ControlNet 用一张**控制图**（canny 边缘 / depth 深度 / pose 姿态 / soft_edge 软边缘）引导构图。需要**两个动作**：

1. **把 ControlNet 模型挂到管线** —— 用 **QuantFunc ControlNet Loader** 或 **QuantFunc ControlNet Auto Loader**（pipeline 进 → pipeline 出，和 LoRA Loader 同款流程，引擎只挂一个 ControlNet）；
2. **提供控制图 + 参数** —— 用 **QuantFunc Control Image** 节点，把它的 `control_image` 输出接到 **Generate** 的 `control_image` 输入。

> ⚠️ **ControlNet 仅对文生图（t2i）生效**，在编辑 / img2img 路径上 `control_image` 会被忽略。

### 连线

```
[加载] → Build Pipeline → QuantFunc ControlNet Auto Loader → QuantFunc Generate → Preview Image
                                                                     ↑ control_image
Load Image → QuantFunc Control Image ────────────────────────────────┘
```

> **完整 workflow 例子：** [`workflow_sample/QuantFunc-ControlNet.json`](../workflow_sample/QuantFunc-ControlNet.json)
> （示例用 `QuantFunc/Qwen-Image-Edit-Series` 模型 + `instantx_qwen_control.safetensors` ControlNet + canny 控制图。）

![ControlNet 连线](assets/basic-gen-controlnet.png)
<!-- TODO-SCREENSHOT: 导入 QuantFunc-ControlNet.json，截全图连线：ModelAutoLoader → BuildPipeline → ControlNet Auto Loader → Generate → Preview；LoadImage → Control Image → Generate.control_image -->

### QuantFunc ControlNet Loader / Auto Loader

| 参数 | 说明 |
|------|------|
| `pipeline` | 从 Build Pipeline 接入 |
| `controlnet_path`（Loader）/ `controlnet`（Auto Loader 下拉） | ControlNet `.safetensors` 路径，或 InstantX 模型目录；Auto Loader 从 `ComfyUI/models/controlnet/` 扫描 |
| `arch` | ControlNet 架构：`auto`（默认，自动识别）/ `instantx_qwen` / `pai-fun` / `zimage-fun` |

### QuantFunc Control Image

| 参数 | 默认 | 说明 |
|------|------|------|
| `image` | — | 控制图（canny / depth / pose 等，来自预处理器或 Load Image） |
| `control_type` | `canny` | 控制图类型：`canny` / `depth` / `pose` / `soft_edge`（union ControlNet 可接受多种，**必须与你喂进的图匹配**） |
| `control_scale` | `0.5` | 控制强度（0 = 关）。⚠️ **太高会把图洗白 / 出噪点**（少步 lighting 模型会放大这个效应） |
| `control_guidance_start` | `0.0` | 从调度的这个比例**开始**注入控制（0 = 第一步） |
| `control_guidance_end` | `1.0` | 到这个比例**停止**注入（1 = 最后一步；调低如 0.7 可减少后期噪点） |

**`control_scale` 起始推荐值**（先从低值起，按需逐步加强）：

| ControlNet | 推荐起始 `control_scale` |
|------------|------------------------|
| PAI Fun-Control（`pai-fun`） | `0.2`–`0.4`（设 1.0 会在 lighting 模型上变纯噪点） |
| InstantX / ZImage（`instantx_qwen` / `zimage-fun`） | `0.5`–`0.9` |

> 输出被洗白 / 发噪，先**降低 `control_scale`**（这是最常见原因），再考虑降低 `control_guidance_end`。

---

## 四、常见问题

**Q：文生图出来是糊的 / 结构乱？**
A：多半是把 base 模型当蒸馏模型跑了。把 `steps` 提到 20–30、`guidance_scale` 提到 3.5 试试。反之若你用的是 lightning 蒸馏模型，`guidance_scale` 必须是 `0`。

**Q：img2img 改得太多 / 太少？**
A：调 Image List 的 `init_img_strength`：太多改动 → 调低（0.2–0.4 强保留源图）；几乎没变 → 调高（越接近 1.0 越像纯文生图）。

**Q：接了 ControlNet 没效果 / 报错？**
A：确认两件事都做了——（1）用 ControlNet (Auto) Loader 把模型挂进了管线；（2）用 Control Image 提供了控制图并接到 Generate 的 `control_image`。还要确认走的是**文生图**路径（连了 `ref_images` / `init_img` 的编辑 / img2img 路径会忽略 ControlNet）。

**Q：ControlNet 出图洗白 / 噪点重？**
A：先把 `control_scale` 降下来（PAI Fun-Control 约 0.2–0.4，InstantX/ZImage 约 0.5–0.9），必要时把 `control_guidance_end` 从 1.0 降到 0.7 左右。

**Q：`control_type` 选哪个？**
A：与你实际喂进 Control Image 的那张图匹配——喂 canny 边缘图就选 `canny`，喂深度图就选 `depth`，以此类推。
