# 图像编辑：参考图编辑 · 局部重绘 · 色彩匹配 · 尺寸对齐

**这篇解决什么场景：** 用一张（或多张）图 + 提示词**改图** —— 整图编辑、**局部重绘（inpaint 遮罩）**、**色彩匹配**，以及**输出尺寸 / 像素级对齐**。像素级对齐、输出与原图同尺寸同比例的关键是把尺寸模式设成 **`origin`**（本文重点讲）。

> 本文对照插件当前代码（`nodes.py`）逐字段核对。纯文生图 / 以图生图（img2img）见 [基础文生图、img2img 与 ControlNet](basic-t2i-img2img-controlnet_zh.md)；模型加载见 [模型加载与 API Key](model-loading-and-apikey_zh.md)。

---

## 零、前置：编辑模式怎么触发、需要什么模型

- **触发方式**：把 **QuantFunc Image List** 的输出接到 **QuantFunc Generate** 的 `ref_images` 输入 —— 只要 `ref_images` 连上，Generate 就自动进入**编辑模式**（不再是纯文生图）。
- **需要编辑能力的模型**：参考图编辑走 vision-edit 路径，需要**编辑模型**（如 **Qwen-Image-Edit 系**，带 vision encoder）。普通 t2i 模型不支持参考图编辑。
- **核心连线**：

```
[加载编辑模型] → Build Pipeline → QuantFunc Generate → Preview Image
                                        ↑ ref_images
Load Image → QuantFunc Image List ──────┘
             （main_image / mask / 尺寸 / color_match 都在这里配）
```

> **完整 workflow 例子：** [`QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json) 里标题为 **`Sample for edit Image`** 的分组（示例用 `QuantFunc/Qwen-Image-Edit-Series` 模型 + Load Image → Image List → Generate）。

![图像编辑整体连线](../assets/edit-overview.png)
<!-- TODO-SCREENSHOT: 导入 QuantFunc-Sample-WorkFlow-All-In-One.json，截 `Sample for edit Image` 分组：加载(Qwen-Image-Edit-Series)→Build Pipeline→Generate；Load Image→Image List→Generate.ref_images 的完整连线 -->

---

## 一、QuantFunc Image List —— 编辑的中枢节点

编辑相关的输入几乎都在这个节点上。它把主图、参考图、遮罩、尺寸、色彩匹配打包成一个 `images` 输出，接到 Generate 的 `ref_images`。

| 参数 | 类型 | 默认 | 取值 | 含义与何时改 |
|------|------|------|------|--------------|
| `main_image` | IMAGE | — | — | **编辑主图**（被编辑 / 被遮罩区域重绘的那张）。整图编辑与局部重绘都用它。**与 `init_img` 互斥**（`init_img` 是 img2img，见文档2；两者只能连一个，都连时 `init_img` 优先） |
| `ref_image_2` … `ref_image_10` | IMAGE | 无 | 最多 9 张 | **附加参考图**（第 2–10 张，可选）。用于多图参考编辑（如“把 A 的物体放到 B 的场景”）。不需要就不连 |
| `main_image_mask` | MASK | 无 | — | **局部重绘遮罩**（连上即触发 inpaint）。白=重绘、黑=保留。见 §三 |
| `mask_config` | QUANTFUNC_MASK_CONFIG | 默认值 | 来自 Mask Config 节点 | 遮罩高级参数（strength / grow / blur / no_snap）。不连=全用默认。见 §三 |
| `main_image_resize` | 下拉 | `720` | `720` / `1024` / `origin` | **主图缩放模式**。决定输出尺寸与像素对齐，见 §二（**要像素级对齐 / 输出同原图尺寸用 `origin`**） |
| `ref_image_resize_others` | 下拉 | `720` | `720` / `1024` / `origin` | 第 2–10 张参考图的缩放模式，见 §二 |
| `color_match` | FLOAT | `0.0` | 0–1，步长 0.05 | 输出色彩向主参考图匹配的强度，见 §四 |
| `init_img` | IMAGE | 无 | — | **img2img 源图**（属文档2，不是编辑）。与 `main_image` 互斥 |
| `init_img_strength` | FLOAT | `0.6` | 0–1 | 仅 `init_img` 连接时生效（见文档2） |

> **必须连主图或源图之一**：Image List 要求 `main_image`（编辑）或 `init_img`（img2img）**至少连一个**，都不连会报错 `connect either main_image (edit) or init_img (t2i img2img)`。

---

## 二、尺寸与像素级对齐（`main_image_resize` / `ref_image_resize_others`）

这是编辑里最容易踩的点。两个尺寸下拉都有三档：

| 取值 | 行为 | 输出尺寸 | 适用场景 |
|------|------|----------|----------|
| **`720`（默认）** | 把图的**长边裁到 720 px**（按比例缩放） | 长边 720 的新尺寸 | 快速预览、省显存 |
| **`1024`** | 把图的**长边裁到 1024 px** | 长边 1024 的新尺寸 | 更高质量，稍慢、稍占显存 |
| **`origin`** | **保留原始尺寸**，只把每条边**对齐到 16 的倍数** | ≈ 原图尺寸（每边至多微调 15 px），**保持原比例** | **像素级对齐**；要求输出与原图**同尺寸同比例** |

### 为什么像素级对齐要用 `origin`

- `720` / `1024` 会**改变图像尺寸**（把长边缩放到固定值），输出图的分辨率和原图不一致，做不到与原图逐像素对齐。
- **`origin` 保留原始分辨率**（仅对齐到 16 的倍数，这是 VAE 下采样的硬性要求），所以编码用的就是原图的几何，输出会**与原图保持相同比例、相同尺寸**，可以和原图逐像素叠合。
- 因此：**做局部重绘、或需要输出和原图对齐 / 同尺寸时，把 `main_image_resize` 设成 `origin`**（如果参考图也要对齐，`ref_image_resize_others` 一并设 `origin`）。

> **代价**：`origin` 编码的是全分辨率图，显存和耗时高于 `720`。大图 + `origin` 可能 OOM —— 显存紧张时先用 `720` 出草稿、确认构图后再切 `origin` 出终稿；或配合管线的 `vram_budget` / VAE tiling。

![尺寸模式对比：720 vs origin](../assets/edit-resize-origin.png)
<!-- TODO-SCREENSHOT: QuantFunc Image List 节点，突出 main_image_resize 下拉展开（720/1024/origin），可并排放一张 720 输出与一张 origin 输出对比说明尺寸差异 -->

---

## 三、局部重绘（inpaint 遮罩）

**只重绘一块区域、其余保留原图**，用遮罩实现。

### 遮罩语义

- **白色（1）= 重绘**，**黑色（0）= 保留**（等价 ComfyUI 的 `SetLatentNoiseMask`）。
- 只有白色区域被模型重绘；黑色区域在最后一步通过 snap **完整贴回原图**（除非关掉 snap，见下）。

### 连线

```
Load Image (或 LoadImageMask)  →  main_image_mask  ┐
Load Image (原图)              →  main_image       ┼─→ QuantFunc Image List → Generate.ref_images
QuantFunc Mask Config          →  mask_config      ┘（可选，不连走默认）
```

- 遮罩会**自动对齐到主图的像素尺寸**（节点内部按主图 H×W 做 bilinear 缩放），所以你给主图接任意缩放节点时，不必单独给遮罩再配一个缩放。
- 但如果你的遮罩来自一个和主图**不同分辨率**的来源、又想精确控制缩放方式，用 **QuantFunc Mask Scale By**（见 §三.3）。

### 三.1 QuantFunc Mask Config —— 遮罩高级四参数

不连这个节点时，Image List 用下面这组**默认值**（与 ComfyUI 的 `VAEEncodeForInpaint` / `SetLatentNoiseMask` 一致）：

| 参数 | 类型 | 默认 | 取值 | 含义 / 效果 / 何时改 |
|------|------|------|------|----------------------|
| `mask_strength` | FLOAT | `1.0` | 0–1，步长 0.05 | 遮罩强度倍数。`1.0` = 完全重绘遮罩区；**`0` = 关闭 inpaint**（等于没接遮罩）。想让重绘区“轻改”而非“重画”时调低（如 0.5–0.8） |
| `mask_grow` | INT | `6` | 0–64 | 遮罩**像素膨胀** N（对齐 ComfyUI `grow_mask_by`）。向外扩大白色区，让重绘区与保留区**接缝过渡更自然**。接缝生硬 / 有硬边时调大；想精确只改遮罩内像素时调 0 |
| `mask_blur` | FLOAT | `0.0` | 0–64，步长 0.5 | 遮罩**高斯模糊** sigma（像素，对齐 ComfyUI `MaskBlur`）。`0` = 边界硬切；调大让边界**柔和羽化**，与 `mask_grow` 配合能进一步消接缝 |
| `mask_no_snap` | BOOLEAN | `False` | 开 / 关 | 是否**关闭**最后一步“把未遮罩区 snap 回原图”。默认关（= snap 开，保留区严格不变，跟 ComfyUI 一致）。**开启** = 让模型决定整张图，边界过渡最柔和，但**保留区会轻微飘移**（不再逐像素等于原图）。要严格保留区就别开 |

**相互作用**：`mask_grow` + `mask_blur` 一起用来消接缝（先膨胀再羽化）；`mask_no_snap` 与它们正交 —— 开了 no_snap 追求柔和过渡，就会牺牲保留区的严格一致性。`mask_strength=0` 会让整个 inpaint 失效（引擎侧 `inpaint_strength>0` 才走遮罩路径）。

![Mask Config 四参数](../assets/edit-mask-config.png)
<!-- TODO-SCREENSHOT: QuantFunc Mask Config 节点，展示 mask_strength / mask_grow / mask_blur / mask_no_snap 四个参数标签与默认值 -->

### 三.2 完整局部重绘步骤

1. 加载**编辑模型**（Qwen-Image-Edit 系）→ Build Pipeline；
2. **Load Image** 加载原图 → Image List 的 `main_image`；
3. 用 **Load Image（含 alpha）/ LoadImageMask** 得到遮罩 → Image List 的 `main_image_mask`（白=要改的区域）；
4. （可选）**QuantFunc Mask Config** 调 grow / blur 消接缝 → Image List 的 `mask_config`；
5. **`main_image_resize` 设 `origin`**（局部重绘几乎总是要像素对齐，避免重绘区错位）；
6. Image List → Generate 的 `ref_images`；`prompt` 写你要在遮罩区生成的内容；
7. 出图，把 Generate 的 `image` 接 Preview / Save。

### 三.3 QuantFunc Mask Scale By —— 遮罩同比缩放

镜像 ComfyUI 的 `ImageScaleBy`（官方那个不接 MASK 类型）。**当你用 `ImageScaleBy` 缩放了主图、需要把遮罩缩放到同样比例**时用它。

| 参数 | 类型 | 默认 | 取值 | 说明 |
|------|------|------|------|------|
| `mask` | MASK | — | — | 要缩放的遮罩 |
| `scale_by` | FLOAT | `1.0` | 0.01–8.0，步长 0.01 | 缩放比例。**要和主图 `ImageScaleBy` 的值设成一样** |
| `method` | 下拉 | `bilinear` | `bilinear` / `nearest` / `bicubic` | 插值方式。`bilinear` 平滑（默认）；`nearest` 保留硬边（适合硬边遮罩）；`bicubic` 质量最高但稍慢 |

> 用法：`LoadImageMask → QuantFunc Mask Scale By（scale_by 同主图）→ main_image_mask`。

---

## 四、色彩匹配（`color_match`）

编辑后有时输出色调会偏离原图，`color_match` 把输出的色彩分布**向主参考图匹配**（插件侧解码后做 Reinhard、Lab 空间后处理）。

| 值 | 效果 |
|----|------|
| `0.0`（默认） | 不校正 —— 最锐利，但可能与原图有色偏 |
| `0.3`–`0.5` | 平衡（**推荐**）—— 色彩更接近原图，细节基本保留 |
| `1.0` | 完全匹配 —— 色彩最忠实，细节略软 |

**相互作用 / 限制**：
- 仅**单图编辑**（输出 1 张）时生效；**多图层输出**（如 QwenImage Layered，N>1）**跳过** color_match（每层没有唯一参考）。
- `init_img`（img2img）路径里 `color_match` 固定为 `0`（不适用）。
- 它是**解码后的后处理**，不影响生成本身，只调最终色彩。

---

## 五、多张参考图

`ref_image_2` … `ref_image_10` 提供最多 9 张附加参考图（加主图共 10 张）。

- 主图（`main_image`）用 `main_image_resize` 的尺寸；参考图 2–10 用 `ref_image_resize_others` 的尺寸（可各自不同）。
- 多参考图显存开销更大，`ref_image_resize_others` 保持 `720`（默认）更省显存；设 `origin` 会按各图原尺寸编码，图大可能 OOM。

---

## 六、常见坑 / FAQ / 错误对照

| 现象 / 错误 | 原因 | 处理 |
|-------------|------|------|
| 报错 `connect either main_image (edit) or init_img` | Image List 主图和源图都没连 | 连 `main_image`（编辑）或 `init_img`（img2img）之一 |
| 编辑输出和原图**尺寸 / 位置对不上** | `main_image_resize` 是 `720`/`1024`（改了尺寸） | 改成 **`origin`**（保留原尺寸、像素对齐） |
| 局部重绘**重绘区错位 / 只改了一部分** | 遮罩没对齐，或尺寸模式不是 origin | 用 `origin`；确认遮罩白=要改的区域；遮罩会自动对齐主图尺寸 |
| 重绘区**接缝生硬 / 有硬边** | grow / blur 太小 | 调大 Mask Config 的 `mask_grow`（如 8–16）+ `mask_blur`（如 4–8） |
| 保留区（黑色）**也变了** | 开了 `mask_no_snap`，或 `mask_strength` 过强漫出 | 关掉 `mask_no_snap`（默认就是关）；必要时缩小遮罩白色区 |
| 参考图编辑**完全没生效 / 报模型不支持** | 用了非编辑模型（纯 t2i） | 换 Qwen-Image-Edit 系等**带 vision encoder 的编辑模型** |
| 大图 + `origin` **OOM** | 全分辨率编码显存不够 | 先 `720` 出草稿，或降分辨率 / 设 `vram_budget` / 开 tiled VAE |
| 输出**色调偏** | 未做色彩匹配 | 把 `color_match` 设 0.3–0.5 |
