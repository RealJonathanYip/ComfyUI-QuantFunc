# Ideogram 4：结构化 / 分区域提示词文生图

**这篇解决什么场景：** 用 **Ideogram 4** 模型文生图。它的特色是配套一个 **Ideogram 4 Prompt Builder** 节点 —— 用**结构化字段 + 画布上的分区域（bbox）编辑器**来写提示词（背景、风格、每个区域画什么），比一行纯文本更可控。

> 本文对照插件当前代码（`nodes.py` 的 `Ideogram4PromptBuilderKJ`，与仓库 `origin/main` 一致）核对。
> **完整 workflow 例子：** [`workflow_sample/QuantFunc-Ideogram4.json`](../workflow_sample/QuantFunc-Ideogram4.json)

---

## 一、整体连线

比普通文生图多一个 **Ideogram 4 Prompt Builder KJ** 节点，它产出结构化 `prompt` 喂给 Generate。

```
[加载 Ideogram-4 模型] → Build Pipeline → QuantFunc Generate → Preview Image
                                              ↑ prompt / width / height
                    Ideogram 4 Prompt Builder KJ ┘
```

![Ideogram4 整体连线：Model Loader → Build Pipeline → Generate；Ideogram 4 Prompt Builder → Generate 的 prompt / width / height](../assets/ideogram4-overview.png)

- **加载**：用 Ideogram-4 系模型（示例是 `ideogram-4-fp8`，用 Model Loader 指目录，见 [模型加载](model-loading-and-apikey_zh.md)）。
- **Prompt Builder**：产出 `prompt`（结构化 JSON 字符串）+ `width` / `height`，直接接到 Generate 的对应输入。
- **Generate**：Ideogram 4 是**完整 CFG 模型**（非蒸馏），示例用 `steps=12`、`guidance_scale=7.0`、`true_cfg_scale=4.0`（对比蒸馏模型的 8 步 / 0 / 1.0）。CFG / 采样细节见 [调度器与采样器](scheduler-and-sampler_zh.md)。

> **提示**：Ideogram 4 跑**串行 true-CFG**（cond + uncond 两趟），FBCache 提速时负向分支有独立的 `fbcache_uncond` 阈值（见调度器手册 §五）。

---

## 二、Ideogram 4 Prompt Builder KJ —— 结构化提示词节点

![Ideogram 4 Prompt Builder KJ 节点](../assets/node-ideogram4-prompt.png)

它把提示词拆成几类字段 + 一个可视化区域编辑器（节点下方的 “Ideogram 4 editor” 面板），最终拼成一段 Ideogram 4 期望格式的 `prompt` JSON。

| 参数 | 类型 | 默认 | 含义 |
|------|------|------|------|
| `width` / `height` | INT | `1024` / `1024` | 画布长宽（也是 bbox 度量的像素网格）。**Ideogram 4 需要 16 的倍数** |
| `high_level_description` | STRING | 空 | 整图一句话总览（留空则省略） |
| `background` | STRING | 空 | **场景背景描述（必填）** |
| `style` | 下拉 | — | 风格预设下拉（动态） |
| `aesthetics` / `lighting` / `medium` | STRING | 空 | 风格描述子（留空则省略）：美学 / 光照 / 媒介 |
| `import_mode` | 下拉 | `when empty` | 接了 `import_json` 时怎么用它：`when empty` = 仅在编辑器还没有区域时用它做种子（之后编辑器优先，方便你继续改）；`always` = 总是用导入的 |
| `output_format` | — | `compact` | 输出 JSON 格式：`compact`（默认，Ideogram 4 期望的）/ `pretty`（缩进，易读） |
| `image`（可选） | IMAGE | — | 作为编辑器背景 / 预览背景的参考图 |
| `import_json`（可选） | STRING | 空 | 完整 caption JSON，连上按 `import_mode` 载入编辑器 |
| `bboxes`（可选） | BOUNDING_BOX | — | 像素空间的框（`{x,y,width,height}`），编辑器还没有区域时用它做种子 |

> `style_palette_data` / `elements_data` / `bg_brightness` 这些字段由**节点内的编辑器 UI 管理**（在节点下方的 “Ideogram 4 editor” 面板里操作：加区域、画框、调背景亮度、选风格色板），你一般不直接填。

**输出**：`prompt`（结构化提示词 JSON）、`preview`（预览图）、`bboxes`、`width`、`height` —— 把 `prompt` / `width` / `height` 接到 Generate 即可。

---

## 三、怎么用（最简流程）

1. **Model Loader** 指向 Ideogram-4 模型目录 → **Build Pipeline**；
2. 加 **Ideogram 4 Prompt Builder KJ**：填 `background`（必填），按需填 `high_level_description` / `aesthetics` / `lighting` / `medium`，在下方编辑器里加分区域（可选）；
3. Prompt Builder 的 `prompt` / `width` / `height` → **Generate** 的对应输入；
4. Generate 设 `steps≈12`、`guidance_scale≈7.0`、`true_cfg_scale≈4.0`（Ideogram 4 是完整 CFG 模型）；
5. `image` → Preview / Save。

---

## 四、常见坑 / FAQ

| 现象 | 原因 | 处理 |
|------|------|------|
| 报错 / 尺寸异常 | `width`/`height` 不是 16 的倍数 | 改成 16 的倍数 |
| 输出很糊 / 结构乱 | 当成蒸馏模型跑了（步数 / CFG 太低） | Ideogram 4 是**完整 CFG 模型**：`steps≈12`、`guidance_scale≈7.0`、`true_cfg_scale≈4.0` |
| 提示词没生效 / 结构没体现 | 没把 Prompt Builder 的 `prompt` 接到 Generate | 连 `prompt`（以及 `width`/`height`）到 Generate |
| 分区域没体现 | 编辑器里没加区域，或 `import_mode` 覆盖了编辑 | 在 “Ideogram 4 editor” 面板加区域；`import_json` 只想做种子时用 `when empty` |
| 显存 / 速度 | 完整 CFG 跑两趟 | 提速可试 `fbcache` + `fbcache_uncond`（见调度器手册） |
