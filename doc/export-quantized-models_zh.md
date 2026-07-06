# 导出量化模型（含 LoRA 融合）

Lighting 后端每次加载模型都会做**运行时量化**。**导出功能**把运行时量化后的模型持久化到磁盘，后续加载完全跳过重新量化；如果叠加了 LoRA，还能把 LoRA **永久融合**进导出的权重。导出后的模型：

- **跳过运行时量化**：加载即用，通常加载速度提升 2 倍以上；
- **LoRA 已融合**（可选）：无需每次运行时再挂 LoRA 节点；
- **可分享**：打包好直接发给他人使用。

![导出工作流全貌](../assets/export-overview.png)

> 本文对照插件当前代码（`nodes.py` 的 `QuantFuncExport`）逐参数核对，节点名与参数以你 ComfyUI 里实际看到的为准。

---

## 一、导出能做什么（使用场景）

| 场景 | 说明 |
|------|------|
| 跳过运行时量化 | 导出 Lighting 运行时量化的模型，避免每次启动都重新量化 |
| LoRA 融合 | 把你调好的 LoRA（含强度配置）永久融合进模型 |
| 模型分发 | 把配置好的模型打包分享给团队成员 |
| 多 LoRA 合并 | 把多个 LoRA 合并为单一模型，简化工作流 |

---

## 二、导出链路（先看这条线）

导出链路就是在正常加载管线后面接一个 **Export** 节点：

```
QuantFunc Model Loader → QuantFunc Build Pipeline →（可选 LoRA）→ QuantFunc Export
```

> **Export 节点接的是 Build Pipeline 输出的 `pipeline`**，不是直接接 Model Loader。示例工作流 [`workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json) 里的「Sample for exportation」分组就是这条链。加载节点 / Build Pipeline 的逐参数详解见 [模型加载与 API Key](model-loading-and-apikey_zh.md)。

---

## 三、两种导出来源（后端由引擎自动检测）

后端类型（lighting / svdq）由引擎从权重元数据**自动检测**，Model Loader 上没有 `model_backend` 参数——你只需**指向不同的权重**。

### A. 导出 Lighting 运行时量化模型（指向 FP16 原模型）

用 Lighting 运行时量化后觉得效果不错，就把量化结果导出。Model Loader 配置与 [模型加载与 API Key](model-loading-and-apikey_zh.md) B 小节一致：

| 参数 | 设置 |
|------|------|
| `model_dir` | 你的 FP16 基础模型目录，例如 `/path/to/Qwen-Image-Edit-2511` |
| `transformer_path` | **留空** —— Lighting 会从 FP16 运行时量化 |
| `prequant_weights`（可选） | 预量化调制权重路径，低显存 GPU 推荐 |

![配置 Model Loader 节点](../assets/node-model-loader.png)

> `device`、`precision_config` 在 **Build Pipeline** 上设置（详见 [模型加载与 API Key](model-loading-and-apikey_zh.md) 第四节）。调制层优化（高显存自动融合 / 低显存用 `prequant_weights`）见同文 B 小节，导出时的选择会写进模型元数据、加载时自动启用。

### B. 从现成 SVDQ 模型导出

已有现成的 SVDQ 量化模型、想把指定 LoRA 永久融合进去再导出：

| 参数 | 设置 |
|------|------|
| `model_dir` | QuantFunc 模型目录，例如 `/path/to/QuantFunc-Model` |
| `transformer_path` | SVDQ Transformer 权重路径，例如 `/path/to/QuantFunc-Model/transformer/model.safetensors`（也兼容旧版 nunchaku 权重） |

引擎会从这些权重**自动识别为 svdq 后端**。

> SVDQ 导出若叠加 LoRA，**必须**加 **QuantFunc LoRA Config** 节点（控制合并策略）。LoRA / LoRA Config 各参数（`merge_method` 的 auto/itc/awsvd/rop/concat、`max_rank`）见 [节点速查](node-reference_zh.md) §四。

---

## 四、添加 LoRA（可选，融合进导出权重）

在 **Build Pipeline** 和 **Export** 之间插入 **QuantFunc LoRA Auto Loader**（或 **QuantFunc LoRA**）节点：

```
Build Pipeline → LoRA (scale=0.8) → LoRA (scale=1.2) → Export
```

每个 LoRA 节点：选择 / 填写 LoRA `.safetensors`；`scale` = LoRA 强度（`0.0`–`2.0`，默认 `1.0`）。你在这里设置的强度会被**永久融合**到导出的模型中。

![添加 LoRA 节点](../assets/node-lora-auto-loader.png)

> **Lighting 后端**叠加 LoRA 直接串接即可，**不需要** LoRA Config 节点；**SVDQ 后端**必须再接一个 **LoRA Config**（见 [节点速查](node-reference_zh.md) §四）。

---

## 五、Export 节点参数

在 **QuantFunc Export** 节点中：

| 参数 | 说明 |
|------|------|
| `export_path` | 导出目录，例如 `/path/to/my-exported-model` |
| `export_format` | `diffusers`（HF 标准目录，每组件一个 safetensors，可当 `model_dir` 重新加载）或 `comfy_checkpoint`（单文件全家桶，可当 ComfyUI checkpoint 直接加载） |
| `export_mode` | `all` —— 导出完整模型（推荐，含 VAE、tokenizer 等）；`custom` —— 自定义选择组件（仅 `diffusers`；`comfy_checkpoint` 强制 `all`） |

`export_mode = custom` 时用下面三个开关选择组件：

| 参数 | 默认 | 说明 |
|------|------|------|
| `export_transformer` | `True` | 导出 Transformer（量化权重 + 融合的 LoRA） |
| `export_text_encoder` | `False` | 导出文本编码器 |
| `export_vision_encoder` | `False` | 导出视觉编码器 |

> **推荐 `export_mode = all`**，导出的模型完整独立，可直接作为 `model_dir` 使用。

![配置 Export 节点](../assets/node-export.png)

---

## 六、执行导出 + 产物结构

点击 **Queue Prompt**。导出过程会：

1. 加载基础模型；
2. 应用所有 LoRA（按配置的强度和合并策略）；
3. 执行运行时量化（Lighting 从 FP16 量化时）；
4. 把所有量化后的模型权重保存到指定目录。

`diffusers` 格式导出完成后，目录结构类似：

```
my-exported-model/
├── model_index.json
├── transformer/
│   └── *.safetensors    ← 量化权重（含融合的 LoRA）
├── vae/
├── tokenizer/
├── text_encoder/
└── scheduler/
```

（`comfy_checkpoint` 格式则是单个 `model.safetensors` 全家桶。）

---

## 七、加载导出的模型

导出的模型有两种加载方式（后端由引擎从导出权重自动识别，无需选择）：

**方式 A：作为完整模型**（推荐，适用于 `all` / `diffusers` 导出）

| 参数 | 设置 |
|------|------|
| `model_dir` | `/path/to/my-exported-model` |
| `transformer_path` | 留空或指向导出的 Transformer 权重 |

**方式 B：仅替换 Transformer 权重**

| 参数 | 设置 |
|------|------|
| `model_dir` | 原始基础模型目录 |
| `transformer_path` | `/path/to/my-exported-model/transformer/model.safetensors` |

> 使用导出模型时**不需要**再加之前的 LoRA 节点——LoRA 已经融合进去了。`comfy_checkpoint` 全家桶则用插件的 checkpoint 加载适配路径加载。

---

## 八、完整示例：从头到尾

假设你有：基础模型 `/models/FLUX.1-dev/`、风格 LoRA `/loras/anime-style.safetensors`（强度 0.8）、细节 LoRA `/loras/detail-enhancer.safetensors`（强度 1.2）。

**导出流程：**

```
Model Loader                     Build Pipeline        Export
  model_dir: /models/FLUX.1-dev/    device: 0            export_path: /models/my-anime-flux/
  transformer_path: (空)            precision_config:    export_format: diffusers
                                      [auto-derive]      export_mode: all
      ↓ model/clip/vae                   ↓ pipeline
  Build Pipeline ───────────────→ LoRA (anime-style, 0.8)
                                       ↓
                                  LoRA (detail-enhancer, 1.2)
                                       ↓
                                  Export
```

**使用导出模型：** Model Loader（`model_dir: /models/my-anime-flux/`、`transformer_path` 留空）→ Build Pipeline → Generate。无需 LoRA 节点，加载即用！

---

## 九、常见问题（FAQ）

**Q：导出的模型可以再叠加新 LoRA 吗？**
A：SVDQ 后端导出的模型可以继续叠加新 LoRA（需要 LoRA Config 节点）。但 **Lighting 后端导出的模型目前不支持再叠加新 LoRA**——要换 LoRA 组合，请从原始 FP16 模型重新导出。

**Q：导出需要多长时间？**
A：取决于模型大小和后端。Lighting 导出包含运行时量化的时间（几分钟）；SVDQ 导出更快，因为权重已经是量化的。

**Q：导出的模型体积有多大？**
A：INT4 量化后的 Transformer 权重通常只有 FP16 的 1/4 左右。完整模型（含 VAE、tokenizer）的总大小取决于各组件。

**Q：`custom` 导出模式什么时候用？**
A：只想更新 Transformer 权重（比如换了 LoRA 组合）、而 VAE / tokenizer 不变时，用 `custom` 只导出 Transformer 可以省时省空间。
