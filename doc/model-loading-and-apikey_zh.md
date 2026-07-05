# 模型加载与 API Key

本文讲清楚两件事：

1. **不同场景下有哪些加载模型的方式，它们各自适合谁、怎么连线**；
2. **API Key 是什么、有什么用、去哪里拿、在哪里填**。

> 本文对照插件当前代码（`nodes.py` / `nodes_format_adapters.py` / `model_auto_loader.py`）编写，节点名与参数以你 ComfyUI 里实际看到的为准。

---

## 一、加载模型的整体思路

QuantFunc 的加载分两步走，几乎所有工作流都是这个形状：

```
[加载节点：拿到 model / clip / vae]  →  QuantFunc Build Pipeline  →  QuantFunc Generate  →  Preview Image
```

- **加载节点**只负责“**指向权重文件**”，输出 ComfyUI 原生的 `MODEL` / `CLIP` / `VAE` 三个接口（和官方 UNETLoader / CLIPLoader / VAELoader 同类型），本身**不做真正的 torch 加载**（零加载，只记录路径）。
- **QuantFunc Build Pipeline** 才是真正“**组装成可运行管线**”的节点：它接收 `model`/`clip`/`vae`，加上 `device`（选哪张 GPU）、`precision_config`（逐层精度）、可选的 `pipeline_config`（高级旋钮）和 `api_key`，输出一个 `QUANTFUNC_PIPELINE`。
- 之后 **Generate** 拿这个 pipeline 出图。

> 综合示例工作流（同时演示三种加载方式）：
> [`workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json`](../workflow_sample/QuantFunc-Sample-WorkFlow-All-In-One.json)

![加载 → Build Pipeline → Generate 的整体连线](assets/model-loading-overview.png)
<!-- TODO-SCREENSHOT: 导入 QuantFunc-Sample-WorkFlow-All-In-One.json，截“文生图”那一组，展示 加载节点 → QuantFunc Build Pipeline → QuantFunc Generate → Preview Image 的完整连线 -->

---

## 二、三大类加载方式（按你的场景选一种）

| 场景 | 用哪些节点 | 说明 |
|------|-----------|------|
| **A. 我没有本地模型 / 想一键下载** | **QuantFunc Model Auto Loader** | 从下拉菜单选模型系列，自动判定 GPU 变体并从 ModelScope / HuggingFace 下载，无需填任何路径。 |
| **B. 我已有本地模型目录** | **QuantFunc Model Loader** | 指向本地 diffusers 目录（或预量化模型），可选单独指定 transformer / 预量化调制权重。 |
| **C. 我已有拆分好的组件文件 / 单文件全家桶** | **QuantFunc Pick Diffusion Model / Pick CLIP / Pick VAE**（分组件）或 **QuantFunc Pick Checkpoint**（全家桶单文件） | ComfyUI 原生风格的“按文件名挑选”，从 `models/` 下各目录扫描。 |

三种方式的输出都是 `MODEL`/`CLIP`/`VAE`，都接到同一个 **Build Pipeline**，可以按需混用。

### A. 一键自动下载 —— QuantFunc Model Auto Loader

最省事的方式，适合新手或没有本地模型的用户。

| 参数 | 说明 |
|------|------|
| `model_series` | 模型系列，可选：`QuantFunc/Qwen-Image-Edit-Series`、`QuantFunc/Qwen-Image-Series`、`QuantFunc/Z-Image-Series`、`QuantFunc/Klein-4B-Series`、`QuantFunc/Klein-9B-Series` |
| `data_source` | 下载源：`modelscope`（国内推荐）或 `huggingface` |
| `transformer`（可选） | 具体的 Transformer 权重变体，格式 `Series/name`。选 `None` 则用基础模型自带的默认 Transformer |

**GPU 变体是自动判定的**：节点内部根据你的 GPU 计算能力（SM）自动选择——

- `50x-above`：Blackwell（SM120+，如 RTX 50 系）——文本编码器用 FP4 基础模型；
- `50x-below`：其他所有卡（SM<120，含 RTX 20 / 30 / 40）——文本编码器用 INT4 基础模型。

> 你不需要手动选 50x-above / 50x-below，节点会替你选对；`transformer` 下拉里列出的变体也会按你的 GPU 过滤。首次使用会下载（取决于网速），之后走本地缓存。

![Model Auto Loader 参数](assets/model-loading-autoloader.png)
<!-- TODO-SCREENSHOT: QuantFunc-ControlNet.json 或 All-In-One 里的 QuantFunc Model Auto Loader 节点，展示 model_series / data_source / transformer 三个参数 -->

### B. 本地目录加载 —— QuantFunc Model Loader

适合你已经把 diffusers 格式模型下载到本地的情况。

| 参数 | 说明 |
|------|------|
| `model_dir` | 基础模型目录（内含 `model_index.json`，以及 `transformer/`、`vae/`、`text_encoder/` 等子目录） |
| `transformer_path` | Transformer 权重路径（`.safetensors` 文件或目录）。留空则用 `model_dir/transformer/` 下的第一个 `.safetensors` |
| `prequant_weights`（可选） | 预量化调制权重 `.safetensors` 路径（仅 Lighting 后端；低显存 GPU 推荐） |

> **什么是 diffusers 格式？** 目录里有 `model_index.json`，并带 `transformer/`、`vae/`、`text_encoder/` 等子目录。
> `model_dir` 内的组件文件（transformer / text_encoder / vae）由节点在标准 HF 目录布局里自动定位。

### C. 组件挑选 / 全家桶 —— Pick 系列（ComfyUI 原生风格）

当你已经有**拆分好的各组件文件**，或一个**单文件全家桶 checkpoint** 时使用。这些节点都是“零加载”——只记录路径，交给 Build Pipeline 真正加载。

| 节点（显示名） | 扫描目录 | 输出 |
|------|----------|------|
| **QuantFunc Pick Diffusion Model (zero-load)** | `models/diffusion_models/`、`models/checkpoints/`、`models/unet/` | `MODEL` |
| **QuantFunc Pick CLIP (zero-load)** | `models/text_encoders/`、`models/clip/`，以及 Model Auto Loader 下载的 `models/QuantFunc/<series>/<base>/text_encoder/`（BF16 官方文本编码器，中文提示词画质更好，以 `[QF]` 前缀标注） | `CLIP` |
| **QuantFunc Pick VAE (zero-load)** | `models/vae/` | `VAE` |
| **QuantFunc Pick Checkpoint (zero-load, bundled)** | `models/checkpoints/` | 一次性输出 `MODEL`/`CLIP`/`VAE`（单文件全家桶，三路同源，按 key 前缀切片） |

> 你也可以直接用 **ComfyUI 官方**的 UNETLoader / CLIPLoader / VAELoader / CheckpointLoaderSimple 喂给 Build Pipeline——它接收的就是标准 `MODEL`/`CLIP`/`VAE` 接口。Pick 系列的区别是“零加载”，更快、显存占用更小，且只服务 QuantFunc 分支。

![三种加载方式的分组](assets/model-loading-three-ways.png)
<!-- TODO-SCREENSHOT: All-In-One 工作流“加载模型”区域，把 A(Model Auto Loader) / B(Model Loader) / C(Pick* + Pick Checkpoint) 三组框在一起 -->

---

## 三、支持的模型格式（后端自动检测）

Build Pipeline 读一次 safetensors 头部（只读 JSON 元数据，不读权重，通常 < 1 MB），**自动判定**用哪个后端加载，你**不用手动选后端**：

| 检测到的格式 | 依据 | 走的后端 | 含义 |
|------|------|----------|------|
| **原精度 diffusers 基础模型** | 目录有 `model_index.json`，权重是 FP16/BF16 | **Lighting**（运行时量化） | 加载时把 FP16 权重量化为 4bit 加速，无需预量化模型。**必须配 precision_config**（见下节） |
| **Lighting 预量化模型** | 元数据 `method` 为 `lighting` / `lighting_precomputed` / `flux2klein_runtime`，或权重键含 `._qweight` | **Lighting** | QuantFunc Lighting 导出的预量化权重，加载即用，跳过运行时量化 |
| **SVDQ 预量化模型** | 元数据 `method=svdquant` / `model_class` 含 `Nunchaku`，或权重键含裸 `.qweight` + LoRA 旁挂 | **SVDQ** | Nunchaku SVDQ 预量化权重 |
| **全家桶单文件 checkpoint** | 单文件里同时含 `model.diffusion_model.` + `text_encoder(s).` + `vae.` 键前缀（≥2 类共存） | 自动识别为 bundled checkpoint | 一个文件打包 transformer + 文本编码器 + VAE，自动切片，无需 CheckpointLoader |

> 检测无法确定时安全回退到 **Lighting**（若文件缺失，引擎会给出清晰报错）。

---

## 四、precision_config（逐层精度表）

`precision_config` 是 **Build Pipeline** 上的一个下拉输入，定义 Transformer 每一层的量化精度，是画质/速度平衡的核心。它的取值有四类：

| 选项 | 含义 |
|------|------|
| **`[auto-derive]`（默认）** | 对**没有量化元数据的原精度 diffusers 基础模型 / 全家桶**，自动识别模型并按你选的 GPU 加载匹配的精度表（Blackwell SM120+ 用 FP4，其他用 INT4）；SVDQ / 已量化模型保留自身元数据里的逐层配置 |
| **`[none]`** | 从不注入精度表 |
| **`[builtin] xxx.json`** | 使用插件内置（`<plugin>/configs` 或 `$QUANTFUNC_CONFIGS_DIR`）的固定精度表 |
| **`[series] yyy.json`** | 使用 QuantFunc 模型系列预设（按需从 ModelScope 下载） |

也可以把这个下拉**转成输入接口**，从 **QuantFunc Precision Config Loader (path)** 或 **QuantFunc Precision Config Auto Loader** 用绝对路径喂进来。

> **重点：加载“原精度 diffusers 基础模型”时，精度表几乎是必需的**——默认的 `[auto-derive]` 会替你自动挑一份；如果你选了 `[none]` 又没有量化元数据，模型会停留在原精度（很慢、很占显存）。**预量化 / SVDQ / 全家桶模型自带精度信息，不需要额外配。**
>
> precision_config **与模型架构绑定**，不同系列不能混用。

![Build Pipeline 上的 precision_config](assets/model-loading-precision-config.png)
<!-- TODO-SCREENSHOT: QuantFunc Build Pipeline 节点，突出 precision_config 下拉（展开显示 [auto-derive]/[none]/[builtin]/[series]），以及 device / api_key 输入 -->

---

## 五、组件级自动加载节点（可选，进阶）

除了上面的整体加载，还有一组**只负责下载/定位单个组件、输出字符串路径**的节点，方便你精细拼装：

| 节点 | 输出 | 作用 |
|------|------|------|
| **QuantFunc Base Series Model Auto Loader** | `model_dir` | 从 QuantFunc 系列自动下载基础模型（自动 GPU 变体） |
| **QuantFunc Base Model Auto Loader** | `model_dir` | 从本地 `ComfyUI/models/diffusers/` 里挑一个带 `model_index.json` 的目录 |
| **QuantFunc Base Model Auto Loader with Download** | `model_dir` | 从上游仓库发现并下载基础模型到 `models/diffusers/` |
| **QuantFunc Transformer Auto Loader** | `transformer_path` | 从 `models/QuantFunc/transformer/` 挑 `.safetensors` |
| **QuantFunc Prequant Auto Loader** | `prequant_weights` | 自动下载预量化调制权重 |
| **QuantFunc Precision Config Auto Loader** | `precision_config` | 自动下载逐层精度表，接到 Build Pipeline 的 `precision_config` |

> 这些是可选的“便利件”。最简单的路子仍是 **Model Auto Loader → Build Pipeline**，不需要它们。

---

## 六、API Key —— 作用、获取、填写

### 6.1 API Key 是什么、有什么用

`api_key`（形如 `qf_xxxxxxxx`）是你的**授权/许可密钥**，用于：

- **许可校验**：引擎初始化时向许可服务器（`server_url`，默认 `https://service.quantfunc.com`）校验；
- **解密受保护模型**：加载 QuantFunc 受保护（key-protected）模型时，用它下载并解密该模型的 key map。

### 6.2 免费测试期间：开箱即用，无需配置

> **免费测试期间，所有客户端共享同一个测试 API Key。**

插件已经**内置了这个共享测试 key**，随插件一起分发，位置在紧挨着 `libquantfunc.so` 的配置文件里：

- Linux：`bin/linux/config.json`
- Windows：`bin/windows/config.json`

内容形如：

```json
{
    "server_url": "https://service.quantfunc.com",
    "api_key": "qf_xxxxxxxx...（内置测试 key）"
}
```

所以**在免费测试期间，你什么都不用填、也不用注册**，直接用即可。

### 6.3 正式发布后：到官网注册获取自己的 Key

> **正式发布后，请到我们官网 https://www.quantfunc.com 注册获取你自己的 API Key。**

拿到自己的 key 后，有两种填法（任选其一）：

1. **在节点里填**（推荐）：在 **QuantFunc Build Pipeline** 节点的 `api_key` 输入框里填入你的 `qf_xxx`。**节点里填的值会覆盖 config.json**。
2. **改配置文件**：编辑 `bin/<os>/config.json` 里的 `api_key`（以及需要时的 `server_url`）。当 Build Pipeline 的 `api_key` 留空时，就回退到这个文件里的值。

> 修改 `api_key` **不会**触发管线重建——插件内部对认证凭据做了热更新（`set_api_key`），所以切换 key 不需要重新加载模型。

![Build Pipeline 的 api_key 输入框](assets/model-loading-apikey.png)
<!-- TODO-SCREENSHOT: QuantFunc Build Pipeline 节点，突出可选的 api_key 输入框（展开 optional 输入），旁边可放一个 config.json 内容的示意 -->

---

## 七、常见问题

**Q：加载报错 “no transformer .safetensors found”？**
A：Model Loader 在 `model_dir/transformer/` 下找不到权重。确认 `model_dir` 是标准 diffusers 布局（含 `model_index.json` + `transformer/`），或显式填 `transformer_path`。

**Q：原精度 diffusers 基础模型出图很慢、很占显存？**
A：多半是没接精度表。Build Pipeline 的 `precision_config` 保持默认 `[auto-derive]`，或用 Precision Config (Auto) Loader 接一份匹配你模型系列的精度表。

**Q：Model Auto Loader 下拉是空的 / 只有 None？**
A：下拉是联网懒加载的。检查网络与 `data_source`（国内选 `modelscope`），首次会在后台刷新目录；保存的工作流里选过的值即便当前不在列表里也会在执行时解析下载，不会因“Value not in list”而失败。

**Q：需要自己填 API Key 吗？**
A：**免费测试期间不需要**——插件已内置共享测试 key。正式发布后到 https://www.quantfunc.com 注册，把 key 填到 Build Pipeline 的 `api_key` 或 `config.json`。

**Q：50x-below 和 50x-above 怎么选？**
A：不用手动选，Auto Loader 会按你的 GPU 自动判定（RTX 50 系 = 50x-above，其余 = 50x-below）。若你手动混用基础模型与 Transformer 权重，务必用**同一个变体**。
