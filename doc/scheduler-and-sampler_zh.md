# 调度器与采样器（scheduler / sampler）完整参数手册

**这篇解决什么：** 把 **QuantFunc Generate** 上和采样有关的参数一次讲清 —— **采样器**（`sampler_name`，22 种）、**调度器**（`scheduler`，9 种）、以及配套的 `sampler_eta` / `sampler_s_noise` / 各 `*_order`、CFG（`true_cfg_scale` / `guidance_scale`）、FBCache 加速（`fbcache` / `fbcache_uncond`）。

> 本文的参数值 / 默认 / 枚举一律对插件当前代码（`nodes.py` 的 `QuantFuncGenerate` INPUT_TYPES，与仓库 `origin/main` 一致）核对。基础用法见 [基础文生图、img2img 与 ControlNet](basic-t2i-img2img-controlnet_zh.md)。

下图是 **QuantFunc Generate** 节点，本文讲的参数都在它上面（`sampler_name` / `scheduler` / `sampler_eta` / `sampler_s_noise` / `sampler_*_order` / `true_cfg_scale` / `fbcache` 等）：

![QuantFunc Generate 节点：采样器 / 调度器 / CFG / FBCache 参数](../assets/node-generate.png)

---

## 零、先分清：采样器 ≠ 调度器（两个正交的旋钮）

- **采样器（sampler）** = **怎么积分每一步**（用什么数值方法从当前 latent 走到下一步）。
- **调度器（scheduler）** = **sigma 噪声曲线长什么样**（各步的噪声强度怎么分布）。

两者**正交**：任意采样器都能配任意调度器。名字与 ComfyUI 的对应下拉**完全一致**。

> ⚠️ **蒸馏 / 少步模型**（如 Klein 这种走 explicit-sigma 的）会**忽略 `scheduler` 的类型** —— 调度器只对走 `set_timesteps(mu)` 路径的**基础模型**（QwenImage / ZImage）起塑形作用。

---

## 一、采样器 `sampler_name`（22 种，默认 `euler`）

按“是否引入随机噪声”分三类。**选择要点在每类开头。**

### 1.1 确定性采样器（`ddim` 是例外，见下；其余不受 `sampler_eta` / `sampler_s_noise` 影响）

> **选择要点**：追求**可复现、稳定**用这一类。少步蒸馏模型首选 `euler`；要更高质量且步数够（20+）用 `dpmpp_2m`。

| 采样器 | 阶数 / 说明 |
|--------|-----------|
| **`euler`（默认）** | 1 阶，最快。少步 / 蒸馏模型首选 |
| `heun` | 2 阶，约 2× 慢（每步 2 次模型评估） |
| `heunpp2` | Heun++（最高到 3 阶） |
| **`dpmpp_2m`** | 2 阶多步，**质量推荐**（base 模型 20+ 步） |
| `dpm_2` | DPM-Solver 2 阶（2 NFE/步） |
| `lms` | 线性多步（用 `sampler_solver_order` 1–4 控制阶数） |
| `ipndm` / `ipndm_v` | Adams 隐式多步 |
| `res_multistep` | RES 2 阶多步 |
| `gradient_estimation` | Euler + 梯度动量 |
| `ddim` | DDIM。`eta=0` 确定性；**`eta>0` 时变随机**（注入 ancestral 噪声，只受 `sampler_eta` 影响；引擎 `DDIMSampler` 不吃 `sampler_s_noise`） |

### 1.2 随机 / 祖先采样器（用 `sampler_eta` + `sampler_s_noise`）

> **选择要点**：想要**更多细节 / 多样性**（每步注入噪声）用这一类，但可复现性下降。`lcm` 专给 LCM 蒸馏 / lightning 模型。

| 采样器 | 说明 |
|--------|------|
| `euler_ancestral` | Euler 祖先采样 |
| `dpm_2_ancestral` | DPM-2 祖先 |
| `dpmpp_2s_ancestral` | DPM++(2S) 祖先 |
| `dpmpp_sde` | DPM++(SDE) 随机 |
| `dpmpp_2m_sde` | DPM++(2M) SDE |
| `dpmpp_2m_sde_heun` | 上者 + Heun 校正 |
| `dpmpp_3m_sde` | DPM++(3M) SDE |
| `res_multistep_ancestral` | RES 2 阶 + 噪声 |
| **`lcm`** | LCM，**给蒸馏 / lightning 模型** |

### 1.3 SA-Solver（用 `sampler_eta` / `sampler_s_noise` / `sampler_predictor_order` / `sampler_corrector_order`）

| 采样器 | 说明 |
|--------|------|
| `sa_solver` | 随机 Adams 预测器 |
| `sa_solver_pece` | `sa_solver` + 校正器评估（PECE，更准更慢） |

---

## 二、采样器配套参数（只对特定采样器生效）

| 参数 | 默认 | 取值 | 只对哪些采样器生效 | 含义 |
|------|------|------|-------------------|------|
| `sampler_eta` | `0.0` | `0`–`1.0`，步 0.05 | 随机 / 祖先 + SA-Solver（§1.2/§1.3）+ **`ddim`（eta>0 时）** | 随机性（eta）噪声尺度。`0` = 确定性 |
| `sampler_s_noise` | `1.0` | `0`–`2.0`，步 0.05 | 随机 / 祖先（SDE）采样器（§1.2/§1.3）。**注意：`ddim` 不吃 s_noise，只吃 eta** | SDE 噪声倍数。`1.0` = ComfyUI 默认；`0` = 不注入噪声；`>1` = 更多 |
| `sampler_solver_order` | `4` | 1–4 | **仅 `lms`** | 线性多步 Adams-Bashforth 阶数。阶越低越稳、越高越准 |
| `sampler_predictor_order` | `3` | 1–4 | **仅 `sa_solver` / `sa_solver_pece`** | SA-Solver 预测器 Adams 阶。越高越准但需更多历史 |
| `sampler_corrector_order` | `4` | 1–4 | **仅 `sa_solver` / `sa_solver_pece`** | SA-Solver 校正器 Adams 阶。越高校正越准 |

> 给非对应采样器设这些参数**无副作用**（引擎忽略）。所以默认值放着不用管，只在用 lms / SA-Solver 时才调。

---

## 三、调度器 `scheduler`（9 种，默认 `normal`）

| 调度器 | 曲线含义 |
|--------|----------|
| **`normal`（默认）** | 原生 FlowMatchEuler flow 曲线（不变） |
| `karras` | Karras et al. 2022（rho=7） |
| `exponential` | sigma 上几何（exp-linspace）间隔 |
| `sgm_uniform` | ComfyUI normal(sgm=True) |
| `simple` | 按模型自带 sigma 表跨步 |
| `ddim_uniform` | 均匀索引跨步（DDIM） |
| `beta` | 在 sigma 表上取 beta.ppf(a=b=0.6) |
| `linear_quadratic` | ComfyUI 线性-二次（Mochi） |
| `kl_optimal` | KL-optimal 调度（sd-webui #15608） |

> ⚠️ **再次强调**：蒸馏 / 少步模型（走 explicit-sigma，如 Klein）**忽略 scheduler 类型**；调度器只对走 `set_timesteps(mu)` 的基础模型（QwenImage / ZImage）塑形。所以在蒸馏模型上改 scheduler 常常“没感觉”，那是正常的。

---

## 四、CFG（引导强度）

| 参数 | 默认 | 说明 |
|------|------|------|
| `guidance_scale` | `0.0` | 蒸馏引导强度（distillation guidance）。蒸馏 / 少步模型用 `0`；base 模型约 `3.5` |
| `true_cfg_scale` | `1.0` | **经典 CFG**（需要负向提示词）。`1.0` = **关闭**（少步 / 蒸馏模型的正确值）；base / 非蒸馏模型才升到 `4.0` 左右 |

> 少步 / 蒸馏模型：`guidance_scale=0` + `true_cfg_scale=1.0`（都关）。base 模型：`guidance_scale≈3.5`，需要负向提示词时 `true_cfg_scale≈4.0`。

---

## 五、FBCache（首块缓存跳步加速，`fbcache` / `fbcache_uncond`）

FBCache（First-Block Cache，TeaCache 式跳步）通过**在模型变化很小的步复用 transformer 缓存的 block-1..N 残差**来提速。

| 参数 | 默认 | 说明 |
|------|------|------|
| `fbcache` | `0.0` | **COND（正向）分支**阈值 = block-0 输出的**累积相对 L1 变化预算**。累积和低于它时，剩余 transformer 块被**跳过**（廉价复用）；越过就跑一次完整步并重置预算。`0.0` = 关（比特精确）。越高跳得越多 = 越快但**渐进有损**（更软 / 提示词忠实度下降）。典型 `0.03`–`0.12`，按模型 + 步数调（少步蒸馏模型用更小值）。 |
| `fbcache_uncond` | `0.0` | **UNCOND（负向）分支**阈值。仅对**串行 true-CFG** 模型有意义（正向 + 负向两趟：Ideogram4、QwenImage-Layered）。`0` = 负向分支不缓存。单缓存模型（QwenImage t2i / ZImage / Wan）只用上面的 `fbcache`。 |

> **注意**：FBCache 是 **create-time**（建管线时读）旋钮，按**模型家族**（QwenImage / ZImage / Ideogram4 / Wan lighting）分别读取；没有 FBCache 路径的家族（如 Klein）忽略它。改这个值会**重建管线**。

---

## 六、快速推荐

| 场景 | sampler | scheduler | guidance / CFG | fbcache |
|------|---------|-----------|----------------|---------|
| **少步蒸馏 / lightning**（4–8 步） | `euler`（或 `lcm`） | `normal`（多半被忽略） | `guidance_scale=0`、`true_cfg_scale=1.0` | `0` 起，急速时试 `0.03`–`0.06` |
| **完整 base 模型**（20–30 步） | `dpmpp_2m` | `karras` 或 `normal` | `guidance_scale≈3.5`、需负向时 `true_cfg_scale≈4.0` | `0`（要质量）；提速试 `0.05`–`0.12` |
| **要更多细节 / 多样性** | 祖先类（`euler_ancestral` / `dpmpp_2m_sde`） | `normal` / `karras` | 同上 | `0` |
