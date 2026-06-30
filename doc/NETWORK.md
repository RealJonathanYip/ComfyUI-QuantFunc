# QuantFunc — Network Transparency

**This document tells you exactly what the QuantFunc engine sends over the network, what it
NEVER sends, and how to verify both yourself in five minutes.** Every claim is anchored to a
specific line in the (closed-source) engine; the claims are written so that a packet capture
confirms them rather than contradicts them.

> **Headline:** the text-to-image / image-to-image **generation path makes ZERO network calls.**
> The only network traffic the engine ever makes is a small **license heartbeat** and, *only for
> publisher-protected models*, a **key-map** request. Neither carries your prompts, images, or any
> user content. There is **no telemetry/analytics SDK anywhere** in the engine — only two source
> files make any HTTP call at all (`src/auth/AuthClient.cpp` and `src/auth/KeyMapClient.cpp`).

---

## 1. Every ENGINE endpoint, and the EXACT fields it sends

*(This section covers the closed C++ **engine** — `service.quantfunc.com` via `src/auth/`. The
open **plugin Python** also makes a small number of network calls — an automatic startup
update-check and optional downloads — enumerated separately in §1b below, so this is not the whole
picture by itself.)*

`server_url` is **not hardcoded** — it is supplied via `config.json` / CLI / API options (the
shipped plugin config points it at `https://service.quantfunc.com`). Setting it empty skips auth
entirely (`AuthClient.cpp:965`).

Every request below is signed by `request_signing::signRequest`, which appends **two signing
fields** to the body before it is sent:

- **`_t`** — the client's estimate of the current server unix timestamp, **sent TO the server to
  prevent replay attacks** (the server rejects a request whose `_t` is not recent).
  `RequestSigning.cpp:95-96`. *(Clock calibration is a separate, CLIENT-side step — NOT what `_t`
  is for: on an HTTP 401 the client reads the `X-Server-Time` response header and adjusts its own
  offset, `AuthClient.cpp:702-706`.)*
- **`_s`** — an `HMAC-SHA256` of the request keyed on your `api_key`: request authentication +
  tamper-evidence. `RequestSigning.cpp:101-106`.

**`_t` and `_s` carry ZERO user content.** They exist only to make each request recent and
tamper-evident. **You WILL see `_t` and `_s` on the wire — that is expected and correct.**

**Compression (matters for the verify recipe below):** the license **heartbeat** (endpoint 1) is
sent as **raw, uncompressed** POST fields (`AuthClient.cpp:681`) — so after you decrypt TLS you see
the four fields in plaintext. The **key-map** requests (endpoints 2–5) are **gzip-compressed**
before sending (`Content-Encoding: gzip`, `KeyMapClient.cpp:316`) — so during a model export/load
a TLS-decrypted capture shows a **binary gzip blob**, which you must `gunzip` to read.

| # | Endpoint | When | EXACT body fields | Code |
|---|---|---|---|---|
| 1 | `POST {server_url}/api/auth/refresh` | License check — on init + a background refresh (1/3-lifetime rule) | `device_id`, `api_key`, `_t`, `_s` — **raw, not compressed** | body `AuthClient.cpp:666`; `_t`/`_s` `RequestSigning.cpp:106`; POST `AuthClient.cpp:691` |
| 2 | `POST {server_url}/api/models/keymap/batch` | Protected-model **bundle EXPORT** — the **PRIMARY** publisher path (sent for every protected bundle exported with `obfuscation_percent > 0`) | `api_key`, `items_json` (a JSON array of `{model_id, encrypted_map}` pairs), `_t`, `_s` — **gzip** | url `KeyMapClient.cpp:467`; body `:474-475`; caller `PipelineLoader.cpp:1962` |
| 3 | `POST {server_url}/api/models/keymap` | Protected-model EXPORT (single-model upload). The **PRIMARY** path for per-component LoRA-export key upload (`finishGEMMLoRAExport`) and for several `from_pretrained` protected-model uploads — sent on a **successful** export with no batch failure. **ALSO** the per-entry fallback when the batch endpoint (#2) fails | `api_key`, `model_id`, `encrypted_map`, `_t`, `_s` — **gzip** | url `KeyMapClient.cpp:378`; body `:396`; AES-encrypt `:384`; primary callers `ComponentImpl.cpp:1656`, `PipelineLoader.cpp:2251/2266/2333`; batch-fallback caller `PipelineLoader.cpp:1969` |
| 4 | `POST {server_url}/api/models/keymap/download` | Protected-model **LOAD** — decrypt one model's key map | `api_key`, `model_id`, `device_id`, `_t`, `_s` — **gzip** | live fn `downloadKeyMapEncrypted`: url `KeyMapClient.cpp:661`, body `:667-669` (all live callers: `ComponentImpl.cpp:1164`, `SDXLPipeline.cpp:116`, `sdxl_factory.cpp:109`). *(A same-URL sibling `downloadKeyMap` at `:509`/`:515-517` is currently unused.)* |
| 5 | `POST {server_url}/api/models/keymap/batch/download` | Protected-model **LOAD (batch)** — **FORTHCOMING**: implemented in the engine but has **no live caller yet**; documented now so a future deploy generates no undocumented traffic | `api_key`, `device_id`, `model_ids` (CSV), `_t`, `_s` — **gzip** | url `KeyMapClient.cpp:587`; body `:592-594` |

### What each field is

- **`device_id`** = `hex(SHA256(GPU_UUID))` — a **one-way SHA-256 hash of the GPU's hardware UUID
  only**. It is **not** your hostname, MAC address, username, IP, or `/etc/machine-id`. It is the
  minimal hardware-binding fingerprint for license enforcement, and it is irreversible. The full
  `computeDeviceId` is `AuthClient.cpp:108-148`; the SHA-256 step itself is `:134-141` (under
  `HAVE_AUTH`, which production builds use — `:108-126` is only the UUID-string formatting).
- **`api_key`** — your license key.
- **`model_id`** / **`model_ids`** — the identifier(s) of publisher-protected/published model(s)
  (key-map endpoints 2–5 only; `model_ids` is a comma-separated list for the batch-load endpoint).
- **`encrypted_map`** / **`items_json`** — the model's tensor-name→UUID map, **AES-encrypted
  client-side** before it ever leaves your machine. `items_json` (batch-export) is a JSON array of
  `{model_id, encrypted_map}` pairs, each `encrypted_map` being the same AES-encrypted DRM
  metadata. Publisher EXPORT paths only; contains no prompts, images, or outputs.
- **`_t`, `_s`** — request signing fields (a replay-prevention timestamp + an HMAC authentication
  tag — see above). No user content.

> The model-name obfuscation (`encrypted_map`) remaps **model tensor names** for publisher DRM. It
> never obfuscates code or behavior, applies only on the protected-model export/load paths, and is
> off for ordinary unprotected models.

---

## 1b. Plugin-side (Python) network calls

Separately from the engine, the **open plugin Python** makes network calls — all to **ModelScope**
(`www.modelscope.cn`; the `QuantFunc/Plugin` repo, plus upstream base-model repos for discovery),
except user-initiated model downloads which may also hit **Hugging Face**. **None carries user
content** — they read public version manifests, list public repo files, or download the engine
library / dependency bundles / models.

| When | What | Endpoint(s) / API | Code |
|---|---|---|---|
| **Automatic, every startup** (background thread) | **Engine update-check** — read `version.json` (is a newer engine lib published?) + `{version}/verify.json` (the SHA-256 integrity manifest for the installed lib) | GET `{RAW}/version.json`; GET `{RAW}/{version}/verify.json` | `check_for_updates()` `auto_update.py:804`; `:192-193`, `:443-446` |
| **Automatic, every startup** (background thread) | **Dropdown resource-cache refresh** — list the public `.safetensors`/`.json` files under each model series to populate the node dropdowns | ModelScope `HubApi.get_model_files` | `refresh_cache_background()` → `_list_ms_dir` `model_auto_loader.py:288` |
| **Automatic, every startup** (background thread) | **Base-model repo discovery** — list available base-model repos for the BaseModelAutoLoader dropdown | ModelScope `HubApi.list_models` | `refresh_base_model_repos_background()` → `_search_base_model_repos` (`api.list_models` `model_auto_loader.py:629`) |
| **Conditional** — lib missing or SHA-256 mismatch | **Engine-lib download / self-heal** — download the `.so`/`.dll`, SHA-256-verify **before** replacing | GET `{RAW}/<platform-path>` | `auto_update.py:282`, `:549-570` |
| **Conditional** — worker can't load the lib (missing platform deps), typically first run | **CUDA dependency-bundle download** — fetch + zip-slip-guarded extract of the deps zip | ModelScope (SDK `model_file_download` / urllib) | `_download_dep_zip` → `_download_from_modelscope` `lib_setup.py:329`; trigger `nodes.py:918` |
| **User-initiated** — you click download in a node | **Model downloads** | ModelScope `snapshot_download` / Hugging Face `snapshot_download` / `hf_hub_download` | `model_auto_loader.py:443/452/491/496/717/763` |
| **Opt-in ONLY** (`QUANTFUNC_PLUGIN_AUTOPULL=1`, **OFF by default**) | **Plugin self-update** — `git pull --rebase` of the plugin's own source from its git remote. Off by default (the normal update path is ComfyUI-Manager); this is the C1 opt-in escape hatch | the plugin's git remote (e.g. `github.com`) | `__init__.py` opt-in block (gated on the env var) |

`{RAW}` = `_MODELSCOPE_RAW_URL = https://www.modelscope.cn/models/QuantFunc/Plugin/resolve/master`
(`auto_update.py:138`).

> **Important for the verify recipe (§3):** because of the THREE automatic startup checks above
> (update-check + dropdown cache + base-model repo discovery), one or more connections to
> `www.modelscope.cn` **will appear at ComfyUI startup even if you never generate or download
> anything** — these are public version-manifest reads + repo file listings (no user content, no
> data egress). *(Gating them behind explicit consent / an opt-out is a planned follow-up; today
> they run in background threads and never block node loading.)*

---

## 2. WE NEVER SEND

The engine **never** transmits any of the following — there is no code path that places them in a
request body, and the generation path makes no network calls at all:

- ❌ prompt text
- ❌ input / reference images
- ❌ generated output images
- ❌ file listings or directory contents
- ❌ system inventory (CPU/RAM/OS details, installed software, drive contents) — beyond the
  one-way GPU-UUID hash used for licensing
- ❌ telemetry, analytics, usage tracking, crash pings, or "phone-home" of any kind

**Why you can trust this:** a grep of the entire engine for outbound HTTP (`curl_easy_perform`)
matches **only** `src/auth/AuthClient.cpp` and `src/auth/KeyMapClient.cpp`. There is no analytics
SDK (Sentry/Mixpanel/PostHog/Amplitude/Datadog/etc.) anywhere in the codebase. The `generate()` /
`generate_edit()` denoise path issues **no** network calls.

---

## 3. Verify it yourself (copy-paste, 5 minutes)

You do not have to take our word for any of the above. With the plugin running inside ComfyUI,
find the ComfyUI process id (`pgrep -f main.py` on Linux) and run:

### Linux — what hosts does it connect to?
```bash
# Every outbound connection the process makes. Expect EXACTLY: service.quantfunc.com (engine
# license heartbeat), AND www.modelscope.cn AT STARTUP (the plugin's three automatic startup
# checks — update-check + dropdown cache + base-model repo discovery, see §1b — these appear
# even if you generate/download nothing), plus ModelScope/HuggingFace only WHEN you download
# models. No other hosts in the default config. (If you set the opt-in QUANTFUNC_PLUGIN_AUTOPULL=1,
# you will ALSO see the plugin's git remote — e.g. github.com — at startup; see the §1b opt-in row.)
sudo strace -f -e trace=connect -p <comfyui_pid> 2>&1 | grep -i 'sin_addr\|sin6_addr'
```

### Linux — capture the license traffic and read the fields
```bash
# See the license HEARTBEAT on the wire. It is sent RAW (uncompressed), so after TLS-decrypt you
# observe exactly device_id, api_key, _t, _s in plaintext — and nothing else:
sudo tcpdump -A -s0 host service.quantfunc.com
# HTTPS note: bodies are TLS-encrypted on the wire. To read plaintext, run the traffic through
# mitmproxy (install its CA) or set SSLKEYLOGFILE and decrypt in Wireshark.
```

> **Scope of this recipe (important — so a capture confirms the doc rather than contradicting it):**
> the **heartbeat** (endpoint 1) is uncompressed, so the four plaintext fields are directly visible
> after TLS-decrypt. The **key-map** traffic (endpoints 2–5, seen *only* during a protected-model
> **export or load**) is **gzip-compressed** — after TLS-decrypt it appears as a binary gzip blob;
> pipe it through `gunzip` to read its fields (`api_key` / `model_id`(s) / the AES-encrypted map /
> `_t` / `_s`). Either way you will find **no** prompts, images, or outputs — and the text-to-image
> generation path emits no network traffic at all.

### Linux — what files does it write?
```bash
# Every file the engine opens for writing. You will see only the model cache dir, the plugin
# bin/ directory (lib swap), and temp staging — never your documents/home tree:
sudo strace -f -e trace=openat,write -p <comfyui_pid> 2>&1 | grep -iE 'O_WRONLY|O_CREAT'
```

### Windows
Use **Process Monitor (ProcMon)**: filter `Process Name` = your ComfyUI/python process, then
watch the **Network** and **File Write** event classes. You will see the same finite, bounded set:
one license host, optional model-download hosts, and writes confined to the cache + plugin `bin/`.

---

## 4. Offline / air-gapped operation

The engine already supports an **empty `server_url`**, which skips authentication entirely
(`AuthClient.cpp:965`) — the engine then makes **no** network calls at all. A formal *valid-paid*
offline-license mode (so paid users can run fully air-gapped without disabling licensing) is on
the roadmap; the license check is designed to **degrade, not brick**, on a transient network
outage during a valid token window.

---

*Code references are to the closed-source engine repository (`src/auth/`). The plugin Python is
fully open and public, so the wrapper that calls these endpoints is itself auditable. Line numbers
are accurate as of the engine version shipped with this plugin release; the field set is stable.*
