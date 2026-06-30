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

## 1. Every endpoint, and the EXACT fields it sends

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
| 3 | `POST {server_url}/api/models/keymap` | Protected-model EXPORT — single-model **FALLBACK** (used only if the batch endpoint above fails) | `api_key`, `model_id`, `encrypted_map`, `_t`, `_s` — **gzip** | url `KeyMapClient.cpp:378`; body `:396`; AES-encrypt `:384`; fallback caller `PipelineLoader.cpp:1969` |
| 4 | `POST {server_url}/api/models/keymap/download` | Protected-model **LOAD** — decrypt one model's key map | `api_key`, `model_id`, `device_id`, `_t`, `_s` — **gzip** | url `KeyMapClient.cpp:509`; body `:515-517` |
| 5 | `POST {server_url}/api/models/keymap/batch/download` | Protected-model **LOAD (batch)** — **FORTHCOMING**: implemented in the engine but has **no live caller yet**; documented now so a future deploy generates no undocumented traffic | `api_key`, `device_id`, `model_ids` (CSV), `_t`, `_s` — **gzip** | url `KeyMapClient.cpp:587`; body `:592-594` |

### What each field is

- **`device_id`** = `hex(SHA256(GPU_UUID))` — a **one-way SHA-256 hash of the GPU's hardware UUID
  only**. It is **not** your hostname, MAC address, username, IP, or `/etc/machine-id`. It is the
  minimal hardware-binding fingerprint for license enforcement, and it is irreversible.
  `AuthClient.cpp:108-126`.
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
# Every outbound connection the process makes (you'll see only service.quantfunc.com,
# plus ModelScope/HuggingFace IF you are downloading models):
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
