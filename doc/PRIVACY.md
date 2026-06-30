# QuantFunc — Privacy Policy

This policy states, in plain language, exactly what data the QuantFunc license server
receives and stores. It is the governance counterpart to the technical network audit in
[`NETWORK.md`](NETWORK.md), which shows the same facts at the wire level and tells you how
to verify them yourself.

> **In one sentence:** the QuantFunc license server receives and stores **only** a one-way
> hash of your GPU's UUID, your license key, and request timestamps — and **never** any of
> your prompts, images, generated outputs, files, or system information.

## What the license server receives and stores

When the engine performs its license heartbeat (`POST /api/auth/refresh`) or, for
publisher-protected models, a key-map request, the server receives **only**:

| Field | What it is | Why |
|---|---|---|
| `device_id` | `hex(SHA256(GPU_UUID))` — a **one-way, irreversible** hash of your GPU's hardware UUID only. **Not** your hostname, MAC, username, IP, or `/etc/machine-id`. | Binds a license to hardware (license enforcement). |
| `api_key` | Your license key. | Identifies your entitlement. |
| `_t` (timestamp) + `_s` (HMAC signature) | A request timestamp and an `HMAC-SHA256` keyed on your `api_key`. | Clock calibration + replay-attack prevention. Carry no user content. |
| `model_id` | The identifier of a **publisher-protected** model (key-map requests only). | Fetch/store the model's encrypted key map. |

That is the complete set. (Standard transport-level metadata such as your source IP is
inherently visible to any server you connect to over the internet; we do not use it to
profile or track you, and the engine sends no IP-derived identifier in the request body.)

## What we NEVER receive or store

The engine has **no code path** that transmits, and the server has no facility that stores,
any of the following:

- ❌ prompt text
- ❌ input or reference images
- ❌ generated output images
- ❌ file names, file listings, or directory contents
- ❌ system inventory (CPU/RAM/OS, installed software, drive contents) beyond the GPU-UUID hash
- ❌ telemetry, analytics, usage tracking, or behavioral profiling

**The text-to-image / image-to-image generation path makes zero network calls** — your
creative inputs and outputs never leave your machine via the engine. This is verifiable: see
the `tcpdump` / `strace` recipes in [`NETWORK.md`](NETWORK.md).

## Retention

License records (`device_id`, `api_key`, timestamps) are retained for the duration of the
license relationship and as required for billing/anti-abuse.
*(⚠ PENDING MAINTAINER CONFIRMATION — set the exact retention period and deletion process.)*

## Third-party model downloads (optional, user-initiated)

If you choose to download models, the plugin contacts **ModelScope** and/or **Hugging Face**.
These are optional, user-initiated downloads governed by those services' own privacy policies;
they are not part of the license heartbeat. A manual-install / offline workflow avoids them
entirely.

## Offline operation

The engine supports running with no license server configured (it then makes no network calls
at all). A valid-paid offline-license mode for air-gapped/regulated users is on the roadmap.

## Changes and contact

Questions about this policy or your data: `privacy@quantfunc.com`
*(⚠ PENDING MAINTAINER CONFIRMATION of the real address.)*
For security reports, see [`../SECURITY.md`](../SECURITY.md).

---

*This policy is published by the commercializing legal entity behind QuantFunc.
(⚠ PENDING MAINTAINER CONFIRMATION of the legal entity name, e.g. "MYNDEER LLC", the
retention period, and the contact addresses.) The technical claims here are backed by the
code-anchored audit in [`NETWORK.md`](NETWORK.md).*
