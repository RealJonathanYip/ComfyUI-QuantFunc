# QuantFunc — Privacy Policy

This policy states, in plain language, exactly what data the QuantFunc license server
receives and stores. It is the governance counterpart to the technical network audit in
[`NETWORK.md`](NETWORK.md), which shows the same facts at the wire level and tells you how
to verify them yourself.

> **In one sentence:** the engine sends the license server **only** a one-way hash of your GPU's
> UUID, your license key, and request timestamps — **never** your prompts, images, outputs, files,
> or system information (a *code-verifiable* fact about the open client); and MYNDEER's policy is
> that the closed server **retains only** those same license fields (an *operator commitment*).
> These are two different kinds of guarantee and are kept honestly distinct below.

## What the license server receives and stores

When the engine performs its license heartbeat (`POST /api/auth/refresh`) or, for
publisher-protected models, a key-map request, the server receives **only**:

| Field | What it is | Why |
|---|---|---|
| `device_id` | `hex(SHA256(GPU_UUID))` — a **one-way, irreversible** hash of your GPU's hardware UUID only. **Not** your hostname, MAC, username, IP, or `/etc/machine-id`. | Binds a license to hardware (license enforcement). |
| `api_key` | Your license key. | Identifies your entitlement. |
| `_t` (timestamp) + `_s` (HMAC) | A recent server-time estimate (`_t`) and an `HMAC-SHA256` keyed on your `api_key` (`_s`). | Replay-attack prevention (`_t`, the server checks it is recent) + request authentication (`_s`). Carry no user content. |
| `model_id` | The identifier of a **publisher-protected** model (key-map requests only). | Fetch/store the model's encrypted key map. |

That is the complete set. (Standard transport-level metadata such as your source IP is
inherently visible to any server you connect to over the internet; we do not use it to
profile or track you, and the engine sends no IP-derived identifier in the request body.)

## What is NEVER sent (client) vs what the server retains (operator policy)

These are two distinct kinds of guarantee. We keep them honestly separate: one is provable from
code, the other is a policy promise about a server you cannot read.

### ① CLIENT — code-verifiable (provable, and checkable by you)

The engine has **no code path** that transmits any of the following — verifiable from the open
plugin Python, the `src/auth/` network surface, and your own capture (see [`NETWORK.md`](NETWORK.md)):

- ❌ prompt text
- ❌ input or reference images
- ❌ generated output images
- ❌ file names, file listings, or directory contents
- ❌ system inventory (CPU/RAM/OS, installed software, drive contents) beyond the GPU-UUID hash
- ❌ telemetry, analytics, usage tracking, or behavioral profiling

This is a claim about **code**, and it is independently checkable. **The text-to-image /
image-to-image generation path makes zero network calls** — your creative inputs and outputs never
leave your machine via the engine.

### ② SERVER — operator policy commitment

The license server is **closed** — it is not in this repository and cannot be verified from source.
As a binding **operational policy**, **MYNDEER** (the operator behind QuantFunc; its exact
registered legal entity is being finalized — see the note at the foot of this document)
**commits that the license server retains only the `device_id` hash, your `api_key`, and request
timestamps, and stores none of the user content listed in ① above.** This is a policy promise about
the closed server (stated deliberately as a commitment, not as a code-verifiable fact), so the
trust posture stays honest about which claim is which. *(Before this policy is published at ship
time, the operator must replace this with the confirmed legal entity name; an unconfirmed entity
cannot bind the commitment.)*

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
