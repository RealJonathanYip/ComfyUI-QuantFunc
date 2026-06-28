#!/usr/bin/env python3
"""SHA-256 artifact integrity manifest tool for the ComfyUI-QuantFunc plugin.

From plugin version 0.0.12 every shipped engine-library artifact carries a SHA-256
recorded in a PER-VERSION manifest `{version}/verify.json` on ModelScope
(QuantFunc/Plugin). The plugin verifies the locally-installed lib against it on
startup (see the plugin's auto_update.py).

This producer tool builds and checks that manifest. It is intentionally
self-contained (stdlib only) and is the CONSOLIDATION step: the per-version
manifest is built ONCE on ONE host that holds ALL the version's artifacts, then
uploaded once — never incrementally / concurrently from parallel build hosts
(that would lose-update a platform's hashes).

Manifest format (per-version {version}/verify.json), platform-grouped, schema'd:
    {
      "schema": 1,
      "linux": { "libquantfunc.so": "<sha256hex>", "libquantfunc-12.so": "<sha256hex>" },
      "win32": { "quantfunc.dll": "<sha256hex>", "quantfunc-12.dll": "<sha256hex>" }
    }

Subcommands:
  build  --version V --out PATH --artifact PLAT:FILE ... --expect PLAT:FILENAME ...
         compute sha256 of each --artifact, write ONE complete manifest atomically.
         FAIL-CLOSED if --expect is empty or any --expect'd (plat,filename) is not
         satisfied by a provided --artifact. --version must be >= 0.0.12.
  check  --manifest PATH --expect PLAT:FILENAME ... [--artifact PLAT:FILE ...]
         assert every --expect'd entry exists in the manifest, and that each
         provided --artifact's recomputed sha matches. Non-zero exit on any miss.
  selftest
         round-trip + adversarial self-checks; non-zero exit on any failure.

Usage example (release/distribute orchestrator, after pulling all artifacts back):
  verify_manifest.py build --version 0.0.12 --out /tmp/0.0.12/verify.json \
      --artifact linux:/path/libquantfunc.so   --artifact linux:/path/libquantfunc-12.so \
      --artifact win32:/path/quantfunc.dll      --artifact win32:/path/quantfunc-12.dll \
      --expect   linux:libquantfunc.so          --expect   linux:libquantfunc-12.so \
      --expect   win32:quantfunc.dll             --expect   win32:quantfunc-12.dll
  # then upload /tmp/0.0.12/verify.json to ModelScope QuantFunc/Plugin at 0.0.12/verify.json
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile

SCHEMA_VERSION = 1
VERIFY_FLOOR = "0.0.12"           # the version from which manifests exist (fail-closed below)
_KNOWN_PLATFORMS = ("linux", "win32")
_CHUNK = 1 << 20                  # 1 MiB


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _parse_version(v):
    parts = []
    for tok in str(v).split("."):
        try:
            parts.append(int(tok))
        except ValueError:
            parts.append(0)
    return parts


def _ver_cmp(a, b):
    ap, bp = _parse_version(a), _parse_version(b)
    n = max(len(ap), len(bp))
    ap += [0] * (n - len(ap))
    bp += [0] * (n - len(bp))
    return (ap > bp) - (ap < bp)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _split_pair(s, what):
    """'plat:value' -> (plat, value). 'value' may itself contain ':' (win paths)."""
    if ":" not in s:
        raise SystemExit("[verify_manifest] {} must be PLATFORM:VALUE, got {!r}".format(what, s))
    plat, value = s.split(":", 1)
    plat = plat.strip()
    if plat not in _KNOWN_PLATFORMS:
        raise SystemExit(
            "[verify_manifest] {} platform {!r} not one of {}".format(what, plat, _KNOWN_PLATFORMS)
        )
    if not value:
        raise SystemExit("[verify_manifest] {} value empty in {!r}".format(what, s))
    return plat, value


def _atomic_write_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _require_floor(version):
    if _ver_cmp(version, VERIFY_FLOOR) < 0:
        raise SystemExit(
            "[verify_manifest] REFUSED: version {} is below the {} floor — "
            "manifests only exist from {} onward".format(version, VERIFY_FLOOR, VERIFY_FLOOR)
        )


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #
def cmd_build(args):
    _require_floor(args.version)

    expects = [_split_pair(e, "--expect") for e in (args.expect or [])]
    if not expects:
        raise SystemExit(
            "[verify_manifest] REFUSED: build requires at least one --expect "
            "PLAT:FILENAME (the completeness matrix); refusing to write a manifest "
            "whose completeness is undeclared."
        )

    artifacts = [_split_pair(a, "--artifact") for a in (args.artifact or [])]
    if not artifacts:
        raise SystemExit("[verify_manifest] REFUSED: build requires at least one --artifact")

    manifest = {"schema": SCHEMA_VERSION}
    for plat, filepath in artifacts:
        if not os.path.isfile(filepath):
            raise SystemExit(
                "[verify_manifest] artifact not found: {}:{}".format(plat, filepath)
            )
        name = os.path.basename(filepath)
        try:
            digest = _sha256(filepath)
            size = os.path.getsize(filepath)
        except OSError as e:
            raise SystemExit("[verify_manifest] cannot read artifact {}: {}".format(filepath, e))
        pm = manifest.setdefault(plat, {})
        prior = pm.get(name)
        if prior is not None and prior != digest:
            raise SystemExit(
                "[verify_manifest] REFUSED: two different {}:{} artifacts hash differently "
                "but share a basename — ambiguous manifest key".format(plat, name)
            )
        pm[name] = digest
        print("[verify_manifest] {}/{}  sha256={}  ({} bytes)".format(plat, name, digest, size))

    # FAIL-CLOSED completeness: every declared --expect must be present.
    missing = [
        "{}:{}".format(plat, fname)
        for plat, fname in expects
        if fname not in manifest.get(plat, {})
    ]
    if missing:
        raise SystemExit(
            "[verify_manifest] REFUSED (incomplete): the following --expect'd "
            "artifacts were not provided as --artifact: {}. A partial ship must not "
            "produce a partial manifest.".format(", ".join(sorted(missing)))
        )

    try:
        _atomic_write_json(args.out, manifest)
    except OSError as e:
        raise SystemExit("[verify_manifest] cannot write manifest {}: {}".format(args.out, e))
    print("[verify_manifest] wrote {} (schema={}, {} platform(s))".format(
        args.out, SCHEMA_VERSION,
        len([k for k in manifest if k != "schema"])))
    return 0


# --------------------------------------------------------------------------- #
# check
# --------------------------------------------------------------------------- #
def cmd_check(args):
    try:
        with open(args.manifest, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as e:
        raise SystemExit("[verify_manifest] cannot read manifest {}: {}".format(args.manifest, e))
    if not isinstance(manifest, dict):
        raise SystemExit("[verify_manifest] manifest is not a JSON object")
    if not (args.expect or args.artifact):
        raise SystemExit("[verify_manifest] check requires at least one --expect or --artifact")

    s = manifest.get("schema", 1)
    if not isinstance(s, int) or isinstance(s, bool):
        raise SystemExit("[verify_manifest] manifest 'schema' must be an integer, got {!r}".format(s))

    problems = []

    for plat, fname in (_split_pair(e, "--expect") for e in (args.expect or [])):
        pm = manifest.get(plat)
        if not isinstance(pm, dict) or fname not in pm:
            problems.append("missing in manifest: {}:{}".format(plat, fname))
        elif not isinstance(pm[fname], str) or not pm[fname]:
            problems.append("non-string/empty sha: {}:{}".format(plat, fname))

    for plat, filepath in (_split_pair(a, "--artifact") for a in (args.artifact or [])):
        name = os.path.basename(filepath)
        pm = manifest.get(plat)
        recorded = pm.get(name) if isinstance(pm, dict) else None
        if recorded is None:
            problems.append("artifact not in manifest: {}:{}".format(plat, name))
            continue
        if not os.path.isfile(filepath):
            problems.append("artifact file not found: {}".format(filepath))
            continue
        try:
            actual = _sha256(filepath)
        except OSError as e:
            problems.append("cannot read artifact {}: {}".format(filepath, e))
            continue
        if actual != recorded:
            problems.append(
                "sha MISMATCH {}:{}  manifest={}  actual={}".format(plat, name, recorded, actual)
            )
        else:
            print("[verify_manifest] OK {}:{}  {}".format(plat, name, actual))

    if problems:
        for p in problems:
            print("[verify_manifest] FAIL: {}".format(p))
        raise SystemExit(1)
    print("[verify_manifest] check passed")
    return 0


# --------------------------------------------------------------------------- #
# selftest
# --------------------------------------------------------------------------- #
def cmd_selftest(_args):
    import io
    import contextlib

    failures = []

    def expect(cond, msg):
        if cond:
            print("  PASS: {}".format(msg))
        else:
            print("  FAIL: {}".format(msg))
            failures.append(msg)

    def run(argv):
        """Run main() with argv, capturing exit code + stdout."""
        buf = io.StringIO()
        code = 0
        try:
            with contextlib.redirect_stdout(buf):
                code = main(argv)
        except SystemExit as e:
            code = e.code if isinstance(e.code, int) else 1
            # SystemExit messages go to stderr; we only care about the code here
        return code, buf.getvalue()

    with tempfile.TemporaryDirectory() as d:
        lib_so = os.path.join(d, "libquantfunc.so")
        lib_so12 = os.path.join(d, "libquantfunc-12.so")
        dll = os.path.join(d, "quantfunc.dll")
        with open(lib_so, "wb") as f:
            f.write(b"fake-linux-cu13-engine-bytes")
        with open(lib_so12, "wb") as f:
            f.write(b"fake-linux-cu12-engine-bytes")
        with open(dll, "wb") as f:
            f.write(b"fake-windows-cu13-engine-bytes")
        out = os.path.join(d, "0.0.12", "verify.json")

        # 1) build round-trip
        code, _ = run([
            "build", "--version", "0.0.12", "--out", out,
            "--artifact", "linux:" + lib_so, "--artifact", "linux:" + lib_so12,
            "--artifact", "win32:" + dll,
            "--expect", "linux:libquantfunc.so", "--expect", "linux:libquantfunc-12.so",
            "--expect", "win32:quantfunc.dll",
        ])
        expect(code == 0 and os.path.isfile(out), "build writes a manifest")

        man = {}
        if os.path.isfile(out):
            with open(out, encoding="utf-8") as f:
                man = json.load(f)
        expect(man.get("schema") == SCHEMA_VERSION and isinstance(man.get("schema"), int),
               "manifest carries integer schema={}".format(SCHEMA_VERSION))
        expect(man.get("linux", {}).get("libquantfunc.so") == _sha256(lib_so),
               "recorded sha matches the real file digest")

        # 2) check matches
        code, _ = run([
            "check", "--manifest", out,
            "--expect", "linux:libquantfunc.so", "--expect", "win32:quantfunc.dll",
            "--artifact", "linux:" + lib_so, "--artifact", "win32:" + dll,
        ])
        expect(code == 0, "check passes on the unmodified artifacts")

        # 3) tamper -> check detects mismatch
        with open(lib_so, "ab") as f:
            f.write(b"TAMPER")
        code, _ = run([
            "check", "--manifest", out, "--artifact", "linux:" + lib_so,
        ])
        expect(code != 0, "check FAILS (non-zero) on a tampered artifact")

        # 4) <0.0.12 floor refusal
        code, _ = run([
            "build", "--version", "0.0.11", "--out", out,
            "--artifact", "linux:" + lib_so, "--expect", "linux:libquantfunc.so",
        ])
        expect(code != 0, "build REFUSES version < 0.0.12 (fail-closed floor)")

        # 5) empty --expect refusal
        code, _ = run([
            "build", "--version", "0.0.12", "--out", out, "--artifact", "linux:" + lib_so,
        ])
        expect(code != 0, "build REFUSES an empty --expect (undeclared completeness)")

        # 6) incomplete: --expect not satisfied by --artifact
        code, _ = run([
            "build", "--version", "0.0.12", "--out", out,
            "--artifact", "linux:" + lib_so,
            "--expect", "linux:libquantfunc.so", "--expect", "win32:quantfunc.dll",
        ])
        expect(code != 0, "build REFUSES when an --expect'd artifact is missing (completeness gate)")

        # 7) corrupt/garbage manifest -> clean non-zero exit, NOT a traceback
        bad = os.path.join(d, "bad.json")
        with open(bad, "w") as f:
            f.write("{ this is not valid json")
        code, _ = run(["check", "--manifest", bad, "--expect", "linux:libquantfunc.so"])
        expect(code != 0, "check exits non-zero (clean) on a corrupt manifest")

        # 8) two different artifacts sharing a basename -> fail-closed
        sub_a, sub_b = os.path.join(d, "a"), os.path.join(d, "b")
        os.makedirs(sub_a)
        os.makedirs(sub_b)
        dup_a, dup_b = os.path.join(sub_a, "libquantfunc.so"), os.path.join(sub_b, "libquantfunc.so")
        with open(dup_a, "wb") as f:
            f.write(b"AAA")
        with open(dup_b, "wb") as f:
            f.write(b"BBB")
        code, _ = run([
            "build", "--version", "0.0.12", "--out", out,
            "--artifact", "linux:" + dup_a, "--artifact", "linux:" + dup_b,
            "--expect", "linux:libquantfunc.so",
        ])
        expect(code != 0, "build REFUSES two different artifacts sharing a basename")

        # 9) malformed PLAT:VALUE (no colon) -> clean non-zero exit, not a traceback
        code, _ = run(["check", "--manifest", out, "--expect", "win32quantfunc.dll"])
        expect(code != 0, "malformed PLAT:VALUE (no colon) exits cleanly")

    if failures:
        print("\n[verify_manifest] SELFTEST FAILED: {} check(s) failed".format(len(failures)))
        return 1
    print("\n[verify_manifest] SELFTEST PASSED")
    return 0


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="build a per-version verify.json from a complete artifact set")
    b.add_argument("--version", required=True, help="plugin/lib version (>= %s)" % VERIFY_FLOOR)
    b.add_argument("--out", required=True, help="output manifest path (e.g. <version>/verify.json)")
    b.add_argument("--artifact", action="append", metavar="PLAT:FILE", help="repeatable")
    b.add_argument("--expect", action="append", metavar="PLAT:FILENAME",
                   help="repeatable; the required completeness matrix (fail-closed)")
    b.set_defaults(func=cmd_build)

    c = sub.add_parser("check", help="verify artifacts/completeness against a manifest")
    c.add_argument("--manifest", required=True)
    c.add_argument("--expect", action="append", metavar="PLAT:FILENAME", help="repeatable")
    c.add_argument("--artifact", action="append", metavar="PLAT:FILE", help="repeatable")
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("selftest", help="run built-in self-checks")
    s.set_defaults(func=cmd_selftest)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
