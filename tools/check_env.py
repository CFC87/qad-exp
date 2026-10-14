#!/usr/bin/env python3
"""Verify this machine against env/versions.toml and write reports/env_report.{json,md}.

    python3 tools/check_env.py            # everything installed by setup/setup.sh
    python3 tools/check_env.py --deep     # also sha256 every model weight file against Hugging Face's recorded hash

Status per item: ok, FAIL (a strict pin differs or is missing), warn (host-level difference: driver, OS, docker), skip
(not installed, e.g. an optional model). Exit code 1 if any item is FAIL. Send the report together with any results.
Stdlib only (system python3 >= 3.11).
"""
import argparse, hashlib, json, os, re, subprocess, sys, tomllib
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
V = tomllib.loads((REPO / "env" / "versions.toml").read_text())
UV = REPO / ".tools" / "uv" / "uv"
rows = []


ENV = dict(os.environ, UV_CACHE_DIR=str(REPO / ".cache" / "uv"))   # uv's interpreter cache stays inside the repository


def sh(cmd, cwd=None):
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), cwd=cwd, capture_output=True, text=True, timeout=600, env=ENV)
        return r.returncode, r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return 1, str(e)


def add(cat, name, expected, found, status):
    rows.append(dict(category=cat, item=name, expected=str(expected), found=str(found), status=status))


def eq(cat, name, expected, found, host=False):
    add(cat, name, expected, found, "ok" if str(expected) == str(found) else ("warn" if host else "FAIL"))


def norm(n):
    return re.sub(r"[-_.]+", "-", n).lower()


PY = tuple(int(x) for x in V["tools"]["python"].split("."))
MARK_VARS = {"sys_platform": "linux", "platform_system": "Linux", "os_name": "posix", "platform_machine": "x86_64",
             "platform_python_implementation": "CPython", "implementation_name": "cpython"}


def marker_applies(marker):
    """Evaluate a PEP 508 environment marker for this Linux x86_64 CPython (enough for uv export output)."""
    def ver(m):
        return f"({PY} {m.group(2)} {tuple(int(x) for x in m.group(3).split('.'))})"
    e = re.sub(r"(python_full_version|python_version)\s*(<=|>=|==|!=|<|>)\s*'([0-9.]+)'", ver, marker)
    for k, v in MARK_VARS.items():
        e = re.sub(rf"\b{k}\b", repr(v), e)
    return bool(eval(e, {"__builtins__": {}}))  # noqa: S307 (only literals and comparisons remain)


def parse_pins(text):
    out = {}
    for line in text.splitlines():
        line = line.split("#")[0].strip()
        if not line or line.startswith("-") or "==" not in line:
            continue
        spec, _, marker = line.partition(";")
        if marker.strip() and not marker_applies(marker.strip()):
            continue
        n, v = spec.split("==", 1)
        out[norm(n.split("[")[0])] = v.strip()
    return out


def compare_env(cat, name, want, have):
    miss = sorted(k for k in want if k not in have)
    extra = sorted(k for k in have if k not in want)
    diff = sorted(f"{k} {want[k]} != {have[k]}" for k in want if k in have and want[k] != have[k])
    if not have:
        add(cat, name, f"{len(want)} packages", "not installed", "FAIL")
    elif miss or extra or diff:
        add(cat, name, f"{len(want)} packages", f"missing {miss[:5]} extra {extra[:5]} differ {diff[:5]}", "FAIL")
    else:
        add(cat, name, f"{len(want)} packages", "identical", "ok")


def host():
    h = V["host"]
    rc, out = sh("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader")
    gpus = [l.split(", ") for l in out.splitlines()] if rc == 0 else []
    eq("host", "GPU model", h["gpu_name"], ",".join(sorted({g[0] for g in gpus})) or "none", host=True)
    eq("host", "GPUs per node", h["gpus_per_node"], len(gpus), host=True)
    eq("host", "driver", h["driver"], ",".join(sorted({g[1] for g in gpus})) or "none", host=True)
    rc, out = sh("nvidia-smi")
    m = re.search(r"CUDA Version: ([0-9.]+)", out)
    eq("host", "CUDA (driver)", h["cuda_driver"], m.group(1) if m else "none", host=True)
    osr = dict(l.split("=", 1) for l in Path("/etc/os-release").read_text().splitlines() if "=" in l)
    found = osr.get("PRETTY_NAME", "").strip('"')
    add("host", "OS", h["os"], found, "ok" if found.startswith(h["os"]) else "warn")
    rc, out = sh("docker version --format '{{.Server.Version}}'")
    eq("host", "docker", h["docker"], out if rc == 0 else "missing", host=True)
    rc, out = sh("nvidia-container-cli --version")
    m = re.search(r"cli-version: ([0-9.]+)", out)
    eq("host", "NVIDIA Container Toolkit", h["nvidia_container_toolkit"], m.group(1) if m else "missing", host=True)
    rc, out = sh("df -BG --output=avail " + str(REPO))
    add("host", "free disk at repo", "-", out.splitlines()[-1].strip() if rc == 0 else "?", "ok")


def tools():
    rc, out = sh([str(UV), "--version"])
    eq("tools", "uv", V["tools"]["uv"], out.split()[1] if rc == 0 else "missing")


def images():
    for k, i in V["images"].items():
        rc, out = sh(["docker", "image", "inspect", "--format", "{{json .RepoDigests}}", f"{i['ref']}@{i['digest']}"])
        ok = rc == 0 and i["digest"] in out
        add("images", k, f"{i['ref']}@{i['digest'][:19]}…", "present" if ok else "missing", "ok" if ok else "FAIL")


def sender():
    p = REPO / ".cache" / "vllm-sender" / "vllm" / "_version.py"
    if not p.exists():
        add("sender", "vLLM weight-transfer package", "from the vllm image", "not extracted", "skip")
        return
    m = re.search(r"__commit_id__ = .*?['\"]g?([0-9a-f]{7,})", p.read_text())
    eq("sender", "vLLM weight-transfer package", "98dff2a81", m.group(1)[:9] if m else "?")


def sources():
    for k, s in V["sources"].items():
        d = REPO / "third_party" / k
        if not d.exists():
            add("sources", k, s["commit"][:12], "not checked out", "FAIL")
            continue
        rc, head = sh(["git", "-C", str(d), "rev-parse", "HEAD"])
        rc2, st = sh(["git", "-C", str(d), "status", "--porcelain", "--untracked-files=no"])
        found = head[:12] + (" (modified)" if st else "")
        add("sources", k, s["commit"][:12], found, "ok" if head == s["commit"] and not st else "FAIL")
        if "submodule_LightCPVerifier" in s:
            rc, sub = sh(["git", "-C", str(d / "LightCPVerifier"), "rev-parse", "HEAD"])
            eq("sources", f"{k}/LightCPVerifier", s["submodule_LightCPVerifier"][:12], sub[:12])
    for proj, extra in (("tau2-bench", ["--no-dev"]), ("IFBench", [])):
        d = REPO / "third_party" / proj
        if not (d / ".venv").exists():
            add("venvs", f"{proj}/.venv", "uv.lock", "not installed", "FAIL")
            continue
        rc, lock = sh([str(UV), "export", "--frozen", "--no-hashes", "--format", "requirements-txt", *extra], cwd=d)
        rc2, have = sh([str(UV), "pip", "freeze", "--python", str(d / ".venv" / "bin" / "python")])
        compare_env("venvs", f"{proj}/.venv (uv.lock)", parse_pins(lock), parse_pins(have))


def venvs():
    for k, v in V["venvs"].items():
        py = REPO / f".venv-{k}" / "bin" / "python"
        want = parse_pins((REPO / v["lock"]).read_text())
        if not py.exists():
            compare_env("venvs", f".venv-{k}", want, {})
            continue
        rc, ver = sh([str(py), "-c", "import platform;print(platform.python_version())"])
        eq("venvs", f".venv-{k} python", V["tools"]["python"], ver)
        rc, have = sh([str(UV), "pip", "freeze", "--python", str(py)])
        compare_env("venvs", f".venv-{k}", want, parse_pins(have))


def models(deep):
    for k, m in V["models"].items():
        d = REPO / "models" / k
        meta = list((d / ".cache" / "huggingface" / "download").rglob("*.metadata")) if d.exists() else []
        if not meta:
            add("models", k, m["revision"][:12], "not downloaded", "skip" if m.get("optional") else "FAIL")
            continue
        revs = {p.read_text().splitlines()[0] for p in meta}
        found = ",".join(sorted(r[:12] for r in revs))
        ok = revs == {m["revision"]}
        if ok and deep:
            bad = []
            for p in meta:
                etag = p.read_text().splitlines()[1].strip('"')
                f = d / p.relative_to(d / ".cache" / "huggingface" / "download").with_suffix("")
                if len(etag) == 64 and f.exists():  # LFS files: the etag is the content sha256
                    h = hashlib.sha256()
                    with open(f, "rb") as fh:
                        for b in iter(lambda: fh.read(1 << 24), b""):
                            h.update(b)
                    if h.hexdigest() != etag:
                        bad.append(f.name)
            found += f"; sha256 {'all match' if not bad else 'MISMATCH ' + ','.join(bad[:3])}"
            ok = not bad
        add("models", k, m["revision"][:12], found, "ok" if ok else "FAIL")


def data():
    m = tomllib.loads((REPO / "data" / "MANIFEST.toml").read_text())
    for name, a in m["assets"].items():
        p = REPO / a["path"]
        if not p.exists():
            add("data", a["path"], a["sha256"][:12], "not fetched", "FAIL")
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        eq("data", a["path"], a["sha256"][:12], h.hexdigest()[:12])


def eval_data():
    """Evaluation files against eval/data/SHA256SUMS, IFBench's NLTK data against eval/data/NLTK_SHA256SUMS, the unpacked
    LiveCodeBench Pro test cases and the judge image (all made by setup/setup.sh eval-data / judge)."""
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 24), b""):
                h.update(b)
        return h.hexdigest()
    for sums, base, cat in ((REPO / "eval" / "data" / "SHA256SUMS", REPO / "eval" / "data", "eval data"),
                            (REPO / "eval" / "data" / "NLTK_SHA256SUMS", REPO / ".cache" / "nltk_data", "eval data (NLTK)")):
        for line in sums.read_text().splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            want, name = line.split()
            p = base / name
            if not p.exists():
                add(cat, name, want[:12], "missing: setup/setup.sh eval-data (gated sets need HF_TOKEN)", "FAIL")
            else:
                eq(cat, name, want[:12], sha(p)[:12])
    lcb = REPO / "eval" / "data" / "lcb_pro.jsonl"
    if lcb.exists():
        pids = {json.loads(l)["problem_id"] for l in open(lcb)}
        have = {p for p in pids if (REPO / ".cache" / "lcbpro_judge" / "problems" / p / "config.yaml").exists()}
        add("eval data", "LiveCodeBench Pro test cases (unpacked)", f"{len(pids)} problems", f"{len(have)} problems",
            "ok" if have == pids else "FAIL")
    rc, out = sh(["docker", "image", "inspect", "--format", "{{.Id}}", "lightcpverifier:qadx"])
    add("eval data", "LiveCodeBench Pro judge image", "lightcpverifier:qadx", out[:19] if rc == 0 else "not built (setup/setup.sh judge)",
        "ok" if rc == 0 else "FAIL")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="hash every model weight file")
    a = ap.parse_args()
    host(); tools(); images(); sender(); sources(); venvs(); models(a.deep); data(); eval_data()
    rc, commit = sh(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    rc2, dirty = sh(["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"])
    rep = dict(time=datetime.now(timezone.utc).isoformat(timespec="seconds"), hostname=os.uname().nodename,
               repo_commit=commit, repo_dirty=bool(dirty), items=rows)
    out = REPO / "reports"
    out.mkdir(exist_ok=True)
    (out / "env_report.json").write_text(json.dumps(rep, indent=1))
    md = [f"# Environment report {rep['time']}", "", f"repo {commit[:12]}{' (modified)' if dirty else ''}", "",
          "| category | item | expected | found | status |", "|---|---|---|---|---|"]
    md += [f"| {r['category']} | {r['item']} | {r['expected']} | {r['found']} | {r['status']} |" for r in rows]
    (out / "env_report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    n = {s: sum(r["status"] == s for r in rows) for s in ("ok", "FAIL", "warn", "skip")}
    print(f"\n{n['ok']} ok, {n['FAIL']} FAIL, {n['warn']} warn, {n['skip']} skip -> reports/env_report.md")
    sys.exit(1 if n["FAIL"] else 0)


if __name__ == "__main__":
    main()
