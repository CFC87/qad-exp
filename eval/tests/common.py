"""Shared helpers of the evaluation tests (CPU only)."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parents[1]
REPO = EVAL.parent
LIB = EVAL / "lib"
TESTS = EVAL / "tests"
CLIENT_PY = REPO / ".venv-client" / "bin" / "python"
BFCL_PY = REPO / ".venv-bfcl" / "bin" / "python"
TAU2_PY = REPO / "third_party" / "tau2-bench" / ".venv" / "bin" / "python"
TOOLS_PY = REPO / ".venv-tools" / "bin" / "python"
IFBENCH_PY = REPO / "third_party" / "IFBench" / ".venv" / "bin" / "python"

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def run(cmd, **kw):
    kw.setdefault("check", True)
    return subprocess.run([str(c) for c in cmd], **kw)


class Report:
    def __init__(self, title):
        self.title, self.rows, self.fail = title, [], 0

    def add(self, name, ok, detail=""):
        self.rows.append((name, ok, detail))
        self.fail += ok is False
        print(f"{'PASS' if ok else 'SKIP' if ok is None else 'FAIL'}  {name}  {detail}", flush=True)

    def done(self):
        n = {k: sum(1 for r in self.rows if r[1] is k) for k in (True, False, None)}
        print(f"\n{self.title}: {n[True]} passed, {n[False]} failed, {n[None]} skipped")
        sys.exit(1 if self.fail else 0)
