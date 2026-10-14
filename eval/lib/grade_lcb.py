"""Grade LiveCodeBench v6 generations by executing them against the benchmark's test cases (runs in .venv-client).

The code is the last ```python block after </think>. It runs in a subprocess (`python -I`, this environment's
interpreter) with CPU / memory / process limits in a temp dir, the same shape as LiveCodeBench's own runner: at most 30
tests per problem (public then private), 8 s wall time per test; stdout compared exactly, line-wise with trailing
whitespace stripped, as JSON for function-style problems, or as a float within 1e-6. Writes <arm>/lcb_v6.grades.jsonl.
Test cases: data/lcb_test6.jsonl (livecodebench/code_generation_lite test6.jsonl at a pinned revision, fetch_data.sh).

    grade_lcb.py --results DIR [--workers N] <arm> ...
"""

import argparse
import base64
import json
import os
import pickle
import re
import resource
import subprocess
import sys
import tempfile
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

MAX_TESTS = 30          # cap per problem; private sets run to ~40 cases
PER_TEST_TIMEOUT = 8    # seconds
sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
from grade import final_part  # noqa: E402

SOURCES = {"lcb_v6": "lcb_test6.jsonl"}


def decode_tests(s):
    if not s:
        return []
    try:
        return json.loads(s)
    except Exception:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(s.encode("utf-8")))))


def extract_code(text):
    fp = final_part(text)
    if fp is None:
        return None
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", fp, re.S)
    return blocks[-1].strip() if blocks else None


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (4 << 30, 4 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (PER_TEST_TIMEOUT + 2, PER_TEST_TIMEOUT + 2))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))


# LiveCodeBench's own runner prepends these; LeetCode-style solutions rely on `List` etc.
PRELUDE = """import sys, math, collections, itertools, functools, heapq, bisect, string, re, random
from typing import *
from collections import *
from functools import *
from itertools import *
from heapq import *
from bisect import *
sys.setrecursionlimit(100000)
"""

FUNC_HARNESS = """
import json, sys
_inputs = [json.loads(l) for l in sys.stdin.read().split("\\n") if l.strip() != ""]
_res = Solution().{func}(*_inputs)
print(json.dumps(_res))
"""


def run_one(args):
    qid, j, code, tests, func_name = args
    if code is None:
        return dict(qid=qid, j=j, correct=0, why="no_code", n_tests=0, passed=0)
    passed = 0
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "sol.py"
        body = PRELUDE + (code if func_name is None else code + FUNC_HARNESS.format(func=func_name))
        src.write_text(body)
        for t in tests:
            inp = t["input"] if func_name is None else "\n".join(str(t["input"]).split("\n"))
            try:
                p = subprocess.run([sys.executable, "-I", str(src)], input=inp, capture_output=True,
                                   text=True, timeout=PER_TEST_TIMEOUT, cwd=td, preexec_fn=_limits,
                                   env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
            except subprocess.TimeoutExpired:
                return dict(qid=qid, j=j, correct=0, why="timeout", n_tests=len(tests), passed=passed)
            except Exception as e:
                return dict(qid=qid, j=j, correct=0, why=f"runner_error:{type(e).__name__}", n_tests=len(tests), passed=passed)
            if p.returncode != 0:
                return dict(qid=qid, j=j, correct=0, why="runtime_error", n_tests=len(tests), passed=passed)
            got, want = p.stdout.strip(), str(t["output"]).strip()
            ok = got == want or [l.rstrip() for l in got.splitlines()] == [l.rstrip() for l in want.splitlines()]
            if not ok and func_name is not None:
                try:
                    ok = json.loads(got) == json.loads(want)
                except Exception:
                    ok = False
            if not ok:
                try:  # numeric tolerance for stdout answers
                    ok = abs(float(got) - float(want)) < 1e-6
                except Exception:
                    ok = False
            if not ok:
                return dict(qid=qid, j=j, correct=0, why="wrong_answer", n_tests=len(tests), passed=passed)
            passed += 1
    return dict(qid=qid, j=j, correct=1, why="passed", n_tests=len(tests), passed=passed)


def load_problems(bench="lcb_v6"):
    probs = {}
    for r in map(json.loads, open(paths.DATA / SOURCES[bench])):
        tests = decode_tests(r["public_test_cases"]) + decode_tests(r["private_test_cases"])
        meta = json.loads(r["metadata"] or "{}")
        probs[f"lcb-{r['question_id']}"] = (tests[:MAX_TESTS], meta.get("func_name") if r["starter_code"] else None)
    return probs


def main(results, arms, workers, bench="lcb_v6"):
    probs = load_problems(bench)
    for arm in arms:
        path = Path(results) / arm / f"{bench}.jsonl"
        if not path.exists():
            print("skip", arm, flush=True)
            continue
        jobs = []
        for r in map(json.loads, open(path)):
            tests, func = probs[r["qid"]]
            jobs.append((r["qid"], r["j"], extract_code(r["text"]), tests, func))
        with ProcessPoolExecutor(workers) as ex:
            out = list(ex.map(run_one, jobs, chunksize=1))
        with open(Path(results) / arm / f"{bench}.grades.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        acc = 100 * sum(o["correct"] for o in out) / max(len(out), 1)
        print(f"{arm}/{bench}: {len(out)} samples, pass rate {acc:.1f}%", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(paths.RESULTS))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("QADX_LCB_WORKERS", "24")))
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    main(a.results, a.arms, a.workers)
