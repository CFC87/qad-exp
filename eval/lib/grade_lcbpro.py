"""Grade LiveCodeBench Pro generations with the official judge (LightCPVerifier + go-judge in Docker; .venv-client).

    grade_lcbpro.py --results DIR <arm> ...     <arm>/lcb_pro.jsonl -> <arm>/lcb_pro.grades.jsonl

The code is taken from the post-</think> answer with the official extractor (third_party/LiveCodeBench-Pro/util.py:
the last ```cpp block containing #include, else the last brace-matched int main). Test data come from the
QAQAQAQAQ/LiveCodeBench-Pro-Testcase zips at a pinned revision (testlib checker, time and memory limits in
config.yaml; gated: needs HF_TOKEN). The judge image is built by eval/setup_judge.sh from the pinned LightCPVerifier with
go-judge pinned to v1.9.6. The judge measures CPU time, but keep the machine otherwise idle while grading to avoid
spurious time-limit failures (run.py grades LiveCodeBench Pro only after all generation jobs have finished).
The 5 interactive problems listed in data/lcbpro_excluded.json are graded but left out of the score (summary.py).
"""

import argparse
import json
import os
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

sys.path.insert(0, str(paths.LCBPRO_DIR))
from util import extract_longest_cpp_code  # noqa: E402  (official extractor)

IMAGE = "lightcpverifier:qadx"
CONTAINER = f"{os.environ.get('QADX_CONTAINER_PREFIX', 'qadx')}_lcbpro_judge"
JUDGE_DIR = paths.CACHE / "lcbpro_judge"
WORKERS = int(os.environ.get("LCBPRO_JUDGE_WORKERS", "16"))
TESTCASE_REPO = "QAQAQAQAQ/LiveCodeBench-Pro-Testcase"
TESTCASE_REV = "5257736c0a4e30ba0949d41c56a257c323d9c600"
SPECIAL = ("<|im_end|>", "<|endoftext|>")


def final_part(text):
    if "</think>" not in text:
        return None
    out = text.split("</think>")[-1]
    for s in SPECIAL:
        out = out.replace(s, "")
    return out


def extract(text):
    """(code or None, reason or None) for one generation."""
    fp = final_part(text)
    code = extract_longest_cpp_code(fp) if fp is not None else None
    return code, ("no_final" if fp is None else ("no_code" if not code else None))


def ensure_problem(pid):
    d = JUDGE_DIR / "problems" / pid
    if not (d / "config.yaml").exists():
        d.mkdir(parents=True, exist_ok=True)
        zipfile.ZipFile(hf_hub_download(TESTCASE_REPO, f"{pid}.zip", repo_type="dataset", revision=TESTCASE_REV)).extractall(d)


def start_judge():
    for sub in ("problems", "submissions", "data"):
        (JUDGE_DIR / sub).mkdir(parents=True, exist_ok=True)
    subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)
    subprocess.run(["docker", "run", "-d", "--name", CONTAINER, "--privileged", "--shm-size", "4g",
                    "-e", f"JUDGE_WORKERS={WORKERS}", "-e", f"GJ_PARALLELISM={WORKERS}",
                    "-v", f"{JUDGE_DIR}/problems:/app/problems", "-v", f"{JUDGE_DIR}/submissions:/app/submissions",
                    "-v", f"{JUDGE_DIR}/data:/app/data", "-p", "127.0.0.1::8081", IMAGE], check=True, capture_output=True)
    port = subprocess.run(["docker", "port", CONTAINER, "8081"], capture_output=True, text=True, check=True).stdout.split(":")[-1].strip()
    base = f"http://127.0.0.1:{port}"
    for _ in range(60):
        try:
            if requests.get(f"{base}/health", timeout=2).ok:
                return base
        except requests.RequestException:
            pass
        time.sleep(1)
    raise RuntimeError("judge did not become healthy: " + subprocess.run(["docker", "logs", CONTAINER], capture_output=True, text=True).stdout[-2000:])


def judge_one(base, pid, code):
    sid = requests.post(f"{base}/submit", json={"pid": pid, "lang": "cpp", "code": code}, timeout=60).json()["sid"]
    while True:
        r = requests.get(f"{base}/result/{sid}", timeout=60)
        if r.status_code == 200:
            j = r.json()
            if j.get("status") == "done":
                return j.get("result") or "unknown"
            if j.get("status") == "error":  # compile errors come back as status=error, as in the official harness
                return "compile_error_or_judge_error"
        time.sleep(1)


def main(results, arms):
    meta = {r["qid"]: r for r in map(json.loads, open(paths.DATA / "lcb_pro.jsonl"))}
    base = start_judge()
    try:
        for arm in arms:
            recs = [json.loads(l) for l in open(Path(results) / arm / "lcb_pro.jsonl")]
            for pid in {meta[r["qid"]]["problem_id"] for r in recs}:
                ensure_problem(pid)
            jobs = [(r, *extract(r["text"])) for r in recs]
            with ThreadPoolExecutor(WORKERS) as ex:
                verdicts = list(ex.map(lambda x: x[2] or judge_one(base, meta[x[0]["qid"]]["problem_id"], x[1]), jobs))
            out = [dict(qid=r["qid"], j=r["j"], correct=int(v == "Accepted"), why=v) for (r, _, _), v in zip(jobs, verdicts)]
            with open(Path(results) / arm / "lcb_pro.grades.jsonl", "w") as f:
                for o in out:
                    f.write(json.dumps(o) + "\n")
            print(f"{arm}/lcb_pro: pass@1 {100 * sum(o['correct'] for o in out) / max(len(out), 1):.1f} ({len(out)} samples)")
    finally:
        subprocess.run(["docker", "rm", "-f", CONTAINER], capture_output=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(paths.RESULTS))
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    main(a.results, a.arms)
