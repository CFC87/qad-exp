"""Build eval/data/lcb_pro.jsonl: LiveCodeBench Pro, 2025 Q2 split (Codeforces, Apr-Jun 2025), problems with tests.

The newest split (2025 Q3) has no public test cases, so Q2 is the newest one that can be judged locally. The problem
set is gated on Hugging Face (accept the terms with the account of HF_TOKEN), so it is rebuilt here rather than
committed; both repositories are pinned to a revision and the output is byte-identical to the reference build (sha256
in eval/data/SHA256SUMS). The prompt is the official harness's prompt, read verbatim from
third_party/LiveCodeBench-Pro/api_interface.py, followed by the problem statement, as its benchmark.py does.
Run with .venv-tools:  .venv-tools/bin/python eval/prep/prep_lcbpro.py [--out FILE]
"""

import argparse
import ast
import glob
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, snapshot_download

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "eval" / "data" / "lcb_pro.jsonl"
PROBLEMS = ("QAQAQAQAQ/LiveCodeBench-Pro", "adebffce047dddb7768a86bace6aea4f7425e3bc")
TESTCASES = ("QAQAQAQAQ/LiveCodeBench-Pro-Testcase", "5257736c0a4e30ba0949d41c56a257c323d9c600")
SPLIT = "quater_2025_4_6"


def official_prompt():
    src = (REPO_ROOT / "third_party/LiveCodeBench-Pro/api_interface.py").read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(ast.unparse(t) == "self.prompt" for t in node.targets):
            return node.value.value
    raise RuntimeError("official prompt not found")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    prompt = official_prompt()
    d = snapshot_download(PROBLEMS[0], repo_type="dataset", revision=PROBLEMS[1], allow_patterns=f"data/{SPLIT}-*")
    df = pd.read_parquet(glob.glob(f"{d}/data/{SPLIT}-*.parquet")[0])
    tests = {f[:-4] for f in HfApi().list_repo_files(TESTCASES[0], repo_type="dataset", revision=TESTCASES[1]) if f.endswith(".zip")}
    rows = [dict(qid=f"lcbpro-{r.problem_id}", prompt=prompt + r.problem_statement, answer="", type="lcbpro",
                 problem_id=r.problem_id, difficulty=r.difficulty, time_limit_ms=int(r.time_limit),
                 memory_limit_mb=int(r.memory_limit), split=SPLIT)
            for r in df.itertuples() if r.problem_id in tests]
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(rows)} of {len(df)} {SPLIT} problems have tests, "
          f"sha256 {hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
