"""Grading of one arm's benchmarks: every benchmark ends up with <arm>/<bench>.grades.jsonl rows {qid, j, correct, why}.

Graded here (.venv-client): competitions (math), GPQA / MMLU-Redux (multiple choice), MRCR. Dispatched to their own
graders and environments: LiveCodeBench v6 (grade_lcb.py), LiveCodeBench Pro (grade_lcbpro.py, Docker judge), IFBench
(grade_ifbench.py, IFBench's env), LongBench Pro (grade_lbp.py, .venv-tools). tau2 and BFCL are graded by run_phase.py
(tau2_convert.py / bfcl_to_rows.py) when their phase ends. A grades file is rewritten whenever its generations are newer.

    grade.py --results DIR --arm A [--benches b1,b2 | --phase S2] [--force]
"""

import argparse
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

MATH = ("aime26", "hmmt_feb26", "hmmt_feb25", "hmmt_nov25")
MCQ = ("gpqa_diamond", "mmlu_redux_1k")
WORKERS = int(os.environ.get("QADX_GRADE_WORKERS", "16"))


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def final_part(text):
    """The answer after the reasoning: text after the last </think>, without end-of-turn tokens (None: no </think>)."""
    if "</think>" not in text:
        return None
    return text.split("</think>")[-1].replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        if s[k] == "{":
            depth += 1
        elif s[k] == "}":
            depth -= 1
            if depth == 0:
                return s[j + 1 : k]
    return None


def norm(s):
    s = s.strip().strip("$").strip()
    for a, b in [("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\ ", ""), (" ", "")]:
        s = s.replace(a, b)
    return s.rstrip(".")


def grade_math(args):
    """Last \\boxed{} after </think>: normalized exact match, else math-verify equivalence."""
    text, gold = args
    fp = final_part(text)
    if fp is None:
        return 0, "no_final"
    pred = last_boxed(fp)
    if pred is None:
        return 0, "no_boxed"
    if norm(pred) == norm(gold):
        return 1, "exact"
    # imported outside the try: a missing math_verify must fail loudly, not grade every non-verbatim answer wrong
    from math_verify import parse, verify

    try:
        g = parse(f"${gold}$", parsing_timeout=5)
        p = parse(f"$\\boxed{{{pred}}}$", parsing_timeout=5)
        if g and p and verify(g, p, timeout_seconds=5):
            return 1, "math_verify"
    except Exception:
        pass
    return 0, "wrong"


def grade_mcq(text, gold):
    fp = final_part(text)
    if fp is None:
        return 0, "no_final"
    # The model writes the requested JSON field loosely: `answer": "C"`, `answer`: "C", `answer: "C"`, **Answer: C**.
    quote = r"[`\"'\*\s]*"
    pats = [
        rf"answer{quote}(?:field{quote})?[:：=]?{quote}\(?([A-J])(?![A-Za-z])",
        rf"answer\s+is{quote}\(?([A-J])(?![A-Za-z])",
        # boxed letter, also as Qwen3 writes it: \boxed{\text{C}}, \boxed{\text{D)}\n}, \boxed{\textbf{A}}
        r"\\boxed\{\s*(?:\\(?:text|textbf|mathrm|mathbf)\{\s*)?\(?([A-J])\s*[).:]?\s*\}?\s*\}",
        r"\n\s*\**\(?([A-J])\)?\**\s*$",
    ]
    hits = [(m.start(), m.group(1)) for p in pats for m in re.finditer(p, fp, re.I)]
    if hits:
        return int(max(hits)[1] == gold), "parsed"
    hits = [(m.start(), m.group(1)) for p in MCQ_FALLBACK for m in re.finditer(p, fp)]
    if hits:
        return int(max(hits)[1] == gold), "parsed_fallback"
    return 0, "no_answer"


# Answer forms the patterns above miss. Tried only when they find nothing, so no answer they parse changes; the letter
# must be an upper-case option label written as the answer.
MCQ_FALLBACK = [
    # \boxed{\text{answer}: "C"}, $$\text{answer}: \text{D}$$, \text{answer}} = \text{B}, \text{answer: } \text{C}
    r"\\text\{\s*(?i:answer)\s*(?:\}\s*)*[:：=]\s*\}?\s*(?:\\(?:text|textbf|mathrm|boxed)\{\s*)?[\"'`]?\(?([A-J])(?![A-Za-z0-9])",
    # option label plus option text in the box: \boxed{\text{A) 0.85}}, \boxed{B) 4}, \boxed{\text{B: The mass ...}}
    r"\\boxed\{\s*(?:\\(?:text|textbf|mathrm|mathbf)\{\s*)?\(?([A-J])(?:\)|:\s)",
    # $$\text{B) 4}$$ without a box
    r"\$\$\s*\\text\{\s*\(?([A-J])\)",
    # **Answer:** ```text\nB\n```
    r"(?i:answer)[:*\s]*```[a-z]*\s*\(?([A-J])\)?\s*```",
    # quoted letter in the box: \boxed{"C"}, \boxed{\text{"D"}}, \boxed{"C) 50%"}
    r"\\boxed\{\s*(?:\\(?:text|textbf|mathrm|mathbf)\{\s*)?[\"'`]\(?([A-J])(?:\)|[\"'`])",
    # JSON with escaped quotes: {\"answer\": \"D\"}
    r"(?i:answer)\\[\"']\s*[:：]\s*\\[\"']\(?([A-J])(?![A-Za-z0-9])",
    # "the answer is: C", "The correct choice is: "A""
    r"(?i:answer|choice)\s+is\s*[:：]\s*[\"'`*]*\(?([A-J])(?![A-Za-z0-9])",
    # the final part is only the (quoted) letter or "letter) option text": "D", C, "D) Health care"
    r"^\s*[\"'`*]*\(?([A-J])(?:\)[^\n]*)?[\"'`*]*\s*$",
    # a quoted letter closing the answer: ... Only Statement I is true. "A"
    r"[\"']\(?([A-J])\)?[\"']\s*\.?\s*$",
]


def grade_mrcr(text, item):
    """OpenAI MRCR: 0 unless the answer starts with the required random prefix, else the difflib SequenceMatcher ratio
    against the reference, both with the prefix stripped."""
    from difflib import SequenceMatcher

    fp = final_part(text)
    if fp is None or not fp.startswith(item["prefix"]):
        return 0.0, "no_prefix"
    return SequenceMatcher(None, fp[len(item["prefix"]):], item["answer"][len(item["prefix"]):]).ratio(), "scored"


def grade_inline(recs, bench, workers=WORKERS):
    """[(correct, why)] for a math / MCQ / MRCR benchmark."""
    if bench in MATH:
        with ProcessPoolExecutor(workers) as ex:
            return list(ex.map(grade_math, [(r["text"], r["answer"]) for r in recs], chunksize=8))
    if bench in MCQ:
        return [grade_mcq(r["text"], r["answer"]) for r in recs]
    if bench == "mrcr8":
        meta = {it["qid"]: it for it in load_jsonl(paths.DATA / f"{bench}.jsonl")}
        return [grade_mrcr(r["text"], meta[r["qid"]]) for r in recs]
    raise ValueError(bench)


def stale(results, arm, bench):
    g, gp = Path(results) / arm / f"{bench}.jsonl", Path(results) / arm / f"{bench}.grades.jsonl"
    return g.exists() and (not gp.exists() or g.stat().st_mtime > gp.stat().st_mtime)


def grade(results, arm, benches, force=False, workers=WORKERS, lcbpro=True):
    """Grade the given benchmarks of one arm; returns {bench: 'graded' | 'up to date' | 'missing' | 'skipped' | 'failed'}."""
    results = Path(results)
    status = {}
    for bench in benches:
        raw = results / arm / f"{bench}.jsonl"
        if bench.startswith("tau2_") or bench == "bfcl_mt":
            status[bench] = "graded" if (results / arm / f"{bench}.grades.jsonl").exists() else "missing"
            continue
        if not raw.exists():
            status[bench] = "missing"
            continue
        if not (force or stale(results, arm, bench)):
            status[bench] = "up to date"
            continue
        cmd = None
        if bench in MATH or bench in MCQ or bench == "mrcr8":
            recs = load_jsonl(raw)
            grades = grade_inline(recs, bench, workers)
            with open(results / arm / f"{bench}.grades.jsonl", "w") as f:
                for r, (c, why) in zip(recs, grades):
                    f.write(json.dumps(dict(qid=r["qid"], j=r["j"], correct=c, why=why)) + "\n")
            status[bench] = "graded"
            continue
        if bench == "lcb_v6":
            cmd = [sys.executable, str(paths.LIB / "grade_lcb.py"), "--results", str(results), arm]
        elif bench == "ifbench":
            cmd = [str(paths.IFBENCH_PY), str(paths.LIB / "grade_ifbench.py"), "--results", str(results), arm]
        elif bench == "longbench_pro":
            cmd = [str(paths.TOOLS_PY), str(paths.LIB / "grade_lbp.py"), "--results", str(results), arm]
        elif bench == "lcb_pro":
            if not lcbpro:
                status[bench] = "skipped"
                continue
            from grade_lcbpro import IMAGE  # noqa: PLC0415
            if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode:
                print(f"lcb_pro: judge image {IMAGE} missing (eval/setup_judge.sh), not grading", file=sys.stderr)
                status[bench] = "skipped"
                continue
            cmd = [sys.executable, str(paths.LIB / "grade_lcbpro.py"), "--results", str(results), arm]
        r = subprocess.run(cmd, check=False)
        status[bench] = "graded" if r.returncode == 0 and (results / arm / f"{bench}.grades.jsonl").exists() else "failed"
    return status


if __name__ == "__main__":
    import phases as P

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(paths.RESULTS))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--benches", default="")
    ap.add_argument("--phase", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-lcbpro", action="store_true", help="leave LiveCodeBench Pro ungraded (judge needs an idle machine)")
    a = ap.parse_args()
    benches = [b for b in a.benches.split(",") if b] or (P.benches(a.phase) if a.phase else
                                                         [b for ph in P.ORDER for b in P.benches(ph)])
    st = grade(a.results, a.arm, benches, a.force, lcbpro=not a.no_lcbpro)
    print(json.dumps({a.arm: st}))
    sys.exit(1 if "failed" in st.values() else 0)
