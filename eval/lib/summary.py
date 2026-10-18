"""Scores per arm, by benchmark and by report category, from the graded files -> <results>/summary.{md,json}.

Benchmark score = mean over questions of the per-question mean over its samples (k), in %. LiveCodeBench Pro leaves out
the interactive problems in data/lcbpro_excluded.json; tau2 pools the tasks of its three domains; a sample with no
grade row counts as 0. Categories (equal-weight means of their benchmark scores):
    reasoning   AIME 2026, HMMT Feb 2025, HMMT Feb 2026, HMMT Nov 2025, GPQA Diamond
    code        LiveCodeBench v6, LiveCodeBench Pro
    long        LongBench Pro, MRCR 8-needle
    bfcl        BFCL multi-turn
    tau2        tau2 (airline + retail + telecom tasks pooled)
MMLU-Redux 1k and IFBench are no-damage checks, reported but in no category. Comparisons against the PTQ / BF16
baselines (recovery = (arm - PTQ) / (BF16 - PTQ)) are made per question from the grades files, not here.

The first section of summary.md, "Report", holds the scores to report: the five category scores and every benchmark
score to two decimals, one column per arm, with the repository commit, the evaluation spec and whether every phase is
complete under it and graded. Recovery against the baselines is computed from these category scores.

    summary.py --results DIR [--arms a,b] [--report]     (--report: print only the Report for these arms, write nothing)
"""

import argparse
import json
import statistics as st
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402
import phases as P  # noqa: E402

COMPS = ["aime26", "hmmt_feb25", "hmmt_feb26", "hmmt_nov25"]
TAU2 = ["tau2_airline", "tau2_retail", "tau2_telecom"]
CATS = {"reasoning": COMPS + ["gpqa_diamond"], "code": ["lcb_v6", "lcb_pro"], "long": ["longbench_pro", "mrcr8"],
        "bfcl": ["bfcl_mt"], "tau2": ["tau2"]}
CHECKS = ["mmlu_redux_1k", "ifbench"]
BENCHES = COMPS + ["gpqa_diamond", "mmlu_redux_1k", "ifbench", "lcb_v6", "lcb_pro", "longbench_pro", "mrcr8"] + TAU2 + ["bfcl_mt"]
GROUPS = {"tau2": TAU2, "comps": COMPS}


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def samples(results, arm):
    """{bench: {qid: [(j, correct, truncated, completion_tokens), ...]}} from the generations and their grades."""
    out = {}
    excluded = set(json.load(open(paths.DATA / "lcbpro_excluded.json")))
    for bench in BENCHES:
        p = Path(results) / arm / f"{bench}.jsonl"
        if not p.exists():
            continue
        recs = load_jsonl(p)
        if bench == "lcb_pro":  # interactive problems packaged as default tasks: the checker rejects the reference answer
            recs = [r for r in recs if r["qid"].removeprefix("lcbpro-") not in excluded]
        gp = Path(results) / arm / f"{bench}.grades.jsonl"
        gm = {(g["qid"], g["j"]): g for g in load_jsonl(gp)} if gp.exists() else {}
        d = defaultdict(list)
        for r in recs:
            c = gm.get((r["qid"], r["j"]), {}).get("correct", 0)
            d[r["qid"]].append((r.get("j", 0), float(c), r["finish_reason"] == "length", r["completion_tokens"]))
        out[bench] = {q: sorted(v) for q, v in d.items()}
        out[bench + "@graded"] = bool(gm)
    return out


def qmeans(s, bench):
    return {f"{b}|{q}": st.fmean(x[1] for x in v) for b in GROUPS.get(bench, [bench]) for q, v in s.get(b, {}).items() if v}


def score(s, bench):
    m = qmeans(s, bench)
    return 100 * st.fmean(m.values()) if m else None


def diag(s, bench):
    xs = [x for b in GROUPS.get(bench, [bench]) for v in s.get(b, {}).values() for x in v]
    if not xs:
        return None
    return dict(samples=len(xs), questions=sum(len(s.get(b, {})) for b in GROUPS.get(bench, [bench])),
                truncated_pct=100 * st.fmean(x[2] for x in xs), mean_tokens=st.fmean(x[3] for x in xs))


def arm_summary(results, arm):
    s = samples(results, arm)
    benches = {b: dict(score=score(s, b), graded=all(s.get(m + "@graded", False) for m in GROUPS.get(b, [b]) if m in s),
                       **diag(s, b)) for b in BENCHES + ["comps", "tau2"] if diag(s, b)}
    cats = {}
    for c, bs in CATS.items():
        sc = [score(s, b) for b in bs]
        cats[c] = st.fmean(sc) if all(x is not None for x in sc) else None
    status = {}
    for ph in P.ORDER:
        f = Path(results) / arm / f"{ph}.status.json"
        if f.exists():
            d = json.load(open(f))  # complete under the current spec only (a status without tasks was written under v1)
            status[ph] = d.get("complete") is True and d.get("tasks", P.V1_TASKS.get(ph, P.PHASES[ph]["tasks"])) == P.PHASES[ph]["tasks"]
        else:
            status[ph] = None
    return dict(categories=cats, benchmarks=benches, phases=status)


def commit():
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=paths.REPO, capture_output=True, text=True).stdout.strip()
        return f"{h} ({'local changes' if dirty else 'clean'})" if h else "unknown"
    except OSError:
        return "unknown"


def report(out):
    """Rows = categories and benchmarks (k = samples per question under the current spec), columns = arms; last row:
    every phase complete under the current spec and every benchmark graded."""
    k = {sp.split(":")[0]: sp.split(":")[1] for ph in P.ORDER for sp in P.specs(ph)}
    k["tau2"] = k["tau2_retail"]
    f2 = lambda v: "—" if v is None else f"{v:.2f}"  # noqa: E731
    arms = list(out)
    L = ["## Report", "", f"Commit {commit()} · evaluation spec v{P.SPEC_VERSION} · scores in %, two decimals", "",
         "| | " + " | ".join(arms) + " |", "|---|" + "---|" * len(arms)]
    for c in CATS:
        L.append(f"| **{c}** | " + " | ".join(f2(out[x]["categories"][c]) for x in arms) + " |")
    for b in COMPS + ["gpqa_diamond", "lcb_v6", "lcb_pro", "longbench_pro", "mrcr8", "bfcl_mt", "tau2"] + CHECKS:
        L.append(f"| {b} (k={k[b]}) | " + " | ".join(f2(out[x]["benchmarks"].get(b, {}).get("score")) for x in arms) + " |")
    ok = []
    for x in arms:
        bad = [p for p in P.ORDER if out[x]["phases"][p] is not True] + [b for b, d in out[x]["benchmarks"].items() if not d["graded"]]
        ok.append("yes" if not bad else "NO: " + ", ".join(bad))
    L.append("| complete + graded | " + " | ".join(ok) + " |")
    return L + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(paths.RESULTS))
    ap.add_argument("--arms", default="")
    ap.add_argument("--report", action="store_true", help="print only the Report for these arms; write nothing")
    a = ap.parse_args()
    root = Path(a.results)
    arms = [x for x in a.arms.split(",") if x] or sorted(p.name for p in root.iterdir() if p.is_dir() and p.name != "logs")
    out = {x: arm_summary(root, x) for x in arms if (root / x).is_dir()}
    if a.report:
        print("\n".join(report(out)))
        return
    f = lambda v: "—" if v is None else f"{v:.1f}"  # noqa: E731
    L = ["# Scores", ""] + report(out) + ["## Categories", "",
         "Benchmark score = mean over questions of the per-question mean over samples (%). Categories are "
         "equal-weight means of their benchmarks (eval/lib/summary.py).", "",
         "| arm | " + " | ".join(CATS) + " | " + " | ".join(CHECKS) + " |", "|---|" + "---|" * (len(CATS) + len(CHECKS))]
    for x, v in out.items():
        L.append(f"| {x} | " + " | ".join(f(v["categories"][c]) for c in CATS) + " | "
                 + " | ".join(f(v["benchmarks"].get(b, {}).get("score")) for b in CHECKS) + " |")
    L += ["", "## By benchmark", "", "| arm | benchmark | score | questions | samples | truncated % | mean output tokens | graded |",
          "|---|---|---|---|---|---|---|---|"]
    for x, v in out.items():
        for b, d in v["benchmarks"].items():
            L.append(f"| {x} | {b} | {f(d['score'])} | {d['questions']} | {d['samples']} | {d['truncated_pct']:.1f} | "
                     f"{d['mean_tokens']:.0f} | {'yes' if d['graded'] else 'NO'} |")
    L += ["", "## Phases complete", "", "| arm | " + " | ".join(P.ORDER) + " |", "|---|" + "---|" * len(P.ORDER)]
    for x, v in out.items():
        L.append(f"| {x} | " + " | ".join({True: "yes", False: "INCOMPLETE", None: "—"}[v["phases"][p]] for p in P.ORDER) + " |")
    (root / "summary.md").write_text("\n".join(L) + "\n")
    json.dump(out, open(root / "summary.json", "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
