"""Grade IFBench with its own evaluation_lib (runs in third_party/IFBench/.venv).

Only the post-</think> response is graded (IFBench: "process the output to extract the answer without the reasoning
chains"). Score: prompt-level LOOSE accuracy, the one the IFBench paper reports; strict and instruction-level are kept
alongside. IFBench downloads its NLTK data (punkt, punkt_tab, stopwords, averaged_perceptron_tagger_eng) on first use.

  grade_ifbench.py --results DIR <arm> ...    <arm>/ifbench.jsonl -> <arm>/ifbench.grades.jsonl
  grade_ifbench.py --selftest                 re-grade IFBench's sample outputs and diff against its eval files
"""

import argparse
import copy
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

IFB = paths.IFBENCH_DIR
sys.path.insert(0, str(IFB))
import evaluation_lib as E  # noqa: E402
import langdetect  # noqa: E402

SPECIAL = ("<|im_end|>", "<|endoftext|>")


def final_part(text):
    if "</think>" not in text:
        return None
    out = text.split("</think>")[-1]
    for s in SPECIAL:
        out = out.replace(s, "")
    return out.strip()


def grade_one(row, response):
    res = {}
    for name, fn in (("strict", E.test_instruction_following_strict), ("loose", E.test_instruction_following_loose)):
        # The official run grades strict first, which drops None-valued kwargs in place; the loose checker relies on
        # that. Drop them up front so each call sees what the official run sees.
        kwargs = [{k: v for k, v in kw.items() if v is not None} for kw in copy.deepcopy(row["kwargs"])]
        inp = E.InputExample(key=row["key"], instruction_id_list=row["instruction_id_list"],
                             prompt=row["prompt"], kwargs=kwargs)
        random.seed(0)
        langdetect.DetectorFactory.seed = 0
        out = fn(inp, {row["prompt"]: response})
        res[name] = (int(out.follow_all_instructions), out.follow_instruction_list)
    return res


def selftest():
    rows = {r["prompt"]: r for r in map(json.loads, open(IFB / "data" / "IFBench_test.jsonl"))}
    samples = [json.loads(l) for l in open(IFB / "data" / "sample_output.jsonl")]
    ref = {m: {r["prompt"]: r for r in map(json.loads, open(IFB / "eval" / f"eval_results_{m}.jsonl"))} for m in ("strict", "loose")}
    mism = n = 0
    for s in samples:
        if s["prompt"] not in ref["strict"] or s["prompt"] not in rows:
            continue
        n += 1
        g = grade_one(rows[s["prompt"]], s["response"])
        for m in ("strict", "loose"):
            r = ref[m][s["prompt"]]
            if bool(g[m][0]) != r["follow_all_instructions"] or g[m][1] != r["follow_instruction_list"]:
                mism += 1
    acc = {m: sum(r["follow_all_instructions"] for r in ref[m].values()) / len(ref[m]) for m in ref}
    print(f"selftest: {n} samples, {mism} mismatches vs official eval files; official prompt-level strict {acc['strict']:.3f} loose {acc['loose']:.3f}")
    return mism == 0


def main(results, arms):
    rows = {f"ifbench-{r['key']}": r for r in map(json.loads, open(IFB / "data" / "IFBench_test.jsonl"))}
    for arm in arms:
        path = Path(results) / arm / "ifbench.jsonl"
        if not path.exists():
            continue
        out = []
        for r in map(json.loads, open(path)):
            resp = final_part(r["text"])
            if resp is None:
                out.append(dict(qid=r["qid"], j=r["j"], correct=0, strict=0, inst_loose=0.0, why="no_final"))
                continue
            g = grade_one(rows[r["qid"]], resp)
            il = g["loose"][1]
            out.append(dict(qid=r["qid"], j=r["j"], correct=g["loose"][0], strict=g["strict"][0],
                            inst_loose=sum(il) / max(len(il), 1), why="graded"))
        with open(Path(results) / arm / "ifbench.grades.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        n = max(len(out), 1)
        print(f"{arm}/ifbench: prompt loose {100 * sum(o['correct'] for o in out) / n:.1f}, strict {100 * sum(o['strict'] for o in out) / n:.1f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(paths.RESULTS))
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("arms", nargs="*")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    main(a.results, a.arms)
