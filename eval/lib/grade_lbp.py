"""Score LongBench Pro generations with the official metric functions (runs in .venv-tools: jieba, rouge, pytrec_eval).

    grade_lbp.py --results DIR <arm> ...     <arm>/longbench_pro.jsonl -> <arm>/longbench_pro.grades.jsonl

The prediction is the post-</think> answer, which is what the official harness scores (message.content). Each item's
metric (NDCG, Pairwise_Accuracy, Accuracy, F1_Score or SubEM) was read from the official evaluation.py when the prompt
file was built; all of them parse the text after the last "[Answer]" marker. An empty prediction scores 0, as in the
official calculate_metric. Scores are in [0, 1] and averaged like accuracies.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paths  # noqa: E402

sys.path.insert(0, str(paths.LBP_MODULES))
import utils as lbp  # noqa: E402  (official metrics)

SPECIAL = ("<|im_end|>", "<|endoftext|>")


def prediction(text):
    if "</think>" not in text:
        return ""
    out = text.split("</think>")[-1]
    for s in SPECIAL:
        out = out.replace(s, "")
    return out.strip()


def main(results, arms, stem="longbench_pro"):
    meta = {r["qid"]: r for r in map(json.loads, open(paths.DATA / "longbench_pro.jsonl"))}
    for arm in arms:
        out = []
        for r in map(json.loads, open(Path(results) / arm / f"{stem}.jsonl")):
            it, pred = meta[r["qid"]], prediction(r["text"])
            if not pred:
                score, why = 0.0, "no_final"
            else:
                score = float(getattr(lbp, it["metric"])(it["answer"], pred))
                assert 0.0 <= score <= 1.0, (r["qid"], score)
                why = it["metric"]
            out.append(dict(qid=r["qid"], j=r["j"], correct=score, why=why))
        with open(Path(results) / arm / f"{stem}.grades.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        print(f"{arm}/{stem}: {100 * sum(o['correct'] for o in out) / max(len(out), 1):.1f} ({len(out)} samples)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(paths.RESULTS))
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args()
    main(a.results, a.arms)
