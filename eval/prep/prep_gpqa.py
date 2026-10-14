"""Build eval/data/gpqa_diamond.jsonl from Idavidrein/gpqa (gated: accept its terms on Hugging Face, export HF_TOKEN).

GPQA's license asks not to redistribute the questions in plain text, so the file is rebuilt here instead of committed.
The four options are shuffled with a per-question seed (sha256 of the Record ID), followed by the Qwen3 card's
multiple-choice instruction; the output is byte-identical to the reference build (sha256 in eval/data/SHA256SUMS).
Run with .venv-tools:  .venv-tools/bin/python eval/prep/prep_gpqa.py [--out FILE]
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from datasets import load_dataset

REPO = ("Idavidrein/gpqa", "83022cefff930aea54f654c0b282e74b9eeda5c6")
OUT = Path(__file__).resolve().parents[1] / "data" / "gpqa_diamond.jsonl"
MCQ_SUFFIX = 'Please show your choice in the `answer` field with only the choice letter, e.g., `"answer": "C"`.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    rows = []
    for r in load_dataset(REPO[0], "gpqa_diamond", split="train", revision=REPO[1]):
        options = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
        options = [o.strip() for o in options]
        seed = int(hashlib.sha256(r["Record ID"].encode()).hexdigest(), 16) % (2**32)
        order = list(range(4))
        random.Random(seed).shuffle(order)
        letters = "ABCD"
        body = "\n".join(f"{letters[i]}) {options[j]}" for i, j in enumerate(order))
        gold = letters[order.index(0)]
        rows.append({"qid": f"gpqa-{r['Record ID']}", "prompt": f"{r['Question'].strip()}\n\n{body}\n\n{MCQ_SUFFIX}",
                     "answer": gold, "type": "mcq"})
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{a.out}: {len(rows)} questions, sha256 {hashlib.sha256(open(a.out, 'rb').read()).hexdigest()}")


if __name__ == "__main__":
    main()
