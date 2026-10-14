#!/usr/bin/env bash
# Download the evaluation files that are too large for git into eval/data/ and verify them against eval/data/SHA256SUMS.
# Idempotent: files whose sha256 already matches are not downloaded again.
#   longbench_pro.jsonl, mrcr8.jsonl   release assets (gzipped) of this repository, tag data-v1 (public download)
#   lcb_test6.jsonl                    LiveCodeBench v6 test cases (grading), Hugging Face at a pinned revision
# The two gated files (gpqa_diamond.jsonl, lcb_pro.jsonl) are rebuilt by eval/prep/*.py instead (HF_TOKEN).
set -euo pipefail
cd "$(dirname "$0")/data"
GH_REPO=${QADX_GH_REPO:-CFC87/qad-exp}
TAG=data-v1
LCB_REV=0fe84c3912ea0c4d4a78037083943e8f0c4dd505   # livecodebench/code_generation_lite

want() { awk -v f="$1" '$1 !~ /^#/ && $2 == f {print $1}' SHA256SUMS; }
ok() { [ -s "$1" ] && [ "$(sha256sum "$1" | cut -c1-64)" = "$(want "$1")" ]; }

for f in longbench_pro.jsonl mrcr8.jsonl; do
  if ok "$f"; then echo "$f: ok"; continue; fi
  curl -fsSL "https://github.com/$GH_REPO/releases/download/$TAG/$f.gz" | gunzip -c > "$f.part" && mv "$f.part" "$f"
  ok "$f" || { echo "$f: sha256 does not match SHA256SUMS" >&2; exit 1; }
  echo "$f: downloaded, ok"
done

f=lcb_test6.jsonl
if ok "$f"; then echo "$f: ok"; else
  curl -fsSL -o "$f.part" "https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/$LCB_REV/test6.jsonl"
  mv "$f.part" "$f"
  ok "$f" || { echo "$f: sha256 does not match SHA256SUMS" >&2; exit 1; }
  echo "$f: downloaded, ok"
fi

[ -s gpqa_diamond.jsonl ] || echo "gpqa_diamond.jsonl: not built yet (gated): .venv-tools/bin/python eval/prep/prep_gpqa.py" >&2
[ -s lcb_pro.jsonl ] || echo "lcb_pro.jsonl: not built yet (gated): .venv-tools/bin/python eval/prep/prep_lcbpro.py" >&2
sha256sum --quiet -c --ignore-missing SHA256SUMS && echo "all present files match SHA256SUMS"
