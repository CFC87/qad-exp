#!/usr/bin/env bash
# Install everything pinned in env/versions.toml. Idempotent: rerun safely.
#
#   setup/setup.sh                 # all steps except optional models
#   setup/setup.sh images venvs    # only some steps
#
# Steps: uv, images, sender, sources, venvs, models, data, eval-data, judge.  Add "optional-models" to also fetch
# models marked optional.
# Prerequisites: Ubuntu 24.04, NVIDIA driver with CUDA 13.0, docker + NVIDIA Container Toolkit, git, curl, python3 >= 3.11.
# The NeMo image is on nvcr.io: run `docker login nvcr.io` (user $oauthtoken, an NGC API key) first if the pull is refused.
# The gated evaluation sets (GPQA, LiveCodeBench Pro) need `export HF_TOKEN=...` of an account that accepted their
# terms on Hugging Face; everything else is downloaded without credentials.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
V() { python3 "$REPO/tools/versions.py" "$@"; }
step() { echo; echo "=== $*"; }

# Keep every tool, cache and download inside the repository (only the docker images live outside it). A Hugging Face
# token saved by `hf auth login` in the default location stays usable; HF_TOKEN in the environment takes precedence.
export UV_PYTHON_INSTALL_DIR=$REPO/.tools/python UV_CACHE_DIR=$REPO/.cache/uv
if [ -z "${HF_HOME:-}" ]; then
  [ -z "${HF_TOKEN:-}" ] && [ -z "${HF_TOKEN_PATH:-}" ] && [ -f "$HOME/.cache/huggingface/token" ] && export HF_TOKEN_PATH=$HOME/.cache/huggingface/token
  export HF_HOME=$REPO/.cache/huggingface
fi
export NLTK_DATA=$REPO/.cache/nltk_data
UV=$REPO/.tools/uv/uv
STEPS=${*:-uv images sender sources venvs models data eval-data judge}
has() { [[ " $STEPS " == *" $1 "* ]]; }

if has uv; then
  step "uv $(V get tools.uv) and CPython $(V get tools.python) (repo-local, under .tools/)"
  want=$(V get tools.uv)
  if [ ! -x "$UV" ] || [ "$($UV --version | awk '{print $2}')" != "$want" ]; then
    # unmanaged install: the binary goes to .tools/uv, no receipt, no PATH change
    curl -LsSf "https://astral.sh/uv/$want/install.sh" | env UV_UNMANAGED_INSTALL="$REPO/.tools/uv" sh
  fi
  $UV python install --no-bin "$(V get tools.python)"
fi

if has images; then
  for i in vllm nemo; do
    ref=$(V image $i); tag="$(V get images.$i.ref):$(V get images.$i.tag)"
    step "image $i: $ref"
    docker image inspect "$ref" >/dev/null 2>&1 || docker pull "$ref"
    docker tag "$ref" "$tag"
  done
fi

if has sender; then
  # The trainer pushes weights to the vLLM servers with vLLM's own NCCL weight-transfer code; take that package from
  # the pinned serving image itself, so sender and receivers are the same build.
  step "vLLM weight-transfer sender package from the vllm image -> .cache/vllm-sender"
  if [ ! -f .cache/vllm-sender/vllm/_version.py ]; then
    mkdir -p .cache/vllm-sender
    cid=$(docker create "$(V image vllm)")
    docker cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm" .cache/vllm-sender/vllm
    docker rm -f "$cid" >/dev/null
  fi
fi

if has sources; then
  mkdir -p third_party
  for s in $(V keys sources); do
    url=$(V get sources.$s.url); sha=$(V get sources.$s.commit)
    step "source $s @ ${sha:0:12}"
    if [ ! -d "third_party/$s/.git" ]; then
      if [ "$s" = longcontext ]; then
        git clone -q --filter=blob:none --sparse "$url" "third_party/$s"
        git -C "third_party/$s" sparse-checkout set "$(V get sources.$s.sparse)"
      else
        git clone -q "$url" "third_party/$s"
      fi
    fi
    git -C "third_party/$s" fetch -q origin "$sha" 2>/dev/null || true
    git -C "third_party/$s" checkout -q "$sha"
  done
  git -C third_party/LiveCodeBench-Pro submodule update -q --init --recursive
  (cd third_party/tau2-bench && $UV sync -q --frozen --no-dev --python "$(V get tools.python)")
  (cd third_party/IFBench && $UV sync -q --frozen --python "$(V get tools.python)")
fi

if has venvs; then
  for v in $(V keys venvs); do
    lock=$(V get venvs.$v.lock)
    step "venv .venv-$v from $lock"
    [ -x ".venv-$v/bin/python" ] || $UV venv -q --python "$(V get tools.python)" ".venv-$v"
    idx=()
    if python3 "$REPO/tools/versions.py" get venvs.$v.index_url >/dev/null 2>&1; then
      idx=(--index-url "$(V get venvs.$v.index_url)" --extra-index-url "$(V get venvs.$v.extra_index_url)" --index-strategy unsafe-best-match)
    fi
    # sync = the venv holds exactly the lock's packages, nothing more
    $UV pip sync -q --python ".venv-$v/bin/python" "${idx[@]}" "$lock"
  done
  # nvidia-cutlass-dsl has an install-order race that can corrupt libs-cu13; reinstalling the locked version fixes it.
  $UV pip install -q --python .venv-client/bin/python --index-url https://pypi.org/simple --force-reinstall --no-deps \
    "nvidia-cutlass-dsl-libs-cu13==$(grep -i '^nvidia-cutlass-dsl-libs-cu13==' env/locks/client.lock.txt | cut -d= -f3)"
fi

if has models || has optional-models; then
  HF=$REPO/.venv-tools/bin/hf
  [ -x "$HF" ] || { echo "models need .venv-tools (run the venvs step first)" >&2; exit 1; }
  for m in $(V keys models); do
    opt=$(python3 -c "import tomllib;print(tomllib.load(open('env/versions.toml','rb'))['models']['$m'].get('optional', False))")
    if [ "$opt" = True ] && ! has optional-models; then echo "skip optional model $m"; continue; fi
    repo=$(V get models.$m.repo); rev=$(V get models.$m.revision)
    step "model $m: $repo @ ${rev:0:12}"
    ex=()
    for e in $(python3 -c "import tomllib;print(' '.join(tomllib.load(open('env/versions.toml','rb'))['models']['$m'].get('exclude', [])))"); do
      ex+=(--exclude "$e")
    done
    "$HF" download "$repo" --revision "$rev" --local-dir "models/$m" "${ex[@]}"
  done
fi

if has data; then
  # Training data: release assets listed in data/MANIFEST.toml (public HTTPS download, no login), each checked against
  # its sha256 before it is put in place.
  step "data: training lists and scenario cards (data/MANIFEST.toml)"
  python3 - <<'PY'
import hashlib, os, tempfile, tomllib, urllib.request
from pathlib import Path
m = tomllib.load(open("data/MANIFEST.toml", "rb"))
base = lambda a: f"https://github.com/{m['repo']}/releases/download/{a.get('release', m['release'])}"
def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()
for name, a in m["assets"].items():
    dst = Path(a["path"])
    if dst.exists() and sha(dst) == a["sha256"]:
        print(f"ok      {dst}"); continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dst.parent, suffix=".part"); os.close(fd)
    urllib.request.urlretrieve(f"{base(a)}/{name}", tmp)
    if sha(tmp) != a["sha256"]:
        os.unlink(tmp); raise SystemExit(f"sha256 mismatch for {name}")
    os.replace(tmp, dst)
    print(f"fetched {dst}")
PY
fi

if has eval-data; then
  # Evaluation files (eval/data/SHA256SUMS): the large prompt files from the release and the LiveCodeBench v6 test
  # cases (public); GPQA Diamond and LiveCodeBench Pro are gated on Hugging Face and rebuilt locally from pinned
  # revisions when HF_TOKEN is set (an account that accepted their terms).
  step "eval-data: evaluation prompt and test files (eval/data/SHA256SUMS)"
  eval/fetch_data.sh
  ok_sum() { (cd "$1" && [ -s "$2" ] && grep -E "  $2\$" "$3" | sha256sum --quiet -c - >/dev/null 2>&1); }
  for g in gpqa_diamond:prep_gpqa lcb_pro:prep_lcbpro; do
    f=${g%%:*}.jsonl; script=${g#*:}
    if ok_sum eval/data "$f" SHA256SUMS; then echo "$f: ok"; continue; fi
    # gated: the Hugging Face account of HF_TOKEN (or of `hf auth login`) must have accepted the dataset's terms
    .venv-tools/bin/python eval/prep/$script.py && ok_sum eval/data "$f" SHA256SUMS && echo "$f: built, ok" || echo "WARNING: \
$f (gated) not built: set HF_TOKEN for an account that accepted the dataset's terms on Hugging Face, then rerun \
'setup/setup.sh eval-data' (phases S2 / S4 need it)"
  done
  # IFBench's NLTK data, checked against eval/data/NLTK_SHA256SUMS (the grader would otherwise fetch the latest at run time)
  third_party/IFBench/.venv/bin/python -c "
import nltk
for p in ('punkt', 'punkt_tab', 'stopwords', 'averaged_perceptron_tagger_eng'):
    assert nltk.download(p, download_dir='$NLTK_DATA', quiet=True), p"
  (cd "$NLTK_DATA" && grep -v '^#' "$REPO/eval/data/NLTK_SHA256SUMS" | sha256sum --quiet -c -) && echo "nltk data: ok" \
    || { echo "NLTK data differ from eval/data/NLTK_SHA256SUMS" >&2; exit 1; }
  # LiveCodeBench Pro test cases (gated, pinned revision), unpacked for the judge so that grading needs no token
  if ok_sum eval/data lcb_pro.jsonl SHA256SUMS; then
    .venv-client/bin/python -c "
import json, sys; sys.path.insert(0, 'eval/lib')
import grade_lcbpro as G
pids = sorted({json.loads(l)['problem_id'] for l in open('eval/data/lcb_pro.jsonl')})
for p in pids: G.ensure_problem(p)
print(f'lcb_pro test cases: {len(pids)} problems ready')"
  fi
  (cd eval/data && sha256sum --quiet -c --ignore-missing SHA256SUMS) && echo "eval data: every present file matches SHA256SUMS"
fi

if has judge; then
  step "judge: LiveCodeBench Pro judge image (CPU-only docker build)"
  eval/setup_judge.sh
fi

step "done; verify with: python3 tools/check_env.py"
