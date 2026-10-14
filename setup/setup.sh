#!/usr/bin/env bash
# Install everything pinned in env/versions.toml. Idempotent: rerun safely.
#
#   setup/setup.sh                 # all steps except optional models
#   setup/setup.sh images venvs    # only some steps
#
# Steps: uv, images, sender, sources, venvs, models, data.  Add "optional-models" to also fetch models marked optional.
# Prerequisites: Ubuntu 24.04, NVIDIA driver with CUDA 13.0, docker + NVIDIA Container Toolkit, git, curl, python3 >= 3.11.
# The NeMo image is on nvcr.io: run `docker login nvcr.io` (user $oauthtoken, an NGC API key) first if the pull is refused.
# Gated Hugging Face repos/datasets need `export HF_TOKEN=...`.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
V() { python3 "$REPO/tools/versions.py" "$@"; }
step() { echo; echo "=== $*"; }

export UV_PYTHON_INSTALL_DIR=$REPO/.tools/python
UV=$REPO/.tools/uv/uv
STEPS=${*:-uv images sender sources venvs models data}
has() { [[ " $STEPS " == *" $1 "* ]]; }

if has uv; then
  step "uv $(V get tools.uv) and CPython $(V get tools.python) (repo-local, under .tools/)"
  want=$(V get tools.uv)
  if [ ! -x "$UV" ] || [ "$($UV --version | awk '{print $2}')" != "$want" ]; then
    curl -LsSf "https://astral.sh/uv/$want/install.sh" | env UV_INSTALL_DIR="$REPO/.tools/uv" UV_NO_MODIFY_PATH=1 sh
  fi
  $UV python install "$(V get tools.python)"
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
  # Release assets listed in data/MANIFEST.toml (training lists, scenario cards; evaluation files are fetched by the
  # evaluation's own step). Needs the GitHub CLI logged in with read access to this repository (`gh auth login`).
  step "data: release assets of data/MANIFEST.toml"
  command -v gh >/dev/null || { echo "the data step needs the GitHub CLI (gh)" >&2; exit 1; }
  python3 - <<'PY'
import hashlib, subprocess, sys, tempfile, tomllib
from pathlib import Path
m = tomllib.load(open("data/MANIFEST.toml", "rb"))
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
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        subprocess.run(["gh", "release", "download", m["release"], "-R", m["repo"], "-p", name, "-D", tmp], check=True)
        got = Path(tmp) / name
        if sha(got) != a["sha256"]:
            sys.exit(f"sha256 mismatch for {name}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        got.replace(dst)
    print(f"fetched {dst}")
PY
fi

step "done; verify with: python3 tools/check_env.py"
