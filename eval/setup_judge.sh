#!/usr/bin/env bash
# Build the LiveCodeBench Pro judge image (lightcpverifier:qadx) from the pinned LightCPVerifier submodule of
# third_party/LiveCodeBench-Pro. Only the grading of LiveCodeBench Pro (phase S4) needs it.
# Upstream's Dockerfile installs npm@latest (needs a newer Node than its base image has) and the latest go-judge release
# (v1.12 cannot create cgroups in this container). The build therefore uses a patched copy of the build context under
# .cache/ (third_party/ stays unmodified): go-judge pinned to v1.9.6, current when the pinned LightCPVerifier commit was
# made, and the bundled npm kept. The base image (ubuntu:22.04) and its apt packages are fetched at build time.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=third_party/LiveCodeBench-Pro/LightCPVerifier
CTX=.cache/lightcpverifier-build
IMAGE=lightcpverifier:qadx
if docker image inspect "$IMAGE" >/dev/null 2>&1; then echo "$IMAGE present"; exit 0; fi
[ -f "$SRC/Dockerfile" ] || { echo "$SRC missing: run setup/setup.sh sources" >&2; exit 1; }
rm -rf "$CTX"; mkdir -p "$CTX"
tar -C "$SRC" --exclude=.git -cf - . | tar -C "$CTX" -xf -
python3 - "$CTX/Dockerfile" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = s.replace("RUN npm install -g npm@latest", "# npm@latest needs a newer Node than 20; the bundled npm 10.8 is enough (pinned build)")
s = re.sub(r"url=\$\(curl -fsSL https://api.github.com/repos/criyle/go-judge/releases/latest.*?\n\s*curl -fsSL \"\$url\"",
           "curl -fsSL https://github.com/criyle/go-judge/releases/download/v1.9.6/go-judge_1.9.6_linux_amd64v2.tar.gz", s, flags=re.S)
open(p, "w").write(s)
PY
grep -q "go-judge_1.9.6" "$CTX/Dockerfile" && ! grep -q "RUN npm install -g npm@latest" "$CTX/Dockerfile" \
  || { echo "LightCPVerifier Dockerfile patch did not apply" >&2; exit 1; }
docker build -t "$IMAGE" "$CTX"
echo "built $IMAGE"
