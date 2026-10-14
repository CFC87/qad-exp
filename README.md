# qad-exp

Experiments on NVFP4 quantization-aware distillation (QAD) and evaluation of the resulting checkpoints, packaged to run
on NVIDIA B200 nodes with an environment that is pinned exactly: container images by registry digest, Python packages by
lock file, third-party code by commit, model weights by Hugging Face revision.

Rules for running experiments here, setup notes and background: [`CLAUDE.md`](CLAUDE.md) (written for Claude Code, useful
for anyone).

## Requirements

- 1 or 2 nodes, 8x NVIDIA B200 each.
- Ubuntu 24.04, NVIDIA driver 580.126.09 (CUDA 13.0), Docker with the NVIDIA Container Toolkit, `git`, `curl`,
  `python3` >= 3.11.
- Access to `nvcr.io` (`docker login nvcr.io`, user `$oauthtoken`, an NGC API key) and to Hugging Face (`export
  HF_TOKEN=...` for gated repositories and datasets).
- Disk: about 250 GB for images, environments and the default models.

## Setup

```bash
setup/setup.sh                    # uv + Python, images, sources, environments, models (all idempotent)
python3 tools/check_env.py        # verify everything against env/versions.toml
```

`setup/setup.sh <step> ...` runs single steps: `uv`, `images`, `sender`, `sources`, `venvs`, `models`,
`optional-models`. Everything is installed inside the repository (`.tools/`, `.venv-*`, `third_party/`, `models/`,
`.cache/`); nothing is installed system-wide.

## Pinned versions

All versions live in [`env/versions.toml`](env/versions.toml); the lock files are in `env/locks/`.

| Part | Pin |
|---|---|
| Serving (evaluation, training rollouts) | `vllm/vllm-openai:v0.29.0` @ `sha256:c2914767…` (vLLM 0.29.0, torch 2.13.0+cu130) |
| Training, PTQ, export | `nvcr.io/nvidia/nemo:26.08.01` @ `sha256:fdd6e9c7…` with NVIDIA Model-Optimizer `06ad0218` (0.47.0rc2) |
| Python | CPython 3.12.13, uv 0.12.12; environments `client`, `tools`, `bfcl` from `env/locks/` |
| Benchmarks | τ²-bench `fc0055d`, IFBench `1c40f0c`, LiveCodeBench Pro `d614eef`, LongBench Pro (`caskcsg/longcontext` `49ee319`), BFCL `bfcl-eval==2026.3.23` |
| Models | Qwen3-8B, Qwen3-30B-A3B, Nemotron Nano 9B v2, Nemotron 3 Nano 30B-A3B: BF16 and NVIDIA NVFP4 releases at fixed revisions |

## Environment report

`tools/check_env.py` writes `reports/env_report.md` and `reports/env_report.json`. Every pinned item must be `ok`;
host-level differences (driver, OS, Docker, GPU count) are listed as `warn`. `--deep` additionally checks the sha256 of
every model weight file. Please send the report with every set of results.

## Layouts

Each experiment runs on either layout:

| | Training | Generation (vLLM, 1 GPU per server) |
|---|---|---|
| 1 node | GPUs 0-3: pipeline parallel 4 | GPUs 4-7: 4 servers |
| 2 nodes | node 0, GPUs 0-7: pipeline parallel 8 | node 1, GPUs 0-7: 8 servers |

Training uses pipeline parallelism only, so the computation of every layer is the same as in the reference runs;
evaluation serves every model on single GPUs (tensor parallel 1).

## Experiments

Added per experiment under `experiments/`, each with its own README, configuration and run script.
