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
- Docker access to `nvcr.io` for the NeMo image (if the pull is refused: `docker login nvcr.io`, user `$oauthtoken`,
  password an NGC API key).
- A Hugging Face token (`export HF_TOKEN=...`) of an account that accepted the terms of the gated evaluation sets
  `Idavidrein/gpqa`, `QAQAQAQAQ/LiveCodeBench-Pro` and `QAQAQAQAQ/LiveCodeBench-Pro-Testcase`. Everything else
  downloads without credentials.
- Disk:
  - about 100 GB of docker images, plus about 40 GB in the repository (environments, the default models, data);
  - about 500 GB per training chain while it runs, which `chain.py clean` frees after its exports (see the experiment
    README).

## Setup

```bash
git clone https://github.com/CFC87/qad-exp.git && cd qad-exp
export HF_TOKEN=...               # gated evaluation sets (see Requirements)
setup/setup.sh                    # everything below; idempotent, rerun after an interruption
python3 tools/check_env.py        # verify everything; writes reports/env_report.md
```

`setup/setup.sh <step> ...` runs single steps:
- `uv`, `images`, `sender`, `sources`, `venvs`: the tools, images, third-party code and Python environments;
- `models`: the models;
- `data`: the training lists, from this repository's release `data-v1`;
- `eval-data`: the evaluation files. The two gated sets are rebuilt with `HF_TOKEN`;
- `judge`: the LiveCodeBench Pro judge image;
- `optional-models`: models not needed by the current experiments.

Everything is installed inside the repository: `.tools/`, `.venv-*`, `third_party/`, `models/`, `data/`, `eval/data/`,
`.cache/`. Nothing is installed system-wide. Every downloaded file is checked against a pinned commit, digest,
revision or sha256.

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

## Repository layout

| Path | Contents |
|---|---|
| `env/` | pinned versions (`versions.toml`) and Python lock files |
| `setup/setup.sh`, `tools/check_env.py` | installation and its verification |
| `data/` | training lists (`MANIFEST.toml`: release assets and sha256; `subsets/<name>/SUBSET.json`: what each list is) |
| `train/` | the QAD trainer (runs in the NeMo image), the rollout controller and multi-turn environments (`onpolicy/`), the launcher (`launch/chain.py`), CPU tests (`tests/`) |
| `eval/` | the evaluation harness (vLLM serving, benchmarks, graders) |
| `experiments/` | one directory per experiment: README with what to run and send back, configuration files |

## Experiments

- [`experiments/pairs-r2`](experiments/pairs-r2/README.md): five QAD runs on mixtures of two training domains, each a
  60 / 120 / 180-step WSD chain, every budget evaluated on the full benchmark set.

## Evaluation

[`eval/README.md`](eval/README.md): Qwen3-8B checkpoints (BF16, NVIDIA NVFP4, QAD exports) on the full benchmark set with
`eval/run.py`, one vLLM server per GPU, results comparable with the reference runs; data steps, the τ² API variables,
outputs, tests.

## Tests without GPUs

```bash
.venv-client/bin/python eval/tests/test_dry_run.py                                       # evaluation scheduler (after setup)
cd train && for t in tests/test_*.py; do ../.venv-client/bin/python $t || break; done   # needs models/Qwen3-8B and data/
```
