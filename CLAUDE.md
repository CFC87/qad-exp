# Guidance for Claude Code in this repository

## What this repository is

Experiments on NVFP4 quantization-aware distillation (QAD) of LLMs and on evaluating the resulting checkpoints. The
results are compared number by number with reference runs that were made on exactly the same software: the same
container images (by digest), Python packages (by lock file), third-party code (by commit) and model weights (by
Hugging Face revision). That comparison is the point of running here, so **matching the pinned environment exactly
matters more than getting a run to start quickly.**

Why so strict:
- Scores move with the serving stack. Moving the same checkpoint from one vLLM release to another has changed
  benchmark scores by up to about 2 points. That is as large as many of the effects being measured.
- NVFP4 numerics depend on the kernels and on details in specific library versions: FP4 GEMM, FP8 KV cache with
  scale 1.0, and how the quantizer configuration is read.
- Chat templates are rendered client-side with a specific `transformers`. A different version can change the prompt
  tokens.

## Rules

1. **Never change a pinned version.**
   - Do not upgrade, downgrade or add packages in the `.venv-*` environments.
   - Do not pull an image by tag or a newer digest.
   - Do not edit or re-checkout `third_party/` sources.
   - Do not download models at another revision.
   - All pins are in `env/versions.toml` and `env/locks/`.
2. **Install only with `setup/setup.sh` and verify with `python3 tools/check_env.py`.** Run experiments only when every
   non-host item is `ok`.
3. **When something fails, do not work around it by changing versions, flags or code.**
   - Stop, and report to the person you work with: the exact error, the step, and the output of `tools/check_env.py`.
   - Fixes that touch versions or experiment code belong in this repository: they are made upstream and pulled again.
4. **Host-level differences are reported, not "fixed".** Driver, OS, Docker and GPU count show up as `warn`. Do not
   change drivers or system packages to clear a warning.
5. **Keep the parallel layout as specified.** Use only the two supported layouts (see `README.md`):
   - Training uses pipeline parallelism only. Do not add tensor, data, context or expert parallelism.
   - Every vLLM server uses exactly 1 GPU (tensor parallel 1).
   - Do not change any of this for speed: it changes the numerics.
6. **Do not change experiment settings.** This covers batch sizes, learning rate and schedule, sampling parameters,
   seeds, token limits, data lists and prompts. Only knobs that an experiment's README marks as free may be changed.
   Examples of free knobs: paths, ports, concurrency.
7. **Never print or commit secrets.** API keys and tokens go in environment variables or in the files an
   experiment's README names, outside git.

8. **Never delete anything under `out/` or `data/` yourself**: training checkpoints, logs, exports or data. A chain
   itself keeps only its newest saves while it trains, as designed; when it is done, each segment keeps the weights-only
   checkpoint of its last step (and of its `keep_weights` steps), the full checkpoints later branches start from, its
   logs and its exports. These are what later runs start from: do not remove or overwrite them. To redo something,
   move the old outputs aside (for example into `out/old/`). If the disk runs short, stop and report; do not free space
   yourself.

## First, after pulling: are the trained checkpoints still there?

Earlier versions of the experiment READMEs said to run `chain.py <config> clean` after a run's exports. That command
deleted `out/train/<segment>/`, the run's trained checkpoints. It has been removed from the repository. Before anything
else, run this in the repository on every node used for training (with the `--out` you used, default `out/`), and send
the whole output to the person you work with:

```bash
shopt -s nullglob; set -- out/train/*/; [ $# -gt 0 ] || echo "no directory under out/train"
for d in "$@"; do s=$(basename "$d"); t="$d/checkpoints/latest_checkpointed_iteration.txt"
  if [ ! -x "$d" ] || { [ -e "$t" ] && [ ! -r "$t" ]; }; then echo "$s: UNREADABLE (list it with sudo)"; continue; fi
  it=$(cat "$t" 2>/dev/null | tr -dc 0-9)
  if [ -n "$it" ] && [ -d "$d/checkpoints/iter_$(printf %07d "$it")" ]; then echo "$s: last checkpoint iter $it present"
  else echo "$s: NO checkpoint"; fi; done
grep -H "cleaned: trainer checkpoints" out/*.chain.log   # chains that were cleaned: their checkpoints are gone
ls out/exports/
```

Report which runs still have their last checkpoint and which were cleaned. Do not start, stop or delete anything because
of this check.

## Setup walkthrough

```bash
git clone https://github.com/CFC87/qad-exp.git && cd qad-exp
export HF_TOKEN=...               # gated evaluation sets: an account that accepted their terms on Hugging Face
# docker login nvcr.io            # only if the NeMo image pull is refused (user $oauthtoken, password an NGC API key)
setup/setup.sh                    # mostly downloads (~100 GB of images, ~55 GB in the repository); rerun after a failure
python3 tools/check_env.py        # writes reports/env_report.md and .json
```

Notes:
- uv and CPython are installed into `.tools/` at the pinned versions. A warning that `uv` is "shadowed by other
  commands in your PATH" is harmless: the scripts call the repo-local `.tools/uv/uv` by path.
- The `venvs` step uses `uv pip sync`, so each environment ends up with exactly its lock file's packages. The client
  environment pulls CUDA 13.0 PyTorch wheels from `download.pytorch.org`.
- After installing the client environment, the `venvs` step reinstalls `nvidia-cutlass-dsl-libs-cu13` on purpose. The
  package has an install-order race that can silently corrupt it.
- The `sender` step copies the `vllm` package out of the pinned serving image into `.cache/vllm-sender/`. The trainer
  uses it to push weights to the vLLM servers, so sender and receivers are the same build.
- The `models` step downloads Qwen3-8B and Qwen3-8B-NVFP4 (about 22 GB). `optional-models` adds the other pinned
  models (Qwen3-30B-A3B and the two Nemotrons with their NVFP4 releases, gpt-oss-120b; about 300 GB): needed by
  `experiments/round4`, `experiments/mix60`, `experiments/mix60b` and `experiments/mix60c`; gpt-oss-120b itself is not
  used by any experiment.
- `tools/check_env.py --deep` hashes every model weight file against Hugging Face's recorded sha256. It takes a while.
  Run it once after the download.
- The `data` step downloads the training lists from this repository's releases `data-v1` to `data-v5`; no login is
  needed. It checks each sha256 against `data/MANIFEST.toml`.
- The `eval-data` step does three things:
  - downloads the evaluation files;
  - rebuilds the two gated sets (GPQA, LiveCodeBench Pro) from pinned Hugging Face revisions, and unpacks the
    LiveCodeBench Pro test cases (13 GB);
  - fetches IFBench's NLTK data and checks everything against `eval/data/SHA256SUMS` and `eval/data/NLTK_SHA256SUMS`.

  The gated sets need `HF_TOKEN` from an account that accepted their terms. Without it the step warns and skips them;
  set it and rerun `setup/setup.sh eval-data`. Never commit these two files: their licenses do not allow
  redistribution.

## Layouts

| | Training | Generation (vLLM servers, 1 GPU each) |
|---|---|---|
| 1 node | GPUs 0-3, pipeline parallel 4 | GPUs 4-7, 4 servers |
| 2 nodes | node 0 GPUs 0-7, pipeline parallel 8 | node 1 GPUs 0-7, 8 servers |

Evaluation-only experiments run one single-GPU server per GPU. Run scripts take the layout as an argument; do not
invent other splits.

## Running an experiment

Each experiment directory under `experiments/` has a README: read it fully before starting. In short:
- Training goes through `train/launch/chain.py <config> <action> --layout 1node|2node`. Run `dryrun` first: it starts
  nothing and shows what would run.
- External model APIs (the multi-turn simulator during training, the τ² user simulator during evaluation) are
  configured only through environment variables, never in files. Each experiment README names the variables and the
  model each one must serve. If the person you work with has not provided them, ask; do not substitute another model.
- τ² API errors (user simulator, judge) are retried per request by the transport, never by re-running conversations.
  A conversation an API error still ended is a fallback case that must be counted and reported (`eval/README.md`, "tau2
  API errors and retries").
- Long runs: launch them with `nohup ... &` or in `tmux`, and follow `out/<name>.chain.log`. A `CHAIN_STOP` line means
  the chain stopped on a problem: report it (rule 3).

## Reporting results

- Each experiment writes its outputs under its own directory, as its README describes.
- Scores are reported as the **Report** section of `eval/results/summary.md` (`eval/lib/summary.py --arms ... --report`),
  every row as printed, for every arm the experiment README lists. Do not recompute, round, select or reformat them.
- With every set of results, send `reports/env_report.md` and `reports/env_report.json` from the same machine. Also
  send the repository commit (`git rev-parse HEAD`), and say whether anything was changed locally (`git status`).
- For τ², also report per arm and domain how many conversations an API error ended, how many were re-run and how many
  are still left out (expected 0 each).
- If a run needed any deviation from these rules, state it explicitly next to the results. Results with an undisclosed
  deviation cannot be compared.
