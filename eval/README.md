# Evaluation of Qwen3-8B checkpoints

This harness evaluates Qwen3-8B checkpoints (BF16, NVIDIA's NVFP4 release, NVFP4 QAD exports) so that the numbers are
directly comparable with the reference results: same serving image and flags, same prompts and chat-template rendering,
same sampling, output caps, context windows and YaRN settings, same number of samples, concurrency and request order,
same graders. Read `CLAUDE.md` first; its rules apply here (nothing below is a free knob unless it says so).

## What runs

One **job** = one checkpoint ("arm") x one **phase**. A phase is one vLLM server on one GPU running one or more
benchmarks; `eval/run.py` runs the jobs over the GPUs of a node, one job per GPU, a GPU taking the next job when free.

| Phase | Benchmarks (`bench:k:cap`) | Concurrency | Window | Typical time on one B200 (QAD checkpoint) |
|---|---|---|---|---|
| S1 | `aime26:8:38912`, `hmmt_feb26:8:38912`, `hmmt_feb25:8:38912`, `hmmt_nov25:8:38912` | 160 | 40,960 | 1.0 h |
| S2 | `gpqa_diamond:8:32768`, `ifbench:4:32768`, `mmlu_redux_1k:1:32768` | 160 | 40,960 | 0.6 h |
| S3 | `lcb_v6:4:32768` | 160 | 40,960 | 0.5 h |
| S4 | `lcb_pro:8:32768` | 160 | 40,960 | 1.2 h |
| L1 | `longbench_pro:1:32768` | 64 | 131,072, static YaRN x4 | 2.4 h |
| M | `mrcr8:8:32768` | 64 | 131,072, static YaRN x4 | 0.5 h |
| T1 | `tau2_retail:4:32768` (4 trials of 114 tasks) | 48 | 40,960 | 0.9 h |
| T2 | `tau2_airline:4:32768`, `tau2_telecom:4:32768` (50 + 114 tasks) | 48 per domain | 40,960 | 1.9 h |
| B | `bfcl_mt:4:32768` (BFCL v4 multi-turn, 4 categories x 200, 4 replicates) | 128 (4 x 32 threads) | 40,960 | 0.65 h |

`k` = samples per question (tau2: trials; BFCL: replicates); the cap is per request (tau2 / BFCL: per model turn).
Times are medians of the reference runs of QAD checkpoints (server start + generation, without grading); NVIDIA's
NVFP4 release takes about 1.3x and BF16 about 1.8x as long (longer outputs); tau2 also depends on the API.
One checkpoint is ~9.7 GPU-hours; on 8 GPUs it finishes in ~2.8 h (T1 and T2 share one API key and run one after the
other) plus grading. With several checkpoints the tau2 phases are usually the bottleneck: they run one at a time per API
key (see `--tau2-per-key`).

Settings common to every phase (`lib/models.py`, `lib/serve.py`, `lib/run_phase.py`):
- Image `vllm/vllm-openai@sha256:c2914767…` (vLLM 0.29.0, from `env/versions.toml`), tensor parallel 1,
  `--gpu-memory-utilization 0.90 --max-num-seqs 256 --seed 0`, prefix caching off except in the tau2 / BFCL phases,
  `--language-model-only`. KV cache dtype and generation defaults come from the checkpoint (FP8 KV for NVFP4).
- Requests: `/v1/completions` with the prompt rendered client-side by the checkpoint's tokenizer (`.venv-client`,
  transformers 5.17.0), thinking mode on, `skip_special_tokens=false`; sampling T 0.6, top-p 0.95, top-k 20, min-p 0,
  no penalties, no per-request seed; output budget `min(cap, window - prompt - 64)`. All (question, sample) requests are
  sent in one fixed shuffled order per benchmark (seed from the benchmark name), the same for every arm.
- LongBench Pro prompts are truncated in the middle to `min(224,000, window - cap - 512)` = 97,792 tokens.
- tau2-bench v1.0.1: base split, seed 300, `--max-retries 0`, agent through vLLM's hermes / qwen3 parsers
  (`--override-generation-config '{"max_new_tokens": 32768}'`), user simulator and retail judge gpt-5.2 (below).
- BFCL: bfcl-eval 2026.3.23 with the handler `qadx/qwen3-ct` (`lib/bfcl_handler.py`): chat template with tools,
  output parsed like vLLM's qwen3 + hermes parsers, `max_tokens = min(32768, window - prompt)`.

## Requirements

- Setup done (`setup/setup.sh`) and `python3 tools/check_env.py` all `ok` apart from host warnings and `eval data`
  items that the steps below create. The built-in arms `bf16` / `nvfp4` use `models/Qwen3-8B` / `models/Qwen3-8B-NVFP4`.
- Docker with the NVIDIA Container Toolkit; the GPUs used must be free (no other process holding memory).
- **Hugging Face token** (`export HF_TOKEN=...`) whose account accepted the terms of `Idavidrein/gpqa`,
  `QAQAQAQAQ/LiveCodeBench-Pro` and `QAQAQAQAQ/LiveCodeBench-Pro-Testcase` (gated; the last one is also read while grading).
- **tau2 API** (phases T1 / T2): an OpenAI-compatible endpoint serving **gpt-5.2**. It plays the customer in every tau2
  conversation and judges the retail tasks' natural-language assertions; requests are sent with `reasoning_effort:
  "none"` and temperature 0. Set in the environment (never in files of this repository):

  ```bash
  export QADX_API_BASE=https://<endpoint>/v1   # must end in /v1
  export QADX_API_KEY=...                      # used by T2, and by T1 when there is no second key
  export QADX_API_KEY_2=...                    # optional: T1 uses it, so T1 and T2 can run at the same time
  export QADX_API_RPS=1.2                      # optional: requests per second per running tau2 phase (default 1.2; 0 = no throttle)
  # the model id the endpoint receives is $QADX_TAU2_MODEL_PREFIX$QADX_TAU2_MODEL, default "openai/openai/gpt-5.2"
  # (a LiteLLM proxy's id for gpt-5.2); for an endpoint that serves it as plain "gpt-5.2":
  export QADX_TAU2_MODEL_PREFIX=""
  ```
  Do not substitute another model: the user simulator changes tau2 scores. All tau2 API calls go through a local rate
  limiter (`lib/api_throttle.py`) that waits and retries on 429 / 5xx instead of failing the conversation.
- CPU: grading runs on the node (LiveCodeBench v6 runs 24 worker processes; LiveCodeBench Pro judges in a container).
- Disk: about 1 GB of outputs per checkpoint (generations, tau2 / BFCL trajectories).

## Steps

Run from the repository root.

```bash
# 1. data and judge image: done by setup/setup.sh (steps eval-data and judge); to redo only these:
#    setup/setup.sh eval-data judge
# 2. check: every evaluation file present and matching eval/data/SHA256SUMS, judge image built
python3 tools/check_env.py
# 3. dry run: the plan, every job's docker command and first request; starts nothing
.venv-client/bin/python eval/run.py --arm qad060=/path/to/export-s060 --dry-run
# 4. the run (all 9 phases of each arm, all GPUs of the node)
nohup .venv-client/bin/python eval/run.py --arm qad060=/path/to/export-s060 --arm qad120=/path/to/export-s120 \
    > eval-run.log 2>&1 &
tail -f eval/results/logs/scheduler.log
```

Options of `run.py`:
- `--arm NAME=PATH` (repeatable): a checkpoint directory in Hugging Face format (config, tokenizer, weights; no
  symlinks leaving the directory, it is mounted alone into the server container). `--arm bf16` / `--arm nvfp4` are the
  built-in baselines. One arm name = one checkpoint, for good: results go to `eval/results/<NAME>/`.
- `--phases S1,S2,...` (default all), `--gpus 0,1,...` (default all GPUs of the node).
- `--shard i/n`: two nodes split one job list. Run the identical command with `--shard 0/2` on one node and
  `--shard 1/2` on the other; the jobs are split by expected time.
  - Every checkpoint must exist on both nodes at the same path.
  - Give each node its own API key(s), or the two nodes share one key's rate.
  - Afterwards, copy one node's `eval/results/<arm>/` directories into the other's `eval/results/`, then run the same
    command with `--grade-only` there to write the combined `summary.md`.
- `--tau2-per-key N` (default 1): tau2 phases running at once per API key. Raise it only if the endpoint's rate limit
  allows N x 2-3 simultaneous conversations x 48.
- `--grade-only`: grade what is there and write the summary, without starting servers.
- `--limit N`: pilot on the first N items per benchmark, written to `<NAME>__limitN/` (never mixed with real results).
- `--results DIR`, `--no-compile-cache`, `--force` (rerun phases already marked complete; finished samples are still
  skipped).

Interruptions: run the same command again. Finished samples are skipped (generations are appended per sample; BFCL skips
finished entries; an interrupted tau2 domain resumes with tau2's `--auto-resume`; tau2 conversations that an API error
ended are re-run with their original seeds), phases marked complete are not restarted, grading is redone where the
generations are newer. Ctrl-C stops all jobs and their containers. A job is limited to 12 h.

## Outputs

```
eval/results/
  summary.md, summary.json         scores by benchmark and category per arm (lib/summary.py), phases complete
  timing.jsonl                     generation time and output tokens per benchmark
  <arm>/arm.json                   checkpoint path and sha256 of its config / quant config / tokenizer config, repo commit
  <arm>/<bench>.jsonl              one row per sample: qid, j, prompt / completion tokens, finish_reason, text
  <arm>/<bench>.grades.jsonl       one row per sample: qid, j, correct (0/1, or a score in [0, 1]), why
  <arm>/<phase>.status.json        complete or not, what is missing
  <arm>/tau2_<domain>.jsonl / .grades.jsonl / .left_out.jsonl, <arm>/tau2_runs/   tau2 rows and full trajectories
  <arm>/bfcl_mt.jsonl / .grades.jsonl, <arm>/bfcl_runs/                          BFCL rows and bfcl's own result files
  logs/scheduler.log, logs/<arm>_<phase>.log, logs/server_<arm>_<phase>.log, logs/tau2_*, logs/bfcl_*, logs/grade_*
```

Scores: benchmark score = mean over questions of the per-question mean over samples. Categories: reasoning (AIME 2026,
three HMMT sets, GPQA; equal weights), code (LiveCodeBench v6, LiveCodeBench Pro without its 5 interactive problems),
long context (LongBench Pro, MRCR), BFCL, tau2 (tasks of the three domains pooled); MMLU-Redux and IFBench are
no-damage checks. Recovery against the baselines is computed from the per-sample grades. **Send back** the whole
`eval/results/` directory (with `tau2_runs/` and `bfcl_runs/`), `reports/env_report.{md,json}`, and the output of
`git rev-parse HEAD` and `git status`.

## Graders

| Benchmark | Grader | Environment |
|---|---|---|
| AIME / HMMT | last `\boxed{}` after `</think>`: normalized exact match, else math-verify 0.9.0 | `.venv-client` |
| GPQA, MMLU-Redux | answer letter (JSON field, `Answer: X`, boxed, fallback forms) after `</think>` | `.venv-client` |
| MRCR | 0 unless the answer starts with the required prefix, else difflib ratio | `.venv-client` |
| LiveCodeBench v6 | last ```python block after `</think>`, run against ≤30 tests, 8 s wall time per test | `.venv-client` |
| LiveCodeBench Pro | official extractor + LightCPVerifier judge (`eval/setup_judge.sh`), tests at a pinned revision | `.venv-client` + Docker |
| IFBench | official evaluation_lib, prompt-level loose, on the text after `</think>` | IFBench's env |
| LongBench Pro | official metric functions on the text after `</think>` | `.venv-tools` |
| tau2 | tau2's reward; empty agent reply / context overflow = 0; API-side failures left out and re-run | tau2's env |
| BFCL | bfcl's own multi-turn checker | `.venv-bfcl` |

Grading starts after each phase, one grading job at a time; LiveCodeBench Pro is graded at the end, when no server runs
(its judge measures time). LiveCodeBench v6 uses wall-clock test timeouts: grade on a machine that is not overloaded.
IFBench fetches its NLTK data on first use; `grade_ifbench.py --selftest` re-grades IFBench's own sample outputs
(288 samples; 6 mismatches against IFBench's stored files are upstream's and appear in the reference too).

## Tests (CPU only)

```bash
.venv-client/bin/python eval/tests/test_dry_run.py      # scheduler: dry run on 8 GPUs, shards, scheduling loop (after setup: models, eval data)
```

Before release this harness was checked against the reference harness the published numbers come from, on the
reference's raw outputs:
- every request body of every benchmark was identical, for three checkpoints and in the same send order, and so were
  every serve, tau2 and BFCL command;
- every sample re-graded to the same grade, and the category scores matched.

The only difference: LiveCodeBench v6 samples whose verdict is a wall-clock `timeout` can pass or fail when executed
again (a handful per 700 samples, in both directions between repeated runs).

## Data

`eval/data/SHA256SUMS` lists every file the evaluation reads. In git: the competition sets, MMLU-Redux 1k, IFBench,
LiveCodeBench v6 prompts and the LiveCodeBench Pro exclusion list. Rebuilt locally because their licenses or gates do
not allow redistribution: GPQA Diamond (`prep/prep_gpqa.py`, `Idavidrein/gpqa` at a pinned revision, options shuffled
with a per-question seed) and LiveCodeBench Pro 2025Q2 (`prep/prep_lcbpro.py`, both repositories pinned). Downloaded
by `fetch_data.sh`: LongBench Pro and MRCR 8-needle prompt files (release assets `data-v1`, gzipped) and the
LiveCodeBench v6 test cases (`livecodebench/code_generation_lite` at a pinned revision).
