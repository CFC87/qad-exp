# Evaluation of QAD checkpoints

This harness evaluates checkpoints of four models (BF16, NVFP4 releases, NVFP4 QAD exports) so that the numbers are
directly comparable with the reference results: same serving image and flags, same prompts and chat-template rendering,
same sampling, output caps, context windows and YaRN settings, same number of samples, concurrency and request order,
same graders. Read `CLAUDE.md` first; its rules apply here (nothing below is a free knob unless it says so).

- **Qwen3-8B**: the sections below, up to [Other models](#other-models-qwen3-30b-a3b-nemotron-nano-9b-v2-nemotron-3-nano-30b-a3b).
- **Qwen3-30B-A3B, Nemotron Nano 9B v2, Nemotron 3 Nano 30B-A3B** (the round-4 exports of `experiments/round4`): each
  with its own protocol, the settings of its reference PTQ and BF16 results; see [Other models](#other-models-qwen3-30b-a3b-nemotron-nano-9b-v2-nemotron-3-nano-30b-a3b).
  Everything else (requirements, outputs, tau2 API errors, graders, data) is shared.

`eval/run.py` recognises the model of every arm from its checkpoint's config (model type and hidden size) and runs that
model's protocol (`lib/models.py`, `lib/phases.py`); `--model` only confirms it.

## What runs

One **job** = one checkpoint ("arm") x one **phase**. A phase is one vLLM server on one GPU running one or more
benchmarks; `eval/run.py` runs the jobs over the GPUs of a node, one job per GPU, a GPU taking the next job when free.

| Phase | Benchmarks (`bench:k:cap`) | Concurrency | Window | Typical time on one B200 (QAD checkpoint) |
|---|---|---|---|---|
| S1 | `aime26:16:38912`, `hmmt_feb26:16:38912`, `hmmt_feb25:16:38912`, `hmmt_nov25:16:38912` | 160 | 40,960 | 1.9 h |
| S2 | `gpqa_diamond:16:32768`, `ifbench:4:32768`, `mmlu_redux_1k:1:32768` | 160 | 40,960 | 1.0 h |
| S3 | `lcb_v6:8:32768` | 160 | 40,960 | 0.8 h |
| S4 | `lcb_pro:8:32768` | 160 | 40,960 | 1.2 h |
| L1 | `longbench_pro:1:32768` | 64 | 131,072, static YaRN x4 | 2.4 h |
| M | `mrcr8:8:32768` | 64 | 131,072, static YaRN x4 | 0.5 h |
| T1 | `tau2_retail:4:32768` (4 trials of 114 tasks) | 48 | 40,960 | 0.9 h |
| T2 | `tau2_airline:4:32768`, `tau2_telecom:4:32768` (50 + 114 tasks) | 48 per domain | 40,960 | 1.9 h |
| B | `bfcl_mt:4:32768` (BFCL v4 multi-turn, 4 categories x 200, 4 replicates) | 128 (4 x 32 threads) | 40,960 | 0.65 h |

`k` = samples per question (tau2: trials; BFCL: replicates); the cap is per request (tau2 / BFCL: per model turn).
Times are medians of the reference runs of QAD checkpoints (server start + generation, without grading); NVIDIA's
NVFP4 release takes about 1.3x and BF16 about 1.8x as long (longer outputs); tau2 also depends on the API.
One checkpoint is ~11.3 GPU-hours; on 8 GPUs it finishes in ~2.8 h (T1 and T2 share one API key and run one after the
other) plus grading. With several checkpoints the tau2 phases are usually the bottleneck: they run one at a time per API
key (see `--tau2-per-key`).

**Evaluation spec v2.** The four competitions and GPQA Diamond take 16 samples per question and LiveCodeBench v6 takes
8 (spec v1: 8, 8 and 4); nothing else changed. Results made under v1 are not extended to the new counts. When a phase
runs, a benchmark file made with another k is moved to `<arm>/superseded/k<old>/` (with its grades) and the benchmark
is generated afresh; a phase whose status file was written under v1 (S1, S2, S3) is not counted as complete. So after
`git pull`, running the same `eval/run.py` command again redoes exactly the changed benchmarks of every arm already
evaluated and keeps everything else.

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

- Setup done (`setup/setup.sh`, which also fetches and checks every evaluation file, IFBench's NLTK data and the
  LiveCodeBench Pro test cases, and builds the judge image), and `python3 tools/check_env.py` all `ok` apart from host
  warnings. Grading then needs no Hugging Face token and downloads nothing. The built-in arms `bf16` / `nvfp4` use
  `models/Qwen3-8B` / `models/Qwen3-8B-NVFP4`.
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
  export QADX_TAU2_RPS=1.2                      # optional: requests per second per running tau2 phase (default 1.2; 0 = no throttle)
  # the model id the endpoint receives is $QADX_TAU2_MODEL_PREFIX$QADX_TAU2_MODEL, default "openai/openai/gpt-5.2"
  # (a LiteLLM proxy's id for gpt-5.2); for an endpoint that serves it as plain "gpt-5.2":
  export QADX_TAU2_MODEL_PREFIX=""
  ```
  Do not substitute another model: the user simulator changes tau2 scores. All tau2 API calls go through a local rate
  limiter (`lib/api_throttle.py`) that waits and retries on 429 / 5xx instead of failing the conversation; see
  [tau2 API errors and retries](#tau2-api-errors-and-retries).
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
ended are re-run with their original seeds, a fallback only: see the next section), phases marked complete are not
restarted, grading is redone where the generations are newer. Ctrl-C stops all jobs and their containers. A job is limited to 12 h.

## tau2 API errors and retries

A tau2 score counts every (task, trial) exactly once, with a conversation that ran to its end. An error is handled
according to who caused it.

- **The model under test**: an agent reply with neither text nor a tool call, or a prompt longer than the 40,960 window
  of the local server.
  - The conversation scores 0 and is never re-run.
  - This is also why tau2's `--max-retries` stays 0. tau2's own retry re-runs every failed conversation, the model's own
    failures included, so it can only turn failures into successes.
- **The API side**: the user simulator or the retail judge, with rate limits, timeouts, connection resets or server
  errors.
  - These are retried **per request, inside the conversation**. `lib/api_throttle.py` waits and retries every 429,
    500, 502, 503, 504 and 529 and every connection error or timeout, for up to 3 hours.
  - The conversation only pauses, and the retry does not decide which conversations are kept.
  - Anything between the throttle and the model provider (a proxy or bridge) must do the same for its own transient
    errors, or pass them on with one of those statuses so that the throttle retries them. A transient error that
    reaches tau2 under another status (for example a throttling error returned as 400) ends the conversation.
- **A conversation that an API error still ended** is a fallback case.
  - It is left out of scoring (`<arm>/tau2_<domain>.left_out.jsonl`), and the next `eval/run.py` re-runs it with its
    original seed. Each such re-run leaves a backup `tau2_runs/simulations/<domain>/results.before_rerun_<time>.json`.
  - The re-run keeps the set complete, but the result is no longer equal to a clean run. An API error is more likely to
    hit a long conversation (more calls), so the conversations that end up kept lean towards those that finished without
    one. The more conversations were re-run, the larger the lean.
  - Expected: 0 per arm and domain. Report the number with the scores: conversations ended by an API error, re-run, and
    still left out, per arm and domain.
  - If it is not 0, report it, with the error texts from `left_out.jsonl`, before doing anything else.
    - Transient errors (rate limit, timeout, 5xx, connection): the fix is in the transport, which must retry them. After
      that fix, regenerate the arm's tau2 phases from scratch. Move `tau2_runs/simulations/<domain>/` and
      `tau2_<domain>.*` of that arm to `<arm>/superseded/tau2-<date>/`, then run
      `eval/run.py --arm ... --phases T1,T2 --force`.
    - Other errors (content filter, invalid request): report them with their texts; do not re-run.

## Outputs

```
eval/results/
  summary.md, summary.json         "Report" (the scores to report), scores by benchmark and category per arm
                                   (lib/summary.py), phases complete
  timing.jsonl                     generation time and output tokens per benchmark
  <arm>/arm.json                   checkpoint path and sha256 of its config / quant config / tokenizer config, repo commit,
                                   model (absent: Qwen3-8B)
  <arm>/<bench>.jsonl              one row per sample: qid, j, prompt / completion tokens, finish_reason, text
  <arm>/<bench>.grades.jsonl       one row per sample: qid, j, correct (0/1, or a score in [0, 1]), why
  <arm>/<phase>.status.json        complete or not, what is missing, the phase's tasks and spec version
  <arm>/<bench>.spec               samples per question (k) the benchmark's file was generated with
  <arm>/superseded/k<old>/         files made under an earlier spec (not used)
eval/reference/baselines.json      reference PTQ / BF16 scores by benchmark and category (spec v2), used by the Report
  <arm>/tau2_<domain>.jsonl / .grades.jsonl / .left_out.jsonl, <arm>/tau2_runs/   tau2 rows and full trajectories
  <arm>/bfcl_mt.jsonl / .grades.jsonl, <arm>/bfcl_runs/                          BFCL rows and bfcl's own result files
  logs/scheduler.log, logs/<arm>_<phase>.log, logs/server_<arm>_<phase>.log, logs/tau2_*, logs/bfcl_*, logs/grade_*
```

Scores: benchmark score = mean over questions of the per-question mean over samples. Categories: reasoning (AIME 2026,
three HMMT sets, GPQA; equal weights), code (LiveCodeBench v6, LiveCodeBench Pro without its 5 interactive problems),
long context (LongBench Pro, MRCR), BFCL, tau2 (tasks of the three domains pooled); MMLU-Redux and IFBench are
no-damage checks. Recovery against the baselines is computed from the category scores.

**Report** = the first section of `summary.md`: one column per arm; the five category scores and every benchmark score
to two decimals; the repository commit, the evaluation spec, and a last row saying whether every phase is complete under
the current spec and graded. With `eval/reference/baselines.json` (the reference PTQ = NVIDIA's NVFP4 release and BF16
= Qwen3-8B, evaluated under spec v2 with these settings) it also has the columns `PTQ (ref)` and `BF16 (ref)` and a
`recovery <category>` row per category = (arm − PTQ) / (BF16 − PTQ). Report all of it for every arm, as printed. For a chosen set of arms (writes nothing):
`.venv-client/bin/python eval/lib/summary.py --arms nvfp4,<arm>,... --report`. **When files can be sent**, send back
the whole
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
| tau2 | tau2's reward; empty agent reply / context overflow = 0; API errors retried per request (a conversation one still ended: left out, re-run, reported) | tau2's env |
| BFCL | bfcl's own multi-turn checker | `.venv-bfcl` |

Grading starts after each phase, one grading job at a time; LiveCodeBench Pro is graded at the end, when no server runs
(its judge measures time). LiveCodeBench v6 uses wall-clock test timeouts: grade on a machine that is not overloaded.
IFBench fetches its NLTK data on first use; `grade_ifbench.py --selftest` re-grades IFBench's own sample outputs
(288 samples; 6 mismatches against IFBench's stored files are upstream's and appear in the reference too).

## Other models: Qwen3-30B-A3B, Nemotron Nano 9B v2, Nemotron 3 Nano 30B-A3B

Each of these models is evaluated with the output caps and windows of its reference PTQ and BF16 results in
`eval/reference/baselines.json` (they differ between the models on purpose) and with Qwen3-8B's samples per question
(spec v2: competitions and GPQA Diamond 16, LiveCodeBench v6 8). Only the five report categories run (no IFBench, no
MMLU-Redux). Each model's protocol is version 2 (`summary.md` prints `evaluation spec <model> v2`); its version 1 took 8,
8 and 4 samples, and results made under it are handled as for Qwen3-8B: moved to `<arm>/superseded/k<old>/` and
generated afresh, never extended.

| Phase | Qwen3-30B-A3B | Nemotron Nano 9B v2 | Nemotron 3 Nano 30B-A3B | Concurrency |
|---|---|---|---|---|
| S1 | competitions `:16:38912` | competitions `:16:32768` | competitions `:16:131072` | 160 |
| G | `gpqa_diamond:16:32768` | `gpqa_diamond:16:32768` | `gpqa_diamond:16:131072` | 160 |
| S3 | `lcb_v6:8:32768` | `lcb_v6:8:32768` | `lcb_v6:8:131072` | 160 |
| S4 | `lcb_pro:8:32768` | — | `lcb_pro:8:131072` | 160 |
| L | `longbench_pro:1:32768`, `mrcr8:8:32768` | same | same | 64 |
| T | `tau2_airline:4:32768`, `tau2_retail:4:32768`, `tau2_telecom:4:32768` (one server, the three domains at once) | same | same | 48 per domain |
| B | `bfcl_mt:4:32768` | same | same | 128 (4 x 32 threads) |
| Window | 40,960; L 131,072 with static YaRN x4 | 131,072 (no YaRN) | S1, G, S3, S4 262,144; L, T, B 131,072 (no YaRN) | |

Competitions = `aime26`, `hmmt_feb26`, `hmmt_feb25`, `hmmt_nov25`. Nemotron Nano 9B v2 has no LiveCodeBench Pro phase (it
does not follow that benchmark's C++ instruction), so its code category is LiveCodeBench v6 alone.

Model-specific settings (`lib/models.py`; everything not listed is as for Qwen3-8B):
- **Qwen3-30B-A3B**: Qwen3-8B's protocol (same model card sampling, chat template rendering, hermes / qwen3 parsers, BFCL
  handler `qadx/qwen3-ct`).
- **Nemotron Nano 9B v2**: sampling T 0.6, top-p 0.95, top-k off (-1), min-p 0, everywhere. Server
  `--mamba-ssm-cache-dtype float32`; no prefix caching in any phase. Prompts: NVIDIA's evaluation prompts for the
  competitions, GPQA and LiveCodeBench v6 (NeMo-Skills; rebuilt from the same items, same options and option order); the
  rendered prompt is counted and sent without the tokenizer's BOS (`"add_special_tokens": false`). tau2 and BFCL: NVIDIA's
  tool-call parser `nemotron_json` and a text-based reasoning parser, loaded as vLLM plugins from `lib/vllm_plugins/`
  (mounted read-only into the container); BFCL with the handler `qadx/nemotron-nano-v2-ct`, which uses the server's chat
  endpoint with these parsers (prompt counted by the server's `/tokenize`).
- **Nemotron 3 Nano 30B-A3B**: sampling T 1.0, top-p 1.0, top-k off for the completion benchmarks; T 0.6, top-p 0.95, top-k
  off for tau2 and BFCL (the card's tool-calling setting). Server `--mamba-ssm-cache-dtype float32`, and for NVFP4
  checkpoints `--moe-backend flashinfer_cutlass` (vLLM 0.29's default NVFP4 MoE backend gives wrong outputs for this
  model); no prefix caching in any phase. NVIDIA's evaluation prompts as above (their Nemotron 3 variant, with a system
  message for LiveCodeBench v6), no BOS. tau2 and BFCL: vLLM's `qwen3_coder` tool parser and `nemotron_v3` reasoning
  parser; BFCL handler `qadx/nemotron3-nano-ct` (chat endpoint, as above).

### Running the round-4 exports

The exports of `experiments/round4` are registered by name: `--arm r4-n9-s200` alone means the checkpoint
`out/exports/r4-n9-s200` (give `NAME=PATH` for another location). Run one model per command, from the repository root:

```bash
# dry run first: the plan, every serve command and first request (no GPU, no server)
.venv-client/bin/python eval/run.py --arm r4-n9-s200 --arm r4-n9-s220 --arm r4-n9-s240 --dry-run
nohup .venv-client/bin/python eval/run.py --arm r4-n9-s200 --arm r4-n9-s220 --arm r4-n9-s240 > eval-r4-n9.log 2>&1 &
nohup .venv-client/bin/python eval/run.py --arm r4-n3-s200 --arm r4-n3-s220 --arm r4-n3-s240 > eval-r4-n3.log 2>&1 &
nohup .venv-client/bin/python eval/run.py --arm r4-q30-s150 --arm r4-q30-s160 --arm r4-q30-s180 \
    --arm r4-q30-s210 --arm r4-q30-s240 > eval-r4-q30.log 2>&1 &
```

- Each model needs only its exports (the tokenizer and chat template are read from the export); the reference
  checkpoints are not run. Their scores are in `eval/reference/baselines.json`.
- **tau2 API keys:** each arm's tau2 runs as one phase (T, about 2.9 h). With one key the T phases of a model run one
  after another and are the bottleneck (Qwen3-30B-A3B: 5 x 2.9 h). With `QADX_API_KEY_2` set, the arms alternate
  between the two keys (`(API lane 1)` / `(API lane 2)` in the dry-run plan), so two T phases run at a time.
- Typical times on 8 B200 GPUs (server start and generation; estimated from reference PTQ runs, the competitions, GPQA
  and LiveCodeBench v6 at twice their spec-v1 time): Qwen3-30B-A3B about 13 GPU-hours per arm (L 3.5 h, T 2.9 h, B 1.5 h,
  the rest about 5 h); Nemotron Nano 9B v2 about 10 h per arm; Nemotron 3 Nano about 12 h per arm. A model's arms finish
  in roughly the larger of (GPU-hours / 8) and the tau2 chain.
- `--shard`, `--grade-only`, `--limit`, resuming and the outputs work as for Qwen3-8B. `<arm>/arm.json` records the
  model; an arm name cannot be reused for another model or checkpoint.

### What to report

The **Report** of each model, printed for its arms (one Report per model when arms of several models are given):

```bash
.venv-client/bin/python eval/lib/summary.py --arms r4-n9-s200,r4-n9-s220,r4-n9-s240 --report
```

It shows the model's `PTQ (ref)` and `BF16 (ref)` columns from `eval/reference/baselines.json` when that file holds them
for the model's current spec, the category scores (reasoning, code, long, bfcl, tau2), each category's `recovery` =
(arm - PTQ) / (BF16 - PTQ), every benchmark score with its k, and the `complete + graded` row. Report every row as printed, with the tau2 API-error counts per arm and domain
(see [tau2 API errors and retries](#tau2-api-errors-and-retries)). Some reference gaps are small (Nemotron Nano 9B v2:
reasoning 0.8 points; Nemotron 3 Nano: long context 1.1 points, tau2 BF16 below PTQ), so a recovery there is read with
the scores next to it.

## Tests (CPU only)

```bash
.venv-client/bin/python eval/tests/test_dry_run.py      # scheduler: dry run on 8 GPUs, shards, scheduling loop (after setup: models, eval data)
```

The test also dry-runs every other model whose tokenizer it finds (`models/<BF16 or NVFP4 directory>`, after
`setup/setup.sh optional-models`, or `--tokenizer-dir MODEL=DIR`) and checks that model's phases, windows, server
flags, sampling, prompts, tau2 and BFCL commands; models without a tokenizer are skipped.

Before release this harness was checked against the reference harness the published numbers come from, on the
reference's raw outputs (for Qwen3-8B, and for the other three models on their reference PTQ / BF16 results):
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
