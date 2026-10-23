# mix60: five-domain mixtures with 60 steps of data per domain

On-policy QAD (NVFP4) of **Qwen3-30B-A3B**, **Nemotron Nano 9B v2** and **Nemotron 3 Nano 30B-A3B** on a mixture of
five training domains with 60 training steps of data for every domain, each model with its own list built by the same
method. The last step of each run is exported and evaluated on the five report categories.

| Config | Model | List | Steps (decay) | Exported | Learning rate |
|---|---|---|---|---|---|
| `q30_stage1.toml` | Qwen3-30B-A3B | `mix60_s1_q30_s20261023`: reasoning, code, function calling, customer service (native RoPE, 40,960) | 240 (last 40) | `mix60-q30-s240` | 1e-5 → 1e-6 |
| `q30_stage2.toml` | Qwen3-30B-A3B | `mix_r4_s2_long_q30_s20261021`: long context (YaRN x4, 131,072), from stage 1's step 240 | 60 (last 15) | `mix60-q30-s300` | 1e-5 → 1e-6 |
| `n9.toml` | Nemotron Nano 9B v2 | `mix60_n9_s20261023`: all five domains (131,072) | 300 (last 50) | `mix60-n9-s300` | 1e-6 → 1e-7 |
| `n3.toml` | Nemotron 3 Nano 30B-A3B | `mix60_n3_s20261023`: all five domains (131,072) | 300 (last 50) | `mix60-n3-s300` | 1e-5 → 1e-6 |

Shared settings are in `base.toml` and `base_mt.toml`. Do not change them (CLAUDE.md, rule 6). The only free knobs are
the layout (`--layout`) and the output directory (`--out`); the ports are fixed in `train/launch/chain.py`.

## Design

- **Data.** The five domains of `experiments/round4`: reasoning (math competition and StackExchange problems, plus a
  fixed science part), code (hard competitive-programming problems once each, plus generated problems), multi-turn
  function calling (one assistant turn generated at a cut point, the conversation before it as history), multi-turn
  customer service (played end to end with a simulated customer and simulated tools, plus 10 % long math rows), and
  long context (documents and retrieval). Each model's list is cut where each domain holds about the packed tokens of
  **60 steps** for that model (round 4: function calling and customer service 30). The selection method is the same
  for every model, not the rows:
  - function calling: the round-4 list was too short for 60 steps, so it is continued in its own order, with at most 80
    conversations per tool set instead of 40 (at 40 every remaining long task was in a full tool set); the rows after
    those of the reference 30-step run (the Nemotron lists: all of them) come in a seeded random order, so the shorter
    continuation rows do not all sit in the decay;
  - customer service: for the two Nemotrons the conversations are selected by the same method in the model's own
    tokens (length bins and the 126,976-token prompt limit) instead of taking the Qwen3 rows;
  - reasoning, code and long context: as round 4.

  The domains are interleaved so that every prefix of a list holds the same fraction of each, followed by a 10 %
  buffer. `SUBSET.json` describes each list.
- **Qwen3-30B-A3B trains in two stages** as in round 4: stage 1 trains the four other domains with the native RoPE,
  stage 2 continues from stage 1's step-240 weights on the long-context data with YaRN (a new optimizer and schedule).
  Long context already had 60 steps in round 4, so stage 2 uses round 4's stage-2 list (the same file). The two
  Nemotrons train all five domains in one run.
- **Multi-turn environment** (`base_mt.toml`, the only place it is set):
  - Customer-service conversations: the simulator plays the customer from the conversation's scenario card and answers
    the tools in a closed world. Its tool results are checked by program rules (schema, ids and values the
    conversation holds, argument errors as the list's recordings show them, `arg_index`) and by a review, and
    resampled when a check fails; its customer messages are checked and reviewed the same way. These calls ask the
    simulator for medium reasoning effort. A role-specific handoff tool (for example a transfer to a specialist) ends
    the episode like a transfer to a human agent. A conversation without a scenario card is left out.
  - Function-calling conversations inside the customer-service portion: user turns replayed, tool results simulated,
    as in round 4.
  - The function-calling portion stays single turn at its cut, as in round 4.
  - An episode stops `deadline_s` (3,600 s) after its rollout step started (round 4: 1,200 s).
  - This environment was chosen by a blind paired reading of the same conversations played in it and in round 4's
    environment, by the same student: for customer service it had fewer severe episodes and less training on turns
    after a first material problem; for the function-calling conversations round 4's environment was kept.
- **Training** as in `experiments/round4`: the NVFP4 student generates at its deployment sampling (Qwen3 and Nano v2
  T 0.6, top-p 0.95; Nemotron 3 Nano T 1.0, top-p 1.0); full-vocabulary forward KL to the BF16 teacher; 36 packs of
  131,072 tokens per step; WSD schedule with a 5-step warmup; new weights pushed into the vLLM servers after every step.
- **Start checkpoints and model-specific parts**: as in `experiments/round4` (the same start checkpoints, PTQ students
  and step-0 checks).

## Requirements

- Setup done and `python3 tools/check_env.py` all `ok` on every node used, plus `setup/setup.sh optional-models data`
  (Qwen3-30B-A3B (+ NVFP4) and both Nemotrons (BF16 + NVFP4); the mix60 lists, the scenario cards and the argument index,
  release `data-v3`) and once `python3 tools/check_env.py --deep` (every model file against Hugging Face's sha256).
- **Simulator API** for every config except `q30_stage2.toml` and `smoke_q30_long.toml`: an OpenAI-compatible endpoint
  serving **gpt-oss-120b** (the launcher sends the reasoning effort: low, medium for the customer-service checks). Set:

  ```bash
  export QADX_API_BASE=https://<endpoint>/v1   # must end in /v1
  export QADX_API_KEY=...                      # a second key in QADX_API_KEY_2 doubles the throughput
  export QADX_SIM_MODEL=<the endpoint's model id for gpt-oss-120b>    # default "openai/gpt-oss-120b"
  export QADX_SIM_RPS=1.0                      # simulator requests per second per key; raise it if the endpoint allows
  ```

  A customer-service episode makes more simulator calls than in round 4 (reviews and resampling, at medium effort), so
  a second key helps more than before.
- **Disk** under `out/` while a chain trains (estimates): Nano 9B v2 about 400 GB; the 30B-class models about 1.2 TB
  with `--layout 1node` (all four pipeline stages write their optimizer state on the training node, and a new full
  checkpoint is written before the previous one is removed), Qwen3-30B-A3B about 1.4 TB while stage 2 runs and stage 1
  is not cleaned yet. Exports: about 17 GB (30B class) and 7 GB (Nano 9B v2) each. Under `models/`: about 300 GB for
  `optional-models`, plus each Nemotron's start checkpoint built by `prepare`.
- **Host memory, 30B-class models with `--layout 1node`:** 70 % of the optimizer state lives in CPU memory (the
  `optimizer_overrides` of the config; fp32 numerics kept). Measured per training process (one pipeline stage, 131,072-token
  packs, 2 steps; process RSS in the trainer's `offload optimizer step` lines, GB = 1e9 bytes): about 105-125 GB while
  training, more while a checkpoint is written, so about 500 GB for the four stages of a node, plus the save.
- **GPU memory, same setting, measured:** the trainer's line `offload optimizer step 2: ... GPU max allocated <x> GB`
  (PyTorch's peak allocation in step 2, GB = 1e9 bytes), per stage: Qwen3-30B-A3B 168.3 / 161.5 / 154.5 / 150.1,
  Nemotron 3 Nano 158.3 / 164.5 / 155.1 / 149.4. The largest, 168.3 GB, is 156.7 GiB of the B200's 179.06 GiB
  (183,359 MiB in `nvidia-smi`, which shows more than PyTorch allocates: the CUDA context and cached blocks).

## Order of work: one model per node

Three nodes, one model each, all with `--layout 1node`. Run every command from the repository root of that node. Each
node does its own setup, preparation, step-0 check, smoke run, training and evaluation; the nodes do not depend on each
other.

**Before pulling this version**, on a node with an earlier clone:
- Pull only when no chain is running on that node. A pull in the middle of a chain mixes two versions of the code in
  one run.
- `git status` must show no local changes. If a file was changed locally, `git stash` before `git pull` and do not
  re-apply the change (`git stash list` keeps it for reference).
- Round 4's outputs can stay: the mix60 chains have their own names (`mix60-*`). `prepare` finds the start checkpoint
  and the PTQ student of round 4 and keeps them; a step-0 check that passed with round 4's version need not be repeated.

**On every node first** (after the general setup of `README.md`):

```bash
setup/setup.sh optional-models data      # the models (already there after round 4) and the mix60 lists, cards and argument index
python3 tools/check_env.py --deep        # all ok (host warnings only)
export QADX_API_BASE=... QADX_API_KEY=... QADX_SIM_MODEL=...   # the training simulator (gpt-oss-120b), see Requirements
E=experiments/mix60
```

**Node A: Qwen3-30B-A3B** (two chains, same `--out`)

```bash
python3 train/launch/chain.py $E/q30_stage1.toml prepare        # skipped parts are logged as "exists"
python3 train/launch/chain.py $E/q30_stage1.toml check-step0    # only if not passed with round 4's version (pass criteria below)
python3 train/launch/chain.py $E/smoke_q30.toml run --layout 1node        # stage-1 path, 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_q30_long.toml run --layout 1node   # stage-2 path (from the smoke's step 2, YaRN), 2 steps; smoke checks
python3 train/launch/chain.py $E/smoke_q30_long.toml clean && python3 train/launch/chain.py $E/smoke_q30.toml clean   # after both passed
python3 train/launch/chain.py $E/q30_stage1.toml run --layout 1node   # 240 steps: exports mix60-q30-s240
python3 train/launch/chain.py $E/q30_stage2.toml run --layout 1node   # 60 steps from stage 1's step 240: mix60-q30-s300
```

**Node B: Nemotron Nano 9B v2**

```bash
python3 train/launch/chain.py $E/n9.toml prepare
python3 train/launch/chain.py $E/n9.toml check-step0            # only if not passed with round 4's version (pass criteria below)
python3 train/launch/chain.py $E/smoke_n9.toml run --layout 1node   # 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_n9.toml clean                 # after the checks passed
python3 train/launch/chain.py $E/n9.toml run --layout 1node     # 300 steps: exports mix60-n9-s300
```

**Node C: Nemotron 3 Nano 30B-A3B**

```bash
python3 train/launch/chain.py $E/n3.toml prepare
python3 train/launch/chain.py $E/n3.toml check-step0            # only if not passed with round 4's version (pass criteria below)
python3 train/launch/chain.py $E/smoke_n3.toml run --layout 1node   # 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_n3.toml clean                 # after the checks passed
python3 train/launch/chain.py $E/n3.toml run --layout 1node     # 300 steps: exports mix60-n3-s300
```

Run long commands with `nohup ... &` or in `tmux` and follow `out/<name>.chain.log`. `dryrun` prints every segment's
configuration without starting anything.

**Step-0 check, pass criteria** (`out/check-step0.txt`). A is the student exported at step 0, B the start checkpoint.
The file has one line per tensor kind (`<kind> N tensors, M bitwise equal`, with padded columns; when some differ,
how much: the largest relative difference of a scale, the share of differing FP8 block scales or FP4 values, and the
worst tensor), for Qwen3-30B-A3B one line `input_scale reference: 18432 routed-expert input scales of B taken as their
layer and projection maximum`, two lines `only in A: <count> (<count per kind>) [first 5 names]` and `only in B: ...`
(with count 0: `only in A: 0 []`), and a last line `VERDICT PASS (...)` or `VERDICT FAIL (...)` with the reasons.
Pass = `VERDICT PASS`. What it checks:
- Nemotron Nano 9B v2, Nemotron 3 Nano: every kind line has M = N (bitwise). Nemotron 3 Nano: the `weight` line
  counts the router gates as "equal in value, other dtype" (exported in fp32, stored in bf16); they are part of M.
- Qwen3-30B-A3B: equal up to rounding, so many tensors are not bitwise equal and that is expected. NVIDIA made this
  release with an older ModelOpt whose scale arithmetic rounds differently in the last bit of fp32 ((amax x 1/6) x 1/448
  where the pinned version computes amax x 1/2688). Every `weight_scale_2` and `input_scale` must be within relative
  2.4e-7 (two fp32 units in the last place); the FP8 block scales and FP4 values that differ at all (by any amount:
  the last-bit difference moves them across a rounding boundary) must be at most 0.2 % of each kind. Its routed
  experts' input scales are compared with the largest one of their layer and projection (gate and up together, down
  alone): the release calibrated them per expert, vLLM's NVFP4 MoE kernels serve one activation scale per layer and
  projection, so the student's experts all take that maximum at train start. Reference run (our B200s, same images and
  pins): `input_scale 18624 tensors, 6715 bitwise equal`, `weight_scale_2 18624 tensors, 6464 bitwise equal` (both at
  most 1.76e-07), `weight_scale 18624 tensors, 17514 bitwise equal` (block scales differing 0.00072), `weight 18867
  tensors, 7664 bitwise equal` (FP4 values differing 0.0008), `input_scale reference: 18432 routed-expert input scales
  ...`, `only in A: 0 []`, `only in B: 96 (k_scale 48, v_scale 48)`, `VERDICT PASS`.
- `only in A` / `only in B`: k_scale and v_scale only, in the per-kind counts (scale-1.0 tensors of the FP8 KV cache
  that one side writes and the other does not).

Anything else, or `VERDICT FAIL`: stop and report.

**Smoke runs, what to check.** Everything in this experiment was tested before release except what needs the GPUs:
training on GPUs 0-3 with four rollout servers on GPUs 4-7; memory on your nodes (measured on our B200s, see
Requirements; the smoke confirms it on yours: the 30B-class models keep 70 % of the optimizer state in CPU memory, and
stage 2 of Qwen3-30B-A3B trains at 131,072 tokens with YaRN); the weight push into the servers after each step; the
multi-turn episodes (for Nemotron 3 Nano through the parse service); and the time per step. A smoke run tests these in
2 steps of the real list. Its numbers are not results. Do all five checks before `clean`, which removes
`out/train/<segment>` and with it `metrics.jsonl`. In the commands, `N` is the smoke's name and `S` its segment:

| Config | `N` | `S` |
|---|---|---|
| `smoke_q30.toml` | `smoke-mix60-q30` | `smoke-mix60-q30-s002` |
| `smoke_q30_long.toml` | `smoke-mix60-q30-long` | `smoke-mix60-q30-long-s002` |
| `smoke_n9.toml` | `smoke-mix60-n9` | `smoke-mix60-n9-s002` |
| `smoke_n3.toml` | `smoke-mix60-n3` | `smoke-mix60-n3-s002` |

While a smoke runs, log memory every 30 s from a second terminal, and stop it with Ctrl-C when the smoke has ended
(columns: time, host memory available in GiB as `free -g` prints it, memory used on GPUs 0-3 in MiB as `nvidia-smi`
prints it):

```bash
N=smoke-mix60-n3   # this node's smoke
while sleep 30; do echo "$(date +%T) $(free -g | awk '/^Mem:/{print $7}') $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -4 | tr '\n' ' ')"; done > out/$N.mem.log
```

After it ends (`N=...; S=...` as in the table):

1. **Finished.**

   ```bash
   grep -E "SEGMENT_DONE|EXPORTED|CHAIN_STOP|parse service" out/$N.chain.log
   ```

   Pass: `SEGMENT_DONE` and `EXPORTED`, no `CHAIN_STOP`. Nemotron 3 Nano: also `parse service up`, and no
   `parse service not answering`.

2. **Memory.**

   ```bash
   awk 'NR==1||$2<h{h=$2} {for(i=3;i<=6;i++) if($i>g) g=$i} END{print "host available min " h " GiB; GPUs 0-3 max " g " MiB"}' out/$N.mem.log
   grep -ciE "out of memory" out/runs/$S/trainer.log             # the count must be 0
   grep "offload optimizer step 2" out/runs/$S/trainer.log       # 30B-class models with --layout 1node: one line per stage
   ```

   Pass: no out-of-memory error. The 30B-class models also print four `offload optimizer step 2` lines (70 % of the
   optimizer state in CPU memory); their `GPU max allocated` and `process RSS` are the measures of the reference
   numbers under Requirements (GB), so compare those with them, not the `nvidia-smi` column. Report before starting
   the run if a training GPU went above 178,000 MiB in the memory log (within about 5 GiB of the 183,359 MiB total) or
   the host had less than 20 GiB available at any point. An out-of-memory error: stop and report. The CPU offload
   fraction is the knob; do not change it yourself.

3. **Weight push.**

   ```bash
   grep -c "weight push: export" out/runs/$S/trainer.log
   python3 -c "import json,sys; [print(r['step'], r['split'], *('%s %.4f' % (f, r.get('kd/%s/all/%s' % (r['split'], f), float('nan'))) for f in ('kl', 'top1', 'nll_s'))) for r in map(json.loads, open(sys.argv[1]))]" out/train/$S/metrics.jsonl
   ```

   Every pipeline rank logs every push, so the count is 4 per push: 8 for a 2-step smoke, 12 for `smoke_q30_long`
   (its start weights are pushed before step 0 as well).

   In `metrics.jsonl`, step 1 trains on rollouts of the checkpoint the servers loaded from disk, and step 2 on
   rollouts of the weights the trainer pushed after step 1. Those are nearly the same weights: in reference runs
   (Qwen3-8B, Nano 9B v2) these values moved by less than 10 % from one step to the next, up to 20 % on long-context
   data (`kl` 0.03-0.06, `top1` 0.93-0.95, `nll_s` 0.20-0.29). Nemotron 3 Nano samples at T 1.0, so its values can
   sit outside these ranges; for every model only the step 2 / step 1 comparison below counts.
   A push that leaves wrong weights on the servers (wrong names, scales or layout) shows as a jump. A push that leaves
   the servers unchanged cannot show in 2 steps.

   Pass: the push count above, and in the `train` rows, step 2 against step 1:
   - `kl` and `nll_s` at most 1.5 times step 1's;
   - `top1` at most 0.02 lower.

   `smoke_q30_long` has no rollouts from disk: all of them come after a push, on long-context data with YaRN. Pass:
   step 2 against step 1 as above, and in both steps `nll_s` below 1.0 and `top1` above 0.8 (Qwen3-8B long-context
   runs with YaRN: `nll_s` 0.23-0.29, `top1` 0.93-0.94). Servers that run a different model than the trainer (for
   example without YaRN) give values far outside these.

4. **Multi-turn episodes**: customer service and function calling (not for `smoke_q30_long`, which has none).

   ```bash
   python3 -c "import json,sys,collections; print(collections.Counter(str(json.loads(l)['ended'])[:40] for l in open(sys.argv[1])).most_common())" out/runs/$S/episodes.jsonl
   grep -c "scenario cards:" out/runs/$S/controller.log    # the card files were read (a line with the number of cards)
   ```

   Pass: most episodes end `complete`, `transfer call` or `transfer`. A Qwen3-8B reference of this environment on 337
   conversations of these lists: customer service 52 % `transfer call`, 21 % `complete`, 21 % `max_user_turns`, 3 %
   `stopped: loop`, 2 % `stopped: repeated rejected call`, 1 % `transfer`, 1 % `user_invalid`; function calling 87 %
   `complete`, the rest `stopped: ...`, `max_turns` and `window`. `stopped: step deadline`, `pending_limit` and
   `tool call without a responder` are normal in small numbers. A few `dropped: env error` are simulator API errors.

   Stop and report if more than a quarter end `stopped: unparsable or empty reply` or `dropped: ...`.
   `dropped: generation error` means the student's server kept failing, or for Nemotron 3 Nano also the parse
   service: send `docker logs qadx-parse` too.

5. **Time per step.**

   ```bash
   python3 -c "import json,sys; t=[r['started'] for r in map(json.loads, open(sys.argv[1])) if r.get('route') == 'pack']; print(' '.join('%.1f min' % ((b - a) / 60) for a, b in zip(t, t[1:])))" out/runs/$S/steps.jsonl
   ```

   This is the time between the first two rollout batches (numbered 0 and 1 in `steps.jsonl`; they are steps 1 and 2
   in `metrics.jsonl`): generation, one training step and the push. Report it. The run takes about its number of steps
   times this; the first step is a little slower.

Send the output of the five checks for every smoke run, also when all pass.

**Reminders.**
- Run one chain at a time per node. Evaluate a model only after its chains are done.
- The smoke runs need the simulator API variables too (all except `smoke_q30_long.toml`).
- Nemotron 3 Nano: the launcher starts and stops the parse service container `qadx-parse` (port 8295). It restarts the
  service by itself if it stops answering. Do not stop it by hand while a chain runs.
- If anything does not match this README (the step-0 check, a smoke check, a `CHAIN_STOP`), stop and report. Do not
  change settings, code or versions to get past it (`CLAUDE.md`, rules 3 and 6).
- A `CHAIN_STOP` leaves the trainer, the servers and the controller running, and `out/runs/<segment>/trainer.log` is
  only written when a segment ends normally. Save the logs first, then free the node:

  ```bash
  docker logs qadx-train-<segment> > out/<segment>.trainer.log 2>&1
  docker logs qadx-parse > out/qadx-parse.log 2>&1                  # Nemotron 3 Nano
  python3 train/launch/chain.py <config> stop --layout 1node        # removes the containers above
  ```

  Send `out/<name>.chain.log`, `out/<segment>.trainer.log` (or `out/runs/<segment>/trainer.log` if it exists),
  `out/qadx-parse.log` for Nemotron 3 Nano, and `reports/env_report.md`.

**Disk.**
- `clean` a chain once its exports exist. It keeps the exports.
- Smoke exports are not needed after the smoke checks and the smokes' `clean` (`clean` refuses while a chain's export
  is missing): then `rm -rf out/exports/smoke-mix60-*` (the one of `smoke_q30_long` is `smoke-mix60-q30-s004`).
- Qwen3-30B-A3B: run `clean` of `q30_stage1.toml` only after stage 2 has started, that is once
  `out/mix60-q30-long.chain.log` shows `INIT_CKPT_OK iter_0000240`. Stage 2 lays out its start from stage 1's step-240
  checkpoint when it launches, into `out/train/mix60-q30-long-s060.init`. A restart of stage 2 still finds it after
  stage 1 is cleaned.
- If stage 2 stops before its first full checkpoint, the chain prints how to start it again. That keeps the `.init`
  start.

With two nodes for one model: `--layout 2node --peer <rollout host>`, as described in `experiments/pairs-r2/README.md`
(network, ports, `--peer-ip`). Keep one layout for a whole model. What a run leaves behind and how to resume a stopped
segment work as in `experiments/pairs-r2/README.md`.

## Evaluation

On the same node, after its chains are done (the training's GPUs and servers must be free). The exports are evaluated
with the current samples per question of `eval/README.md` (competitions and GPQA Diamond 16, LiveCodeBench v6 8);
round-4 results made with the earlier counts are not compared with them. Setup: `eval/README.md`, "Requirements"
(gated sets via `HF_TOKEN`, the tau2 API: **gpt-5.2** as the customer and judge, not the training's simulator model):

```bash
export QADX_API_BASE=... QADX_API_KEY=...   # tau2: an endpoint serving gpt-5.2 (eval/README.md)
# export QADX_API_KEY_2=...                 # optional second key: two tau2 phases at a time
# dry run first (plan, serve commands, first requests; starts nothing), then the run; one model per node:
.venv-client/bin/python eval/run.py --arm mix60-q30-s240 --arm mix60-q30-s300 --dry-run
nohup .venv-client/bin/python eval/run.py --arm mix60-q30-s240 --arm mix60-q30-s300 > eval-mix60-q30.log 2>&1 &
nohup .venv-client/bin/python eval/run.py --arm mix60-n9-s300 > eval-mix60-n9.log 2>&1 &   # node B
nohup .venv-client/bin/python eval/run.py --arm mix60-n3-s300 > eval-mix60-n3.log 2>&1 &   # node C
```

A registered name such as `mix60-n9-s300` means `out/exports/mix60-n9-s300` (the default `--out`). Phases, times, the
second API key and reruns: `eval/README.md`, "Other models".

The Report shows these models' scores without the PTQ (ref) / BF16 (ref) columns and recovery rows: their reference
results in `eval/reference/baselines.json` were made with the earlier samples per question and are not updated.

## What to report

For each model: the **Report** for its exported arms (`.venv-client/bin/python eval/lib/summary.py --arms
mix60-q30-s240,mix60-q30-s300 --report`, likewise `mix60-n9-s300` and `mix60-n3-s300`; every row as printed), whether
each chain finished (`CHAIN_DONE` in `out/mix60-q30.chain.log`, `out/mix60-q30-long.chain.log`, `out/mix60-n9.chain.log`,
`out/mix60-n3.chain.log`), `out/check-step0.txt` per model (if the step-0 check was run for this version), the output of
the five smoke checks of each smoke run, the τ² API-error counts per arm and domain, and any deviation from this README.
The τ² counts, as printed:

```bash
grep -E "re-running|left out" eval/results/logs/mix60-*_T*.log    # per tau2 run: re-run, and left out of scoring
wc -l eval/results/mix60-*/tau2_*.left_out.jsonl                  # still left out now (expected 0)
```
