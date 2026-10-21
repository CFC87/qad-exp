# Round 4: five-domain mixtures on three more models

On-policy QAD (NVFP4) of **Qwen3-30B-A3B**, **Nemotron Nano 9B v2** and **Nemotron 3 Nano 30B-A3B** on a mixture of
five training domains, each model with its own list built by the same method. Weights at the listed steps are exported
and evaluated on the five report categories.

| Config | Model | List | Steps (decay) | Exported | Learning rate |
|---|---|---|---|---|---|
| `q30_stage1.toml` | Qwen3-30B-A3B | `mix_r4_s1_q30_s20261021`: reasoning, code, function calling, customer service (native RoPE, 40,960) | 180 (last 30) | `r4-q30-s150`, `-s160`, `-s180` | 1e-5 → 1e-6 |
| `q30_stage2.toml` | Qwen3-30B-A3B | `mix_r4_s2_long_q30_s20261021`: long context (YaRN x4, 131,072), from stage 1's step 180 | 60 (last 15) | `r4-q30-s210`, `-s240` | 1e-5 → 1e-6 |
| `n9.toml` | Nemotron Nano 9B v2 | `mix_r4_n9_s20261021`: all five domains (131,072) | 240 (last 40) | `r4-n9-s200`, `-s220`, `-s240` | 1e-6 → 1e-7 |
| `n3.toml` | Nemotron 3 Nano 30B-A3B | `mix_r4_n3_s20261021`: all five domains (131,072) | 240 (last 40) | `r4-n3-s200`, `-s220`, `-s240` | 1e-5 → 1e-6 |

Shared settings are in `base.toml` and `base_mt.toml`. Do not change them (CLAUDE.md, rule 6). The only free knobs are
the layout (`--layout`), paths and ports.

## Design

- **Data.** Five domains: reasoning (math competition and StackExchange problems, plus a fixed science part), code
  (hard competitive-programming problems once each, plus generated problems), multi-turn function calling (one
  assistant turn generated at a cut point, the conversation before it as history), multi-turn customer service (played
  end to end with a simulated customer and simulated tools, plus 10 % long math rows), and long context (documents and
  retrieval). The domains are interleaved so that every prefix of a list holds the same fraction of each, followed by a
  10 % buffer. Each model's list is cut where each domain holds about the packed tokens of N steps for that model
  (reasoning 60, code 60, function calling 30, customer service 30, long context 60). `SUBSET.json` describes each list.
- **Qwen3-30B-A3B trains in two stages.** Its long-context data needs YaRN x4, which hurts short inputs, so stage 1 trains
  the four other domains with the native RoPE and stage 2 continues from stage 1's step-180 weights on the long-context
  data with YaRN (a new optimizer and schedule). The two Nemotrons have no such conflict and train all five domains in
  one run.
- **Training** as in `experiments/pairs-r2`: the NVFP4 student generates at its deployment sampling (Qwen3 and Nano v2
  T 0.6, top-p 0.95; Nemotron 3 Nano T 1.0, top-p 1.0); full-vocabulary forward KL to the BF16 teacher; 36 packs of
  131,072 tokens per step; WSD schedule with a 5-step warmup; new weights pushed into the vLLM servers after every step.
- **Start checkpoints.** Qwen3-30B-A3B starts from NVIDIA's release `nvidia/Qwen3-30B-A3B-NVFP4`. Each Nemotron starts
  from a checkpoint that `prepare` builds from the BF16 model: the release's quantized layers, quantized from BF16 at the
  release's static scales, so every tensor QAD does not train equals the release. It is checked tensor by tensor against
  `train/prep/ptqs/<name>/tensors.sha256`.
- **Model-specific parts** (automatic, `train/launch/chain.py` `MODELS`): Nemotron servers keep the Mamba state in
  fp32 and run without prefix caching; Nemotron 3 Nano uses the FlashInfer CUTLASS MoE backend; the MoE models' servers
  load `train/vllm_moe_graph_patch.py` so CUDA graphs stay valid across weight updates; Nemotron 3 Nano's multi-turn
  turns are parsed by vLLM 0.29's own parser, run as a CPU-only service on the training node (`qadx-parse`, port 8295);
  the MoE models keep pack padding out of the router statistics.

## Requirements

- Setup done and `python3 tools/check_env.py` all `ok` on every node used, plus:

  ```bash
  setup/setup.sh optional-models data    # Qwen3-30B-A3B (+ NVFP4) and both Nemotrons (BF16 + NVFP4), and the round-4 lists
  python3 tools/check_env.py --deep      # once: every model file against Hugging Face's sha256
  ```

- **Simulator API** for every config except `q30_stage2.toml`: as in `experiments/pairs-r2` (gpt-oss-120b; variables
  `QADX_API_BASE`, `QADX_API_KEY` [`QADX_API_KEY_2`], `QADX_SIM_MODEL`, `QADX_SIM_RPS`).
- **Disk** under `out/` while a chain trains (estimates): Nano 9B v2 about 400 GB; the 30B-class models about 1.2 TB
  with `--layout 1node` (all four pipeline stages write their optimizer state on the training node, and a new full
  checkpoint is written before the previous one is removed), Qwen3-30B-A3B about 1.4 TB while stage 2 runs and stage 1
  is not cleaned yet. Exports: about 17 GB (30B class) and 7 GB (Nano 9B v2) each. Under `models/`: about 300 GB for
  `optional-models`, plus each Nemotron's start checkpoint built by `prepare`.
- **Host memory, 30B-class models with `--layout 1node`:** half of the optimizer state lives in CPU memory (the
  `optimizer_overrides` of the config; fp32 numerics kept), about 250 GB on the training node (estimate). With
  `--layout 2node` (8 pipeline stages) the optimizer stays on the GPUs (estimate; the smoke run shows it).

## Order of work: one model per node

Three nodes, one model each, all with `--layout 1node`. Run every command from the repository root of that node. Each
node does its own setup, preparation, step-0 check, smoke run, training and evaluation; the nodes do not depend on each
other.

**Before pulling this version**, on a node that already has a clone (for example from `experiments/pairs-r2`):
- Pull only when no chain is running on that node. A pull in the middle of a chain mixes two versions of the code in
  one run.
- `git status` must show no local changes. If `train/launch/chain.py` or another file was changed locally, `git stash`
  before `git pull` and do not re-apply the change.
  - For example, a local fix to the removal of old saves while a chain trains. That removal stopped a chain's first
    segment after its first save. This version fixes it.
  - `git stash list` keeps the old change for reference.

**On every node first** (after the general setup of `README.md`):

```bash
setup/setup.sh optional-models data      # the round-4 models and lists (the gpt-oss-120b download is not needed; let it run)
python3 tools/check_env.py --deep        # all ok (host warnings only); --deep hashes every model file once
export QADX_API_BASE=... QADX_API_KEY=... QADX_SIM_MODEL=...   # the training simulator (gpt-oss-120b), see Requirements
E=experiments/round4
```

**Node A: Qwen3-30B-A3B** (two chains, same `--out`)

```bash
python3 train/launch/chain.py $E/q30_stage1.toml prepare        # the PTQ student (one GPU)
python3 train/launch/chain.py $E/q30_stage1.toml check-step0    # GPUs 0-3; read out/check-step0.txt (pass criteria below)
python3 train/launch/chain.py $E/smoke_q30.toml run --layout 1node        # stage-1 path, 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_q30_long.toml run --layout 1node   # stage-2 path (from the smoke's step 2, YaRN), 2 steps; smoke checks
python3 train/launch/chain.py $E/smoke_q30_long.toml clean && python3 train/launch/chain.py $E/smoke_q30.toml clean   # after both passed
python3 train/launch/chain.py $E/q30_stage1.toml run --layout 1node   # 180 steps: exports r4-q30-s150, -s160, -s180
python3 train/launch/chain.py $E/q30_stage2.toml run --layout 1node   # 60 steps from stage 1's step 180: r4-q30-s210, -s240
```

**Node B: Nemotron Nano 9B v2**

```bash
python3 train/launch/chain.py $E/n9.toml prepare                # start checkpoint (CPU, a few minutes; must log "629 / 629 tensors match"), PTQ student
python3 train/launch/chain.py $E/n9.toml check-step0            # one GPU
python3 train/launch/chain.py $E/smoke_n9.toml run --layout 1node   # 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_n9.toml clean                 # after the checks passed
python3 train/launch/chain.py $E/n9.toml run --layout 1node     # 240 steps: exports r4-n9-s200, -s220, -s240
```

**Node C: Nemotron 3 Nano 30B-A3B**

```bash
python3 train/launch/chain.py $E/n3.toml prepare                # start checkpoint (CPU, up to about 1.5 h on few cores; "24147 / 24147 tensors match"), PTQ student
python3 train/launch/chain.py $E/n3.toml check-step0            # GPUs 0-3
python3 train/launch/chain.py $E/smoke_n3.toml run --layout 1node   # 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_n3.toml clean                 # after the checks passed
python3 train/launch/chain.py $E/n3.toml run --layout 1node     # 240 steps: exports r4-n3-s200, -s220, -s240
```

Run long commands with `nohup ... &` or in `tmux` and follow `out/<name>.chain.log`.

**Step-0 check, pass criteria** (`out/check-step0.txt`): every weight, weight_scale, weight_scale_2 and input_scale
identical, except:
- k_scale / v_scale on one side only (FP8-cast Q/K/V write scale 1.0 tensors that the start checkpoint does not carry);
- Nemotron 3 Nano: the router gates counted "equal in value, other dtype" (exported in fp32, stored in bf16);
- Qwen3-30B-A3B: the routed experts' down_proj input_scale. The release calibrated them per expert; vLLM's NVFP4 MoE
  kernels serve one activation scale per layer and projection, the largest of the experts', so the student's experts
  all take that largest value at train start. Expected: input_scale differs for routed-expert down_proj only (about
  6,100 of 18,624), each equal to the largest down_proj input_scale of its layer in the release.

Anything else: stop and report.

**Smoke runs, what to check.** Everything in this experiment was tested before release except what needs the GPUs:
training on GPUs 0-3 with four rollout servers on GPUs 4-7; memory (the 30B-class models keep half of the optimizer
state in CPU memory; stage 2 of Qwen3-30B-A3B trains at 131,072 tokens with YaRN); the weight push into the servers
after each step; the customer-service episodes (for Nemotron 3 Nano through the parse service); and the time per step.
A smoke run tests these in 2 steps of the real list. Its numbers are not results. Do all five checks before `clean`,
which removes `out/train/<segment>` and with it `metrics.jsonl`. In the commands, `N` is the smoke's name and `S` its
segment:

| Config | `N` | `S` |
|---|---|---|
| `smoke_q30.toml` | `smoke-r4-q30` | `smoke-r4-q30-s002` |
| `smoke_q30_long.toml` | `smoke-r4-q30-long` | `smoke-r4-q30-long-s002` |
| `smoke_n9.toml` | `smoke-r4-n9` | `smoke-r4-n9-s002` |
| `smoke_n3.toml` | `smoke-r4-n3` | `smoke-r4-n3-s002` |

While a smoke runs, log memory every 30 s from a second terminal, and stop it with Ctrl-C when the smoke has ended
(columns: time, host memory available in GB, memory used on GPUs 0-3 in MiB):

```bash
N=smoke-r4-n3   # this node's smoke
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
   awk 'NR==1||$2<h{h=$2} {for(i=3;i<=6;i++) if($i>g) g=$i} END{print "host available min " h " GB; GPUs 0-3 max " g " MiB"}' out/$N.mem.log
   grep -ciE "out of memory" out/runs/$S/trainer.log           # the count must be 0
   grep -m2 "offload optimizer step" out/runs/$S/trainer.log   # 30B-class models: these lines must be there
   ```

   Pass: no out-of-memory error. The 30B-class models also print `offload optimizer step` lines (half of the
   optimizer state in CPU memory). Report before starting the run if a training GPU went above 178,000 MiB (within
   5 GB of the 183,359 MiB total) or the host had less than 20 GB available at any point. An out-of-memory error: stop
   and report. The CPU offload fraction is the knob; do not change it yourself.

3. **Weight push.**

   ```bash
   grep -c "weight push: export" out/runs/$S/trainer.log
   python3 -c "import json,sys; [print(r['step'], r['split'], *('%s %.4f' % (f, r.get('kd/%s/all/%s' % (r['split'], f), float('nan'))) for f in ('kl', 'top1', 'nll_s'))) for r in map(json.loads, open(sys.argv[1]))]" out/train/$S/metrics.jsonl
   ```

   Step 1 trains on rollouts of the checkpoint the servers loaded from disk; step 2 trains on rollouts of the weights
   the trainer pushed after step 1. Two steps barely move the model: from one step to the next these values change by
   about 5-10 %. A larger jump means the servers do not hold the trainer's weights.

   Pass: at least one push line per step, and in the `train` rows, step 2 against step 1:
   - `kl` and `nll_s` at most 1.5 times step 1's;
   - `top1` at most 0.02 lower.

   `smoke_q30_long` is different: every one of its rollouts comes after a push, with YaRN. Compare it with step 1 of
   `smoke_q30` instead: `kl` and `nll_s` at most twice as large, `top1` at most 0.05 lower.

4. **Customer-service episodes** (not for `smoke_q30_long`, which has none).

   ```bash
   python3 -c "import json,sys,collections; print(collections.Counter(str(json.loads(l)['ended'])[:40] for l in open(sys.argv[1])).most_common())" out/runs/$S/episodes.jsonl
   ```

   Pass: most episodes end `complete` or `transfer...`. A reference run had about 1.5 % `stopped: unparsable or empty
   reply`, and a few `dropped: env error` are simulator API errors.

   Stop and report if more than a quarter end `stopped: unparsable or empty reply` or `dropped: ...`. For Nemotron 3
   Nano, `dropped: generation error` means the parse service failed: send `docker logs qadx-parse` too.

5. **Time per step.**

   ```bash
   python3 -c "import json,sys; t=[r['started'] for r in map(json.loads, open(sys.argv[1])) if r.get('route') == 'pack']; print(' '.join('%.1f min' % ((b - a) / 60) for a, b in zip(t, t[1:])))" out/runs/$S/steps.jsonl
   ```

   This is the time from step 0's rollouts to step 1's: generation, one training step and the push. Report it. The run
   takes about its number of steps times this; the first step is a little slower.

Send the output of the five checks for every smoke run, also when all pass.

**Reminders.**
- Run one chain at a time per node. Evaluate a model only after its chains are done.
- The smoke runs need the simulator API variables too (all except `smoke_q30_long.toml`).
- Nemotron 3 Nano: the launcher starts and stops the parse service container `qadx-parse` (port 8295). It restarts the
  service by itself if it stops answering. Do not stop it by hand while a chain runs.
- If anything does not match this README (the step-0 check, a smoke check, a `CHAIN_STOP`), stop and report. Send
  `out/<name>.chain.log`, `out/runs/<segment>/trainer.log` and `reports/env_report.md`. Do not change settings, code
  or versions to get past it (`CLAUDE.md`, rules 3 and 6).

**Disk.** `clean` a chain once its exports exist. Qwen3-30B-A3B: run `clean` of `q30_stage1.toml` only after stage 2
has started (stage 2 lays out its start from stage 1's step-180 checkpoint when it launches; a restart of stage 2 still
finds it after stage 1 is cleaned).

With two nodes for one model: `--layout 2node --peer <rollout host>`, as described in `experiments/pairs-r2/README.md`
(network, ports, `--peer-ip`). Keep one layout for a whole model. `dryrun` prints every segment's configuration without
starting anything. What a run leaves behind and how to resume a stopped segment work as in `experiments/pairs-r2/README.md`.

## Evaluation

On the same node, after its chains are done (the training's GPUs and servers must be free). Each model is evaluated with
the settings of its own reference PTQ and BF16 results, so each category's recovery is read against them; the reference
checkpoints are not run (their scores are in `eval/reference/baselines.json`). Setup: `eval/README.md`, "Requirements"
(gated sets via `HF_TOKEN`, the tau2 API: **gpt-5.2** as the customer and judge, not the training's simulator model):

```bash
export QADX_API_BASE=... QADX_API_KEY=... [QADX_API_KEY_2=...]   # tau2: an endpoint serving gpt-5.2 (eval/README.md)
# dry run first (plan, serve commands, first requests; starts nothing), then the run; one model per node:
.venv-client/bin/python eval/run.py --arm r4-q30-s150 --arm r4-q30-s160 --arm r4-q30-s180 --arm r4-q30-s210 --arm r4-q30-s240 --dry-run
nohup .venv-client/bin/python eval/run.py --arm r4-q30-s150 --arm r4-q30-s160 --arm r4-q30-s180 --arm r4-q30-s210 --arm r4-q30-s240 > eval-r4-q30.log 2>&1 &
nohup .venv-client/bin/python eval/run.py --arm r4-n9-s200 --arm r4-n9-s220 --arm r4-n9-s240 > eval-r4-n9.log 2>&1 &   # node B
nohup .venv-client/bin/python eval/run.py --arm r4-n3-s200 --arm r4-n3-s220 --arm r4-n3-s240 > eval-r4-n3.log 2>&1 &   # node C
```

A registered name such as `r4-n9-s200` means `out/exports/r4-n9-s200`. Phases, times, the second API key and reruns:
`eval/README.md`, "Other models" and "Running the round-4 exports".

## What to report

For each model: the **Report** for its exported arms (`.venv-client/bin/python eval/lib/summary.py --arms
r4-n9-s200,r4-n9-s220,r4-n9-s240 --report`, likewise for the other models; every row as printed), whether
each chain finished (`CHAIN_DONE` in `out/<name>.chain.log`), `out/check-step0.txt` per model, the output of the five
smoke checks of each smoke run, the τ² API-error counts per arm and domain, and any deviation from this README.
Some reference gaps are small (`eval/README.md`, "What to report"); a recovery there is read with the scores next to it.
