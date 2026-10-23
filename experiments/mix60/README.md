# mix60: five-domain mixtures with 60 steps of data per domain

On-policy QAD (NVFP4) of **Qwen3-30B-A3B**, **Nemotron Nano 9B v2** and **Nemotron 3 Nano 30B-A3B** on a mixture of
five training domains with 60 training steps of data for every domain, each model with its own list built by the same
method. Weights at the listed steps are exported and evaluated on the five report categories.

| Config | Model | List | Steps (decay) | Exported | Learning rate |
|---|---|---|---|---|---|
| `q30_stage1.toml` | Qwen3-30B-A3B | `mix60_s1_q30_s20261023`: reasoning, code, function calling, customer service (native RoPE, 40,960) | 240 (last 40) | `mix60-q30-s200`, `-s220`, `-s240` | 1e-5 → 1e-6 |
| `q30_stage2.toml` | Qwen3-30B-A3B | `mix_r4_s2_long_q30_s20261021`: long context (YaRN x4, 131,072), from stage 1's step 240 | 60 (last 15) | `mix60-q30-s270`, `-s300` | 1e-5 → 1e-6 |
| `n9.toml` | Nemotron Nano 9B v2 | `mix60_n9_s20261023`: all five domains (131,072) | 300 (last 50) | `mix60-n9-s250`, `-s280`, `-s300` | 1e-6 → 1e-7 |
| `n3.toml` | Nemotron 3 Nano 30B-A3B | `mix60_n3_s20261023`: all five domains (131,072) | 300 (last 50) | `mix60-n3-s250`, `-s280`, `-s300` | 1e-5 → 1e-6 |

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

As `experiments/round4` (models, host and GPU memory, disk), plus:

- `setup/setup.sh data` for the mix60 lists, the scenario cards and the argument index (release `data-v3`).
- **Simulator API** for every config except `q30_stage2.toml` and `smoke_q30_long.toml`: as in `experiments/round4`
  (gpt-oss-120b; variables `QADX_API_BASE`, `QADX_API_KEY` [`QADX_API_KEY_2`], `QADX_SIM_MODEL`, `QADX_SIM_RPS`). A
  customer-service episode makes more simulator calls than in round 4 (reviews and resampling, at medium effort), so a
  second key helps more than before.
- **Disk** under `out/` while a chain trains: as round 4 (Nano 9B v2 about 400 GB; the 30B-class models about 1.2 TB
  with `--layout 1node`, Qwen3-30B-A3B about 1.4 TB while stage 2 runs and stage 1 is not cleaned yet).

## Order of work: one model per node

Three nodes, one model each, all with `--layout 1node`. Run every command from the repository root of that node.

**Before pulling this version**, on a node with an earlier clone:
- Pull only when no chain is running on that node.
- `git status` must show no local changes (see `experiments/round4/README.md`).
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
python3 train/launch/chain.py $E/q30_stage1.toml check-step0    # only if not passed with round 4's version (pass criteria: experiments/round4/README.md)
python3 train/launch/chain.py $E/smoke_q30.toml run --layout 1node        # stage-1 path, 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_q30_long.toml run --layout 1node   # stage-2 path (from the smoke's step 2, YaRN), 2 steps; smoke checks
python3 train/launch/chain.py $E/smoke_q30_long.toml clean && python3 train/launch/chain.py $E/smoke_q30.toml clean   # after both passed
python3 train/launch/chain.py $E/q30_stage1.toml run --layout 1node   # 240 steps: exports mix60-q30-s200, -s220, -s240
python3 train/launch/chain.py $E/q30_stage2.toml run --layout 1node   # 60 steps from stage 1's step 240: mix60-q30-s270, -s300
```

**Node B: Nemotron Nano 9B v2**

```bash
python3 train/launch/chain.py $E/n9.toml prepare
python3 train/launch/chain.py $E/n9.toml check-step0            # only if not passed with round 4's version
python3 train/launch/chain.py $E/smoke_n9.toml run --layout 1node   # 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_n9.toml clean                 # after the checks passed
python3 train/launch/chain.py $E/n9.toml run --layout 1node     # 300 steps: exports mix60-n9-s250, -s280, -s300
```

**Node C: Nemotron 3 Nano 30B-A3B**

```bash
python3 train/launch/chain.py $E/n3.toml prepare
python3 train/launch/chain.py $E/n3.toml check-step0            # only if not passed with round 4's version
python3 train/launch/chain.py $E/smoke_n3.toml run --layout 1node   # 2 steps; then the smoke checks below
python3 train/launch/chain.py $E/smoke_n3.toml clean                 # after the checks passed
python3 train/launch/chain.py $E/n3.toml run --layout 1node     # 300 steps: exports mix60-n3-s250, -s280, -s300
```

Run long commands with `nohup ... &` or in `tmux` and follow `out/<name>.chain.log`. `dryrun` prints every segment's
configuration without starting anything.

**Smoke runs, what to check**: the five checks of `experiments/round4/README.md` ("Smoke runs, what to check"), with
these names, and check 4 as below:

| Config | `N` | `S` |
|---|---|---|
| `smoke_q30.toml` | `smoke-mix60-q30` | `smoke-mix60-q30-s002` |
| `smoke_q30_long.toml` | `smoke-mix60-q30-long` | `smoke-mix60-q30-long-s002` |
| `smoke_n9.toml` | `smoke-mix60-n9` | `smoke-mix60-n9-s002` |
| `smoke_n3.toml` | `smoke-mix60-n3` | `smoke-mix60-n3-s002` |

Check 4, **multi-turn episodes** (not for `smoke_q30_long`):

```bash
python3 -c "import json,sys,collections; print(collections.Counter(str(json.loads(l)['ended'])[:40] for l in open(sys.argv[1])).most_common())" out/runs/$S/episodes.jsonl
grep -c "scenario cards:" out/runs/$S/controller.log    # the card files were read (a line with the number of cards)
```

Pass: most episodes end `complete`, `transfer call` or `transfer`. A Qwen3-8B reference of this environment on 337
conversations of these lists: customer service 52 % `transfer call`, 21 % `complete`, 21 % `max_user_turns`, 3 %
`stopped: loop`, 2 % `stopped: repeated rejected call`, 1 % `transfer`, 1 % `user_invalid`; function calling 87 %
`complete`, the rest `stopped: ...`, `max_turns` and `window`. `stopped: step deadline`, `pending_limit` and
`tool call without a responder` are normal in small numbers. Stop and report if more than a quarter end
`stopped: unparsable or empty reply` or `dropped: ...`.

**Reminders, `CHAIN_STOP` and disk**: as in `experiments/round4/README.md`, with the mix60 names (the smoke exports are
`smoke-mix60-*`; run `clean` of `q30_stage1.toml` only after `out/mix60-q30-long.chain.log` shows
`INIT_CKPT_OK iter_0000240`).

## Evaluation

On the same node, after its chains are done, as in `experiments/round4/README.md` ("Evaluation"), with the mix60
exports:

```bash
.venv-client/bin/python eval/run.py --arm mix60-q30-s200 --arm mix60-q30-s220 --arm mix60-q30-s240 --arm mix60-q30-s270 --arm mix60-q30-s300 --dry-run
nohup .venv-client/bin/python eval/run.py --arm mix60-q30-s200 --arm mix60-q30-s220 --arm mix60-q30-s240 --arm mix60-q30-s270 --arm mix60-q30-s300 > eval-mix60-q30.log 2>&1 &
nohup .venv-client/bin/python eval/run.py --arm mix60-n9-s250 --arm mix60-n9-s280 --arm mix60-n9-s300 > eval-mix60-n9.log 2>&1 &   # node B
nohup .venv-client/bin/python eval/run.py --arm mix60-n3-s250 --arm mix60-n3-s280 --arm mix60-n3-s300 > eval-mix60-n3.log 2>&1 &   # node C
```

## What to report

As in `experiments/round4/README.md` ("What to report"), for the mix60 arms and chains (`out/mix60-q30.chain.log`,
`out/mix60-q30-long.chain.log`, `out/mix60-n9.chain.log`, `out/mix60-n3.chain.log`).
