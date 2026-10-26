# mix60b: Qwen3-30B-A3B with the long-context stage first, and the updated customer-service environment

**What to run: only Qwen3-30B-A3B is trained again.** The Nemotron Nano 9B v2 and Nemotron 3 Nano runs of mix60 are
not repeated and keep their mix60 results. No new data: the lists, scenario cards and argument index are mix60's files,
already on disk after mix60 (`setup/setup.sh data` reports them as present).

The same data, settings and model as `experiments/mix60`'s Qwen3-30B-A3B run, with two changes:

1. **The stage order is reversed.** In mix60 the long-context stage came after the 240 mixed steps; in that order the
   long-context stage took back much of what the other domains had gained. mix60b trains long context **first**, from
   the NVFP4 start, and the mixed list **second**, from the long-context stage's last step.
2. **The customer-service environment is updated** (`base_mt.toml`): mix60's environment plus three switches on the
   customer-service entries (all off by default in `train/onpolicy/multiturn.py`, so mix60 and round 4 are unchanged):
   - `code_v3`: the one-time code the recorded customer received stays hidden on the scenario card until a send
     succeeds in the episode, and is given to the customer after it (customers had offered a card code before any send,
     or said that a sent code never arrived);
   - `log_v3`: rejected customer messages and the reasons are logged with the episode (logging only);
   - `close_v4`: the customer's end. There is no "the representative awaits your answer" gate any more; every end (a
     stop marker or a closing message) goes to a narrowed goal audit, which refuses an end only when it can name a card
     goal that is still open. The episode ends exactly when the audit and every other check allowed the end, and an
     allowed end skips the customer reviewer. Customers had kept going after their tasks were done (repeating done
     items, asking for a code again) until the turn limit. The tool side, the fact check and both reviewers are
     unchanged.

   This environment was compared with mix60's by a blind paired reading of the same conversations (two readers,
   disagreements adjudicated). Customers who did not close fell by about half and turn-limit cuts by 40 %; severe
   episodes were equal, and less training fell on turns after a first material problem. Tasks completed (as judged by
   the readers) were lower: in 21 of the 34 conversations with fewer tasks done the readers found no environment
   problem (most of them the student handing over to a human agent itself).

| Config | List | Steps (decay) | Start | Exported | Learning rate |
|---|---|---|---|---|---|
| `q30_stage1.toml` | `mix_r4_s2_long_q30_s20261021`: long context (YaRN x4, 131,072) | 60 (last 15) | the NVFP4 start | `mix60b-q30-s060` (not evaluated) | 1e-5 → 1e-6 |
| `q30_stage2.toml` | `mix60_s1_q30_s20261023`: reasoning, code, function calling, customer service (native RoPE, 40,960) | 240 (last 40) | stage 1's step 60 (new optimizer and schedule) | `mix60b-q30-s300` | 1e-5 → 1e-6 |

Everything else is mix60's: the lists, scenario cards and argument index (no new data), `../mix60/base.toml`, batch
and pack sizes, sampling, the WSD schedule with a 5-step warmup, the optimizer offload on one node. Do not change them
(CLAUDE.md, rule 6). The only free knobs are the layout (`--layout`) and the output directory (`--out`).

## Requirements

As `experiments/mix60/README.md`, "Requirements", for its Qwen3-30B-A3B run. The simulator API variables are needed for
stage 2 and `smoke_q30.toml` only (stage 1 has no multi-turn data).

## Running

Pull first (`git status` clean, as mix60's README says). mix60's outputs can stay: the mix60b chains have their own
names (`mix60b-*`).

```bash
setup/setup.sh optional-models data      # nothing new to download after mix60
python3 tools/check_env.py --deep        # all ok (host warnings only)
(cd train && ../.venv-client/bin/python tests/test_chain_mix60b.py && ../.venv-client/bin/python tests/test_multiturn_v4.py)
export QADX_API_BASE=... QADX_API_KEY=... QADX_SIM_MODEL=...   # the training simulator (gpt-oss-120b): stage 2 and smoke_q30
E=experiments/mix60b
python3 train/launch/chain.py $E/q30_stage1.toml prepare        # finds mix60's start checkpoint and PTQ student ("exists")
python3 train/launch/chain.py $E/q30_stage1.toml check-step0    # only if it never passed for this model with mix60 or round 4
python3 train/launch/chain.py $E/smoke_q30_long.toml run --layout 1node   # stage-1 path (NVFP4 start, YaRN), 2 steps; smoke checks
python3 train/launch/chain.py $E/smoke_q30.toml run --layout 1node        # stage-2 path (from the smoke's step 2), 2 steps; smoke checks
python3 train/launch/chain.py $E/smoke_q30.toml clean && python3 train/launch/chain.py $E/smoke_q30_long.toml clean   # after both passed
python3 train/launch/chain.py $E/q30_stage1.toml run --layout 1node   # 60 steps of long context: exports mix60b-q30-s060
python3 train/launch/chain.py $E/q30_stage2.toml run --layout 1node   # 240 mixed steps from stage 1's step 60: mix60b-q30-s300
```

The step-0 check uses the same start checkpoint and PTQ student as mix60 and round 4: if it passed there, it need not be
repeated. Otherwise run it and go on only with `VERDICT PASS` in `out/check-step0.txt` (pass criteria:
`experiments/mix60/README.md`, "Order of work").

**Smoke checks.** The five checks of `experiments/mix60/README.md` ("Smoke runs, what to check"), with these names and
these differences (the stage roles are swapped):

| Config | `N` | `S` |
|---|---|---|
| `smoke_q30_long.toml` (run first) | `smoke-mix60b-q30-long` | `smoke-mix60b-q30-long-s002` |
| `smoke_q30.toml` (run second) | `smoke-mix60b-q30` | `smoke-mix60b-q30-s002` |

- Check 3, weight push:
  - `smoke_q30_long` starts from the checkpoint on disk, so the push count is **8**. Step 1 trains on rollouts of the
    disk checkpoint (YaRN servers), step 2 on rollouts after the first push. Pass: step 2 against step 1 as in mix60,
    and in both steps `nll_s` below 1.0 and `top1` above 0.8 (servers without YaRN give values far outside these).
  - `smoke_q30` starts from `smoke_q30_long`'s weights, which are pushed before step 0, so the count is **12** and all
    rollouts come after a push. Pass: step 2 against step 1 as in mix60.
- Check 4, multi-turn episodes: `smoke_q30` only (`smoke_q30_long` has none). With this environment a Qwen3-8B
  reference on 100 customer-service conversations ended 40 % `complete`, 36 % `transfer call`, 10 % `max_user_turns`,
  5 % `stopped: loop`, 4 % `transfer`, 3 % `user_invalid`, the rest `stopped: ...`. An episode may now end `complete`
  on a closing message while the representative's last message asked a question; that is the new end rule. Stop and
  report as mix60 says (more than a quarter `stopped: unparsable or empty reply` or `dropped: ...`).
- The memory log and checks 1, 2 and 5 as in mix60, with `N` and `S` from the table.

Send the output of the five checks for both smoke runs, also when all pass. The reminders and `CHAIN_STOP` handling of
mix60's README apply (replace the mix60 names by these).

**Disk.** `clean` a chain once its exports exist. Run `clean` of `q30_stage1.toml` only after stage 2 has started, that
is once `out/mix60b-q30.chain.log` shows `INIT_CKPT_OK iter_0000060` (stage 2 lays out its start from stage 1's step-60
checkpoint into `out/train/mix60b-q30-s240.init` when it launches). Smoke exports are not needed after the smokes'
`clean`: then `rm -rf out/exports/smoke-mix60b-*`.

## Evaluation

After stage 2, on the same node, as mix60's evaluation (`experiments/mix60/README.md`, "Evaluation"; the tau2 API:
gpt-5.2 as the customer and judge). Only the final export is evaluated:

```bash
.venv-client/bin/python eval/run.py --arm mix60b-q30-s300 --dry-run
nohup .venv-client/bin/python eval/run.py --arm mix60b-q30-s300 > eval-mix60b-q30.log 2>&1 &
```

## What to report

The **Report** of `mix60b-q30-s300` (`.venv-client/bin/python eval/lib/summary.py --arms mix60b-q30-s300 --report`, every
row as printed); whether both chains finished (`CHAIN_DONE` in `out/mix60b-q30-long.chain.log` and
`out/mix60b-q30.chain.log`); `out/check-step0.txt` if the step-0 check was run; the output of the five smoke checks of
both smoke runs; the τ² API-error counts, as printed; and any deviation from this README.

```bash
grep -E "re-running|left out" eval/results/logs/mix60b-q30-s300_T*.log    # per tau2 run: re-run, and left out of scoring
wc -l eval/results/mix60b-q30-s300/tau2_*.left_out.jsonl                  # still left out now (expected 0)
```
