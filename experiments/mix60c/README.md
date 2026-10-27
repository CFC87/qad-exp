# mix60c: the two Nemotrons, 90 more steps after mix60

**What to run: two runs that continue mix60's Nemotron runs**, each on the node of its mix60 run and with the same
`--out`: Nemotron Nano 9B v2 from the step-300 weights of mix60's `n9.toml` run, and Nemotron 3 Nano from those of its
`n3.toml` run. Each trains 90 more steps on a new list of its own. The other runs of mix60, mix60b and mix60b-q8 are not
affected.

| Config | List | Steps (decay) | Start | Exported | Learning rate |
|---|---|---|---|---|---|
| `n9.toml` | `mix60c_n9_s20261027`: all five domains (131,072) | 90 (last 15) | mix60's `mix60-n9-s300` weights (new optimizer and schedule) | `mix60c-n9-s390` | 1e-6 → 1e-7 |
| `n3.toml` | `mix60c_n3_s20261027`: all five domains (131,072) | 90 (last 15) | mix60's `mix60-n3-s300` weights (new optimizer and schedule) | `mix60c-n3-s390` | 1e-5 → 1e-6 |

Everything not listed here is mix60's: `../mix60/base.toml` (batch and pack sizes, sampling, the WSD schedule with a
5-step warmup), each model's start checkpoint and PTQ student, its learning rate and (Nemotron 3 Nano) the optimizer
offload on one node. Do not change them (CLAUDE.md, rule 6). The only free knobs are the layout (`--layout`) and the
output directory (`--out`, which must be the one of the model's mix60 run).

## Design

- **Start.** The weights-only checkpoint of the mix60 run's last step, `out/train/mix60-<m>-s300/checkpoints/iter_0000300`
  (`init_from`, as mix60b's stage 2 starts from its stage 1): a new optimizer and a new WSD schedule, step 0. The
  export `out/exports/mix60-<m>-s300` cannot be used instead: it holds the NVFP4 weights, not the trained BF16 weights.
- **Data.** More data of code and customer service, **30 steps** each; reasoning, function calling and long context
  **10 steps** each, so that they keep being trained. Each model's list continues **its own mix60 list**: every portion
  continues that list's selection in the same order and by the same method, past every row it holds (also its buffer; no
  row of it comes back in the same domain). Steps of data are counted as in mix60 (about the packed tokens of that many
  steps for that model), with a 10 % buffer per portion, and the five portions are interleaved so that every prefix of
  a list holds the same fraction of each.

  | Domain | Steps | Rows (+ buffer): Nano 9B v2 / Nemotron 3 Nano | What the continuation holds |
  |---|---|---|---|
  | code | 30 | 9,587 (+959) / 7,125 (+712) | generated problems only, 70 % with a long reference answer (≥ 12K tokens), 30 % shorter: each mix60 list already used every hard original problem once |
  | customer service | 30 | 3,067 (+307) / 2,309 (+231) | the model's customer-service list in its order: about 58-60 % customer-service conversations, 30-32 % multi-turn function-calling conversations, 10 % long math rows; length and depth bins as mix60's |
  | reasoning | 10 | 2,597 (+260) / 2,338 (+234) | math only: the fixed science part is all in the mix60 lists |
  | function calling | 10 | 2,603 (+260) / 2,462 (+246) | the model's function-calling list in its order past mix60's rows, then shuffled. Nano 9B v2: 94 tool sets, the second 40 conversations of tool sets that had 40 in the list's first 20,000 rows. Nemotron 3 Nano: 754 tool sets (its mix60 list stopped before row 20,000) |
  | long context | 10 | 1,482 (+148) / 1,606 (+161) | the same dataset shares as mix60's |

  `data/subsets/mix60c_<m>_s20261027/SUBSET.json` describes each list; its columns `mix_domain` and `mix_part` name
  each row's portion.
- **Multi-turn environment** (`base_mt.toml`): mix60b's, that is mix60's with `code_v3`, `log_v3` and `close_v4` on the
  customer-service entries (see `experiments/mix60b/README.md`), plus the customer cards of the two lists' new
  customer-service conversations (`data/subsets/mix60c_cards_s20261027/`), loaded after mix60's. Every
  customer-service row of both lists has a card.
- **Training** as mix60's run of each model.

## Requirements

As `experiments/mix60/README.md`, "Requirements", for its Nemotron runs (one node per model, the simulator API
variables, disk, host and GPU memory). Before anything else, on the node of each mix60 run, check that its last
checkpoint is still there (`m` = `n9` or `n3`):

```bash
m=n9   # n3 on the Nemotron 3 Nano node
cat out/train/mix60-$m-s300/checkpoints/latest_checkpointed_iteration.txt   # must print 300
ls -d out/train/mix60-$m-s300/checkpoints/iter_0000300                       # must exist
```

If either fails, **stop and report**: that run cannot start from the export.

## Running

Pull first (`git status` clean, as mix60's README says). On each node, with its model's `m`:

```bash
setup/setup.sh data                       # fetches the two lists and the new card files (release data-v5); the rest prints ok
python3 tools/check_env.py --deep         # all ok (host warnings only)
(cd train && ../.venv-client/bin/python tests/test_chain_mix60c.py && ../.venv-client/bin/python tests/test_multiturn_v4.py)
export QADX_API_BASE=... QADX_API_KEY=... QADX_SIM_MODEL=...   # the training simulator (gpt-oss-120b), as mix60
E=experiments/mix60c; m=n9                                      # m=n3 on the Nemotron 3 Nano node
python3 train/launch/chain.py $E/$m.toml dryrun --layout 1node        # starts nothing; out/dryrun/mix60c-$m-s090/trainer.json has CL_PRETRAINED
python3 train/launch/chain.py $E/smoke_$m.toml run --layout 1node     # 2 steps from mix60-$m-s300; smoke checks below
python3 train/launch/chain.py $E/$m.toml run --layout 1node           # 90 steps: exports mix60c-$m-s390
```

The step-0 check and `prepare` are mix60's (the same start checkpoints and PTQ students); they need not be repeated.

**Smoke checks.** The five checks of `experiments/mix60/README.md` ("Smoke runs, what to check") with
`N=smoke-mix60c-$m` and `S=smoke-mix60c-$m-s002`, and these differences:

- Check 1, finished: also `INIT_CKPT_OK iter_0000300` in `out/smoke-mix60c-$m.chain.log`. Nemotron 3 Nano: also the
  parse-service lines as in mix60.
- Check 3, weight push: the start weights are pushed before step 0, so the count is **12** and all rollouts come after a
  push. Pass: step 2 against step 1 as in mix60.
- Check 4, multi-turn episodes: with this environment a Qwen3-8B reference on 100 customer-service conversations ended
  40 % `complete`, 36 % `transfer call`, 10 % `max_user_turns`, 5 % `stopped: loop`, 4 % `transfer`, 3 % `user_invalid`,
  the rest `stopped: ...`. An episode may end `complete` on a closing message while the representative's last message
  asked a question; that is the end rule of `close_v4`. Stop and report as mix60 says (more than a quarter
  `stopped: unparsable or empty reply` or `dropped: ...`). The line `scenario cards:` of `controller.log` counts the
  cards read, and no customer-service row may lack one:
  `grep -c '"mt_no_card"' out/runs/$S/steps.jsonl` must print 0.
- The memory log and checks 2 and 5 as in mix60 (Nemotron 3 Nano: the 30B-class lines of check 2).

Send the output of the five checks of each smoke run, also when all pass. The reminders and `CHAIN_STOP` handling of
mix60's README apply (replace the mix60 names by these).

**Disk.** Keep `out/train/mix60-<m>-s300` and `out/train/mix60c-<m>-s090`: they are the trained checkpoints (a later
run starts from them); do not remove them by hand (CLAUDE.md, rule 8).

## Evaluation

After a chain is done, on its node, as mix60's evaluation (`experiments/mix60/README.md`, "Evaluation"; the tau2 API:
gpt-5.2 as the customer and judge). Only the final exports are evaluated:

```bash
.venv-client/bin/python eval/run.py --arm mix60c-$m-s390 --dry-run
nohup .venv-client/bin/python eval/run.py --arm mix60c-$m-s390 > eval-mix60c-$m.log 2>&1 &
```

## What to report

For each model: the **Report** of `mix60c-<m>-s390` (`.venv-client/bin/python eval/lib/summary.py --arms mix60c-<m>-s390
--report`, every row as printed); whether the chain finished (`CHAIN_DONE` in `out/mix60c-<m>.chain.log`); the output of
the five smoke checks; the τ² API-error counts, as printed; and any deviation from this README.

```bash
grep -E "re-running|left out" eval/results/logs/mix60c-*_T*.log    # per tau2 run: re-run, and left out of scoring
wc -l eval/results/mix60c-*/tau2_*.left_out.jsonl                  # still left out now (expected 0)
```
