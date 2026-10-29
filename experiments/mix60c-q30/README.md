# mix60c-q30: Qwen3-30B-A3B, 130 more steps after mix60b

**What to run: one run that continues mix60b's Qwen3-30B-A3B run**, on the node of that run and with the same `--out`:
from the weights of mix60b's stage 2 at its last step (`mix60b-q30-s300` = 60 long-context steps + 240 mixed steps),
130 more steps on a new list. No other run is affected.

| Config | List | Steps (decay) | Start | Exported | Learning rate |
|---|---|---|---|---|---|
| `q30.toml` | `mix60c_q30_s20261027`: customer service, reasoning, code, function calling (native RoPE, 40,960) | 130 (last 25) | mix60b stage 2's `mix60b-q30-s240` weights (new optimizer and schedule) | `mix60c-q30-s430` | 1e-5 → 1e-6 |

Everything not listed here is mix60b's stage 2 (`experiments/mix60b/q30_stage2.toml`): `../mix60/base.toml` (batch and
pack sizes, sampling, the WSD schedule with a 5-step warmup), the start checkpoint and PTQ student, the learning rate,
the validation size and the optimizer offload on one node. Do not change them (CLAUDE.md, rule 6). The only free knobs
are the layout (`--layout`) and the output directory (`--out`, which must be the one of the mix60b run).

## Design

- **Start.** The weights-only checkpoint of mix60b stage 2's last step,
  `out/train/mix60b-q30-s240/checkpoints/iter_0000240` (`init_from`, as mix60b's stage 2 starts from its stage 1): a new
  optimizer and a new WSD schedule, step 0. The segment is named after its own 240 steps; its export carries the total,
  `mix60b-q30-s300`. The export itself cannot be used instead: it holds the NVFP4 weights, not the trained BF16 weights.
  The WSD decay covers the last 25 steps.
- **Data.** **Customer service 60 steps**, **reasoning 30**, **code 30**, **function calling 10**; no long context. The
  list continues **mix60b stage 2's list** (`mix60_s1_q30_s20261023`): every portion continues that list's selection in
  the same order and by the same method, past every row it holds (also its buffer). Steps of data are counted as in
  mix60 (about the packed tokens of that many steps for this model), with a 10 % buffer per portion, and the four
  portions are interleaved so that every prefix of the list holds the same fraction of each.

  | Domain | Steps | Rows (+ buffer) | What the continuation holds |
  |---|---|---|---|
  | customer service | 60 | 7,248 (+725) | the Qwen3 customer-service list in its order: about 58 % customer-service conversations, 32 % multi-turn function-calling conversations, 10 % long math rows |
  | reasoning | 30 | 9,528 (+953) | 90 % math (the math sequence continued) and 10 % science by rows: the fixed science part is all in the mix60 list, so 953 of its science problems come again, in that list's order; the buffer is math |
  | code | 30 | 9,900 (+990) | generated problems only, 70 % with a long reference answer (≥ 12K tokens), 30 % shorter: the mix60 list already used every hard original problem once |
  | function calling | 10 | 4,460 (+446) | first all 1,214 long, deep rows of the mix60 list's function-calling portion (prompt ≥ 16,000 tokens and ≥ 10 turns deep; about 24.6M of the 38.9M packed tokens), taken again; then 3,246 rows of the Qwen3 function-calling list in its order past mix60's rows; shuffled. The list's rows past mix60's are short and shallow (the long, deep ones are selected first), so the long, deep rows come first |

  The science rows and the long, deep function-calling rows are the only rows of the mix60 list that come back in their
  own domain.
  `data/subsets/mix60c_q30_s20261027/SUBSET.json` describes the list; its columns `mix_domain` and `mix_part` name each
  row's portion.
- **Multi-turn environment** (`base_mt.toml`): mix60b's, that is mix60's with `code_v3`, `log_v3` and `close_v4` on the
  customer-service entries (see `experiments/mix60b/README.md`), plus the customer cards of the new customer-service
  conversations (`data/subsets/mix60c_cards_s20261027/`), loaded after mix60's. The list shares conversations with the
  other continuation lists, so their card files are loaded too (experiments/mix60c's, then those of a Qwen3-8B
  continuation list that is not trained here), then this list's. Every customer-service row of the list has a card.
- **Training** as mix60b's stage 2.

## Requirements

As `experiments/mix60/README.md`, "Requirements", for its Qwen3-30B-A3B run (one node, the simulator API variables,
disk, host and GPU memory). Before anything else, on the node of the mix60b run, check that stage 2's last checkpoint is
still there:

```bash
cat out/train/mix60b-q30-s240/checkpoints/latest_checkpointed_iteration.txt   # must print 240
ls -d out/train/mix60b-q30-s240/checkpoints/iter_0000240                       # must exist
```

If either fails, **stop and report**: the run cannot start from the export.

## Running

Pull first (`git status` clean, as mix60's README says). On the node of the mix60b run:

```bash
setup/setup.sh data                       # fetches the list and the new card files (release data-v6); the rest prints ok
python3 tools/check_env.py --deep         # all ok (host warnings only)
(cd train && ../.venv-client/bin/python tests/test_chain_mix60c_q30.py && ../.venv-client/bin/python tests/test_multiturn_v4.py)
export QADX_API_BASE=... QADX_API_KEY=... QADX_SIM_MODEL=...   # the training simulator (gpt-oss-120b), as mix60
E=experiments/mix60c-q30
python3 train/launch/chain.py $E/q30.toml dryrun --layout 1node        # starts nothing; out/dryrun/mix60c-q30-s130/trainer.json has CL_PRETRAINED
python3 train/launch/chain.py $E/smoke_q30.toml run --layout 1node     # 2 steps from mix60b-q30-s240; smoke checks below
python3 train/launch/chain.py $E/q30.toml run --layout 1node           # 130 steps: exports mix60c-q30-s430
```

The step-0 check and `prepare` are mix60's (the same start checkpoint and PTQ student); they need not be repeated.

**Smoke checks.** The five checks of `experiments/mix60/README.md` ("Smoke runs, what to check") with
`N=smoke-mix60c-q30` and `S=smoke-mix60c-q30-s002`, and these differences:

- Check 1, finished: also `INIT_CKPT_OK iter_0000240` in `out/smoke-mix60c-q30.chain.log`.
- Check 3, weight push: the start weights are pushed before step 0, so the count is **12** and all rollouts come after a
  push. Pass: step 2 against step 1 as in mix60.
- Check 4, multi-turn episodes: with this environment a Qwen3-8B reference on 100 customer-service conversations ended
  40 % `complete`, 36 % `transfer call`, 10 % `max_user_turns`, 5 % `stopped: loop`, 4 % `transfer`, 3 % `user_invalid`,
  the rest `stopped: ...`. An episode may end `complete` on a closing message while the representative's last message
  asked a question; that is the end rule of `close_v4`. Stop and report as mix60 says (more than a quarter
  `stopped: unparsable or empty reply` or `dropped: ...`). The line `scenario cards:` of `controller.log` counts the
  cards read, and no customer-service row may lack one:
  `grep -c '"mt_no_card"' out/runs/$S/steps.jsonl` must print 0.
- The memory log and checks 2 and 5 as in mix60 (the 30B-class lines of check 2).

Send the output of the five smoke checks, also when all pass. The reminders and `CHAIN_STOP` handling of mix60's README
apply (replace the mix60 names by these).

**Disk.** Keep `out/train/mix60b-q30-s240` and `out/train/mix60c-q30-s130`: they are the trained checkpoints (a later
run starts from them); do not remove them by hand (CLAUDE.md, rule 8).

## Evaluation

After the chain is done, on its node, as mix60's evaluation (`experiments/mix60/README.md`, "Evaluation"; the tau2 API:
gpt-5.2 as the customer and judge). Only the final export is evaluated:

```bash
.venv-client/bin/python eval/run.py --arm mix60c-q30-s430 --dry-run
nohup .venv-client/bin/python eval/run.py --arm mix60c-q30-s430 > eval-mix60c-q30.log 2>&1 &
```

## What to report

The **Report** of `mix60c-q30-s430` (`.venv-client/bin/python eval/lib/summary.py --arms mix60c-q30-s430 --report`, every
row as printed); whether the chain finished (`CHAIN_DONE` in `out/mix60c-q30.chain.log`); the output of the five smoke
checks; the τ² API-error counts, as printed; and any deviation from this README.

```bash
grep -E "re-running|left out" eval/results/logs/mix60c-q30-s430_T*.log    # per tau2 run: re-run, and left out of scoring
wc -l eval/results/mix60c-q30-s430/tau2_*.left_out.jsonl                  # still left out now (expected 0)
```
