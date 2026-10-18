# Pairs of training domains (five runs)

On-policy QAD of Qwen3-8B (NVFP4 W4A4, FP8 KV cache) on mixtures of two training domains, each mixture trained as one
WSD chain whose 60-, 120- and 180-step budgets are exported and evaluated on the full benchmark set.

| Config | Mixture (50 / 50 by packed tokens) | Multi-turn | Needs the simulator API |
|---|---|---|---|
| `reas_agent.toml` | reasoning + customer-service conversations | customer service end to end | yes |
| `reas_fc.toml` | reasoning + multi-turn function calling | function calling end to end | yes |
| `code_fc.toml` | code + multi-turn function calling | function calling end to end | yes |
| `code_agent.toml` | code + customer-service conversations | customer service end to end | yes |
| `reas_code.toml` | reasoning + code | none | no |

Shared settings are in `base.toml`. Do not change them (CLAUDE.md, rule 6).

> **Evaluation spec v2 (update).** The four competitions and GPQA Diamond now take 16 samples per question and
> LiveCodeBench v6 takes 8 (were 8, 8 and 4); nothing else in the evaluation or the training changed. After `git pull`,
> rerun the evaluation commands below for every arm already evaluated (the NVFP4 environment check included), exactly
> as before: the changed benchmarks (phases S1, S2, S3) are regenerated from scratch, the files made under the old
> counts are moved to `eval/results/<arm>/superseded/`, and every other phase is kept. Report only results made
> under v2 (the Report line says `evaluation spec v2`). See [`eval/README.md`](../../eval/README.md).

## Design

- **Data:** two domain lists, all of their rows, each in its own order, interleaved so that every prefix holds about half
  the packed training tokens of each domain. The lists are in `data/subsets/<name>/`; `SUBSET.json` describes each one.
- **Training:** on-policy distillation. The NVFP4 student generates its own answers with vLLM at deployment sampling
  (T 0.6, top-p 0.95, top-k 20). The trainer then minimises the full-vocabulary forward KL to the BF16 teacher on those
  answers.
  - The trainer uses a fake-quant NVFP4 student. Its step 0 equals the released `nvidia/Qwen3-8B-NVFP4` checkpoint bit
    for bit.
  - After every step the new weights are pushed into the vLLM servers.
  - 36 packs of 131,072 tokens per step; learning rate 1e-5 after a 5-step warmup.
- **Budgets:** WSD schedule with three budgets.

  | Budget | Decay at the end | Branches from |
  |---|---|---|
  | 60 steps | 15 steps | — |
  | 120 steps | 20 steps | step 45 of the 60-step run |
  | 180 steps | 30 steps | step 100 of the 120-step run |

  Each budget therefore equals a run of that length from scratch, and the chain trains 60 + 75 + 80 = 215 steps in all.
- **Multi-turn data:** played end to end. The student writes every assistant turn; a language model plays the user
  (customer service: from an offline scenario card) and simulates the tools.

## Requirements

- Setup done and `python3 tools/check_env.py` all `ok` (warnings about the host only), on every node used.
- **Simulator API** for the four pairs with multi-turn data, and for `smoke_mt.toml`: an OpenAI-compatible endpoint
  serving **gpt-oss-120b** (reasoning effort low; the launcher sends it). Set:

  ```bash
  export QADX_API_BASE=https://<endpoint>/v1   # must end in /v1
  export QADX_API_KEY=...                      # a second key in QADX_API_KEY_2 doubles the throughput
  export QADX_SIM_MODEL=<the endpoint's model id for gpt-oss-120b>    # default "openai/gpt-oss-120b"
  export QADX_SIM_RPS=1.0                      # simulator requests per second per key; raise it if the endpoint allows
  ```

  Each multi-turn step makes a few hundred simulator calls. The API rate, not the GPUs, usually sets the step time of
  these four pairs.
- **Disk:** about 500 GB free under `out/` per chain while it trains (two full checkpoints with optimizer state plus the
  weights-only checkpoints of the budgets). Each export is about 6 GB.
- Optional: `WANDB_PROJECT` (plus `WANDB_ENTITY` / `WANDB_API_KEY`) to log training to Weights & Biases.

## Order of work

Run every command from the repository root, on the training node.

```bash
E=experiments/pairs-r2
# 1. Build the PTQ student once (one GPU, ~10 min; downloads cnn_dailymail for a calibration that does not affect the result).
python3 train/launch/chain.py $E/reas_code.toml prepare        # any pair config: same model and checkpoint
# 2. Step 0 must equal the released checkpoint bit for bit (one GPU, ~15 min); read out/check-step0.txt.
#    Pass: every weight, weight_scale, weight_scale_2 and input_scale identical. Only the release has k_scale /
#    v_scale (1.0), and that is expected. Anything else: stop and report.
python3 train/launch/chain.py $E/reas_code.toml check-step0
# 3. Environment check of the evaluation (all 8 GPUs, ~3-4 h; tau2 needs the gpt-5.2 endpoint, see Evaluation):
#    NVIDIA's NVFP4 release on every phase. Send eval/results/ when it is done; we compare it with our numbers.
.venv-client/bin/python eval/run.py --arm nvfp4
# 4. Smoke tests of the launcher, about 1 hour and 15 minutes; they are not results. Free their disk afterwards.
python3 train/launch/chain.py $E/smoke.toml run --layout 1node
python3 train/launch/chain.py $E/smoke_mt.toml run --layout 1node     # needs the simulator API variables
python3 train/launch/chain.py $E/smoke.toml clean; python3 train/launch/chain.py $E/smoke_mt.toml clean
# 5. The five chains, one at a time per node (each holds all 8 GPUs of the layout); after each, evaluate its exports
#    (see Evaluation) and run `clean`.
python3 train/launch/chain.py $E/reas_code.toml run --layout 1node
python3 train/launch/chain.py $E/reas_agent.toml run --layout 1node   # multi-turn: simulator API variables
...
```

With two nodes, run on the training node and add `--layout 2node --peer <rollout host>`.

The rollout node needs:
- passwordless ssh from the training node;
- docker;
- this repository at the same path, with setup done (images, models).

`--peer-ip` sets the address the servers are reached at, if `hostname -I` there does not give it first.

Network between the nodes:
- the rollout node's ports 8021-8028 (vLLM servers, HTTP) must be reachable from the training node;
- the training node's port 29611 (weight-push rendezvous) and NCCL's own ports must be reachable from the rollout node,
  because the new weights go to the servers over NCCL after every step.

`NCCL_*` / `GLOO_*` variables set in the shell that runs `chain.py` are passed into every container, for example
`NCCL_SOCKET_IFNAME` to pick the interface.

`dryrun` writes every segment's controller configuration and trainer command to `out/dryrun/` without starting anything.

### What a run does and leaves behind

- `out/<name>.chain.log`: one line per event. `SEGMENT_DONE` and `EXPORTED` lines mark progress; `CHAIN_STOP` means it
  stopped on a problem.
- `out/runs/<segment>/`: controller logs, one line per step and per rollout (`steps.jsonl`, `rollouts.jsonl`, and
  `episodes.jsonl` for multi-turn data), plus the logs of the controller and the API throttles.
- `out/train/<segment>/checkpoints/`: trainer checkpoints. Pruned to what later budgets need.
- `out/exports/<name>-s060`, `-s120`, `-s180`: Hugging Face NVFP4 checkpoints for evaluation.

### Disk after a chain

When the chain is done, `out/train/<segment>/` still holds the weights-only checkpoints of every budget and the full
checkpoints the branches started from (several hundred GB). Once the three exports exist, `clean` deletes them; it keeps
the controller logs and the exports. Run it before starting the next chain if disk is short:

```bash
python3 train/launch/chain.py $E/<pair>.toml clean
```

### When something stops

- The chain stops at the first failure: a trainer exit other than 0, a missing checkpoint, a server that does not come
  up, or no new step for 120 minutes.
- The state is left as it is. Report the last lines of `out/<name>.chain.log` and of `docker logs qadx-train-<segment>`.
- To continue a segment from its newest full checkpoint:

  ```bash
  python3 train/launch/chain.py $E/<pair>.toml stop --layout 1node
  python3 train/launch/chain.py $E/<pair>.toml resume <segment> <step> --layout 1node   # step: out/train/<segment>/checkpoints/non_persistent/latest_checkpointed_iteration.txt
  python3 train/launch/chain.py $E/<pair>.toml run --layout 1node                      # continues with the remaining segments
  ```

  Do not restart a chain from scratch over an existing `out/` without saying so in the report.

## Evaluation

The evaluation data and the judge image are installed by `setup/setup.sh`. See [`eval/README.md`](../../eval/README.md)
for the τ² API variables. τ² needs an endpoint serving **gpt-5.2**, which is a different model from the training
simulator. The same `QADX_API_*` variables are used, so set them to the gpt-5.2 endpoint while evaluating.

1. **Environment check, once, before the first training run** (step 3 of the order of work). Evaluate NVIDIA's NVFP4
   release on all phases and send the results. We compare them with our numbers for the same checkpoint before the
   training results are interpreted.

   ```bash
   nohup .venv-client/bin/python eval/run.py --arm nvfp4 > eval-nvfp4.log 2>&1 &     # ~3-4 h on 8 GPUs (rerun after the v2 update: ~2.5 h)
   ```

2. **After each chain:** its three exports, all phases.

   ```bash
   P=pair-reas-agent      # the config's name
   nohup .venv-client/bin/python eval/run.py --arm $P-s060=out/exports/$P-s060 --arm $P-s120=out/exports/$P-s120 \
       --arm $P-s180=out/exports/$P-s180 > eval-$P.log 2>&1 &                                    # ~8-10 h on 8 GPUs
   ```

   With two nodes, either evaluate different pairs on different nodes (simplest), or split one job list. To split:
   - copy the exports to the second node at the same path;
   - run the identical command on both, with `--shard 0/2` and `--shard 1/2`, giving each node its own τ² API key;
   - copy the second node's `eval/results/<arm>/` directories into the first node's `eval/results/`;
   - rebuild `summary.md` there with the same command plus `--grade-only`.

The evaluation needs all GPUs of the node. Do not start it while a chain trains on that node.

Rough time per pair on one node: training 1-1.5 days (the four multi-turn pairs depend on the simulator API rate),
then evaluation about half a day. With two nodes, running two pairs at the same time (each on its own node with
`--layout 1node`) finishes the five pairs sooner than running each pair on both nodes.

## What to report

The scores, for the NVFP4 environment check and for each pair's three exports: the **Report** section of
`eval/results/summary.md` (see `eval/README.md`, Outputs), printed for those arms with

```bash
.venv-client/bin/python eval/lib/summary.py --arms nvfp4,$P-s060,$P-s120,$P-s180 --report
```

Every row as printed: the five categories (reasoning, code, long, bfcl, tau2) and the benchmarks behind them to two
decimals, the commit / spec line, and the `complete + graded` row (anything other than `yes` is reported as it is, not
fixed). Recovery is computed from these category scores against the reference PTQ and BF16 baselines, so nothing else
is needed for it. Also report, per pair, whether the chain finished (`CHAIN_DONE` in `out/<name>.chain.log`) and any
deviation from this README.

## What to send back (when files can be sent)

- **Once per node:**
  - `out/check-step0.txt`;
  - `reports/env_report.md` and `.json` (after `python3 tools/check_env.py`);
  - `git rev-parse HEAD` and `git status`.
- **For each pair:** `out/<name>.chain.log`, and `out/runs/*/steps.jsonl` and `out/runs/*/controller.json` of its
  segments.
- **The whole `eval/results/` directory** after each evaluation. It holds every arm's generations, grades, τ² and BFCL
  trajectories, logs, `timing.jsonl` and `summary.md`; send it complete rather than selected files.
- **Anything that deviated from this README**, stated explicitly.
