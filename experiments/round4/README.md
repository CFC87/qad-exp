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
  checkpoint is written before the previous one is removed). Exports: about 17 GB (30B class) and 7 GB (Nano 9B v2) each.
- **Host memory, 30B-class models with `--layout 1node`:** half of the optimizer state lives in CPU memory (the
  `optimizer_overrides` of the config; fp32 numerics kept), about 250 GB on the training node (estimate). With
  `--layout 2node` (8 pipeline stages) the optimizer stays on the GPUs (estimate; the smoke run shows it).

## Order of work: one model per node

Three nodes, one model each, all with `--layout 1node`. Run every command from the repository root of that node. Each
node does its own setup, preparation, step-0 check, smoke run, training and evaluation; the nodes do not depend on each
other.

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
python3 train/launch/chain.py $E/smoke_q30.toml run --layout 1node && python3 train/launch/chain.py $E/smoke_q30.toml clean
python3 train/launch/chain.py $E/q30_stage1.toml run --layout 1node   # 180 steps: exports r4-q30-s150, -s160, -s180
python3 train/launch/chain.py $E/q30_stage2.toml run --layout 1node   # 60 steps from stage 1's step 180: r4-q30-s210, -s240
```

**Node B: Nemotron Nano 9B v2**

```bash
python3 train/launch/chain.py $E/n9.toml prepare                # start checkpoint (CPU, must print "629 / 629 tensors match"), PTQ student
python3 train/launch/chain.py $E/n9.toml check-step0            # one GPU
python3 train/launch/chain.py $E/smoke_n9.toml run --layout 1node && python3 train/launch/chain.py $E/smoke_n9.toml clean
python3 train/launch/chain.py $E/n9.toml run --layout 1node     # 240 steps: exports r4-n9-s200, -s220, -s240
```

**Node C: Nemotron 3 Nano 30B-A3B**

```bash
python3 train/launch/chain.py $E/n3.toml prepare                # start checkpoint (CPU, "24147 / 24147 tensors match"), PTQ student
python3 train/launch/chain.py $E/n3.toml check-step0            # GPUs 0-3
python3 train/launch/chain.py $E/smoke_n3.toml run --layout 1node && python3 train/launch/chain.py $E/smoke_n3.toml clean
python3 train/launch/chain.py $E/n3.toml run --layout 1node     # 240 steps: exports r4-n3-s200, -s220, -s240
```

Run long commands with `nohup ... &` or in `tmux` and follow `out/<name>.chain.log`.

**Step-0 check, pass criteria** (`out/check-step0.txt`): every weight, weight_scale, weight_scale_2 and input_scale
identical, except:
- k_scale / v_scale on one side only (FP8-cast Q/K/V write scale 1.0 tensors that the start checkpoint does not carry);
- Nemotron 3 Nano: the router gates counted "equal in value, other dtype" (exported in fp32, stored in bf16);
- Qwen3-30B-A3B: the routed experts' down_proj input_scale equal to the largest of its layer's experts. The release
  calibrated them per expert; the student has one input quantizer per expert group, and vLLM serves the max of them as
  well. So input_scale shows differences for those experts only.

Anything else: stop and report.

**Smoke run, what to check** (2 steps; not a result): `SEGMENT_DONE` and `EXPORTED` in `out/<smoke name>.chain.log`;
`weight push` lines in `docker logs qadx-train-<segment>` (one per step); for Nemotron 3 Nano `parse service up` in the
chain log; no `CHAIN_STOP`. An out-of-memory error in the trainer on a 30B-class model: stop and report (the CPU offload
fraction is the knob; do not change it yourself).

**Disk.** `clean` a chain once its exports exist. Qwen3-30B-A3B: run `clean` of `q30_stage1.toml` only after stage 2
has started (stage 2 starts from stage 1's step-180 checkpoint, which `clean` removes).

With two nodes for one model: `--layout 2node --peer <rollout host>`, as described in `experiments/pairs-r2/README.md`
(network, ports, `--peer-ip`). Keep one layout for a whole model. `dryrun` prints every segment's configuration without
starting anything. What a run leaves behind and how to resume a stopped segment work as in `experiments/pairs-r2/README.md`.

## Evaluation

Each model is evaluated with the settings of its own reference PTQ and BF16 results, so the recovery of each category is
read against them: see `eval/README.md` for the per-model settings and commands.

## What to report

For each model: the **Report** section of `eval/results/summary.md` for its exported arms (every row as printed), whether
each chain finished (`CHAIN_DONE` in `out/<name>.chain.log`), `out/check-step0.txt` per model, the τ² API-error counts per
arm and domain, and any deviation from this README.
