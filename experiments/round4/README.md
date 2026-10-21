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

## Order of work

Run every command from the repository root, on the training node.

```bash
E=experiments/round4
# 1. Start checkpoints and PTQ students, once per model (one GPU each; the Nemotron start checkpoint is built on the CPU
#    first and must print "N / N tensors match"; anything else: stop and report).
python3 train/launch/chain.py $E/n9.toml prepare
python3 train/launch/chain.py $E/q30_stage1.toml prepare
python3 train/launch/chain.py $E/n3.toml prepare
# 2. Step 0 must equal the start checkpoint (n9: one GPU; q30 / n3: GPUs 0-3); read out/check-step0.txt after each.
#    Pass: every weight, weight_scale, weight_scale_2 and input_scale identical. Expected and fine: k_scale / v_scale
#    on one side only; Nemotron 3 Nano's router gate counted "equal in value, other dtype"; Qwen3-30B-A3B's routed
#    experts' down_proj input_scale equal to the largest of its layer's experts (the release calibrated them per
#    expert; the student has one input quantizer per expert group and vLLM serves the max of them as well), so input_scale
#    shows differences for those experts only. Anything else: stop and report.
python3 train/launch/chain.py $E/n9.toml check-step0
python3 train/launch/chain.py $E/q30_stage1.toml check-step0
python3 train/launch/chain.py $E/n3.toml check-step0
# 3. Smoke runs (2 steps each, simulator API variables needed); not results. Check in out/<name>.chain.log that the
#    segment finished, in docker logs of the trainer the "weight push" lines, and for n3 the "parse service up" line.
#    Then free their disk.
python3 train/launch/chain.py $E/smoke_n9.toml run --layout 1node
python3 train/launch/chain.py $E/smoke_q30.toml run --layout 1node
python3 train/launch/chain.py $E/smoke_n3.toml run --layout 1node
for m in n9 q30 n3; do python3 train/launch/chain.py $E/smoke_$m.toml clean; done
# 4. The runs, one at a time per node. Qwen3-30B-A3B: stage 2 after stage 1, with the same --out (stage 2 reads stage
#    1's step-180 weights). After each run, evaluate its exports (Evaluation) and run `clean` if disk is short.
python3 train/launch/chain.py $E/n9.toml run --layout 1node
python3 train/launch/chain.py $E/q30_stage1.toml run --layout 1node
python3 train/launch/chain.py $E/q30_stage2.toml run --layout 1node
python3 train/launch/chain.py $E/n3.toml run --layout 1node
```

With two nodes: `--layout 2node --peer <rollout host>`, as described in `experiments/pairs-r2/README.md` (network,
ports, `--peer-ip`). Keep one layout for a whole model: do not train stage 1 and stage 2 of Qwen3-30B-A3B on different
layouts without saying so. `dryrun` prints every segment's configuration without starting anything.

What a run leaves behind, how to resume a stopped segment, and `clean` work as in `experiments/pairs-r2/README.md`.
`clean` of `q30_stage1.toml` must wait until stage 2 has started (stage 2 starts from its step-180 checkpoint).

## Evaluation

Each model is evaluated with the settings of its own reference PTQ and BF16 results, so the recovery of each category is
read against them: see `eval/README.md` for the per-model settings and commands.

## What to report

For each model: the **Report** section of `eval/results/summary.md` for its exported arms (every row as printed), whether
each chain finished (`CHAIN_DONE` in `out/<name>.chain.log`), `out/check-step0.txt` per model, the τ² API-error counts per
arm and domain, and any deviation from this README.
