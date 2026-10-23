#!/usr/bin/env python3
"""On-policy QAD training as a WSD chain on one node (8 GPUs) or two nodes (8 + 8 GPUs).

    python3 train/launch/chain.py EXPERIMENT.toml prepare                        # PTQ student (once per model)
    python3 train/launch/chain.py EXPERIMENT.toml check-step0                    # step 0 == the evaluated checkpoint
    python3 train/launch/chain.py EXPERIMENT.toml run   --layout 1node           # train every budget, then export
    python3 train/launch/chain.py EXPERIMENT.toml run   --layout 2node --peer HOST
    python3 train/launch/chain.py EXPERIMENT.toml status | stop | export | resume SEGMENT STEP [layout args]
    python3 train/launch/chain.py EXPERIMENT.toml dryrun --layout 1node             # write configs, print commands
    python3 train/launch/chain.py EXPERIMENT.toml clean                             # after export: drop trainer checkpoints

Layouts (README.md): 1node = GPUs 0-3 train (pipeline parallel 4), GPUs 4-7 serve rollouts (one vLLM server each);
2node = this node's GPUs 0-7 train (pipeline parallel 8), the peer's GPUs 0-7 serve rollouts. Run the script on the
training node. For 2node the peer must be reachable by passwordless ssh, have docker, and hold this repository at the
same path (setup/setup.sh run there too: images, models).

The chain (budgets b_0 < b_1 < ..., decays d_i): segment 0 trains from the PTQ student to b_0 with a WSD schedule that
decays over its last d_0 steps and keeps its full checkpoint (weights + optimizer) at b_0 - d_0; segment i > 0 branches
from segment i-1's kept full checkpoint at b_{i-1} - d_{i-1} and trains to b_i, decaying over its last d_i steps. So
every budget equals a run of that length from scratch while the shared constant-LR part is trained once. The prompt
order continues across a branch exactly (onpolicy/ledger.py). After the last segment every budget is exported to a
Hugging Face NVFP4 checkpoint under OUT/exports/.

A segment that stops (crash, stall) leaves its state as it is: `resume SEGMENT STEP` continues it from its newest full
checkpoint STEP (OUT/train/<segment>/checkpoints/non_persistent/latest_checkpointed_iteration.txt); then run the chain
again.

Outputs (OUT = --out, default <repo>/out): train/<segment>/ (trainer checkpoints), runs/<segment>/ (controller logs:
controller.json, driver.log, rollouts.jsonl, steps.jsonl, episodes.jsonl), exports/<name>-s<budget>/, chain.log.
Files written by containers are owned by root; this script changes them only through a container, so the host user
needs no sudo.
"""
import argparse, json, os, shlex, signal, socket, subprocess, sys, time, tomllib, urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
VERS = tomllib.loads((REPO / "env" / "versions.toml").read_text())
IMG_VLLM = f"{VERS['images']['vllm']['ref']}@{VERS['images']['vllm']['digest']}"
IMG_NEMO = f"{VERS['images']['nemo']['ref']}@{VERS['images']['nemo']['digest']}"
PY = REPO / ".venv-client" / "bin" / "python"
ONPOLICY = REPO / "train" / "onpolicy"
CTL_PORT, SRV_PORT0, THR_PORT0, TORCH_PORT = 18800, 8021, 8290, 29500
YARN = {"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
                            "rope_theta": 1000000, "beta_fast": 32.0, "beta_slow": 1.0, "truncate": True}}
QWEN3_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0}   # each model's deployment sampling
QWEN3_SERVE = ["--kv-cache-dtype", "fp8", "--enable-prefix-caching"]
NEMOTRON_SERVE = ["--mamba-ssm-cache-dtype", "float32", "--no-enable-prefix-caching"]
# Per model (the experiment's `model` = the BF16 folder under models/): layers, the subset's length column for its
# tokenizer and chat template, rollout sampling, vLLM flags, the Megatron PTQ recipe (train/recipes/), and:
#   yarn       Qwen3: a `rope = "yarn"` experiment runs YaRN x4 (servers: --hf-overrides; trainer: a YaRN config copy)
#   tok_ckpt   tokenizer from the experiment's `checkpoint` (Nemotron: the start checkpoint has the same ids and template)
#   moe_graph  vLLM's MoE weight updates under CUDA graphs need train/vllm_moe_graph_patch.py in every server worker
#   pad_id     MoE: the packs' pad id, kept out of the router statistics (qad_distill CL_MOE_PAD_ID)
#   bad_ids    responses are cut at these token ids (untrained embedding rows); mock_vocab: mock-data ids for tests
#   parse      multi-turn turns are parsed by vLLM 0.29's own parser (train/onpolicy/parse_service.py)
#   kv         the export's KV cache algorithm; release: NVIDIA's NVFP4 release whose exclude list the export carries
#   bias_rate0 the router expert bias moves every step whatever the LR: the step-0 check freezes it
#   step0_gpus GPUs (pipeline stages) of the step-0 check: a 30B-class student does not fit one GPU
#   step0_cmp  extra arguments of the step-0 comparison (train/tests/compare_nvfp4_ckpt.py); none = bitwise equality.
#              Qwen3-30B-A3B: NVIDIA's release was made by an older ModelOpt whose scale arithmetic rounds differently
#              in the last bit of fp32 ((amax x 1/6) x 1/448 against the pinned amax x 1/2688), so the step-0 export
#              matches it up to that rounding; its routed experts' input scales are compared with their layer maximum
MODELS = {
    "Qwen3-8B": dict(layers=36, length_col="len_qwen3", sampling=QWEN3_SAMPLING, serve=QWEN3_SERVE,
                     recipe="nvfp4_w4a4-qkv_fp8_cast_mcore.yaml", yarn=True, kv="FP8", step0_gpus=1),
    "Qwen3-30B-A3B": dict(layers=48, length_col="len_qwen3", sampling=QWEN3_SAMPLING, serve=QWEN3_SERVE,
                          recipe="nvfp4_w4a4-qkv_fp8_cast_mcore.yaml", yarn=True, kv="FP8", moe_graph=True, pad_id=151643,
                          step0_gpus=4, step0_cmp=["--expert-input-max", "--scale-rtol", "2.4e-7", "--max-share", "0.002"]),
    "NVIDIA-Nemotron-Nano-9B-v2": dict(layers=56, length_col="len_n9",
                                       sampling={"temperature": 0.6, "top_p": 0.95, "top_k": -1, "min_p": 0.0},
                                       serve=NEMOTRON_SERVE, recipe="nemotron_nano_v2_nvfp4_w4a4_mcore.yaml", tok_ckpt=True,
                                       bad_ids="train/onpolicy/n9_degenerate_ids.json", mock_vocab=1100, kv=None,
                                       release="NVIDIA-Nemotron-Nano-9B-v2-NVFP4", step0_gpus=1),
    "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": dict(layers=52, length_col="len_n3",
                                                sampling={"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0},
                                                serve=NEMOTRON_SERVE + ["--moe-backend", "flashinfer_cutlass"],
                                                recipe="nemotron_3_nano_nvfp4_w4a4_mcore.yaml", tok_ckpt=True, moe_graph=True,
                                                pad_id=11, parse="nemotron_v3", kv="FP8", bias_rate0=True,
                                                release="NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4", step0_gpus=4),
}
PARSE_PORT = 8295


# ----------------------------------------------------------------------------------------------------------- config
def load_toml(path):
    path = Path(path).resolve()
    cfg = tomllib.loads(path.read_text())
    if "include" in cfg:
        base = load_toml(path.parent / cfg.pop("include"))
        for k, v in cfg.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = {**base[k], **v}
            else:
                base[k] = v
        cfg = base
    return cfg


class Chain:
    def __init__(self, exp, args):
        self.cfg = load_toml(exp)
        c = self.cfg
        self.name = c["name"]
        self.budgets, self.decays = list(c["budgets"]), list(c["decays"])
        assert len(self.budgets) == len(self.decays) >= 1, "budgets and decays must have the same length"
        for i, (b, d) in enumerate(zip(self.budgets, self.decays)):
            assert b > d, f"budget {b} is not longer than its decay {d}"
            assert i == 0 or b > self.budgets[i - 1], "budgets must increase"
            assert i == 0 or b - d >= self.budgets[i - 1] - self.decays[i - 1], f"segment {i} would decay before its branch point"
        assert self.budgets[0] - self.decays[0] >= c["warmup"], "the first branch point is inside the warmup"
        self.out = Path(args.out or REPO / "out").resolve()
        self.layout = getattr(args, "layout", None)
        self.peer = getattr(args, "peer", None)
        self.peer_ip = getattr(args, "peer_ip", None)
        self.model = c["model"]
        assert self.model in MODELS, f"model {self.model}: one of {sorted(MODELS)}"
        self.spec = MODELS[self.model]
        self.models = REPO / "models"
        self.subset = REPO / "data" / "subsets" / c["subset"]
        self.ptq = f"ptq-{self.model}-nvfp4-qkv"
        self.yarn = c.get("rope", "plain") == "yarn"
        assert not self.yarn or self.spec.get("yarn"), f"{self.model} has no YaRN mode"
        # the HF folder the trainer (student and teacher structure, weight push) reads: a YaRN config copy for YaRN runs
        self.hf_train = f"/out/hfy/{self.model}" if self.yarn else f"/models/{self.model}"
        # extra weights-only checkpoints kept and exported besides each budget (each a multiple of eval_every inside a
        # segment), and an offset added to the step in export names (a run continuing another one from its step N)
        self.keep_weights = sorted(int(x) for x in c.get("keep_weights", []))
        assert all(x % c["eval_every"] == 0 and 0 < x <= self.budgets[-1] for x in self.keep_weights), "keep_weights"
        self.export_offset = int(c.get("export_offset", 0))
        # init_from: a segment of another chain (same OUT) whose last weights-only checkpoint this chain starts from
        # (new optimizer and schedule, step 0)
        self.init_from = c.get("init_from")
        self.log_path = self.out / f"{self.name}.chain.log"

    # -------------------------------------------------------------------------------------------------- utilities
    def log(self, msg):
        line = f"[{datetime.now(timezone.utc).strftime('%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.out.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def die(self, msg):
        self.log(f"CHAIN_STOP: {msg}")
        sys.exit(1)

    def seg(self, i):
        return f"{self.name}-s{self.budgets[i]:03d}"

    def keep(self, i):  # segment i keeps its full checkpoint here for the next branch (not the last segment)
        return self.budgets[i] - self.decays[i] if i < len(self.budgets) - 1 else None

    def bp(self, i):
        return self.budgets[i] - self.decays[i]

    def kw(self, i):
        """keep_weights steps of segment i (after its branch point, up to its budget)"""
        lo = self.bp(i - 1) if i > 0 else 0
        return [x for x in self.keep_weights if lo < x <= self.budgets[i]]

    def export_name(self, step):  # export_prefix: one name for the exports of a run trained in two chains
        return f"{self.cfg.get('export_prefix', self.name)}-s{step + self.export_offset:03d}"

    def export_steps(self):
        return [(i, x) for i in range(len(self.budgets)) for x in self.kw(i) + [self.budgets[i]]]

    def sched_json(self, i):
        return json.dumps({"lr_decay_style": "WSD", "lr_wsd_decay_style": "minus_sqrt", "lr_wsd_decay_iters": self.decays[i],
                           "start_weight_decay": 0.0, "end_weight_decay": 0.0, "weight_decay": 0.0,
                           "override_opt_param_scheduler": True})

    def layout_spec(self):
        if self.layout == "1node":
            return dict(train_gpus=[0, 1, 2, 3], serve_host=None, serve_ip="127.0.0.1", serve_gpus=[4, 5, 6, 7])
        if self.layout == "2node":
            if not self.peer:
                self.die("2node needs --peer HOST (the rollout node)")
            ip = self.peer_ip or sh(f"ssh {shlex.quote(self.peer)} hostname -I", capture=True).split()[0]
            return dict(train_gpus=list(range(8)), serve_host=self.peer, serve_ip=ip, serve_gpus=list(range(8)))
        self.die("--layout 1node or 2node")

    def pp_split(self, pp):
        n = self.spec["layers"]
        if n % pp == 0:
            return n // pp, n // pp
        for f in range(n // pp, 0, -1):  # equal first / last stages, the middle stages equal and >= them
            if (n - 2 * f) % (pp - 2) == 0 and (n - 2 * f) // (pp - 2) >= f:
                return f, f
        self.die(f"no pipeline split of {n} layers over {pp} stages")

    def root(self, cmd):
        """Run a shell command as root inside the NeMo image with OUT mounted at /out (files written by containers)."""
        return sh(["docker", "run", "--rm", "-v", f"{self.out}:/out", "--entrypoint", "bash", IMG_NEMO, "-c", cmd], capture=True)

    def env_passthrough(self):
        return [a for k, v in os.environ.items() if k.startswith(("NCCL_", "GLOO_")) for a in ("-e", f"{k}={v}")]

    # ------------------------------------------------------------------------------------------------- components
    def nemo_run(self, gpus, name, cmd, env=None, detach=True):
        mo = REPO / "third_party" / "Model-Optimizer"
        sp = "/opt/venv/lib/python3.12/site-packages"
        (REPO / ".cache" / "torch_extensions").mkdir(parents=True, exist_ok=True)
        (REPO / ".cache" / "triton").mkdir(parents=True, exist_ok=True)
        (REPO / ".cache" / "huggingface").mkdir(parents=True, exist_ok=True)
        argv = ["docker", "run", "--name", name, "--gpus", f'"device={",".join(map(str, gpus))}"', "--net=host", "--ipc=host",
                "--ulimit", "memlock=-1", "--ulimit", "stack=67108864",
                "-v", f"{mo}:/opt/Model-Optimizer", "-v", f"{mo}/modelopt:{sp}/modelopt", "-v", f"{mo}/modelopt_recipes:{sp}/modelopt_recipes",
                "-v", f"{REPO / 'train'}:/cl:ro", "-v", f"{self.models}:/models:ro", "-v", f"{self.out / 'train'}:/out",
                "-v", f"{REPO / '.cache' / 'vllm-sender'}:/v029:ro",
                "-v", f"{REPO / '.cache' / 'torch_extensions'}:/root/.cache/torch_extensions",
                "-v", f"{REPO / '.cache' / 'triton'}:/root/.triton", "-v", f"{REPO / '.cache' / 'huggingface'}:/root/.cache/huggingface",
                "-w", "/opt/Model-Optimizer/examples/megatron_bridge", *self.env_passthrough()]
        for k, v in (env or {}).items():
            argv += ["-e", k if v is None else f"{k}={v}"]
        argv += (["-d"] if detach else ["--rm"]) + [IMG_NEMO] + cmd
        (self.out / "train").mkdir(parents=True, exist_ok=True)
        sh(["docker", "rm", "-f", name], check=False, quiet=True)
        return sh(argv, capture=True)

    def servers_up(self, L, window):
        serve, extra = list(self.spec["serve"]), []
        if self.yarn:
            serve += ["--hf-overrides", json.dumps(YARN)]
        if self.spec.get("moe_graph"):  # the patch module, imported in every worker before the model is built
            serve += ["--worker-extension-cls", "vllm_moe_graph_patch.MoeGraphPatchExt"]
            extra = ["-v", f"{REPO / 'train'}:/ops:ro", "-e", "PYTHONPATH=/ops"]
        urls = []
        for j, g in enumerate(L["serve_gpus"]):
            port = SRV_PORT0 + j
            name = f"qadx-vroll-{j}"
            argv = ["docker", "run", "-d", "--name", name, "--gpus", f"device={g}", "--ipc=host", "--net=host",
                    "-e", "VLLM_SERVER_DEV_MODE=1", "-e", "VLLM_HTTP_TIMEOUT_KEEP_ALIVE=60", *self.env_passthrough(), *extra,
                    "-v", f"{self.models}:/models:ro", "-v", f"{REPO / '.cache' / 'vllm' / str(j)}:/root/.cache/vllm",  # one cache per server
                    IMG_VLLM, f"/models/{self.cfg['checkpoint']}", "--served-model-name", "student", "--port", str(port),
                    "--max-model-len", str(window), "--gpu-memory-utilization", "0.88", "--max-num-seqs", str(self.cfg["max_num_seqs"]),
                    *serve, "--weight-transfer-config", '{"backend":"nccl"}']
            self.remote(L, f"docker rm -f {name} >/dev/null 2>&1; mkdir -p {shlex.quote(str(REPO / '.cache' / 'vllm' / str(j)))}; " + shlex.join(argv))
            urls.append(f"http://{L['serve_ip']}:{port}")
        for u in urls:
            for _ in range(160):
                if http_ok(u + "/health"):
                    break
                time.sleep(15)
            else:
                self.die(f"server {u} did not come up (docker logs on {'the peer' if L['serve_host'] else 'this node'})")
        self.log(f"servers up: {','.join(urls)}")
        return urls

    def servers_down(self, L):
        for j in range(len(L["serve_gpus"])):
            self.remote(L, f"docker rm -f qadx-vroll-{j} >/dev/null 2>&1; true", check=False)

    def remote(self, L, cmd, check=True):
        if L["serve_host"]:
            return sh(["ssh", L["serve_host"], cmd], check=check, capture=True)
        return sh(["bash", "-c", cmd], check=check, capture=True)

    def throttles_up(self, run_dir):
        """One rate limiter per API key (onpolicy/sim_throttle.py) for the multi-turn user / tool simulator."""
        if not self.cfg.get("multiturn"):
            return []
        keys = [k for k in (os.environ.get("QADX_API_KEY"), os.environ.get("QADX_API_KEY_2")) if k]
        base = os.environ.get("QADX_API_BASE")
        if not keys or not base:
            self.die("multi-turn data needs QADX_API_BASE and QADX_API_KEY (optionally QADX_API_KEY_2) for the simulator model")
        urls = []
        for i, key in enumerate(keys):
            port = THR_PORT0 + i
            if http_ok(f"http://127.0.0.1:{port}/stats"):
                self.die(f"port {port} is already served (a throttle from an earlier run?): run `stop` first")
            env = dict(os.environ, OPENAI_API_BASE=base, OPENAI_API_KEY=key)
            self.spawn(run_dir, f"throttle{i}", [str(PY), str(ONPOLICY / "sim_throttle.py"), "--port", str(port),
                                                 "--rps", os.environ.get("QADX_SIM_RPS", "1.0")], env=env)
            for _ in range(30):
                if http_ok(f"http://127.0.0.1:{port}/stats"):
                    break
                time.sleep(2)
            else:
                self.die(f"throttle {i} did not come up")
            urls.append(f"http://127.0.0.1:{port}")
        self.log(f"API throttles up: {len(urls)} key(s), {os.environ.get('QADX_SIM_RPS', '1.0')}/s each at most")
        self.parse_up()
        return urls

    def parse_up(self):
        """Multi-turn turns of a model whose evaluation parses them with vLLM 0.29's own parser (Nemotron 3 Nano): that
        parser as a CPU-only service in the serving image on this node (train/onpolicy/parse_service.py), as the
        controller's vLLM 0.25.1 has an older copy of it."""
        if not (self.spec.get("parse") and self.cfg.get("multiturn")):
            return
        url = f"http://127.0.0.1:{PARSE_PORT}"
        if http_ok(url + "/health"):
            self.die(f"port {PARSE_PORT} is already served (a parse service from an earlier run?): run `stop` first")
        sh(["docker", "rm", "-f", "qadx-parse"], check=False, quiet=True)
        sh(["docker", "run", "-d", "--name", "qadx-parse", "--net=host", "--restart", "unless-stopped", "--entrypoint", "python3",
            "-v", f"{self.models / self.cfg['checkpoint']}:/tok:ro", "-v", f"{ONPOLICY}:/ops:ro", IMG_VLLM,
            "/ops/parse_service.py", "--port", str(PARSE_PORT), "--parser", self.spec["parse"]], capture=True)
        for _ in range(60):
            if http_ok(url + "/health"):
                break
            time.sleep(2)
        else:
            self.die("parse service did not come up (docker logs qadx-parse)")
        self.log(f"parse service up: {url} (vLLM 0.29 {self.spec['parse']})")

    def controller_cfg(self, i, run_dir, urls, thr_urls, start_step):
        c = self.cfg
        meta = json.loads((self.subset / "SUBSET.json").read_text())
        mt = {}
        for key, m in (c.get("multiturn") or {}).items():
            m = dict(m)
            m["sim"] = {"url": thr_urls, "model": os.environ.get("QADX_SIM_MODEL", c["simulator"]["model"]),
                        "extra": {"reasoning_effort": c["simulator"]["reasoning_effort"]}, "timeout": 300}
            # the entry's own environment settings (multiturn.py sim_kw, e.g. tool_v2), over the tool-result cap; an
            # arg_index is a data file of this repository
            kw = dict(m.get("sim_kw") or {})
            if kw.get("arg_index"):
                kw["arg_index"] = str(REPO / kw["arg_index"])
            m["sim_kw"] = {"tool_max_tokens": 8192, **kw}
            m["deadline_s"] = c.get("deadline_s", 1200)   # an episode stops this long after its step started
            mt[key] = m
        cards = [str(REPO / p) for p in c.get("scenario_cards", [])]
        cfg = dict(vllm=urls, model="student", W=131072, window=c["window"], n_packs=c["n_packs"], max_packs=c["gbs"],
                   target_packs=c["target_packs"], cap=c["cap"], steps=self.budgets[i], start_step=start_step, port=CTL_PORT,
                   log_dir=str(run_dir), tokenizer_dir=str(self.models / (c["checkpoint"] if self.spec.get("tok_ckpt") else self.model)),
                   add_bos=False, subset=str(self.subset), train_ranks=meta["train_ranks"], val_ranks=meta["val_ranks"],
                   length_col=self.spec["length_col"], val_n=c["val_n"],
                   val_every=c["eval_every"], sample_every=c["eval_every"], n_samples=8, sampling=self.spec["sampling"],
                   expected_response=json.loads((REPO / c["expected_json"]).read_text()),
                   bad_ids_file=str(REPO / self.spec["bad_ids"]) if self.spec.get("bad_ids") else None,
                   hist_sup=bool(c["hist_sup"]), multiturn=mt, scenario_cards=cards or None, pad_id=self.spec.get("pad_id"),
                   parse_url=f"http://127.0.0.1:{PARSE_PORT}" if self.spec.get("parse") and mt else None,
                   wait_init=bool(self.init_from) and i == 0)  # started from a trained checkpoint: step 0 waits for its push
        return cfg

    def controller_up(self, i, run_dir, urls, thr_urls, start_step):
        cfg = self.controller_cfg(i, run_dir, urls, thr_urls, start_step)
        (run_dir / "controller.json").write_text(json.dumps(cfg, indent=1))
        self.spawn(run_dir, "controller", [str(PY), "rollout.py", "--config", str(run_dir / "controller.json")], cwd=ONPOLICY)
        for _ in range(60):
            if http_ok(f"http://127.0.0.1:{CTL_PORT}/status"):
                break
            time.sleep(5)
        else:
            self.die("controller did not come up (runs/<segment>/controller.log)")
        self.log(f"controller up: {http_get(f'http://127.0.0.1:{CTL_PORT}/status')}")

    def spawn(self, run_dir, name, argv, cwd=None, env=None):
        logf = open(run_dir / f"{name}.log", "a")
        p = subprocess.Popen(argv, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        (run_dir / f"{name}.pid").write_text(str(p.pid))

    def kill_spawned(self, run_dir):
        for pf in run_dir.glob("*.pid"):
            try:
                os.killpg(int(pf.read_text()), signal.SIGTERM)
            except (ProcessLookupError, ValueError):
                pass
            pf.unlink()

    def trainer_cmd(self, i, L, urls):
        c, s = self.cfg, self.seg(i)
        pp = len(L["train_gpus"])
        first, last = self.pp_split(pp)
        env = {"HF_HUB_OFFLINE": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "PYTHONPATH": "/v029",
               "CL_ONPOLICY_FEED": f"http://127.0.0.1:{CTL_PORT}", "CL_WEIGHT_PUSH_URL": ",".join(urls),
               "CL_WEIGHT_PUSH_HF": self.hf_train, "CL_WEIGHT_PUSH_SENDER": "0", "CL_CHUNKED_KD": "8192",
               "CL_IMPORT_INPUT_SCALE": f"/models/{c['checkpoint']}", "CL_CKPT_LATEST_FULL": "1",
               "CL_PROVIDER_OVERRIDES": json.dumps({"num_layers_in_first_pipeline_stage": first, "num_layers_in_last_pipeline_stage": last,
                                                    "calculate_per_token_loss": True, "seq_length": 131072}),
               "CL_SCHEDULER_OVERRIDES": self.sched_json(i)}
        if self.keep(i) is not None:
            env["CL_KEEP_FULL_STEPS"] = str(self.keep(i))
        if self.spec.get("pad_id") is not None:
            env["CL_MOE_PAD_ID"] = str(self.spec["pad_id"])
        opt = (c.get("optimizer_overrides") or {}).get(self.layout)
        if opt:  # e.g. partial CPU offload of the optimizer (30B class, 1node)
            env["CL_OPTIMIZER_OVERRIDES"] = json.dumps(opt)
        if self.init_from and i == 0:
            env["CL_PRETRAINED"] = f"/out/{s}.init/checkpoints"
        cmd = ["torchrun", "--nnodes", "1", "--nproc-per-node", str(pp), "--master-addr", "127.0.0.1", "--master-port", str(TORCH_PORT),
               "/cl/qad_distill.py", "--pp_size", str(pp), "--student_hf_path", self.hf_train,
               "--student_megatron_path", f"/out/{self.ptq}", "--teacher_hf_path", self.hf_train, "--use_mock_data",
               "--seq_length", "131072", "--mbs", "1", "--gbs", str(c["gbs"]), "--train_iters", str(self.budgets[i]),
               "--eval_interval", str(c["eval_every"]), "--eval_iters", "1", "--save_interval", str(c["eval_every"]), "--no_async_save",
               "--log_interval", "1", "--lr", str(c["lr"]), "--min_lr", str(c["min_lr"]), "--lr_warmup_iters", str(c["warmup"]),
               "--recompute_granularity", "full", "--recompute_method", "uniform", "--recompute_num_layers", "1", "--output_dir", f"/out/{s}"]
        if os.environ.get("WANDB_PROJECT"):
            cmd += ["--wandb_project", os.environ["WANDB_PROJECT"], "--wandb_exp_name", s]
            if os.environ.get("WANDB_ENTITY"):
                cmd += ["--wandb_entity", os.environ["WANDB_ENTITY"]]
            env["WANDB_API_KEY"] = None   # passed through from this shell's environment (-e NAME), never written out
        return env, cmd, pp, first, last

    def trainer_up(self, i, L, urls, start_step):
        s = self.seg(i)
        env, cmd, pp, first, last = self.trainer_cmd(i, L, urls)
        self.nemo_run(L["train_gpus"], f"qadx-train-{s}", cmd, env=env)
        self.log(f"trainer up: {s}, GPUs {L['train_gpus']}, pipeline parallel {pp} (layers {first} / ... / {last}), "
                 f"start step {start_step}, to {self.budgets[i]}, decay {self.decays[i]}, keep full at {self.keep(i)}")

    def stop_all(self, L=None):
        """Stop this chain's trainer containers, controller and API throttles, and (with a layout) the rollout servers."""
        for i in range(len(self.budgets)):
            sh(["docker", "rm", "-f", f"qadx-train-{self.seg(i)}"], check=False, quiet=True)
            rd = self.out / "runs" / self.seg(i)
            if rd.exists():
                self.kill_spawned(rd)
        sh(["docker", "rm", "-f", "qadx-parse"], check=False, quiet=True)
        if L:
            self.servers_down(L)
        self.log("stopped")

    # -------------------------------------------------------------------------------------------------- segments
    def state(self, i):
        s = self.seg(i)
        if (self.out / "runs" / s / "CHAIN_DONE").exists():
            return "done"
        r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", f"qadx-train-{s}"], capture_output=True, text=True)
        if r.returncode == 0:  # the container exists (docker prints an empty line and exits 1 when it does not)
            return "running" if r.stdout.strip() == "true" else "exited"
        return "orphan" if (self.out / "runs" / s / "steps.jsonl").exists() else "new"

    def preflight(self, L):
        if not (self.out / "train" / self.ptq).exists():
            self.die(f"no PTQ student {self.out / 'train' / self.ptq}: run `prepare` first")
        if not (self.subset / "subset.parquet").exists():
            self.die(f"no subset {self.subset} (setup/setup.sh data)")
        # the controller skips a missing card file without a word, and a customer-service row without its card is
        # dropped, so every listed data file must be there
        data = list(self.cfg.get("scenario_cards", [])) + [m["sim_kw"]["arg_index"] for m in (self.cfg.get("multiturn") or {}).values()
                                                           if (m.get("sim_kw") or {}).get("arg_index")]
        missing = sorted({p for p in data if not (REPO / p).exists()})
        if missing:
            self.die(f"missing data files {missing} (setup/setup.sh data)")
        busy = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l", capture=True, shell=True).strip()
        if busy != "0":
            self.die(f"GPUs of this node are in use ({busy} processes)")
        if L["serve_host"]:
            b = self.remote(L, "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l").strip()
            if b != "0":
                self.die(f"GPUs of the peer are in use ({b} processes)")

    def launch(self, i, L, resume_step=None):
        s = self.seg(i)
        run_dir = self.out / "runs" / s
        if resume_step is not None:
            start, act = resume_step, "resume"
        elif i == 0:
            start, act = 0, "start"
        else:
            start, act = self.bp(i - 1), "branch"
            parent = self.seg(i - 1)
            if (run_dir / "steps.jsonl").exists():
                self.die(f"{run_dir} already has a controller log (an earlier branch attempt?): to retry the branch, run `stop`, "
                         f"remove out/runs/{s} and out/train/{s} (root-owned: docker run --rm -v <out>:/out --entrypoint rm "
                         f"<nemo image> -rf /out/train/{s}), then run the chain again")
            run_dir.mkdir(parents=True, exist_ok=True)
            r = subprocess.run([sys.executable, str(ONPOLICY / "ledger.py"), "branch", str(self.out / "runs" / parent), str(run_dir),
                                str(start), str(self.cfg["eval_every"])], capture_output=True, text=True)
            (run_dir / "driver.log").open("a").write(r.stdout + r.stderr)
            if r.returncode:
                self.die(f"ledger branch failed: {r.stderr[-500:]}")
            out = sh(["docker", "run", "--rm", "-v", f"{self.out}:/out", "--entrypoint", "bash", IMG_NEMO, "-c",
                      branch_cmd(f"/out/train/{parent}/checkpoints", f"/out/train/{s}/checkpoints", start)], capture=True, check=False)
            if "BRANCH_CKPT_OK" not in out:
                self.die(f"branch checkpoint setup failed: {out[-500:]}")
            self.log(out.strip().splitlines()[-1])
        run_dir.mkdir(parents=True, exist_ok=True)
        if self.init_from and i == 0 and resume_step is None:
            out = sh(["docker", "run", "--rm", "-v", f"{self.out}:/out", "--entrypoint", "bash", IMG_NEMO, "-c",
                      init_cmd(f"/out/train/{self.init_from}/checkpoints", f"/out/train/{s}.init/checkpoints")], capture=True, check=False)
            if "INIT_CKPT_OK" not in out:
                self.die(f"start checkpoint from {self.init_from} failed: {out[-500:]}")
            self.log(out.strip().splitlines()[-1])
        if self.yarn:
            self.hfy_up()
        self.preflight(L)
        self.log(f"launch {s} ({act} at step {start}, to step {self.budgets[i]}, decay {self.decays[i]}, keep full at {self.keep(i)})")
        thr = self.throttles_up(run_dir)
        urls = self.servers_up(L, self.cfg["window"])
        self.controller_up(i, run_dir, urls, thr, start)
        self.trainer_up(i, L, urls, start)
        self.log(f"TRAINING_STARTED {s}")

    def hfy_up(self):
        """OUT/train/hfy/<model>: the BF16 model with a YaRN x4 config (max_position_embeddings 131072; beta_fast 32,
        beta_slow 1 and truncate true written out: Megatron-Bridge does not fill in HF's defaults), the other files
        symlinked to /models/<model> (resolved inside the containers), for the trainer's student, teacher and weight push."""
        d = f"/out/train/hfy/{self.model}"
        cfg = json.loads((self.models / self.model / "config.json").read_text())
        cfg["rope_scaling"] = {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768, "beta_fast": 32.0,
                               "beta_slow": 1.0, "truncate": True}
        cfg["max_position_embeddings"] = 131072
        files = [f.name for f in (self.models / self.model).iterdir() if f.is_file() and f.name != "config.json"]
        self.root(f"mkdir -p {d} && cd {d} && " + " && ".join(f"ln -sfn /models/{self.model}/{shlex.quote(f)} {shlex.quote(f)}" for f in files)
                  + f" && printf %s {shlex.quote(json.dumps(cfg, indent=2))} > config.json")
        self.log(f"YaRN config folder {d} ({len(files)} files linked)")

    def wait_done(self, i, L):
        s = self.seg(i)
        f = self.out / "runs" / s / "steps.jsonl"
        last, since, n = -1, time.time(), 0
        stall = self.cfg.get("stall_min", 120) * 60
        parse = f"http://127.0.0.1:{PARSE_PORT}/health" if self.spec.get("parse") and self.cfg.get("multiturn") else None
        parse_down = 0
        while self.state(i) == "running":
            if n % 10 == 0:
                self.trim(i)
            n += 1
            # the parse service (stateless) must stay up: without it every multi-turn episode of the model ends as
            # "dropped: generation error" and is requeued, so the multi-turn domains would silently drop out of the steps
            parse_down = 0 if not parse or http_ok(parse) else parse_down + 1
            if parse_down == 3:
                self.log("parse service not answering for 3 min: restarted (docker logs qadx-parse)")
                sh(["docker", "restart", "qadx-parse"], check=False, quiet=True)
            elif parse_down >= 6:
                self.die("parse service down after a restart (docker logs qadx-parse): the trainer keeps running; report this")
            sz = f.stat().st_size if f.exists() else 0
            if sz != last:
                last, since = sz, time.time()
            elif time.time() - since > stall:
                self.die(f"{s}: no new step for {stall // 60} min (controller: {http_get(f'http://127.0.0.1:{CTL_PORT}/status')})")
            time.sleep(60)
        code = sh(["docker", "inspect", "-f", "{{.State.ExitCode}}", f"qadx-train-{s}"], capture=True, check=False, quiet=True).strip()
        (self.out / "runs" / s).mkdir(parents=True, exist_ok=True)
        sh(["docker", "logs", "--tail", "200", f"qadx-train-{s}"], check=False, capture=True, quiet=True,
           stdout_file=self.out / "runs" / s / "trainer_tail.log")
        sh(["docker", "logs", f"qadx-train-{s}"], check=False, capture=True, quiet=True,
           stdout_file=self.out / "runs" / s / "trainer.log")  # the whole log: the container is removed below
        if code != "0":
            self.die(f"{s}: the trainer exited with {code} (docker logs qadx-train-{s})")
        it = f"iter_{self.budgets[i]:07d}"
        k = self.keep(i)
        chk = f"test -d /out/train/{s}/checkpoints/{it} && echo W"
        if k is not None:
            chk += f"; test -d /out/train/{s}/checkpoints/non_persistent/iter_{k:07d} && echo K"
        got = self.root(chk)
        if "W" not in got or (k is not None and "K" not in got):
            self.die(f"{s}: the checkpoint at {self.budgets[i]} or the kept full one at {k} is missing")
        sh(["docker", "rm", "-f", f"qadx-train-{s}"], check=False, quiet=True)
        self.kill_spawned(self.out / "runs" / s)
        self.servers_down(L)
        sh(["docker", "rm", "-f", "qadx-parse"], check=False, quiet=True)  # started again by the next launch
        (self.out / "runs" / s / "CHAIN_DONE").write_text(json.dumps(dict(run=s, steps=self.budgets[i], keep_full=k,
                                                                          done=datetime.now(timezone.utc).isoformat(timespec="seconds"))))
        self.log(f"SEGMENT_DONE {s}: {it}" + (f", full checkpoint at {k} kept" if k is not None else ""))

    def trim(self, i):
        """While segment i trains: drop its weights-only saves older than the newest one (except its last step and keep
        step), and the parent's kept full checkpoint once this segment has a full checkpoint of its own past the branch."""
        s, k = self.seg(i), self.keep(i)
        keep_w = {self.budgets[i]} | ({k} if k is not None else set()) | set(self.kw(i))
        cmd = (f"cd /out/train/{s}/checkpoints 2>/dev/null || exit 0; W=$(cat latest_checkpointed_iteration.txt 2>/dev/null); "
               f"for d in iter_*; do [ -d \"$d\" ] || continue; n=$((10#${{d#iter_}})); "
               f"[ -n \"$W\" ] && [ $n -lt $W ] && ! echo ' {' '.join(map(str, keep_w))} ' | grep -q \" $n \" && rm -rf \"$d\" && echo \"trim $d\"; done")
        if i > 0:
            p, b = self.seg(i - 1), self.bp(i - 1)
            cmd += (f"; F=$(cat non_persistent/latest_checkpointed_iteration.txt 2>/dev/null); "
                    f"if [ -n \"$F\" ] && [ $F -gt {b} ] && [ -d non_persistent/iter_$(printf %07d $F) ]; then "
                    f"rm -rf /out/train/{p}/checkpoints/non_persistent/iter_{b:07d} && echo 'trim parent full {b}'; fi")
        out = self.root(cmd + "; true").strip()  # the loop's last test (the newest save is never removed) must not fail the call
        if out:
            self.log(f"{s}: {' '.join(out.split())}")

    def prune(self):
        """Of each finished segment keep the weights-only checkpoints of its last step and keep step and its kept full one."""
        for i in range(len(self.budgets)):
            s, k = self.seg(i), self.keep(i)
            if not (self.out / "runs" / s / "CHAIN_DONE").exists():
                continue
            kw = {f"iter_{self.budgets[i]:07d}"} | ({f"iter_{k:07d}"} if k is not None else set()) | {f"iter_{x:07d}" for x in self.kw(i)}
            kf = f"non_persistent/iter_{k:07d}" if k is not None else ""
            out = self.root(f"cd /out/train/{s}/checkpoints 2>/dev/null || exit 0; "
                            f"for d in iter_*; do case \" {' '.join(sorted(kw))} \" in *\" $d \"*) ;; *) rm -rf \"$d\" && echo \"$d\";; esac; done; "
                            f"for d in non_persistent/iter_*; do [ -e \"$d\" ] || continue; [ \"$d\" = '{kf}' ] || {{ rm -rf \"$d\" && echo \"$d\"; }}; done").strip()
            if out:
                self.log(f"prune {s}: {' '.join(out.split())}")

    # ---------------------------------------------------------------------------------------------------- export
    def export(self):
        """Each finished segment's keep_weights steps and budget -> OUT/exports/<name>-s<step + export_offset>/. Checks:
        NVFP4, the model's KV algorithm, Qwen3 with its native RoPE config (YaRN only ever comes from the evaluation's
        --yarn); Nemotron: the quantized layers equal the NVIDIA release's layer map, whose exclude list is then written
        (train/prep/fix_ptq_config.py)."""
        for i, step in self.export_steps():
            s = self.seg(i)
            if not (self.out / "runs" / s / "CHAIN_DONE").exists():
                self.log(f"export: {s} is not done, skipped")
                continue
            d = self.export_name(step)
            dst = self.out / "exports" / d
            if (dst / "hf_quant_config.json").exists():
                self.log(f"{dst} exists: not exported again")
                continue
            self.log(f"export {s} step {step}")
            (self.out / "train" / "exports").mkdir(parents=True, exist_ok=True)
            self.nemo_run([0], f"qadx-export-{d}", ["torchrun", "--nproc-per-node", "1", "/cl/export_hf.py",
                                                    "--hf_model_name_or_path", f"/models/{self.model}",
                                                    "--megatron_path", f"/out/{s}/checkpoints/iter_{step:07d}",
                                                    "--export_unified_hf_path", f"/out/exports/{d}"],
                          env={"HF_HUB_OFFLINE": "1"}, detach=False)
            uid, gid = os.getuid(), os.getgid()
            self.root(f"chown -R {uid}:{gid} /out/train/exports/{d}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.replace(self.out / "train" / "exports" / d, dst)
            if self.spec.get("release"):
                ref = json.loads((self.models / self.spec["release"] / "hf_quant_config.json").read_text())["quantization"]["exclude_modules"]
                rp = dst / "release_exclude.json"
                rp.write_text(json.dumps(ref))
                fx = sh([str(PY), str(REPO / "train" / "prep" / "fix_ptq_config.py"), str(dst), str(rp), str(self.spec["kv"] or "none")],
                        capture=True, check=False)
                rp.unlink()
                self.log(fx.strip())
                if "PTQ_CHECK OK" not in fx:
                    self.die(f"export {dst}: its quantized layers differ from the release's layer map")
            q = json.loads((dst / "hf_quant_config.json").read_text())["quantization"]
            conf = json.loads((dst / "config.json").read_text())
            rope_ok = not self.spec.get("yarn") or (not conf.get("rope_scaling") and
                                                     (conf.get("rope_parameters") or {}).get("rope_type", "default") == "default")
            ok = (q["quant_algo"] == "NVFP4" and q.get("kv_cache_quant_algo") == self.spec["kv"] and any(dst.glob("*.safetensors"))
                  and rope_ok)
            if not ok:
                self.die(f"export {dst} failed its checks: {q}")
            self.log(f"EXPORTED {dst}")

    # ------------------------------------------------------------------------------------------------------- run
    def run(self):
        L = self.layout_spec()
        self.log(f"chain {self.name}: budgets {self.budgets}, decays {self.decays}, layout {self.layout}")
        for i in range(len(self.budgets)):
            st = self.state(i)
            if st == "done":
                self.log(f"{self.seg(i)} done before")
            elif st == "orphan":
                self.die(f"{self.seg(i)} has controller logs but no trainer: `resume {self.seg(i)} <step>` with its newest full checkpoint, then run again")
            elif st in ("running", "exited"):
                self.wait_done(i, L)
            else:
                self.launch(i, L)
                self.wait_done(i, L)
            self.prune()
        self.export()
        self.log(f"CHAIN_DONE {self.name}")

    def resume(self, seg_name, step):
        L = self.layout_spec()
        i = [self.seg(j) for j in range(len(self.budgets))].index(seg_name)
        self.launch(i, L, resume_step=step)
        self.wait_done(i, L)
        self.prune()

    def prepare(self):
        """Once per model: (1) Nemotron: the start checkpoint models/<checkpoint> (train/prep/build_ptqs.py: quantized
        from BF16 at the NVIDIA release's static scales, CPU, checked tensor by tensor against
        train/prep/ptqs/<checkpoint>/tensors.sha256; Qwen3 starts from NVIDIA's release itself); (2) the Megatron PTQ
        student (fake-quant NVFP4 W4A4 with the model's recipe in train/recipes/). Calibration does not matter for the
        student: weight scales come from max|W| (routed experts: imported at train start), Q/K/V use a constant amax, and
        every activation scale is replaced at train start by the start checkpoint's (CL_IMPORT_INPUT_SCALE)."""
        ck = self.models / self.cfg["checkpoint"]
        if self.spec.get("release") and not (ck / "hf_quant_config.json").exists():
            ref = REPO / "train" / "prep" / "ptqs" / self.cfg["checkpoint"]
            self.log(f"building the start checkpoint {ck} (CPU)")
            mo = REPO / "third_party" / "Model-Optimizer"
            sp = "/opt/venv/lib/python3.12/site-packages"
            tmp = f"/models/{self.cfg['checkpoint']}.build"  # moved to models/<checkpoint> only after every tensor matched
            root_m = lambda c: sh(["docker", "run", "--rm", "-v", f"{self.models}:/models", "--entrypoint", "bash", IMG_NEMO, "-c", c],
                                  check=False, capture=True, quiet=True)
            root_m(f"rm -rf {tmp} /models/{self.cfg['checkpoint']}")
            out = sh(["docker", "run", "--rm", "--entrypoint", "python3", "-v", f"{mo}/modelopt:{sp}/modelopt:ro",
                      "-v", f"{self.models}:/models", "-v", f"{REPO / 'train'}:/train:ro", IMG_NEMO, "/train/prep/build_ptqs.py",
                      "--bf16", f"/models/{self.model}", "--release", f"/models/{self.spec['release']}",
                      "--configs", f"/train/prep/ptqs/{self.cfg['checkpoint']}", "--out", tmp,
                      "--check", f"/train/prep/ptqs/{self.cfg['checkpoint']}/tensors.sha256"], capture=True, check=False)
            lines = [l for l in out.splitlines() if l.startswith("build_ptqs")]
            self.log("\n".join(lines) or out[-1000:])
            if not any("tensors match" in l for l in lines) or "differ" in "".join(lines):
                root_m(f"rm -rf {tmp}")
                self.die(f"start checkpoint {ck} does not match {ref / 'tensors.sha256'}: report this")
            root_m(f"mv {tmp} /models/{self.cfg['checkpoint']} && chown -R {os.getuid()}:{os.getgid()} /models/{self.cfg['checkpoint']}")
            if not (ck / "hf_quant_config.json").exists():
                self.die(f"start checkpoint {ck}: moving it in place failed")
        elif self.spec.get("release"):
            self.log(f"{ck} exists")
        dst = self.out / "train" / self.ptq
        if dst.exists():
            self.log(f"{dst} exists")
            return
        self.log(f"building {dst}")
        self.nemo_run([0], "qadx-ptq", ["torchrun", "--nproc-per-node", "1", "quantize.py", "--hf_model_name_or_path", f"/models/{self.model}",
                                        "--recipe", f"/cl/recipes/{self.spec['recipe']}", "--calib_dataset_name", "cnn_dailymail",
                                        "--calib_num_samples", "64", "--seq_length", "1024", "--skip_generate",
                                        "--export_megatron_path", f"/out/{self.ptq}"], detach=False)
        if not dst.exists():
            self.die("PTQ failed")
        self.log(f"PTQ student ready: {dst}")

    def dryrun(self):
        """Write each segment's controller.json to OUT/dryrun/<segment>/ and print the trainer command; starts nothing."""
        L = self.layout_spec()
        if self.cfg.get("multiturn"):
            missing = [v for v in ("QADX_API_BASE", "QADX_API_KEY", "QADX_SIM_MODEL") if not os.environ.get(v)]
            if missing:
                print(f"WARNING: multi-turn data: {', '.join(missing)} not set (a real run needs QADX_API_BASE and QADX_API_KEY; "
                      f"QADX_SIM_MODEL defaults to {self.cfg['simulator']['model']!r}, the id of gpt-oss-120b at your endpoint may differ)")
        urls = [f"http://{L['serve_ip']}:{SRV_PORT0 + j}" for j in range(len(L["serve_gpus"]))]
        thr = [f"http://127.0.0.1:{THR_PORT0 + j}" for j in range(2)] if self.cfg.get("multiturn") else []
        for i in range(len(self.budgets)):
            d = self.out / "dryrun" / self.seg(i)
            d.mkdir(parents=True, exist_ok=True)
            start = 0 if i == 0 else self.bp(i - 1)
            (d / "controller.json").write_text(json.dumps(self.controller_cfg(i, d, urls, thr, start), indent=1))
            env, cmd, pp, first, last = self.trainer_cmd(i, L, urls)
            (d / "trainer.json").write_text(json.dumps(dict(env=env, cmd=cmd, gpus=L["train_gpus"]), indent=1))
            print(f"{self.seg(i)}: start {start}, to {self.budgets[i]}, decay {self.decays[i]}, keep {self.keep(i)}, "
                  f"PP {pp} ({first}/../{last}), servers {urls} -> {d}")

    def check_step0(self):
        """Step 0 of training must be the start checkpoint: load the PTQ student, import the checkpoint's static scales,
        run one step at learning rate 0 (weights unchanged), save, export to Hugging Face format, and compare every
        tensor with the checkpoint (the comparison's last line is VERDICT PASS / FAIL). Expected: every weight,
        weight_scale, weight_scale_2 and input_scale identical -- Qwen3-30B-A3B equal up to the last-bit rounding of
        NVIDIA's older ModelOpt, its routed experts' input scales against their layer maximum (MODELS step0_cmp); a
        float tensor in another dtype counts as equal when its values are (Nemotron 3 Nano: the router gate is exported
        in fp32, stored in bf16); FP8-cast k_scale / v_scale = 1.0 may exist on one side only. GPUs 0.. (one for an
        8B-class model; four, pipeline parallel, for a 30B-class one, whose student with its optimizer does not fit one
        GPU). Nemotron 3 Nano: the router expert bias moves every step whatever the learning rate, so it is frozen here."""
        d = "check-step0"
        n = self.spec["step0_gpus"]
        self.root(f"rm -rf /out/train/{d} /out/train/{d}-hf")
        env = {"HF_HUB_OFFLINE": "1", "CL_NO_OPTIM_CKPT": "1", "CL_IMPORT_INPUT_SCALE": f"/models/{self.cfg['checkpoint']}"}
        if self.spec.get("mock_vocab"):
            env["CL_MOCK_VOCAB"] = str(self.spec["mock_vocab"])
        if self.spec.get("bias_rate0"):
            env["CL_PROVIDER_OVERRIDES"] = json.dumps({"moe_router_bias_update_rate": 0.0})
        opt = (self.cfg.get("optimizer_overrides") or {}).get("1node")
        if n > 1 and opt:  # as the 1node training: part of the optimizer state on the CPU (no effect at learning rate 0)
            env["CL_OPTIMIZER_OVERRIDES"] = json.dumps(opt)
        if n > 1:
            f, l = self.pp_split(n)
            ov = json.loads(env.get("CL_PROVIDER_OVERRIDES", "{}"))
            ov.update(num_layers_in_first_pipeline_stage=f, num_layers_in_last_pipeline_stage=l)
            env["CL_PROVIDER_OVERRIDES"] = json.dumps(ov)
        rc = self.nemo_run(list(range(n)), "qadx-check-step0", [
            "torchrun", "--nproc-per-node", str(n), "--master-addr", "127.0.0.1", "--master-port", str(TORCH_PORT),
            "/cl/qad_distill.py", "--pp_size", str(n), "--student_hf_path", f"/models/{self.model}",
            "--student_megatron_path", f"/out/{self.ptq}", "--teacher_hf_path", f"/models/{self.model}", "--use_mock_data",
            "--seq_length", "2048", "--mbs", "1", "--gbs", str(n), "--train_iters", "1", "--eval_iters", "0", "--lr", "0", "--min_lr", "0",
            "--lr_warmup_iters", "0", "--save_interval", "1", "--no_async_save", "--log_interval", "1", "--output_dir", f"/out/{d}"],
            env=env, detach=False)
        self.log("\n".join(l for l in rc.splitlines() if "imported" in l or "Traceback" in l or "Error" in l)[-2000:] or "step done")
        self.nemo_run([0], "qadx-check-step0-export", ["torchrun", "--nproc-per-node", "1", "/cl/export_hf.py",
                                                       "--hf_model_name_or_path", f"/models/{self.model}",
                                                       "--megatron_path", f"/out/{d}/checkpoints/iter_0000001",
                                                       "--export_unified_hf_path", f"/out/{d}-hf"], env={"HF_HUB_OFFLINE": "1"}, detach=False)
        self.root(f"chmod -R a+rX /out/train/{d}-hf")
        r = subprocess.run([str(PY), str(REPO / "train" / "tests" / "compare_nvfp4_ckpt.py"), str(self.out / "train" / f"{d}-hf"),
                            str(self.models / self.cfg["checkpoint"]), *self.spec.get("step0_cmp", [])], capture_output=True, text=True)
        out = r.stdout if "VERDICT " in r.stdout else r.stdout + r.stderr  # stderr only without a verdict: VERDICT stays the last line
        (self.out / f"{d}.txt").write_text(out)
        print(out)
        self.root(f"rm -rf /out/train/{d} /out/train/{d}-hf")  # compared: the result is in {d}.txt (rerun to repeat)
        self.log(f"step-0 comparison written to {self.out / (d + '.txt')}")

    def clean(self):
        """After every budget is exported: remove the chain's trainer checkpoints (OUT/train/<segment>), keeping the
        controller logs (OUT/runs) and the exports. Refuses while an export is missing."""
        missing = [x for _, x in self.export_steps() if not (self.out / "exports" / self.export_name(x) / "hf_quant_config.json").exists()]
        if missing:
            self.die(f"not cleaned: steps {missing} are not exported yet")
        for i in range(len(self.budgets)):
            self.root(f"rm -rf /out/train/{self.seg(i)} /out/train/{self.seg(i)}.init")
        self.log(f"cleaned: trainer checkpoints of {self.name} removed (exports and controller logs kept)")

    def status(self):
        for i in range(len(self.budgets)):
            print(f"{self.seg(i)}: {self.state(i)}")
        print("controller:", http_get(f"http://127.0.0.1:{CTL_PORT}/status") or "down")


def init_cmd(src, dst):
    """A chain's start checkpoint from another chain's last weights-only checkpoint (init_from): its latest iteration laid
    out in DST with its trackers, shard files hard-linked, everything else copied; qad_distill reads it as a pretrained
    checkpoint (CL_PRETRAINED: weights only, step 0, a new optimizer and schedule)."""
    return f"""set -e
S={src}; D={dst}
if test -s $D/latest_checkpointed_iteration.txt; then  # laid out before (the tracker is written last)
  IT=$(printf iter_%07d $(cat $D/latest_checkpointed_iteration.txt | tr -dc 0-9))
  test -d $D/$IT && {{ echo "INIT_CKPT_OK $IT: exists"; exit 0; }}
fi
it=$(cat $S/latest_checkpointed_iteration.txt 2>/dev/null | tr -dc 0-9)
test -n "$it" || {{ echo INIT_CKPT_FAIL no $S/latest_checkpointed_iteration.txt; exit 1; }}
IT=$(printf iter_%07d $it)
test -d $S/$IT || {{ echo INIT_CKPT_FAIL no $S/$IT; exit 1; }}
rm -rf $D/$IT  # an interrupted earlier attempt
mkdir -p $D/$IT; cd $S/$IT
find . -mindepth 1 -type d -printf '%P\\0' | while IFS= read -r -d '' d; do mkdir -p "$D/$IT/$d"; done
nl=0; nc=0
while IFS= read -r -d '' f; do
  case "$f" in __*.distcp) ln "$f" "$D/$IT/$f"; nl=$((nl + 1)); continue ;; esac
  cp -p --sparse=always "$f" "$D/$IT/$f"; nc=$((nc + 1))
done < <(find . -type f -printf '%P\\0')
cp -p $S/$IT/train_state.pt $D/latest_train_state.pt
echo $it > $D/latest_checkpointed_iteration.txt
echo "INIT_CKPT_OK $IT from $S: $nl shard files linked, $nc files copied"
"""


def branch_cmd(parent, branch, step):
    """The branch's checkpoint directory from the parent's full checkpoint at STEP: shard files hard-linked (a written
    checkpoint never changes), everything else copied; the branch's trackers point at that iteration, so the trainer
    resumes from it (weights, optimizer, scheduler and RNG state)."""
    it = f"iter_{step:07d}"
    return f"""set -e
P={parent}; B={branch}; PI=$P/non_persistent/{it}; BI=$B/non_persistent/{it}
test -d $PI || {{ echo BRANCH_CKPT_FAIL no full checkpoint $PI; exit 1; }}
test -e $B && {{ echo BRANCH_CKPT_FAIL $B exists; exit 1; }}
mkdir -p $BI; cd $PI
find . -mindepth 1 -type d -printf '%P\\0' | while IFS= read -r -d '' d; do mkdir -p "$BI/$d"; done
nl=0; nc=0
while IFS= read -r -d '' f; do
  case "$f" in __*.distcp) ln "$f" "$BI/$f"; nl=$((nl + 1)); continue ;; esac
  cp -p --sparse=always "$f" "$BI/$f"; nc=$((nc + 1))
done < <(find . -type f -printf '%P\\0')
test -f $PI/train_state.pt && cp -p $PI/train_state.pt $B/non_persistent/latest_train_state.pt
echo {step} > $B/non_persistent/latest_checkpointed_iteration.txt
echo "BRANCH_CKPT_OK {it}: $nl shard files linked, $nc files copied"
"""


# ---------------------------------------------------------------------------------------------------------- helpers
def sh(cmd, check=True, capture=False, quiet=False, shell=False, stdout_file=None):
    if isinstance(cmd, str) and not shell:
        cmd = ["bash", "-c", cmd]
    if stdout_file:
        with open(stdout_file, "w") as f:
            return str(subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, shell=shell).returncode)
    r = subprocess.run(cmd, capture_output=True, text=True, shell=shell)
    if check and r.returncode:
        raise SystemExit(f"command failed ({r.returncode}): {cmd if isinstance(cmd, str) else shlex.join(cmd)}\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return (r.stdout + ("" if quiet else r.stderr)) if capture else r.stdout


def http_get(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()[:300]
    except Exception:  # noqa: BLE001
        return ""


def http_ok(url):
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment")
    ap.add_argument("action", choices=["prepare", "check-step0", "run", "status", "stop", "export", "resume", "dryrun", "clean"])
    ap.add_argument("args", nargs="*", help="resume: SEGMENT STEP")
    ap.add_argument("--layout", choices=["1node", "2node"])
    ap.add_argument("--peer", help="2node: ssh host of the rollout node")
    ap.add_argument("--peer-ip", help="2node: its IP as seen from this node (default: first of `hostname -I` there)")
    ap.add_argument("--out", help="output root (default <repo>/out)")
    a = ap.parse_args()
    ch = Chain(a.experiment, a)
    if a.action == "prepare":
        ch.prepare()
    elif a.action == "run":
        ch.run()
    elif a.action == "status":
        ch.status()
    elif a.action == "stop":
        ch.stop_all(ch.layout_spec() if a.layout else None)
    elif a.action == "export":
        ch.export()
    elif a.action == "resume":
        ch.resume(a.args[0], int(a.args[1]))
    elif a.action == "dryrun":
        ch.dryrun()
    elif a.action == "check-step0":
        ch.check_step0()
    elif a.action == "clean":
        ch.clean()


if __name__ == "__main__":
    main()
