"""Scheduler tests (CPU only, nothing is started):

1. run.py --dry-run for a fake arm on 8 GPUs: a checkpoint directory holding only a Qwen3-8B tokenizer and config
   (no weights). Checks the plan (9 jobs, every GPU busy in the first wave, tau2 jobs never overlapping on one API key),
   one serve command per job on its own GPU / port, a first request for every completion benchmark, the tau2 and BFCL
   commands, and that the dry run wrote nothing and started no container.
2. --shard 0/2 and 1/2 split the jobs of two arms disjointly and completely.
3. The real scheduling loop (run_jobs) with a stub in place of run_phase.py: every job runs exactly once, at most one
   job per GPU at a time, tau2 jobs serialized per API key, an incomplete job is reported and the others go on.
4. The other models (Qwen3-30B-A3B, Nemotron Nano 9B v2, Nemotron 3 Nano), each whose tokenizer is found: run.py
   --dry-run for two fake NVFP4 arms with two API keys (the model recognised from the config; its phases, windows,
   YaRN, server flags, parsers and plugins, prefix caching, first-request sampling / prompts / add_special_tokens / caps,
   tau2 commands and agent sampling, BFCL handler and first request, alternating tau2 lanes), a --model mismatch refused,
   and the scheduling loop passing --model and --api-lane to the phase runner.
5. Evaluation spec v2 of the other models (no tokenizer needed): Qwen3-8B's samples per question (competitions and GPQA
   Diamond 16, LiveCodeBench v6 8) with each model's own caps; a benchmark file made under v1 (stamped k, or unstamped)
   is moved to <arm>/superseded/k<old>/ and its stamp rewritten for v2, a current one is kept; a phase whose status was
   written for the v1 tasks (or has no tasks field) is not complete (run.py and summary.py), one for the current tasks is.
    .venv-client/bin/python eval/tests/test_dry_run.py [--tokenizer-from DIR] [--tokenizer-dir MODEL=DIR ...]
Tokenizer source: --tokenizer-from, else models/Qwen3-8B; other models: --tokenizer-dir, else models/<BF16 or NVFP4 dir>.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
sys.path.insert(0, str(C.LIB))
import run_phase as RP  # noqa: E402

STUB = '''import json, os, sys, time, random
a = dict(zip(sys.argv[1::2], sys.argv[2::2]))
t0 = time.time()
time.sleep(0.3 + random.random() * 0.5)
with open(os.environ["STUB_LOG"], "a") as f:
    f.write(json.dumps(dict(arm=a["--arm"], phase=a["--phase"], gpu=int(a["--gpu"]), t0=t0, t1=time.time(),
                            model=a.get("--model"), lane=a.get("--api-lane"))) + "\\n")
sys.exit(3 if (a["--arm"], a["--phase"]) == ("fake2", "S3") else 0)
'''


def tokenizer_source(arg):
    for c in [arg, C.REPO / "models" / "Qwen3-8B"]:
        if c and (Path(c) / "tokenizer_config.json").exists():
            return Path(c)
    print("SKIP: no Qwen3-8B tokenizer found (--tokenizer-from)")
    sys.exit(0)


def fake_checkpoint(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja"):
        if (src / f).exists():
            shutil.copy(src / f, dst / f)
    return dst


# The other models: tokenizer / config directories under models/ (BF16 or NVIDIA's NVFP4 release), expected settings.
OTHER = {
    "qwen3-30b-a3b": dict(dirs=("Qwen3-30B-A3B-NVFP4", "Qwen3-30B-A3B"), phases=["S1", "G", "S3", "S4", "L", "T", "B"],
                          windows=dict(S1=40960, G=40960, S3=40960, S4=40960, L=131072, T=40960, B=40960), yarn={"L"},
                          flags=[], parse_phases={"T"}, plugins=False, prefix=True,
                          sampling=(0.6, 0.95, 20), agent=(0.6, 0.95, 20), add_special=None, comps_cap=38912,
                          other_cap=32768, prompt_start=None, handler="qadx/qwen3-ct"),
    "nemotron-nano-9b-v2": dict(dirs=("NVIDIA-Nemotron-Nano-9B-v2", "NVIDIA-Nemotron-Nano-9B-v2-NVFP4"),
                                phases=["S1", "G", "S3", "L", "T", "B"], windows=dict.fromkeys(["S1", "G", "S3", "L", "T", "B"], 131072),
                                yarn=set(), flags=["--mamba-ssm-cache-dtype float32"], parse_phases={"T", "B"}, plugins=True,
                                prefix=False, sampling=(0.6, 0.95, -1), agent=(0.6, 0.95, -1), add_special=False,
                                comps_cap=32768, other_cap=32768, prompt_start="Solve the following math problem. Make sure",
                                handler="qadx/nemotron-nano-v2-ct"),
    "nemotron-3-nano-30b-a3b": dict(dirs=("NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4"),
                                    phases=["S1", "G", "S3", "S4", "L", "T", "B"],
                                    windows=dict(S1=262144, G=262144, S3=262144, S4=262144, L=131072, T=131072, B=131072),
                                    yarn=set(), flags=["--mamba-ssm-cache-dtype float32 --moe-backend flashinfer_cutlass"],
                                    parse_phases={"T", "B"}, plugins=False, prefix=False, sampling=(1.0, 1.0, -1),
                                    agent=(0.6, 0.95, -1), add_special=False, comps_cap=131072, other_cap=131072,
                                    prompt_start="Solve the following math problem step by step.",
                                    handler="qadx/nemotron3-nano-ct"),
}


def other_tokenizer(model, given):
    for c in [given.get(model)] + [C.REPO / "models" / d for d in OTHER[model]["dirs"]]:
        if c and (Path(c) / "tokenizer_config.json").exists() and (Path(c) / "config.json").exists():
            return Path(c)
    return None


def check_other_model(model, src, tmp, env, rep):
    """Dry run of two fake NVFP4 arms of one model (section 4)."""
    exp = OTHER[model]
    cks = []
    for i in (1, 2):
        ck = fake_checkpoint(src, tmp / f"ckpt-{model}-{i}")
        (ck / "hf_quant_config.json").write_text(json.dumps({"quantization": {"quant_algo": "NVFP4", "kv_cache_quant_algo": None}}))
        cks.append(ck)
    results = tmp / f"results_{model}"
    env2 = dict(env, QADX_API_KEY_2="dummy-key-2")
    r = subprocess.run([str(C.CLIENT_PY), str(C.EVAL / "run.py"), "--arm", f"x1={cks[0]}", "--arm", f"x2={cks[1]}",
                        "--gpus", "0,1,2,3,4,5,6,7", "--dry-run", "--results", str(results)], capture_output=True, text=True, env=env2)
    out = r.stdout
    (tmp / f"dry_run_{model}.txt").write_text(out + "\n--- stderr ---\n" + r.stderr)
    tag = f"{model}:"
    rep.add(f"{tag} dry run exits 0", r.returncode == 0, r.stderr.strip()[-500:] if r.returncode else "")
    plan = re.findall(r"t\+\s*([\d.]+) h\s+GPU (\d)\s+(x[12])/(\w+)\s.*?(?:\(API lane (\d)\))?\s+\[(\S+)\]$", out, re.M)
    rep.add(f"{tag} plan: the model's phases for each arm, model recognised from the config",
            sorted((p[2], p[3]) for p in plan) == sorted((a, ph) for a in ("x1", "x2") for ph in exp["phases"])
            and all(p[5] == model for p in plan), " ".join(f"{p[2]}/{p[3]}" for p in plan))
    lanes = {p[2]: p[4] for p in plan if p[3] == "T"}
    rep.add(f"{tag} tau2: one phase per arm, the two arms on API lanes 1 and 2", lanes == {"x1": "1", "x2": "2"}, str(lanes))
    serves = {(a, ph): s for a, ph, s in re.findall(r"## (x[12]) (\w+) on GPU \d: .*?\nserve: (.*)", out)}
    win = {ph: int(re.search(r"--max-model-len (\d+)", s).group(1)) for (a, ph), s in serves.items() if a == "x1"}
    rep.add(f"{tag} serve: windows per phase", win == exp["windows"], str(win))
    rep.add(f"{tag} serve: YaRN exactly for {sorted(exp['yarn']) or 'no phase'}",
            {ph for (a, ph), s in serves.items() if a == "x1" and "rope_type" in s} == exp["yarn"])
    rep.add(f"{tag} serve: model flags after the common ones" + (f" ({exp['flags'][0]})" if exp["flags"] else " (none)"),
            all(s.split("--seed 0", 1)[1].startswith(" " + exp["flags"][0]) if exp["flags"] else "--mamba" not in s and "--moe-backend" not in s
                for s in serves.values()))
    rep.add(f"{tag} serve: prefix caching in tau2 / BFCL only " + ("as for Qwen3" if exp["prefix"] else "never (hybrid model)"),
            all(("--enable-prefix-caching" in s) == (exp["prefix"] and ph in ("T", "B")) for (a, ph), s in serves.items()))
    rep.add(f"{tag} serve: tool / reasoning parsers in {sorted(exp['parse_phases'])}, generation cap override in T only",
            all(("--tool-call-parser" in s) == (ph in exp["parse_phases"]) and ("--override-generation-config" in s) == (ph == "T")
                for (a, ph), s in serves.items()))
    plug = [s for (a, ph), s in serves.items() if "--tool-parser-plugin" in s]
    rep.add(f"{tag} serve: parser plugins " + ("loaded from eval/lib/vllm_plugins, mounted read-only" if exp["plugins"] else "none"),
            (sorted(ph for (a, ph), s in serves.items() if "--tool-parser-plugin" in s and a == "x1") == sorted(exp["parse_phases"])
             and all(f"{C.LIB / 'vllm_plugins'}:{C.LIB / 'vllm_plugins'}:ro" in s and (C.LIB / "vllm_plugins" / "nemotron_toolcall_parser.py").exists()
                     for s in plug)) if exp["plugins"] else not plug)
    posts = {b: json.loads(j) for b, j in re.findall(r"^(\w+): first request = .*?\nPOST /v1/completions (\{.*\})$", out, re.M)}
    want = {"aime26", "hmmt_feb26", "hmmt_feb25", "hmmt_nov25", "gpqa_diamond", "lcb_v6", "longbench_pro", "mrcr8"} | (
        {"lcb_pro"} if "S4" in exp["phases"] else set())
    rep.add(f"{tag} first request for every completion benchmark", set(posts) == want, f"{sorted(want ^ set(posts))}")
    t, p, k = exp["sampling"]
    rep.add(f"{tag} first requests: sampling T {t} / top-p {p} / top-k {k}, add_special_tokens "
            + ("absent" if exp["add_special"] is None else str(exp["add_special"])),
            all(b["temperature"] == t and b["top_p"] == p and b["top_k"] == k and b["min_p"] == 0.0 and b["skip_special_tokens"] is False
                and b.get("add_special_tokens", None) == exp["add_special"] and b["model"] == "eval-model" for b in posts.values()))
    caps = {b: v["max_tokens"] for b, v in posts.items()}
    rep.add(f"{tag} first requests: budgets within the caps",
            all(c <= (exp["comps_cap"] if b.startswith(("aime", "hmmt")) else exp["other_cap"] if b not in ("longbench_pro", "mrcr8") else 32768)
                for b, c in caps.items()), json.dumps(caps))
    ps = posts.get("aime26", {}).get("prompt", "")
    rep.add(f"{tag} competition prompt: " + ("NVIDIA's evaluation prompt" if exp["prompt_start"] else "the item as it is"),
            (exp["prompt_start"] in ps) if exp["prompt_start"] else ("Please reason step by step" in ps), ps[:120].replace("\n", " "))
    tau2 = re.findall(r"^tau2 (\w+): (.*)$", out, re.M)
    ta, tp, tk = exp["agent"]
    agent_ok = all(json.loads(re.search(r"--agent-llm-args (\{.*?\}) --user-llm", c).group(1)) | {"api_base": None} ==
                   {"api_base": None, "api_key": "EMPTY", "temperature": ta, "top_p": tp, "top_k": tk, "min_p": 0.0} for _, c in tau2)
    rep.add(f"{tag} tau2: 3 domains per arm in one phase, seed 300, 4 trials, agent sampling T {ta} / top-p {tp} / top-k {tk}",
            sorted(d for d, _ in tau2) == sorted(["airline", "retail", "telecom"] * 2) and agent_ok
            and all("--seed 300" in c and "--num-trials 4" in c and "--max-concurrency 48" in c for _, c in tau2))
    keys = re.findall(r"^## (x[12]) T on.*?\n.*?\nuser simulator / judge: .* with (QADX_API_KEY(?:_2)?) ", out, re.M)
    rep.add(f"{tag} tau2: the arms use the two keys", dict(keys) == {"x1": "QADX_API_KEY", "x2": "QADX_API_KEY_2"}, str(keys))
    bf = re.findall(r"^bfcl: .*$", out, re.M)
    rep.add(f"{tag} bfcl: handler {exp['handler']}, 4 generate + 4 evaluate commands per arm, first request rendered",
            len(bf) == 16 and all(f"--model {exp['handler']} " in c for c in bf) and out.count("bfcl first request (multi_turn_base_0") == 2,
            next((l for l in out.splitlines() if "could not render" in l), "")[:300])
    r2 = subprocess.run([str(C.CLIENT_PY), str(C.EVAL / "run.py"), "--arm", f"x1={cks[0]}", "--model", "qwen3-8b", "--dry-run",
                         "--results", str(results)], capture_output=True, text=True, env=env)
    rep.add(f"{tag} --model qwen3-8b refused for this checkpoint", r2.returncode != 0 and "checkpoint" in r2.stderr,
            r2.stderr.strip()[-200:])
    return cks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-from")
    ap.add_argument("--tokenizer-dir", action="append", default=[], help="MODEL=DIR: tokenizer and config of another model")
    a = ap.parse_args()
    given = dict(x.split("=", 1) for x in a.tokenizer_dir)
    rep = C.Report("scheduler")
    tmp = Path(tempfile.mkdtemp(prefix="qadx_sched_"))
    ck = fake_checkpoint(tokenizer_source(a.tokenizer_from), tmp / "ckpt-fake")
    results = tmp / "results"
    env = dict(os.environ, QADX_API_BASE="https://upstream.invalid/v1", QADX_API_KEY="dummy-key")
    env.pop("QADX_API_KEY_2", None)
    # only this harness's containers (qadx_vllm_gpu<i>): other containers on a shared host come and go independently
    ours = lambda: [n for n in subprocess.run(["docker", "ps", "-a", "--format", "{{.Names}}"], capture_output=True,
                                              text=True).stdout.split() if n.startswith("qadx_vllm_gpu")]
    before = ours()
    r = subprocess.run([str(C.CLIENT_PY), str(C.EVAL / "run.py"), "--arm", f"fake={ck}", "--gpus", "0,1,2,3,4,5,6,7",
                        "--dry-run", "--results", str(results)], capture_output=True, text=True, env=env)
    out = r.stdout
    (tmp / "dry_run.txt").write_text(out + "\n--- stderr ---\n" + r.stderr)
    rep.add("dry run exits 0", r.returncode == 0, r.stderr.strip()[-500:] if r.returncode else "")
    plan = re.findall(r"t\+\s*([\d.]+) h\s+GPU (\d)\s+fake/(\w+)\s", out)
    rep.add("plan: 9 jobs, one per phase", sorted(p[2] for p in plan) == sorted(["S1", "S2", "S3", "S4", "L1", "M", "T1", "T2", "B"]),
            " ".join(f"{p[2]}@GPU{p[1]}" for p in plan))
    first = [p for p in plan if float(p[0]) == 0.0]
    rep.add("plan: first wave uses all 8 GPUs once", sorted(int(p[1]) for p in first) == list(range(8)))
    t = {p[2]: float(p[0]) for p in plan}
    hours = {"T1": 0.9, "T2": 1.9}
    rep.add("plan: tau2 jobs do not overlap on a single API key",
            t["T1"] >= t["T2"] + hours["T2"] - 1e-6 or t["T2"] >= t["T1"] + hours["T1"] - 1e-6, f"T1 at {t['T1']} h, T2 at {t['T2']} h")
    serves = re.findall(r"## fake (\w+) on GPU (\d): .*?\nserve: (.*)", out)
    ok = len(serves) == 9 and all(f"--gpus device={g} " in s and f"--port {8100 + int(g)} " in s and f"--name qadx_vllm_gpu{g} " in s
                                   and s.count(str(ck)) == 3 for _, g, s in serves)
    rep.add("serve commands: own GPU, port and container per job; checkpoint mounted read-only", ok, f"{len(serves)} commands")
    gpu_of = {ph: g for ph, g, _ in serves}
    rep.add("serve commands follow the plan's GPU assignment", all(gpu_of[p[2]] == p[1] for p in plan))
    img = json.loads(subprocess.run([str(C.CLIENT_PY), "-c", "import sys; sys.path.insert(0, 'eval/lib'); import paths, json; print(json.dumps(paths.vllm_image()))"],
                                    capture_output=True, text=True, cwd=C.REPO).stdout)
    rep.add("serve commands use the pinned image digest", all(f" {img} " in s for _, _, s in serves), img)
    yarn = {ph: "rope_type" in s and "--max-model-len 131072" in s for ph, _, s in serves}
    rep.add("YaRN and 131,072 window exactly for L1 and M", {k for k, v in yarn.items() if v} == {"L1", "M"})
    posts = re.findall(r"^(\w+): first request = .*?\nPOST /v1/completions (\{.*\})$", out, re.M)
    want = {"aime26", "hmmt_feb26", "hmmt_feb25", "hmmt_nov25", "gpqa_diamond", "ifbench", "mmlu_redux_1k", "lcb_v6",
            "lcb_pro", "longbench_pro", "mrcr8"}
    bodies = {b: json.loads(j) for b, j in posts}
    rep.add("first request printed for every completion benchmark", set(bodies) == want, f"{sorted(set(want) - set(bodies))} missing")
    samp = all(b["temperature"] == 0.6 and b["top_p"] == 0.95 and b["top_k"] == 20 and b["min_p"] == 0.0
               and b["presence_penalty"] == 0.0 and b["repetition_penalty"] == 1.0 and b["skip_special_tokens"] is False
               and b["model"] == "eval-model" for b in bodies.values())
    caps = {b: v["max_tokens"] for b, v in bodies.items()}
    rep.add("first requests: model-card sampling, budgets within the caps", samp and all(
        (c <= 38912 if b.startswith(("aime", "hmmt")) else c <= 32768) for b, c in caps.items()), json.dumps(caps))
    tau2 = re.findall(r"^tau2 (\w+): (.*)$", out, re.M)
    rep.add("tau2 commands: 3 domains, seed 300, 4 trials, no retries, concurrency 48",
            sorted(d for d, _ in tau2) == ["airline", "retail", "telecom"] and all(
                all(x in c for x in ("--seed 300", "--num-trials 4", "--max-retries 0", "--max-concurrency 48",
                                     "--task-split-name base", f"--user-llm {RP.tau2_api_config(1, require=False)['model']}")) for _, c in tau2))
    bf = re.findall(r"^bfcl: .*$", out, re.M)
    rep.add("bfcl: 4 generate + 4 evaluate commands, first request rendered",
            len(bf) == 8 and "bfcl first request (multi_turn_base_0" in out and "--num-threads 32" in bf[0])
    after = ours()
    rep.add("dry run started no container and wrote no results", sorted(after) == sorted(before) and not any(
        f for f in results.rglob("*") if f.is_file()) if results.exists() else sorted(after) == sorted(before))

    # 2. shards
    def planned(shard):
        o = subprocess.run([str(C.CLIENT_PY), str(C.EVAL / "run.py"), "--arm", f"fake={ck}", "--arm", f"fake2={ck}",
                            "--gpus", "0,1,2,3,4,5,6,7", "--dry-run", "--results", str(results), "--shard", shard,
                            "--phases", "S1,S2,S3,L1,T1,T2"], capture_output=True, text=True, env=env).stdout
        return {m for m in re.findall(r"GPU \d\s+(fake2?/\w+)\s", o)}
    s0, s1 = planned("0/2"), planned("1/2")
    rep.add("--shard 0/2 and 1/2: disjoint and complete", not (s0 & s1) and len(s0 | s1) == 12, f"{len(s0)} + {len(s1)} jobs")

    # 3. the scheduling loop with a stub phase runner
    sys.path.insert(0, str(C.EVAL))
    sys.path.insert(0, str(C.LIB))
    import run as RUN
    import serve
    stub = tmp / "stub_phase.py"
    stub.write_text(STUB)
    log = tmp / "stub_log.jsonl"
    os.environ["STUB_LOG"] = str(log)
    os.environ.pop("QADX_API_KEY_2", None)
    RUN.PHASE_RUNNER, RUN.POLL_S = stub, 0.1
    serve.wait_gpu_free = lambda *x, **k: True
    res2 = tmp / "results_stub"
    (res2 / "logs").mkdir(parents=True)
    RUN.LOG = res2 / "logs" / "scheduler.log"
    jobs = [RUN.Job(arm, str(ck), ph, i) for i, (arm, ph) in enumerate((x, p) for x in ("fake", "fake2") for p in
                                                                      ("S1", "S2", "S3", "L1", "M", "T1", "T2", "B"))]
    jobs = sorted(jobs, key=lambda j: (-j.hours, j.idx))
    args = argparse.Namespace(force=False, limit=None, no_compile_cache=True, tau2_per_key=1)
    t0 = time.time()
    failed, _ = RUN.run_jobs(jobs, [0, 1, 2], args, res2)
    runs = C.load_jsonl(log)
    rep.add("stub run: every job ran exactly once", sorted((r["arm"], r["phase"]) for r in runs) == sorted((j.arm, j.phase) for j in jobs),
            f"{len(runs)} runs in {time.time() - t0:.1f} s")
    overlap = lambda a, b: a["t0"] < b["t1"] and b["t0"] < a["t1"]  # noqa: E731
    rep.add("stub run: never two jobs on one GPU at a time", not any(
        overlap(a, b) for i, a in enumerate(runs) for b in runs[i + 1:] if a["gpu"] == b["gpu"]))
    t2 = [r for r in runs if r["phase"] in ("T1", "T2")]
    rep.add("stub run: tau2 jobs serialized on one API key", not any(overlap(a, b) for i, a in enumerate(t2) for b in t2[i + 1:]))
    rep.add("stub run: the incomplete job is reported, the rest complete", [j.name for j in failed] == ["fake2/S3"],
            str([j.name for j in failed]))
    rep.add("stub run: Qwen3-8B jobs get no --model / --api-lane (the phase runner's defaults)",
            all(r["model"] is None and r["lane"] is None for r in runs))

    # 4. the other models
    stub_ck = None
    for model in OTHER:
        src = other_tokenizer(model, given)
        if src is None:
            rep.add(f"{model}: dry run", None, f"no tokenizer (models/{OTHER[model]['dirs'][0]} or --tokenizer-dir {model}=DIR)")
            continue
        cks = check_other_model(model, src, tmp, env, rep)
        stub_ck = stub_ck or (model, cks)
    if stub_ck:
        model, cks = stub_ck
        os.environ["QADX_API_KEY_2"] = "dummy-key-2"
        log3 = tmp / "stub_log_other.jsonl"
        os.environ["STUB_LOG"] = str(log3)
        res3 = tmp / "results_stub_other"
        (res3 / "logs").mkdir(parents=True)
        RUN.LOG = res3 / "logs" / "scheduler.log"
        jobs = [RUN.Job(arm, str(ck), ph, i, model, n) for i, (n, arm, ck, ph) in
                enumerate((n, f"o{n}", ck, ph) for n, ck in enumerate(cks) for ph in ("T", "S1"))]
        failed, _ = RUN.run_jobs(sorted(jobs, key=lambda j: (-j.hours, j.idx)), [0, 1], args, res3)
        runs = C.load_jsonl(log3)
        rep.add(f"stub run ({model}): --model and alternating --api-lane passed to the phase runner",
                not failed and sorted((r["arm"], r["phase"], r["model"], r["lane"]) for r in runs) ==
                sorted([("o0", "T", model, "1"), ("o0", "S1", model, None), ("o1", "T", model, "2"), ("o1", "S1", model, None)]),
                str(sorted((r["arm"], r["phase"], r["model"], r["lane"]) for r in runs)))
        os.environ.pop("QADX_API_KEY_2", None)
    # 5. evaluation spec v2 of the other models
    import phases as P
    import summary as SUM
    res5 = tmp / "results_spec"
    RP.RES = res5
    for model in OTHER:
        tbl, v1 = P.table(model), P.MODEL_V1_TASKS[model]
        kc = {sp.split(":")[0]: (int(sp.split(":")[1]), int(sp.split(":")[2])) for ph in tbl for sp in P.specs(ph, model)}
        k1 = {sp.split(":")[0]: (int(sp.split(":")[1]), int(sp.split(":")[2])) for ph in v1 for sp in v1[ph].split(",")}
        want_k = {b: 16 for b in P.COMPS_ORDER} | {"gpqa_diamond": 16, "lcb_v6": 8} | ({"lcb_pro": 8} if "S4" in tbl else {})
        rep.add(f"{model}: spec v2, samples per question as Qwen3-8B's, caps as in v1",
                P.spec_version(model) == 2 and all(kc[b][0] == k for b, k in want_k.items()) and sorted(v1) == ["G", "S1", "S3"]
                and all(k1[b][0] == (4 if b == "lcb_v6" else 8) and k1[b][1] == kc[b][1] for b in k1)
                and all(P.v1_k(b, kc[b][0], model) == k1.get(b, kc[b])[0] for b in kc), json.dumps(kc))
        RP.set_model(model)
        arm = f"spec-{model}"
        d = res5 / arm
        d.mkdir(parents=True)
        (d / "arm.json").write_text(json.dumps({"model": model}))
        for b, stamp in (("aime26", {"k": 8, "spec_version": 1, "model": model}), ("gpqa_diamond", None),
                         ("lcb_v6", {"k": 8, "spec_version": 2, "model": model})):
            (d / f"{b}.jsonl").write_text("")
            (d / f"{b}.grades.jsonl").write_text("")
            if stamp:
                (d / f"{b}.spec").write_text(json.dumps(stamp) + "\n")
            RP.start_bench(arm, b, kc[b][0])
        moved = sorted(str(f.relative_to(d / "superseded")) for f in (d / "superseded").rglob("*") if f.is_file())
        stamps = {b: json.loads((d / f"{b}.spec").read_text()) for b in ("aime26", "gpqa_diamond", "lcb_v6")}
        rep.add(f"{model}: v1 results (stamped and unstamped) moved to superseded/k8, the current one kept, stamps v2",
                moved == ["k8/aime26.grades.jsonl", "k8/aime26.jsonl", "k8/aime26.spec", "k8/gpqa_diamond.grades.jsonl",
                          "k8/gpqa_diamond.jsonl"] and not (d / "aime26.jsonl").exists() and (d / "lcb_v6.jsonl").exists()
                and all(v == {"k": kc[b][0], "spec_version": 2, "model": model} for b, v in stamps.items()), f"{moved} {stamps}")
        job = RUN.Job(arm, "ck", "S1", 0, model)
        comp = {}
        for name, st in (("v1 tasks", dict(complete=True, tasks=v1["S1"])), ("no tasks field", dict(complete=True)),
                         ("current tasks", dict(complete=True, tasks=tbl["S1"]["tasks"]))):
            (d / "S1.status.json").write_text(json.dumps(st))
            comp[name] = (RUN.phase_complete(res5, job), SUM.arm_summary(res5, arm, model)["phases"]["S1"])
        rep.add(f"{model}: S1 complete only under the current tasks (run.py and summary.py)",
                comp == {"v1 tasks": (False, False), "no tasks field": (False, False), "current tasks": (True, True)}, str(comp))
    RP.set_model("qwen3-8b")
    print(f"work dir {tmp}")
    rep.done()


if __name__ == "__main__":
    main()
