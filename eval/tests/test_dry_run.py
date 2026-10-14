"""Scheduler tests (CPU only, nothing is started):

1. run.py --dry-run for a fake arm on 8 GPUs: a checkpoint directory holding only a Qwen3-8B tokenizer and config
   (no weights). Checks the plan (9 jobs, every GPU busy in the first wave, tau2 jobs never overlapping on one API key),
   one serve command per job on its own GPU / port, a first request for every completion benchmark, the tau2 and BFCL
   commands, and that the dry run wrote nothing and started no container.
2. --shard 0/2 and 1/2 split the jobs of two arms disjointly and completely.
3. The real scheduling loop (run_jobs) with a stub in place of run_phase.py: every job runs exactly once, at most one
   job per GPU at a time, tau2 jobs serialized per API key, an incomplete job is reported and the others go on.
    .venv-client/bin/python eval/tests/test_dry_run.py [--tokenizer-from DIR]
Tokenizer source: --tokenizer-from, else models/Qwen3-8B.
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
    f.write(json.dumps(dict(arm=a["--arm"], phase=a["--phase"], gpu=int(a["--gpu"]), t0=t0, t1=time.time())) + "\\n")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer-from")
    a = ap.parse_args()
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
    print(f"work dir {tmp}")
    rep.done()


if __name__ == "__main__":
    main()
