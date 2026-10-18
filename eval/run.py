#!/usr/bin/env python3
"""Evaluate Qwen3-8B checkpoints on one node: (arm x phase) jobs, one vLLM server per GPU, a GPU takes the next job
when it is free. Run with the client environment:

    .venv-client/bin/python eval/run.py --arm qad030=/path/to/checkpoint [--arm nvfp4 --arm bf16] [options]

    --phases S1,S2,...   phases to run (default: all of S1 S2 S3 S4 L1 M T1 T2 B; see eval/lib/phases.py)
    --gpus 0,1,...       GPUs to use (default: all GPUs of the node)
    --shard i/n          run only this node's share of the job list (n nodes given the identical arguments)
    --dry-run            print the plan, every job's docker serve command and first request; start nothing
    --results DIR        output directory (default eval/results)
    --grade-only         (re)grade what is there and write the summary; start no server
    --tau2-per-key N     tau2 jobs running at once per API key (default 1)
    --limit N            pilot: first N items per benchmark, written under <arm>__limitN/
    --force              rerun phases already marked complete (finished samples are still skipped)
    --no-compile-cache   do not mount the per-GPU vLLM compile cache under .cache/vllm

Jobs run longest first. A job that stops early is resumed by running the same command again: finished samples are
skipped, complete phases are not restarted. After each phase its benchmarks are graded (one grading job at a time);
LiveCodeBench Pro is graded at the end, when no server is running, because its judge measures time. Outputs:
<results>/<arm>/<bench>.jsonl (generations), <bench>.grades.jsonl, <phase>.status.json, tau2_runs/, bfcl_runs/;
<results>/logs/ (scheduler.log, <arm>_<phase>.log, server_*.log, tau2_*, bfcl_*); <results>/summary.{md,json}.
"""

import argparse
import hashlib
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import models as M  # noqa: E402
import paths  # noqa: E402
import phases as P  # noqa: E402
import serve  # noqa: E402

JOB_TIMEOUT_S = 12 * 3600
POLL_S = 10
PHASE_RUNNER = paths.LIB / "run_phase.py"  # tests substitute a stub
SLOWDOWN = {"bf16": 1.8, "nvfp4": 1.3}  # built-in baselines write longer outputs than QAD checkpoints (time estimates only)
DATA_FILES = {  # benchmark -> files under eval/data it needs (generation and grading)
    "lcb_v6": ["lcb_v6.jsonl", "lcb_test6.jsonl"], "lcb_pro": ["lcb_pro.jsonl", "lcbpro_excluded.json"],
    "mrcr8": ["mrcr8.jsonl"], "longbench_pro": ["longbench_pro.jsonl"],
}
LOG = None


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG:
        with open(LOG, "a") as f:
            f.write(line + "\n")


class Job:
    def __init__(self, arm, ckpt, phase, idx):
        self.arm, self.ckpt, self.phase, self.idx = arm, ckpt, phase, idx
        self.hours = P.PHASES[phase]["hours"] * SLOWDOWN.get(arm, 1.0)
        self.lane = P.PHASES[phase].get("api_lane")
        if self.lane == 2 and not os.environ.get("QADX_API_KEY_2"):
            self.lane = 1  # one key: both tau2 phases share it

    @property
    def name(self):
        return f"{self.arm}/{self.phase}"


def gpu_list(spec):
    if spec:
        return [int(x) for x in spec.split(",") if x != ""]
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        n = sum(1 for l in out.splitlines() if l.startswith("GPU "))
        return list(range(n)) if n else list(range(8))
    except Exception:
        return list(range(8))


def lpt_shards(jobs, n):
    """Longest-processing-time partition of the jobs over n nodes (deterministic for identical arguments)."""
    loads, parts = [0.0] * n, [[] for _ in range(n)]
    for j in sorted(jobs, key=lambda j: (-j.hours, j.idx)):
        i = min(range(n), key=lambda i: (loads[i], i))
        loads[i] += j.hours
        parts[i].append(j)
    return parts, loads


def simulate(jobs, gpus, per_key):
    """Estimated start GPU / time of each job under the scheduler's rules (for the dry-run plan)."""
    free = {g: 0.0 for g in gpus}
    lane_busy = {1: [], 2: []}
    plan, pending, t = [], list(jobs), 0.0
    while pending:
        g = min(free, key=lambda g: (free[g], g))
        t = free[g]
        pick = None
        for j in pending:
            if j.lane is None or sum(1 for e in lane_busy[j.lane] if e > t) < per_key:
                pick = j
                break
        if pick is None:  # every pending job waits for an API lane
            free[g] = min(e for l in lane_busy.values() for e in l if e > t)
            continue
        pending.remove(pick)
        end = t + pick.hours
        if pick.lane:
            lane_busy[pick.lane].append(end)
        free[g] = end
        plan.append((pick, g, t))
    return plan, max(free.values()) if free else 0.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def check_data(phases):
    """Missing or altered prompt / test files the selected phases need (empty list = fine)."""
    want = {}
    for line in (paths.DATA / "SHA256SUMS").read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, f = line.split()
            want[f] = h
    need = set()
    for ph in phases:
        for b in P.benches(ph):
            if b.startswith("tau2_") or b == "bfcl_mt":
                continue
            need.update(DATA_FILES.get(b, [f"{b}.jsonl"]))
    bad = []
    for f in sorted(need):
        p = paths.DATA / f
        if not p.exists():
            bad.append(f"{f}: missing (eval/fetch_data.sh or eval/prep/*)")
        elif f in want and sha256(p) != want[f]:
            bad.append(f"{f}: sha256 differs from data/SHA256SUMS")
    return bad


def preflight(jobs, gpus, dry):
    problems = []
    for arm, ckpt in sorted({(j.arm, j.ckpt) for j in jobs}):
        problems += M.check_checkpoint(ckpt)
    problems += check_data(sorted({j.phase for j in jobs}))
    if any(j.lane for j in jobs) and not (os.environ.get("QADX_API_BASE") and os.environ.get("QADX_API_KEY")):
        problems.append("tau2 phases need QADX_API_BASE and QADX_API_KEY (eval/README.md)")
    if not dry:
        if subprocess.run(["docker", "image", "inspect", paths.vllm_image()], capture_output=True).returncode:
            problems.append(f"serving image {paths.vllm_image()} not present (setup/setup.sh images)")
        for g in gpus:
            used = serve.gpu_memory_used(g)
            if used is not None and used >= 2000:
                problems.append(f"GPU {g} has {used} MiB in use")
    return problems


def bind_arm(results, arm, ckpt):
    """<results>/<arm>/arm.json records the checkpoint; refuse to mix two checkpoints under one arm name."""
    d = Path(results) / arm
    d.mkdir(parents=True, exist_ok=True)
    f = d / "arm.json"
    if f.exists():
        old = json.load(open(f))
        if old["checkpoint"] != ckpt:
            raise SystemExit(f"arm {arm} already holds results of {old['checkpoint']}; use another arm name for {ckpt}")
        return
    info = dict(arm=arm, checkpoint=ckpt)
    for name in ("config.json", "hf_quant_config.json", "generation_config.json", "tokenizer_config.json"):
        p = Path(ckpt) / name
        if p.exists():
            info[f"sha256_{name}"] = sha256(p)
    rc = subprocess.run(["git", "-C", str(paths.REPO), "rev-parse", "HEAD"], capture_output=True, text=True)
    info.update(repo_commit=rc.stdout.strip(), image=paths.vllm_image(), created=time.strftime("%Y-%m-%dT%H:%M:%S"))
    f.write_text(json.dumps(info, indent=1) + "\n")


def phase_complete(results, job):
    """Complete under the current spec: a status file without a tasks field was written under spec v1 (phases.V1_TASKS)."""
    f = Path(results) / job.arm / f"{job.phase}.status.json"
    if not f.exists():
        return False
    st = json.load(open(f))
    tasks = st.get("tasks", P.V1_TASKS.get(job.phase, P.PHASES[job.phase]["tasks"]))
    return st.get("complete") is True and tasks == P.PHASES[job.phase]["tasks"]


class Grader(threading.Thread):
    """Grades finished phases one at a time; LiveCodeBench Pro is collected and graded by finish()."""

    def __init__(self, results):
        super().__init__(daemon=True)
        self.results, self.q, self.lcbpro, self.status = results, queue.Queue(), set(), {}

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                return
            arm, phase = item
            benches = P.benches(phase)
            if "lcb_pro" in benches:
                self.lcbpro.add(arm)
            self._grade(arm, benches, lcbpro=False)
            self.q.task_done()

    def _grade(self, arm, benches, lcbpro):
        cmd = [sys.executable, str(paths.LIB / "grade.py"), "--results", str(self.results), "--arm", arm,
               "--benches", ",".join(benches)] + ([] if lcbpro else ["--no-lcbpro"])
        with open(paths.logs_dir(self.results) / f"grade_{arm}.log", "a") as lf:
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=lf, text=True)
            lf.write(r.stdout)
        try:
            st = json.loads(r.stdout.strip().splitlines()[-1])[arm]
        except Exception:
            st = {b: "failed" for b in benches}
        self.status.setdefault(arm, {}).update(st)
        log(f"graded {arm}: " + ", ".join(f"{b} {s}" for b, s in st.items()))

    def finish(self):
        self.q.put(None)
        self.q.join()
        for arm in sorted(self.lcbpro):
            log(f"grading {arm}/lcb_pro (LiveCodeBench Pro judge)")
            self._grade(arm, ["lcb_pro"], lcbpro=True)


def run_jobs(jobs, gpus, args, results):
    grader = Grader(results)
    grader.start()
    pending = [j for j in jobs]
    for j in [j for j in pending if phase_complete(results, j) and not args.force]:
        log(f"{j.name}: complete, not rerun (--force reruns it)")
        pending.remove(j)
        grader.q.put((j.arm, j.phase))
    running, failed, blocked = {}, [], {}
    stop = {"flag": False}

    def on_signal(sig, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    def kill(gpu, job, proc):
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=180)
        except (subprocess.TimeoutExpired, ProcessLookupError):
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        subprocess.run(["docker", "rm", "-f", serve.container(gpu)], capture_output=True)

    while (pending or running) and not stop["flag"]:
        for g in gpus:
            if g in running or not pending or time.time() < blocked.get(g, 0):
                continue
            lanes = [r[0].lane for r in running.values() if r[0].lane]
            job = next((j for j in pending if j.lane is None or lanes.count(j.lane) < args.tau2_per_key), None)
            if job is None:
                break
            if not serve.wait_gpu_free(g, timeout_s=120):
                log(f"GPU {g} is not free (memory in use); retrying it in 10 min")
                blocked[g] = time.time() + 600
                continue
            pending.remove(job)
            cmd = [sys.executable, str(PHASE_RUNNER), "--arm", job.arm, "--checkpoint", job.ckpt,
                   "--phase", job.phase, "--gpu", str(g), "--results", str(results)]
            cmd += (["--limit", str(args.limit)] if args.limit else []) + (["--no-compile-cache"] if args.no_compile_cache else [])
            lf = open(paths.logs_dir(results) / f"{job.arm}_{job.phase}.log", "a")
            proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
            running[g] = (job, proc, time.time(), lf)
            log(f"START {job.name} on GPU {g} (port {serve.port(g)}; {len(pending)} pending)")
        time.sleep(POLL_S)
        for g, (job, proc, t0, lf) in list(running.items()):
            rc = proc.poll()
            if rc is None and time.time() - t0 > JOB_TIMEOUT_S:
                log(f"{job.name}: timeout after {JOB_TIMEOUT_S / 3600:.0f} h, stopping it")
                kill(g, job, proc)
                rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            del running[g]
            subprocess.run(["docker", "rm", "-f", serve.container(g)], capture_output=True)
            el = (time.time() - t0) / 3600
            if rc == 0:
                log(f"END {job.name} on GPU {g}: complete in {el:.2f} h")
            else:
                failed.append(job)
                log(f"END {job.name} on GPU {g}: {'INCOMPLETE' if rc == 3 else f'FAILED (exit {rc})'} after {el:.2f} h; "
                    f"see logs/{job.arm}_{job.phase}.log; rerun the same command to resume")
            grader.q.put((job.arm, job.phase))
    if stop["flag"]:
        log(f"interrupted: stopping {len(running)} running job(s); rerun the same command to resume")
        for g, (job, proc, t0, lf) in running.items():
            kill(g, job, proc)
            lf.close()
        sys.exit(130)
    grader.finish()
    return failed, grader.status


def main():
    global LOG
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME (built-in: bf16, nvfp4) or NAME=/checkpoint/dir")
    ap.add_argument("--phases", default=",".join(P.ORDER))
    ap.add_argument("--gpus", default="")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--results", default=str(paths.RESULTS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--grade-only", action="store_true")
    ap.add_argument("--tau2-per-key", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-compile-cache", action="store_true")
    args = ap.parse_args()
    arms = [M.resolve_arm(s) for s in args.arm]
    if len({a for a, _ in arms}) != len(arms):
        raise SystemExit("arm names must be unique")
    if args.limit:
        arms = [(f"{a}__limit{args.limit}", c) for a, c in arms]
    phases = [p for p in args.phases.split(",") if p]
    for p in phases:
        if p not in P.PHASES:
            raise SystemExit(f"unknown phase {p}; phases: {' '.join(P.ORDER)}")
    gpus = gpu_list(args.gpus)
    si, sn = (int(x) for x in args.shard.split("/"))
    results = Path(args.results).resolve()
    jobs = [Job(a, c, p, i) for i, (a, c, p) in enumerate((a, c, p) for a, c in arms for p in phases)]
    parts, loads = lpt_shards(jobs, sn)
    jobs = sorted(parts[si], key=lambda j: (-j.hours, j.idx))

    if args.dry_run:
        print(f"# dry run: {len(jobs)} jobs on GPUs {gpus} (shard {si}/{sn}; estimated load per shard "
              f"{', '.join(f'{x:.1f} h' for x in loads)})")
        problems = preflight(jobs, gpus, dry=True)
        for p in problems:
            print(f"PREFLIGHT: {p}")
        plan, span = simulate(jobs, gpus, args.tau2_per_key)
        print(f"# plan (estimated hours on B200; makespan ~{span:.1f} h + grading)")
        for j, g, t in plan:
            print(f"  t+{t:5.1f} h  GPU {g}  {j.name:<28} ~{j.hours:.1f} h" + (f"  (API lane {j.lane})" if j.lane else ""))
        print()
        sys.path.insert(0, str(paths.LIB))
        import run_phase as RP

        RP.RES, RP.LOGS = results, paths.logs_dir(results)
        for j, g, _ in plan:
            try:
                RP.dry_run(j.arm, j.ckpt, j.phase, g, args.limit, not args.no_compile_cache)
            except Exception as e:  # noqa: BLE001  (e.g. a data file not fetched yet: listed by the preflight above)
                print(f"## {j.name} on GPU {g}: cannot show its requests: {type(e).__name__}: {e}\n")
        return

    results.mkdir(parents=True, exist_ok=True)
    paths.logs_dir(results).mkdir(parents=True, exist_ok=True)
    LOG = paths.logs_dir(results) / "scheduler.log"
    for a, c in arms:
        bind_arm(results, a, c)
    if args.grade_only:
        g = Grader(results)
        for a, _ in arms:
            g._grade(a, [b for p in phases for b in P.benches(p)], lcbpro=True)
    else:
        problems = preflight(jobs, gpus, dry=False)
        if problems:
            for p in problems:
                log(f"PREFLIGHT: {p}")
            raise SystemExit("preflight failed; nothing started")
        log(f"{len(jobs)} jobs on GPUs {gpus} (shard {si}/{sn}): " + ", ".join(j.name for j in jobs))
        failed, _ = run_jobs(jobs, gpus, args, results)
        if failed:
            log("not complete: " + ", ".join(j.name for j in failed) + " (rerun the same command to resume)")
    subprocess.run([sys.executable, str(paths.LIB / "summary.py"), "--results", str(results),
                    "--arms", ",".join(a for a, _ in arms)], stdout=subprocess.DEVNULL)
    log(f"summary: {results / 'summary.md'}")


if __name__ == "__main__":
    main()
