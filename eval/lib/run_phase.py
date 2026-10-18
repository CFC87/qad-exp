"""Run one phase (phases.py) for one arm on one GPU: start vLLM, generate, stop vLLM. Runs in .venv-client.

    run_phase.py --arm NAME --checkpoint DIR --phase S1 --gpu 0 [--results DIR] [--limit N] [--dry-run]

Completion benchmarks go through /v1/completions with the chat template rendered client-side by the checkpoint's own
tokenizer (transformers from the client lock), so the prompt tokens are identical across arms and machines. Every
(question, sample) request is shuffled with a seed derived from the benchmark name only, so all arms send requests in the
same order. Results are appended per sample to <results>/<arm>/<bench>.jsonl; a rerun skips the samples already there,
unless the file was made with another number of samples per question (an earlier evaluation spec, phases.py): then it is
moved to <arm>/superseded/k<old>/ and the benchmark is generated afresh, never extended.
tau2 and BFCL run their own clients against the same server (run_tau2 / run_bfcl).
Exit code 0 when every sample of the phase is present, 3 when the phase finished incomplete.
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import models as M  # noqa: E402
import paths  # noqa: E402
import phases as P  # noqa: E402
import serve  # noqa: E402

RES = paths.RESULTS
LOGS = paths.logs_dir()
TAU2_SEED = 300           # tau2's official seed; tau2 seeds every agent call from it
TAU2_CLI = paths.LIB / "tau2_run.py"
BFCL_HANDLER = "qadx/qwen3-ct"
BFCL_CATEGORIES = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
BFCL_PER_CATEGORY = 200


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def stable_seed(*parts):
    return int(hashlib.sha256("/".join(map(str, parts)).encode()).hexdigest(), 16) % (2**31 - 1)


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------- completion benchmarks


def load_items(bench, limit=None):
    items = load_jsonl(paths.DATA / f"{bench}.jsonl")
    return items[:limit] if limit else items


def render(tok, msgs, thinking=True):
    """Chat template with the generation prompt, thinking mode on (Qwen3 templates read enable_thinking)."""
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def prompt_text(tok, it, max_tokens, max_model_len):
    """The exact prompt string of one item."""
    prompt = it.get("prompt", "")
    if it.get("truncate_middle"):
        # LongBench Pro: keep the head and tail tokens, as the official model_manager._truncate_prompt does, at its
        # default length or less if the output budget would not fit in the window. A prompt with no more characters
        # than the limit cannot have more tokens, so only long ones are tokenized.
        limit = min(it["truncate_middle"], max_model_len - min(it.get("max_tokens", max_tokens), max_tokens) - 512)
        if len(prompt) > limit:
            ids = tok(prompt, add_special_tokens=False)["input_ids"]
            if len(ids) > limit:
                prompt = tok.decode(ids[:limit // 2] + ids[-(limit // 2):], skip_special_tokens=True)
    msgs = it.get("messages") or [{"role": "user", "content": prompt}]  # MRCR items are multi-turn conversations
    text = render(tok, msgs, it.get("enable_thinking", True))
    return text + it.get("prefill", "")


def task_order(bench, items, k, done):
    """(item index, sample index) pairs still to run, in the fixed shuffled order. The permutation depends only on the
    benchmark name and the number of pending tasks, so every arm sends the same sequence."""
    tasks = [(qi, j) for qi, it in enumerate(items) for j in range(k) if (it["qid"], j) not in done]
    random.Random(stable_seed(bench, "order")).shuffle(tasks)
    return tasks


def request_body(tok, it, text, max_tokens, max_model_len):
    """The /v1/completions payload. No per-request seed (the server seed is fixed); the output budget leaves room for
    the prompt inside the window: min(cap, window - prompt tokens - 64)."""
    budget = min(it.get("max_tokens", max_tokens), max_model_len - len(tok(text, add_special_tokens=True)["input_ids"]) - 64)
    return dict(model=M.SERVED_NAME, prompt=text, max_tokens=budget, skip_special_tokens=False, **M.SAMPLING)


def done_samples(out_path):
    done = set()
    if out_path.exists():
        for r in load_jsonl(out_path):
            done.add((r["qid"], r["j"]))
    return done


def made_with_k(arm, bench, k):
    """The k the existing <bench>.jsonl of arm was generated with: its .spec stamp, else (no stamp: written under spec
    v1) phases.V1_K; None when there is no file."""
    d = RES / arm
    if not (d / f"{bench}.jsonl").exists():
        return None
    stamp = d / f"{bench}.spec"
    return int(json.loads(stamp.read_text())["k"]) if stamp.exists() else P.V1_K.get(bench, k)


def start_bench(arm, bench, k):
    """Before generating: a file made with another k is moved to <arm>/superseded/k<old>/ (with its grades and stamp),
    so the benchmark starts afresh; then the stamp records k."""
    d = RES / arm
    d.mkdir(parents=True, exist_ok=True)
    old = made_with_k(arm, bench, k)
    if old is not None and old != k:
        dst = d / "superseded" / f"k{old}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in (d / f"{bench}.jsonl", d / f"{bench}.grades.jsonl", d / f"{bench}.spec"):
            if f.exists():
                shutil.move(str(f), str(dst / f.name))
        log(f"{arm}/{bench}: results made with k={old} moved to {dst}; generating k={k} afresh")
    (d / f"{bench}.spec").write_text(json.dumps(dict(k=k, spec_version=P.SPEC_VERSION)) + "\n")


async def generate(arm, bench, k, max_tokens, concurrency, tok, gpu, max_model_len, limit=None):
    import aiohttp

    items = load_items(bench, limit)
    out_path = RES / arm / f"{bench}.jsonl"
    start_bench(arm, bench, k)
    done = done_samples(out_path)
    order = task_order(bench, items, k, done)
    log(f"{arm}/{bench}: {len(items)} questions x k={k}, {len(done)} already done, {len(order)} to run")
    if not order:
        return
    texts = {qi: prompt_text(tok, items[qi], max_tokens, max_model_len) for qi in sorted({qi for qi, _ in order})}
    tasks = [(items[qi], j, texts[qi]) for qi, j in order]
    base = f"http://127.0.0.1:{serve.port(gpu)}"
    sem = asyncio.Semaphore(concurrency)
    fout = open(out_path, "a")
    stats = {"n": 0, "tokens": 0, "t0": time.time()}

    async def one(session, it, j, text):
        seed = None
        body = request_body(tok, it, text, max_tokens, max_model_len)
        async with sem:
            for attempt in range(3):
                try:
                    t = time.time()
                    async with session.post(f"{base}/v1/completions", json=body,
                                            timeout=aiohttp.ClientTimeout(total=4 * 3600)) as r:
                        resp = await r.json()
                    if "choices" not in resp:
                        raise RuntimeError(str(resp)[:500])
                    ch = resp["choices"][0]
                    rec = dict(
                        qid=it["qid"], j=j, seed=seed, answer=it["answer"], type=it["type"],
                        finish_reason=ch.get("finish_reason"),
                        prompt_tokens=resp["usage"]["prompt_tokens"],
                        completion_tokens=resp["usage"]["completion_tokens"],
                        latency=time.time() - t, text=ch["text"],
                    )
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    stats["n"] += 1
                    stats["tokens"] += rec["completion_tokens"]
                    if stats["n"] % 50 == 0:
                        el = time.time() - stats["t0"]
                        log(f"{arm}/{bench}: {stats['n']}/{len(tasks)} done, {stats['tokens'] / el:.0f} out tok/s")
                    return
                except Exception as e:
                    log(f"request failed ({attempt}): {it['qid']} j={j}: {e!r}"[:300])
                    await asyncio.sleep(10)
            log(f"GIVING UP {it['qid']} j={j}")

    conn = aiohttp.TCPConnector(limit=concurrency + 8)
    async with aiohttp.ClientSession(connector=conn) as session:
        await asyncio.gather(*(one(session, *t) for t in tasks))
    fout.close()
    el = time.time() - stats["t0"]
    log(f"{arm}/{bench}: finished {stats['n']} samples in {el / 60:.1f} min, {stats['tokens'] / max(el, 1):.0f} out tok/s")
    with open(RES / "timing.jsonl", "a") as f:
        f.write(json.dumps(dict(arm=arm, bench=bench, n=stats["n"], out_tokens=stats["tokens"], seconds=el,
                                concurrency=concurrency)) + "\n")


def first_requests(phase, tok, limit=None, arm=None):
    """For --dry-run: the first request each benchmark of a completion phase would send (respecting samples already
    present for arm)."""
    cfg = P.PHASES[phase]
    out = []
    for spec in P.specs(phase):
        bench, k, mt = spec.split(":")
        items = load_items(bench, limit)
        done = done_samples(RES / arm / f"{bench}.jsonl") if arm and made_with_k(arm, bench, int(k)) == int(k) else set()
        order = task_order(bench, items, int(k), done)
        if not order:
            out.append((bench, None, None))
            continue
        qi, j = order[0]
        text = prompt_text(tok, items[qi], int(mt), cfg["max_model_len"])
        out.append((bench, (items[qi]["qid"], j, len(order)), request_body(tok, items[qi], text, int(mt), cfg["max_model_len"])))
    return out


def complete_completions(arm, phase, limit=None):
    missing = {}
    for spec in P.specs(phase):
        bench, k, _ = spec.split(":")
        items = load_items(bench, limit)
        done = done_samples(RES / arm / f"{bench}.jsonl")
        want = {(it["qid"], j) for it in items for j in range(int(k))}
        if want - done:
            missing[bench] = len(want - done)
    return missing


# ------------------------------------------------------------------------------------------------------------ tau2
# tau2-bench v1.0.1 (third_party/tau2-bench, its own environment): base task split, official seed 300, no retries of
# failed simulations (a retried simulation only ever turns failures into successes). The user simulator and the retail
# natural-language judge are an external OpenAI-compatible model (default gpt-5.2, no reasoning, temperature 0),
# configured by environment variables (tau2_api_config). All API calls go through api_throttle.py: a token bucket and
# waits-and-retries on 429 / 5xx, so a rate-limit burst delays simulations instead of failing them.

TAU2_API_ARGS = {"reasoning_effort": "none", "extra_body": {"temperature": 0}}


def tau2_api_config(lane, require=True):
    """Endpoint, key, litellm model id and request rate for one API lane (1 or 2) from QADX_API_* variables.

    litellm's "openai/" provider prefix is stripped before the request, so the endpoint receives
    QADX_TAU2_MODEL_PREFIX + QADX_TAU2_MODEL (default "openai/openai/gpt-5.2", the id a LiteLLM proxy uses for gpt-5.2;
    set QADX_TAU2_MODEL_PREFIX="" for an endpoint that serves it as "gpt-5.2"). With an id litellm does not know, it
    drops a top-level temperature, so the args carry temperature 0 in extra_body."""
    e = os.environ
    base, key = e.get("QADX_API_BASE"), e.get("QADX_API_KEY_2" if lane == 2 and e.get("QADX_API_KEY_2") else "QADX_API_KEY")
    if require and not (base and key):
        raise SystemExit("tau2 needs QADX_API_BASE and QADX_API_KEY (an OpenAI-compatible endpoint serving gpt-5.2); "
                         "see eval/README.md")
    model = "openai/" + e.get("QADX_TAU2_MODEL_PREFIX", "openai/openai/") + e.get("QADX_TAU2_MODEL", "gpt-5.2")
    return dict(base=base, key=key, model=model, rps=float(e.get("QADX_TAU2_RPS", "1.2")),
                key_name="QADX_API_KEY_2" if lane == 2 and e.get("QADX_API_KEY_2") else "QADX_API_KEY")


def tau2_results_state(path):
    """None (no results.json), 'complete' or 'partial' (an interrupted run: resumed with --auto-resume)."""
    if not path.exists():
        return None
    try:
        d = json.load(open(path))
        return "complete" if len(d["simulations"]) >= len(d["tasks"]) * d["info"]["num_trials"] else "partial"
    except Exception:
        return "partial"


def tau2_strip_api_failures(arm, domain, res):
    """Simulations an API error ended (listed by tau2_convert.py in tau2_<domain>.left_out.jsonl with error_class
    "api") say nothing about the arm and are re-run: they are removed from results.json (a backup is kept) so that
    tau2 --auto-resume runs exactly them again, with their original seeds. Unclassified errors stay listed for a look
    by hand. Returns the number removed."""
    lo = RES / arm / f"tau2_{domain}.left_out.jsonl"
    if not (res.exists() and lo.exists()):
        return 0
    drop = {(r["qid"].split(":", 1)[1], r["j"]) for r in load_jsonl(lo) if r.get("error_class") == "api"}
    if not drop:
        return 0
    d = json.load(open(res))
    keep = [x for x in d["simulations"] if (str(x["task_id"]), x["trial"]) not in drop]
    n = len(d["simulations"]) - len(keep)
    if n:
        shutil.copy(res, res.with_name(f"results.before_rerun_{time.strftime('%Y%m%d%H%M%S')}.json"))
        d["simulations"] = keep
        tmp = res.with_name("results.json.part")
        with open(tmp, "w") as f:
            json.dump(d, f)
        tmp.rename(res)
    return n


def tau2_commands(arm, specs, concurrency, gpu, api, limit=None):
    """(domain, results.json, command or None) per domain; None = results complete, convert only."""
    root = RES / arm / "tau2_runs"
    agent_args = {"api_base": f"http://127.0.0.1:{serve.port(gpu)}/v1", "api_key": "EMPTY",
                  **{x: M.SAMPLING[x] for x in ("temperature", "top_p", "top_k", "min_p")}}
    out = []
    for spec in specs:
        bench, k, _ = spec.split(":")
        domain = bench.removeprefix("tau2_")
        res = root / "simulations" / domain / "results.json"
        state = tau2_results_state(res)
        if state == "complete":
            out.append((domain, res, None))
            continue
        cmd = [str(paths.TAU2_PY), str(TAU2_CLI), "run", "--domain", domain, "--task-split-name", "base",
               "--agent-llm", f"openai/{M.SERVED_NAME}", "--agent-llm-args", json.dumps(agent_args),
               "--user-llm", api["model"], "--user-llm-args", json.dumps(TAU2_API_ARGS),
               "--num-trials", k, "--seed", str(TAU2_SEED), "--max-retries", "0", "--max-concurrency", str(concurrency),
               "--save-to", domain] + (["--num-tasks", str(limit)] if limit else []) + \
              (["--auto-resume"] if state == "partial" else [])
        out.append((domain, res, cmd))
    return out


async def run_tau2(arm, specs, concurrency, gpu, lane, limit=None):
    root = RES / arm / "tau2_runs"
    root.mkdir(parents=True, exist_ok=True)
    if not (root / "tau2").exists():  # tau2 reads DATA_DIR/tau2/domains and writes DATA_DIR/simulations
        (root / "tau2").symlink_to(paths.TAU2_DIR / "data" / "tau2")
    api = tau2_api_config(lane)
    env = {k: v for k, v in os.environ.items() if not k.startswith("QADX_API_KEY")}
    env.update(TAU2_DATA_DIR=str(root), TAU2_NL_JUDGE_MODEL=api["model"], TAU2_NL_JUDGE_ARGS=json.dumps(TAU2_API_ARGS))
    throttle = throttle_log = None
    if api["rps"] > 0:
        tport = serve.THROTTLE_PORT_BASE + int(gpu)
        throttle_log = open(LOGS / f"tau2_{arm}_throttle.log", "a")
        throttle = subprocess.Popen([sys.executable, str(paths.LIB / "api_throttle.py"), "--port", str(tport),
                                     "--rps", str(api["rps"])], stdout=throttle_log, stderr=subprocess.STDOUT,
                                    env=dict(env, QADX_THROTTLE_UPSTREAM=api["base"], QADX_THROTTLE_KEY=api["key"]))
        await asyncio.sleep(3)
        env.update(OPENAI_API_BASE=f"http://127.0.0.1:{tport}/v1", OPENAI_API_KEY="via-throttle")
    else:
        env.update(OPENAI_API_BASE=api["base"], OPENAI_API_KEY=api["key"])
    for spec in specs:
        domain = spec.split(":")[0].removeprefix("tau2_")
        n = tau2_strip_api_failures(arm, domain, root / "simulations" / domain / "results.json")
        if n:
            log(f"tau2 {domain}: re-running {n} simulations that an API error ended")
    procs = []
    for domain, out, cmd in tau2_commands(arm, specs, concurrency, gpu, api, limit):
        if cmd is None:
            log(f"tau2 {domain}: {out} complete, converting only")
            procs.append((domain, out, None, None))
            continue
        logf = open(LOGS / f"tau2_{arm}_{domain}.log", "a")
        denv = dict(env, TAU2_EMPTY_LOG=str(LOGS / f"tau2_{arm}_{domain}_empty.jsonl"))  # empty agent replies
        log(f"tau2 {domain}: {spec_k(specs, domain)} trials, seed {TAU2_SEED}, concurrency {concurrency}, user "
            f"{api['model']} via {'throttle %.2f req/s' % api['rps'] if throttle else 'direct'} ({api['key_name']})")
        procs.append((domain, out, await asyncio.create_subprocess_exec(*cmd, stdout=logf, stderr=subprocess.STDOUT,
                                                                          env=denv, cwd=str(root)), logf))
    try:
        for domain, out, p, logf in procs:
            if p is not None:
                rc = await p.wait()
                logf.close()
                log(f"tau2 {domain}: exit {rc}")
    finally:
        if throttle:
            throttle.terminate()
            throttle_log.close()
    for domain, out, p, logf in procs:
        if out.exists():
            subprocess.run([sys.executable, str(paths.LIB / "tau2_convert.py"), str(out), str(RES / arm), domain], check=False)
        else:
            log(f"tau2 {domain}: no results.json")


def spec_k(specs, domain):
    return next(s.split(":")[1] for s in specs if s.split(":")[0] == f"tau2_{domain}")


def complete_tau2(arm, phase):
    missing = {}
    for b in P.benches(phase):
        domain = b.removeprefix("tau2_")
        state = tau2_results_state(RES / arm / "tau2_runs" / "simulations" / domain / "results.json")
        lo = RES / arm / f"{b}.left_out.jsonl"
        n_left = len(load_jsonl(lo)) if lo.exists() else 0
        if state != "complete" or n_left or not (RES / arm / f"{b}.grades.jsonl").exists():
            missing[b] = f"results {state}, {n_left} simulations left out (API errors)"
    return missing


# ------------------------------------------------------------------------------------------------------------ BFCL
# BFCL multi-turn (bfcl-eval 2026.3.23 in .venv-bfcl): bfcl_handler.py renders with the checkpoint's chat template and
# parses like vLLM's qwen3 reasoning + hermes tool parsers, sends the model-card sampling and max_tokens = min(cap,
# window - prompt). bfcl caches backend state in module globals, so every replicate's generate and evaluate is its own
# process; generate skips entries already in the replicate's result files, so an interrupted phase resumes.


def bfcl_env(gpu, max_model_len, max_tokens, root):
    env = {k: v for k, v in os.environ.items() if not k.startswith("REMOTE_OPENAI_")}  # would redirect bfcl's client
    env.update(BFCL_PROJECT_ROOT=str(root), LOCAL_SERVER_ENDPOINT="127.0.0.1", LOCAL_SERVER_PORT=str(serve.port(gpu)),
               QADX_BFCL_SERVED_MODEL=M.SERVED_NAME, QADX_BFCL_MAX_MODEL_LEN=str(max_model_len),
               QADX_BFCL_MAX_TOKENS=str(max_tokens),
               QADX_BFCL_SAMPLING=json.dumps({x: M.SAMPLING[x] for x in ("temperature", "top_p", "top_k", "min_p",
                                                                         "presence_penalty", "repetition_penalty")}))
    return env


def bfcl_commands(checkpoint, k, threads, limit=None):
    run = [str(paths.BFCL_PY), str(paths.LIB / "bfcl_run.py")]
    gen = [run + ["generate", "--model", BFCL_HANDLER, "--test-category", "multi_turn", "--skip-server-setup",
                  "--local-model-path", checkpoint, "--num-threads", str(threads), "--result-dir", f"result_j{j}"]
           + (["--run-ids"] if limit else []) for j in range(k)]
    ev = [run + ["evaluate", "--model", BFCL_HANDLER, "--test-category", "multi_turn", "--result-dir", f"result_j{j}",
                 "--score-dir", f"score_j{j}"] + (["--partial-eval"] if limit else []) for j in range(k)]
    return gen, ev


async def run_bfcl(arm, checkpoint, spec, concurrency, gpu, max_model_len, limit=None):
    _, k, mt = spec.split(":")
    k = int(k)
    root = RES / arm / "bfcl_runs"
    root.mkdir(parents=True, exist_ok=True)
    env = bfcl_env(gpu, max_model_len, mt, root)
    if limit:  # first <limit> entries per category (pilots)
        (root / "test_case_ids_to_generate.json").write_text(
            json.dumps({c: [f"{c}_{i}" for i in range(limit)] for c in BFCL_CATEGORIES}))
    threads = max(1, concurrency // k)
    gen, ev = bfcl_commands(checkpoint, k, threads, limit)
    log(f"bfcl multi-turn: {k} replicates x {threads} threads, handler {BFCL_HANDLER}, max_tokens {mt}, served len {max_model_len}")
    procs = []
    for j in range(k):
        logf = open(LOGS / f"bfcl_{arm}_gen_j{j}.log", "a")
        procs.append((j, logf, await asyncio.create_subprocess_exec(*gen[j], stdout=logf, stderr=subprocess.STDOUT,
                                                                     env=env, cwd=str(root))))
    for j, logf, p in procs:
        log(f"bfcl generate j={j}: exit {await p.wait()}")
        logf.close()
    hdir = BFCL_HANDLER.replace("/", "_")
    for j in range(k):
        with open(LOGS / f"bfcl_{arm}_eval_j{j}.log", "a") as logf:
            subprocess.run(ev[j], stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=str(root), check=False)
        subprocess.run([sys.executable, str(paths.LIB / "bfcl_to_rows.py"), "--result", str(root / f"result_j{j}" / hdir),
                        "--score", str(root / f"score_j{j}" / hdir), "--j", str(j), "--records", str(RES / arm / "bfcl_mt.jsonl"),
                        "--grades", str(RES / arm / "bfcl_mt.grades.jsonl"), "--expect", str(limit or BFCL_PER_CATEGORY)]
                       + (["--append"] if j else []), check=False)


def complete_bfcl(arm, phase, limit=None):
    k = int(P.specs(phase)[0].split(":")[1])
    g = RES / arm / "bfcl_mt.grades.jsonl"
    n = len({(r["qid"], r["j"]) for r in load_jsonl(g)}) if g.exists() else 0
    want = k * len(BFCL_CATEGORIES) * (limit or BFCL_PER_CATEGORY)
    return {} if n == want else {"bfcl_mt": f"{n} of {want} entries graded"}


# ------------------------------------------------------------------------------------------------------------ main


def write_status(arm, phase, t0, missing, rc=None):
    st = dict(arm=arm, phase=phase, complete=not missing, missing=missing, seconds=round(time.time() - t0),
              finished=time.strftime("%Y-%m-%dT%H:%M:%S"), exit_code=rc, tasks=P.PHASES[phase]["tasks"],
              spec_version=P.SPEC_VERSION)
    (RES / arm / f"{phase}.status.json").write_text(json.dumps(st, indent=1) + "\n")
    return st


def dry_run(arm, checkpoint, phase, gpu, limit, compile_cache):
    from transformers import AutoTokenizer

    cfg = P.PHASES[phase]
    kind = P.kind(phase)
    cmd = serve.serve_cmd(checkpoint, gpu, cfg["max_model_len"], serve.phase_server_args(cfg, kind), compile_cache)
    print(f"## {arm} {phase} on GPU {gpu}: {cfg['tasks']}  (concurrency {cfg['concurrency']})")
    print("serve: " + " ".join(cmd))
    if kind == "completions":
        tok = AutoTokenizer.from_pretrained(checkpoint)
        for bench, first, body in first_requests(phase, tok, limit, arm):
            if body is None:
                print(f"{bench}: all samples present, nothing to send")
                continue
            p = body["prompt"]
            shown = dict(body, prompt=(p if len(p) <= 600 else p[:300] + f" [... {len(p) - 600} chars ...] " + p[-300:]))
            print(f"{bench}: first request = {first[0]} j={first[1]} (of {first[2]} pending); prompt {len(p)} chars, "
                  f"{len(tok(p, add_special_tokens=True)['input_ids'])} tokens, sha256 {hashlib.sha256(p.encode()).hexdigest()[:16]}")
            print("POST /v1/completions " + json.dumps(shown, ensure_ascii=False))
    elif kind == "tau2":
        api = tau2_api_config(cfg["api_lane"], require=False)
        print(f"user simulator / judge: {api['model']} at {api['base'] or '<QADX_API_BASE unset>'} with {api['key_name']}"
              f"{' (set)' if api['key'] else ' (unset)'}, throttle {api['rps']} req/s on port {serve.THROTTLE_PORT_BASE + int(gpu)}")
        for domain, out, c in tau2_commands(arm, P.specs(phase), cfg["concurrency"], gpu, api, limit):
            print(f"tau2 {domain}: " + ("complete, convert only" if c is None else " ".join(c)))
    else:
        k = int(P.specs(phase)[0].split(":")[1])
        gen, ev = bfcl_commands(checkpoint, k, max(1, cfg["concurrency"] // k), limit)
        for c in gen + ev:
            print("bfcl: " + " ".join(c))
        import tempfile

        with tempfile.TemporaryDirectory() as td:  # bfcl creates lock files under its project root on import
            env = bfcl_env(gpu, cfg["max_model_len"], P.specs(phase)[0].split(":")[2], td)
            r = subprocess.run([str(paths.BFCL_PY), str(paths.LIB / "bfcl_handler.py"), "--first-request", checkpoint],
                               capture_output=True, text=True, env=env, cwd=td)
        first = [l for l in r.stdout.splitlines() if l.startswith("bfcl first request")]
        print(first[0] if r.returncode == 0 and first else f"(could not render the first BFCL request: {r.stderr.strip()[-500:]})")
    print()


async def main():
    global RES, LOGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--phase", required=True, choices=P.ORDER)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--results", default=str(paths.RESULTS))
    ap.add_argument("--limit", type=int, default=None, help="first N items per benchmark (pilot runs only)")
    ap.add_argument("--dry-run", action="store_true", help="print the serve command and first requests, start nothing")
    ap.add_argument("--no-compile-cache", action="store_true")
    a = ap.parse_args()

    def on_term(*_):  # the scheduler stops a job with SIGTERM: unwind so that the server container is removed
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)
    RES, LOGS = Path(a.results), paths.logs_dir(a.results)
    LOGS.mkdir(parents=True, exist_ok=True)
    cc = not a.no_compile_cache
    if a.dry_run:
        dry_run(a.arm, a.checkpoint, a.phase, a.gpu, a.limit, cc)
        return 0
    from transformers import AutoTokenizer

    cfg, kind = P.PHASES[a.phase], P.kind(a.phase)
    (RES / a.arm).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.checkpoint)
    cmd = serve.serve_cmd(a.checkpoint, a.gpu, cfg["max_model_len"], serve.phase_server_args(cfg, kind), cc)
    log(f"starting server: {' '.join(cmd)}")
    proc = serve.start(cmd, a.gpu, LOGS / f"server_{a.arm}_{a.phase}.log", cc)
    try:
        serve.wait_healthy(proc, a.gpu, log=log)
        if kind == "tau2":
            await run_tau2(a.arm, P.specs(a.phase), cfg["concurrency"], a.gpu, cfg["api_lane"], a.limit)
        elif kind == "bfcl":
            await run_bfcl(a.arm, a.checkpoint, P.specs(a.phase)[0], cfg["concurrency"], a.gpu, cfg["max_model_len"], a.limit)
        else:
            for spec in P.specs(a.phase):
                bench, k, mt = spec.split(":")
                await generate(a.arm, bench, int(k), int(mt), cfg["concurrency"], tok, a.gpu, cfg["max_model_len"], a.limit)
    finally:
        serve.stop(proc, a.gpu, log=log)
    missing = (complete_tau2(a.arm, a.phase) if kind == "tau2" else complete_bfcl(a.arm, a.phase, a.limit)
               if kind == "bfcl" else complete_completions(a.arm, a.phase, a.limit))
    st = write_status(a.arm, a.phase, t0, missing, 0 if not missing else 3)
    log(f"phase {a.phase} {'complete' if st['complete'] else 'INCOMPLETE: ' + json.dumps(missing)} in {st['seconds'] / 60:.1f} min")
    return 0 if st["complete"] else 3


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
