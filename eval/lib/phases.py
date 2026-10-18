"""The evaluation phases of one Qwen3-8B checkpoint. Stdlib only.

A phase is one vLLM server lifetime on one GPU running one or more benchmarks. Task specs are bench:k:max_tokens, where
k is the number of samples per question (tau2: trials per task; BFCL: independent replicates of the whole set) and
max_tokens the output cap per request (tau2 / BFCL: per model turn / step).

    S1  competitions        AIME 2026, HMMT Feb 2026, HMMT Feb 2025, HMMT Nov 2025   k=16, cap 38,912
    S2  GPQA Diamond k=16 (primary); IFBench k=4, MMLU-Redux 1k k=1 (no-damage checks)
    S3  LiveCodeBench v6     k=8
    S4  LiveCodeBench Pro 2025Q2  k=8
    L1  LongBench Pro        k=1, static YaRN x4, 131,072 window
    M   MRCR 8-needle        k=8, static YaRN x4, 131,072 window
    T1  tau2 retail          4 trials (API lane 2)
    T2  tau2 airline + telecom   4 trials each (API lane 1)
    B   BFCL v4 multi-turn (base, miss_func, miss_param, long_context)   4 replicates

hours: typical wall time on one B200 for a QAD checkpoint (server start and generation, without grading); NVIDIA's NVFP4
release takes about 1.3x and BF16 about 1.8x as long (longer outputs). Used to order jobs (longest first) and to split
them across nodes.

Evaluation spec v2: the four competitions and GPQA Diamond take 16 samples per question, LiveCodeBench v6 takes 8
(spec v1: 8, 8 and 4). Everything else is unchanged. Results made under v1 are never extended to the new k: run_phase.py
moves such a benchmark's files to <results>/<arm>/superseded/ and generates it afresh (V1_K below), and run.py does not
count a phase as complete unless its status file was written under the current spec.
"""

PHASES = {
    "S1": dict(tasks="aime26:16:38912,hmmt_feb26:16:38912,hmmt_feb25:16:38912,hmmt_nov25:16:38912",
               concurrency=160, max_model_len=40960, yarn=False, hours=1.9),
    "S2": dict(tasks="gpqa_diamond:16:32768,ifbench:4:32768,mmlu_redux_1k:1:32768",
               concurrency=160, max_model_len=40960, yarn=False, hours=1.0),
    "S3": dict(tasks="lcb_v6:8:32768", concurrency=160, max_model_len=40960, yarn=False, hours=0.8),
    "S4": dict(tasks="lcb_pro:8:32768", concurrency=160, max_model_len=40960, yarn=False, hours=1.2),
    "L1": dict(tasks="longbench_pro:1:32768", concurrency=64, max_model_len=131072, yarn=True, hours=2.4),
    "M": dict(tasks="mrcr8:8:32768", concurrency=64, max_model_len=131072, yarn=True, hours=0.5),
    "T1": dict(tasks="tau2_retail:4:32768", concurrency=48, max_model_len=40960, yarn=False, hours=0.9, api_lane=2),
    "T2": dict(tasks="tau2_airline:4:32768,tau2_telecom:4:32768", concurrency=48, max_model_len=40960, yarn=False,
               hours=1.9, api_lane=1),
    "B": dict(tasks="bfcl_mt:4:32768", concurrency=128, max_model_len=40960, yarn=False, hours=0.65),
}
ORDER = list(PHASES)
SPEC_VERSION = 2
# spec v1 task strings of the phases v2 changed (a status file without a spec field was written under v1) and the
# v1 sample counts of the benchmarks whose k changed (a results file without a .spec stamp was made under v1)
V1_TASKS = {"S1": "aime26:8:38912,hmmt_feb26:8:38912,hmmt_feb25:8:38912,hmmt_nov25:8:38912",
            "S2": "gpqa_diamond:8:32768,ifbench:4:32768,mmlu_redux_1k:1:32768", "S3": "lcb_v6:4:32768"}
V1_K = {"aime26": 8, "hmmt_feb26": 8, "hmmt_feb25": 8, "hmmt_nov25": 8, "gpqa_diamond": 8, "lcb_v6": 4}


def specs(phase):
    return [t for t in PHASES[phase]["tasks"].split(",") if t]


def benches(phase):
    return [t.split(":")[0] for t in specs(phase)]


def kind(phase):
    """'tau2', 'bfcl' or 'completions'."""
    b = benches(phase)[0]
    return "tau2" if b.startswith("tau2_") else "bfcl" if b == "bfcl_mt" else "completions"
