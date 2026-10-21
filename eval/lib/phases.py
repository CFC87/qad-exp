"""The evaluation phases of one checkpoint, per model (Qwen3-8B below; the other models at the end). Stdlib only.

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



# ------------------------------------------------------------------------------------------------- the other models
# Qwen3-30B-A3B, Nemotron Nano 9B v2 and Nemotron 3 Nano 30B-A3B: every phase as their reference PTQ / BF16 results were
# evaluated (eval/reference/baselines.json), so that recovery is read against them. Their samples per question and caps
# differ from Qwen3-8B's spec v2 on purpose: they are those references' settings. Only the five report categories are
# run (no IFBench, no MMLU-Redux).
#
#   Qwen3-30B-A3B (window 40,960; L with static YaRN x4 at 131,072)
#     S1  competitions k=8, cap 38,912     G  GPQA Diamond k=8          S3  LiveCodeBench v6 k=4
#     S4  LiveCodeBench Pro k=8            L  LongBench Pro k=1 + MRCR k=8
#     T   tau2 airline + retail + telecom, 4 trials each, one server    B  BFCL multi-turn, 4 replicates
#   Nemotron Nano 9B v2 (window 131,072, no YaRN): S1 competitions k=8, cap 32,768; G GPQA k=8; S3 LiveCodeBench v6 k=4
#     (no LiveCodeBench Pro: this model does not follow its C++ instruction); L, T, B as above
#   Nemotron 3 Nano 30B-A3B: S1, G, S3, S4 with cap 131,072 in a 262,144 window (NVIDIA's evaluation budget); L, T, B as
#     above in a 131,072 window
# hours: typical wall time on one B200 (server start and generation); tau2 also depends on the API. api_lane "alt": the
# one tau2 phase of an arm takes API lane 1 or 2 by arm (run.py), so two arms' tau2 phases can run at once with two keys.

COMPS_ORDER = ("aime26", "hmmt_feb26", "hmmt_feb25", "hmmt_nov25")


def _comps(k, cap):
    return ",".join(f"{b}:{k}:{cap}" for b in COMPS_ORDER)


_LONG = "longbench_pro:1:32768,mrcr8:8:32768"
_TAU2 = "tau2_airline:4:32768,tau2_retail:4:32768,tau2_telecom:4:32768"
MODEL_PHASES = {
    "qwen3-8b": PHASES,
    "qwen3-30b-a3b": {
        "S1": dict(tasks=_comps(8, 38912), concurrency=160, max_model_len=40960, yarn=False, hours=1.0),
        "G": dict(tasks="gpqa_diamond:8:32768", concurrency=160, max_model_len=40960, yarn=False, hours=0.4),
        "S3": dict(tasks="lcb_v6:4:32768", concurrency=160, max_model_len=40960, yarn=False, hours=0.5),
        "S4": dict(tasks="lcb_pro:8:32768", concurrency=160, max_model_len=40960, yarn=False, hours=1.1),
        "L": dict(tasks=_LONG, concurrency=64, max_model_len=131072, yarn=True, hours=3.5),
        "T": dict(tasks=_TAU2, concurrency=48, max_model_len=40960, yarn=False, hours=2.9, api_lane="alt"),
        "B": dict(tasks="bfcl_mt:4:32768", concurrency=128, max_model_len=40960, yarn=False, hours=1.5),
    },
    "nemotron-nano-9b-v2": {
        "S1": dict(tasks=_comps(8, 32768), concurrency=160, max_model_len=131072, yarn=False, hours=0.9),
        "G": dict(tasks="gpqa_diamond:8:32768", concurrency=160, max_model_len=131072, yarn=False, hours=0.8),
        "S3": dict(tasks="lcb_v6:4:32768", concurrency=160, max_model_len=131072, yarn=False, hours=0.5),
        "L": dict(tasks=_LONG, concurrency=64, max_model_len=131072, yarn=False, hours=1.5),
        "T": dict(tasks=_TAU2, concurrency=48, max_model_len=131072, yarn=False, hours=2.9, api_lane="alt"),
        "B": dict(tasks="bfcl_mt:4:32768", concurrency=128, max_model_len=131072, yarn=False, hours=1.5),
    },
    "nemotron-3-nano-30b-a3b": {
        "S1": dict(tasks=_comps(8, 131072), concurrency=160, max_model_len=262144, yarn=False, hours=1.2),
        "G": dict(tasks="gpqa_diamond:8:131072", concurrency=160, max_model_len=262144, yarn=False, hours=0.8),
        "S3": dict(tasks="lcb_v6:4:131072", concurrency=160, max_model_len=262144, yarn=False, hours=0.6),
        "S4": dict(tasks="lcb_pro:8:131072", concurrency=160, max_model_len=262144, yarn=False, hours=1.3),
        "L": dict(tasks=_LONG, concurrency=64, max_model_len=131072, yarn=False, hours=0.8),
        "T": dict(tasks=_TAU2, concurrency=48, max_model_len=131072, yarn=False, hours=2.9, api_lane="alt"),
        "B": dict(tasks="bfcl_mt:4:32768", concurrency=128, max_model_len=131072, yarn=False, hours=1.5),
    },
}
# Evaluation spec per model: Qwen3-8B's is v2 (above); the other models' protocols are at their first version.
MODEL_SPEC_VERSION = {"qwen3-8b": SPEC_VERSION, "qwen3-30b-a3b": 1, "nemotron-nano-9b-v2": 1, "nemotron-3-nano-30b-a3b": 1}
DEFAULT_MODEL = "qwen3-8b"


def table(model=DEFAULT_MODEL):
    return MODEL_PHASES[model]


def order(model=DEFAULT_MODEL):
    return list(MODEL_PHASES[model])


def spec_version(model=DEFAULT_MODEL):
    return MODEL_SPEC_VERSION[model]


def v1_tasks(phase, model=DEFAULT_MODEL):
    """The tasks a status file without a tasks field was written for (spec v1 of Qwen3-8B; no such files otherwise)."""
    return V1_TASKS.get(phase, PHASES[phase]["tasks"]) if model == DEFAULT_MODEL else None


def v1_k(bench, k, model=DEFAULT_MODEL):
    """The k a results file without a .spec stamp was made with (spec v1 of Qwen3-8B; otherwise unstamped = k)."""
    return V1_K.get(bench, k) if model == DEFAULT_MODEL else k


def specs(phase, model=DEFAULT_MODEL):
    return [t for t in MODEL_PHASES[model][phase]["tasks"].split(",") if t]


def benches(phase, model=DEFAULT_MODEL):
    return [t.split(":")[0] for t in specs(phase, model)]


def kind(phase, model=DEFAULT_MODEL):
    """'tau2', 'bfcl' or 'completions'."""
    b = benches(phase, model)[0]
    return "tau2" if b.startswith("tau2_") else "bfcl" if b == "bfcl_mt" else "completions"
