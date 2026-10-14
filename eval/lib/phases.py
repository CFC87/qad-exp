"""The evaluation phases of one Qwen3-8B checkpoint. Stdlib only.

A phase is one vLLM server lifetime on one GPU running one or more benchmarks. Task specs are bench:k:max_tokens, where
k is the number of samples per question (tau2: trials per task; BFCL: independent replicates of the whole set) and
max_tokens the output cap per request (tau2 / BFCL: per model turn / step).

    S1  competitions        AIME 2026, HMMT Feb 2026, HMMT Feb 2025, HMMT Nov 2025   k=8, cap 38,912
    S2  GPQA Diamond k=8 (primary); IFBench k=4, MMLU-Redux 1k k=1 (no-damage checks)
    S3  LiveCodeBench v6     k=4
    S4  LiveCodeBench Pro 2025Q2  k=8
    L1  LongBench Pro        k=1, static YaRN x4, 131,072 window
    M   MRCR 8-needle        k=8, static YaRN x4, 131,072 window
    T1  tau2 retail          4 trials (API lane 2)
    T2  tau2 airline + telecom   4 trials each (API lane 1)
    B   BFCL v4 multi-turn (base, miss_func, miss_param, long_context)   4 replicates

hours: typical wall time on one B200 for a QAD checkpoint (server start and generation, without grading); NVIDIA's NVFP4
release takes about 1.3x and BF16 about 1.8x as long (longer outputs). Used to order jobs (longest first) and to split
them across nodes.
"""

PHASES = {
    "S1": dict(tasks="aime26:8:38912,hmmt_feb26:8:38912,hmmt_feb25:8:38912,hmmt_nov25:8:38912",
               concurrency=160, max_model_len=40960, yarn=False, hours=1.0),
    "S2": dict(tasks="gpqa_diamond:8:32768,ifbench:4:32768,mmlu_redux_1k:1:32768",
               concurrency=160, max_model_len=40960, yarn=False, hours=0.6),
    "S3": dict(tasks="lcb_v6:4:32768", concurrency=160, max_model_len=40960, yarn=False, hours=0.5),
    "S4": dict(tasks="lcb_pro:8:32768", concurrency=160, max_model_len=40960, yarn=False, hours=1.2),
    "L1": dict(tasks="longbench_pro:1:32768", concurrency=64, max_model_len=131072, yarn=True, hours=2.4),
    "M": dict(tasks="mrcr8:8:32768", concurrency=64, max_model_len=131072, yarn=True, hours=0.5),
    "T1": dict(tasks="tau2_retail:4:32768", concurrency=48, max_model_len=40960, yarn=False, hours=0.9, api_lane=2),
    "T2": dict(tasks="tau2_airline:4:32768,tau2_telecom:4:32768", concurrency=48, max_model_len=40960, yarn=False,
               hours=1.9, api_lane=1),
    "B": dict(tasks="bfcl_mt:4:32768", concurrency=128, max_model_len=40960, yarn=False, hours=0.65),
}
ORDER = list(PHASES)


def specs(phase):
    return [t for t in PHASES[phase]["tasks"].split(",") if t]


def benches(phase):
    return [t.split(":")[0] for t in specs(phase)]


def kind(phase):
    """'tau2', 'bfcl' or 'completions'."""
    b = benches(phase)[0]
    return "tau2" if b.startswith("tau2_") else "bfcl" if b == "bfcl_mt" else "completions"
