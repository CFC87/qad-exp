"""Convert a tau2 results.json into per-sample files for one arm and domain.

<arm>/tau2_<domain>.jsonl           one row per simulation (qid "<domain>:<task_id>", j = trial), diagnostics
<arm>/tau2_<domain>.grades.jsonl    {qid, j, correct, why}: correct = tau2's reward (0/1)
<arm>/tau2_<domain>.left_out.jsonl  simulations left out of scoring (to be re-run)

Failed simulations (termination infrastructure_error): tau2 cannot tell who failed. A reply with neither text nor a tool
call (thinking cut off by the token budget: the reasoning parser leaves content empty) or a context overflow on the
local server is the model's failure and scores 0, as the leaderboard counts it. An error on the API side (user
simulator or judge: rate limit, connection, server error) says nothing about the arm: the simulation is left out and
listed. Unclassified errors are left out and listed too, for a look by hand.
usage: tau2_convert.py RESULTS_JSON ARM_DIR DOMAIN
"""

import json
import sys
from pathlib import Path

# tau2's message check raises "AssistantMessage must have either content or tool_calls" for an empty agent reply
# (tau2/data_model/message.py); the same error for a UserMessage is the user simulator's (API side).
MODEL_ERR = ("assistantmessage must have either content or tool_calls",)
# A prompt longer than the served window (40,960 for Qwen3, 131,072 for the Nemotrons) is a limit of the setting, the same
# for every arm: scored 0 like a model failure but counted separately.
OVERFLOW_ERR = ("maximum context length", "prompt is too long", "contextwindowexceeded")
API_ERR = ("usermessage must have either content or tool_calls", "ratelimit", "rate limit", "apiconnection",
           "connection", "timeout", "timed out", "internalserver", "service unavailable", "502", "503", "504", "529",
           "overloaded", "authentication")


def classify(sim):
    """'overflow' / 'model' / 'api' / 'unknown' for an infrastructure error."""
    err = json.dumps(sim.get("info") or {}).lower()
    if any(k in err for k in OVERFLOW_ERR):
        return "overflow"
    if any(k in err for k in MODEL_ERR):
        return "model"
    if any(k in err for k in API_ERR):
        return "api"
    return "unknown"


def finish(m):
    raw = m.get("raw_data") or {}
    ch = (raw.get("choices") or [{}])[0] if isinstance(raw, dict) else {}
    return ch.get("finish_reason")


def convert(results_json, arm_dir, domain):
    d = json.load(open(results_json))
    rows, grades, left_out = [], [], []
    for s in d["simulations"]:
        qid, j, term = f"{domain}:{s['task_id']}", s["trial"], s["termination_reason"]
        msgs = s.get("messages") or []
        agent = [m for m in msgs if m["role"] == "assistant"]
        usage = [m.get("usage") or {} for m in agent]
        err = classify(s) if term == "infrastructure_error" else None
        if err in ("api", "unknown"):
            left_out.append(dict(qid=qid, j=j, error_class=err, info=json.dumps(s.get("info"))[:500]))
            continue
        ri = s.get("reward_info") or {}
        reward = float(ri.get("reward") or 0.0) if err is None else 0.0
        trunc = any(finish(m) == "length" for m in agent)
        rows.append(dict(
            qid=qid, j=j, type="tau2", seed=s.get("seed"), termination_reason=term, error_class=err,
            finish_reason="length" if trunc else "stop",
            completion_tokens=sum(u.get("completion_tokens") or 0 for u in usage),
            max_prompt_tokens=max((u.get("prompt_tokens") or 0 for u in usage), default=0),
            agent_calls=len(agent), user_calls=sum(1 for m in msgs if m["role"] == "user"),
            tool_calls=sum(len(m.get("tool_calls") or []) for m in agent),
            leaked_markup=any(("<tool_call>" in (m.get("content") or "")) or ("<think>" in (m.get("content") or "")) for m in agent),
            reward_breakdown=ri.get("reward_breakdown"), user_cost=s.get("user_cost"), duration=s.get("duration")))
        grades.append(dict(qid=qid, j=j, correct=reward, why=err and f"infra_{err}" or term))
    arm_dir = Path(arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    for name, recs in ((f"tau2_{domain}.jsonl", rows), (f"tau2_{domain}.grades.jsonl", grades),
                       (f"tau2_{domain}.left_out.jsonl", left_out)):
        with open(arm_dir / name, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    n = len(rows)
    print(f"tau2 {domain}: {n} runs kept (mean reward {sum(g['correct'] for g in grades) / max(n, 1):.3f}), "
          f"{sum(r['error_class'] == 'model' for r in rows)} model-caused failures and "
          f"{sum(r['error_class'] == 'overflow' for r in rows)} context overflows scored 0, {len(left_out)} left out "
          f"(api/unknown errors), {sum(r['finish_reason'] == 'length' for r in rows)} with a truncated agent turn, "
          f"{sum(r['leaked_markup'] for r in rows)} with tool/think markup leaked into content")
    return n, len(left_out)


if __name__ == "__main__":
    convert(*sys.argv[1:4])
