"""Convert one replicate's bfcl result + score files (multi-turn) into per-entry rows (run by run_phase.run_bfcl).

  python bfcl_to_rows.py --result <root>/result_j0/qadx_qwen3-ct --score <root>/score_j0/qadx_qwen3-ct \
      --j 0 --records out/bfcl_mt.jsonl --grades out/bfcl_mt.grades.jsonl [--expect 200] [--append]

bfcl's score file holds a header {accuracy, correct_count, total_count} and one record per FAILED entry only, so
correctness is "id present in the result file and absent from the failure list". Both files are deduplicated by id
(result: last record wins; score: first record wins, conflicting error types abort) and the counts are cross-checked
against the header. bfcl evaluate itself aborts on duplicate result ids (length assert), so a duplicate here means the
files were edited or concatenated after evaluation.

Rows (one per entry): qid "<category>/<index>", j, correct 0/1, why (bfcl error_type or "pass"), plus diagnostics:
turns, steps_per_turn, hit_step_cap, output_tokens, max_step_output_tokens, truncated_steps, context_overflow,
max_prompt_tokens, parse_disagree (steps where the stock regex and hermes differ), hermes_errors, implicit_think_end,
latency_s, inference_error. --key-prefix names the handler's metadata keys (<prefix>_steps, <prefix>_error).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

CATS = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
STEP_CAP = 20  # bfcl MAXIMUM_STEP_LIMIT: force-quit once a turn has executed more than 20 steps (21 model calls)


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def flat(x):
    return [v for t in x for v in (t if isinstance(t, list) else [t])] if isinstance(x, list) else []


def rows_for_category(result_dir, score_dir, cat, j, expect, warn, prefix="qadx"):
    rp = result_dir / "multi_turn" / f"BFCL_v4_{cat}_result.json"
    sp = score_dir / "multi_turn" / f"BFCL_v4_{cat}_score.json"
    if not rp.exists():
        warn(f"{cat}: no result file {rp}")
        return []
    res_list = load_jsonl(rp)
    res = {}
    for r in res_list:
        res[r["id"]] = r
    if len(res) != len(res_list):
        warn(f"{cat}: {len(res_list) - len(res)} duplicate result records dropped (last kept)")
    score = load_jsonl(sp)
    header, fails = score[0], score[1:]
    failed = {}
    for f in fails:
        et = f.get("error", {}).get("error_type", "unknown")
        if f["id"] in failed and failed[f["id"]] != et:
            raise SystemExit(f"{cat}: conflicting failure records for {f['id']}: {failed[f['id']]} vs {et}")
        failed.setdefault(f["id"], et)
    if len(failed) != len(fails):
        warn(f"{cat}: {len(fails) - len(failed)} duplicate failure records dropped")
    if header["total_count"] != len(res) or header["total_count"] - header["correct_count"] != len(failed):
        raise SystemExit(f"{cat}: score header {header} inconsistent with {len(res)} results / {len(failed)} failures")
    if not set(failed) <= set(res):
        raise SystemExit(f"{cat}: failures for ids missing from the result file: {sorted(set(failed) - set(res))[:5]}")
    if expect and len(res) != expect:
        warn(f"{cat}: {len(res)} entries, expected {expect}")
    out = []
    for eid, r in sorted(res.items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1])):
        result = r["result"]
        inf_err = not isinstance(result, list)
        steps = r.get(f"{prefix}_steps", [])
        calls = [s for s in steps if "event" not in s]
        spt = [len(t) for t in result] if not inf_err else []
        out_tok = flat(r.get("output_token_count", []))
        why = failed.get(eid, "pass")
        out.append({
            "qid": f"{cat}/{eid.rsplit('_', 1)[1]}", "j": j, "bfcl_id": eid, "category": cat, "type": "bfcl_mt",
            "correct": int(eid not in failed), "why": why,
            "turns": len(spt), "steps_per_turn": spt,
            "hit_step_cap": why == "multi_turn:force_terminated" or any(n > STEP_CAP for n in spt),
            "output_tokens": int(sum(out_tok)), "completion_tokens": int(sum(out_tok)),
            "max_step_output_tokens": int(max(out_tok, default=0)),
            "truncated_steps": sum(s.get("finish_reason") == "length" for s in calls),
            "finish_reason": "length" if any(s.get("finish_reason") == "length" for s in calls) else "stop",
            "context_overflow": any(s.get("finish_reason") == "context_overflow" for s in calls),
            "max_prompt_tokens": max((s.get("prompt_tokens", 0) for s in calls), default=0),
            "model_calls": len(calls),
            "parse_disagree": sum(1 for s in calls if "strict" in s.get("parse", {}) and s["parse"]["strict"] != s["parse"]["hermes_n"]),
            "hermes_errors": sum(1 for s in calls if s.get("parse", {}).get("hermes") == "error"),
            "implicit_think_end": sum(1 for s in calls if s.get("parse", {}).get("implicit_end")),
            "latency_s": round(sum(flat(r.get("latency", []))), 2),
            "inference_error": (r.get(f"{prefix}_error") or "error") if inf_err else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", required=True, type=Path)
    ap.add_argument("--score", required=True, type=Path)
    ap.add_argument("--j", required=True, type=int)
    ap.add_argument("--records", required=True, type=Path)
    ap.add_argument("--grades", type=Path)
    ap.add_argument("--expect", type=int, default=200)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--key-prefix", default="qadx")
    a = ap.parse_args()
    warns = []
    rows = []
    for cat in CATS:
        rows += rows_for_category(a.result, a.score, cat, a.j, a.expect, warns.append, a.key_prefix)
    for w in warns:
        print("WARNING", w, file=sys.stderr)
    mode = "a" if a.append else "w"
    a.records.parent.mkdir(parents=True, exist_ok=True)
    with open(a.records, mode) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if a.grades:
        with open(a.grades, mode) as f:
            for r in rows:
                f.write(json.dumps({"qid": r["qid"], "j": r["j"], "correct": r["correct"], "why": r["why"]}) + "\n")
    by = Counter()
    for r in rows:
        by[(r["category"], "n")] += 1
        by[(r["category"], "ok")] += r["correct"]
    accs = [by[(c, "ok")] / by[(c, "n")] for c in CATS if by[(c, "n")]]
    print(f"j={a.j}: " + ", ".join(f"{c.removeprefix('multi_turn_')} {by[(c, 'ok')]}/{by[(c, 'n')]}" for c in CATS if by[(c, 'n')])
          + (f"; overall (mean of categories) {sum(accs) / len(accs):.4f}" if accs else ""))


if __name__ == "__main__":
    main()
