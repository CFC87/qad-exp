"""mix60b configurations of train/launch/chain.py (no docker, no GPU): Qwen3-30B-A3B in the reversed order (long context
first from the NVFP4 start, then the mixed list from its step 60), the environment of mix60b/base_mt.toml (mix60's with
code_v3, log_v3 and close_v4 on the customer-service entries), export names, the stage-2 start; mix60 and round 4 unchanged.
Run from train/:  ../.venv-client/bin/python tests/test_chain_mix60b.py"""
import argparse, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train" / "launch"))
import chain  # noqa: E402

M60B, M60, R4 = REPO / "experiments" / "mix60b", REPO / "experiments" / "mix60", REPO / "experiments" / "round4"
CS = ("SFT-Agentic-v2/interactive_agent", "Agentic-v1/interactive_agent", "RL-Agentic-Conversational-Tool-Use-Pivot-v1")
FC = ("SFT-Agentic-v2/tool_calling", "Agentic-v1/tool_calling", "RL-Agentic-Function-Calling-Pivot-v1")
CS_KW = {"tool_max_tokens": 8192, "tool_v2": True, "sim_ctx": 65536, "user_effort": "medium", "tool_effort": "medium",
         "arg_index": str(REPO / "data" / "subsets" / "arg_index_mix_v2.json")}


def ns(out, layout="1node"):
    return argparse.Namespace(out=str(out), layout=layout, peer=None, peer_ip=None)


def ctl(path, out):
    c = chain.Chain(path, ns(out))
    return c, c.controller_cfg(0, out, ["http://127.0.0.1:8021"], ["http://127.0.0.1:8290", "http://127.0.0.1:8291"], 0)


def main():
    out = Path(tempfile.mkdtemp(prefix="qadx_m60b_"))
    # 1. stage 2 (mixed list) and its smoke: mix60's environment plus the three customer-service switches
    _, ref = ctl(M60 / "q30_stage1.toml", out)
    for f in ("q30_stage2", "smoke_q30"):
        c, cc = ctl(M60B / f"{f}.toml", out)
        mt = cc["multiturn"]
        assert sorted(mt) == sorted(CS + FC), (f, sorted(mt))
        for k in CS:
            assert mt[k]["sim_kw"] == {**CS_KW, "code_v3": True, "log_v3": True, "close_v4": True}, (f, k, mt[k]["sim_kw"])
            assert {x: y for x, y in mt[k].items() if x != "sim_kw"} == {x: y for x, y in ref["multiturn"][k].items() if x != "sim_kw"}, (f, k)
        for k in FC:
            assert mt[k] == ref["multiturn"][k], (f, k, mt[k])
        assert cc["scenario_cards"] == ref["scenario_cards"], f
        print(f"{f}: multi-turn environment ok (mix60's plus code_v3, log_v3, close_v4 on customer service)")
    # stage 1 (long context) and its smoke: no multi-turn data
    for f in ("q30_stage1", "smoke_q30_long"):
        _, cc = ctl(M60B / f"{f}.toml", out)
        assert not cc["multiturn"] and not cc["scenario_cards"], (f, cc["multiturn"])
    # 2. mix60 and round 4 unchanged
    for f in ("q30_stage1", "n9", "n3"):
        _, cc = ctl(M60 / f"{f}.toml", out)
        assert all("close_v4" not in m["sim_kw"] and "code_v3" not in m["sim_kw"] for m in cc["multiturn"].values()), f
        _, cc = ctl(R4 / f"{f}.toml", out)
        assert all(m["sim_kw"] == {"tool_max_tokens": 8192} for m in cc["multiturn"].values()), f
    print("mix60 and round 4: environment unchanged")
    # 3. schedules, export names, starts: stage 1 from the NVFP4 start, stage 2 from stage 1's step 60
    want = {"q30_stage1": ([60], [15], ["mix60b-q30-s060"], 131072, True), "q30_stage2": ([240], [40], ["mix60b-q30-s300"], 40960, False)}
    for f, (b, d, names, win, yarn) in want.items():
        c = chain.Chain(M60B / f"{f}.toml", ns(out))
        assert (c.budgets, c.decays) == (b, d) and [c.export_name(x) for _, x in c.export_steps()] == names, (f, c.budgets, c.export_steps())
        assert c.cfg["window"] == win and bool(c.yarn) == yarn, (f, c.cfg["window"], c.yarn)
    c1, c2 = chain.Chain(M60B / "q30_stage1.toml", ns(out)), chain.Chain(M60B / "q30_stage2.toml", ns(out))
    assert not c1.init_from and c2.init_from == c1.seg(0) == "mix60b-q30-long-s060", (c1.init_from, c2.init_from, c1.seg(0))
    s1, s2 = chain.Chain(M60B / "smoke_q30_long.toml", ns(out)), chain.Chain(M60B / "smoke_q30.toml", ns(out))
    assert s2.init_from == s1.seg(0) == "smoke-mix60b-q30-long-s002"
    assert [s2.export_name(x) for _, x in s2.export_steps()] == ["smoke-mix60b-q30-s004"], s2.export_steps()
    print("schedules, export names, starts ok")
    print("mix60b chain test passed")


if __name__ == "__main__":
    main()
