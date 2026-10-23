"""mix60 configurations of train/launch/chain.py (no docker, no GPU): the multi-turn environment of base_mt.toml reaches
the controller as set (customer service: tool_v2 settings with the argument index as a repository file; function
calling inside the customer-service portion: round 4's replay entry; the episode deadline), round 4's configurations
are unchanged by it, export names and the stage-2 start, and the pre-flight stop on a missing card file.
Run from train/:  ../.venv-client/bin/python tests/test_chain_mix60.py"""
import argparse, json, shutil, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train" / "launch"))
import chain  # noqa: E402

M60, R4 = REPO / "experiments" / "mix60", REPO / "experiments" / "round4"
CS = ("SFT-Agentic-v2/interactive_agent", "Agentic-v1/interactive_agent", "RL-Agentic-Conversational-Tool-Use-Pivot-v1")
FC = ("SFT-Agentic-v2/tool_calling", "Agentic-v1/tool_calling", "RL-Agentic-Function-Calling-Pivot-v1")
CS_KW = {"tool_max_tokens": 8192, "tool_v2": True, "sim_ctx": 65536, "user_effort": "medium", "tool_effort": "medium",
         "arg_index": str(REPO / "data" / "subsets" / "arg_index_mix_v2.json")}
CARDS = ["tau2_cut_mt_25k_s20261001/scenario_cards_v2_final.parquet", "tau2_r3_t3_s20261014/customer_cards_fill_v1.parquet",
         "mix60_cards_s20261023/customer_cards_fill_n9.parquet", "mix60_cards_s20261023/customer_cards_fill_n3.parquet",
         "tau2_r3_t3_s20261014/customer_cards_v3.parquet", "mix60_cards_s20261023/customer_cards_v3_q.parquet",
         "mix60_cards_s20261023/customer_cards_v3_n9.parquet", "mix60_cards_s20261023/customer_cards_v3_n3.parquet",
         "mix60_cards_s20261023/customer_cards_alias.parquet"]


def ns(out, layout="1node"):
    return argparse.Namespace(out=str(out), layout=layout, peer=None, peer_ip=None)


def ctl(path, out):
    c = chain.Chain(path, ns(out))
    return c, c.controller_cfg(0, out, ["http://127.0.0.1:8021"], ["http://127.0.0.1:8290", "http://127.0.0.1:8291"], 0)


def main():
    out = Path(tempfile.mkdtemp(prefix="qadx_m60_"))
    # 1. the environment as base_mt.toml sets it, in every mix60 config with multi-turn data
    for f in ("q30_stage1", "n9", "n3", "smoke_q30", "smoke_n9", "smoke_n3"):
        c, cc = ctl(M60 / f"{f}.toml", out)
        mt = cc["multiturn"]
        assert sorted(mt) == sorted(CS + FC), (f, sorted(mt))
        for k in CS:
            m = mt[k]
            assert (m["env"], m["mode"], m["require_card"], m["log"]) == ("sim", "each", True, True), (f, k, m)
            assert m["sim_kw"] == CS_KW and m["deadline_s"] == 3600, (f, k, m)
        for k in FC:
            m = mt[k]
            assert (m["env"], m["mode"], m["log"]) == ("replay_tools", "each", True) and "require_card" not in m, (f, k, m)
            assert m["sim_kw"] == {"tool_max_tokens": 8192} and m["deadline_s"] == 3600, (f, k, m)
        for m in mt.values():
            assert m["sim"]["extra"] == {"reasoning_effort": "low"} and m["sim"]["timeout"] == 300, (f, m["sim"])
        assert cc["scenario_cards"] == [str(REPO / "data" / "subsets" / p) for p in CARDS], (f, cc["scenario_cards"])
        print(f"{f}: multi-turn environment ok")
    # stage 2 has no multi-turn data
    c2, cc2 = ctl(M60 / "q30_stage2.toml", out)
    assert not cc2["multiturn"] and not cc2["scenario_cards"], cc2["multiturn"]
    # 2. round 4 unchanged: its entries keep the tool-result cap only and the 1,200 s deadline
    for f in ("q30_stage1", "n9", "n3"):
        _, cc = ctl(R4 / f"{f}.toml", out)
        assert all(m["sim_kw"] == {"tool_max_tokens": 8192} and m["deadline_s"] == 1200 for m in cc["multiturn"].values()), (f, cc["multiturn"])
    print("round 4: multi-turn settings unchanged")
    # 3. schedules, export names, the stage-2 start
    want = {"q30_stage1": ([240], [40], ["mix60-q30-s240"]), "q30_stage2": ([60], [15], ["mix60-q30-s300"]),
            "n9": ([300], [50], ["mix60-n9-s300"]), "n3": ([300], [50], ["mix60-n3-s300"])}
    for f, (b, d, names) in want.items():
        c = chain.Chain(M60 / f"{f}.toml", ns(out))
        assert (c.budgets, c.decays) == (b, d) and c.kw(0) == [] and [c.export_name(x) for _, x in c.export_steps()] == names, (f, c.budgets, c.export_steps())
    c1 = chain.Chain(M60 / "q30_stage1.toml", ns(out))
    assert c2.init_from == c1.seg(0) == "mix60-q30-s240" and c2.yarn and c2.cfg["window"] == 131072
    s1, s2 = chain.Chain(M60 / "smoke_q30.toml", ns(out)), chain.Chain(M60 / "smoke_q30_long.toml", ns(out))
    assert s2.init_from == s1.seg(0) == "smoke-mix60-q30-s002"
    assert [s2.export_name(x) for _, x in s2.export_steps()] == ["smoke-mix60-q30-s004"], s2.export_steps()
    print("schedules, export names, stage-2 start ok")
    # 4. pre-flight: a missing card or argument-index file stops the chain before anything starts
    c = chain.Chain(M60 / "n9.toml", ns(out))
    (out / "train" / c.ptq).mkdir(parents=True, exist_ok=True)
    c.cfg["scenario_cards"] = c.cfg["scenario_cards"] + ["data/subsets/no_such_list/cards.parquet"]
    logged = []
    c.log = logged.append
    try:
        c.preflight(c.layout_spec())
        raise AssertionError("pre-flight passed with a missing card file")
    except SystemExit:
        pass
    assert logged and logged[-1].startswith("CHAIN_STOP: missing data files") and "no_such_list/cards.parquet" in logged[-1], logged
    print("pre-flight stops on a missing data file")
    shutil.rmtree(out, ignore_errors=True)
    print("mix60 chain test passed")


if __name__ == "__main__":
    main()
