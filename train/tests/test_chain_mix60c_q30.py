"""mix60c-q30 configurations of train/launch/chain.py (no docker, no GPU): Qwen3-30B-A3B continued from mix60b's stage 2
(init_from its step-240 segment, a new optimizer and schedule), 130 steps with the last 25 decayed, exported as
mix60c-q30-s430; mix60b's multi-turn environment with mix60's card files followed by the continuation lists' card files;
the list and card files in data/MANIFEST.toml; everything else as mix60b's stage 2.
Run from train/:  ../.venv-client/bin/python tests/test_chain_mix60c_q30.py"""
import argparse, sys, tempfile, tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train" / "launch"))
import chain  # noqa: E402

E, M60B = REPO / "experiments" / "mix60c-q30", REPO / "experiments" / "mix60b"
C = "data/subsets/mix60c_cards_s20261027/"
NEW_CARDS = [C + f"customer_cards_{f}.parquet" for f in (
    "fill_n9", "fill_n3", "fill_q30", "v3_n9", "v3_n3", "v3_q8", "v3_q30", "alias", "alias_n3", "alias_q8", "alias_q30")]
V5 = {C + f"customer_cards_{f}.parquet" for f in ("fill_n9", "fill_n3", "v3_n9", "v3_n3", "alias", "alias_n3")}
LIST = "data/subsets/mix60c_q30_s20261027/subset.parquet"


def ns(out, layout="1node"):
    return argparse.Namespace(out=str(out), layout=layout, peer=None, peer_ip=None)


def ctl(path, out):
    c = chain.Chain(path, ns(out))
    return c, c.controller_cfg(0, out, ["http://127.0.0.1:8021"], ["http://127.0.0.1:8290", "http://127.0.0.1:8291"], 0)


def main():
    out = Path(tempfile.mkdtemp(prefix="qadx_m60cq30_"))
    # 1. environment: mix60b's multi-turn entries, mix60's card files then the continuation lists'
    _, ref = ctl(M60B / "q30_stage2.toml", out)
    for f in ("q30", "smoke_q30"):
        c, cc = ctl(E / f"{f}.toml", out)
        assert cc["multiturn"] == ref["multiturn"], f
        assert cc["scenario_cards"] == ref["scenario_cards"] + [str(REPO / p) for p in NEW_CARDS], (f, cc["scenario_cards"])
        assert c.cfg["subset"] == "mix60c_q30_s20261027", f
        print(f"{f}: multi-turn environment ok (mix60b's), cards = mix60's + {len(NEW_CARDS)} files of the continuation lists")
    # 2. everything else as mix60b's stage 2
    a, b = chain.Chain(M60B / "q30_stage2.toml", ns(out)).cfg, chain.Chain(E / "q30.toml", ns(out)).cfg
    differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
    assert differ == {"name", "subset", "budgets", "decays", "init_from", "export_prefix", "export_offset", "scenario_cards"}, sorted(differ)
    print("q30: settings as mix60b's stage 2 except", sorted(differ))
    # 3. schedule, start, export names
    s2, c = chain.Chain(M60B / "q30_stage2.toml", ns(out)), chain.Chain(E / "q30.toml", ns(out))
    assert (c.budgets, c.decays) == ([130], [25]) and (c.cfg["lr"], c.cfg["min_lr"]) == (s2.cfg["lr"], s2.cfg["min_lr"])
    assert c.init_from == s2.seg(0) == "mix60b-q30-s240", (c.init_from, s2.seg(0))
    assert [s2.export_name(x) for _, x in s2.export_steps()] == ["mix60b-q30-s300"], s2.export_steps()
    assert [c.export_name(x) for _, x in c.export_steps()] == ["mix60c-q30-s430"], c.export_steps()
    s = chain.Chain(E / "smoke_q30.toml", ns(out))
    assert s.init_from == "mix60b-q30-s240" and s.seg(0) == "smoke-mix60c-q30-s002", (s.init_from, s.seg(0))
    assert [s.export_name(x) for _, x in s.export_steps()] == ["smoke-mix60c-q30-s302"], s.export_steps()
    print("q30: schedule, start from mix60b-q30-s240, export names ok")
    # 4. the list and card files are release assets with a sha256
    man = tomllib.loads((REPO / "data" / "MANIFEST.toml").read_text())["assets"]
    paths = {v["path"]: v for v in man.values()}
    for p in [LIST] + NEW_CARDS:
        rel = "data-v5" if p in V5 else "data-v6"
        assert p in paths and paths[p].get("release") == rel and len(paths[p]["sha256"]) == 64, p
    print("list and card files in data/MANIFEST.toml (data-v5, data-v6)")
    print("mix60c-q30 chain test passed")


if __name__ == "__main__":
    main()
