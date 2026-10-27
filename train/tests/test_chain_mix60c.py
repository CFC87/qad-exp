"""mix60c configurations of train/launch/chain.py (no docker, no GPU): the two Nemotrons continued from mix60's n9 / n3
runs (init_from their step-300 segments, a new optimizer and schedule), 90 steps with the last 15 decayed, exported as
mix60c-<m>-s390; mix60b's multi-turn environment with mix60's card files followed by the two lists' new ones; the lists
and card files in data/MANIFEST.toml; everything else as mix60's runs.
Run from train/:  ../.venv-client/bin/python tests/test_chain_mix60c.py"""
import argparse, sys, tempfile, tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train" / "launch"))
import chain  # noqa: E402

M60C, M60B, M60 = REPO / "experiments" / "mix60c", REPO / "experiments" / "mix60b", REPO / "experiments" / "mix60"
NEW_CARDS = [f"data/subsets/mix60c_cards_s20261027/{f}.parquet" for f in (
    "customer_cards_fill_n9", "customer_cards_fill_n3", "customer_cards_v3_n9", "customer_cards_v3_n3", "customer_cards_alias",
    "customer_cards_alias_n3")]
LISTS = [f"data/subsets/mix60c_{m}_s20261027/subset.parquet" for m in ("n9", "n3")]


def ns(out, layout="1node"):
    return argparse.Namespace(out=str(out), layout=layout, peer=None, peer_ip=None)


def ctl(path, out):
    c = chain.Chain(path, ns(out))
    return c, c.controller_cfg(0, out, ["http://127.0.0.1:8021"], ["http://127.0.0.1:8290", "http://127.0.0.1:8291"], 0)


def main():
    out = Path(tempfile.mkdtemp(prefix="qadx_m60c_"))
    # 1. environment: mix60b's multi-turn entries, mix60's card files then this list's
    _, ref = ctl(M60B / "q30_stage2.toml", out)
    for m in ("n9", "n3"):
        _, ref60 = ctl(M60 / f"{m}.toml", out)
        for f in (m, f"smoke_{m}"):
            c, cc = ctl(M60C / f"{f}.toml", out)
            assert cc["multiturn"] == ref["multiturn"], f
            assert cc["scenario_cards"] == ref60["scenario_cards"] + [str(REPO / p) for p in NEW_CARDS], (f, cc["scenario_cards"])
            assert c.cfg["subset"] == f"mix60c_{m}_s20261027", f
            print(f"{f}: multi-turn environment ok (mix60b's), cards = mix60's + {len(NEW_CARDS)} files of the two lists")
        # 2. everything else as mix60's run of the model
        a, b = chain.Chain(M60 / f"{m}.toml", ns(out)).cfg, chain.Chain(M60C / f"{m}.toml", ns(out)).cfg
        differ = {k for k in set(a) | set(b) if a.get(k) != b.get(k)}
        assert differ == {"name", "subset", "budgets", "decays", "init_from", "export_prefix", "export_offset", "multiturn",
                          "scenario_cards"}, (m, sorted(differ))
        print(f"{m}: settings as mix60's run except", sorted(differ))
        # 3. schedule, start, export names
        m60, c = chain.Chain(M60 / f"{m}.toml", ns(out)), chain.Chain(M60C / f"{m}.toml", ns(out))
        assert (c.budgets, c.decays) == ([90], [15]) and (c.cfg["lr"], c.cfg["min_lr"]) == (m60.cfg["lr"], m60.cfg["min_lr"]), m
        assert c.init_from == m60.seg(0) == f"mix60-{m}-s300", (c.init_from, m60.seg(0))
        assert [c.export_name(x) for _, x in c.export_steps()] == [f"mix60c-{m}-s390"], c.export_steps()
        s = chain.Chain(M60C / f"smoke_{m}.toml", ns(out))
        assert s.init_from == f"mix60-{m}-s300" and s.seg(0) == f"smoke-mix60c-{m}-s002", (s.init_from, s.seg(0))
        assert [s.export_name(x) for _, x in s.export_steps()] == [f"smoke-mix60c-{m}-s302"], s.export_steps()
        print(f"{m}: schedule, start from mix60-{m}-s300, export names ok")
    # 4. the list and card files are release assets with a sha256
    man = tomllib.loads((REPO / "data" / "MANIFEST.toml").read_text())["assets"]
    paths = {v["path"]: v for v in man.values()}
    for p in LISTS + NEW_CARDS:
        assert p in paths and paths[p].get("release") == "data-v5" and len(paths[p]["sha256"]) == 64, p
    print("list and card files in data/MANIFEST.toml (data-v5)")
    print("mix60c chain test passed")


if __name__ == "__main__":
    main()
