"""mix60b-q8 configurations of train/launch/chain.py (no docker, no GPU): Qwen3-8B, three arms in the mix60b order (long
context 60 steps from the NVFP4 start, then 240 mixed steps from its step 60): sft (the SFT-trace lists, forward KL), mkl
(kd_reverse 0.5) and rkl (kd_reverse 1.0); schedules, export names, starts, the loss weight given to the trainer, the lists,
the multi-turn environment of stage 2 (mix60b's).  Run from train/:  ../.venv-client/bin/python tests/test_chain_mix60b_q8.py"""
import argparse, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train" / "launch"))
import chain  # noqa: E402

E = REPO / "experiments" / "mix60b-q8"
LISTS = {"sft": ("mix60sft_s2_long_q8_s20261026", "mix60sft_s1_q8_s20261026"),
         "mkl": ("mix60_s2_long_q8_s20261023", "mix60_s1_q8_s20261023"), "rkl": ("mix60_s2_long_q8_s20261023", "mix60_s1_q8_s20261023")}
KD = {"sft": None, "mkl": "0.5", "rkl": "1.0"}


def ns(out):
    return argparse.Namespace(out=str(out), layout="1node", peer=None, peer_ip=None)


def main():
    out = Path(tempfile.mkdtemp(prefix="qadx_m60bq8_"))
    _, ref = None, chain.Chain(REPO / "experiments" / "mix60b" / "q30_stage2.toml", ns(out))
    ref_mt = ref.controller_cfg(0, out, ["http://127.0.0.1:8021"], ["http://127.0.0.1:8290"], 0)["multiturn"]
    for arm in ("sft", "mkl", "rkl"):
        for st, (b, d, win, yarn) in (("stage1", ([60], [15], 131072, True)), ("stage2", ([240], [40], 40960, False))):
            for f in (f"{arm}_{st}", f"smoke_{arm}{'_long' if st == 'stage1' else ''}"):
                c = chain.Chain(E / f"{f}.toml", ns(out))
                env, cmd, pp, *_ = c.trainer_cmd(0, c.layout_spec(), ["http://127.0.0.1:8021"])
                cc = c.controller_cfg(0, out, ["http://127.0.0.1:8021"], ["http://127.0.0.1:8290"], 0)
                assert c.model == "Qwen3-8B" and c.cfg["checkpoint"] == "Qwen3-8B-NVFP4", f
                assert c.cfg["subset"] == LISTS[arm][0 if st == "stage1" else 1], (f, c.cfg["subset"])
                assert env.get("CL_KD_REVERSE") == KD[arm], (f, env.get("CL_KD_REVERSE"))
                assert c.cfg["window"] == win and bool(c.yarn) == yarn, f
                if st == "stage1":
                    assert not cc["multiturn"], f
                else:
                    assert cc["multiturn"] == ref_mt, f
                if not f.startswith("smoke"):
                    assert (c.budgets, c.decays) == (b, d), (f, c.budgets)
                    names = [c.export_name(x) for _, x in c.export_steps()]
                    assert names == [f"mix60b-q8-{arm}-s{'060' if st == 'stage1' else '300'}"], (f, names)
                    assert c.init_from == (None if st == "stage1" else f"mix60b-q8-{arm}-long-s060"), (f, c.init_from)
                else:
                    assert c.init_from == (None if st == "stage1" else f"smoke-mix60b-q8-{arm}-long-s002"), (f, c.init_from)
        print(f"{arm}: lists, schedules, starts, loss weight, environment ok")
    print("mix60b-q8 chain test passed")


if __name__ == "__main__":
    main()
