"""Round-4 additions of train/launch/chain.py on fake checkpoint directories (docker and the NeMo image needed, no GPU):
keep_weights steps survive trim and prune; init_from lays out another chain's last weights-only checkpoint as this chain's
start (shards hard-linked, trackers at that iteration); export names with export_prefix / export_offset; the per-model
settings of the round-4 experiments in their dry-run configs. Run from train/:  ../.venv-client/bin/python tests/test_chain_round4.py"""
import argparse, json, os, shutil, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train" / "launch"))
sys.path.insert(0, str(REPO / "train" / "tests"))
import chain  # noqa: E402
from test_chain_ckpt_ops import ls, make_ckpt  # noqa: E402

R4 = REPO / "experiments" / "round4"


def ns(out, layout="1node"):
    return argparse.Namespace(out=str(out), layout=layout, peer=None, peer_ip=None)


def main():
    out = Path(tempfile.mkdtemp(prefix="qadx_r4_"))
    # 1. keep_weights: a 240-step segment keeps 200 and 220 besides 240 through trim and prune
    ch = chain.Chain(R4 / "n9.toml", ns(out))
    s = ch.seg(0)
    assert ch.kw(0) == [200, 220] and [ch.export_name(x) for _, x in ch.export_steps()] == ["r4-n9-s200", "r4-n9-s220", "r4-n9-s240"]
    p = out / "train" / s / "checkpoints"
    make_ckpt(p, [190, 200, 210, 220, 230], [230], 230, 230)
    ch.trim(0)
    assert ls(p) == ["iter_0000200", "iter_0000220", "iter_0000230", "non_persistent/iter_0000230"], ls(p)
    ch.root(f"cd /out/train/{s}/checkpoints && mkdir iter_0000240 && echo w > iter_0000240/__0_0.distcp && echo 240 > latest_checkpointed_iteration.txt")
    (out / "runs" / s).mkdir(parents=True)
    (out / "runs" / s / "CHAIN_DONE").write_text("{}")
    ch.prune()
    assert ls(p) == ["iter_0000200", "iter_0000220", "iter_0000240"], ls(p)
    print("keep_weights ok:", ls(p))
    # 2. init_from: Qwen3-30B-A3B stage 2 starts from stage 1's last weights-only checkpoint
    c1 = chain.Chain(R4 / "q30_stage1.toml", ns(out))
    c2 = chain.Chain(R4 / "q30_stage2.toml", ns(out))
    assert c2.init_from == c1.seg(0) == "r4-q30-s180", (c2.init_from, c1.seg(0))
    assert [c1.export_name(x) for _, x in c1.export_steps()] == ["r4-q30-s150", "r4-q30-s160", "r4-q30-s180"]
    assert [c2.export_name(x) for _, x in c2.export_steps()] == ["r4-q30-s210", "r4-q30-s240"]
    q = out / "train" / c1.seg(0) / "checkpoints"
    make_ckpt(q, [150, 160, 180], [], 180, 180)
    ch.root(f"echo t > /out/train/{c1.seg(0)}/checkpoints/iter_0000180/train_state.pt")
    o = ch.root(chain.init_cmd(f"/out/train/{c1.seg(0)}/checkpoints", f"/out/train/{c2.seg(0)}.init/checkpoints"))
    assert "INIT_CKPT_OK iter_0000180" in o, o
    d = out / "train" / f"{c2.seg(0)}.init" / "checkpoints"
    assert os.stat(d / "iter_0000180" / "__0_0.distcp").st_ino == os.stat(q / "iter_0000180" / "__0_0.distcp").st_ino
    assert (d / "latest_checkpointed_iteration.txt").read_text().strip() == "180"
    assert (d / "latest_train_state.pt").read_text().strip() == "t"
    o2 = ch.root(chain.init_cmd(f"/out/train/{c1.seg(0)}/checkpoints", f"/out/train/{c2.seg(0)}.init/checkpoints"))
    assert "exists" in o2, o2
    # stage 1 cleaned after stage 2 laid out its start: a restart of stage 2 still finds it
    ch.root(f"mv /out/train/{c1.seg(0)} /out/train/{c1.seg(0)}.away")
    o3 = ch.root(chain.init_cmd(f"/out/train/{c1.seg(0)}/checkpoints", f"/out/train/{c2.seg(0)}.init/checkpoints"))
    assert "INIT_CKPT_OK iter_0000180: exists" in o3, o3
    ch.root(f"mv /out/train/{c1.seg(0)}.away /out/train/{c1.seg(0)}")
    # an interrupted earlier attempt (no tracker yet) is laid out again
    ch.root(f"rm /out/train/{c2.seg(0)}.init/checkpoints/latest_checkpointed_iteration.txt "
            f"/out/train/{c2.seg(0)}.init/checkpoints/iter_0000180/__0_0.distcp")
    o4 = ch.root(chain.init_cmd(f"/out/train/{c1.seg(0)}/checkpoints", f"/out/train/{c2.seg(0)}.init/checkpoints"))
    assert "INIT_CKPT_OK iter_0000180 from" in o4 and (d / "iter_0000180" / "__0_0.distcp").exists(), o4
    # clean (exports present) removes both chains' checkpoints and stage 2's start
    for c in (c1, c2):
        for _, x in c.export_steps():
            (out / "exports" / c.export_name(x)).mkdir(parents=True, exist_ok=True)
            (out / "exports" / c.export_name(x) / "hf_quant_config.json").write_text("{}")
        c.clean()
    assert not (out / "train" / c1.seg(0)).exists() and not (out / "train" / f"{c2.seg(0)}.init").exists()
    # the stage-2 smoke starts from the stage-1 smoke
    s1, s2 = chain.Chain(R4 / "smoke_q30.toml", ns(out)), chain.Chain(R4 / "smoke_q30_long.toml", ns(out))
    assert s2.init_from == s1.seg(0) == "smoke-r4-q30-s002" and s2.yarn and s2.cfg["window"] == 131072
    assert [s2.export_name(x) for _, x in s2.export_steps()] == ["smoke-r4-q30-s004"], s2.export_steps()
    print("init_from ok:", o.strip().splitlines()[-1])
    # 3. per-model settings in the dry-run configs
    want = {"q30_stage1": dict(len="len_qwen3", pad=151643, parse=None, init=None, hf="/models/Qwen3-30B-A3B", pp=4),
            "q30_stage2": dict(len="len_qwen3", pad=151643, parse=None, init="/out/r4-q30-long-s060.init/checkpoints",
                               hf="/out/hfy/Qwen3-30B-A3B", pp=4),
            "n9": dict(len="len_n9", pad=None, parse=None, init=None, hf="/models/NVIDIA-Nemotron-Nano-9B-v2", pp=4),
            "n3": dict(len="len_n3", pad=11, parse="http://127.0.0.1:8295", init=None,
                       hf="/models/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", pp=4)}
    for f, w in want.items():
        c = chain.Chain(R4 / f"{f}.toml", ns(out))
        c.dryrun()
        dd = out / "dryrun" / c.seg(0)
        cc, tt = json.loads((dd / "controller.json").read_text()), json.loads((dd / "trainer.json").read_text())
        e, cmd = tt["env"], tt["cmd"]
        assert cc["length_col"] == w["len"] and cc.get("pad_id") == w["pad"] and cc.get("parse_url") == w["parse"], (f, cc)
        assert cc["wait_init"] == bool(w["init"]) and e.get("CL_PRETRAINED") == w["init"], (f, e)
        assert e.get("CL_MOE_PAD_ID") == (str(w["pad"]) if w["pad"] else None), (f, e)
        assert cmd[cmd.index("--student_hf_path") + 1] == w["hf"] == e["CL_WEIGHT_PUSH_HF"], (f, cmd)
        assert cmd[cmd.index("--pp_size") + 1] == str(w["pp"])
        assert ("CL_OPTIMIZER_OVERRIDES" in e) == f.startswith(("q30", "n3")), (f, e)
        assert bool(cc["multiturn"]) == (f != "q30_stage2"), f
        print(f"{f}: dry-run settings ok")
    ch.root("rm -rf /out/train /out/runs /out/dryrun /out/exports")
    shutil.rmtree(out, ignore_errors=True)
    print("round-4 chain test passed")


if __name__ == "__main__":
    main()
