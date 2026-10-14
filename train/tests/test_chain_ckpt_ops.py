"""Checkpoint file operations of train/launch/chain.py on fake checkpoint directories (docker and the NeMo image needed,
no GPU): the WSD branch (shards hard-linked, trackers pointing at the branch step), trim while a segment trains, and
prune after a segment. Run from train/:  ../.venv-client/bin/python tests/test_chain_ckpt_ops.py"""
import argparse, os, shutil, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "train" / "launch"))
import chain  # noqa: E402

EXP = REPO / "experiments" / "pairs-r2" / "smoke.toml"   # budgets 6, 8; decays 2, 2; checkpoints every 2 steps


def make_ckpt(ckdir, iters_weights, iters_full, latest_w, latest_f):
    for it in iters_weights:
        d = ckdir / f"iter_{it:07d}"
        d.mkdir(parents=True)
        (d / "__0_0.distcp").write_bytes(b"w" * 64)
        (d / ".metadata").write_text("m")
    for it in iters_full:
        d = ckdir / "non_persistent" / f"iter_{it:07d}"
        (d / "modelopt_state").mkdir(parents=True)
        for r in range(4):
            (d / f"__{r}_0.distcp").write_bytes(bytes([r]) * 128)
        (d / "modelopt_state" / "__0_0.distcp").write_bytes(b"q" * 16)
        (d / ".metadata").write_text("m")
        (d / "train_state.pt").write_bytes(b"s" + str(it).encode())
    (ckdir / "latest_checkpointed_iteration.txt").write_text(str(latest_w))
    (ckdir / "non_persistent").mkdir(exist_ok=True)
    (ckdir / "non_persistent" / "latest_checkpointed_iteration.txt").write_text(str(latest_f))


def ls(root):
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_dir() and p.name.startswith("iter_"))


def main():
    out = Path(tempfile.mkdtemp(prefix="qadx_ckpt_ops_"))
    ch = chain.Chain(EXP, argparse.Namespace(out=str(out), layout="1node", peer=None, peer_ip=None))
    s0, s1 = ch.seg(0), ch.seg(1)
    p = out / "train" / s0 / "checkpoints"
    make_ckpt(p, [2, 4, 6], [4, 6], 6, 6)
    (out / "runs" / s0).mkdir(parents=True)
    (out / "runs" / s0 / "CHAIN_DONE").write_text("{}")
    # prune the finished segment 0: keep weights 6 (budget) and 4 (keep step), full 4; drop weights 2 and full 6
    ch.prune()
    got = ls(p)
    assert got == ["iter_0000004", "iter_0000006", "non_persistent/iter_0000004"], got
    print("prune ok:", got)
    # branch segment 1 from the kept full checkpoint at 4
    o = ch.root(chain.branch_cmd(f"/out/train/{s0}/checkpoints", f"/out/train/{s1}/checkpoints", 4))
    assert "BRANCH_CKPT_OK" in o, o
    b = out / "train" / s1 / "checkpoints"
    bi, pi = b / "non_persistent" / "iter_0000004", p / "non_persistent" / "iter_0000004"
    for r in range(4):
        assert os.stat(bi / f"__{r}_0.distcp").st_ino == os.stat(pi / f"__{r}_0.distcp").st_ino, "shard not hard-linked"
    assert os.stat(bi / "modelopt_state" / "__0_0.distcp").st_ino != os.stat(pi / "modelopt_state" / "__0_0.distcp").st_ino
    assert (b / "non_persistent" / "latest_checkpointed_iteration.txt").read_text().strip() == "4"
    assert (b / "non_persistent" / "latest_train_state.pt").read_bytes() == b"s4"
    print("branch ok:", o.strip().splitlines()[-1])
    # a branch over an existing directory is refused
    r = subprocess.run(["docker", "run", "--rm", "-v", f"{out}:/out", "--entrypoint", "bash", chain.IMG_NEMO, "-c",
                        chain.branch_cmd(f"/out/train/{s0}/checkpoints", f"/out/train/{s1}/checkpoints", 4)], capture_output=True, text=True)
    assert r.returncode != 0 and "BRANCH_CKPT_FAIL" in r.stdout, r.stdout
    print("branch over an existing directory refused ok")
    # segment 1 trains: weights 6 and 8 written (latest 8), its own full checkpoint at 6 (> branch point 4)
    # (written as root, like the trainer container writes them)
    ch.root(f"cd /out/train/{s1}/checkpoints && for it in 0000006 0000008; do mkdir iter_$it && echo w > iter_$it/__0_0.distcp; done "
            f"&& echo 8 > latest_checkpointed_iteration.txt && mkdir non_persistent/iter_0000006 "
            f"&& echo 6 > non_persistent/latest_checkpointed_iteration.txt")
    ch.trim(1)
    assert ls(b) == ["iter_0000008", "non_persistent/iter_0000004", "non_persistent/iter_0000006"], ls(b)
    assert not (p / "non_persistent" / "iter_0000004").exists(), "the parent's kept full checkpoint should be gone"
    assert (bi / "__0_0.distcp").exists(), "the branch's hard links must survive the parent's removal"
    print("trim ok:", ls(b), "| parent:", ls(p))
    ch.root(f"rm -rf /out/train /out/runs")
    shutil.rmtree(out)
    print("checkpoint operations test passed")


if __name__ == "__main__":
    main()
