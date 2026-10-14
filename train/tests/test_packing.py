"""onpolicy/packing.py: THD batch layout and online packing invariants (numpy only).

    python3 tests/test_packing.py
"""
import os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onpolicy'))
from packing import OnlinePacker, Pack, Sample, to_batch  # noqa: E402

rng = np.random.default_rng(0)


def rand_sample(pmax, rmax, tag):
    p = rng.integers(1, pmax); r = rng.integers(1, rmax)
    return Sample(list(rng.integers(0, 150000, p)), list(rng.integers(0, 150000, r)), {'tag': tag})


def check_batch(pack):
    b = to_batch(pack)
    W = pack.capacity
    tok, lab, m, pos = b['tokens'][0], b['labels'][0], b['loss_mask'][0], b['position_ids'][0]
    cu = b['cu_seqlens_q']
    assert tok.shape == (W,) and b['tokens'].shape == (1, W)
    assert cu[0] == 0 and cu[-1] == W and np.all(np.diff(cu) > 0)
    assert b['max_seqlen_q'] == int(np.diff(cu).max())
    assert m.sum() == sum(len(s.response) for s in pack.samples), 'every response token is supervised exactly once'
    o = 0
    for s in pack.samples:
        ids = list(s.prompt) + list(s.response)
        n, p = len(ids), len(s.prompt)
        assert list(tok[o:o + n]) == ids
        assert list(lab[o:o + n - 1]) == ids[1:]
        assert list(pos[o:o + n]) == list(range(n)), 'positions restart per sample'
        sup = np.flatnonzero(m[o:o + n])
        assert list(sup) == list(range(p - 1, n - 1)), 'loss exactly on the positions predicting response tokens'
        assert [lab[o + i] for i in sup] == list(s.response)
        o += n
    assert m[o:].sum() == 0, 'padding is never trained on'
    assert list(cu[1:1 + len(pack.samples)]) == list(np.cumsum([len(s) for s in pack.samples]))
    return b


# 1. layout of a hand-made pack
pk = Pack(64)
for s in (Sample([1, 2, 3], [4, 5]), Sample([6], [7, 8, 9]), Sample([10, 11, 12, 13], [14])):
    pk.add(s)
b = check_batch(pk)
assert list(b['cu_seqlens_q']) == [0, 5, 9, 14, 64]
assert list(b["loss_mask"][0][:14]) == [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0]  # S3: prompt 4 tokens -> position 9+3

# 2. online packing: every sample lands in exactly one pack or is dropped; nothing overflows
for trial in range(20):
    W, n = 4096, 8
    pk = OnlinePacker(n, capacity=W, ship_at=0.97)
    samples = [rand_sample(900, 900, i) for i in range(80)]
    ready = []
    for s in samples:
        ready += pk.add(s)
    ready += pk.finish()
    assert len(ready) == n and len({id(p) for p in ready}) == n
    placed = [s.meta['tag'] for p in ready for s in p.samples]
    dropped = [s.meta['tag'] for s in pk.dropped]
    assert sorted(placed + dropped) == list(range(80)) and len(set(placed)) == len(placed)
    for p in ready:
        assert p.used <= W
        check_batch(p)
    st = pk.stats()
    assert abs(st['fill'] - sum(p.used for p in ready) / (n * W)) < 1e-12
# 3. max_packs > n_packs: a rollout that fits no open pack gets a new pack; drops only beyond max_packs
for trial in range(20):
    W, n, mx = 4096, 8, 11
    pk = OnlinePacker(n, capacity=W, ship_at=0.97, max_packs=mx)
    samples = [rand_sample(900, 900, i) for i in range(int(rng.integers(30, 120)))]
    ready = []
    for s in samples:
        ready += pk.add(s)
    ready += pk.finish()
    assert len(ready) <= mx and all(p.samples for p in ready)
    placed = [s.meta['tag'] for p in ready for s in p.samples]
    dropped = [s.meta['tag'] for s in pk.dropped]
    assert sorted(placed + dropped) == list(range(len(samples))) and len(set(placed)) == len(placed)
    if dropped:
        assert len(ready) == mx, 'dropped only when every allowed pack exists'
    total = sum(len(s) for s in samples)
    if total <= 0.9 * n * W:
        assert not dropped, 'room to spare: nothing dropped'
    for p in ready:
        assert p.used <= W
        check_batch(p)
    st = pk.stats()
    assert st['packs'] == len(ready) and abs(st['fill'] - sum(p.used for p in ready) / (len(ready) * W)) < 1e-12

# 4. an empty pack: all padding, loss 0, attention in short segments
from packing import PAD_SEG  # noqa: E402
b = check_batch(Pack(131072))
assert b['loss_mask'].sum() == 0 and b['max_seqlen_q'] == PAD_SEG and len(b['cu_seqlens_q']) == 131072 // PAD_SEG + 1
print('packing tests passed')
