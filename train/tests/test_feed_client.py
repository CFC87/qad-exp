"""onpolicy/feed_client.py against the real Feed server from rollout.py (CPU; needs torch).

    ../.venv-client/bin/python tests/test_feed_client.py
"""
import asyncio, os, sys, threading, time

import numpy as np
import torch
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onpolicy'))
from feed_client import FeedIterator, post_step_done  # noqa: E402
from packing import Pack, Sample, to_batch  # noqa: E402
from rollout import Feed  # noqa: E402

W, NP, PORT = 256, 3, 18711
rng = np.random.default_rng(0)


def pack(tag):
    p = Pack(W)
    while True:
        s = Sample(list(rng.integers(0, 1000, rng.integers(5, 40))), list(rng.integers(0, 1000, rng.integers(1, 30))), {'t': tag})
        if p.room() < len(s):
            return p
        p.add(s)


async def main():
    feed = Feed()
    app = web.Application(); app.add_routes([web.get('/pack/{step}/{seq}', feed.get_pack), web.post('/step_done/{step}', feed.step_done)])
    r = web.AppRunner(app); await r.setup(); await web.TCPSite(r, '127.0.0.1', PORT).start()
    ref = {}
    for k in (5,):  # step 5 published up front; step 6 pack 1 published late
        for i in range(NP):
            ref[(k, i)] = to_batch(pack((k, i))); await feed.publish(k, i, ref[(k, i)])
    ref[(6, 0)] = to_batch(pack((6, 0))); await feed.publish(6, 0, ref[(6, 0)])
    got, times = [], []

    def trainer():
        it = FeedIterator('http://127.0.0.1:%d' % PORT, NP, start_step=5, log=lambda m: None)
        for _ in range(NP + 2):
            got.append(next(it)); times.append(time.time())
        post_step_done('http://127.0.0.1:%d' % PORT, 5)

    th = threading.Thread(target=trainer); th.start()
    await asyncio.sleep(1.5)
    t_late = time.time()
    ref[(6, 1)] = to_batch(pack((6, 1))); await feed.publish(6, 1, ref[(6, 1)])
    while th.is_alive():
        await asyncio.sleep(0.1)
    order = [(5, 0), (5, 1), (5, 2), (6, 0), (6, 1)]
    for b, key in zip(got, order):
        e = ref[key]
        assert b['tokens'].dtype == torch.int64 and b['tokens'].shape == (1, W)
        assert b['loss_mask'].dtype == torch.float32 and b['cu_seqlens_q'].dtype == torch.int32
        assert b['max_seqlen_q'].dim() == 0 and int(b['max_seqlen_q']) == e['max_seqlen_q']
        assert b['attention_mask'] is None
        for k in ('tokens', 'labels', 'loss_mask', 'position_ids', 'cu_seqlens_q'):
            assert np.array_equal(b[k].numpy(), e[k]), k
    assert times[4] >= t_late, 'the late pack must be waited for'
    assert 5 in feed.done and not any(k[0] == 5 for k in feed.packs), 'step_done frees the step'
    await r.cleanup()
    print('feed client test passed')


asyncio.run(main())
