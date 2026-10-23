"""WSD branches and resumes on the controller side (onpolicy/ledger.py, rollout.py), CPU only.

A. ledger.py on hand-made logs: the latest attempt of each step after a resume, the ranks a step gave back, the cut
   for a branch (training steps < start, validations e < start // val_every), refusal to overwrite.
B. The real controller against a mock vLLM and a mock trainer (as test_rollout_subset.py), the expected response set
   far too low so that every step over-submits and gives rollouts back:
   1. parent: steps 0-4, then a crash and a resume at its step 3 (the log holds steps 3 and 4 twice), on to step 6;
   2. branches at step 5 (5 % VAL_EVERY != 0) and at step 4 (a validation step): the ledger is cut, a new controller
      starts at that step, the mock trainer resumes there and runs to step 8.
   Checks: the branch draws from the parent's position (its first prompt at the start step is the parent's, the
   prompts the parent gave back at the step before are drawn again, at the start step or, when they are more than that
   step draws, later; none is used up untrained), no prompt is trained twice along the lineage
   (parent before the start + branch after it), every trained rank is a training rank, the validations continue
   without gaps or repeats, the cumulative totals continue.
    ../.venv-client/bin/python tests/test_branch_ledger.py
"""
import asyncio, io, json, os, random, shutil, sys, tempfile

import aiohttp
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from aiohttp import web

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onpolicy'))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
import ledger  # noqa: E402
import rollout  # noqa: E402
from prompts import SubsetSource  # noqa: E402

W, NP, CAP, VAL_EVERY, N_TRAIN, N_VAL = 8192, 2, 1024, 2, 260, 10
rng = random.Random(0)


def part_a():
    d = tempfile.mkdtemp()
    P = lambda step, rank, route='pack': dict(step=step, route=route, rank=rank)
    # attempt 0: steps 0..4 (val e=0 after step 2's pack block, e=1 after step 4's), crash; attempt 1 resumes at 3
    roll = ([P(0, 0), P(0, 1)] + [P(1, 2), P(1, 3)] + [P(2, 4), P(2, 5)] + [P(0, 90, 'val')] + [P(3, 6), P(3, 7)] +
            [P(4, 8)] + [P(1, 90, 'val')] +
            [P(3, 6), P(3, 9)] + [P(4, 7), P(4, 10)] + [P(1, 91, 'val')] + [P(5, 11)])
    steps = ([dict(step=s, route='pack', requeued_ranks=[], cumulative=dict(prompts=10 * (s + 1))) for s in range(5)] +
             [dict(step=0, route='val'), dict(step=1, route='val')] +
             [dict(step=3, route='pack', requeued_ranks=[9], cumulative=dict(prompts=31)),
              dict(step=4, route='pack', requeued_ranks=[], cumulative=dict(prompts=41)),
              dict(step=1, route='val'), dict(step=5, route='pack', requeued_ranks=[], cumulative=dict(prompts=51))])
    steps[2]['requeued_ranks'] = [5]  # step 2 gave rank 5 back; attempt 0's step 3 did not draw it again
    for f, rows in (('rollouts.jsonl', roll), ('steps.jsonl', steps)):
        with open(os.path.join(d, f), 'w') as o:
            o.writelines(json.dumps(r) + '\n' for r in rows)
    la = ledger.latest_attempt(ledger.read(os.path.join(d, 'rollouts.jsonl')))
    got = [(r['route'], r['step'], r['rank']) for r in la]
    want = [('pack', 0, 0), ('pack', 0, 1), ('pack', 1, 2), ('pack', 1, 3), ('pack', 2, 4), ('pack', 2, 5),
            ('val', 0, 90), ('pack', 3, 6), ('pack', 3, 9), ('pack', 4, 7), ('pack', 4, 10), ('val', 1, 91), ('pack', 5, 11)]
    assert got == want, got
    # used before 5: 0-4 and 5 (given back at step 2) out, 6 and 9 at step 3 (9 given back), 7 and 10 at step 4
    assert ledger.used_ranks(d, 5) == {0, 1, 2, 3, 4, 6, 7, 10}, ledger.used_ranks(d, 5)
    assert ledger.used_ranks(d, 3) == {0, 1, 2, 3, 4}
    assert ledger.carried_totals(d, 5) == dict(prompts=41) and ledger.carried_totals(d, 4) == dict(prompts=31)
    c = d + '/child'
    info = ledger.branch(d, c, 4, VAL_EVERY)
    cr = ledger.read(c + '/rollouts.jsonl')
    assert [(r['route'], r['step'], r['rank']) for r in cr] == want[:9] + [('val', 1, 91)], cr  # steps < 4, val e < 2 (e=1: attempt 1's)
    cs = ledger.read(c + '/steps.jsonl')
    assert [(r['route'], r['step']) for r in cs] == [('pack', 0), ('pack', 1), ('pack', 2), ('val', 0), ('pack', 3), ('val', 1)], cs
    assert info['used_prompts'] == 6 and info['cumulative'] == dict(prompts=31), info
    try:
        ledger.branch(d, c, 4, VAL_EVERY)
        raise AssertionError('overwrote an existing branch log')
    except SystemExit as e:
        assert 'not overwritten' in str(e)
    try:
        ledger.branch(d, d + '/child2', 7, VAL_EVERY)
        raise AssertionError('branched beyond the parent')
    except SystemExit as e:
        assert 'has not reached' in str(e)
    c3 = ledger.branch(d, d + '/child3', 3, VAL_EVERY)  # 3 % 2 != 0: validations e < 1
    assert [(r['route'], r['step']) for r in ledger.read(d + '/child3/rollouts.jsonl')] == \
        [('pack', 0), ('pack', 0), ('pack', 1), ('pack', 1), ('pack', 2), ('pack', 2), ('val', 0)]
    assert c3['used_prompts'] == 5
    shutil.rmtree(d)
    print('A. ledger on hand-made logs: ok')


async def mock_completions(request):
    body = await request.json()
    mt = body['max_tokens']
    n = min(int(rng.lognormvariate(6.5, 0.8)), mt)
    fin = 'length' if n == mt else 'stop'
    ids = [rng.randrange(1000, 150000) for _ in range(max(n - 2, 0))]
    if fin == 'stop':
        cut = rng.randrange(0, max(len(ids), 1))
        ids = ids[:cut] + [rollout.THINK_END] + ids[cut:] + [151645]
    ids = ids[:mt]
    await asyncio.sleep(len(ids) / 500000)
    return web.json_response({'choices': [{'token_ids': ids, 'finish_reason': fin, 'text': '',
                                           'logprobs': {'token_logprobs': [-rng.random() for _ in ids]}}]})


async def trainer(port, start, end, seen, crash_after=None):
    """Megatron as the trainer (qad_distill.py) drives it: resumed at iteration `start` (its checkpoint), it evaluates
    after its steps k-1 with k % VAL_EVERY == 0, except at k == start (done before that checkpoint)."""
    async with aiohttp.ClientSession() as s:
        if start > 0:  # resumed: the trainer pushes the loaded weights and reports step start - 1 done
            async with s.post('http://127.0.0.1:%d/step_done/%d' % (port, start - 1)) as r:
                assert (await r.json())['ok']
        for k in range(start, end):
            if k > start and k % VAL_EVERY == 0:
                e = k // VAL_EVERY - 1
                for i in range(NP):
                    async with s.get('http://127.0.0.1:%d/val/%d/%d' % (port, e, i)) as r:
                        np.load(io.BytesIO(await r.read()))
                seen['val'].append(e)
            for i in range(NP):
                async with s.get('http://127.0.0.1:%d/pack/%d/%d' % (port, k, i)) as r:
                    np.load(io.BytesIO(await r.read()))
            async with s.get('http://127.0.0.1:%d/stats/pack/%d' % (port, k)) as r:
                seen['stats'].append(await r.json())
            if crash_after is not None and k == crash_after:
                return  # dies before reporting the step: the controller already logged it
            async with s.post('http://127.0.0.1:%d/step_done/%d' % (port, k)) as r:
                assert (await r.json())['ok']


def make_src(tmp, tok):
    src = SubsetSource(tmp, tok, ranks=(0, N_TRAIN))
    for d in set(src.ds):  # far too low: every step over-submits and gives rollouts back
        src.expected[d] = 60.0
    return src


async def controller(tmp, log_dir, tok, vi, start, steps, port, crash_after=None):
    """rollout.main()'s resume path (ledger.used_ranks -> mark_used) and main_async against the mock."""
    src = make_src(tmp, tok)
    if start:
        src.mark_used(ledger.used_ranks(log_dir, start))
    cfg = dict(vllm=['http://127.0.0.1:18731'], model='m', W=W, n_packs=NP, cap=CAP, steps=steps, start_step=start,
               port=port, log_dir=log_dir, sampling=rollout.QWEN3_SAMPLING, topup_max_prompt=4096, val_every=VAL_EVERY,
               val_items=vi, ds_names=sorted(set(src.ds) | {x['dataset'] for x in vi}), sample_every=0, n_samples=0)
    seen = dict(val=[], stats=[])
    ctl = asyncio.ensure_future(rollout.main_async(cfg, src))
    await asyncio.sleep(0.5)
    await asyncio.wait_for(trainer(port, start, steps if crash_after is None else crash_after + 1, seen, crash_after), 600)
    await asyncio.sleep(0.5)
    ctl.cancel()
    try:
        await ctl
    except (asyncio.CancelledError, Exception):
        pass
    await asyncio.sleep(0.3)
    return seen


def trained(log_dir, lo, hi):
    """rank -> step for the prompts trained in steps lo..hi-1 of this log's latest attempts (generated minus given back);
    each step's set must equal its drawn_ranks."""
    roll = [r for r in ledger.latest_attempt(ledger.read(log_dir + '/rollouts.jsonl')) if r.get('route', 'pack') == 'pack']
    st = {r['step']: r for r in ledger.latest_attempt(ledger.read(log_dir + '/steps.jsonl')) if r.get('route', 'pack') == 'pack'}
    out = {}
    for s in range(lo, hi):
        back = set(st[s].get('requeued_ranks') or ())
        got = {r['rank'] for r in roll if r['step'] == s} - back
        assert got == set(st[s]['drawn_ranks']), 'step %d: rollouts minus given back != drawn_ranks' % s
        for rk in got:
            assert rk not in out, 'rank %d trained twice (steps %d and %d)' % (rk, out[rk], s)
            out[rk] = s
    return out


async def part_b():
    tmp = tempfile.mkdtemp()
    t = pq.read_table(os.path.join(REPO, 'data', 'subsets', 'mix_reas_code_s20261008', 'subset.parquet'))
    t = t.filter(pc.less(t.column('len_qwen3'), 1200)).slice(0, N_TRAIN + N_VAL)
    t = t.set_column(t.column_names.index('rank'), 'rank', pa.array(np.arange(N_TRAIN + N_VAL), pa.int64()))
    pq.write_table(t, tmp + '/subset.parquet')
    mock = web.Application(); mock.add_routes([web.post('/v1/completions', mock_completions)])
    mr = web.AppRunner(mock); await mr.setup(); await web.TCPSite(mr, '127.0.0.1', 18731).start()
    tok = os.path.join(REPO, 'models', 'Qwen3-8B')
    vs = SubsetSource(tmp, tok, ranks=(N_TRAIN, N_TRAIN + N_VAL))
    vi = [vs.next() for _ in range(N_VAL)]
    parent = tmp + '/parent'; os.makedirs(parent)
    s1 = await controller(tmp, parent, tok, vi, 0, 7, 18732, crash_after=3)  # steps 0-3, dies before reporting 3
    s2 = await controller(tmp, parent, tok, vi, 3, 7, 18733)                 # resume at 3: step 3 written twice in a row
    st = ledger.latest_attempt(ledger.read(parent + '/steps.jsonl'))
    reqd = sum(len(r.get('requeued_ranks') or ()) for r in st if r.get('route') == 'pack')
    assert reqd >= 5, 'the mock must make steps give rollouts back (got %d)' % reqd
    raw = [r['step'] for r in ledger.read(parent + '/steps.jsonl') if r.get('route') == 'pack']
    assert raw.count(3) == 2 and raw[raw.index(3) + 1] == 3, raw  # back to back: only the run ids tell them apart
    runs = [r['run'] for r in ledger.read(parent + '/steps.jsonl') if r.get('route') == 'pack']
    assert len(set(runs)) == 2 and all(r for r in runs), runs
    la = [r for r in ledger.latest_attempt(ledger.read(parent + '/steps.jsonl')) if r.get('route') == 'pack']
    assert [r['step'] for r in la] == list(range(7)) and la[3]['run'] == runs[-1], 'step 3 of the resumed start'
    assert all(r.get('drawn_ranks') is not None for r in la)
    tp = trained(parent, 0, 7)
    assert set(tp) <= set(range(N_TRAIN)), 'only training ranks'
    # the controller's own resume: what the resumed start drew at step 3 begins where the lineage before 3 ends
    first3 = min(r['rank'] for r in ledger.latest_attempt(ledger.read(parent + '/rollouts.jsonl')) if r.get('route') == 'pack' and r['step'] == 3)
    assert first3 == min(set(range(N_TRAIN)) - ledger.used_ranks(parent, 3)), 'resume did not continue at the lineage position'
    print('B1. parent: %d steps (step 3 twice, back to back), %d prompts trained, %d given back' % (7, len(tp), reqd))
    for B, port in ((5, 18734), (4, 18736)):
        child = tmp + '/child%d' % B
        info = ledger.branch(parent, child, B, VAL_EVERY)
        sc = await controller(tmp, child, tok, vi, B, 9, port)
        tc_before, tc_after = trained(child, 0, B), trained(child, B, 9)
        assert tc_before == {r: s for r, s in tp.items() if s < B}, 'the branch keeps the parent lineage before %d' % B
        assert not set(tc_before) & set(tc_after), 'a prompt trained twice across the branch point'
        assert set(tc_after) <= set(range(N_TRAIN))
        pb = [r for r in ledger.latest_attempt(ledger.read(parent + '/rollouts.jsonl')) if r.get('route') == 'pack' and r['step'] == B]
        cb = [r for r in ledger.read(child + '/rollouts.jsonl') if r.get('route') == 'pack' and r['step'] == B]
        back = set(next(r for r in ledger.latest_attempt(ledger.read(parent + '/steps.jsonl'))
                        if r.get('route') == 'pack' and r['step'] == B - 1).get('requeued_ranks') or ())
        assert min(r['rank'] for r in cb) == min(r['rank'] for r in pb), 'first prompt at step %d differs' % B
        # a prompt given back comes again in a later step (rollout.py), in rank order before any new prompt: at step B
        # unless the backlog of given-back prompts is larger than step B draws; it is never used up without training
        miss = back - {r['rank'] for r in cb}
        later = {r['rank'] for r in ledger.read(child + '/rollouts.jsonl') if r.get('route') == 'pack' and r['step'] > B}
        used_end = ledger.used_ranks(child, 9)
        assert all(m in later or m not in used_end for m in miss), \
            'prompts given back at step %d neither drawn later nor still unused: %s' % (B - 1, sorted(m for m in miss if m not in later and m in used_end))
        assert not miss or min(miss) > min(r['rank'] for r in cb), 'a given-back prompt was skipped for a higher rank at step %d' % B
        cs = [r for r in ledger.read(child + '/steps.jsonl')]
        ve = [r['step'] for r in cs if r.get('route') == 'val']
        assert sorted(ve) == list(range(len(ve))) and len(set(ve)) == len(ve), 'validations %s' % ve
        assert sc['val'] == [e for e in range(B // VAL_EVERY, 9 // VAL_EVERY) if (e + 1) * VAL_EVERY > B], sc['val']
        cB = next(r for r in cs if r.get('route') == 'pack' and r['step'] == B)
        assert cB['cumulative']['prompts'] == info['cumulative']['prompts'] + cB['finished'], 'cumulative totals'
        ps = sorted(r['step'] for r in cs if r.get('route') == 'pack')
        assert ps == list(range(9)), ps
        print('B2. branch at %d: first prompt %d = parent, %d given-back prompts of step %d drawn again, %d trained '
              'after the branch, none twice; validations %s; totals continue' % (
                  B, min(r['rank'] for r in cb), len(back), B - 1, len(tc_after), ve))
    await mr.cleanup()
    shutil.rmtree(tmp)


if __name__ == '__main__':
    part_a()
    asyncio.run(part_b())
    print('branch ledger test passed')
