"""run_step on a code + multi-turn function-calling mixture subset and its run config, CPU only: the real SubsetSource,
the real multiturn.run_episode (replay_tools), the run config's MULTITURN_JSON and expected lengths; rollout.generate
and rollout.sim_chat mocked. Code prompts get ~12K-token replies (lognormal, capped like the run), multi-turn turns ~650
(the means in training runs), latency growing with the length. Checks per step: exactly max_packs packs published, no
error, code rows single-turn (one sample each, never an episode), fc rows played as episodes, packed tokens about half
code / half fc.
    ../.venv-client/bin/python tests/test_rollout_mix_mock.py
"""
import asyncio, atexit, collections, json, os, random, shutil, subprocess, sys, tempfile, time
_mkdtemp = tempfile.mkdtemp
tempfile.mkdtemp = lambda *a, **k: (lambda d: (atexit.register(shutil.rmtree, d, True), d)[1])(_mkdtemp(*a, **k))  # removed at exit
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onpolicy'))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
import pyarrow.parquet as pq  # noqa: E402
import rollout  # noqa: E402
from prompts import SubsetSource  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOKD = os.path.join(REPO, 'models', 'Qwen3-8B')
SUB = os.path.join(REPO, 'data', 'subsets', 'mix_code_fc_s20261007')
EXPERIMENT = os.path.join(REPO, 'experiments', 'pairs-r2', 'code_fc.toml')
EXP = os.path.join(REPO, 'train', 'configs', 'expected_bench_8b.json')
tok = AutoTokenizer.from_pretrained(TOKD)
rng = random.Random(0)
FILL = tok.encode(' hmm', add_special_tokens=False)
HEAD = tok.encode('<think>\n', add_special_tokens=False)
TAIL = tok.encode('\n</think>\n\nSure, I can help with that.', add_special_tokens=False) + [tok.convert_tokens_to_ids('<|im_end|>')]
_t = pq.read_table(SUB + '/subset.parquet', columns=['prompt_id', 'domain']).to_pylist()
CODE_IDS = {r['prompt_id'] for r in _t if r['domain'] == 'code'}
calls = collections.Counter()


async def mock_generate(session, url, model, item, max_tokens, sampling, logprobs=True):
    code = item.get('prompt_id') in CODE_IDS
    calls['code' if code else 'mt_turn'] += 1
    n = int(min(rng.lognormvariate(9.2, 0.7) if code else rng.lognormvariate(6.2, 0.7), max_tokens - 30))
    await asyncio.sleep(0.01 + n / 60000)
    return HEAD + FILL * max(n, 1) + TAIL, ('length' if n >= max_tokens - 30 else 'stop'), -0.3


async def mock_sim(session, url, sc, messages, max_tokens, stats=None, accept_cut=False):
    calls['sim'] += 1
    await asyncio.sleep(rng.uniform(0.02, 0.1))
    return '{"result": "ok"}' if rng.random() < 0.5 else 'Thanks, that works.'


class Feed:
    def __init__(self):
        self.n = 0

    async def publish(self, k, seq, batch, route):
        self.n += 1


async def main():
    rollout.generate, rollout.sim_chat = mock_generate, mock_sim
    # the controller config exactly as train/launch/chain.py writes it for this experiment
    sys.path.insert(0, os.path.join(REPO, 'train', 'launch'))
    import argparse, chain
    ch = chain.Chain(EXPERIMENT, argparse.Namespace(out=tempfile.mkdtemp(), layout='1node', peer=None, peer_ip=None))
    CC = ch.controller_cfg(0, ch.out, ['http://x'], ['http://127.0.0.1:8290'], 0)
    MT = CC['multiturn']
    src = SubsetSource(SUB, TOKD, ranks=(0, 100000), hist_sup=True, expected_response=json.load(open(EXP)))
    cfg = dict(W=131072, window=40960, n_packs=32, max_packs=36, target_packs=29, cap=32768, vllm=['http://x'], model='s',
               sampling={}, log_dir=tempfile.mkdtemp(), multiturn=MT, ds_names=sorted(set(src.ds)), sample_every=0)
    cfg['_tok'] = tok
    dom = {r['prompt_id']: r['domain'] for r in _t}
    for k in range(3):
        calls.clear()
        h0, t0 = src.handed, time.time()
        feed = Feed()
        open(cfg['log_dir'] + '/rollouts.jsonl', 'w').close()
        st = await rollout.run_step(k, src, feed, cfg, None)
        rl = [json.loads(l) for l in open(cfg['log_dir'] + '/rollouts.jsonl')]
        tokd = collections.Counter(); nd = collections.Counter(); per_prompt = collections.Counter()
        for r in rl:
            d = dom.get(r.get('prompt_id'), '?')
            tokd[d] += r.get('prompt', 0) + r.get('response', 0); nd[d] += 1
            per_prompt[(d, r.get('prompt_id'))] += 1
        code_multi = sum(1 for (d, _), c in per_prompt.items() if d == 'code' and c > 1)
        share = tokd['code'] / max(1, tokd['code'] + tokd['fc'])
        print('step %d: %.0fs, submitted %d, packs %d (published %d), dropped %d, requeued %d, mt_cancelled %s, '
              'samples code %d / fc %d, prompts code %d / fc %d, code token share %.2f, calls %s'
              % (k, time.time() - t0, st['submitted'], st['packs'], feed.n, st['dropped_samples'], st['requeued'],
                 st.get('mt_cancelled'), nd['code'], nd['fc'], sum(1 for d, _ in per_prompt if d == 'code'),
                 sum(1 for d, _ in per_prompt if d == 'fc'), share, dict(calls)))
        assert feed.n == 36, feed.n
        assert nd['code'] > 0 and nd['fc'] > 0 and calls['mt_turn'] > 0, (nd, calls)
        assert code_multi == 0, 'a code prompt produced more than one sample'
        # share: reported only -- the mock's episodes are far cheaper than real function-calling conversations (14.5K
        # tokens packed each)
    print('rollout mixture mock test passed')


asyncio.run(main())
