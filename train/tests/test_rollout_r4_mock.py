"""run_step on the round-4 lists of Qwen3-30B-A3B (both stages), Nemotron Nano 9B v2 and Nemotron 3 Nano with their
controller configs exactly as train/launch/chain.py writes them, CPU only: the real SubsetSource (each model's tokenizer
and length column), the real multiturn.run_episode (customer service end to end with scenario cards, function calling
inside it replayed), and for Nemotron 3 Nano the real parse service (vLLM 0.29's parser, train/onpolicy/parse_service.py,
started here in the serving image); rollout.generate and rollout.sim_chat mocked. The mock writes each model's own
format: reasoning, then a reply or a tool call to a tool of the prompt (Qwen3 <tool_call>{json}, Nano v2 <TOOLCALL>[json],
Nemotron 3 Nano <tool_call><function=...>). Checks per step: exactly max_packs packs published, pad id of the model in the
pack padding, the function-calling rows (subset *_b1) one sample each, the customer-service rows played as episodes, no
generation error; Nemotron 3 Nano: turns parsed by the service.
The mix60 configs (experiments/mix60) run the same way, given as mix60/<config>; their customer-service episodes also
go through the checked environment (sim_kw tool_v2).
Needs docker and the models of the round-4 experiments (setup/setup.sh optional-models).
    ../.venv-client/bin/python tests/test_rollout_r4_mock.py [q30_stage1 q30_stage2 n9 n3 mix60/q30_stage1 mix60/n9 ...]
"""
import argparse, asyncio, atexit, collections, json, os, random, re, shutil, subprocess, sys, tempfile, time
_mkdtemp = tempfile.mkdtemp
tempfile.mkdtemp = lambda *a, **k: (lambda d: (atexit.register(shutil.rmtree, d, True), d)[1])(_mkdtemp(*a, **k))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onpolicy'))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'train', 'launch'))
import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import rollout  # noqa: E402
import multiturn  # noqa: E402
import chain  # noqa: E402
from prompts import SubsetSource  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

TOOLNAME = re.compile(r'"name":\s*"([A-Za-z0-9_.\-]+)"|<name>([A-Za-z0-9_.\-]+)</name>')  # JSON tools (Qwen3, Nano v2) / XML (Nemotron 3 Nano)
calls = collections.Counter()


def fmt_call(fam, name):
    if fam == 'n3':
        return '<tool_call>\n<function=%s>\n</function>\n</tool_call>\n' % name
    if fam == 'n9':
        return '<TOOLCALL>[{"name": "%s", "arguments": {}}]</TOOLCALL>' % name
    return '<tool_call>\n{"name": "%s", "arguments": {}}\n</tool_call>' % name


def make_mock(tok, fam, eos, rng, single):
    fill = tok.encode(' hmm', add_special_tokens=False)
    head = tok.encode('<think>\n', add_special_tokens=False) if fam == 'qwen3' else []  # Nemotron prompts end inside <think>

    async def mock_generate(session, url, model, item, max_tokens, sampling, logprobs=True):
        one = item.get('prompt_id') in single
        calls['single' if one else 'mt_turn'] += 1
        n = int(min(rng.lognormvariate(7.5 if one else 6.2, 0.7), max_tokens - 60))
        tail = '\n</think>\n'
        if not one and rng.random() < 0.4:  # a call to one of the conversation's tools
            names = [x or y for x, y in TOOLNAME.findall(tok.decode(item['ids'][:20000]))] if item.get('ids') else []
            tail += fmt_call(fam, names[0]) if names else 'Sure, I can help with that.'
            calls['tool_call'] += bool(names)
        else:
            tail += 'Sure, I can help with that.'
        await asyncio.sleep(0.005 + n / 80000)
        return head + fill * max(n, 1) + tok.encode(tail, add_special_tokens=False) + [eos], 'stop', -0.3
    return mock_generate


async def mock_sim(session, url, sc, messages, max_tokens, stats=None, accept_cut=False, effort=None):
    calls['sim'] += 1
    await asyncio.sleep(0.01)
    return '{"result": "ok"}' if random.random() < 0.5 else 'Thanks, that works.'


class Feed:
    def __init__(self, pad):
        self.n, self.pad, self.pad_ok = 0, pad, True

    async def publish(self, k, seq, batch, route):
        self.n += 1
        t, lm = np.asarray(batch['tokens']).reshape(-1), np.asarray(batch['loss_mask']).reshape(-1)
        if lm.sum() == 0:  # an empty pack: all padding
            self.pad_ok &= bool((t == self.pad).all())


def parse_service_up(checkpoint_dir):
    subprocess.run(['docker', 'rm', '-f', 'qadx-parse-test'], capture_output=True)
    subprocess.run(['docker', 'run', '-d', '--name', 'qadx-parse-test', '--net=host', '--entrypoint', 'python3',
                    '-v', f'{checkpoint_dir}:/tok:ro', '-v', f'{REPO}/train/onpolicy:/ops:ro', chain.IMG_VLLM,
                    '/ops/parse_service.py', '--port', '8297', '--parser', 'nemotron_v3'], check=True, capture_output=True)
    atexit.register(subprocess.run, ['docker', 'rm', '-f', 'qadx-parse-test'], capture_output=True)
    for _ in range(90):
        if chain.http_ok('http://127.0.0.1:8297/health'):
            return 'http://127.0.0.1:8297'
        time.sleep(2)
    raise SystemExit('parse service did not come up')


async def one(exp):
    d, exp = exp.split('/') if '/' in exp else ('round4', exp)  # e.g. mix60/n9: another experiment's config
    ch = chain.Chain(os.path.join(REPO, 'experiments', d, exp + '.toml'),
                     argparse.Namespace(out=tempfile.mkdtemp(), layout='1node', peer=None, peer_ip=None))
    cc = ch.controller_cfg(0, ch.out, ['http://x'], ['http://127.0.0.1:8290'] if ch.cfg.get('multiturn') else [], 0)
    fam = {'len_qwen3': 'qwen3', 'len_n9': 'n9', 'len_n3': 'n3'}[cc['length_col']]
    tok = AutoTokenizer.from_pretrained(cc['tokenizer_dir'])
    pad = cc['pad_id'] if cc.get('pad_id') is not None else (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    meta = json.load(open(os.path.join(cc['subset'], 'SUBSET.json')))
    t = pq.read_table(cc['subset'] + '/subset.parquet', columns=['rank', 'prompt_id', 'dataset', 'subset', 'mix_domain']).to_pylist()
    mt = cc['multiturn']
    single = {r['prompt_id'] for r in t if not ('%s/%s' % (r['dataset'], r['subset']) in mt or r['dataset'] in mt)}
    single -= {r['prompt_id'] for r in t if ('%s/%s' % (r['dataset'], r['subset']) in mt or r['dataset'] in mt)}  # the mock's turn lengths
    rng = random.Random(0)
    rollout.generate, rollout.sim_chat = make_mock(tok, fam, tok.eos_token_id, rng, single), mock_sim
    if fam == 'n3' and cc['multiturn']:
        multiturn.PARSE_URL = parse_service_up(cc['tokenizer_dir'])
    src = SubsetSource(cc['subset'], cc['tokenizer_dir'], ranks=tuple(meta['train_ranks']), hist_sup=True,
                       expected_response=cc['expected_response'], length_col=cc['length_col'], add_bos=False)
    cfg = dict(W=131072, window=cc['window'], n_packs=cc['n_packs'], max_packs=cc['max_packs'], target_packs=cc['target_packs'],
               cap=cc['cap'], vllm=['http://x'], model='s', sampling=cc['sampling'], log_dir=tempfile.mkdtemp(),
               multiturn=cc['multiturn'], ds_names=sorted(set(src.ds)), sample_every=0, pad_id=pad,
               scenario_cards=cc['scenario_cards'], bad_ids_file=cc.get('bad_ids_file'))
    cfg['_tok'], cfg['_fam'] = tok, fam
    if cfg.get('bad_ids_file'):
        cfg['_bad_ids'] = frozenset(json.load(open(cfg['bad_ids_file']))['ids'])
    rollout.load_cards(cfg)
    dom = {r['rank']: r['mix_domain'] for r in t}  # by rank: a conversation can be in two domains (cut / end to end)
    for k in range(2):
        calls.clear()
        feed = Feed(pad)
        open(cfg['log_dir'] + '/rollouts.jsonl', 'w').close()
        t0 = time.time()
        st = await rollout.run_step(k, src, feed, cfg, None)
        rl = [json.loads(l) for l in open(cfg['log_dir'] + '/rollouts.jsonl')]
        per = collections.Counter((dom.get(r.get('rank'), '?'), r.get('rank')) for r in rl if r.get('route') == 'pack')
        bfcl_multi = sum(1 for (d, _), c in per.items() if d == 'bfcl' and c > 1)
        tau_multi = sum(1 for (d, _), c in per.items() if d == 'tau2' and c > 1)
        doms = collections.Counter(d for d, _ in per)
        errs = [r for r in rl if 'error' in str(r.get('finish', ''))]
        print('%s step %d: %.0fs, packs %d (published %d), prompts by domain %s, τ² episodes with >1 sample %d, calls %s'
              % (exp, k, time.time() - t0, st['packs'], feed.n, dict(doms), tau_multi, dict(calls)), flush=True)
        assert feed.n == cfg['max_packs'], feed.n
        assert feed.pad_ok, 'an empty pack holds tokens other than the pad id %d' % pad
        if bfcl_multi:
            bad = [rk for (d, rk), c in per.items() if d == 'bfcl' and c > 1][:2]
            for r in rl:
                if r.get('rank') in bad:
                    print('  DEBUG', {k: r.get(k) for k in ('route', 'step', 'rank', 'prompt_id', 'dataset', 'finish', 'prompt', 'response', 'turn', 'episode')})
        assert bfcl_multi == 0, 'a function-calling (*_b1) row produced more than one sample'
        assert not errs, errs[:2]
        if cc['multiturn']:
            assert calls['mt_turn'] > 0 and calls['sim'] > 0, calls
            if any((m.get('sim_kw') or {}).get('tool_v2') for m in cc['multiturn'].values()):  # mix60: the checked environment ran
                assert st.get('mt_user_samples', 0) > 0, {k: v for k, v in st.items() if k.startswith('mt_')}
    if fam == 'n3' and cc['multiturn']:
        assert calls['tool_call'] > 0, 'no tool call was mocked for the parse service'
    print('%s ok (family %s, pad id %d)' % (exp, fam, pad), flush=True)


async def main(exps):
    for e in exps:
        await one(e)
    print('round-4 rollout mock test passed')


if __name__ == '__main__':
    asyncio.run(main(sys.argv[1:] or ['q30_stage1', 'q30_stage2', 'n9', 'n3']))
