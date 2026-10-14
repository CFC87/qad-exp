"""On-policy rollout controller and pack feed (runs on the rollout machine next to `vllm serve`).

Per step k:
  1. wait until the trainer reports step k-1 done (weights already pushed to vLLM); step 0 starts at once;
  2. bulk: draw prompts until prompt + expected response reaches `bulk` (0.95) of the step's token target
     (target_packs x W, default n_packs) and submit them all at once: the longest generation sets the rollout time;
  3. top-up: after each completion, gap = target - (finished + in-flight estimate); while gap > 0, submit
     prompts expected to be short (they finish quickly and do not lengthen the tail);
  4. every finished rollout goes to OnlinePacker (n_packs open packs, best fit, ship at 97%; a rollout that fits
     none gets a new pack, up to max_packs); each shipped pack is published at /pack/<k>/<seq> for the trainer's data
     iterator; at the end the open packs with data are published and empty packs (loss 0) complete the step to
     exactly max_packs (default n_packs; = the trainer's GBS). So everything generated in a step is trained in that
     step (on-policy, and no bias against the long rollouts that finish last). Only rollouts beyond max_packs are
     dropped; with a fixed subset their prompts go back to the source and are generated again in a later step;
  5. statistics: one line per rollout in <log_dir>/rollouts.jsonl (dataset, prompt_id, rank, prompt and response
     length, finish reason, whether </think> was written, thinking / answer lengths, share of repeated 32-grams, mean
     log p of the sampled tokens) and one line per step in <log_dir>/steps.jsonl (token accounting incl. cumulative
     totals, length percentiles, truncation / no-</think> / repetition rates, self log p, per dataset, times, and every
     `sample_every` steps a few decoded samples); the step's line is also served at /stats/<route>/<k> for the trainer,
     which logs it to wandb next to its own metrics.
Validation (cfg val_every, val_items): when step k with k % val_every == 0 (k > 0) starts, the fixed validation prompts
are generated first with the same weights and published at /val/<e>/<seq> (e = k / val_every - 1), then step k's own
rollouts; the trainer evaluates them (forward only) right after its step k - 1. Both run on vLLM at the same time.
Requests use the prompt as token ids (the exact rendering the lengths were measured on) and return token ids;
sampling is the deployment setting (Qwen3: T 0.6, top_p 0.95, top_k 20, min_p 0), max_tokens = min(cap, window - n), where
window is the model's context (config 'window', default W; 40,960 for plain-RoPE Qwen3 runs) and W the pack length.

HTTP (aiohttp):  GET /pack/<k>/<seq>, /val/<e>/<seq> -> npz (block until published)   POST /step_done/<k>  (trainer)
                 GET /stats/<route>/<k> -> json (blocks until that step's statistics exist)   GET /status -> json
"""
import asyncio, io, json, os, random, sys, time, zlib

import aiohttp
import numpy as np
from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from packing import OnlinePacker, Pack, Sample, PAD_ID_DEFAULT, THINK_END, to_batch  # noqa: E402
import multiturn as MT  # noqa: E402
import ledger  # noqa: E402

QWEN3_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0)


def make_think_end(tok, marker='</think>'):
    """f(response ids) -> index of the last token of the first </think>, or None. Qwen3: one special token (151668).
    Nemotron Nano V2: plain text, '</' 'think' then a token starting with '>' ('>' merges with the newlines after it)."""
    ids = tok.encode(marker, add_special_tokens=False)
    if len(ids) == 1:
        t = ids[0]
        return lambda r: r.index(t) if t in r else None
    head = tok.encode(marker[:-1], add_special_tokens=False)
    L = len(head)
    gt = {i for i in range(len(tok)) if tok.convert_ids_to_tokens(i) is not None
          and tok.decode([i]).startswith(marker[-1])}

    def find(r):
        for i in range(len(r) - L):
            if r[i:i + L] == head and r[i + L] in gt:
                return i + L
        text = tok.decode(r)  # any other segmentation of the marker
        k = text.find(marker)
        if k < 0:
            return None
        lo, hi, end = 0, len(r) - 1, k + len(marker)
        while lo < hi:
            m = (lo + hi) // 2
            if len(tok.decode(r[:m + 1])) >= end:
                hi = m
            else:
                lo = m + 1
        return lo
    return find


class Feed:
    def __init__(self):
        self.packs = {}          # (route, step, seq) -> npz bytes
        self.cond = asyncio.Condition()
        self.done = {}           # step -> time the trainer reported it done
        self.stats = {}          # (route, step) -> statistics of that step
        self.status = {}

    async def publish(self, step, seq, batch, route='pack'):
        buf = io.BytesIO()
        np.savez(buf, **batch)
        async with self.cond:
            self.packs[(route, step, seq)] = buf.getvalue()
            self.cond.notify_all()

    async def put_stats(self, route, step, st):
        async with self.cond:
            self.stats[(route, step)] = st
            self.cond.notify_all()

    async def _serve_pack(self, route, request):
        k, i = int(request.match_info['step']), int(request.match_info['seq'])
        async with self.cond:
            await self.cond.wait_for(lambda: (route, k, i) in self.packs)
            body = self.packs[(route, k, i)]
        return web.Response(body=body, content_type='application/octet-stream')

    async def get_pack(self, request):
        return await self._serve_pack('pack', request)

    async def get_val(self, request):
        return await self._serve_pack('val', request)

    async def get_stats(self, request):
        route, k = request.match_info['route'], int(request.match_info['step'])
        async with self.cond:
            await self.cond.wait_for(lambda: (route, k) in self.stats)
            st = self.stats[(route, k)]
        return web.json_response(st)

    async def step_done(self, request):
        k = int(request.match_info['step'])
        async with self.cond:
            self.done[k] = time.time()
            for key in [x for x in self.packs if x[0] == 'pack' and x[1] <= k]:  # every stage has read them by now
                del self.packs[key]
            self.cond.notify_all()
        return web.json_response({'ok': True, 'step': k})

    async def wait_done(self, k):
        async with self.cond:
            await self.cond.wait_for(lambda: k in self.done)

    async def get_status(self, request):
        return web.json_response(self.status)


async def generate(session, url, model, item, max_tokens, sampling, logprobs=True):
    """(response token ids, finish reason, mean log p of the sampled tokens or None)."""
    body = dict(model=model, prompt=item['ids'], max_tokens=max_tokens, return_token_ids=True,
                skip_special_tokens=False, **sampling)
    if logprobs:
        body['logprobs'] = 0  # the sampled token's log p only
    for attempt in range(5):
        try:
            async with session.post(url + '/v1/completions', json=body, timeout=aiohttp.ClientTimeout(total=None)) as r:
                js = await r.json()
            ch = js['choices'][0]
            lp = None
            tl = (ch.get('logprobs') or {}).get('token_logprobs') if isinstance(ch.get('logprobs'), dict) else None
            if tl:
                v = [x for x in tl if x is not None]
                lp = float(np.mean(v)) if v else None
            return list(ch['token_ids']), ch.get('finish_reason'), lp
        except Exception as e:  # a dropped connection must not take the step down
            print('request retry %d for %s: %r' % (attempt, item['prompt_id'], e), flush=True)
            if not (attempt == 0 and isinstance(e, aiohttp.ServerDisconnectedError)):
                await asyncio.sleep(5 * (attempt + 1))  # a stale keep-alive connection is retried at once
    raise RuntimeError('rollout failed for %s' % item['prompt_id'])


async def sim_chat(session, url, sc, messages, max_tokens, stats=None, accept_cut=False):
    """One user / tool simulator call (OpenAI-compatible chat, sc = the multi-turn config's "sim") -> the reply text.
    Up to 3 attempts, each with a timeout (sc "timeout", default 300 s; behind sim_throttle.py a rate-limited key makes
    a request wait there while the throttle retries). An error, an empty reply (a reasoning simulator such as gpt-oss
    used up max_tokens on its reasoning) or a cut-off one (finish_reason length: e.g. half a JSON tool result) is
    retried; the third failure raises, which ends that episode as an env error (its finished turns kept in mode each).
    stats: counts mt_sim_empty / mt_sim_cutoff / mt_sim_error per failed attempt. accept_cut: a cut-off reply
    (non-empty) is returned as it is (tool results)."""
    body = dict(model=sc['model'], messages=messages, max_tokens=max_tokens, temperature=sc.get('temperature', 0.7),
                top_p=sc.get('top_p', 0.8), **sc.get('extra', {}))
    for attempt in range(3):
        kind = 'mt_sim_error'
        try:
            async with session.post(url + '/v1/chat/completions', json=body,
                                    timeout=aiohttp.ClientTimeout(total=sc.get('timeout', 300))) as r:
                j = await r.json(content_type=None)
            ch = j['choices'][0]
            txt = ((ch.get('message') or {}).get('content') or '').strip()
            # a lone surrogate (half an emoji escaped as "\ud83d" in the API's JSON) cannot be encoded as UTF-8 and
            # would crash the controller when it writes episodes.jsonl; replace it
            txt = txt.encode('utf-8', 'replace').decode('utf-8')
            if txt and accept_cut and ch.get('finish_reason') == 'length':
                if stats is not None:
                    stats['mt_sim_cut_kept'] = stats.get('mt_sim_cut_kept', 0) + 1
                return txt
            if not txt or ch.get('finish_reason') == 'length':
                kind = 'mt_sim_empty' if not txt else 'mt_sim_cutoff'
                raise ValueError('simulator reply %s' % ('empty' if not txt else 'cut off at max_tokens'))
            return txt
        except Exception:
            if stats is not None:
                stats[kind] = stats.get(kind, 0) + 1
            if attempt == 2:
                raise
            await asyncio.sleep(sc.get('retry_sleep', 5))


def hist_tokens(s):
    """tokens of a sample's history spans (history supervision; 0 without)"""
    return sum(b - a for a, b in s.meta.get('hist_spans') or ())


def repeated_share(ids, G=32):
    """Share of positions whose 32-gram (ending there) already occurred earlier in the response."""
    n = len(ids)
    if n <= G:
        return 0.0
    a = np.asarray(ids, np.uint64)
    h = np.zeros(n - G + 1, np.uint64)
    for j in range(G):
        h = h * np.uint64(1000003) + a[j:n - G + 1 + j]
    seen, rep = set(), 0
    for x in h.tolist():
        if x in seen:
            rep += 1
        else:
            seen.add(x)
    return rep / (n - G + 1)


def _pct(xs, qs=(50, 90, 99)):
    if not len(xs):
        return {}
    a = np.asarray(xs)
    return dict(**{'p%d' % q: float(np.percentile(a, q)) for q in qs}, max=int(a.max()), mean=float(a.mean()))


class Dispatch:
    """Several rollout servers (cfg vllm = list of URLs): each request goes to the server with the fewest estimated
    tokens in flight (prompt + expected response), so long prompts and long generations spread evenly."""

    def __init__(self, urls):
        self.urls = [u.strip() for u in urls.split(',')] if isinstance(urls, str) else list(urls)
        self.load = {u: 0.0 for u in self.urls}

    def acquire(self, est_tokens):
        u = min(self.urls, key=lambda x: self.load[x])
        self.load[u] += est_tokens
        return u

    def release(self, u, est_tokens):
        self.load[u] -= est_tokens


def load_cards(cfg):
    """offline scenario cards: prompt_id -> card, for env sim; re-read when a file changed (the card job may still be
    writing ahead of the run, in rank order)"""
    if not cfg.get('scenario_cards'):
        return
    import pyarrow.parquet as pq
    paths = cfg['scenario_cards'] if isinstance(cfg['scenario_cards'], list) else [cfg['scenario_cards']]
    stamp = tuple(os.path.getmtime(os.path.expanduser(p)) if os.path.exists(os.path.expanduser(p)) else 0 for p in paths)
    if stamp == cfg.get('_cards_stamp'):
        return
    cards = {}
    for pth in paths:
        if os.path.exists(os.path.expanduser(pth)):
            t = pq.read_table(os.path.expanduser(pth), columns=['prompt_id', 'card', 'ok']).to_pylist()
            cards.update({r['prompt_id']: r['card'] for r in t if r['ok'] and r['card']})
    cfg['_cards'], cfg['_cards_stamp'] = cards, stamp
    print('scenario cards: %d' % len(cards), flush=True)


async def run_step(k, src, feed, cfg, session, route='pack', fixed=None):
    """One step's rollouts into cfg n_packs packs published on `route`. fixed: a list of prompt items to generate
    (validation) instead of drawing from src."""
    W, n_packs = cfg['W'], cfg['n_packs']
    WIN = cfg.get('window') or W  # the model's context window (generation: max_tokens = min(cap, WIN - prompt)); W is the pack
    load_cards(cfg)
    if '_think_end' not in cfg:  # main() sets both from the tokenizer; Qwen3 defaults for direct callers (tests)
        cfg['_think_end'] = lambda r: r.index(THINK_END) if THINK_END in r else None
    cfg.setdefault('pad_id', PAD_ID_DEFAULT)
    n_out = cfg.get('max_packs', n_packs)  # packs published per step = the trainer's micro-batches (GBS)
    target = cfg.get('target_packs', n_packs) * W  # generation aims at this many packs of tokens
    packer = OnlinePacker(n_packs, W, cfg.get('ship_at', 0.97), max_packs=n_out)
    ds_names = cfg.get('ds_names')
    seq, done_tok, inflight = 0, 0, {}
    t0 = time.time()
    lock = asyncio.Lock()
    tasks = set()
    stats = dict(route=route, step=k, started=t0, submitted=0, finished=0, truncated=0, gen_tokens=0, prompt_tokens=0)
    by_ds = {}
    rlog = []
    texts = []
    touched = {}  # rank -> its last event in this step: 'd' drawn from the source, 'b' given back (ledger.py)

    def draw(**kw):
        it = src.next(**kw)
        if it is not None and it.get('rank') is not None:
            touched[int(it['rank'])] = 'd'
        return it

    def give(rank):
        src.give_back(rank)
        touched[int(rank)] = 'b'

    def mt_cfg(item):  # multi-turn config per "dataset/subset" (e.g. "SFT-Agentic-v2/interactive_agent") or per dataset
        mtd = cfg.get('multiturn') or {}
        key = '%s/%s' % (item['dataset'], item.get('subset'))
        return (key, mtd[key]) if key in mtd else (item['dataset'], mtd.get(item['dataset']))

    def e2e_ratio(key, mtc):  # tokens of an end-to-end episode (all its samples) per n_est token, over the episodes
        a, b = cfg.setdefault('_e2e_cost', {}).get(key, (0.0, 0.0))  # finished so far (sums), with the prior worth
        w = 20 * 7500.0  # ~20 episodes of n_est
        return (a + w * float(mtc.get('cost_prior', 8.0))) / (b + w)

    def est(item):  # the prompt's own prediction (length sets: exp_gen) else the dataset's running mean
        e = item.get('exp')
        n = item.get('n_est') or item['n']  # stored-cut sets: the trained sample is the conversation up to the cut
        if fixed is None:
            key, mtc = mt_cfg(item)
            if mtc and mtc.get('mode') == 'each':  # end to end: one sample per student turn, each with its context
                return n * e2e_ratio(key, mtc)  # (stored conversations: 8.2 x n_est on average)
        return n + min(e if e is not None else src.expected.get(item['dataset'], 4000.0), cfg['cap'], WIN - n)

    async def one(item):
        nonlocal seq, done_tok
        disp = cfg.setdefault('_dispatch', Dispatch(cfg['vllm']))
        # multi-turn config per "dataset/subset" (e.g. "SFT-Agentic-v2/interactive_agent") or per dataset; validation
        # (fixed prompts) stays single-turn so its KL compares across runs
        mt_key, mtc = mt_cfg(item)
        mtc = None if fixed is not None else mtc
        if mtc and mtc.get('require_card') and item['prompt_id'] not in (cfg.get('_cards') or {}):
            # env sim only with a checked offline scenario card (online cards can swap the roles): else single-turn,
            # except in a stored-cut set, whose prompt is the whole conversation (not the cut): that row is dropped
            stats['mt_no_card'] = stats.get('mt_no_card', 0) + 1
            if item.get('cut_at') is not None:
                async with lock:
                    inflight.pop(id(item), None)
                return
            mtc = None
        if mtc:  # multi-turn episode: the student writes every assistant turn (multiturn.py); one or more samples
            async def gen(ids, mtok):
                e = len(ids) + 0.25 * mtok
                u = disp.acquire(e)
                try:
                    return await generate(session, u, cfg['model'], dict(ids=ids, prompt_id=item['prompt_id']), mtok,
                                          cfg['sampling'], cfg.get('logprobs', True))
                finally:
                    disp.release(u, e)
            sim, sim_kw = None, dict(mtc.get('sim_kw') or {})
            if mtc.get('env') in ('sim', 'replay_tools'):  # an OpenAI-compatible chat server simulates users / tools
                sc = mtc['sim']
                urls = sc['url'] if isinstance(sc['url'], list) else [sc['url']]
                url = urls[zlib.crc32(str(item['prompt_id']).encode()) % len(urls)]  # one key (throttle) per episode

                async def sim(messages, max_tokens, accept_cut=False):
                    return await sim_chat(session, url, sc, messages, max_tokens, stats, accept_cut)
                card = (cfg.get('_cards') or {}).get(item['prompt_id'])
                if card and mtc.get('env') == 'sim':
                    sim_kw['card'] = card
            try:
                outs, info = await MT.run_episode(
                    gen, cfg['_tok'], item['msgs'], item['tools'], window=WIN, cap=cfg['cap'],
                    mode=mtc.get('mode', 'last'), env=mtc.get('env', 'replay'), min_gen=mtc.get('min_gen', 1024),
                    fam=cfg.get('_fam', 'qwen3'),
                    max_turns=mtc.get('max_turns', 64), sim=sim, sim_kw=sim_kw,
                    rng=random.Random(zlib.crc32(('%s/%d' % (item['prompt_id'], k)).encode())),
                    cut_at=item.get('cut_at') if mtc.get('stored_cut', True) else None,
                    deadline=t0 + mtc['deadline_s'] if mtc.get('deadline_s') else None)
            except Exception as e:  # never let one episode take the step down
                print('step %d: multi-turn episode %s failed: %r' % (k, item['prompt_id'], e), flush=True)
                stats['mt_failed'] = stats.get('mt_failed', 0) + 1
                outs, info = [], dict(turns=[], ended='failed')
            if 'sim' in info:
                for kk, vv in info['sim'].items():
                    stats['mt_' + kk] = stats.get('mt_' + kk, 0) + vv
            stats['mt_resampled'] = stats.get('mt_resampled', 0) + info.get('resampled', 0)
            if mtc.get('log') and info.get('conversation') is not None:
                cfg.setdefault('_mt_log', []).append(dict(step=k, run=cfg.get('_run'), prompt_id=item['prompt_id'], dataset=item['dataset'],
                                                          subset=item.get('subset'), ended=info['ended'],
                                                          cut_at=info.get('cut_at'), target=info.get('target'),
                                                          resampled=info.get('resampled'), turns=info['turns'],
                                                          sim=info.get('sim'), conversation=info['conversation']))
            stats['mt_episodes'] = stats.get('mt_episodes', 0) + 1
            stats['mt_turns'] = stats.get('mt_turns', 0) + len(info['turns'])
            stats.setdefault('mt_ended', {})
            stats['mt_ended'][info['ended']] = stats['mt_ended'].get(info['ended'], 0) + 1
            cut_short = str(info.get('ended', '')).startswith(('stopped: step deadline', 'dropped: generation error'))
            if not outs:  # not even one turn fitted the window, or the episode was cut before its first turn
                async with lock:
                    inflight.pop(id(item), None)
                    if cut_short and fixed is None and hasattr(src, 'give_back') and item.get('rank') is not None:
                        give(item['rank'])  # nothing of it was trained: it comes again in a later step
                        stats['mt_requeued'] = stats.get('mt_requeued', 0) + 1
                return
            # the cost ratio learns from episodes that ran their course (a deadline / an error cuts an episode short:
            # counting those would lower the ratio, so more episodes would be sent, so more would be cut)
            if mtc.get('mode') == 'each' and item.get('n_est') and not str(info.get('ended', '')).startswith(
                    ('stopped: step deadline', 'dropped:', 'failed')):
                a, b = cfg.setdefault('_e2e_cost', {}).get(mt_key, (0.0, 0.0))
                cfg['_e2e_cost'][mt_key] = (a + sum(len(o['ids']) + len(o['resp']) for o in outs), b + item['n_est'])
        else:
            mt = min(cfg['cap'], WIN - item['n'])
            e_tok = est(item)
            url = disp.acquire(e_tok)
            try:
                resp, fin, lp = await generate(session, url, cfg['model'], item, mt, cfg['sampling'],
                                               cfg.get('logprobs', True))
            finally:
                disp.release(url, e_tok)
            outs = [dict(ids=item['ids'], resp=resp, finish=fin, lp=lp, hist_spans=item.get('hist_spans'))]
        for j, o in enumerate(outs):  # an episode's in-flight estimate is settled (and the gap re-checked) at its last
            await record_sample(item, o, final=j == len(outs) - 1)

    task_item, recorded, full = {}, set(), [False]

    def submit(item):
        t = asyncio.ensure_future(one(item))
        tasks.add(t); task_item[t] = item

    async def record_sample(item, o, final=True):
        nonlocal seq, done_tok
        resp, fin, lp = o['resp'], o['finish'], o['lp']
        n_prompt = len(o['ids'])
        bad = cfg.get('_bad_ids')
        if bad:  # tokens the trainer cannot take (Nano V2: untrained ~0 embedding rows -> Inf gradient): keep the prefix
            cut = next((j for j, t in enumerate(resp) if t in bad), None)
            if cut is not None:
                resp, fin = resp[:cut], 'bad_id'
                stats['bad_id_cut'] = stats.get('bad_id_cut', 0) + 1
                print('step %d: response cut at a degenerate token id (%s) after %d tokens' % (k, route, cut), flush=True)
        te = cfg['_think_end'](resp)
        rep = repeated_share(resp)
        async with lock:
            if final:
                inflight.pop(id(item), None)
            s = Sample(o['ids'], resp, dict(dataset=item['dataset'], prompt_id=item['prompt_id'], finish=fin,
                                            rank=item.get('rank'), think_end=te, hist_spans=o.get('hist_spans')))
            done_tok += len(s)
            stats['finished'] += 1; stats['gen_tokens'] += len(resp); stats['truncated'] += fin == 'length'
            stats['prompt_tokens'] += n_prompt
            b = by_ds.setdefault(item['dataset'], dict(n=0, truncated=0, gen_tokens=0, prompt_tokens=0, no_think=0,
                                                       rep=0.0, lp=0.0, lp_n=0))
            b['n'] += 1; b['truncated'] += fin == 'length'; b['gen_tokens'] += len(resp); b['prompt_tokens'] += n_prompt
            b['no_think'] += te is None; b['rep'] += rep
            if lp is not None:
                b['lp'] += lp; b['lp_n'] += 1
            rlog.append(dict(step=k, route=route, run=cfg.get('_run'), dataset=item['dataset'], prompt_id=item['prompt_id'],
                             rank=item.get('rank'), prompt=n_prompt, response=len(resp), finish=fin,
                             think_end=te is not None, think=(te + 1) if te is not None else len(resp),
                             answer=(len(resp) - te - 1) if te is not None else 0, rep32=round(rep, 4),
                             lp=None if lp is None else round(lp, 4), t=round(time.time() - t0, 1)))
            if len(texts) < 64:
                texts.append((dict(item, ids=o['ids'], n=len(o['ids'])), resp))  # multi-turn: this turn's own prompt
            if fixed is None:
                src.update_expected(item['dataset'], len(resp)); src.record(item['dataset'], len(s))
            recorded.add(item.get('rank'))
            for p in packer.add(s):
                await feed.publish(k, seq, to_batch(p, pad_id=cfg['pad_id'], ds_names=ds_names), route); seq += 1
            if packer.dropped and not full[0] and fixed is None:
                # the packs are full: end-to-end episodes still running would only produce dropped samples and hold
                # up the next step; cancel them, their conversations go back to the source below
                full[0] = True
                me = asyncio.current_task()
                for t, it in task_item.items():
                    if t is not me and not t.done() and (mt_cfg(it)[1] or {}).get('mode') == 'each':
                        t.cancel()
            gap = target - done_tok - sum(inflight.values())
            # no top-up once an end-to-end episode started now could not finish before the step's deadline: it would be
            # cut at once and its conversation used up
            dl = min((v.get('deadline_s') for v in (cfg.get('multiturn') or {}).values() if v.get('deadline_s')), default=None)
            late = dl is not None and time.time() > t0 + dl - cfg.get('topup_margin_s', 300)
            while final and fixed is None and not late and not full[0] and gap > cfg.get('min_gap', 0.02) * W:  # top up
                kw = dict(max_total=int(gap) + int(cfg.get('min_gap', 0.02) * W), cap=cfg['cap']) \
                    if hasattr(src, 'exp') else {}  # SubsetSource: prompt + predicted response must fit the gap
                nxt = draw(max_len=min(cfg.get('topup_max_prompt', 16384), int(gap)), allow_none=True, **kw)
                if nxt is None:
                    break  # nothing short enough to fill the rest
                if est(nxt) > gap + cfg.get('min_gap', 0.02) * W and hasattr(src, 'give_back'):
                    give(nxt['rank'])  # an end-to-end episode (source checks prompt + one response) too big
                    break
                inflight[id(nxt)] = est(nxt); gap -= est(nxt); stats['submitted'] += 1
                submit(nxt)

    async with lock:
        if fixed is not None:
            for item in fixed:
                stats['submitted'] += 1
                submit(item)
        else:
            bulk = 0.0
            while bulk < cfg.get('bulk', 0.95) * target:
                item = draw()
                if item is None:  # a fixed subset ran out: this step takes what is left
                    stats['data_exhausted'] = True
                    break
                if (item.get('n_est') or item['n']) > WIN - cfg.get('min_gen', 4096) and \
                        (mt_cfg(item)[1] or {}).get('mode') != 'each':  # no min_gen answer fits (plain-RoPE
                    stats['skipped_long'] = stats.get('skipped_long', 0) + 1  # Qwen3 at 40,960): dropped, as validation does
                    continue
                inflight[id(item)] = est(item); bulk += est(item); stats['submitted'] += 1
                submit(item)
    while True:
        pending = [t for t in tasks if not t.done()]
        if not pending:
            break
        await asyncio.gather(*pending, return_exceptions=True)
    for t in tasks:
        if not t.cancelled():
            t.result()  # surface errors
    for p in packer.finish():
        await feed.publish(k, seq, to_batch(p, pad_id=cfg['pad_id'], ds_names=ds_names), route); seq += 1
    requeued = 0
    if fixed is None and hasattr(src, 'give_back'):  # beyond max_packs only: their prompts come again in a later step
        trained = {s.meta.get('rank') for p in packer.shipped for s in p.samples}  # end to end: an episode whose
        for r in {s.meta.get('rank') for s in packer.dropped} - trained - {None}:  # turns were partly trained is done
            give(r); requeued += 1
        for t, it in task_item.items():  # cancelled once the packs were full, before any sample: comes again later
            if t.cancelled() and it.get('rank') is not None and it['rank'] not in recorded:
                give(it['rank']); requeued += 1
                stats['mt_cancelled'] = stats.get('mt_cancelled', 0) + 1
    if fixed is None:
        # the prompts this step used up (drawn and not given back) and those it gave back (beyond max_packs, cancelled,
        # cut before a turn, too big for the top-up): a resume or a branch takes the used prompts from these
        # (ledger.used_ranks), so a prompt given back here, or drawn without a logged rollout, is counted right
        stats['drawn_ranks'] = sorted(r for r, e in touched.items() if e == 'd')
        stats['requeued_ranks'] = sorted(r for r, e in touched.items() if e == 'b')
    if fixed is None and getattr(src, 'exhausted', False):  # nothing left (after requeueing): this is the last step
        stats['data_exhausted'] = True
    empty = 0
    while seq < n_out:  # complete the step's micro-batches with empty packs (loss 0)
        await feed.publish(k, seq, to_batch(Pack(W), pad_id=cfg['pad_id'], ds_names=ds_names), route); seq += 1; empty += 1
    assert seq == n_out, 'published %d packs, expected %d' % (seq, n_out)
    # token accounting over what went into packs vs what was generated
    packed = [s for p in packer.shipped for s in p.samples]
    resp_lens = [r['response'] for r in rlog]
    for d, b in by_ds.items():
        b['no_think_rate'] = b['no_think'] / max(b['n'], 1); b['rep_mean'] = b['rep'] / max(b['n'], 1)
        b['lp_mean'] = b['lp'] / b['lp_n'] if b['lp_n'] else None
        b['trunc_rate'] = b['truncated'] / max(b['n'], 1); b['resp_mean'] = b['gen_tokens'] / max(b['n'], 1)
        for x in ('rep', 'lp', 'lp_n'):
            b.pop(x)
    if cfg.get('_e2e_cost'):
        stats['mt_e2e_ratio'] = {kk: round(e2e_ratio(kk, mt_cfg(dict(dataset=kk.split('/')[0], subset=kk.split('/', 1)[-1]))[1] or {}), 2)
                                 for kk in cfg['_e2e_cost']}
    stats.update(packer.stats(), seconds=round(time.time() - t0), empty_packs=empty, requeued=requeued, by_dataset=by_ds,
                 packed_samples=len(packed), packed_tokens=int(sum(len(s) for s in packed)),
                 loss_tokens=int(sum(len(s.response) + hist_tokens(s) for s in packed)),
                 hist_loss_tokens=int(sum(hist_tokens(s) for s in packed)),
                 resp_len=_pct(resp_lens), resp_len_hist=resp_lens,
                 trunc_rate=stats['truncated'] / max(stats['finished'], 1),
                 no_think_rate=float(np.mean([not r['think_end'] for r in rlog])) if rlog else 0.0,
                 rep_mean=float(np.mean([r['rep32'] for r in rlog])) if rlog else 0.0,
                 lp_mean=(float(np.mean([r['lp'] for r in rlog if r['lp'] is not None]))
                          if any(r['lp'] is not None for r in rlog) else None))
    if fixed is None:
        stats.update(shares=src.shares(), expected={d: round(v) for d, v in src.expected.items()})
    se = cfg.get('sample_every', 20)
    if se and k % se == 0 and texts and getattr(src, 'tok', None) is not None:
        rnd = random.Random(1000 * k + (route == 'val'))
        pick = rnd.sample(texts, min(cfg.get('n_samples', 8), len(texts)))
        stats['samples'] = [dict(dataset=it['dataset'], prompt_id=it['prompt_id'], prompt_tokens=it['n'],
                                 response_tokens=len(r),
                                 prompt_tail=src.tok.decode(it['ids'][-600:])[-2000:],
                                 response=src.tok.decode(r)[:20000]) for it, r in pick]
    with open(cfg['log_dir'] + '/rollouts.jsonl', 'a') as f:
        f.writelines(json.dumps(r) + '\n' for r in rlog)
    if cfg.get('_mt_log'):  # multi-turn episodes with "log": true: whole conversations, for reading
        with open(cfg['log_dir'] + '/episodes.jsonl', 'a') as f:
            f.writelines(json.dumps(r) + '\n' for r in cfg['_mt_log'])  # ASCII escapes: no text can fail to encode
        cfg['_mt_log'] = []
    for kk in [x for x in stats if x.startswith('mt_') and isinstance(stats[x], dict)]:
        stats[kk] = dict(stats[kk])
    return stats


async def main_async(cfg, src):
    feed = Feed()
    app = web.Application(client_max_size=1 << 20)
    app.add_routes([web.get('/pack/{step}/{seq}', feed.get_pack), web.get('/val/{step}/{seq}', feed.get_val),
                    web.post('/step_done/{step}', feed.step_done), web.get('/stats/{route}/{step}', feed.get_stats),
                    web.get('/status', feed.get_status)])
    runner = web.AppRunner(app); await runner.setup()
    await web.TCPSite(runner, '0.0.0.0', cfg['port']).start()
    os.makedirs(cfg['log_dir'], exist_ok=True)
    # idle keep-alive connections are dropped before the server drops them (vLLM / uvicorn closes an idle connection
    # after VLLM_HTTP_TIMEOUT_KEEP_ALIVE, default 5 s; aiohttp keeps it 15 s by default and would reuse a closed one,
    # which shows up as "Server disconnected" retries)
    conn = aiohttp.TCPConnector(limit=cfg.get('max_concurrency', 512), keepalive_timeout=cfg.get('keepalive_timeout', 4.0))
    cum = dict(prompts=0, prompt_tokens=0, gen_tokens=0, packed_tokens=0, loss_tokens=0, dropped_tokens=0)
    if cfg.get('start_step', 0) > 0:  # resume / branch: carry the totals on (the latest attempt of step start - 1)
        c0 = ledger.carried_totals(cfg['log_dir'], cfg['start_step'])
        if c0:
            cum = {x: c0.get(x, 0) for x in cum}
    val_every, val_items = cfg.get('val_every'), cfg.get('val_items')
    total_prompts = len(src) if hasattr(src, '__len__') else None
    # this controller start: on every log line, so a resume or a branch keeps each step's lines of the latest start
    cfg['_run'] = '%x-%d' % (time.time_ns(), os.getpid())

    async def write(st):
        st['run'] = cfg.get('_run')
        with open(cfg['log_dir'] + '/steps.jsonl', 'a') as f:
            f.write(json.dumps({k: v for k, v in st.items() if k != 'resp_len_hist'}) + '\n')
        await feed.put_stats(st['route'], st['step'], st)

    async with aiohttp.ClientSession(connector=conn) as session:
        if cfg.get('val_only'):  # validation of a fixed checkpoint (step 0 / final): one batch on /val/0, no training
            st = await run_step(0, src, feed, cfg, session, route='val', fixed=val_items)
            await write(st)
            print('val-only: %d rollouts, mean response %.0f, truncated %d, %d s; serving until stopped' % (
                st['finished'], st['resp_len'].get('mean', 0), st['truncated'], st['seconds']), flush=True)
            feed.status = dict(state='val-only done', last={x: st[x] for x in ('finished', 'truncated', 'seconds')})
            while True:
                await asyncio.sleep(3600)

        async def val_job(e):
            st = await run_step(e, src, feed, cfg, session, route='val', fixed=val_items)
            await write(st)  # at once: the trainer evaluates right after its step
            print('val %d: %d rollouts, mean response %.0f, truncated %d, dropped %d, %d s' % (
                e, st['finished'], st['resp_len'].get('mean', 0), st['truncated'], st['dropped_samples'], st['seconds']),
                flush=True)
            if st['dropped_samples']:  # validation is not requeued: fewer prompts than the other evaluations
                print('WARNING: validation %d dropped %d rollouts beyond max_packs' % (e, st['dropped_samples']), flush=True)

        n_val = (cfg['start_step'] - 1) // val_every if (val_every and cfg['start_step'] > 0) else 0  # batches served
        k = cfg['start_step']
        while k < cfg['steps']:
            if k > 0:  # on resume too: the trainer pushes the resumed weights and reports step start_step - 1 done
                await feed.wait_done(k - 1)
            feed.status = dict(step=k, state='generating', since=time.time())
            vj = None
            if val_every and val_items and k > 0 and k % val_every == 0 and k == cfg['start_step']:
                n_val = k // val_every  # resumed at a checkpoint: the trainer ran this validation before saving it
            elif val_every and val_items and k > 0 and k % val_every == 0:
                vj = asyncio.ensure_future(val_job(k // val_every - 1))  # submitted first: its requests go ahead
                n_val = k // val_every
                await asyncio.sleep(0)
            st = await run_step(k, src, feed, cfg, session)
            cum['prompts'] += st['finished']; cum['prompt_tokens'] += st['prompt_tokens']; cum['gen_tokens'] += st['gen_tokens']
            cum['packed_tokens'] += st['packed_tokens']; cum['loss_tokens'] += st['loss_tokens']
            cum['dropped_tokens'] += st['dropped_tokens']
            st['cumulative'] = dict(cum, subset_share=(getattr(src, 'handed', 0) / total_prompts) if total_prompts else None)
            await write(st)
            if vj is not None:
                await vj
            feed.status = dict(step=k, state='waiting for trainer', since=time.time(),
                               last={x: st[x] for x in ('fill', 'dropped_samples', 'seconds', 'truncated', 'finished')})
            print('step %d: fill %.3f, dropped %d samples, %d s, truncated %d/%d, generated %d tok (cum %.3fB, %s)' % (
                k, st['fill'], st['dropped_samples'], st['seconds'], st['truncated'], st['finished'], st['gen_tokens'],
                (cum['packed_tokens']) / 1e9, 'subset %.1f%%' % (100 * st['cumulative']['subset_share'])
                if st['cumulative']['subset_share'] is not None else ''), flush=True)
            if st.get('data_exhausted'):
                print('subset exhausted at step %d: this was the last step' % k, flush=True)
                break
            k += 1
        last = min(k, cfg['steps'] - 1)
        await feed.wait_done(last)
        if val_every and val_items:
            # the trainer still evaluates after its last step if that step is a multiple of val_every, and once more
            # after training (Megatron's final validation); both see the final weights: generate once, serve twice
            need = (last + 1) // val_every + 1
            if need > n_val:
                st = await run_step(n_val, src, feed, cfg, session, route='val', fixed=val_items)
                await write(st)
                for e in range(n_val + 1, need):
                    async with feed.cond:
                        for i in range(cfg.get('max_packs', cfg['n_packs'])):
                            feed.packs[('val', e, i)] = feed.packs[('val', n_val, i)]
                    await feed.put_stats('val', e, dict(st, step=e))
                print('final validation after step %d: %d rollouts, mean response %.0f (served as val %s)' % (
                    last, st['finished'], st['resp_len'].get('mean', 0), list(range(n_val, need))), flush=True)
            while True:  # keep serving until the trainer has read everything; the driver stops the controller
                await asyncio.sleep(3600)
    await runner.cleanup()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, help='json: vllm, model, W, n_packs, cap, steps, start_step, port, '
                    'log_dir, tokenizer_dir and either weights {dataset: token share} + prompts_root (mixture) or '
                    'subset + train_ranks [lo, hi) (fixed subset in rank order); optional val_ranks, val_n (validation '
                    'prompts: the first val_n of val_ranks), val_every, val_only (validate a fixed checkpoint: one '
                    'batch on /val/0, no training), expected_response, sample_every, n_samples')
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    cfg.setdefault('sampling', QWEN3_SAMPLING)
    tok_dir = os.path.expanduser(cfg['tokenizer_dir'])
    # rendering per model: length column, template kwargs, BOS (see SubsetSource)
    rkw = dict(length_col=cfg.get('length_col', 'len_qwen3'), template_kwargs=cfg.get('template_kwargs'),
               add_bos=cfg.get('add_bos', False))
    if cfg.get('hist_sup'):  # history supervision (subsets v2): the history's assistant turns join the KL
        rkw['hist_sup'] = True
    if cfg.get('subset'):
        from prompts import SubsetSource
        src = SubsetSource(cfg['subset'], tok_dir, ranks=tuple(cfg['train_ranks']),
                           expected_response=cfg.get('expected_response'), **rkw)
        start = cfg.get('start_step', 0)
        if start:  # resume / branch: skip the prompts earlier steps used (ledger.py: the latest attempt of each step,
            used = ledger.used_ranks(cfg['log_dir'], start)  # minus the prompts a step gave back)
            src.mark_used(used)
            print('resume at step %d: %d prompts already used' % (start, len(used)), flush=True)
        if cfg.get('val_ranks'):
            vs = SubsetSource(cfg['subset'], tok_dir, ranks=tuple(cfg['val_ranks']), **rkw)
            n = cfg.get('val_n') or len(vs)
            win, items = cfg.get('window') or cfg['W'], []
            while len(items) < n:  # prompts that leave < min_gen tokens of the window are skipped (plain-RoPE runs)
                x = vs.next()
                if x is None:
                    break
                if x['n'] <= win - cfg.get('min_gen', 4096):
                    items.append(x)
            cfg['val_items'] = items
            print('validation: %d fixed prompts, %d prompt tokens' % (
                len(cfg['val_items']), sum(x['n'] for x in cfg['val_items'])), flush=True)
        # every dataset a pack can hold, the validation prompts' too (the shared validation set of the length sets has
        # datasets their training part lacks)
        cfg.setdefault('ds_names', sorted(set(src.ds) | {x['dataset'] for x in cfg.get('val_items') or []}))
    else:
        from prompts import PromptSource
        src = PromptSource(os.path.expanduser(cfg['prompts_root']), cfg['weights'], tok_dir,
                           seed=cfg.get('seed', 0), expected_response=cfg.get('expected_response'))
        cfg.setdefault('ds_names', sorted(cfg['weights']))
    tok = src.tok
    cfg['_think_end'] = make_think_end(tok)
    cfg['_tok'] = tok  # multi-turn episodes render each turn with the model's own template (multiturn.py)
    cfg['_fam'] = getattr(src, 'fam', 'qwen3')  # and parse it as that model's eval parsers do
    load_cards(cfg)
    if cfg.get('pad_id') is None:
        cfg['pad_id'] = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    assert 0 <= cfg['pad_id'] < len(tok), 'pad id %s outside the vocabulary' % cfg['pad_id']
    if cfg.get('bad_ids_file'):
        cfg['_bad_ids'] = frozenset(json.load(open(os.path.expanduser(cfg['bad_ids_file'])))['ids'])
        print('responses are cut at %d degenerate token ids (%s)' % (len(cfg['_bad_ids']), cfg['bad_ids_file']), flush=True)
    print('tokenizer %s: pad id %d, </think> = %s, BOS %s, length column %s, sampling %s' % (
        tok_dir, cfg['pad_id'], tok.encode('</think>', add_special_tokens=False), rkw['add_bos'], rkw['length_col'],
        cfg['sampling']), flush=True)
    asyncio.run(main_async(cfg, src))


if __name__ == '__main__':
    main()
