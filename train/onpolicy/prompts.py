"""Prompt source for on-policy rollouts: samples prompts from the prompt pool (<root>/<DATASET>/, built by the QAD
data preparation) by a dataset mixture and renders them exactly as the length pass did
(to_chat.py + the model's chat template, generation prompt, thinking on), so rollout prompts, their measured
lengths and the 124K filter all agree.

    src = PromptSource(root, weights={'SFT-SWE-v3': 1.0, ...}, tokenizer_dir=..., seed=0)
    item = src.next()            # dict(dataset, prompt_id, ids, n)          (random dataset by token-share weight)
    item = src.next(max_len=...) # only prompts shorter than max_len tokens (top-up of short rollouts)
    item = src.next(max_len=..., allow_none=True)  # None if no such prompt turns up in 1000 draws

Weights are token shares of the *prompts drawn*: a dataset is drawn with probability proportional to
weight / mean prompt+expected-response length, so each dataset's share of generated tokens follows its weight.
Within a dataset prompts are drawn without replacement in a seeded random order. `record(dataset, tokens)` feeds
back what was actually used; next() corrects the draw towards the target shares (deficit-weighted).
Prompts with len_qwen3 > 126,976 (124K) or a template error are skipped (they belong to the long-context set), and
so are the prompts in <DATASET>/exclude.parquet (overlap with the evaluation sets, written by the data preparation).
Datasets with a groups.parquet (from the data preparation: Finance by report, SWE-v3 by issue) are drawn group-first:
each pass visits every group once in a shuffled order and takes one unused prompt of it, so a report asked 291
questions is not drawn 291 times as often as one asked once.
"""
import os, sys

import numpy as np
import pyarrow.parquet as pq

LIMIT = 126976
# history supervision (v2 subsets): the assistant turns inside a prompt, from after the turn's role header up to and
# including its end-of-turn token (or the next header); the last header is the generation prompt.
# Identical to the data preparation's length pass, which counted hist_<family> with it.
HIST_MARK = {'qwen3': ('<|im_start|>assistant\n', '<|im_end|>'), 'n3': ('<|im_start|>assistant\n', '<|im_end|>'),
             'n9': ('<SPECIAL_11>Assistant\n', '<SPECIAL_12>')}


def hist_char_spans(s, head, end):
    starts, i = [], 0
    while True:
        j = s.find(head, i)
        if j < 0:
            break
        starts.append(j); i = j + 1
    out = []
    for a, nxt in zip(starts[:-1], starts[1:]):
        b0 = a + len(head)
        e = s.find(end, b0, nxt)
        out.append((b0, e + len(end) if e >= 0 else nxt))
    return out


class PromptSource:
    def __init__(self, root, weights, tokenizer_dir, seed=0, length_col='len_qwen3', expected_response=None):
        from transformers import AutoTokenizer
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from to_chat import to_chat
        self.to_chat, self.tok = to_chat, AutoTokenizer.from_pretrained(tokenizer_dir)
        self.rng = np.random.default_rng(seed)
        self.root, self.weights = root, {k: float(v) for k, v in weights.items() if v > 0}
        self.idx, self.pos, self.meanlen = {}, {}, {}
        self.expected = dict(expected_response or {})
        self.used = {k: 0.0 for k in self.weights}
        for d in self.weights:
            L = pq.read_table('%s/%s/lengths.parquet' % (root, d), columns=['prompt_id', length_col, 'error'])
            n = L.column(length_col).to_numpy()
            ok = (n > 0) & (n <= LIMIT) & np.array([e is None for e in L.column('error').to_pylist()])
            xp = '%s/%s/exclude.parquet' % (root, d)
            if os.path.exists(xp):  # overlap with the evaluation sets: never drawn
                bad = set(pq.read_table(xp, columns=['prompt_id']).column(0).to_pylist())
                ok &= np.array([p not in bad for p in L.column('prompt_id').to_pylist()])
            keep = np.flatnonzero(ok)
            self.idx[d] = (keep[self.rng.permutation(len(keep))], n)
            self.ids_by_row = getattr(self, 'ids_by_row', {})
            self.ids_by_row[d] = L.column('prompt_id').to_pylist()
            self.pos[d] = 0
            self.meanlen[d] = float(n[keep].mean()) + self.expected.get(d, 4000.0)
            gp = '%s/%s/groups.parquet' % (root, d)
            if os.path.exists(gp):
                G = pq.read_table(gp)
                assert G.column('prompt_id').to_pylist() == self.ids_by_row[d], 'groups.parquet out of step (%s)' % d
                gcol = G.column('group').to_pylist()
                members = {}
                for i in keep.tolist():
                    members.setdefault(gcol[i], []).append(i)
                for v in members.values():
                    self.rng.shuffle(v)
                self.groups = getattr(self, 'groups', {})
                self.groups[d] = dict(members=members, order=list(members), gpos=0, used={})
                self.rng.shuffle(self.groups[d]['order'])
        self.tables = {}

    def _table(self, d):
        if d not in self.tables:
            self.tables[d] = pq.read_table('%s/%s/prompts.parquet' % (self.root, d), columns=['prompt_id', 'prompt', 'tools'])
        return self.tables[d]

    def _pick_dataset(self):
        tot = sum(self.used.values()) or 1.0
        wsum = sum(self.weights.values())
        score = {}
        for d, w in self.weights.items():
            deficit = max(w / wsum - self.used[d] / tot, 0.0) + 1e-3 * w / wsum  # draw towards the target share
            score[d] = deficit / self.meanlen[d]
        z = sum(score.values())
        ds = list(score)
        return ds[self.rng.choice(len(ds), p=[score[d] / z for d in ds])]

    def next(self, max_len=None, allow_none=False):
        for _ in range(1000):
            d = self._pick_dataset()
            if d in getattr(self, 'groups', {}):
                i = self._draw_grouped(d, max_len)
                if i is None:
                    continue
                return self._render(d, i)
            order, n = self.idx[d]
            if self.pos[d] >= len(order):
                self.pos[d] = 0  # epoch over this dataset: reshuffle
                self.idx[d] = (order[self.rng.permutation(len(order))], n)
                order = self.idx[d][0]
            i = int(order[self.pos[d]])
            if max_len is not None and n[i] >= max_len:
                continue  # a short one is wanted; try another draw (does not consume this prompt)
            self.pos[d] += 1
            return self._render(d, i)
        if allow_none:
            return None
        raise RuntimeError('no prompt shorter than %s found' % max_len)

    def _draw_grouped(self, d, max_len, tries=64):
        """Next group in the pass, one unused prompt of it (shorter than max_len if given)."""
        g = self.groups[d]; n = self.idx[d][1]
        for _ in range(tries):
            if g['gpos'] >= len(g['order']):  # pass over: groups with prompts left, reshuffled
                g['order'] = [k for k in g['members'] if g['used'].get(k, 0) < len(g['members'][k])]
                if not g['order']:  # every prompt used: new epoch
                    g['used'] = {}
                    g['order'] = list(g['members'])
                self.rng.shuffle(g['order']); g['gpos'] = 0
            k = g['order'][g['gpos']]; g['gpos'] += 1
            u = g['used'].get(k, 0)
            if u >= len(g['members'][k]):
                continue
            i = g['members'][k][u]
            if max_len is not None and n[i] >= max_len:
                continue
            g['used'][k] = u + 1
            return i
        return None

    def _render(self, d, i):
        row = self._table(d).slice(i, 1).to_pylist()[0]
        assert row['prompt_id'] == self.ids_by_row[d][i], 'lengths.parquet out of step with prompts.parquet (%s row %d)' % (d, i)
        msgs, tools = self.to_chat(row['prompt'], row['tools'])
        text = self.tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        ids = self.tok(text, add_special_tokens=False)['input_ids']
        return dict(dataset=d, prompt_id=row['prompt_id'], ids=ids, n=len(ids))

    def update_expected(self, dataset, response_tokens, alpha=0.05):
        """Online estimate of the student's response length per dataset (exponential moving average)."""
        old = self.expected.get(dataset, 4000.0)
        new = (1 - alpha) * old + alpha * response_tokens
        self.meanlen[dataset] += new - old
        self.expected[dataset] = new

    def record(self, dataset, tokens):
        self.used[dataset] = self.used.get(dataset, 0.0) + tokens

    def shares(self):
        tot = sum(self.used.values()) or 1.0
        return {d: round(v / tot, 4) for d, v in self.used.items()}


class SubsetSource:
    """Prompts of a fixed subset (<subset_dir>/subset.parquet, built by the data preparation), each used once, in
    `rank` order (the subset's seeded random order), for runs that must be reproducible and see every prompt once.

        src = SubsetSource(subset_dir, tokenizer_dir, ranks=(0, 57506))       # training split
        item = src.next()              # next unused prompt by rank; None when every prompt has been used
        item = src.next(max_len=...)   # first unused prompt shorter than max_len among the next `lookahead` ranks
                                       # (top-up of a nearly full step); skipped prompts stay and come later

    Same rendering as PromptSource (to_chat.py + the model's chat template, generation prompt, thinking on); the rendered
    length must equal the subset's length column (len_qwen3 / len_n9 / len_n3; checked for every prompt). Same interface as PromptSource for rollout.py
    (expected, update_expected, record, shares); `exhausted` is True once every prompt has been handed out.
    """

    def __init__(self, subset_dir, tokenizer_dir, ranks=None, lookahead=2000, expected_response=None,
                 length_col='len_qwen3', template_kwargs=None, add_bos=False, hist_sup=False):
        from transformers import AutoTokenizer
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from to_chat import to_chat
        self.to_chat, self.tok = to_chat, AutoTokenizer.from_pretrained(tokenizer_dir)
        # length_col: the subset's rendered length for this tokenizer and template (len_qwen3, len_n9, len_n3);
        # template_kwargs: as the lengths were rendered (enable_thinking=True; Nemotron templates ignore it);
        # add_bos: prepend the tokenizer's BOS (Nemotron Nano V2: the evaluation harness sends the rendered text to
        # /v1/completions, where vLLM adds <s>, so the evaluated prompts start with it)
        # hist_sup: prompts flagged hist_sup (v2 subsets) carry item['hist_spans'], the token ranges [a, b) of the
        # history's assistant turns inside ids (they join the KL); their token count must equal the subset's
        # hist_<family> (the data preparation's length pass counts the same spans)
        self.length_col = length_col
        self.fam = {'len_qwen3': 'qwen3', 'len_n9': 'n9', 'len_n3': 'n3'}[length_col]
        self.hist = hist_sup
        self.tkw = {'enable_thinking': True} if template_kwargs is None else dict(template_kwargs)
        self.bos = [self.tok.bos_token_id] if add_bos else []
        assert not add_bos or self.tok.bos_token_id is not None
        path = os.path.join(os.path.expanduser(subset_dir), 'subset.parquet')
        cols = ['rank', 'dataset', 'prompt_id', length_col, 'prompt', 'tools']
        names = pq.ParquetFile(path).schema_arrow.names
        if hist_sup:
            assert 'hist_sup' in names and 'hist_' + self.fam in names, 'hist_sup needs a v2 subset (hist_sup, hist_<fam>)'
            cols += ['hist_sup', 'hist_' + self.fam]
        if 'exp_gen' in names:
            cols.append('exp_gen')  # the student's predicted E[min(response, 32K)] per prompt (length sets)
        if 'subset' in names:
            cols.append('subset')  # per-subset multi-turn config (rollout.py)
        for c in ('response', 'len_response'):  # offline traces: a row with a response is trained as written,
            if c in names:                        # without generation (rollout.py); len_response is checked at rendering
                cols.append(c)
        for c in ('cut_at', 'n_est'):  # stored-cut sets (multi-turn): the trained turn and its size
            if c in names:
                cols.append(c)
        t = pq.read_table(path, columns=cols)
        rank = t.column('rank').to_numpy()
        lo, hi = ranks if ranks else (0, int(rank.max()) + 1)
        keep = np.flatnonzero((rank >= lo) & (rank < hi))
        keep = keep[np.argsort(rank[keep], kind='stable')]
        self.rows = t.take(keep)
        self.n = self.rows.column(length_col).to_numpy() + len(self.bos)
        # predicted response per prompt (-1: none, e.g. validation rows); top-up picks prompts by prompt + prediction
        self.exp = (np.nan_to_num(self.rows.column('exp_gen').to_numpy(zero_copy_only=False).astype(float), nan=-1.0)
                    if 'exp_gen' in self.rows.column_names else np.full(len(self.n), -1.0))
        self.exp_onp = self.exp  # without the offline traces (offline_phase(False): their rows are generated)
        if 'len_response' in self.rows.column_names:  # offline traces: the response length is known exactly
            lr = np.nan_to_num(self.rows.column('len_response').to_numpy(zero_copy_only=False).astype(float), nan=-1.0)
            self.exp = np.where(lr >= 0, lr, self.exp)
        self.exp_off = self.exp
        self.offline = (np.array([x is not None for x in self.rows.column('response').to_pylist()])
                        if 'response' in self.rows.column_names else np.zeros(len(keep), bool))  # rows with a trace
        self.use_offline = True  # offline_phase(False): the traces are ignored, every row is generated
        self.ds = self.rows.column('dataset').to_pylist()
        self.used_flag = np.zeros(len(keep), bool)
        self.head, self.lookahead, self.handed = 0, lookahead, 0
        self.expected = dict(expected_response or {})
        self.used = {}

    def __len__(self):
        return len(self.n)

    @property
    def exhausted(self):
        return self.handed >= len(self.n)

    def next(self, max_len=None, allow_none=True, max_total=None, cap=32768):
        """max_len: prompt shorter than this; max_total: prompt + predicted response (exp_gen, capped; 4000 without
        a prediction) shorter than this (top-up: what still fits into the step's remaining gap)."""
        while self.head < len(self.n) and self.used_flag[self.head]:
            self.head += 1
        if self.head >= len(self.n):
            return None
        if max_len is None and max_total is None:
            i = self.head
        else:
            stop = min(self.head + self.lookahead, len(self.n))
            ok = ~self.used_flag[self.head:stop]
            if max_len is not None:
                ok &= self.n[self.head:stop] < max_len
            if max_total is not None:
                e = self.exp[self.head:stop]
                ok &= self.n[self.head:stop] + np.minimum(np.where(e >= 0, e, 4000.0), cap) < max_total
            cand = np.flatnonzero(ok)
            if not len(cand):
                return None
            i = self.head + int(cand[0])
        self.used_flag[i] = True
        self.handed += 1
        return self._render(i)

    def _render(self, i):
        row = self.rows.slice(i, 1).to_pylist()[0]
        msgs, tools = self.to_chat(row['prompt'], row['tools'])
        text = self.tok.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=True, **self.tkw)
        want_hist = self.hist and row['hist_sup']
        enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=want_hist)
        ids = enc['input_ids']
        assert len(ids) == row[self.length_col], 'rendered %d tokens, subset says %d (%s %s)' % (
            len(ids), row[self.length_col], row['dataset'], row['prompt_id'])
        spans = None
        if want_hist:
            st = np.fromiter((o[0] for o in enc['offset_mapping']), np.int64, len(ids))
            b = len(self.bos)
            spans = [[int(np.searchsorted(st, c0)) + b, int(np.searchsorted(st, c1)) + b]
                     for c0, c1 in hist_char_spans(text, *HIST_MARK[self.fam])]
            spans = [s for s in spans if s[1] > s[0]]
            n_h = sum(e - a for a, e in spans)
            assert n_h == row['hist_' + self.fam], 'history %d tokens, subset says %d (%s %s)' % (
                n_h, row['hist_' + self.fam], row['dataset'], row['prompt_id'])
        ids = self.bos + ids
        eg = row.get('exp_gen')
        resp_ids = None
        if row.get('response') is not None and self.use_offline:  # offline trace: the response as written (its loss covers it like a generated one)
            resp_ids = self.tok(row['response'], add_special_tokens=False)['input_ids']
            if row.get('len_response') is not None:
                assert len(resp_ids) == row['len_response'], 'response rendered %d tokens, subset says %d (%s %s)' % (
                    len(resp_ids), row['len_response'], row['dataset'], row['prompt_id'])
            eg = len(resp_ids)
        return dict(dataset=row['dataset'], prompt_id=row['prompt_id'], ids=ids, n=len(ids), rank=int(row['rank']),
                    exp=float(eg) if eg is not None and eg == eg else None, hist_spans=spans, resp_ids=resp_ids,
                    msgs=msgs, tools=tools, subset=row.get('subset'),  # conversation + subset: multi-turn (multiturn.py)
                    cut_at=row.get('cut_at'), n_est=row.get('n_est'))

    def give_back(self, rank):
        """A generated sample that did not fit into this step's packs: its prompt goes back and is drawn again (and
        regenerated with the then-current weights) in a later step, so every prompt is trained on once."""
        i = int(np.flatnonzero(self.rows.column('rank').to_numpy() == rank)[0])
        if self.used_flag[i]:
            self.used_flag[i] = False
            self.handed -= 1
            self.head = min(self.head, i)

    def mark_used(self, ranks):
        """Resume: prompts with these ranks were used by earlier steps."""
        r = self.rows.column('rank').to_numpy()
        hit = np.isin(r, np.fromiter(ranks, np.int64, len(ranks)))
        self.handed += int((hit & ~self.used_flag).sum())
        self.used_flag |= hit

    def offline_phase(self, on):
        """on: a row with an offline trace is trained as written (the default); off: the traces are ignored, every row
        is generated (rollout.py offline_until). The order of the rows is the same either way."""
        self.use_offline = bool(on)
        self.exp = self.exp_off if on else self.exp_onp

    def update_expected(self, dataset, response_tokens, alpha=0.05):
        old = self.expected.get(dataset, 4000.0)
        self.expected[dataset] = (1 - alpha) * old + alpha * response_tokens

    def record(self, dataset, tokens):
        self.used[dataset] = self.used.get(dataset, 0.0) + tokens

    def shares(self):
        tot = sum(self.used.values()) or 1.0
        return {d: round(v / tot, 4) for d, v in self.used.items()}
