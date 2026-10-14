"""Per-token statistics of the KD loss on the last pipeline stage (qad_distill.py, CL_CHUNKED_KD with the on-policy
feed). For every loss position of a pack (feed_client.META gives dataset, position in the response, thinking / answer
segment, label) it accumulates, separately for training and validation packs:

  kl          the loss itself: KL(teacher || student) at that position (full vocabulary, T = 1)
  top1        teacher and student put their highest probability on the same token
  logratio    log p_teacher(y) - log p_student(y) at the sampled token y (the student's own rollout token)
  nll_s/nll_t -log p of the sampled token under student / teacher
  kl>0.1, kl>1, max kl   the tail of the per-token KL

grouped overall, by dataset, by position in the response (0-1K, 1-4K, 4-16K, 16K+) and by segment (thinking up to and
including </think>, answer after it). History spans (history supervision, meta seg 3, rpos -1) are loss positions but
stay out of all / ds / pos -- those keep measuring the student's own rollout, comparable with runs without history
supervision -- and are reported as seg/history and, per dataset, hist_ds/<dataset>; sup = response + history, the
positions the loss averages over (only reported when there is history). The extra quantities need the two output layers once more on the loss positions
(no grad, chunks of 2048 tokens): a small fraction of a step. flush(split) returns flat metric names and resets.
"""
import math, os

import numpy as np
import torch

BUCKET_EDGES = [1024, 4096, 16384]
BUCKET_NAMES = ['0-1k', '1-4k', '4-16k', '16k+']
SEG_NAMES = {1: 'think', 2: 'answer', 3: 'history'}
FIELDS = ['n', 'kl', 'top1', 'logratio', 'nll_s', 'nll_t', 'kl_gt0.1', 'kl_gt1']
STAT_CHUNK = 2048


class KDStats:
    def __init__(self):
        self.names = None
        self.acc = {}   # split -> dict(group kind -> tensor [n_groups, len(FIELDS)] float64)
        self.maxkl = {}

    def _groups(self, split, n_ds, device):
        if split not in self.acc:
            self.acc[split] = dict(all=torch.zeros(1, len(FIELDS), dtype=torch.float64, device=device),
                                   ds=torch.zeros(n_ds, len(FIELDS), dtype=torch.float64, device=device),
                                   pos=torch.zeros(len(BUCKET_NAMES), len(FIELDS), dtype=torch.float64, device=device),
                                   seg=torch.zeros(4, len(FIELDS), dtype=torch.float64, device=device),
                                   hds=torch.zeros(n_ds, len(FIELDS), dtype=torch.float64, device=device),
                                   sup=torch.zeros(1, len(FIELDS), dtype=torch.float64, device=device))
            self.maxkl[split] = 0.0
        return self.acc[split]

    @torch.no_grad()
    def update(self, meta, kl_tok, hs, ht, head_s, head_t):
        """meta: one feed_client.META entry; kl_tok [s] per-token loss of the pack; hs / ht [s, b, h] hidden states;
        head_s / head_t: callables hidden -> logits (the real output layers)."""
        rpos = meta['rpos']
        if len(rpos) != kl_tok.shape[0]:
            return False
        sel_np = np.flatnonzero((rpos >= 0) | (meta['seg'] == 3))
        if not len(sel_np):
            return True
        if self.names is None:
            self.names = list(meta['names'])
        dev = kl_tok.device
        sel = torch.from_numpy(sel_np).to(dev)
        kl = kl_tok[sel].double()
        labels = torch.from_numpy(meta['labels'][sel_np].astype(np.int64)).to(dev)
        top1 = torch.empty(len(sel_np), dtype=torch.float64, device=dev)
        lps = torch.empty(len(sel_np), dtype=torch.float64, device=dev)
        lpt = torch.empty(len(sel_np), dtype=torch.float64, device=dev)
        for j in range(0, len(sel_np), STAT_CHUNK):
            idx = sel[j:j + STAT_CHUNK]
            ls, lt = head_s(hs[idx]), head_t(ht[idx])
            ls = (ls[0] if isinstance(ls, tuple) else ls).float().reshape(len(idx), -1)
            lt = (lt[0] if isinstance(lt, tuple) else lt).float().reshape(len(idx), -1)
            top1[j:j + len(idx)] = (ls.argmax(-1) == lt.argmax(-1)).double()
            y = labels[j:j + len(idx)].unsqueeze(-1)
            lps[j:j + len(idx)] = (ls.gather(-1, y).squeeze(-1) - torch.logsumexp(ls, -1)).double()
            lpt[j:j + len(idx)] = (lt.gather(-1, y).squeeze(-1) - torch.logsumexp(lt, -1)).double()
            del ls, lt
        vals = torch.stack([torch.ones_like(kl), kl, top1, lpt - lps, -lps, -lpt, (kl > 0.1).double(),
                            (kl > 1.0).double()], dim=1)  # [n, FIELDS]
        g = self._groups(meta['split'], len(self.names), dev)
        resp = torch.from_numpy(rpos[sel_np] >= 0).to(dev)  # the response; the rest is history
        ds = torch.from_numpy(meta['ds'][sel_np].astype(np.int64)).to(dev)
        seg = torch.from_numpy(meta['seg'][sel_np].astype(np.int64)).to(dev)
        g['seg'].index_add_(0, seg, vals)
        g['sup'] += vals.sum(0, keepdim=True)  # every loss position (response + history): what the loss averages
        if (~resp).any():
            g['hds'].index_add_(0, ds[~resp], vals[~resp])
        if resp.any():
            vr = vals[resp]
            g['all'] += vr.sum(0, keepdim=True)
            g['ds'].index_add_(0, ds[resp], vr)
            pos = torch.bucketize(torch.from_numpy(rpos[sel_np].astype(np.int64)).to(dev)[resp],
                                  torch.tensor(BUCKET_EDGES, device=dev), right=True)
            g['pos'].index_add_(0, pos, vr)
            self.maxkl[meta['split']] = max(self.maxkl[meta['split']], float(kl[resp].max()))
        d = os.environ.get('CL_DUMP_DIR')
        if d and 'tokens' in meta and meta['split'] in os.environ.get('CL_DUMP_SPLITS', 'val').split(','):
            os.makedirs(d, exist_ok=True)  # per-token log p for a consistency check with vLLM's log p
            np.savez_compressed(os.path.join(d, '%s_%d_%d.npz' % (meta['split'], meta['step'], meta['seq'])),
                                tokens=meta['tokens'], cu=meta['cu'], sel=sel_np, lps=lps.cpu().numpy(),
                                lpt=lpt.cpu().numpy(), kl=kl.cpu().numpy(), ds=meta['ds'][sel_np],
                                names=np.asarray(self.names, dtype='U64'))
        return True

    def flush(self, split):
        """Flat metrics {'kd/<split>/<group>/<field>': value}; empty on stages without statistics."""
        g = self.acc.pop(split, None)
        if g is None:
            return {}
        out = {}

        def put(prefix, row):
            row = row.tolist()
            n = row[0]
            if n <= 0:
                return
            out[prefix + '/tokens'] = n
            for f, v in zip(FIELDS[1:], row[1:]):
                out['%s/%s' % (prefix, f)] = v / n

        put('kd/%s/all' % split, g['all'][0])
        if g['sup'][0, 0] > g['all'][0, 0]:  # only with history supervision (otherwise identical to all)
            put('kd/%s/sup' % split, g['sup'][0])
        for i, name in enumerate(self.names or []):
            put('kd/%s/ds/%s' % (split, name), g['ds'][i])
        for i, name in enumerate(BUCKET_NAMES):
            put('kd/%s/pos/%s' % (split, name), g['pos'][i])
        for i, name in SEG_NAMES.items():
            put('kd/%s/seg/%s' % (split, name), g['seg'][i])
        for i, name in enumerate(self.names or []):
            put('kd/%s/hist_ds/%s' % (split, name), g['hds'][i])
        out['kd/%s/all/kl_max' % split] = self.maxkl.pop(split, 0.0)
        return out


STATS = KDStats()
