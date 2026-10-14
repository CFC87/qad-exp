"""Trainer side of the on-policy pack feed (onpolicy/rollout.py). Imported by qad_distill.py when
CL_ONPOLICY_FEED=http://<rollout host>:<port> is set.

FeedIterator yields one Megatron-Bridge packed micro-batch per next(): pack (step, seq) in order, seq = 0..n_packs-1,
then the next step. The feed blocks until a pack is published, so the trainer simply waits for rollouts; every
pipeline stage runs its own iterator over the same sequence and gets identical inputs. Keys and dtypes follow
training/gpt_step.get_batch_from_iterator: tokens / labels / position_ids int64, loss_mask float32, all [1, W];
cu_seqlens_q / cu_seqlens_kv int32 (device keys); max_seqlen_q / max_seqlen_kv 0-d tensors (host keys);
attention_mask None (THD attention is causal within each cu_seqlens segment).

post_step_done(url, step) tells the controller the optimizer step is done (and, with weight push, that the new weights
are already in vLLM), so it can start generating the next step.
"""
import collections, io, os, time, urllib.request

import numpy as np
import torch


# Per-position metadata of the packs handed out (packing.to_batch meta_*), oldest first, for the trainer's statistics
# (qad_distill.py): the last pipeline stage takes one entry per micro-batch in its loss; other stages never read
# it, hence the bound.
META = collections.deque(maxlen=512)
_WAIT = [0.0]  # seconds spent waiting for packs since the last take_wait()


def take_wait():
    w, _WAIT[0] = _WAIT[0], 0.0
    return w


class FeedIterator:
    """route 'pack' (training) or 'val' (validation rollouts); split names the statistics bucket."""

    def __init__(self, url, n_packs, start_step=0, log=print, route='pack', split='train'):
        self.url, self.n, self.k, self.i, self.log = url.rstrip('/'), n_packs, start_step, 0, log
        self.route, self.split = route, split

    def __iter__(self):
        return self

    def __next__(self):
        u = '%s/%s/%d/%d' % (self.url, self.route, self.k, self.i)
        t0, attempt = time.time(), 0
        while True:
            try:
                data = urllib.request.urlopen(u, timeout=24 * 3600).read()
                break
            except Exception as e:  # controller restarting, network hiccup: keep waiting
                attempt += 1
                if attempt % 6 == 1:
                    self.log('feed: waiting for %s (%r)' % (u, e))
                time.sleep(10)
        z = np.load(io.BytesIO(data))
        b = {k: torch.from_numpy(np.ascontiguousarray(z[k])) for k in ('tokens', 'labels', 'loss_mask', 'position_ids')}
        b['loss_mask'] = b['loss_mask'].float()
        for k in ('cu_seqlens_q', 'cu_seqlens_kv'):
            b[k] = torch.from_numpy(z[k].astype(np.int32))
        for k in ('max_seqlen_q', 'max_seqlen_kv'):
            b[k] = torch.tensor(int(z[k]), dtype=torch.int32)
        b['attention_mask'] = None
        if 'meta_ds' in z.files:
            META.append(dict(split=self.split, step=self.k, seq=self.i, ds=z['meta_ds'], rpos=z['meta_rpos'],
                             seg=z['meta_seg'], names=[str(x) for x in z['meta_ds_names']], labels=z['labels'][0]))
            if os.environ.get('CL_DUMP_DIR'):  # kd_stats dumps per-token log p with the pack (consistency check with vLLM)
                META[-1].update(tokens=z['tokens'][0], cu=z['cu_seqlens_q'])
        waited = time.time() - t0
        _WAIT[0] += waited
        if waited > 60:
            self.log('feed: step %d pack %d arrived after %.0f s' % (self.k, self.i, waited))
        self.i += 1
        if self.i == self.n:
            self.i, self.k = 0, self.k + 1
        return b


def post_step_done(url, step):
    req = urllib.request.Request('%s/step_done/%d' % (url.rstrip('/'), step), data=b'', method='POST')
    for attempt in range(60):
        try:
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            time.sleep(10)
    raise RuntimeError('could not report step %d done to %s' % (step, url))
