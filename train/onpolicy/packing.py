"""Pack rollouts into fixed-size THD micro-batches for Megatron-Bridge distillation.

A sample is (prompt token ids, response token ids): the prompt is the chat-template rendering the rollout was
sampled from, the response is exactly what the student generated (the stop token included when it stopped).
Samples are never split: every sample fits one pack on its own because max_tokens = min(32768, W - prompt).

OnlinePacker.add() places each finished rollout in the open pack with the least room that still fits it (best
fit); a pack is shipped to training as soon as it is at least `ship_at` full. A rollout that fits no open pack gets
a new pack as long as the step has fewer than `max_packs`, so everything generated in a step is trained in that step
(dropping the late, long rollouts and regenerating their prompts later would bias training towards short answers).
Only beyond max_packs is a rollout dropped (its prompt comes back in a later step). finish() ships the open packs
that hold anything; the controller completes the step with empty packs up to max_packs (the trainer's fixed number
of micro-batches); an empty pack is loss 0 and, with its padding cut into short segments, cheap.

to_batch() gives the Megatron-Bridge current packed format (training/gpt_step.py, utils/packed_seq_utils.py):
  tokens, labels, loss_mask, position_ids    [1, W]; labels[i] = tokens[i + 1] inside a sample
  cu_seqlens_q, cu_seqlens_kv                int32 offsets of every sample, then the padding (loss 0) as segments
                                             of at most PAD_SEG tokens, so its attention costs next to nothing
  max_seqlen_q, max_seqlen_kv                longest segment
loss_mask[i] = 1 exactly where the next token (labels[i]) is a response token, so the KD loss covers every
response position including the stop token, and never a prompt or padding position -- except the history spans of a
sample (meta['hist_spans'], token ranges [a, b) of the prompt: the assistant turns of the history, history supervision
of the v2 subsets), whose positions are loss positions too. position_ids restart at 0 in every sample (each sample is
its own sequence; YaRN / RoPE positions as at rollout time).

With ds_names given, the batch also carries per-position metadata for the trainer's statistics (never fed to the
model; the trainer collects them through feed_client.py): meta_ds (int16, index into meta_ds_names), meta_rpos
(int32, index of labels[i] within its response), meta_seg (int8, 1 = labels[i] is thinking up to and including
</think>, 2 = answer after it; a response without </think> is all 1; 3 = a history span, whose meta_rpos stays -1),
all -1 / 0 outside loss positions.
"""
from dataclasses import dataclass, field

import numpy as np

W_DEFAULT = 131072
PAD_ID_DEFAULT = 151643  # <|endoftext|> in the Qwen3 tokenizer; never trained on (loss 0). rollout.py passes the
                         # model's own pad id (Nemotron Nano V2: 12; must be < the vocabulary size)
PAD_SEG = 2048  # padding is cut into attention segments of at most this many tokens
THINK_END = 151668  # </think> in the Qwen3 tokenizer; a Sample's meta['think_end'] (rollout.py) takes precedence


@dataclass
class Sample:
    prompt: list
    response: list
    meta: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.prompt) + len(self.response)


@dataclass
class Pack:
    capacity: int
    samples: list = field(default_factory=list)
    used: int = 0

    def room(self):
        return self.capacity - self.used

    def add(self, s):
        self.samples.append(s)
        self.used += len(s)


class OnlinePacker:
    def __init__(self, n_packs, capacity=W_DEFAULT, ship_at=0.97, max_packs=None):
        self.n_packs, self.capacity, self.ship_at = n_packs, capacity, ship_at
        self.max_packs = max(max_packs or n_packs, n_packs)
        self.open = [Pack(capacity) for _ in range(n_packs)]
        self.shipped, self.dropped = [], []

    def add(self, s):
        """Place one finished rollout; returns the packs that became ready (to hand to training)."""
        L = len(s)
        if L > self.capacity:
            raise ValueError('sample of %d tokens exceeds the pack capacity %d' % (L, self.capacity))
        fits = [p for p in self.open if p.room() >= L]
        if not fits:
            if len(self.shipped) + len(self.open) >= self.max_packs:
                self.dropped.append(s)
                return []
            self.open.append(Pack(self.capacity))
            fits = self.open[-1:]
        best = min(fits, key=lambda p: p.room())
        best.add(s)
        if best.used >= self.ship_at * self.capacity:
            self.open.remove(best)
            self.shipped.append(best)
            return [best]
        return []

    def finish(self):
        """End of step: ship the open packs that hold anything; returns them (the caller pads with empty packs)."""
        rest = [p for p in self.open if p.samples]
        self.open = []
        self.shipped += rest
        return rest

    def stats(self):
        used = sum(p.used for p in self.shipped + self.open)
        n = len(self.shipped) + len([p for p in self.open if p.samples])
        return dict(packs=n, fill=used / (max(n, 1) * self.capacity),
                    dropped_samples=len(self.dropped), dropped_tokens=sum(len(s) for s in self.dropped))


def to_batch(pack, pad_id=PAD_ID_DEFAULT, ds_names=None):
    W = pack.capacity
    tokens = np.full(W, pad_id, np.int64)
    labels = np.full(W, pad_id, np.int64)
    loss_mask = np.zeros(W, np.float32)
    position_ids = np.zeros(W, np.int64)
    if ds_names is not None:
        ds_idx = {d: i for i, d in enumerate(ds_names)}
        m_ds = np.full(W, -1, np.int16); m_rpos = np.full(W, -1, np.int32); m_seg = np.zeros(W, np.int8)
    cu = [0]
    o = 0
    for s in pack.samples:
        ids = np.asarray(list(s.prompt) + list(s.response), np.int64)
        n, p = len(ids), len(s.prompt)
        tokens[o:o + n] = ids
        labels[o:o + n - 1] = ids[1:]
        loss_mask[o + p - 1:o + n - 1] = 1.0  # positions whose next token is a response token
        position_ids[o:o + n] = np.arange(n)
        if ds_names is not None:
            r = n - p
            m_ds[o + p - 1:o + n - 1] = ds_idx[s.meta.get('dataset')]
            m_rpos[o + p - 1:o + n - 1] = np.arange(r)
            if 'think_end' in s.meta:  # index of the last token of </think> in the response, or None
                te = [] if s.meta['think_end'] is None else [s.meta['think_end']]
            else:
                te = np.flatnonzero(np.asarray(s.response) == THINK_END)
            seg = np.ones(r, np.int8)
            if len(te):
                seg[te[0] + 1:] = 2  # after </think>: the answer
            m_seg[o + p - 1:o + n - 1] = seg
        for a, b in s.meta.get('hist_spans') or ():  # history supervision: positions whose next token is in [a, b)
            assert 0 < a < b <= p, (a, b, p)
            loss_mask[o + a - 1:o + b - 1] = 1.0
            if ds_names is not None:
                m_ds[o + a - 1:o + b - 1] = ds_idx[s.meta.get('dataset')]
                m_seg[o + a - 1:o + b - 1] = 3  # meta_rpos stays -1: not a response position
        o += n
        cu.append(o)
    while o < W:  # padding: short segments of their own (loss 0), so an empty pack's attention is cheap
        e = min(o + PAD_SEG, W)
        position_ids[o:e] = np.arange(e - o)
        cu.append(e)
        o = e
    seg = np.diff(cu)
    out = dict(tokens=tokens[None], labels=labels[None], loss_mask=loss_mask[None], position_ids=position_ids[None],
               cu_seqlens_q=np.asarray(cu, np.int32), cu_seqlens_kv=np.asarray(cu, np.int32),
               max_seqlen_q=int(seg.max()), max_seqlen_kv=int(seg.max()))
    if ds_names is not None:
        out.update(meta_ds=m_ds, meta_rpos=m_rpos, meta_seg=m_seg, meta_ds_names=np.asarray(list(ds_names), dtype='U64'))
    return out
