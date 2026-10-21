"""Compare two HF NVFP4 checkpoints tensor by tensor (e.g. the step-0 QAD export against the evaluated NVIDIA checkpoint).

    .venv-client/bin/python train/tests/compare_nvfp4_ckpt.py A_DIR B_DIR

For every tensor name in both: exact equality (bitwise, after viewing as bytes; float tensors of another float dtype count
as equal when their values are, and are reported), and for float tensors the max absolute difference. Summary per
tensor kind (weight, weight_scale, weight_scale_2, input_scale, k_scale, v_scale, other) and the names only in one side.
"""
import glob, os, sys
from collections import defaultdict

import torch
from safetensors import safe_open


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, '*.safetensors'))):
        with safe_open(f, 'pt') as s:
            for k in s.keys():
                out[k] = (f, k)
    return out


def get(ref):
    f, k = ref
    with safe_open(f, 'pt') as s:
        return s.get_tensor(k)


def kind(k):
    for t in ('weight_scale_2', 'weight_scale', 'input_scale', 'k_scale', 'v_scale'):
        if k.endswith(t):
            return t
    return 'weight' if k.endswith('weight') else 'other'


def main():
    A, B = load(sys.argv[1]), load(sys.argv[2])
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    stat = defaultdict(lambda: dict(n=0, equal=0, maxdiff=0.0, worst=None))
    for k in sorted(set(A) & set(B)):
        a, b = get(A[k]), get(B[k])
        s = stat[kind(k)]; s['n'] += 1
        if a.shape == b.shape and a.dtype != b.dtype and a.is_floating_point() and b.is_floating_point() \
                and torch.equal(a.double(), b.double()):
            # the same values in another float dtype (Nemotron 3 Nano: the export writes the router gate in its fp32
            # moe_router_dtype, the PTQ start checkpoint stores it as bf16; the values are the bf16 ones)
            s['equal'] += 1; s.setdefault('dtype_only', 0); s['dtype_only'] += 1; continue
        if a.shape != b.shape or a.dtype != b.dtype:
            s['worst'] = (k, 'shape/dtype %s %s vs %s %s' % (tuple(a.shape), a.dtype, tuple(b.shape), b.dtype)); continue
        if torch.equal(a.reshape(-1).contiguous().view(torch.uint8), b.reshape(-1).contiguous().view(torch.uint8)):
            s['equal'] += 1; continue
        if a.is_floating_point():
            d = (a.float() - b.float()).abs().max().item()
            rel = d / max(b.float().abs().max().item(), 1e-30)
        else:
            d = (a.int() != b.int()).float().mean().item(); rel = d  # share of differing packed bytes
        if d > s['maxdiff']:
            s['maxdiff'], s['worst'] = d, (k, 'max diff %.3g (rel %.3g)' % (d, rel))
    for t, s in sorted(stat.items()):
        print('%-15s %4d tensors, %4d bitwise equal%s%s' % (t, s['n'], s['equal'],
              (' (%d of them equal in value, other dtype)' % s['dtype_only']) if s.get('dtype_only') else '',
              ('; worst %s: %s' % s['worst']) if s['worst'] else ''))
    for side, only in (('A', only_a), ('B', only_b)):  # counted by kind: only 5 names are printed
        by = defaultdict(int)
        for k in only:
            by[kind(k)] += 1
        print('only in %s: %d%s %s' % (side, len(only), (' (%s)' % ', '.join('%s %d' % kv for kv in sorted(by.items()))) if by else '',
                                       only[:5]))


if __name__ == '__main__':
    main()
