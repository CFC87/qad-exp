"""Compare two HF NVFP4 checkpoints tensor by tensor (e.g. the step-0 QAD export against the evaluated NVIDIA checkpoint).

    .venv-client/bin/python train/tests/compare_nvfp4_ckpt.py A_DIR B_DIR [--expert-input-max] [--scale-rtol R --max-share S]

For every tensor name in both: exact equality (bitwise, after viewing as bytes; float tensors of another float dtype count
as equal when their values are, and are reported). Summary per tensor kind (weight, weight_scale, weight_scale_2,
input_scale, k_scale, v_scale, other), and for the tensors that differ:
  - scales and other float tensors: the largest relative difference of an element, |a - b| / |b|;
  - weight_scale (FP8 block scales): the share of block scales that differ;
  - weight (two FP4 values per byte): the share of FP4 values that differ.
Then the names only in one side, counted by kind.

--expert-input-max: B's routed-expert input scales are replaced by the largest one of their layer and projection group
(gate_proj and up_proj together, down_proj alone) before comparing. vLLM's MoE kernels serve one input scale per layer and
projection, so a QAD student takes that maximum where the checkpoint calibrated every expert on its own (Qwen3-30B-A3B).

--scale-rtol R --max-share S: print a verdict. PASS when every scale tensor (weight_scale_2, input_scale, k_scale, v_scale)
is within relative difference R, the shares of differing block scales and FP4 values are at most S, no other tensor
differs, and only k_scale / v_scale are on one side only. Without these flags the verdict asks for bitwise equality.
"""
import argparse, glob, os, re
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


SCALES = ('weight_scale_2', 'input_scale', 'k_scale', 'v_scale')
EXPERT_INPUT = re.compile(r'^(.*\.experts)\.\d+\.(gate_proj|up_proj|down_proj)\.input_scale$')


def expert_input_max(B):
    """name -> the max input scale of its layer's experts for that projection group (gate and up together)."""
    groups = defaultdict(list)
    for k in B:
        m = EXPERT_INPUT.match(k)
        if m:
            groups[(m.group(1), 'down' if m.group(2) == 'down_proj' else 'gate_up')].append(k)
    out = {}
    for ks in groups.values():
        mx = max((get(B[k]) for k in ks), key=lambda t: t.float().max().item())
        for k in ks:
            out[k] = mx
    return out


def fp4_values(t):
    """uint8 tensor of packed E2M1 pairs -> the 4-bit codes (low nibble, high nibble)."""
    t = t.reshape(-1)
    return torch.stack([t & 0xF, t >> 4])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a'); ap.add_argument('b')
    ap.add_argument('--expert-input-max', action='store_true')
    ap.add_argument('--scale-rtol', type=float)
    ap.add_argument('--max-share', type=float)
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    ref = expert_input_max(B) if args.expert_input_max else {}
    only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
    stat = defaultdict(lambda: dict(n=0, equal=0, maxrel=0.0, worst=None, diff_el=0, el=0, other=[]))
    for k in sorted(set(A) & set(B)):
        a, b = get(A[k]), (ref[k] if k in ref else get(B[k]))
        t = kind(k)
        s = stat[t]; s['n'] += 1
        if a.shape == b.shape and a.dtype != b.dtype and a.is_floating_point() and b.is_floating_point() \
                and torch.equal(a.double(), b.double()):
            # the same values in another float dtype (Nemotron 3 Nano: the export writes the router gate in its fp32
            # moe_router_dtype, the PTQ start checkpoint stores it as bf16; the values are the bf16 ones)
            s['equal'] += 1; s.setdefault('dtype_only', 0); s['dtype_only'] += 1; continue
        if a.shape != b.shape or a.dtype != b.dtype:
            s['other'].append(k); s['worst'] = (k, 'shape/dtype %s %s vs %s %s' % (tuple(a.shape), a.dtype, tuple(b.shape), b.dtype))
            s['maxrel'] = float('inf'); continue
        if torch.equal(a.reshape(-1).contiguous().view(torch.uint8), b.reshape(-1).contiguous().view(torch.uint8)):
            s['equal'] += 1
            if t == 'weight' and a.dtype == torch.uint8:
                s['el'] += 2 * a.numel()
            elif t == 'weight_scale':
                s['el'] += a.numel()
            continue
        if t == 'weight' and a.dtype == torch.uint8:  # packed FP4: the share of differing 4-bit values
            va, vb = fp4_values(a), fp4_values(b)
            nd, ne = int((va != vb).sum()), va.numel()
            s['diff_el'] += nd; s['el'] += ne
            share = nd / ne
            if share > s['maxrel']:
                s['maxrel'], s['worst'] = share, (k, '%.3g of its FP4 values differ' % share)
            continue
        if t == 'weight_scale':  # FP8 block scales: the share of differing scales and the largest relative change
            d = a.reshape(-1).view(torch.uint8) != b.reshape(-1).view(torch.uint8)
            s['diff_el'] += int(d.sum()); s['el'] += d.numel()
            af, bf = a.float().reshape(-1)[d], b.float().reshape(-1)[d]
            rel = ((af - bf).abs() / bf.abs().clamp_min(1e-30)).max().item()
            if rel > s['maxrel']:
                s['maxrel'], s['worst'] = rel, (k, '%d of %d block scales differ, largest relative change %.3g' % (int(d.sum()), d.numel(), rel))
            continue
        if a.is_floating_point():
            rel = ((a.float() - b.float()).abs() / b.float().abs().clamp_min(1e-30)).max().item()
            if t not in SCALES:
                s['other'].append(k)
            if rel > s['maxrel']:
                s['maxrel'], s['worst'] = rel, (k, 'largest relative difference %.3g' % rel)
        else:
            s['other'].append(k); s['maxrel'] = float('inf'); s['worst'] = (k, 'integer tensor differs')
    for t, s in sorted(stat.items()):
        extra = ''
        if s['n'] > s['equal']:
            if t == 'weight':
                extra = '; FP4 values differing: %d of %d (%.2g)' % (s['diff_el'], s['el'], s['diff_el'] / max(s['el'], 1))
            elif t == 'weight_scale':
                extra = '; block scales differing: %d of %d (%.2g)' % (s['diff_el'], s['el'], s['diff_el'] / max(s['el'], 1))
            extra += '; worst %s: %s' % s['worst']
        print('%-15s %5d tensors, %5d bitwise equal%s%s' % (t, s['n'], s['equal'],
              (' (%d of them equal in value, other dtype)' % s['dtype_only']) if s.get('dtype_only') else '', extra))
    if ref:
        print('input_scale reference: %d routed-expert input scales of B taken as their layer and projection maximum' % len(ref))
    for side, only in (('A', only_a), ('B', only_b)):  # counted by kind: only 5 names are printed
        by = defaultdict(int)
        for k in only:
            by[kind(k)] += 1
        print('only in %s: %d%s %s' % (side, len(only), (' (%s)' % ', '.join('%s %d' % kv for kv in sorted(by.items()))) if by else '',
                                       only[:5]))
    # verdict
    bad = [k for k in only_a + only_b if kind(k) not in ('k_scale', 'v_scale')]
    for t, s in stat.items():
        bad += s['other']
        if args.scale_rtol is None:
            if s['n'] > s['equal']:
                bad.append('%s: %d tensors not bitwise equal' % (t, s['n'] - s['equal']))
        elif t in SCALES and s['maxrel'] > args.scale_rtol:
            bad.append('%s: relative difference %.3g > %.3g' % (t, s['maxrel'], args.scale_rtol))
        elif t in ('weight', 'weight_scale') and s['diff_el'] / max(s['el'], 1) > (args.max_share or 0.0):
            bad.append('%s: share %.3g > %.3g' % (t, s['diff_el'] / max(s['el'], 1), args.max_share or 0.0))
    crit = 'bitwise' if args.scale_rtol is None else 'scales within %.3g, shares at most %.3g' % (args.scale_rtol, args.max_share or 0.0)
    print('VERDICT %s (%s)%s' % ('PASS' if not bad else 'FAIL', crit, ('; ' + '; '.join(map(str, bad[:5]))) if bad else ''))


if __name__ == '__main__':
    main()
