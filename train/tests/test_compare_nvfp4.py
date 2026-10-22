"""CPU test of compare_nvfp4_ckpt.py on small synthetic NVFP4 checkpoints: bitwise verdicts, the rounding verdict
(--scale-rtol / --max-share) and the routed experts' input-scale maximum (--expert-input-max)."""
import os, subprocess, sys, tempfile

import torch
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))


def ckpt(d, t):
    os.makedirs(d, exist_ok=True)
    save_file({k: v.contiguous() for k, v in t.items()}, os.path.join(d, 'model.safetensors'))


def run(a, b, *flags):
    out = subprocess.run([sys.executable, os.path.join(HERE, 'compare_nvfp4_ckpt.py'), a, b, *flags],
                         capture_output=True, text=True, check=True).stdout
    return out, out.strip().splitlines()[-1]


def main():
    g = torch.Generator().manual_seed(0)
    base = {}
    for e in range(4):
        p = 'model.layers.0.mlp.experts.%d.down_proj.' % e
        base[p + 'weight'] = torch.randint(0, 256, (64, 32), dtype=torch.uint8, generator=g)
        base[p + 'weight_scale'] = (torch.rand(64, 4, generator=g) * 400 + 1).to(torch.float8_e4m3fn)
        base[p + 'weight_scale_2'] = torch.tensor(1e-4 * (e + 1), dtype=torch.float32)
        base[p + 'input_scale'] = torch.tensor(0.01 * (e + 1), dtype=torch.float32)
    base['model.layers.0.self_attn.k_proj.k_scale'] = torch.tensor(1.0)
    with tempfile.TemporaryDirectory() as td:
        A, B = os.path.join(td, 'a'), os.path.join(td, 'b')
        ckpt(B, base)
        # 1. identical (but for a one-sided k_scale): bitwise PASS
        ckpt(A, {k: v for k, v in base.items() if not k.endswith('k_scale')})
        out, v = run(A, B)
        assert v.startswith('VERDICT PASS (bitwise)'), out
        assert 'only in B: 1 (k_scale 1)' in out, out
        # 2. one weight_scale_2 one fp32 ulp off, one FP4 nibble and one FP8 block scale changed
        t = {k: v.clone() for k, v in base.items() if not k.endswith('k_scale')}
        k2 = 'model.layers.0.mlp.experts.1.down_proj.weight_scale_2'
        t[k2] = torch.nextafter(t[k2], torch.tensor(1.0))
        kw = 'model.layers.0.mlp.experts.2.down_proj.weight'
        t[kw][0, 0] ^= 0x01
        ks = 'model.layers.0.mlp.experts.3.down_proj.weight_scale'
        t[ks].view(torch.uint8)[0, 0] += 1
        ckpt(A, t)
        out, v = run(A, B)
        assert v.startswith('VERDICT FAIL (bitwise)'), out
        out, v = run(A, B, '--scale-rtol', '2.4e-7', '--max-share', '0.002')
        assert v.startswith('VERDICT PASS'), out
        assert 'FP4 values differing: 1 of 16384' in out and 'block scales differing: 1 of 1024' in out, out
        out, v = run(A, B, '--scale-rtol', '1e-9', '--max-share', '0.002')
        assert v.startswith('VERDICT FAIL') and 'weight_scale_2: relative difference' in v, out
        out, v = run(A, B, '--scale-rtol', '2.4e-7', '--max-share', '1e-5')
        assert v.startswith('VERDICT FAIL') and 'share' in v, out
        # 3. a student whose experts all carry the layer's largest input scale
        t = {k: v.clone() for k, v in base.items() if not k.endswith('k_scale')}
        for e in range(4):
            t['model.layers.0.mlp.experts.%d.down_proj.input_scale' % e] = torch.tensor(0.04, dtype=torch.float32)
        ckpt(A, t)
        out, v = run(A, B)
        assert v.startswith('VERDICT FAIL') and 'input_scale: 3 tensors not bitwise equal' in v, out
        out, v = run(A, B, '--expert-input-max')
        assert v.startswith('VERDICT PASS (bitwise)'), out
        assert '4 routed-expert input scales of B taken as their layer and projection maximum' in out, out
        # 4. any other tensor that differs fails whatever the tolerances
        t = {k: v.clone() for k, v in base.items() if not k.endswith('k_scale')}
        t['model.norm.weight'] = torch.ones(8, dtype=torch.bfloat16)
        b2 = dict(base); b2['model.norm.weight'] = torch.full((8,), 2.0, dtype=torch.bfloat16)
        ckpt(A, t); ckpt(B, b2)
        out, v = run(A, B, '--scale-rtol', '1.0', '--max-share', '1.0')
        assert v.startswith('VERDICT FAIL') and 'model.norm.weight' in v, out
    print('compare_nvfp4_ckpt test passed')


if __name__ == '__main__':
    main()
