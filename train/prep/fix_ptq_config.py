"""Check an exported Nemotron NVFP4 checkpoint against NVIDIA's release layer map and write the release's exclude list.

    python fix_ptq_config.py EXPORT_DIR RELEASE_EXCLUDE_JSON KV_ALGO|none

1. Weights (the ground truth): every linear module the release keeps in BF16 must have no `weight_scale` in the export,
   and every other linear module must have one; quant_algo NVFP4, group 16, KV algo as given.
2. If they agree, the exclude list is rewritten to the release's (hf_quant_config.json exclude_modules and config.json
   quantization_config.ignore; originals kept as *.modelopt_orig). ModelOpt 0.47.0rc2 writes it for NemotronH with
   an unstripped name-mapping sentinel and merges fully-excluded layers into their parent ("backbone.layers.0"); the
   weights are right, and vLLM should read the same explicit list the release carries.
Prints one line: PTQ_CHECK OK|MISMATCH ... (exit 1 on MISMATCH, files untouched).
"""
import glob, json, os, shutil, struct, sys


def main():
    d, ref_path, kv = sys.argv[1], sys.argv[2], sys.argv[3]
    kv = None if kv == 'none' else kv
    ref = json.load(open(ref_path))
    keys = set()
    for f in glob.glob(d + '/*.safetensors'):
        with open(f, 'rb') as fh:
            n = struct.unpack('<Q', fh.read(8))[0]
            keys |= set(json.loads(fh.read(n)))
    lin = {k[:-len('.weight')] for k in keys if k.endswith('.weight') and ('proj' in k or 'lm_head' in k)}
    quant = {k[:-len('.weight_scale')] for k in keys if k.endswith('.weight_scale')}
    ref_lin = set(ref) & lin
    bad_q = sorted(ref_lin & quant)            # the release keeps it BF16, ours quantized it
    bad_b = sorted((lin - set(ref)) - quant)   # the release quantized it, ours kept it BF16
    hq = json.load(open(d + '/hf_quant_config.json'))
    q = hq.get('quantization', hq)
    ok = not bad_q and not bad_b and q.get('quant_algo') == 'NVFP4' and q.get('group_size') == 16 and q.get('kv_cache_quant_algo') == kv
    print('PTQ_CHECK', 'OK' if ok else 'MISMATCH', f'linear {len(lin)} nvfp4 {len(quant)} release_bf16_linear {len(ref_lin)}',
          f'kv {q.get("kv_cache_quant_algo")}', 'quantized_but_release_bf16', bad_q[:5], len(bad_q),
          'bf16_but_release_quantized', bad_b[:5], len(bad_b), flush=True)
    if not ok:
        sys.exit(1)
    for name in ('hf_quant_config.json', 'config.json'):
        if not os.path.exists(f'{d}/{name}.modelopt_orig'):
            shutil.copy(f'{d}/{name}', f'{d}/{name}.modelopt_orig')
    q['exclude_modules'] = list(ref)
    json.dump(hq, open(d + '/hf_quant_config.json', 'w'), indent=4)
    c = json.load(open(d + '/config.json'))
    qc = c.get('quantization_config') or {}
    for k in ('ignore', 'exclude_modules'):
        if k in qc:
            qc[k] = list(ref)
    json.dump(c, open(d + '/config.json', 'w'), indent=2)
    print('PTQ_CHECK exclude list set to the release list', len(ref), flush=True)


if __name__ == '__main__':
    main()
