"""Build the NVFP4 start checkpoint of a Nemotron QAD run ("-PTQ-NVFP4-nvscale") from the public BF16 model and NVIDIA's
NVFP4 release, so that every tensor QAD does not train equals the release:
  - the quantized modules are the release's (its hf_quant_config exclude list);
  - each quantized weight is quantized from BF16 with ModelOpt's NVFP4QTensor.quantize (block 16) at the release's
    weight_scale_2 (Nemotron 3 Nano: one per layer and projection for the routed experts, the max over the 128 experts);
  - input_scale and weight_scale_2 are the release's;
  - every other tensor (BF16 layers, norms, Mamba parameters, router, expert bias, embeddings) is the BF16 model's, in
    bf16 where the release stores bf16 and the BF16 model fp32 (a PTQ export's dtype; the values are unchanged);
  - config.json / hf_quant_config.json are this repository's copies (train/prep/ptqs/<model>/), the tokenizer and code
    files the BF16 model's.
Deterministic (no calibration: every static scale comes from the release). `--check MANIFEST` compares each tensor's
sha256 with the reference manifest (train/prep/ptqs/<model>/tensors.sha256) and exits non-zero on any difference.
Runs inside the NeMo image with ModelOpt (CPU is enough):
    python3 train/prep/build_ptqs.py --bf16 models/<BF16> --release models/<NVFP4 release> --configs train/prep/ptqs/<name> \
        --out models/<name> [--check train/prep/ptqs/<name>/tensors.sha256]
"""
import argparse, glob, hashlib, json, os, shutil, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def index(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(f, "pt") as st:
            for k in st.keys():
                out[k] = f
    return out


def get(idx, k):
    with safe_open(idx[k], "pt") as st:
        return st.get_tensor(k)


def digest(t):
    return hashlib.sha256(str(t.dtype).encode() + str(tuple(t.shape)).encode() + t.reshape(-1).contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--configs", required=True, help="dir with config.json and hf_quant_config.json for the output")
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", help="per-tensor sha256 manifest to compare with")
    ap.add_argument("--write-manifest", help="write the per-tensor sha256 manifest of the output here")
    ap.add_argument("--shard-gb", type=float, default=5.0)
    a = ap.parse_args()
    from modelopt.torch.quantization.qtensor.nvfp4_tensor import NVFP4QTensor
    t0 = time.time()
    B, R = index(a.bf16), index(a.release)
    quant = sorted(k[: -len(".weight_scale_2")] for k in R if k.endswith(".weight_scale_2"))
    qset = set(quant)
    os.makedirs(a.out, exist_ok=True)
    out, size, shard, files, wmap, digests = {}, 0, 0, [], {}, {}

    def flush():
        nonlocal out, size, shard
        if not out:
            return
        shard += 1
        name = "model-%05d.safetensors" % shard
        save_file(out, os.path.join(a.out, name), metadata={"format": "pt"})
        for k in out:
            wmap[k] = name
        files.append(name)
        out, size = {}, 0

    def put(k, t):
        nonlocal size
        out[k] = t.contiguous()
        digests[k] = digest(out[k])
        size += t.numel() * t.element_size()
        if size >= a.shard_gb * 2 ** 30:
            flush()

    n_q = 0
    for k in sorted(R):  # the release's tensor set and names
        mod = k.rsplit(".", 1)[0]
        if mod in qset:
            if k.endswith(".weight"):
                w = get(B, k)
                ws2 = get(R, mod + ".weight_scale_2").float()
                q, ws, _ = NVFP4QTensor.quantize(w, 16, None, ws2.reshape(()) if ws2.numel() == 1 else ws2)
                put(k, q._quantized_data)
                put(mod + ".weight_scale", ws)
                n_q += 1
            elif k.endswith((".weight_scale_2", ".input_scale")):
                put(k, get(R, k))
            elif k.endswith(".weight_scale"):
                pass  # made with the weight
            else:
                put(k, get(B, k))  # e.g. a bias of a quantized module
        else:  # the BF16 model's tensor (e.g. its bf16 router gate), in bf16 where the release stores bf16 and the BF16
            # model fp32 (Nemotron 3 Nano's Mamba A_log / D: the values are bf16-exact, a PTQ export writes them as bf16)
            t = get(B, k)
            with safe_open(R[k], "pt") as st:
                rdt = st.get_slice(k).get_dtype()
            put(k, t.to(torch.bfloat16) if t.dtype == torch.float32 and rdt == "BF16" else t)
    flush()
    json.dump({"metadata": {"total_size": sum(os.path.getsize(os.path.join(a.out, f)) for f in files)}, "weight_map": wmap},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=2)
    for f in os.listdir(a.bf16):  # tokenizer, chat template, code, generation config
        if not f.endswith(".safetensors") and f not in ("model.safetensors.index.json", "config.json") and os.path.isfile(os.path.join(a.bf16, f)):
            shutil.copy2(os.path.join(a.bf16, f), os.path.join(a.out, f))
    for f in ("config.json", "hf_quant_config.json"):
        shutil.copy2(os.path.join(a.configs, f), os.path.join(a.out, f))
    print("build_ptqs: %d tensors (%d quantized modules re-quantized from BF16) -> %s (%.0f s)" % (len(digests), n_q, a.out, time.time() - t0), flush=True)
    if a.write_manifest:
        open(a.write_manifest, "w").write("".join("%s  %s\n" % (digests[k], k) for k in sorted(digests)))
    if a.check:
        ref = dict(reversed(l.split()) for l in open(a.check) if l.strip())
        bad = sorted(k for k in set(ref) | set(digests) if ref.get(k) != digests.get(k))
        print("build_ptqs: %d / %d tensors match %s%s" % (len(ref) - len(bad), len(ref), a.check,
                                                           ("; differ: %s" % bad[:5]) if bad else ""), flush=True)
        if bad:
            sys.exit(1)


if __name__ == "__main__":
    main()
