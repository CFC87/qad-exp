"""Run ModelOpt's examples/megatron_bridge/export_quantized_megatron_to_hf.py with one fix: the per-expert weight-quantizer
amax of TE grouped experts is loaded from the checkpoint.

Nemotron 3 Nano's routed experts are TEGroupedMLP: one GroupedQuantizer with a TensorQuantizer per expert. When the export
restores the ModelOpt state, only expert 0's quantizer gets an `_amax` buffer, so the checkpoint load has nowhere to put the
other experts' amax (their keys are in the checkpoint) and skips them, and the export then derives those experts'
weight_scale_2 from their own max|W| (unified_export_megatron._grouped_mlp_slicing). A QAD student that took NVIDIA's shared
weight_scale_2 at train start (qad_distill.py item 19) would be exported with other scales than it trained with.

Fix: before the weights are loaded, every enabled expert quantizer without `_amax` gets one shaped like its sibling's and
filled with NaN; the load fills it from the checkpoint; afterwards any NaN left (a key missing from the checkpoint) stops
the export. Models without grouped experts are unchanged (nothing is added).

Usage: same arguments as export_quantized_megatron_to_hf.py, run from examples/megatron_bridge (the container's working
directory), e.g. torchrun --nproc-per-node 1 /cl/export_hf.py --hf_model_name_or_path ... --megatron_path ...
"""
import os, runpy, sys

import torch
import modelopt.torch.utils.plugins.mbridge as _mb
from megatron.core.utils import unwrap_model
from modelopt.torch.quantization.nn import GroupedQuantizer

_orig_load = _mb._load_model_weights_from_checkpoint


def _load_with_expert_amax(checkpoint_path, model, *a, **k):
    added = []
    for chunk in (model if isinstance(model, (list, tuple)) else [model]):
        for name, mod in unwrap_model(chunk).named_modules():
            wq = getattr(mod, "weight_quantizer", None)
            if not isinstance(wq, GroupedQuantizer):
                continue
            ref = next((q for q in wq if getattr(q, "_amax", None) is not None), None)
            if ref is None:
                continue  # nothing calibrated in this module: leave it as ModelOpt has it
            for i, q in enumerate(wq):
                if getattr(q, "_amax", None) is None and getattr(q, "is_enabled", False):
                    q.register_buffer("_amax", torch.full_like(ref._amax, float("nan")))
                    added.append(("%s.weight_quantizer.%d" % (name, i), q))
    out = _orig_load(checkpoint_path, model, *a, **k)
    left = [n for n, q in added if torch.isnan(q._amax).any()]
    print("export_hf: %d per-expert weight amax restored from the checkpoint%s" % (
        len(added) - len(left), (", %d NOT in the checkpoint: %s" % (len(left), left[:4])) if left else ""), flush=True)
    if left:
        raise RuntimeError("export_hf: %d expert weight quantizers without an amax in %s" % (len(left), checkpoint_path))
    return out


_mb._load_model_weights_from_checkpoint = _load_with_expert_amax

if __name__ == "__main__":
    sys.path.insert(0, os.getcwd())
    script = os.path.join(os.getcwd(), "export_quantized_megatron_to_hf.py")
    sys.argv = [script] + sys.argv[1:]
    runpy.run_path(script, run_name="__main__")
