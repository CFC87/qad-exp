"""Keep CUDA graphs valid across in-place NVFP4 MoE weight updates in vLLM 0.29.

Without it the MoE rollout servers would need --enforce-eager: vLLM's native weight transfer reloads
layer by layer (initialize_layerwise_reload / finalize_layerwise_reload) and copies every re-processed PARAMETER and
BUFFER back into its original, graph-captured storage. But ModelOptNvFp4FusedMoE.process_weights_after_loading also
rebuilds `moe_quant_config` and `moe_kernel`, and the quant config holds the reciprocal activation scales
a1_gscale = 1 / w13_input_scale and a2_gscale = 1 / w2_input_scale as freshly allocated tensors that are not module
parameters. After an update the kernel object the graphs were captured with is dropped, its scale tensors are freed,
and the graphs read whatever reuses that memory.

Fix (a port of NeMo-RL's NemoModelOptNvFp4FusedMoE, nemo_rl/modelopt/models/generation/vllm_modelopt.py): on a reload
(the method already has a kernel), run the original processing, copy the two recomputed scales into the ORIGINAL
quant config's tensors (same shape required), and put the original quant config and kernel objects back. Everything
else the kernel reads is a registered parameter that the layerwise reload already copied in place.

Load it in every worker: put this file's directory on PYTHONPATH and pass `--worker-extension-cls vllm_moe_graph_patch.MoeGraphPatchExt`
(vLLM imports the module in each worker before the model is built; the import installs the patch).
"""
import logging

from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4FusedMoE

log = logging.getLogger(__name__)
_orig = ModelOptNvFp4FusedMoE.process_weights_after_loading
_stats = {'first_loads': 0, 'reloads': 0}


def _process_weights_after_loading(self, layer):
    old_kernel = getattr(self, 'moe_kernel', None)
    old_cfg = getattr(self, 'moe_quant_config', None)
    _orig(self, layer)
    if old_kernel is None:  # the initial load: nothing captured yet
        _stats['first_loads'] += 1
        return
    new_cfg = self.moe_quant_config
    if old_cfg is None or new_cfg is None:
        raise RuntimeError('NVFP4 MoE reload: quant config missing')
    for name in ('a1_gscale', 'a2_gscale'):
        old, new = getattr(old_cfg, name, None), getattr(new_cfg, name, None)
        if old is None or new is None:
            raise RuntimeError('NVFP4 MoE reload: %s missing' % name)
        if old.shape != new.shape or old.dtype != new.dtype:
            raise RuntimeError('NVFP4 MoE reload: %s changed shape/dtype %s -> %s' % (name, tuple(old.shape), tuple(new.shape)))
        old.copy_(new)
    self.moe_kernel, self.moe_quant_config = old_kernel, old_cfg
    _stats['reloads'] += 1


ModelOptNvFp4FusedMoE.process_weights_after_loading = _process_weights_after_loading


class MoeGraphPatchExt:
    """worker extension: importing this module installs the patch; moe_graph_patch_stats() reports it ran"""

    def moe_graph_patch_stats(self):
        return dict(_stats)
