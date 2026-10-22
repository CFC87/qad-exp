"""Run ModelOpt's examples/megatron_bridge/distill.py with the patches below (multi-node pipeline parallelism, long
contexts, on-policy training).

1. Every pipeline stage gets tokens and position_ids. Megatron-Bridge 0.6.1 (gpt_step.get_batch) gives them
   only to the first stage and to a stage that holds MTP layers; middle stages get no batch at all unless
   a custom attention mask or packed sequences are used. Qwen3.5-architecture models build their multimodal
   RoPE from position_ids on every stage, so a stage without them fails with "'NoneType' object has no
   attribute 'ndim'" in qwen_vl/modelling_qwen3_vl/rope.py (e.g. a PP=2 MoE model without MTP). position_ids
   are one integer per token.
2. Provider overrides from the CL_PROVIDER_OVERRIDES environment variable (JSON), applied to both the
   student and the teacher provider, e.g. uneven stages for PP=6:
   CL_PROVIDER_OVERRIDES='{"num_layers_in_first_pipeline_stage": 6, "num_layers_in_last_pipeline_stage": 6}'
   distill.py has no flags for these.

3. CL_NO_SAVE=1 skips checkpoint saving (for benchmarks; distill.py always saves at the end).
4. CL_CHUNKED_KD=<tokens> computes the logit KD loss in sequence chunks (see below) for long contexts.
5. GPT datasets do not build the unused [seq, seq] attention mask (always on; see below).
6. CL_NO_OPTIM_CKPT=1 saves/loads checkpoints without optimizer state (disk-limited nodes). CL_CKPT_LATEST_FULL=1
   instead keeps weights-only checkpoints plus one full checkpoint (with optimizer state) of the newest save.
7. CL_TIMING=1|2 prints Megatron timers for every rank (compute vs pipeline waits).
8. CL_GRAD_REPORT=N prints each stage's gradient norm and its largest-gradient parameters for N steps.
9. CL_KD_REVERSE=w (0..1) mixes in the reverse KL: loss = (1-w) KL(teacher||student) + w KL(student||teacher);
   distill.py alone always uses the forward KL. Full-vocabulary loss only (no --logit_kl_topk).
10. CL_ONPOLICY_FEED=http://<rollout host>:<port> trains on the on-policy packs of onpolicy/rollout.py
   (blocking per-pack reads, step-done report after every optimizer step; CL_DIST_TIMEOUT_MIN, default 240).
   With CL_WEIGHT_PUSH_URL (+ CL_WEIGHT_PUSH_HF, _SENDER, _PORT, _BUFFER_GB) the new weights are pushed to the rollout
   vLLM server(s) over NCCL before each step is reported done (onpolicy/weight_push.py).
11. CL_OPTIMIZER_OVERRIDES (JSON) sets OptimizerConfig fields, e.g. bf16 Adam states or optimizer CPU offload. Offload
   keeping fp32 numerics (30B class on PP=4 at 128K): '{"use_precision_aware_optimizer": true, "optimizer_cpu_offload":
   true, "optimizer_offload_fraction": 0.5, "overlap_cpu_optimizer_d2h_h2d": true}'; with it, resuming keeps the fp32
   masters and the Adam step (two fixes, see below), and the first steps print optimizer time and host memory.
12. CL_IMPORT_INPUT_SCALE=<HF NVFP4 checkpoint dir> sets the student's activation quantizers to that checkpoint's static
   input scales (amax = input_scale * 6 * 448) once the student is loaded, before the first step, so step 0 is the
   evaluated checkpoint rather than our own calibration (dense Qwen3: linear_qkv <- q_proj, linear_proj <- o_proj,
   linear_fc1 <- gate_proj, linear_fc2 <- down_proj; the fused inputs have equal scales in the checkpoint, checked;
   NemotronH: Mamba mixer.in_proj / out_proj, MLP linear_fc1 <- up_proj, linear_fc2 <- down_proj, every scale of a
   stage's layers must land on an enabled quantizer).
13. CL_SCHEDULER_OVERRIDES (JSON) sets SchedulerConfig fields (or OptimizerConfig ones such as weight_decay), e.g. WSD.
14. CL_KEEP_FULL_STEPS=96[,..] (with CL_CKPT_LATEST_FULL=1) also saves at these steps and keeps their full checkpoints.
15. CL_DDP_OVERRIDES (JSON) sets DistributedDataParallelConfig fields (debugging, e.g. check_for_nan_in_grad false).
16. CL_MOCK_VOCAB=N limits mock-data token ids to 1..N (tests on models with untrained embedding rows, Nano V2).
17. Mamba2 models: the fused training kernel gets contiguous inputs (mamba_compact.py; 32-bit offset overflow past
   ~52K tokens per micro-batch otherwise). CL_MAMBA_COMPACT=0 disables it.
18. CL_PRETRAINED=<checkpoint dir> starts a new run from a trained checkpoint (weights only; step 0, new optimizer).
19. NemotronH MoE (Nemotron 3 Nano) with CL_IMPORT_INPUT_SCALE: the routed experts' one input quantizer <- the 128
   experts' input_scale (all equal in the checkpoint), their 128 weight quantizers <- the checkpoint's weight_scale_2
   (amax = weight_scale_2 * 6 * 448; NVIDIA's one shared value per layer and projection, max over the experts -- a
   quantize.py PTQ calibrates every expert on its own max|W_e|), shared experts as dense MLPs. Every training launch
   (start, resume, a run from a trained checkpoint) imports them again at train start. ModelOpt's restore gives only expert
   0 of a grouped linear an amax buffer, so a plain export would drop the others' saved amax and derive their
   weight_scale_2 from their own max|W|: exports go through export_hf.py, which loads them from the checkpoint.
20. CL_MOE_PAD_ID=<token id>: MoE models get a padding_mask in their forward that covers each pack's padding tail (the
   trailing run of the rollout's pad id, from where position_ids restart at 0 inside it), so the pack padding neither
   counts for the router expert-bias update nor enters the auxiliary load-balancing loss (Megatron-Bridge passes no
   padding_mask; the packs' padding tails and empty packs would otherwise drive the bias of the experts the pad token
   routes to). The pad id also occurs inside samples (Nemotron 3 Nano's is <|im_end|>, 11, which ends every message):
   those tokens are real and stay in.

Usage: same arguments as distill.py, run from examples/megatron_bridge (the container's working directory).
"""
import json, os, runpy, sys

import megatron.bridge.training.gpt_step as gpt_step
from megatron.bridge import AutoBridge

_orig_from_iterator = gpt_step.get_batch_from_iterator


def _from_iterator_with_positions(data_iterator, include_mtp_inputs=False, *args, **kwargs):
    # include_mtp_inputs=True makes the stage load tokens and position_ids (and nothing it would not use)
    return _orig_from_iterator(data_iterator, True, *args, **kwargs)


gpt_step.get_batch_from_iterator = _from_iterator_with_positions

_MOE_PAD = os.environ.get("CL_MOE_PAD_ID")
if _MOE_PAD:  # 20.: pack padding out of the MoE router statistics
    import inspect as _inspect
    _MOE_PAD = int(_MOE_PAD)

    def _pack_padding(ids, pos):
        # the pack's padding is its tail after the last sample (packing.to_batch): all pad id, position_ids restarting at
        # 0. The pad id also occurs inside samples (Nemotron 3 Nano's is <|im_end|>, which ends every message; Qwen3's
        # <|endoftext|> can end a response): those tokens are real. So the padding starts at the first position 0 inside
        # the trailing run of pad ids (a sample never starts with the pad id); no position_ids: the whole trailing run
        S = ids.shape[-1]
        idx = _torch.arange(S, device=ids.device).expand_as(ids)
        run = idx > _torch.where(ids.ne(_MOE_PAD), idx, _torch.full_like(idx, -1)).max(dim=-1, keepdim=True).values
        if pos is None:
            return run
        start = _torch.where(run & pos.eq(0), idx, _torch.full_like(idx, S)).min(dim=-1, keepdim=True).values
        return idx >= start

    def _with_padding_mask(cls):
        orig = cls.forward
        params = list(_inspect.signature(orig).parameters)
        assert "padding_mask" in params and params[1:3] == ["input_ids", "position_ids"], "%s.forward: %s" % (cls.__name__, params)

        def forward(self, *args, **kwargs):
            if kwargs.get("padding_mask") is None and getattr(self.config, "num_moe_experts", None):
                ids = kwargs.get("input_ids", args[0] if args else None)
                pos = kwargs.get("position_ids", args[1] if len(args) > 1 else None)
                if ids is not None:
                    kwargs["padding_mask"] = _pack_padding(ids, pos)
            return orig(self, *args, **kwargs)

        cls.forward = forward

    import torch as _torch
    from megatron.core.transformer.moe import router as _router

    def _apply_expert_bias(self, routing_map, padding_mask=None):
        # MCore's own (router.py TopKRouter._apply_expert_bias) masks [num_tokens, num_experts] with the flattened
        # [num_tokens] padding mask, which does not broadcast: the first backward of a model with expert bias would fail
        if self.enable_expert_bias and _torch.is_grad_enabled():
            with _torch.no_grad():
                if padding_mask is not None:
                    routing_map = routing_map & (~padding_mask.reshape(-1, 1))
                self.local_tokens_per_expert += routing_map.sum(dim=0)

    _router.TopKRouter._apply_expert_bias = _apply_expert_bias
    _orig_token_count = _router.get_tokens_per_expert_and_token_count

    def _token_count(routing_map, reduce_group, topk=None, with_padding_mask=False):
        # an all-padding micro-batch (the empty packs that fill a step) has 0 valid tokens: the load-balancing loss would
        # be 0 * inf = NaN (and check_for_nan_in_grad stops the run); with at least 1 it is exactly 0
        g, l, t = _orig_token_count(routing_map, reduce_group, topk=topk, with_padding_mask=with_padding_mask)
        if with_padding_mask:
            l, t = _torch.clamp(l, min=1), _torch.clamp(t, min=1)
        return g, l, t

    _router.get_tokens_per_expert_and_token_count = _token_count  # router.py imports it by name
    from megatron.core.models.gpt.gpt_model import GPTModel as _GPTModel
    _with_padding_mask(_GPTModel)
    try:
        from megatron.core.models.hybrid.hybrid_model import HybridModel as _HybridModel
        _with_padding_mask(_HybridModel)
    except ImportError:
        pass
gpt_step._middle_pp_stage_needs_batch = lambda cfg: True  # middle stages consume the batch too

_overrides = json.loads(os.environ.get("CL_PROVIDER_OVERRIDES", "{}") or "{}")
if _overrides:
    _orig_to_provider = AutoBridge.to_megatron_provider

    def _to_provider(self, *a, **k):
        p = _orig_to_provider(self, *a, **k)
        for key, val in _overrides.items():
            setattr(p, key, val)
        return p

    AutoBridge.to_megatron_provider = _to_provider
    print("qad_distill: provider overrides %s" % _overrides, flush=True)

# GPT datasets build a [seq, seq] boolean attention mask per sample by default (create_attention_mask=True),
# although distill.py never uses it (skip_getting_attention_mask_from_dataset=True; attention is causal).
# At 131072 tokens that is 17 GB per sample in the DataLoader workers, enough to kill a worker.
# Always off here.
from megatron.bridge.training import config as _bcfg

for _cls in (_bcfg.MockGPTDatasetConfig, _bcfg.GPTDatasetConfig):
    _orig_post = _cls.__post_init__

    def _post(self, _orig=_orig_post):
        _orig(self)
        self.create_attention_mask = False

    _cls.__post_init__ = _post

if os.environ.get("CL_MOCK_VOCAB"):
    # Mock data only (tests): Megatron's mock documents are the ids 1, 2, ..., length-1 (lengths up to 4095). Nemotron
    # Nano 9B V2 has 75 embedding rows at ~1e-37 (ids 1192..117744: byte tokens no valid UTF-8 text produces, so never
    # trained), and a step on them gives an Inf embedding gradient; CL_MOCK_VOCAB=N keeps mock ids in 1..N instead.
    from megatron.core.datasets import gpt_dataset as _gds
    import numpy as _np

    _mock_n = int(os.environ["CL_MOCK_VOCAB"])

    def _mock_getitem(self, idx):
        length = self.sequence_lengths[idx]
        return _np.int64(_np.concatenate([(_np.arange(length - 1) % _mock_n) + 1, [self.eod_token]]))

    _gds.MockGPTLowLevelDataset.__getitem__ = _mock_getitem
    print("qad_distill: mock token ids limited to 1..%d" % _mock_n, flush=True)

# Mamba2 (NemotronH): mamba_ssm's fused training kernel overflows 32-bit offsets on long micro-batches (Nano V2: wrong
# outputs from ~52K tokens, NaN from ~95K, crash when packed); mamba_compact.py (next to this file) feeds it
# contiguous pieces instead. Always on; CL_MAMBA_COMPACT=0 turns it off.
if os.environ.get("CL_MAMBA_COMPACT", "1") != "0":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mamba_compact as _mamba_compact

    _mamba_compact.install()

_REV = float(os.environ.get("CL_KD_REVERSE", "0") or 0)
if _REV > 0:
    # KL direction. distill.py builds LogitsKLLoss without its reverse flag, so the loss is always the
    # forward KL(teacher || student) (F.kl_div(student_logp, teacher_logp, log_target=True)). With w =
    # CL_KD_REVERSE in (0, 1]: loss = (1 - w) KL(teacher || student) + w KL(student || teacher). Reverse KL
    # penalises the student for probability on tokens the teacher gives almost none (a 10% vs 1% token costs
    # 0.1 ln 10 = 0.23 reverse, ~0.02 forward), the kind of error that survives top-k/top-p sampling.
    # Patched before the chunking wrapper below, so both paths use it. Full vocabulary only.
    assert _REV <= 1.0, "CL_KD_REVERSE is a weight in [0, 1]"
    from modelopt.torch.distill.plugins import megatron as _mkd_rev

    _orig_kl_forward = _mkd_rev.LogitsKLLoss.forward

    def _mixed_forward(self, predictions, targets):
        if isinstance(self, _mkd_rev.TopKLogitsKLLoss):
            raise RuntimeError("CL_KD_REVERSE needs the full-vocabulary loss (drop --logit_kl_topk): reverse KL "
                               "needs the teacher's probability of every token the student puts mass on")
        self._reverse = True
        rev = _orig_kl_forward(self, predictions, targets)
        if _REV >= 1.0:
            return rev
        self._reverse = False
        fwd = _orig_kl_forward(self, predictions, targets)
        return ((1.0 - _REV) * fwd[0] + _REV * rev[0],) + tuple(fwd[1:])

    _mkd_rev.LogitsKLLoss.forward = _mixed_forward

    def _topk_refuses(self, *a, **k):  # TopKLogitsKLLoss has its own forward and would ignore the mix silently
        raise RuntimeError("CL_KD_REVERSE needs the full-vocabulary loss: drop --logit_kl_topk")

    _mkd_rev.TopKLogitsKLLoss.forward = _topk_refuses
    print("qad_distill: KD loss = %.2f forward KL + %.2f reverse KL" % (1.0 - _REV, _REV), flush=True)

_OPT = json.loads(os.environ.get("CL_OPTIMIZER_OVERRIDES", "{}") or "{}")
if _OPT:
    # Optimizer memory options distill.py has no flags for, e.g. for PP=5 at 128K:
    # '{"use_precision_aware_optimizer": true, "exp_avg_dtype": "bfloat16", "exp_avg_sq_dtype": "bfloat16"}' or
    # '{"use_precision_aware_optimizer": true, "optimizer_cpu_offload": true, "optimizer_offload_fraction": 0.5,
    #   "overlap_cpu_optimizer_d2h_h2d": true}' (offload needs the precision-aware code path; its dtypes stay fp32).
    import torch
    import megatron.bridge.training.distill as _bdistill_opt

    _orig_pretrain_opt = _bdistill_opt.pretrain

    def _pretrain_with_opt(config, forward_step_func, *a, **k):
        for key, val in _OPT.items():
            if isinstance(val, str) and hasattr(torch, val):
                val = getattr(torch, val)  # "bfloat16" -> torch.bfloat16
            if not hasattr(config.optimizer, key):
                raise AttributeError("OptimizerConfig has no field %r" % key)
            setattr(config.optimizer, key, val)
        if _OPT.get("optimizer_cpu_offload") and getattr(config.checkpoint, "async_save", False):
            # MCore's async writer does not snapshot CPU tensors, and the next CPU Adam step updates the offloaded
            # masters and moments in place while they are written
            raise RuntimeError("optimizer_cpu_offload needs --no_async_save")
        return _orig_pretrain_opt(config, forward_step_func, *a, **k)

    _bdistill_opt.pretrain = _pretrain_with_opt
    print("qad_distill: optimizer overrides %s" % _OPT, flush=True)

_SCHED = json.loads(os.environ.get("CL_SCHEDULER_OVERRIDES", "{}") or "{}")
if _SCHED:
    # LR / weight-decay schedule fields distill.py has no flags for (it builds cosine with weight decay 0.033), e.g. a
    # WSD schedule: '{"lr_decay_style": "WSD", "lr_wsd_decay_style": "minus_sqrt", "lr_wsd_decay_iters":
    # 24, "start_weight_decay": 0.0, "end_weight_decay": 0.0, "override_opt_param_scheduler": true}'. With
    # override_opt_param_scheduler a resume takes the schedule from the config (only the step count from the
    # checkpoint), so the total steps and the decay length can be changed before the decay starts. Keys that are
    # OptimizerConfig fields (weight_decay) go to the optimizer config.
    import megatron.bridge.training.distill as _bdistill_sched

    _orig_pretrain_sched = _bdistill_sched.pretrain

    def _pretrain_with_sched(config, forward_step_func, *a, **k):
        for key, val in _SCHED.items():
            target = config.scheduler if hasattr(config.scheduler, key) else config.optimizer
            if not hasattr(target, key):
                raise AttributeError("SchedulerConfig / OptimizerConfig have no field %r" % key)
            setattr(target, key, val)
        return _orig_pretrain_sched(config, forward_step_func, *a, **k)

    _bdistill_sched.pretrain = _pretrain_with_sched
    print("qad_distill: scheduler overrides %s" % _SCHED, flush=True)

_PRETRAINED = os.environ.get("CL_PRETRAINED")
if _PRETRAINED:
    # Start from a trained checkpoint (the chain's init_from): Megatron-Bridge's pretrained_checkpoint
    # path. When the run's own checkpoint directory holds nothing to resume, the model weights (and quantizer state) are
    # read from this directory's latest iteration with finetune semantics: step 0, a new optimizer and schedule, no RNG
    # state. The student is still built and restored from --student_megatron_path first (structure + ModelOpt state);
    # these weights then replace its weights. A resume of the run itself prefers its own checkpoints.
    import megatron.bridge.training.distill as _bdistill_pre

    _orig_pretrain_pre = _bdistill_pre.pretrain

    def _pretrain_with_pre(config, forward_step_func, *a, **k):
        config.checkpoint.pretrained_checkpoint = _PRETRAINED
        return _orig_pretrain_pre(config, forward_step_func, *a, **k)

    _bdistill_pre.pretrain = _pretrain_with_pre
    print("qad_distill: start from the trained checkpoint %s (finetune: step 0, new optimizer)" % _PRETRAINED, flush=True)

_DDP = json.loads(os.environ.get("CL_DDP_OVERRIDES", "{}") or "{}")
if _DDP:
    # DistributedDataParallelConfig fields, e.g. '{"check_for_nan_in_grad": false}' to let a step with an Inf gradient
    # finish so CL_GRAD_REPORT can name the parameter (debugging only).
    import megatron.bridge.training.distill as _bdistill_ddp

    _orig_pretrain_ddp = _bdistill_ddp.pretrain

    def _pretrain_with_ddp(config, forward_step_func, *a, **k):
        for key, val in _DDP.items():
            if not hasattr(config.ddp, key):
                raise AttributeError("DistributedDataParallelConfig has no field %r" % key)
            setattr(config.ddp, key, val)
        return _orig_pretrain_ddp(config, forward_step_func, *a, **k)

    _bdistill_ddp.pretrain = _pretrain_with_ddp
    print("qad_distill: ddp overrides %s" % _DDP, flush=True)

if _OPT.get("optimizer_cpu_offload"):
    # Exact resume with optimizer CPU offload (MCore 0.19.1 + Megatron-Bridge 0.6.1 in
    # nemo:26.08.01). The offloaded part keeps fp32 master, fp32 grad copy and fp32 Adam m/v on the CPU (torch AdamW,
    # same update as TE FusedAdam; clipping acts on the same fp32 decoupled grads); it needs
    # use_precision_aware_optimizer with the dtypes left at fp32. Loading a checkpoint with optimizer state
    # (checkpointing._load_checkpoint_from_path): building the load template calls DistributedOptimizer.load_state_dict,
    # which runs a dummy Adam step on the still empty offload optimizer (its copy-back overwrites the bf16 weights,
    # harmless since the model weights are read afterwards); then the model weights and the optimizer state are read;
    # then setup() calls Bridge's sync_hybrid_device_optimizer_fp32_master_copies. Fixes:
    # 1. That sync copies the bf16 weights over the fp32 masters: right after a weights-only load, but after a load with
    #    optimizer state it replaces the masters just read with bf16-rounded ones. And with Bridge's default checkpoint
    #    format (dp_reshardable) MCore does not call _sync_hdo_state_to_sub_optimizers after reading the state (only the
    #    fully_reshardable / fully_sharded_model_space loaders do; dp_reshardable relies on reading in place). When the
    #    optimizer state was loaded: run that sync (idempotent), keep the masters, and require every bf16 weight to equal
    #    its master rounded to bf16 (true at every save: the weights are set from the masters after each step); any
    #    difference stops the run.
    # 2. HybridDeviceOptimizer copies its param_groups, including the 'step' read from the checkpoint, into the
    #    sub-optimizers before every step. The GPU sub-optimizer (TE FusedAdam) takes its bias correction from
    #    group['step'], so after a resume it would stay at S+1. A sub-optimizer group keeps its own 'step' once it has one.
    # 3. The dp_reshardable loader would set every CPU AdamW state['step'] to the load template's value (1, from the
    #    dummy step) after load_state_dict set it to S: 'step' is left out there (as the other loaders do). After the
    #    load all Adam steps (CPU per parameter, TE per group) must equal the checkpoint's; the masters' share of values
    #    finer than bf16 is reported (about 0 would mean they were lost). Async save is refused (CPU tensors are written
    #    while the next CPU Adam step updates them).
    from megatron.core.optimizer.cpu_offloading.hybrid_optimizer import HybridDeviceOptimizer as _HDO
    from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer as _DO
    import megatron.bridge.training.setup as _bsetup_off

    assert _OPT.get("use_precision_aware_optimizer") is True, "optimizer_cpu_offload needs use_precision_aware_optimizer"
    for _k in ("main_grads_dtype", "main_params_dtype", "exp_avg_dtype", "exp_avg_sq_dtype"):
        assert _OPT.get(_k, "float32") == "float32", "keep %s at float32 with the offload" % _k

    _orig_dummy_step = _HDO.dummy_step

    def _dummy_step_marked(self):  # so that the step report below skips it
        self._cl_in_dummy = True
        try:
            return _orig_dummy_step(self)
        finally:
            self._cl_in_dummy = False

    _HDO.dummy_step = _dummy_step_marked
    _orig_set_state = _DO._set_main_param_and_optimizer_states

    def _set_state_mark(self, model_param, tensors):
        self._cl_optim_state_loaded = True
        if isinstance(self.optimizer, _HDO):
            # dp_reshardable carries each CPU AdamW 'step' as a non-persistent object: the load hands back the
            # template's tensor (value 1 from the dummy step), which would overwrite the step S that load_state_dict
            # just set from the checkpoint's param_groups (the other two loaders skip 'step' for this reason)
            tensors = {k: v for k, v in tensors.items() if k != "step"}
        return _orig_set_state(self, model_param, tensors)

    _DO._set_main_param_and_optimizer_states = _set_state_mark
    _orig_sync_masters = _bsetup_off.sync_hybrid_device_optimizer_fp32_master_copies

    def _sync_masters(optimizer):
        subs = [] if optimizer is None else list(getattr(optimizer, "chained_optimizers", None) or [optimizer])
        hdos = [o for o in subs if isinstance(getattr(o, "optimizer", None), _HDO)]
        loaded = [o for o in hdos if getattr(o, "_cl_optim_state_loaded", False)]
        if not loaded:
            return _orig_sync_masters(optimizer)  # weights-only load: the masters come from the weights
        assert len(loaded) == len(hdos), "optimizer state loaded for only some of the offload optimizers"
        import gc
        diff = total = sub_bf16 = 0
        worst = 0.0
        steps_cpu, steps_gpu, group_steps = set(), set(), set()
        ids = set()  # one step tensor per parameter across every CPU sub-optimizer (overlap: one parameter each)
        with torch.no_grad():
            for o in loaded:
                hdo = o.optimizer
                hdo._sync_hdo_state_to_sub_optimizers()
                for p, fp32 in hdo.param_to_fp32_param.items():
                    assert fp32.dtype == torch.float32
                    new = fp32.to(device=p.device).to(p.dtype)
                    n = int((new != p.data).sum())
                    if n:
                        diff += n
                        worst = max(worst, float((new.float() - p.data.float()).abs().max()))
                    sub_bf16 += int((new.float() != fp32.to(device=p.device)).sum())  # masters carry bits below bf16
                    total += p.numel()
                group_steps |= {g.get("step") for g in hdo.param_groups}
                for sub in hdo.sub_optimizers:
                    if sub is hdo.gpu_optimizer:
                        steps_gpu |= {g.get("step") for g in sub.param_groups}
                    else:
                        for st in sub.state.values():
                            t = st.get("step")
                            assert t is not None and t.device.type == "cpu" and t.dtype == torch.float32, st.keys()
                            assert id(t) not in ids, "CPU Adam step tensors shared between parameters"
                            ids.add(id(t))
                            steps_cpu.add(float(t))
                # point state['master_param'] at the live fp32 copies (the loaded ones were copied into them), so
                # the stale copies are freed now rather than after the first step (4 B / param more at 128K otherwise)
                hdo._sync_sub_optimizers_state_to_hdo()
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # the step reports then show each step's own peak
        report = ("%d bf16 weights, masters below bf16 resolution in %.1f%%, CPU Adam step %s, GPU Adam step %s, "
                  "checkpoint step %s" % (total, 100.0 * sub_bf16 / max(total, 1), sorted(steps_cpu), sorted(steps_gpu),
                                         sorted(x for x in group_steps if x is not None)))
        if total == 0 or diff:
            raise RuntimeError("qad_distill: after loading the optimizer state, %d of %d bf16 weights differ from the "
                               "loaded fp32 masters rounded to bf16 (max |diff| %.3g): not resuming (%s)" % (
                                   diff, total, worst, report))
        S = {x for x in group_steps if x is not None}
        if (len(S) != 1 or None in steps_gpu or {float(x) for x in S} != steps_cpu
                or (steps_gpu and {float(x) for x in steps_gpu} != steps_cpu)):
            raise RuntimeError("qad_distill: Adam steps after loading are not all the checkpoint's step (an offload fraction of 0 "
                               "also lands here: without CPU Adam states MCore saves no step): " + report)
        print("qad_distill: optimizer state loaded; fp32 masters kept; " + report, flush=True)
        return True

    _bsetup_off.sync_hybrid_device_optimizer_fp32_master_copies = _sync_masters
    _orig_sync_groups = _HDO._sync_hdo_param_groups_to_sub_optimizers

    def _sync_groups_keep_step(self):
        own = [[g.get("step") for g in o.param_groups] for o in self.sub_optimizers]
        _orig_sync_groups(self)
        for o, steps in zip(self.sub_optimizers, own):
            for g, s in zip(o.param_groups, steps):
                if s is not None:
                    g["step"] = s

    _HDO._sync_hdo_param_groups_to_sub_optimizers = _sync_groups_keep_step
    _orig_hdo_step = _HDO.step

    def _hdo_step_report(self, closure=None):
        # the first real steps: optimizer time (D2H, CPU Adam, H2D) and host memory, for sizing the offload fraction
        import time
        t = time.time()
        out = _orig_hdo_step(self, closure)
        if getattr(self, "_cl_in_dummy", False):
            return out
        self._cl_steps = getattr(self, "_cl_steps", 0) + 1
        if self._cl_steps <= int(os.environ.get("CL_OFFLOAD_REPORT_STEPS", "4")):
            torch.cuda.synchronize()
            mem = dict(l.split(":", 1) for l in open("/proc/meminfo"))
            rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
            ncpu = sum(p.numel() for g in self.cpu_param_groups for p in g["params"])
            ngpu = sum(p.numel() for g in self.gpu_param_groups for p in g["params"])
            # a native fp32 parameter would make MCore 0.19.1's load_state_dict fail (param_to_fp32_param KeyError)
            nfp32 = sum(1 for g in self.param_groups for p in g["params"] if p.dtype == torch.float32)
            steps = sorted({str(g.get("step")) for o in self.sub_optimizers if o is self.gpu_optimizer for g in o.param_groups})
            csteps = sorted({float(st["step"]) for o in self.cpu_optimizers for st in o.state.values() if "step" in st})
            print("qad_distill: offload optimizer step %d: %.1f s; params on CPU %.2fB, on GPU %.2fB (native fp32 tensors: "
                  "%d); Adam step GPU %s CPU %s; process RSS %.1f GB, host MemAvailable %.1f GB, GPU max allocated "
                  "%.1f GB" % (self._cl_steps, time.time() - t, ncpu / 1e9, ngpu / 1e9, nfp32, ",".join(steps), csteps,
                               rss / 1e9, int(mem["MemAvailable"].split()[0]) / 1e6,
                               torch.cuda.max_memory_allocated() / 1e9), flush=True)
            torch.cuda.reset_peak_memory_stats()  # next line: that step's own peak
        return out

    _HDO.step = _hdo_step_report
    print("qad_distill: optimizer CPU offload: resume keeps the fp32 masters and the Adam step", flush=True)

_FEED = os.environ.get("CL_ONPOLICY_FEED")
if _FEED:
    # On-policy QAD: the training data are the rollout controller's packs (onpolicy/rollout.py), not a dataset.
    # Each stage's iterator reads pack (step, seq) in order from the feed, blocking until it is published, so all
    # stages see identical inputs and training overlaps the rollouts of the same step. After every optimizer step all
    # ranks meet, then rank 0 reports the step done so the controller starts the next step's rollouts (with
    # CL_WEIGHT_PUSH_URL, after pushing the new weights). Use with --use_mock_data (the
    # dataset is never built) and CL_PROVIDER_OVERRIDES '{"calculate_per_token_loss": true, "seq_length": 131072}':
    # distill.py only turns on per-token loss for --sft. gbs / mbs = packs per step; mbs must be 1.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "onpolicy"))
    import torch
    import megatron.bridge.training.distill as _bdistill
    import megatron.bridge.training.setup as _bsetup
    from megatron.bridge.training.callbacks import Callback
    from megatron.core.rerun_state_machine import RerunDataIterator
    from feed_client import FeedIterator, post_step_done

    def _feed_iterators(cfg, train_state, *a, **k):
        gbs, mbs = cfg.train.global_batch_size, cfg.train.micro_batch_size
        assert mbs == 1 and gbs >= 1, "on-policy packs: micro batch size 1, global batch size = packs per step"
        say = lambda m: print("qad_distill: " + m, flush=True)
        it = FeedIterator(_FEED, gbs, start_step=train_state.step, log=say)
        valid = None
        v = getattr(cfg, "validation", None)
        if v is not None and (v.eval_iters or 0) > 0:  # validation rollouts: /val/<e>/<seq>, e = evaluations so far
            e0 = 0 if v.skip_train else train_state.step // v.eval_interval
            valid = RerunDataIterator(FeedIterator(_FEED, gbs, start_step=e0, log=say, route="val", split="val"))
        # the replaced data setup is where Megatron-Bridge sets these (data/loaders.py)
        train_state.do_train = (cfg.train.train_iters or 0) > 0 and not (v is not None and v.skip_train)
        train_state.do_valid = valid is not None
        train_state.do_test = False
        print("qad_distill: on-policy feed %s, %d packs per step, starting at step %d%s" % (
            _FEED, gbs, train_state.step, ", validation every %d steps (%d batch)" % (v.eval_interval, v.eval_iters)
            if valid is not None else ""), flush=True)
        return RerunDataIterator(it), valid, None

    _bsetup.setup_data_iterators = _feed_iterators

    _PUSH_URL = os.environ.get("CL_WEIGHT_PUSH_URL")  # rollout vLLM server; unset: no weight push (feed-only tests)

    import json as _json
    import time as _time
    import urllib.request as _url
    import feed_client as _fc
    import kd_stats as _kd

    def _get_stats(route, k):
        try:
            return _json.load(_url.urlopen("%s/stats/%s/%d" % (_FEED.rstrip("/"), route, k), timeout=6 * 3600))
        except Exception as e:  # statistics must never stop training
            print("qad_distill: no rollout stats for %s %d (%r)" % (route, k, e), flush=True)
            return {}

    _RO_KEYS = ("packs", "finished", "submitted", "truncated", "gen_tokens", "prompt_tokens", "packed_tokens", "loss_tokens",
                "hist_loss_tokens", "packed_samples", "dropped_samples", "dropped_tokens", "requeued", "empty_packs", "fill", "seconds",
                "trunc_rate", "no_think_rate", "rep_mean", "lp_mean")

    def _flatten(st, prefix):
        m = {"%s/%s" % (prefix, x): st[x] for x in _RO_KEYS if st.get(x) is not None}
        for q, v in (st.get("resp_len") or {}).items():
            m["%s/resp_len_%s" % (prefix, q)] = v
        for q, v in (st.get("cumulative") or {}).items():
            if v is not None:
                m["cumulative/%s" % q] = v
        for d, b in (st.get("by_dataset") or {}).items():
            for x in ("n", "resp_mean", "trunc_rate", "no_think_rate", "rep_mean", "lp_mean"):
                if b.get(x) is not None:
                    m["%s_ds/%s/%s" % (prefix, d, x)] = b[x]
        return m

    import wandb_keys as _wk

    def _log(context, split, it, st, extra=None):
        m = _kd.STATS.flush(split)
        last = torch.distributed.get_rank() == torch.distributed.get_world_size() - 1
        if not last:
            return
        m.update(_flatten(st, "rollout" if split == "train" else "val_rollout"))
        m.update(extra or {})
        lr = None
        try:
            lr = float(context.optimizer.param_groups[0]["lr"]) if context.optimizer is not None else None
        except Exception:
            pass
        gn = getattr(context, "grad_norm", None)
        save = getattr(context.state.cfg.checkpoint, "save", None)
        if save:
            with open(os.path.join(os.path.dirname(save.rstrip("/")), "metrics.jsonl"), "a") as f:
                f.write(_json.dumps(dict(step=it, split=split, **m)) + "\n")
        wl = context.state.wandb_logger
        if wl is None:
            return
        # wandb only mirrors metrics.jsonl (written above): a wandb failure (e.g. a timed-out artifact upload of the samples
        # table) must not end the run
        try:
            wl.log(_wk.wandb_view(m, split, lr=lr, grad_norm=None if gn is None else float(gn)), step=it)  # sections: wandb_keys.py
            hist = st.get("resp_len_hist")
            if hist:
                wl.log({("rollout" if split == "train" else "val_rollout") + "/response_len_histogram": wl.Histogram(hist)}, step=it)
            if st.get("samples"):
                cols = ["dataset", "prompt_id", "prompt_tokens", "response_tokens", "prompt_tail", "response"]
                wl.log({("rollout" if split == "train" else "val_rollout") + "/sample_responses":
                        wl.Table(columns=cols, data=[[x[c] for c in cols] for x in st["samples"]])}, step=it)
        except Exception as e:
            print("qad_distill: wandb log failed at %s step %d (%r); the step is in metrics.jsonl" % (split, it, e), flush=True)

    class _StepDone(Callback):
        pusher = None
        n_eval = 0
        t_last = None

        def on_eval_end(self, context):
            st = _get_stats("val", self.n_eval)
            self.n_eval += 1
            _log(context, "val", context.state.train_state.step, st)

        def on_train_start(self, context):
            v = getattr(context.state.cfg, "validation", None)
            if v is not None and v.eval_interval and not v.skip_train:
                self.n_eval = context.state.train_state.step // v.eval_interval
            self.t_last = _time.time()
            v0 = os.environ.get("CL_VAL0_METRICS")  # step-0 validation from a separate validation-only run, so the val curves start at 0
            if (v0 and os.path.exists(v0) and context.state.train_state.step == 0 and context.state.wandb_logger is not None
                    and torch.distributed.get_rank() == torch.distributed.get_world_size() - 1):
                try:
                    rec = [json.loads(l) for l in open(v0) if '"val"' in l][-1]
                    rec.pop("split", None); rec.pop("step", None)
                    context.state.wandb_logger.log(_wk.wandb_view(rec, "val"), step=0)
                    print("qad_distill: step-0 validation logged from " + v0, flush=True)
                except Exception as e:
                    print("qad_distill: step-0 validation not logged (%r)" % (e,), flush=True)
            if _PUSH_URL:
                # Weight push (onpolicy/weight_push.py): NCCL group to the rollout server(s), set up once before step 0.
                import socket
                from weight_push import PPWeightPusher
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:  # this node's address towards vLLM
                    sk.connect((_PUSH_URL.split("//")[-1].split(":")[0], 80)); my_ip = sk.getsockname()[0]
                self.pusher = PPWeightPusher(context.model, os.environ["CL_WEIGHT_PUSH_HF"], _PUSH_URL,
                                             int(os.environ.get("CL_WEIGHT_PUSH_SENDER", "0")), my_ip,
                                             port=int(os.environ.get("CL_WEIGHT_PUSH_PORT", "29611")),
                                             buffer_gb=float(os.environ.get("CL_WEIGHT_PUSH_BUFFER_GB", "2")),
                                             log=lambda m: print("qad_distill: " + m, flush=True))
            step = context.state.train_state.step
            if step == 0 and _PRETRAINED and not getattr(context.state.cfg.train, "skip_train", False):
                # started from a trained checkpoint: vLLM serves the PTQ weights, so push these and report step -1 done,
                # which is what the controller (cfg wait_init) waits for before generating step 0
                if self.pusher is not None:
                    self.pusher.push()
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    post_step_done(_FEED, -1)
                print("qad_distill: started from %s, weights pushed before step 0" % _PRETRAINED, flush=True)
            if step > 0 and not getattr(context.state.cfg.train, "skip_train", False):
                # resumed from a checkpoint: vLLM still serves the starting weights, so push the resumed ones and
                # report step - 1 done, which is what lets the controller start generating step `step`
                if self.pusher is not None:
                    self.pusher.push()
                torch.distributed.barrier()
                if torch.distributed.get_rank() == 0:
                    post_step_done(_FEED, step - 1)
                print("qad_distill: resumed at step %d, weights pushed" % step, flush=True)

        def on_train_step_end(self, context):
            step = context.state.train_state.step  # incremented only after this callback
            t_push = _time.time()
            if self.pusher is not None:
                self.pusher.push()  # new weights are in vLLM before the controller hears the step is done
            t_push = _time.time() - t_push
            torch.distributed.barrier()
            if torch.distributed.get_rank() == 0:
                post_step_done(_FEED, step)
            st = _get_stats("pack", step)
            if st.get("data_exhausted"):  # the subset ran out in this step: save and stop after it
                context.state.cfg.train.exit_interval = step + 1
                print("qad_distill: rollout data exhausted at step %d: checkpoint and exit" % (step + 1), flush=True)
            now = _time.time()
            waits = [None] * torch.distributed.get_world_size()  # stage 0's wait is the trainer idling on rollouts
            torch.distributed.all_gather_object(waits, _fc.take_wait())
            extra = {"time/step_s": now - (self.t_last or now), "time/push_s": t_push}
            extra.update({"time/wait_for_packs_s/stage%d" % i: w for i, w in enumerate(waits)})
            self.t_last = now
            _log(context, "train", step + 1, st, extra)

    _orig_bpretrain = _bdistill.pretrain

    def _pretrain_with_feed(config, forward_step_func, *a, **k):
        config.dist.distributed_timeout_minutes = int(os.environ.get("CL_DIST_TIMEOUT_MIN", "240"))
        if getattr(config.model, "calculate_per_token_loss", False):  # distill.py pairs these only for --sft
            config.ddp.average_in_collective = False
        cbs = list(k.pop("callbacks", None) or []) + [_StepDone()]
        return _orig_bpretrain(config, forward_step_func, *a, callbacks=cbs, **k)

    _bdistill.pretrain = _pretrain_with_feed

_IMPORT = os.environ.get("CL_IMPORT_INPUT_SCALE")
if _IMPORT:
    import glob as _glob
    import torch
    import megatron.bridge.training.distill as _bdistill_imp
    from megatron.bridge.training.callbacks import Callback as _Cb

    _HF_OF = {"self_attention.linear_qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
              "self_attention.linear_proj": ["self_attn.o_proj"],
              "mlp.linear_fc1": ["mlp.gate_proj", "mlp.up_proj"],
              "mlp.linear_fc2": ["mlp.down_proj"]}
    # NemotronH (Nemotron Nano 9B V2): HF backbone.layers.N.mixer.* for every layer kind; Megatron MambaLayer
    # mixer.in_proj / out_proj (z, x, B, C, dt fused in both), MLPLayer linear_fc1 = up_proj (relu2, no gate),
    # attention linear_qkv = q/k/v (kept BF16 in our PTQ, so disabled here too)
    _HF_OF_NEMOTRON_H = {"mixer.in_proj": ["mixer.in_proj"], "mixer.out_proj": ["mixer.out_proj"],
                         "mlp.linear_fc1": ["mixer.up_proj"], "mlp.linear_fc2": ["mixer.down_proj"],
                         "self_attention.linear_qkv": ["mixer.q_proj", "mixer.k_proj", "mixer.v_proj"],
                         "self_attention.linear_proj": ["mixer.o_proj"]}

    # NemotronH MoE (Nemotron 3 Nano): routed experts are TEGroupedMLP (one input quantizer, one weight quantizer per
    # expert: weight_quantizer[e]), shared experts a dense MLP; HF mixer.experts.<e>.up_proj / down_proj (relu2, no gate)
    _MOE_EXPERTS = {"mlp.experts.linear_fc1": "mixer.experts.%d.up_proj", "mlp.experts.linear_fc2": "mixer.experts.%d.down_proj"}

    # routed experts (TEGroupedMLP: one input quantizer for all experts): the checkpoint's per-expert input scales may
    # differ (Qwen3-30B-A3B's down_proj: NVIDIA calibrated each expert); vLLM serves the max over the experts, so the
    # student takes that max too (equal scales: that value)
    _EXPERT_RELS = ("mlp.experts.linear_fc1", "mlp.experts.linear_fc2")
    _GROUP_MAX = {}  # Megatron submodule -> the HF modules whose max input scale it takes (Qwen3 MoE experts)

    def _layer_map(ckpt):  # (HF layer root, {Megatron submodule: [HF modules sharing its input_scale]}, strict)
        cfg = json.load(open(os.path.join(ckpt, "config.json")))
        if cfg.get("model_type") == "qwen3_moe":
            # Qwen3 MoE: ModelOpt builds the experts as SequentialMLP (no export rule for fused experts:
            # modelopt/torch/utils/plugins/mbridge.py), one module per expert: local_experts.<e>.linear_fc1 = gate + up,
            # linear_fc2 = down; no shared experts. Each expert's input quantizer takes the max over its layer's experts
            # of that projection (_GROUP_MAX), as vLLM's FlashInfer MoE backends serve one input scale for all experts
            n_exp = int(cfg["num_experts"])
            m = dict(_HF_OF)
            for e in range(n_exp):
                m["mlp.experts.local_experts.%d.linear_fc1" % e] = ["mlp.experts.%d.gate_proj" % e, "mlp.experts.%d.up_proj" % e]
                m["mlp.experts.local_experts.%d.linear_fc2" % e] = ["mlp.experts.%d.down_proj" % e]
                _GROUP_MAX["mlp.experts.local_experts.%d.linear_fc1" % e] = ["mlp.experts.%d.%s" % (x, q) for x in range(n_exp)
                                                                             for q in ("gate_proj", "up_proj")]
                _GROUP_MAX["mlp.experts.local_experts.%d.linear_fc2" % e] = ["mlp.experts.%d.down_proj" % x for x in range(n_exp)]
            return "model.layers", m, True
        if cfg.get("model_type") == "nemotron_h":
            m = dict(_HF_OF_NEMOTRON_H)
            n_exp = int(cfg.get("n_routed_experts") or 0)
            if n_exp:
                m.update({rel: [hf % e for e in range(n_exp)] for rel, hf in _MOE_EXPERTS.items()})
                m.update({"mlp.shared_experts.linear_fc1": ["mixer.shared_experts.up_proj"],
                          "mlp.shared_experts.linear_fc2": ["mixer.shared_experts.down_proj"]})
            return "backbone.layers", m, True
        return "model.layers", _HF_OF, False  # dense Qwen3, unchanged

    def _read_input_scales(path, suffix=".input_scale"):
        from safetensors import safe_open
        out = {}
        for f in sorted(_glob.glob(os.path.join(path, "*.safetensors"))):
            with safe_open(f, "pt") as st:
                for k in st.keys():
                    if k.endswith(suffix):
                        out[k[: -len(suffix)]] = float(st.get_tensor(k).float().item())
        return out

    class _ImportScales(_Cb):
        done = False

        def on_eval_start(self, context):  # --validate_only never calls on_train_start
            self.on_train_start(context)

        def on_train_start(self, context):
            if self.done:
                return
            self.done = True
            from megatron.core.transformer.transformer_layer import TransformerLayer
            kinds = (TransformerLayer,)
            try:
                from megatron.core.ssm.mamba_layer import MambaLayer
                kinds = (TransformerLayer, MambaLayer)
            except ImportError:
                pass
            root, hf_of, strict = _layer_map(_IMPORT)
            scales = _read_input_scales(_IMPORT)
            ws2 = _read_input_scales(_IMPORT, ".weight_scale_2") if any(r in hf_of for r in _MOE_EXPERTS) else {}
            n_set, missing, used, mine = 0, [], set(), set()
            n_w, w_missing, n_maxed = 0, [], 0
            for chunk in context.model:
                for layer in chunk.modules():
                    if not isinstance(layer, kinds):
                        continue
                    i = layer.layer_number - 1  # global, 0-based (TransformerLayer and MambaLayer)
                    for rel, hf_names in hf_of.items():
                        try:
                            mod = layer.get_submodule(rel)
                        except AttributeError:
                            continue  # a layer kind without this module (hybrid: Mamba / MLP / attention layers)
                        q = getattr(mod, "input_quantizer", None)
                        if q is None:
                            continue  # teacher layers (BF16)
                        mine.add(i)
                        if not getattr(q, "is_enabled", True):
                            continue  # kept BF16
                        keys = ["%s.%d.%s" % (root, i, h) for h in hf_names]
                        vals = [scales.get(k) for k in keys]
                        gvals = [scales.get("%s.%d.%s" % (root, i, h)) for h in _GROUP_MAX[rel]] if rel in _GROUP_MAX else vals
                        if any(v is None for v in vals + gvals) or (len(set(vals)) != 1 and rel not in _EXPERT_RELS):
                            missing.append((i, rel, vals[:4])); continue
                        n_maxed += len(set(gvals)) != 1
                        amax = torch.tensor(max(gvals) * 6.0 * 448.0, dtype=torch.float32, device=q.amax.device
                                            if getattr(q, "amax", None) is not None else "cuda")
                        q.amax = amax
                        used.update(keys)
                        n_set += 1
                        if rel in _MOE_EXPERTS:  # routed experts: every expert's weight amax <- the shared weight_scale_2
                            wv = [ws2.get(k) for k in keys]
                            wq = getattr(mod, "weight_quantizer", None)
                            if any(v is None for v in wv) or len(set(wv)) != 1 or wq is None or len(wq) != len(keys):
                                w_missing.append((i, rel, len(set(wv)), None if wq is None else len(wq))); continue
                            for e in range(len(keys)):
                                qe = wq[e]
                                qe = qe[0] if hasattr(qe, "__getitem__") and not hasattr(qe, "amax") else qe  # SequentialQuantizer
                                qe.amax = torch.tensor(wv[0] * 6.0 * 448.0, dtype=torch.float32,
                                                       device=qe.amax.device if getattr(qe, "amax", None) is not None else "cuda")
                                n_w += 1
            # input_scales of this stage's layers that no enabled quantizer took: the checkpoint quantizes a module
            # that is BF16 here (a wrong recipe or a wrong enable state after a pipeline re-split)
            unused = sorted(k for k in scales if int(k.split(".")[2]) in mine and k not in used)
            print("qad_distill: imported %d activation scales from %s%s%s%s" % (
                n_set, _IMPORT, (", NOT FOUND or unequal: %s" % missing[:4]) if missing else "",
                (", %d input_scales of this stage unused: %s" % (len(unused), unused[:4])) if unused else "",
                (", %d routed-expert weight scales (weight_scale_2)%s" % (n_w, (", NOT FOUND or unequal: %s" % w_missing[:4])
                                                                            if w_missing else "")) if ws2 else "")
                + (", %d routed-expert input quantizers set to the max of unequal expert scales" % n_maxed if n_maxed else ""), flush=True)
            if missing:
                raise RuntimeError("CL_IMPORT_INPUT_SCALE: %d quantizers without a matching input_scale" % len(missing))
            if w_missing:
                raise RuntimeError("CL_IMPORT_INPUT_SCALE: %d routed-expert projections without one shared weight_scale_2" % len(w_missing))
            if unused and strict:
                raise RuntimeError("CL_IMPORT_INPUT_SCALE: %d input_scales of quantized modules have no enabled "
                                   "quantizer here" % len(unused))

    _orig_bpretrain_imp = _bdistill_imp.pretrain

    def _pretrain_with_import(config, forward_step_func, *a, **k):
        cbs = [_ImportScales()] + list(k.pop("callbacks", None) or [])
        return _orig_bpretrain_imp(config, forward_step_func, *a, callbacks=cbs, **k)

    _bdistill_imp.pretrain = _pretrain_with_import

_CHUNK = int(os.environ.get("CL_CHUNKED_KD", "0") or 0)
if _CHUNK > 0:
    # Chunked logit distillation, so the last stage never holds [seq, vocab] logits (40 GB per model at
    # 128K tokens x 152K vocab in bf16, twice that in fp32). Both output layers are made to return
    # their input hidden states; the KD loss then runs the real output layers (student with its
    # fake quantizers) on sequence chunks of _CHUNK tokens, each under activation checkpointing so
    # the backward recomputes the chunk's logits instead of keeping them.
    import torch
    import torch.utils.checkpoint as _ckpt
    import megatron.bridge.models.distillation_provider as _dp
    import modelopt.torch.distill as _mtd
    from modelopt.torch.distill.plugins import megatron as _mkd

    _REG = {}
    if _FEED:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "onpolicy"))
        import feed_client as _STATS_FC
        import kd_stats as _STATS_KD

    def _defer(tag, layer):
        orig = layer.forward

        def fwd(input_, *a, **k):
            _REG[tag + "_args"] = (a, k)  # e.g. weight= for tied embeddings; MTP heads call first
            return input_, None

        _REG[tag] = orig
        layer.forward = fwd

    def _chunk_hook(model_chunks):
        for chunk in model_chunks:
            dms = [m for m in chunk.modules() if isinstance(m, _mtd.DistillationModel)]
            for dm in dms:
                s_ol = getattr(dm, "output_layer", None)
                if s_ol is None:
                    continue  # not the last pipeline stage
                if (dm.config.mtp_num_layers or 0) and not _mkd._mtp_excluded_from_quantization(dm):
                    raise RuntimeError("CL_CHUNKED_KD needs MTP excluded from quantization (its CE "
                                       "would see hidden states instead of logits)")
                _defer("s", s_ol)
                _defer("t", dm.teacher_model.output_layer)
                print("qad_distill: chunked KD on, %d tokens per chunk" % _CHUNK, flush=True)
        return model_chunks

    _orig_convert = _dp.convert_to_distillation_provider

    def _convert(*a, **k):
        prov = _orig_convert(*a, **k)
        prov.register_pre_wrap_hook(_chunk_hook)  # appended: runs after the KD conversion hook
        return prov

    _dp.convert_to_distillation_provider = _convert

    def _chunked(orig_forward):
        def forward(self, predictions, targets):
            if "s" not in _REG:
                return orig_forward(self, predictions, targets)
            hs = predictions[0] if isinstance(predictions, tuple) else predictions
            ht = (targets[0] if isinstance(targets, tuple) else targets).detach()
            s_args, t_args = _REG.get("s_args", ((), {})), _REG.get("t_args", ((), {}))

            def one(hs_c, ht_c):
                ls = _REG["s"](hs_c, *s_args[0], **s_args[1])
                with torch.no_grad():
                    lt = _REG["t"](ht_c, *t_args[0], **t_args[1])
                return orig_forward(self, ls, lt)[0]  # [b, chunk]

            parts = [_ckpt.checkpoint(one, hs[i:i + _CHUNK], ht[i:i + _CHUNK], use_reentrant=False)
                     for i in range(0, hs.shape[0], _CHUNK)]
            tp_reduce = not isinstance(self, _mkd.TopKLogitsKLLoss)
            out = torch.cat(parts, dim=1)
            if _FEED and _STATS_FC.META:  # per-token statistics of on-policy packs (onpolicy/kd_stats.py)
                meta = _STATS_FC.META.popleft()
                try:
                    _STATS_KD.STATS.update(meta, out[0].detach(), hs.detach(), ht,
                                           lambda h: _REG["s"](h, *s_args[0], **s_args[1]),
                                           lambda h: _REG["t"](h, *t_args[0], **t_args[1]))
                except Exception as e:  # statistics must never stop training
                    print("qad_distill: kd stats skipped (%r)" % (e,), flush=True)
            return out, tp_reduce, False
        return forward

    _mkd.LogitsKLLoss.forward = _chunked(_mkd.LogitsKLLoss.forward)
    _mkd.TopKLogitsKLLoss.forward = _chunked(_mkd.TopKLogitsKLLoss.forward)

_KEEP_FULL = {int(x) for x in os.environ.get("CL_KEEP_FULL_STEPS", "").split(",") if x.strip()}
if os.environ.get("CL_CKPT_LATEST_FULL") == "1":
    # Every save_interval: a weights-only checkpoint (+ quantizer state) that is kept, as with CL_NO_OPTIM_CKPT, and
    # right after it a full one (optimizer state too) as Megatron-Bridge's global non-persistent checkpoint in
    # <save>/non_persistent, where only the newest is kept (the older one is removed once the new one is written).
    # Resuming prefers the non-persistent checkpoint when it is at least as new as the persistent one and loads the
    # optimizer only from a checkpoint saved with it, so a resume continues Adam where it was.
    import megatron.bridge.training.train as _btrain_ck
    import megatron.bridge.training.checkpointing as _bckpt

    _orig_ck_post_full = _bcfg.CheckpointConfig.__post_init__

    def _ck_post_full(self):
        _orig_ck_post_full(self)
        self.save_optim = False
        self.load_optim = True
        self.non_persistent_ckpt_type = "global"
        self.most_recent_k = -1  # keep every weights-only checkpoint (distill.py sets 5; rank 0 alone would delete)

    _bcfg.CheckpointConfig.__post_init__ = _ck_post_full
    _orig_save_and_time = _btrain_ck.save_checkpoint_and_time

    _saved_step = [None]

    def _save_weights_then_full(state, *a, **k):
        ck = state.cfg.checkpoint
        if state.train_state.step == _saved_step[0]:  # e.g. a CL_KEEP_FULL_STEPS step that is also an exit step
            print("qad_distill: step %d already saved" % state.train_state.step, flush=True)
            return
        _saved_step[0] = state.train_state.step
        if k.pop("non_persistent_ckpt", False):
            full_only = True
        else:
            full_only = False
            ck.save_optim = False
            _orig_save_and_time(state, *a, non_persistent_ckpt=False, **k)
        ck.save_optim = True
        try:
            return _orig_save_and_time(state, *a, non_persistent_ckpt=True, **k)
        finally:
            ck.save_optim = False
            if not full_only:
                print("qad_distill: step %d saved weights-only + full (non_persistent, newest only)"
                      % state.train_state.step, flush=True)

    _btrain_ck.save_checkpoint_and_time = _save_weights_then_full
    _orig_np_cleanup = _bckpt.cleanup_old_non_persistent_checkpoint

    def _np_cleanup_keep_newest(save_dir, leave_ckpt_num=1, *a, **k):
        if os.path.basename(str(save_dir).rstrip("/")) != "non_persistent":
            return _orig_np_cleanup(save_dir, leave_ckpt_num, *a, **k)
        # The original removes only on rank 0 (it assumes a shared filesystem); here every node has its own disk, so
        # every rank removes the older full checkpoints from its own disk and keeps the newest (<= max_iteration).
        import shutil
        mx = k.get("max_iteration")
        its = sorted(int(d[5:]) for d in os.listdir(save_dir)
                     if d.startswith("iter_") and d[5:].isdigit() and (mx is None or int(d[5:]) <= mx))
        its = [it for it in its[:-1] if it not in _KEEP_FULL] + its[-1:]  # CL_KEEP_FULL_STEPS stay
        for it in its[:-1]:
            shutil.rmtree(os.path.join(save_dir, "iter_%07d" % it), ignore_errors=True)
        if its[:-1]:
            print("qad_distill: removed full checkpoint(s) %s from this node, kept %d" % (its[:-1], its[-1]), flush=True)

    _bckpt.cleanup_old_non_persistent_checkpoint = _np_cleanup_keep_newest
    if _KEEP_FULL:
        # CL_KEEP_FULL_STEPS=96[,...]: at these steps also save (weights-only + full) when the step is not on
        # save_interval, and never remove their full checkpoints from non_persistent, e.g. the last constant-LR step
        # before a WSD decay, from which a longer run can branch later with the optimizer state.
        _orig_ck_decide = _btrain_ck.checkpoint_and_decide_exit

        def _ck_decide_keep(state, model, optimizer, opt_param_scheduler, flops, checkpoint_manager, *a, **k):
            step, ck = state.train_state.step, state.cfg.checkpoint
            if step in _KEEP_FULL and ck.save and not (ck.save_interval and step % ck.save_interval == 0):
                kw = {x: k[x] for x in ("train_data_iterator", "pg_collection", "callback_manager", "module_name") if x in k}
                if a:
                    kw["train_data_iterator"] = a[0]
                _btrain_ck.save_checkpoint_and_time(state, model, optimizer, opt_param_scheduler, flops, checkpoint_manager, **kw)
                print("qad_distill: step %d is in CL_KEEP_FULL_STEPS: saved, its full checkpoint is kept" % step, flush=True)
            return _orig_ck_decide(state, model, optimizer, opt_param_scheduler, flops, checkpoint_manager, *a, **k)

        _btrain_ck.checkpoint_and_decide_exit = _ck_decide_keep
        print("qad_distill: full checkpoints kept at steps %s" % sorted(_KEEP_FULL), flush=True)

if os.environ.get("CL_NO_OPTIM_CKPT") == "1":
    # Save and load model weights (+ quantizer state) only, no optimizer state: about 1/8 of the disk
    # (a 36B student's optimizer state is ~96 GB per node at PP=6). Resuming re-initialises Adam.
    _orig_ck_post = _bcfg.CheckpointConfig.__post_init__

    def _ck_post(self):
        _orig_ck_post(self)
        self.save_optim = False
        self.load_optim = False

    _bcfg.CheckpointConfig.__post_init__ = _ck_post

if os.environ.get("CL_TIMING"):
    # Megatron timers per rank (forward/backward compute vs pipeline send/recv waits); level 1 or 2
    _lvl = int(os.environ["CL_TIMING"])
    _orig_log_init = _bcfg.LoggerConfig.__init__

    def _log_init(self, *a, **k):
        _orig_log_init(self, *a, **k)
        self.timing_log_level = _lvl
        self.timing_log_option = "all"

    _bcfg.LoggerConfig.__init__ = _log_init

if os.environ.get("CL_GRAD_REPORT"):
    # For the first N steps print, on every rank, the total gradient norm of its stage and the eight
    # parameters with the largest gradients. train_step zeroes the grad buffers when it starts, so
    # main_grad still holds the step's gradients when it returns (after clipping, which rescales all
    # of them by the same factor, so the ranking is unaffected).
    import math
    import torch
    import megatron.bridge.training.train as _tr

    _n_rep, _cnt = int(os.environ["CL_GRAD_REPORT"]), [0]
    _orig_ts = _tr.train_step

    def _ts(forward_step_func, data_iterator, model, *a, **k):
        out = _orig_ts(forward_step_func, data_iterator, model, *a, **k)
        if _cnt[0] < _n_rep:
            _cnt[0] += 1
            norms = []
            for chunk in model:
                for name, p in chunk.named_parameters():
                    g = getattr(p, "main_grad", None)
                    g = p.grad if g is None else g
                    if g is not None:
                        norms.append((float(g.float().norm()), name, tuple(p.shape)))
            bad = [(name, shape) for n, name, shape in norms if not math.isfinite(n)]
            if bad:
                print("cl_grad[rank %d step %d] non-finite gradients in %d parameters: %s"
                      % (torch.distributed.get_rank(), _cnt[0] + 1, len(bad), "; ".join("%s %s" % x for x in bad[:20])),
                      flush=True)
            norms = [x for x in norms if math.isfinite(x[0])]
            norms.sort(reverse=True)
            tot = math.sqrt(sum(n * n for n, _, _ in norms))
            print("cl_grad[rank %d step %d] stage total %.3e | " % (torch.distributed.get_rank(), _cnt[0], tot)
                  + " ; ".join("%.2e %s %s" % x for x in norms[:8]), flush=True)
            for chunk in model:  # where inside the word embedding does the gradient sit?
                for name, p in chunk.named_parameters():
                    g = getattr(p, "main_grad", None)
                    if g is None or not name.endswith("word_embeddings.weight"):
                        continue
                    rows = g.float().norm(dim=1)
                    top = torch.topk(rows, 10)
                    share = float((top.values ** 2).sum() / (rows ** 2).sum().clamp_min(1e-30))
                    print("cl_grad_emb[rank %d step %d] rows with grad: %d; top-10 rows hold %.1f%% of |g|^2: %s"
                          % (torch.distributed.get_rank(), _cnt[0], int((rows > 0).sum()), 100 * share,
                             ", ".join("%d:%.2e" % (i, v) for v, i in zip(top.values.tolist(), top.indices.tolist()))),
                          flush=True)
        return out

    _tr.train_step = _ts

if os.environ.get("CL_NO_SAVE") == "1":  # benchmarks: distill.py always saves at the end
    import megatron.bridge.training.train as _train

    _train.save_checkpoint_and_time = lambda *a, **k: print("qad_distill: CL_NO_SAVE=1, checkpoint skipped", flush=True)

sys.argv[0] = "distill.py"
sys.path.insert(0, os.getcwd())  # distill.py imports export_distilled_megatron_to_hf from its own directory
runpy.run_path("distill.py", run_name="__main__")
