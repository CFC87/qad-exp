"""mamba_ssm's fused Mamba2 training kernel (ssd_combined.MambaSplitConv1dScanCombinedFn, 2.3.2) with the in_proj output
split into contiguous pieces, for sequences past ~94K tokens.

The fused function reads z, xBC and dt (and writes their gradients) as strided views of the in_proj output zxbcdt
[batch, seq, 2*dim + 2*ngroups*dstate + nheads]; its Triton / CUDA kernels compute element offsets such as
t * stride_seqlen in 32-bit integers. For Nemotron Nano 9B V2 (row stride 22656) the offset passes 2^31 at t = 94,786:
a 131,072-token micro-batch gives NaN gradients (measured: 90,112 tokens fine, 98,304 NaN). Here z, xBC
and dt are copied to contiguous tensors (row strides 10240 / 12288 / 128, the largest 1.6e9 elements at 131K) and their
gradients are allocated separately and concatenated at the end. Same kernels and operations otherwise: where the
original works, the output and the input gradient are bitwise equal and the parameter gradients (sequence reductions
whose order the original does not fix either) agree to bf16 rounding (tested against the original). Costs a few GB of
transient memory per Mamba layer (the copies, and the concatenation in backward).

    import mamba_compact; mamba_compact.install()   # patches megatron.core.ssm.mamba_mixer
"""
import inspect
import textwrap

_FWD_SPLIT = "zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)"
_BWD_SPLIT = "zx0, z, xBC, dt = torch.split(zxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)"
_BWD_GRAD = ("dzxbcdt = torch.empty_like(zxbcdt)\n"
             "        dzx0, dz, dxBC_given, ddt_given = torch.split(dzxbcdt, [2 * d_nonssm, dim, dim + 2 * ctx.ngroups * dstate, nheads], dim=-1)")
_BWD_RET = "return dzxbcdt, dweight, dbias,"
_CONTIG = "z, xBC, dt = z.contiguous(), xBC.contiguous(), dt.contiguous()  # compact: no 32-bit offset overflow"


def build():
    """The patched autograd Function and a drop-in replacement of mamba_split_conv1d_scan_combined."""
    from mamba_ssm.ops.triton import ssd_combined as M

    src = textwrap.dedent(inspect.getsource(M.MambaSplitConv1dScanCombinedFn))
    for s in (_FWD_SPLIT, _BWD_SPLIT, _BWD_GRAD, _BWD_RET):
        assert src.count(s) == 1, "mamba_ssm changed: %r not found exactly once" % s
    src = src.replace("class MambaSplitConv1dScanCombinedFn(", "class MambaSplitConv1dScanCombinedCompactFn(")
    src = src.replace(_FWD_SPLIT, _FWD_SPLIT + "\n        " + _CONTIG)
    src = src.replace(_BWD_SPLIT, _BWD_SPLIT + "\n        " + _CONTIG)
    src = src.replace(_BWD_GRAD, (
        "dzx0 = zxbcdt.new_empty(*zxbcdt.shape[:2], 2 * d_nonssm)\n"
        "        dz, dxBC_given, ddt_given = torch.empty_like(z), torch.empty_like(xBC), torch.empty_like(dt)\n"
        "        dz_buf = dz"))
    src = src.replace(_BWD_RET, (
        "dzxbcdt = torch.cat([dzx0, dz_buf, dxBC_given, ddt_given], dim=-1)\n"
        "        " + _BWD_RET))
    ns = dict(vars(M))
    exec(compile(src, "<mamba_compact>", "exec"), ns)
    Fn = ns["MambaSplitConv1dScanCombinedCompactFn"]
    params = list(inspect.signature(M.mamba_split_conv1d_scan_combined).parameters)

    def mamba_split_conv1d_scan_combined(*args, **kwargs):
        bound = dict(zip(params, args))
        bound.update(kwargs)
        full = inspect.signature(M.mamba_split_conv1d_scan_combined).bind(**bound)
        full.apply_defaults()
        return Fn.apply(*full.arguments.values())

    return Fn, mamba_split_conv1d_scan_combined


def install():
    from megatron.core.ssm import mamba_mixer

    if getattr(mamba_mixer, "_cl_compact", False):
        return
    _, fn = build()
    mamba_mixer.mamba_split_conv1d_scan_combined = fn
    mamba_mixer._cl_compact = True
    print("qad_distill: Mamba fused kernel uses contiguous z / xBC / dt (no 32-bit offset overflow past ~94K tokens)",
          flush=True)
