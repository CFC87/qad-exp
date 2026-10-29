"""Evaluation protocols of the four models and the arm registry. Stdlib only.

An arm is a name (results go to <results>/<arm>/) bound to a local Hugging Face checkpoint directory and to the model
whose protocol it runs. The model is read from the checkpoint's config (model type and hidden size, MODELS[...]["config"]);
`--model` only confirms it. Every arm of a model runs with exactly that model's protocol: nothing depends on the arm.

Built-in arms point at the pinned downloads under models/ (env/versions.toml):
    bf16   Qwen/Qwen3-8B                (upper reference)
    nvfp4  nvidia/Qwen3-8B-NVFP4        (NVIDIA's post-training quantization: the PTQ baseline QAD starts from)
The reference arms of the other three models (REFERENCE_ARMS) and the exports of round 4 and mix60 (EXPORTS, under
out/exports/) are registered by name as well. Any other arm is given as NAME=PATH, e.g. a QAD checkpoint exported to
Hugging Face NVFP4 format.

Per model (MODELS):
    sampling           completion benchmarks (each model card's thinking-mode setting)
    agent_sampling     tau2 and BFCL (the card's tool-calling setting; differs from `sampling` for Nemotron 3 Nano)
    yarn, native_len   static YaRN x4 for the long-context phases (Qwen3 only; the Nemotrons are natively long)
    server_args        vLLM flags of every phase; quant_server_args only for NVFP4 checkpoints
    tool_parser, reasoning_parser (+ *_plugin: files under lib/vllm_plugins/, mounted read-only into the container)
                       server-side parsing for tau2, and for BFCL where bfcl_server_parsing is set
    agent_prefix_cache prefix caching in the tau2 / BFCL phases
    bfcl_handler       lib/bfcl_handler.py handler
    add_special_tokens False: the rendered prompt is sent and counted without the tokenizer's BOS (Nemotrons)
    prompt_style       NVIDIA's evaluation prompts for competitions, GPQA and LiveCodeBench v6 (run_phase.PROMPT_STYLES)
    no_think_marker    thinking turned off by a system-message marker (unused by the current benchmarks)
"""

import json
import re
from pathlib import Path

import paths

# Thinking-mode sampling of the Qwen3 model card, used for every benchmark (reasoning, code, long context, tool use).
# presence_penalty stays 0 (the card allows up to 1.5; it is CPU-bound in vLLM on long traces).
SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)

# Qwen3 is natively 32K (max_position_embeddings 40,960). Its card enables static YaRN x4 only for long inputs because
# it degrades short texts, so only the long-context phases start the server with this override. transformers 5 keeps
# RoPE settings in rope_parameters (rope_theta included), so the override restates the theta.
YARN = {"rope_parameters": {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 32768,
                            "rope_theta": 1000000}}
NATIVE_LEN = 40960

# Server-side parsers for the tau2 agent endpoint (OpenAI chat API with tool calls): both built into vLLM.
TOOL_PARSER = "hermes"
REASONING_PARSER = "qwen3"

# The name the vLLM server registers the model under. Only clients see it (the "model" field of a request); it never
# reaches the model.
SERVED_NAME = "eval-model"

BUILTIN_ARMS = {"bf16": "Qwen3-8B", "nvfp4": "Qwen3-8B-NVFP4"}
ARM_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")

DEFAULT_MODEL = "qwen3-8b"

# Nemotron model cards: Nano 9B v2 thinking T 0.6 / top-p 0.95; Nemotron 3 Nano T 1.0 / top-p 1.0 for reasoning and
# 0.6 / 0.95 for tool calling. Neither card sets top-k (-1: off).
NEMOTRON_T06 = dict(temperature=0.6, top_p=0.95, top_k=-1, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
NEMOTRON_T10 = dict(temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)
# Mamba SSM state in float32 (Nano 9B v2's card requires it, Nemotron 3 Nano's config declares it); passed explicitly.
NEMOTRON_SERVER_ARGS = ["--mamba-ssm-cache-dtype", "float32"]

MODELS = {
    "qwen3-8b": dict(
        title="Qwen3-8B", config=("qwen3", 4096), sampling=SAMPLING, agent_sampling=SAMPLING, yarn=YARN,
        native_len=NATIVE_LEN, server_args=[], quant_server_args=[], tool_parser=TOOL_PARSER,
        reasoning_parser=REASONING_PARSER, tool_parser_plugin=None, reasoning_parser_plugin=None,
        agent_prefix_cache=True, bfcl_handler="qadx/qwen3-ct", bfcl_server_parsing=False, add_special_tokens=True,
        prompt_style=None, no_think_marker=None),
    # Qwen3-30B-A3B: the Qwen3-8B protocol (same card, template, parsers and BFCL handler).
    "qwen3-30b-a3b": dict(
        title="Qwen3-30B-A3B", config=("qwen3_moe", 2048), sampling=SAMPLING, agent_sampling=SAMPLING, yarn=YARN,
        native_len=NATIVE_LEN, server_args=[], quant_server_args=[], tool_parser=TOOL_PARSER,
        reasoning_parser=REASONING_PARSER, tool_parser_plugin=None, reasoning_parser_plugin=None,
        agent_prefix_cache=True, bfcl_handler="qadx/qwen3-ct", bfcl_server_parsing=False, add_special_tokens=True,
        prompt_style=None, no_think_marker=None),
    # Nemotron Nano 9B v2: its template ignores enable_thinking ('/no_think' in a system message turns thinking off).
    # Tool calls: NVIDIA's parser from the model repository (<TOOLCALL>[...]</TOOLCALL>), ported to vLLM 0.29's module
    # layout; "<think>" / "</think>" are plain text in this tokenizer, so a text-based reasoning parser splits reasoning
    # from content the way vLLM's deepseek_r1 parser does. BFCL goes through vLLM's chat endpoint with these parsers.
    # Prefix caching stays off in the agent phases (vLLM marks it experimental for hybrid models).
    "nemotron-nano-9b-v2": dict(
        title="NVIDIA Nemotron Nano 9B v2", config=("nemotron_h", 4480), sampling=NEMOTRON_T06,
        agent_sampling=NEMOTRON_T06, yarn=None, native_len=131072, server_args=NEMOTRON_SERVER_ARGS, quant_server_args=[],
        tool_parser="nemotron_json", reasoning_parser="nemotron_nano_v2_text",
        tool_parser_plugin="nemotron_toolcall_parser.py", reasoning_parser_plugin="nemotron_nano_v2_reasoning_parser.py",
        agent_prefix_cache=False, bfcl_handler="qadx/nemotron-nano-v2-ct", bfcl_server_parsing=True,
        add_special_tokens=False, prompt_style="nemo_v2", no_think_marker="/no_think"),
    # Nemotron 3 Nano 30B-A3B: vLLM's built-in qwen3_coder tool parser and nemotron_v3 reasoning parser. NVFP4 MoE on
    # vLLM 0.29: the default backend returns wrong outputs for this model; FlashInfer CUTLASS (W4A4) is correct.
    "nemotron-3-nano-30b-a3b": dict(
        title="NVIDIA Nemotron 3 Nano 30B-A3B", config=("nemotron_h", 2688), sampling=NEMOTRON_T10,
        agent_sampling=NEMOTRON_T06, yarn=None, native_len=262144, server_args=NEMOTRON_SERVER_ARGS,
        quant_server_args=["--moe-backend", "flashinfer_cutlass"], tool_parser="qwen3_coder",
        reasoning_parser="nemotron_v3", tool_parser_plugin=None, reasoning_parser_plugin=None,
        agent_prefix_cache=False, bfcl_handler="qadx/nemotron3-nano-ct", bfcl_server_parsing=True,
        add_special_tokens=False, prompt_style="nemo_v3", no_think_marker=None),
}
ALIASES = {"q8": "qwen3-8b", "q30": "qwen3-30b-a3b", "n9": "nemotron-nano-9b-v2", "n3": "nemotron-3-nano-30b-a3b"}

# Reference arms of the other models: the checkpoints of the reference PTQ / BF16 results in eval/reference/baselines.json
# (the PTQ of a Nemotron is the round-4 start checkpoint, built by `chain.py <config> prepare`). Their scores are given
# there; they are registered so that their protocol can be inspected (--dry-run) or repeated.
REFERENCE_ARMS = {
    "qwen3-30b-a3b": {"q30-bf16": "Qwen3-30B-A3B", "q30-ptq": "Qwen3-30B-A3B-NVFP4"},
    "nemotron-nano-9b-v2": {"n9-bf16": "NVIDIA-Nemotron-Nano-9B-v2", "n9-ptq": "NVIDIA-Nemotron-Nano-9B-v2-PTQ-NVFP4-nvscale"},
    "nemotron-3-nano-30b-a3b": {"n3-bf16": "NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
                                "n3-ptq": "NVIDIA-Nemotron-3-Nano-30B-A3B-PTQ-NVFP4-nvscale"},
}

# Exports of experiments/round4, experiments/mix60, experiments/mix60b, experiments/mix60b-q8, experiments/mix60c and experiments/mix60c-q30: <name> -> model; the checkpoint is out/exports/<name> unless given
# as NAME=PATH.
EXPORTS_DIR = paths.REPO / "out" / "exports"
EXPORTS = {**{f"r4-q30-s{s}": "qwen3-30b-a3b" for s in (150, 160, 180, 210, 240)},
           **{f"r4-n9-s{s}": "nemotron-nano-9b-v2" for s in (200, 220, 240)},
           **{f"r4-n3-s{s}": "nemotron-3-nano-30b-a3b" for s in (200, 220, 240)},
           **{f"mix60-q30-s{s}": "qwen3-30b-a3b" for s in (240, 300)},
           "mix60-n9-s300": "nemotron-nano-9b-v2",
           "mix60-n3-s300": "nemotron-3-nano-30b-a3b",
           "mix60b-q30-s300": "qwen3-30b-a3b",
           "mix60c-n9-s390": "nemotron-nano-9b-v2",
           "mix60c-n3-s390": "nemotron-3-nano-30b-a3b",
           "mix60c-q30-s430": "qwen3-30b-a3b",
           **{f"mix60b-q8-{a}-s300": "qwen3-8b" for a in ("sft", "mkl", "rkl")}}


def model_id(name):
    """A model id or alias (q8, q30, n9, n3) -> the model id."""
    m = ALIASES.get(name, name)
    if m not in MODELS:
        raise SystemExit(f"unknown model {name!r}; models: {', '.join(MODELS)} (aliases {', '.join(ALIASES)})")
    return m


def spec(model):
    return MODELS[model_id(model)]


def registered_model(name):
    """The model an arm name is registered to (built-in, reference or round-4 export), else None."""
    if name in BUILTIN_ARMS:
        return DEFAULT_MODEL
    for m, arms in REFERENCE_ARMS.items():
        if name in arms:
            return m
    return EXPORTS.get(name)


def model_of_checkpoint(path):
    """The model whose config (model type, hidden size) the checkpoint has, or None."""
    try:
        c = json.loads((Path(path) / "config.json").read_text())
    except (OSError, ValueError):
        return None
    sig = (c.get("model_type"), c.get("hidden_size"))
    return next((m for m, s in MODELS.items() if s["config"] == sig), None)


def resolve_arm(spec_, model=None):
    """'NAME' (registered) or 'NAME=PATH' -> (name, absolute checkpoint dir, model id).

    The model is the one the checkpoint's config identifies; a registered name or an explicit `model` must agree with it.
    Without a readable config (the checkpoint is checked later) it falls back to the registered model, then `model`, then
    Qwen3-8B."""
    name, _, path = spec_.partition("=")
    if not ARM_NAME.match(name):
        raise SystemExit(f"arm name {name!r}: use letters, digits, '_', '.', '-'")
    reg = registered_model(name)
    if not path:
        if name in BUILTIN_ARMS:
            path = paths.MODELS / BUILTIN_ARMS[name]
        elif reg in REFERENCE_ARMS and name in REFERENCE_ARMS[reg]:
            path = paths.MODELS / REFERENCE_ARMS[reg][name]
        elif name in EXPORTS:
            path = EXPORTS_DIR / name
        else:
            raise SystemExit(f"arm {name!r} is not registered; give it as {name}=/path/to/checkpoint")
    path = str(Path(path).expanduser().resolve())
    want = model_id(model) if model else None
    found = model_of_checkpoint(path)
    for src, m in (("--model", want), (f"the registered model of {name}", reg)):
        if m and found and m != found:
            raise SystemExit(f"arm {name}: {path} is a {MODELS[found]['title']} checkpoint, but {src} is {m}")
    if want and reg and want != reg:
        raise SystemExit(f"arm {name} is registered to {reg}, not {want}")
    return name, path, found or reg or want or DEFAULT_MODEL


def is_nvfp4(path):
    """The checkpoint is NVFP4 (its hf_quant_config.json says so): it gets the model's quant_server_args."""
    try:
        q = json.loads((Path(path) / "hf_quant_config.json").read_text())
    except (OSError, ValueError):
        return False
    return (q.get("quantization") or {}).get("quant_algo") == "NVFP4"


def plugin_path(name):
    return str(paths.LIB / "vllm_plugins" / name)


def check_checkpoint(path, model=None):
    """Problems that would stop the server or the client tokenizer, or run the wrong protocol (empty list = fine)."""
    p = Path(path)
    if not p.is_dir():
        return [f"{path}: not a directory"]
    out = [f"{path}: missing {f}" for f in ("config.json", "tokenizer_config.json") if not (p / f).exists()]
    # The server container mounts only this directory: symlinks that leave it would be dangling inside.
    for f in p.iterdir():
        if f.is_symlink() and p not in f.resolve().parents:
            out.append(f"{f}: symlink outside the checkpoint directory (copy the file instead)")
    if model and (p / "config.json").exists():
        found = model_of_checkpoint(path)
        if found != model_id(model):
            out.append(f"{path}: config is {'not one of the evaluated models' if found is None else MODELS[found]['title']},"
                       f" expected {MODELS[model_id(model)]['title']}")
    return out
