"""Qwen3-8B evaluation protocol and the arm registry. Stdlib only.

An arm is a name (results go to <results>/<arm>/) bound to a local Hugging Face checkpoint directory. Two arms are
built in and point at the pinned downloads under models/ (env/versions.toml):
    bf16   Qwen/Qwen3-8B                (upper reference)
    nvfp4  nvidia/Qwen3-8B-NVFP4        (NVIDIA's post-training quantization: the PTQ baseline QAD starts from)
Any other arm is given as NAME=PATH, e.g. a QAD checkpoint exported to Hugging Face NVFP4 format. Every arm, built-in
or not, runs with exactly the same protocol: nothing below depends on the arm.
"""

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


def resolve_arm(spec):
    """'NAME' (built-in) or 'NAME=PATH' -> (name, absolute checkpoint dir)."""
    name, _, path = spec.partition("=")
    if not ARM_NAME.match(name):
        raise SystemExit(f"arm name {name!r}: use letters, digits, '_', '.', '-'")
    if not path:
        if name not in BUILTIN_ARMS:
            raise SystemExit(f"arm {name!r} is not built in ({', '.join(BUILTIN_ARMS)}); give it as {name}=/path/to/checkpoint")
        path = paths.MODELS / BUILTIN_ARMS[name]
    return name, str(Path(path).expanduser().resolve())


def check_checkpoint(path):
    """Problems that would stop the server or the client tokenizer (empty list = fine)."""
    p = Path(path)
    if not p.is_dir():
        return [f"{path}: not a directory"]
    out = [f"{path}: missing {f}" for f in ("config.json", "tokenizer_config.json") if not (p / f).exists()]
    # The server container mounts only this directory: symlinks that leave it would be dangling inside.
    for f in p.iterdir():
        if f.is_symlink() and p not in f.resolve().parents:
            out.append(f"{f}: symlink outside the checkpoint directory (copy the file instead)")
    return out
