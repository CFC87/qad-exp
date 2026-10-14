"""The vLLM server of one phase: the pinned docker image on one GPU, the same flags for every arm.

    docker run --rm --name qadx_vllm_gpu<i> --gpus device=<i> --ipc=host --net=host
        -v <ckpt>:<ckpt>:ro [-v <repo>/.cache/vllm/gpu<i>:/root/.cache/vllm]
        vllm/vllm-openai@sha256:... <ckpt>
        --served-model-name eval-model --host 127.0.0.1 --port <8100+i> --max-model-len <N> --language-model-only
        --gpu-memory-utilization 0.90 --max-num-seqs 256 --no-enable-prefix-caching --seed 0 <phase args>

The checkpoint directory is mounted at its host path, so the path the server loads is the path the client tokenizer
reads. Defaults that matter but are not passed: --kv-cache-dtype auto (FP8 E4M3 KV with scale 1.0 for checkpoints whose
hf_quant_config declares kv_cache_quant_algo FP8, i.e. NVIDIA's NVFP4 release and NVFP4 QAD exports; BF16 KV for the BF16
checkpoint) and --generation-config auto (the checkpoint's generation_config.json supplies any sampling field a request
omits). The server seed is fixed (--seed 0) and requests carry no seed, so with the fixed request order a rerun of the
same phase on the same build replays the same tokens closely; independent repeats need another server seed.
"""

import json
import os
import signal
import subprocess
import time
import urllib.request

import models as M
import paths

PORT_BASE = 8100
THROTTLE_PORT_BASE = 8190
PREFIX = os.environ.get("QADX_CONTAINER_PREFIX", "qadx")


def port(gpu):
    return PORT_BASE + int(gpu)


def container(gpu):
    return f"{PREFIX}_vllm_gpu{gpu}"


def phase_server_args(phase_cfg, kind):
    """Arguments appended for a phase, in the order the reference harness appends them."""
    args = []
    if phase_cfg["yarn"]:
        args += ["--hf-overrides", json.dumps(M.YARN)]
    if kind in ("tau2", "bfcl"):  # agent phases resend long, growing prefixes
        args += ["--enable-prefix-caching"]
    if kind == "tau2":  # tau2 expects tool calls and reasoning parsed server-side, as from an OpenAI endpoint
        args += ["--enable-auto-tool-choice", "--tool-call-parser", M.TOOL_PARSER, "--reasoning-parser", M.REASONING_PARSER]
        # tau2 sends no max_tokens: vLLM rejects a request whose prompt + max_tokens exceeds the window instead of
        # clamping, while the server default max_new_tokens is clamped to window - prompt.
        cap = int(phase_cfg["tasks"].split(",")[0].split(":")[2])
        args += ["--override-generation-config", json.dumps({"max_new_tokens": cap})]
    return args


def vllm_args(gpu, max_model_len, extra):
    return ["--served-model-name", M.SERVED_NAME, "--host", "127.0.0.1", "--port", str(port(gpu)),
            "--max-model-len", str(max_model_len), "--language-model-only", "--gpu-memory-utilization", "0.90",
            "--max-num-seqs", "256", "--no-enable-prefix-caching", "--seed", "0"] + list(extra)


def serve_cmd(checkpoint, gpu, max_model_len, extra, compile_cache=True):
    """The full docker command line. compile_cache mounts a per-GPU torch.compile / FlashInfer autotune cache under
    .cache/vllm (only startup time changes; one directory per GPU so concurrent servers never write the same files)."""
    mounts = ["-v", f"{checkpoint}:{checkpoint}:ro"]
    if compile_cache:
        mounts += ["-v", f"{paths.CACHE / 'vllm' / f'gpu{gpu}'}:/root/.cache/vllm"]
    return (["docker", "run", "--rm", "--name", container(gpu), "--gpus", f"device={gpu}", "--ipc=host", "--net=host"]
            + mounts + [paths.vllm_image(), checkpoint] + vllm_args(gpu, max_model_len, extra))


def start(cmd, gpu, logfile, compile_cache=True):
    subprocess.run(["docker", "rm", "-f", container(gpu)], capture_output=True)  # a container left by a killed run
    if compile_cache:
        (paths.CACHE / "vllm" / f"gpu{gpu}").mkdir(parents=True, exist_ok=True)
    logf = open(logfile, "w")
    return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)


def wait_healthy(proc, gpu, timeout_s=1800, log=print):
    t0 = time.time()
    url = f"http://127.0.0.1:{port(gpu)}/health"
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    log(f"server healthy after {time.time() - t0:.0f}s")
                    return
        except Exception:
            pass
        time.sleep(5)
    raise TimeoutError("server did not become healthy")


def stop(proc, gpu, log=print):
    subprocess.run(["docker", "rm", "-f", container(gpu)], capture_output=True)
    if proc is not None and proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        except ProcessLookupError:
            pass
    log("server stopped")


def gpu_memory_used(gpu):
    """MiB in use on one GPU (None if nvidia-smi is unavailable)."""
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def wait_gpu_free(gpu, limit_mib=2000, timeout_s=300):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        used = gpu_memory_used(gpu)
        if used is None or used < limit_mib:
            return True
        time.sleep(5)
    return False
