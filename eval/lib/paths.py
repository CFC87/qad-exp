"""Repository locations used by the evaluation code. Stdlib only: every environment imports this module.

Defaults are repository-relative; the results directory can be moved with --results (run.py) or QADX_RESULTS.
"""

import os
import tomllib
from pathlib import Path

LIB = Path(__file__).resolve().parent
EVAL = LIB.parent
REPO = EVAL.parent
DATA = EVAL / "data"
THIRD_PARTY = REPO / "third_party"
MODELS = REPO / "models"
CACHE = REPO / ".cache"
RESULTS = Path(os.environ.get("QADX_RESULTS", EVAL / "results"))
NLTK_DIR = CACHE / "nltk_data"   # IFBench's NLTK data, fetched and checked by setup/setup.sh eval-data


def local_caches():
    """Keep downloads and caches inside the repository: Hugging Face (models, datasets, test cases), NLTK, uv. A token
    saved by `hf auth login` in the default location stays usable (HF_TOKEN_PATH). Inherited by every subprocess."""
    default_token = Path.home() / ".cache" / "huggingface" / "token"
    if "HF_HOME" not in os.environ:
        os.environ["HF_HOME"] = str(CACHE / "huggingface")
        if not os.environ.get("HF_TOKEN") and "HF_TOKEN_PATH" not in os.environ and default_token.exists():
            os.environ["HF_TOKEN_PATH"] = str(default_token)
    os.environ.setdefault("NLTK_DATA", str(NLTK_DIR))
    os.environ.setdefault("UV_CACHE_DIR", str(CACHE / "uv"))


local_caches()

# Python environments (all built by setup/setup.sh from env/versions.toml)
CLIENT_PY = REPO / ".venv-client" / "bin" / "python"   # driver, chat templates, math / MCQ / MRCR / LCB / LCB Pro graders
TOOLS_PY = REPO / ".venv-tools" / "bin" / "python"     # data preparation, LongBench Pro metrics
BFCL_PY = REPO / ".venv-bfcl" / "bin" / "python"       # BFCL multi-turn client and evaluator
TAU2_DIR = THIRD_PARTY / "tau2-bench"
TAU2_PY = TAU2_DIR / ".venv" / "bin" / "python"        # tau2-bench's own locked environment
IFBENCH_DIR = THIRD_PARTY / "IFBench"
IFBENCH_PY = IFBENCH_DIR / ".venv" / "bin" / "python"  # IFBench's own locked environment
LBP_MODULES = THIRD_PARTY / "longcontext" / "LongBench-Pro" / "modules"
LCBPRO_DIR = THIRD_PARTY / "LiveCodeBench-Pro"


def versions():
    return tomllib.loads((REPO / "env" / "versions.toml").read_text())


def vllm_image():
    """The serving image by registry digest (tags are mutable)."""
    i = versions()["images"]["vllm"]
    return f"{i['ref']}@{i['digest']}"


def logs_dir(results=None):
    return Path(results or RESULTS) / "logs"
