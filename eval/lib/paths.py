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
