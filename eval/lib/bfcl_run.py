"""Entry point in .venv-bfcl: register the qadx/qwen3-ct handler (bfcl_handler.py), then run bfcl's own CLI.

  BFCL_PROJECT_ROOT=<dir> LOCAL_SERVER_PORT=8100 QADX_BFCL_MAX_MODEL_LEN=40960 ... \
  python bfcl_run.py generate --model qadx/qwen3-ct --test-category multi_turn \
      --skip-server-setup --local-model-path <checkpoint dir> --num-threads 32 --result-dir result_j<j>
  python bfcl_run.py evaluate --model qadx/qwen3-ct --test-category multi_turn \
      --result-dir result_j<j> --score-dir score_j<j>

One process per replicate for generate AND for evaluate: bfcl caches backend instances in module globals keyed by
model name + entry id, so a second pass in the same process would reuse mutated state.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bfcl_handler  # noqa: E402,F401  (registers qadx/qwen3-ct)
from bfcl_eval.__main__ import cli  # noqa: E402

if __name__ == "__main__":
    cli()
