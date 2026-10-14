"""`tau2 ...` (tau2-bench v1.0.1, third_party/tau2-bench) with three runtime patches; runs in tau2-bench's own env.

The tau2 sources are not modified; this wrapper patches modules after import and then calls tau2's CLI.
1. The retail natural-language-assertion judge is hardcoded to gpt-4.1-2025-04-14 at T=0 (tau2/config.py).
   TAU2_NL_JUDGE_MODEL / TAU2_NL_JUDGE_ARGS (JSON) replace the model and its args, so the judge is the same model as
   the user simulator (gpt-5.2 without reasoning at T=0). Through an OpenAI-compatible proxy whose model id litellm
   does not know, litellm silently drops a top-level `temperature`, so the args carry it in extra_body.
2. The judge's reply is parsed with a bare json.loads; a reply wrapped in a ```json fence would raise and make tau2
   re-run the whole simulation, so a surrounding fence is stripped first.
3. An agent reply with neither text nor a tool call makes tau2 raise and drop the message; its usage, finish_reason
   and reasoning tail are appended to the jsonl file TAU2_EMPTY_LOG first (diagnostics only).
Secrets come from the environment (OPENAI_API_KEY / OPENAI_API_BASE), never from --*-llm-args: tau2 writes the llm
args into results.json.
"""

import json
import os
import re
import sys

import tau2.evaluator.evaluator_nl_assertions as nl
from tau2.cli import main

if os.environ.get("TAU2_NL_JUDGE_MODEL"):
    nl.DEFAULT_LLM_NL_ASSERTIONS = os.environ["TAU2_NL_JUDGE_MODEL"]
    nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = json.loads(os.environ.get("TAU2_NL_JUDGE_ARGS", "{}"))


class _FenceTolerantJSON:
    def __getattr__(self, name):
        return getattr(json, name)

    @staticmethod
    def loads(s, *args, **kwargs):
        if isinstance(s, str):
            m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", s, re.S)
            if m:
                s = m.group(1)
        return json.loads(s, *args, **kwargs)


nl.json = _FenceTolerantJSON()

import tau2.data_model.message as _msg  # noqa: E402

_validate = _msg.AssistantMessage.validate


def _logging_validate(self):
    try:
        return _validate(self)
    except ValueError:
        path = os.environ.get("TAU2_EMPTY_LOG")
        if path:
            raw = self.raw_data if isinstance(self.raw_data, dict) else {}
            ch = (raw.get("choices") or [{}])[0] if raw else {}
            m = ch.get("message") or {}
            rec = dict(usage=self.usage, finish_reason=ch.get("finish_reason"),
                       reasoning_tail=str(m.get("reasoning_content") or m.get("reasoning") or "")[-400:],
                       content=str(m.get("content"))[:200])
            with open(path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        raise


_msg.AssistantMessage.validate = _logging_validate

if __name__ == "__main__":
    sys.exit(main())
