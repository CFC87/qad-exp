"""BFCL multi-turn handlers for checkpoints on a local vLLM server (bfcl-eval 2026.3.23 = gorilla 6ea5797).

Import this module before bfcl_eval's CLI runs (bfcl_run.py does). It registers three handlers (no "_" in the names:
bfcl's evaluate maps result-directory names back with "_" -> "/"): qadx/qwen3-ct for the Qwen3 models (below), and
qadx/nemotron-nano-v2-ct / qadx/nemotron3-nano-ct for the Nemotrons (NemotronChatHandler: BFCL's FC path on the server's
chat endpoint, parsed server-side with the tau2 parsers).

qadx/qwen3-ct:

  - Renders with the checkpoint's own chat template: tokenizer.apply_chat_template(messages, tools=[{"type": "function",
    "function": <BFCL's FC-converted schema>}], enable_thinking=True); assistant history as {content,
    reasoning_content, tool_calls}.
  - Sends the rendered text to /v1/completions and splits / parses the output exactly like vLLM's qwen3 reasoning parser
    and hermes tool parser would (all-or-nothing per step).
  - miss_func holdout functions are added to the tool list with BFCL's FC user message (the FC path), not pasted
    into the user turn as a Python repr (the stock prompting path).

Query patches (the stock OSS path has none of them; all handlers):
  - max_tokens = min(QADX_BFCL_MAX_TOKENS [32768], QADX_BFCL_MAX_MODEL_LEN [131072] - prompt_tokens); a step whose prompt
    leaves < QADX_BFCL_MIN_NEW_TOKENS [1024] fails the entry as context_overflow (stock: requests 1000 tokens anyway and
    the server returns 400)
  - context = the served --max-model-len, not the checkpoint's max_position_embeddings
  - sampling sent explicitly from QADX_BFCL_SAMPLING (JSON; default Qwen3 thinking T=0.6 / top_p 0.95 / top_k 20 /
    min_p 0 / presence 0 / repetition 1.0); the CLI --temperature is ignored
  - request model = QADX_BFCL_SERVED_MODEL (stock sends the local checkpoint path as the model name)
  - add_special_tokens=False (the rendered template is sent verbatim)
  - retries on connection / timeout / 5xx (not on 4xx), QADX_BFCL_RETRIES [3]
  - per-step log in the result record, key "qadx_steps": turn, prompt_tokens (client count), server prompt /
    completion tokens, max_tokens, finish_reason ("length" = truncated), parse status
  - QADX_BFCL_APPLY_EXCLUDED [1]: drop the entry's "excluded_function" docs (the data holds out cp/mv/rm on 18
    file-system entries per category in that field, but no code at 6ea5797 reads it)

    python bfcl_handler.py --first-request <checkpoint> [--handler NAME]   print the first request of multi_turn_base_0
                                                                           (no server)
"""

import copy
import json
import os
import sys
import threading
import time
import traceback
from typing import Any

import httpx
import openai
from overrides import override

from bfcl_eval.constants.default_prompts import DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC
from bfcl_eval.constants.enums import ModelStyle
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI
from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from bfcl_eval.model_handler.base_handler import BaseHandler
from bfcl_eval.model_handler.local_inference.qwen_fc import QwenFCHandler
from bfcl_eval.model_handler.utils import convert_to_function_call, convert_to_tool

HANDLER = "qadx/qwen3-ct"
CHAT_HANDLERS = {"qadx/nemotron-nano-v2-ct": "Nemotron Nano 9B v2 (vLLM chat endpoint, nemotron_json + nemotron_nano_v2_text)",
                 "qadx/nemotron3-nano-ct": "Nemotron 3 Nano (vLLM chat endpoint, qwen3_coder + nemotron_v3)"}
STEPS_KEY = "qadx_steps"
DEFAULT_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0)


def _cfg():
    return dict(
        served_model=os.getenv("QADX_BFCL_SERVED_MODEL", "eval-model"),
        max_model_len=int(os.getenv("QADX_BFCL_MAX_MODEL_LEN", "131072")),
        max_tokens=int(os.getenv("QADX_BFCL_MAX_TOKENS", "32768")),
        min_new_tokens=int(os.getenv("QADX_BFCL_MIN_NEW_TOKENS", "1024")),
        sampling=json.loads(os.getenv("QADX_BFCL_SAMPLING", json.dumps(DEFAULT_SAMPLING))),
        retries=int(os.getenv("QADX_BFCL_RETRIES", "3")),
        apply_excluded=os.getenv("QADX_BFCL_APPLY_EXCLUDED", "1") == "1",
        timeout=float(os.getenv("QADX_BFCL_TIMEOUT", "14400")),
    )


class ContextOverflow(RuntimeError):
    pass


_print_lock = threading.Lock()


def _say(msg):
    with _print_lock:
        print(f"[bfcl] {msg}", flush=True)


# ---------------------------------------------------------------- parsing like vLLM (qwen3 reasoning + hermes tools)

THINK_START, THINK_END, TOOL_START, TOOL_END = "<think>", "</think>", "<tool_call>", "</tool_call>"


def split_reasoning_qwen3(text):
    """(reasoning, content) as vLLM's qwen3 reasoning parser returns them for a thinking-mode request.

    The parser starts in the reasoning state; '</think>' ends it, and so does a '<tool_call>' (implicit end, the tag
    stays in content). With neither, everything is reasoning and content is None.
    """
    i, j = text.find(THINK_END), text.find(TOOL_START)
    if i < 0 and j < 0:
        reasoning, content = text, None
    elif i >= 0 and (j < 0 or i < j):
        reasoning, content = text[:i], text[i + len(THINK_END):]
    else:
        reasoning, content = text[:j], text[j:]
    if reasoning.startswith(THINK_START):
        reasoning = reasoning[len(THINK_START):]
    return reasoning, content


def parse_hermes(content):
    """vLLM's Hermes2ProToolParser.extract_tool_calls (non-streaming) on the post-reasoning content.

    Returns (calls, content_before_calls, error). calls is None when there is no '<tool_call>' or when any block fails
    (json.loads error, missing name/arguments): vLLM then returns no tool calls and the whole text as content. A
    trailing unclosed '<tool_call>' block is accepted. Whitespace-only content becomes None, as the delegating parser
    does in auto tool-choice mode.
    """
    if not content or TOOL_START not in content:
        return None, (content if content and content.strip() else None), None
    import re
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", content, re.DOTALL)
    try:
        raw = [json.loads(a if a else b) for a, b in blocks]
        calls = [{"name": r["name"], "arguments": r["arguments"]} for r in raw]
    except Exception as e:  # vLLM logs and falls back to plain content
        return None, content, f"{type(e).__name__}: {e}"
    before = content[: content.find(TOOL_START)]
    return calls, (before if before.strip() else None), None


def calls_to_execution(calls):
    """BFCL execution strings; arguments given as a JSON string are decoded (vLLM ships them as strings)."""
    out = []
    for c in calls:
        args = c["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError(f"arguments is not an object: {args!r}")
        out.append({c["name"]: args})
    return convert_to_function_call(out)


def diagnose(text):
    """Both parsers' view of one raw completion (diagnostics only).

    think_end: '</think>' present; strict: calls the stock regex + json.loads finds after '</think>'; hermes: 'none' |
    'ok' | 'error' and its call count on the vLLM-style split (what is executed); implicit_end: a call was taken from an
    unclosed think block (vLLM treats '<tool_call>' as the end of reasoning).
    """
    post = text.split(THINK_END)[-1].lstrip("\n") if THINK_END in text else text
    strict = len(QwenFCHandler._extract_tool_calls(post))
    reasoning, content = split_reasoning_qwen3(text)
    calls, _, err = parse_hermes(content)
    return {
        "think_end": THINK_END in text,
        "strict": strict,
        "hermes": "error" if err else ("ok" if calls else "none"),
        "hermes_n": len(calls or []),
        "implicit_end": bool(calls) and THINK_END not in text,
    }


# ---------------------------------------------------------------- query patches


class _QueryPatches:
    """Overrides of the stock OSS prompting path. Must come first in the MRO."""

    def inference(self, test_entry: dict, include_input_log: bool, exclude_state_log: bool):
        cfg = _cfg()
        steps = []
        test_entry["_steps"] = steps
        excluded = test_entry.get("excluded_function") or []
        if cfg["apply_excluded"] and excluded:
            before = len(test_entry["function"])
            test_entry["function"] = [f for f in test_entry["function"] if f["name"] not in excluded]
            steps.append({"event": "excluded_functions", "names": excluded, "removed": before - len(test_entry["function"])})
        try:
            result, metadata = super().inference(test_entry, include_input_log, exclude_state_log)
        except Exception as e:
            # Same shape as bfcl's multi_threaded_inference error record, plus the step log; scored as
            # multi_turn:inference_error.
            result = f"Error during inference: {e}"
            metadata = {"traceback": traceback.format_exc(), "qadx_error": type(e).__name__}
        metadata[STEPS_KEY] = steps
        metadata["qadx_config"] = {k: v for k, v in cfg.items() if k != "timeout"}
        return result, metadata

    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        data = super()._pre_query_processing_prompting(test_entry)
        data["_steps"] = test_entry["_steps"]
        data["_turn"] = -1
        data["_missed"] = test_entry.get("missed_function", {})
        return data

    def add_first_turn_message_prompting(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["_turn"] += 1
        return super().add_first_turn_message_prompting(inference_data, first_turn_message)

    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["_turn"] += 1
        return super()._add_next_turn_user_message_prompting(inference_data, user_message)

    def _query_prompting(self, inference_data: dict):
        cfg = _cfg()
        prompt = self._format_prompt(inference_data["message"], inference_data["function"])
        inference_data["inference_input_log"] = {"formatted_prompt": prompt}
        n = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        room = cfg["max_model_len"] - n
        rec = {"turn": inference_data["_turn"], "prompt_tokens": n}
        inference_data["_steps"].append(rec)
        if room < cfg["min_new_tokens"]:
            rec.update(max_tokens=0, finish_reason="context_overflow")
            raise ContextOverflow(f"prompt has {n} tokens; {room} left of {cfg['max_model_len']} < {cfg['min_new_tokens']}")
        max_tokens = min(cfg["max_tokens"], room)
        rec["max_tokens"] = max_tokens
        s = dict(cfg["sampling"])
        std = {k: s.pop(k) for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "seed") if k in s}
        extra = dict(s, add_special_tokens=False)
        t0 = time.time()
        for attempt in range(cfg["retries"] + 1):
            try:
                resp = self.client.completions.create(
                    model=cfg["served_model"], prompt=prompt, max_tokens=max_tokens,
                    extra_body=extra, timeout=cfg["timeout"], **std,
                )
                break
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                if attempt == cfg["retries"]:
                    rec["finish_reason"] = "http_error"
                    raise
                _say(f"retry {attempt + 1} after {type(e).__name__}: {e}"[:300])
                time.sleep(10 * (attempt + 1))
        latency = time.time() - t0
        ch = resp.choices[0]
        rec.update(server_prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens,
                   finish_reason=ch.finish_reason, latency=round(latency, 3))
        if ch.finish_reason == "length":
            _say(f"TRUNCATED turn {rec['turn']}: {resp.usage.completion_tokens} tokens hit max_tokens={max_tokens} (prompt {n})")
        if resp.usage.prompt_tokens != n:
            rec["prompt_token_mismatch"] = True
        rec["parse"] = diagnose(ch.text or "")
        return resp, latency


class Qwen3ChatTemplateHandler(_QueryPatches, QwenFCHandler):
    """Renders with the checkpoint's own chat template, parses like vLLM (qwen3 reasoning + hermes)."""

    @staticmethod
    def to_openai_tools(functions):
        # BFCL's FC conversion (dict->object, float->number + format/description note), wrapped as OpenAI tools; key
        # order name, description, parameters, response = what vLLM's ChatCompletionToolsParam.model_dump() passes to
        # the template (extra fields such as "response" are kept by vLLM).
        return convert_to_tool(copy.deepcopy(functions), GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)

    @override
    def _format_prompt(self, messages, function):
        tools = self.to_openai_tools(function) if function else None
        return self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )

    @override
    def _add_next_turn_user_message_prompting(self, inference_data: dict, user_message: list[dict]) -> dict:
        turn = inference_data["_turn"] + 1
        holdout = inference_data["_missed"].get(str(turn))
        if holdout:
            # bfcl's prompting loop has already built the stock "<python repr of docs>\nI have updated ..." turn;
            # replace it with the FC path's behaviour: new tools in the system prompt, generic user message.
            inference_data["function"].extend(holdout)
            user_message = [{"role": "user", "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC}]
            inference_data["_steps"].append({"event": "holdout_added", "turn": turn, "names": [f["name"] for f in holdout]})
        return super()._add_next_turn_user_message_prompting(inference_data, user_message)

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        text = api_response.choices[0].text
        reasoning, content = split_reasoning_qwen3(text)
        calls, before, err = parse_hermes(content)
        # History content: vLLM returns the post-</think> text with its leading "\n\n"; the Qwen3 template prints
        # past-turn content verbatim after "<|im_start|>assistant\n", so strip leading newlines the way the template
        # itself does when it splits reasoning out of content (and as the stock handler does).
        hist = before if calls else content
        msg = {"role": "assistant", "content": hist.lstrip("\n") if hist else hist, "reasoning_content": reasoning}
        if calls:
            msg["tool_calls"] = [{"type": "function", "function": c} for c in calls]
        return {
            # what the evaluator re-decodes with decode_execute: the post-reasoning content, verbatim
            "model_responses": content or "",
            "reasoning_content": reasoning or "",
            "model_responses_message_for_chat_history": msg,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def decode_execute(self, result, has_tool_call_tag):
        calls, _, err = parse_hermes(result)
        if err:
            raise ValueError(f"tool call parse error: {err}")
        return calls_to_execution(calls or [])

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        calls, _, err = parse_hermes(result)
        if err:
            raise ValueError(f"tool call parse error: {err}")
        return [{c["name"]: c["arguments"]} for c in calls or []]


# ---------------------------------------------------------------- chat endpoint, server-side parsing (Nemotrons)


class NemotronChatHandler(_QueryPatches, OpenAICompletionsHandler):
    """BFCL's FC path against the local vLLM chat endpoint (the Nemotron models): the server renders the checkpoint's own
    template with the tools and parses reasoning and tool calls with the same vLLM parsers as tau2 (run_phase starts it
    with the model's --tool-call-parser / --reasoning-parser, models.py bfcl_server_parsing). Used where re-implementing
    the parser client-side is not sensible (Nano 9B v2's <TOOLCALL> JSON with NVIDIA's plugin, Nemotron 3's qwen3_coder
    XML through vLLM's parser engine).

    Same query patches as qadx/qwen3-ct where they apply: sampling sent explicitly, max_tokens = min(cap, window - prompt)
    with the prompt counted by the server's own /tokenize (identical rendering), context_overflow when fewer than
    QADX_BFCL_MIN_NEW_TOKENS remain, retries on connection / timeout / 5xx, per-step log, excluded functions dropped.
    Holdout functions follow BFCL's FC path (added to the tools, generic user message). History: assistant content and
    tool_calls plus the reasoning as "reasoning" (vLLM hands it to the template, which decides what to show).
    """

    def __init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs):
        BaseHandler.__init__(self, model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.model_style = ModelStyle.OPENAI_COMPLETIONS
        host, port = os.getenv("LOCAL_SERVER_ENDPOINT", "127.0.0.1"), os.getenv("LOCAL_SERVER_PORT", "8100")
        self.base_url = f"http://{host}:{port}"
        self.client = openai.OpenAI(base_url=f"{self.base_url}/v1", api_key="EMPTY", max_retries=0)

    def _pre_query_processing_FC(self, inference_data: dict, test_entry: dict) -> dict:
        data = super()._pre_query_processing_FC(inference_data, test_entry)
        data["_steps"] = test_entry["_steps"]
        data["_turn"] = -1
        return data

    def add_first_turn_message_FC(self, inference_data: dict, first_turn_message: list[dict]) -> dict:
        inference_data["_turn"] += 1
        return super().add_first_turn_message_FC(inference_data, first_turn_message)

    def _add_next_turn_user_message_FC(self, inference_data: dict, user_message: list[dict]) -> dict:
        inference_data["_turn"] += 1
        return super()._add_next_turn_user_message_FC(inference_data, user_message)

    def _count_prompt(self, messages, tools, cfg):
        body = {"model": cfg["served_model"], "messages": messages, "add_generation_prompt": True}
        if tools:
            body["tools"] = tools
        for attempt in range(cfg["retries"] + 1):
            try:
                r = httpx.post(f"{self.base_url}/tokenize", json=body, timeout=600)
                r.raise_for_status()
                return r.json()["count"]
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                if attempt == cfg["retries"]:
                    raise
                _say(f"tokenize retry {attempt + 1} after {type(e).__name__}: {e}"[:300])
                time.sleep(10 * (attempt + 1))

    def _query_FC(self, inference_data: dict):
        cfg = _cfg()
        messages, tools = inference_data["message"], inference_data["tools"]
        inference_data["inference_input_log"] = {"message": repr(messages), "tools": tools}
        rec = {"turn": inference_data["_turn"]}
        inference_data["_steps"].append(rec)
        n = self._count_prompt(messages, tools, cfg)
        rec["prompt_tokens"] = n
        room = cfg["max_model_len"] - n
        if room < cfg["min_new_tokens"]:
            rec.update(max_tokens=0, finish_reason="context_overflow")
            raise ContextOverflow(f"prompt has {n} tokens; {room} left of {cfg['max_model_len']} < {cfg['min_new_tokens']}")
        max_tokens = min(cfg["max_tokens"], room)
        rec["max_tokens"] = max_tokens
        s = dict(cfg["sampling"])
        std = {k: s.pop(k) for k in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "seed") if k in s}
        kwargs = dict(model=cfg["served_model"], messages=messages, max_tokens=max_tokens, extra_body=s,
                      timeout=cfg["timeout"], **std)
        if tools:
            kwargs["tools"] = tools
        t0 = time.time()
        for attempt in range(cfg["retries"] + 1):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as e:
                if attempt == cfg["retries"]:
                    rec["finish_reason"] = "http_error"
                    raise
                _say(f"retry {attempt + 1} after {type(e).__name__}: {e}"[:300])
                time.sleep(10 * (attempt + 1))
        latency = time.time() - t0
        ch = resp.choices[0]
        rec.update(server_prompt_tokens=resp.usage.prompt_tokens, completion_tokens=resp.usage.completion_tokens,
                   finish_reason=ch.finish_reason, latency=round(latency, 3))
        if ch.finish_reason == "length":
            _say(f"TRUNCATED turn {rec['turn']}: {resp.usage.completion_tokens} tokens hit max_tokens={max_tokens} (prompt {n})")
        if resp.usage.prompt_tokens != n:
            rec["prompt_token_mismatch"] = True
        msg = ch.message
        rec["parse"] = {"server_tool_calls": len(msg.tool_calls or []), "content": bool(msg.content and msg.content.strip()),
                        "reasoning": _reasoning(msg) is not None}
        return resp, latency

    def _parse_query_response_FC(self, api_response: Any) -> dict:
        msg = api_response.choices[0].message
        reasoning = _reasoning(msg)
        calls = msg.tool_calls or []
        hist = {"role": "assistant", "content": msg.content}
        if calls:
            hist["tool_calls"] = [{"id": c.id, "type": "function",
                                   "function": {"name": c.function.name, "arguments": c.function.arguments}} for c in calls]
        if reasoning is not None:
            hist["reasoning"] = reasoning
        return {
            # decode_ast / decode_execute (OpenAI FC): [{name: arguments JSON}] per call; plain text when none
            "model_responses": [{c.function.name: c.function.arguments} for c in calls] if calls else (msg.content or ""),
            "reasoning_content": reasoning or "",
            "model_responses_message_for_chat_history": hist,
            "tool_call_ids": [c.id for c in calls],
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }


def _reasoning(msg):
    """The parsed reasoning of a chat reply: vLLM returns it as `reasoning` (older releases: `reasoning_content`)."""
    extra = getattr(msg, "model_extra", None) or {}
    for k in ("reasoning", "reasoning_content"):
        v = extra.get(k, getattr(msg, k, None))
        if v is not None:
            return v
    return None


def register():
    base = MODEL_CONFIG_MAPPING["Qwen/Qwen3-8B-FC"]
    MODEL_CONFIG_MAPPING[HANDLER] = ModelConfig(
        model_name=HANDLER, display_name="Qwen3 (chat template + hermes)", url=base.url, org="qad-exp",
        license=base.license, model_handler=Qwen3ChatTemplateHandler, input_price=None, output_price=None,
        is_fc_model=True, underscore_to_dot=False,
    )
    for name, disp in CHAT_HANDLERS.items():
        MODEL_CONFIG_MAPPING[name] = ModelConfig(
            model_name=name, display_name=disp, url=base.url, org="qad-exp", license=base.license,
            model_handler=NemotronChatHandler, input_price=None, output_price=None, is_fc_model=True,
            underscore_to_dot=False,
        )


register()


# ---------------------------------------------------------------- offline request capture (dry runs, tests)


class _Recorder:
    """Stands in for the OpenAI client: records each completions.create / chat.completions.create call and answers every
    step with a short final message (no tool call), so each turn of an entry costs exactly one request."""

    def __init__(self, reply="<think>\nok\n</think>\n\nDone."):
        self.calls, self.reply = [], reply
        outer = self
        from types import SimpleNamespace as NS

        class _C:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                return NS(choices=[NS(text=outer.reply, finish_reason="stop")],
                          usage=NS(prompt_tokens=0, completion_tokens=1))

        class _Chat:
            @staticmethod
            def create(**kw):
                outer.calls.append(copy.deepcopy(kw))  # bfcl keeps appending to the same message list
                msg = NS(content="Done.", tool_calls=None, model_extra={"reasoning": "ok"})
                return NS(choices=[NS(message=msg, finish_reason="stop")], usage=NS(prompt_tokens=0, completion_tokens=1))
        self.completions = _C()
        self.chat = NS(completions=_Chat())


def offline_requests(checkpoint, entry_ids, handler_name=HANDLER):
    """Run bfcl's own multi-turn loop for some entries against a recording client; returns {entry id: [request kwargs]}.
    For the chat-endpoint handlers the prompt is counted client-side with the checkpoint's chat template (the run uses
    the server's /tokenize)."""
    from transformers import AutoTokenizer
    from bfcl_eval._llm_response_generation import build_handler
    from bfcl_eval.utils import extract_test_category_from_id, load_dataset_entry

    h = build_handler(handler_name, 0.001)
    chat = isinstance(h, NemotronChatHandler)  # remote code of the Nemotron configs is not needed for the tokenizer
    h.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=not chat)
    h.model_path_or_id = checkpoint
    if chat:
        h._count_prompt = lambda messages, tools, cfg: len(h.tokenizer.apply_chat_template(
            messages, tools=tools or None, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"])
    cats = {}
    out = {}
    for eid in entry_ids:
        cat = extract_test_category_from_id(eid)
        if cat not in cats:
            cats[cat] = {e["id"]: e for e in load_dataset_entry(cat)}
        rec = _Recorder()
        h.client = rec
        h.inference(copy.deepcopy(cats[cat][eid]), include_input_log=False, exclude_state_log=True)
        out[eid] = rec.calls
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["--first-request"]:
        handler = sys.argv[sys.argv.index("--handler") + 1] if "--handler" in sys.argv else HANDLER
        kw = offline_requests(sys.argv[2], ["multi_turn_base_0"], handler)["multi_turn_base_0"][0]
        if "prompt" in kw:
            p = kw["prompt"]
            shown = dict(kw, prompt=p if len(p) <= 600 else p[:300] + f" [... {len(p) - 600} chars ...] " + p[-300:])
            print("bfcl first request (multi_turn_base_0, turn 0): POST /v1/completions "
                  + json.dumps(shown, ensure_ascii=False, default=str))
        else:
            shown = dict(kw, tools=f"[{len(kw.get('tools') or [])} tools: "
                                   + ", ".join(t["function"]["name"] for t in kw.get("tools") or []) + "]")
            print("bfcl first request (multi_turn_base_0, turn 0; max_tokens from a client-side prompt count): "
                  "POST /v1/chat/completions " + json.dumps(shown, ensure_ascii=False, default=str))
