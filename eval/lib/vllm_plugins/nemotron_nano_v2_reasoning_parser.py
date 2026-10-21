# SPDX-License-Identifier: Apache-2.0
"""Reasoning parser for Nemotron Nano 9B v2, working on text: in this tokenizer "<think>" and "</think>" are ordinary
text (several vocabulary pieces), so vLLM's token-id parsers (deepseek_r1, qwen3) refuse to start ("could not locate
think start/end tokens in the tokenizer"). NVIDIA's card serves tool calling with no reasoning parser at all, which
leaves the reasoning and "</think>" in `content`: harmless for a client that splits it itself, but tau2 would pass the
reasoning to the user simulator as the agent's reply.

Split as deepseek_r1 does (vllm/reasoning/basic_parsers.py), so the tau2 / BFCL semantics match the Qwen3 runs: the chat
template opens the reasoning ("<think>\\n" at the end of the prompt), a leading "<think>" in the output is dropped,
everything before the first "</think>" is reasoning and the rest is content (None when empty); with no "</think>" the
whole output is reasoning and content is None (thinking cut off by max_tokens -> an empty reply). The tool parser
(nemotron_json) then runs on the content. Loaded with --reasoning-parser-plugin <this file> --reasoning-parser
nemotron_nano_v2_text (eval/lib/models.py). Streaming (unused here: tau2 and BFCL request whole replies) splits on the
accumulated text the same way, except that pieces of a "</think>" split across deltas are already out as reasoning.
Nothing here uses structured outputs, so extract_content_ids returns [] as vLLM's own text-based parser (olmo3) does.
"""

from collections.abc import Sequence

try:  # vLLM 0.29
    from vllm.entrypoints.generate.base.protocol import DeltaMessage
except ImportError:  # earlier vLLM releases
    from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParser, ReasoningParserManager

START, END = "<think>", "</think>"


@ReasoningParserManager.register_module("nemotron_nano_v2_text")
class NemotronNanoV2TextReasoningParser(ReasoningParser):

    @property
    def reasoning_start_str(self) -> str:
        return START

    @property
    def reasoning_end_str(self) -> str:
        return END

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        text = self.model_tokenizer.decode(list(input_ids))
        return text.rfind(END) > text.rfind(START)

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        return []

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        if model_output.startswith(START):
            model_output = model_output[len(START):]
        if END not in model_output:
            return model_output, None
        reasoning, _, content = model_output.partition(END)
        return reasoning, content or None

    def extract_reasoning_streaming(self, previous_text: str, current_text: str, delta_text: str,
                                    previous_token_ids: Sequence[int], current_token_ids: Sequence[int],
                                    delta_token_ids: Sequence[int]) -> DeltaMessage | None:
        if END in previous_text:
            return DeltaMessage(content=delta_text) if delta_text else None
        if END in current_text:
            reasoning, _, content = current_text.partition(END)
            done = len(previous_text.removeprefix(START)) if previous_text.startswith(START) else len(previous_text)
            reasoning = reasoning.removeprefix(START)[done:]
            return DeltaMessage(reasoning=reasoning or None, content=content or None)
        delta = delta_text if not current_text.startswith(START) or len(previous_text) >= len(START) else current_text[len(START):]
        return DeltaMessage(reasoning=delta) if delta else None
