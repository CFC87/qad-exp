# SPDX-License-Identifier: Apache-2.0
"""NVIDIA's tool-call parser for Nemotron Nano 9B v2 (nemotron_toolcall_parser_no_streaming.py from the
nvidia/NVIDIA-Nemotron-Nano-9B-v2 model repository), ported to vLLM's newer module layout: the parsing logic is
unchanged; only the imports (the request class from vllm.entrypoints.openai.chat_completion.protocol, the delta and
tool-call classes from vllm.entrypoints.generate.base.protocol in vLLM 0.29 or vllm.entrypoints.openai.engine.protocol
in earlier releases, ToolParser from vllm.tool_parsers) and the constructor signature (tokenizer, tools=None) follow it.
Loaded with --tool-parser-plugin <this file> --tool-call-parser nemotron_json (eval/lib/models.py).
The model writes <TOOLCALL>[{"name": ..., "arguments": {...}}, ...]</TOOLCALL> after its reasoning. As upstream: the
first <TOOLCALL> block is parsed, a call without "name"/"arguments" is skipped, unparsable JSON returns the text as
content, and the upstream bracket repair prepends "]" when the closing bracket is missing (kept as released).
"""

import json
import re
from collections.abc import Sequence

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
try:  # vLLM 0.29 (the serving image)
    from vllm.entrypoints.generate.base.protocol import (
        DeltaMessage,
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
    )
except ImportError:  # earlier vLLM releases
    from vllm.entrypoints.openai.engine.protocol import (
        DeltaMessage,
        ExtractedToolCallInformation,
        FunctionCall,
        ToolCall,
    )
from vllm.logger import init_logger
from vllm.tool_parsers.abstract_tool_parser import ToolParser, ToolParserManager

logger = init_logger(__name__)


@ToolParserManager.register_module("nemotron_json")
class NemotronJSONToolParser(ToolParser):

    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)

        self.current_tool_name_sent: bool = False
        self.prev_tool_call_arr: list[dict] = []
        self.current_tool_id: int = -1
        self.streamed_args_for_tool: list[str] = []

        self.tool_call_start_token: str = "<TOOLCALL>"
        self.tool_call_end_token: str = "</TOOLCALL>"

        self.tool_call_regex = re.compile(r"<TOOLCALL>(.*?)</TOOLCALL>", re.DOTALL)

    def extract_tool_calls(
        self,
        model_output: str,
        request: ChatCompletionRequest,
    ) -> ExtractedToolCallInformation:

        if self.tool_call_start_token not in model_output:
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

        try:
            str_tool_calls = self.tool_call_regex.findall(model_output)[0].strip()
            if not str_tool_calls.startswith("["):
                str_tool_calls = "[" + str_tool_calls
            if not str_tool_calls.endswith("]"):
                str_tool_calls = "]" + str_tool_calls
            json_tool_calls = json.loads(str_tool_calls)
            tool_calls = []
            for tool_call in json_tool_calls:
                try:
                    tool_calls.append(ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=tool_call["name"],
                            arguments=json.dumps(tool_call["arguments"], ensure_ascii=False)
                            if isinstance(tool_call["arguments"], dict) else tool_call["arguments"],
                        ),
                    ))
                except Exception:
                    continue

            content = model_output[:model_output.rfind(self.tool_call_start_token)]

            return ExtractedToolCallInformation(
                tools_called=True,
                tool_calls=tool_calls,
                content=content if content else None,
            )

        except Exception:
            logger.exception(f"Error in extracting tool call from response. Response: {model_output}")
            return ExtractedToolCallInformation(
                tools_called=False,
                tool_calls=[],
                content=model_output,
            )

    def extract_tool_calls_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:

        raise NotImplementedError("Tool calling is not supported in streaming mode!")
