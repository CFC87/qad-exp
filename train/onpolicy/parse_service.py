"""Reasoning / tool-call parsing of multi-turn rollout turns by vLLM 0.29's own parser -- the code the evaluation servers
run (Nemotron 3 Nano is evaluated with --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder, i.e.
vllm.parser.nemotron_v3.NemotronV3Parser, which parses both). The rollout controller's Python has vLLM 0.25.1,
whose copy of these parsers differs (parameter values stripped instead of one wrapping newline trimmed, no chat-turn
boundary tokens), so the parsing runs here, inside the vLLM 0.29 image, CPU only:

    docker run --rm --net=host -v <tokenizer dir>:/tok:ro -v <dir of this file>:/ops:ro --entrypoint python3 \
        vllm/vllm-openai:v0.29.0 /ops/parse_service.py --port 8295 --parser nemotron_v3

POST /parse {"text": completion text after the generation prompt (stop token removed), "tools": [OpenAI tool dicts]}
  -> {"reasoning": str | null, "content": str | null, "tool_calls": [{"name": str, "arguments": JSON string}]}
GET /health -> ok. Each request builds its parser (the parser keeps per-request state), as the server does per request.
"""
import argparse, json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PARSERS = {"nemotron_v3": ("vllm.parser.nemotron_v3", "NemotronV3Parser")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8295)
    ap.add_argument("--tok", default="/tok")
    ap.add_argument("--parser", default="nemotron_v3", choices=list(PARSERS))
    a = ap.parse_args()
    import importlib
    from transformers import AutoTokenizer
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    tok = AutoTokenizer.from_pretrained(a.tok)
    mod, cls = PARSERS[a.parser]
    Parser = getattr(importlib.import_module(mod), cls)

    def parse(text, tools):
        req = ChatCompletionRequest(model="student", messages=[{"role": "user", "content": "-"}], tools=tools or None)
        p = Parser(tok, tools=req.tools)
        reasoning, content, calls = p.parse(text, req, enable_auto_tools=bool(tools))
        return dict(reasoning=reasoning, content=content,
                    tool_calls=[dict(name=c.name, arguments=c.arguments) for c in (calls or [])])

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            self._send(200, "ok") if self.path == "/health" else self._send(404, "no")

        def do_POST(self):
            if self.path != "/parse":
                return self._send(404, "no")
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                self._send(200, parse(req["text"], req.get("tools")))
            except Exception as e:  # the caller treats it as an unparsable turn
                self._send(500, dict(error=repr(e)[:300]))

    t0 = time.time()
    parse("x\n</think>\nok", [])  # warm up
    print("parse service (%s, vLLM 0.29) on port %d, ready in %.0f s" % (a.parser, a.port, time.time() - t0), flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main())
