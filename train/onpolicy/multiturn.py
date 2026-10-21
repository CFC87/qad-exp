"""Multi-turn on-policy generation: the student writes EVERY assistant turn of a conversation, not only the last one,
so its history is its own.

A stored conversation (the subset row's prompt: depending on the set, everything up to the last user message or
everything before the final assistant message) has K-1 stored assistant turns plus the final turn. The student
regenerates all K turns in order; after each of its turns an environment supplies the next non-assistant messages:

  replay   the stored user / tool messages that followed the stored assistant turn, unchanged. Valid where the later
           user turns do not react to the assistant's content (scripted users: RL-Instruction-Following-Calendar-v2,
           the Structured-Outputs multistep rows, most of CFBench / SysBench) and where the
           stored turns call no tools. A student turn that calls a tool ends the episode there (nothing can answer it).
  tools    (interface only) an executor answers the student's own tool calls; user turns are replayed. Pass
           `executor(call) -> str` (a python sandbox, a repo container, a simulator; may be async).
  sim      an LLM plays the user and the tools (as NeMo-Gym's conversational_tool_use_simulation does), for
           conversations whose later user turns react to the assistant (customer service: SFT-Agentic-v2 /
           Agentic-v1 interactive_agent, the Conversational-Tool-Use-Pivot prefixes). The data has no scenario
           cards, so the stored conversation itself is the scenario: the user simulator is told to be the user of the
           recorded conversation (same goals, same facts, reacting to what the student actually says) and the tool
           simulator treats the recorded tool results as the backend's true state. A student call identical to a
           recorded one (same name and arguments) gets the recorded result without a simulator call. The episode ends
           when the user simulator says ###STOP### (goals handled) or ###TRANSFER### (handed to a human). Pass
           `sim(messages, max_tokens) -> str` (an async chat call to any OpenAI-compatible server).

  replay_tools  tool-calling conversations whose user turns do not react to the assistant (SFT-Agentic-v2 / Agentic-v1
           tool_calling): the stored user messages are replayed one per student turn that ends with a text reply (the
           student may make more or fewer tool calls than the recorded assistant did), and the student's tool calls are
           answered like in `sim` (the recorded result for an identical call, else the tool simulator).

Each student turn is rendered with the model's own chat template from the messages so far (re-tokenized: the
student's earlier turns enter the context as parsed messages, so Qwen3's template drops their reasoning exactly as it
would in deployment). Samples (the trainer's unit) per episode:

  last     one sample: prompt = the final render (with the student's earlier turns), response = the last turn's
           sampled tokens, hist_spans = the student's earlier assistant turns inside the prompt (history supervision:
           the teacher scores their visible tokens).
  each     one sample per student turn (prompt = the render before that turn, response = its sampled tokens, no
           history spans): every turn's reasoning is scored too, at the cost of repeating the prefix per turn.
  cut      one sample at a random student turn k (uniform over the recorded conversation's assistant turns + 1; the
           last generated turn if the episode ends first), like `last` (history spans) but the episode stops at k:
           same training cost per conversation as single-turn data, the turn can fall anywhere (also after the
           student's own mistakes), and nothing after k is generated.

An episode stops early (and keeps what it has) when a turn is truncated, cannot be parsed, calls a tool nothing can
answer, or the next prompt would leave fewer than `min_gen` tokens of the window.
"""
import asyncio
import collections
import inspect
import json
import time
import re

import numpy as np

from prompts import HIST_MARK, hist_char_spans

# as vLLM's hermes tool parser (τ² / BFCL evaluations, Qwen3): a <tool_call> without its closing tag is a call too (a
# closed-tags-only rule passes such a call to the customer as text)
TOOL_CALL = re.compile(r'<tool_call>(.*?)</tool_call>|<tool_call>(.*)', re.S)
# a transfer-to-human / escalation tool (the same pattern the τ² cut sets were selected with)
TRANSFER_TOOL = re.compile(r'^(?!(get|check|list|view|lookup|search|find|fetch|read|query|retrieve)_)'
                          r'.*(transfer.*(human|agent|representative)|escalat)', re.I)  # not check_claim_escalation
READ_TOOL = re.compile(r'^(get|find|list|search|check|lookup|look_up|view|fetch|read|query|retrieve|calculate|compute|'
                       r'validate|verify|show|describe|count|estimate|compare|think)(_|[A-Z]|$)')


def _is_error(txt):
    """a tool result that reports a failure (τ²: "Error: ..."; JSON with an error / failed status)"""
    t = (txt or '').strip()
    if t.lower().startswith('error'):
        return True
    try:
        j = json.loads(t)
    except Exception:
        return False
    return isinstance(j, dict) and bool(j.get('error') or str(j.get('status', '')).lower() in ('error', 'failed', 'failure')
                                        or j.get('success') is False)


def _is_write(call, result):
    """a call that changes the backend's state: not read-like by name, not a transfer, and it did not fail"""
    name = str((call.get('function') or call).get('name') or '')
    return not READ_TOOL.search(name) and not TRANSFER_TOOL.search(name) and not _is_error(result)


def parse_assistant(text):
    """Completion text after the generation prompt -> (chat message, ok), as vLLM 0.25.1's qwen3 reasoning parser plus
    hermes tool parser split it (the τ² / BFCL evaluations; compared case by case against them):
      - reasoning ends at the first </think>, or at the first <tool_call> if that comes earlier (vLLM then executes a
        call drafted inside the reasoning); with neither, the whole text is reasoning and the reply is empty;
      - stray </think> after the first are dropped from the content;
      - content = the text before the first <tool_call>; a <tool_call> without its closing tag is a call too; any call
        that is not a JSON object with a string name and arguments -> no calls, the whole rest is content, ok False.
    An empty reply (no content, no call; τ² fails the simulation on it) is ok False too: like a broken call it is
    resampled, then the episode stops there. Without this split an unclosed <think> stays in the content and the
    student copies it from its own history (8 % of the BF16 model's messages in a test run)."""
    txt = text
    for stop in ('<|im_end|>', '<|endoftext|>'):
        txt = txt.split(stop)[0]
    if txt.startswith('<think>'):
        txt = txt[len('<think>'):]
    e, t = txt.find('</think>'), txt.find('<tool_call>')
    if e >= 0 and (t < 0 or e < t):
        reasoning, rest = txt[:e], txt[e + len('</think>'):].replace('</think>', '')
    elif t >= 0:
        reasoning, rest = txt[:t], txt[t:]
    else:
        reasoning, rest = txt, ''
    calls, ok, content = [], True, rest
    if '<tool_call>' in rest:
        try:
            for a, b in TOOL_CALL.findall(rest):
                j = json.loads(a or b)
                if not isinstance(j, dict) or not isinstance(j.get('name'), str) or 'arguments' not in j:
                    raise ValueError('not a tool call object')
                calls.append({'type': 'function', 'function': {'name': j['name'], 'arguments': j['arguments']}})
            content = rest[:rest.find('<tool_call>')]
        except Exception:
            calls, ok = [], False
    msg = {'role': 'assistant', 'content': content.strip()}
    if reasoning.strip('\n'):
        msg['reasoning_content'] = reasoning.strip('\n')
    if calls:
        msg['tool_calls'] = calls
    if not calls and not msg['content']:
        ok = False
    return msg, ok


N9_TOOLCALL = re.compile(r'<TOOLCALL>(.*?)</TOOLCALL>', re.S)


def parse_assistant_n9(text):
    """Nemotron Nano 9B V2: completion text after the generation prompt (which ends with "<think>\n") -> (chat message,
    ok), as the τ² / BFCL evaluations split it: the evaluation harness's nemotron_nano_v2_text reasoning parser, then
    NVIDIA's nemotron_json tool parser on the content (compared case by case against both):
      - a leading "<think>" is dropped; reasoning = the text before the first "</think>", content = the rest (None when
        empty); with no "</think>" all of it is reasoning and the reply is empty;
      - tool calls: only the first <TOOLCALL>...</TOOLCALL> block; "[" / "]" added as the plugin does (it prepends a "]"
        when the closing bracket is missing, so that JSON never parses); a call whose name is not a string or whose
        arguments are neither an object nor a string is skipped; the arguments are kept as the JSON string vLLM returns;
        content = the text before the LAST <TOOLCALL>; a block that does not parse -> no calls, all content, ok False.
    An empty reply (no content, no call) is ok False, as in parse_assistant."""
    txt = text
    for stop in ('<SPECIAL_12>', '</s>', '<SPECIAL_11>'):
        txt = txt.split(stop)[0]
    if txt.startswith('<think>'):
        txt = txt[len('<think>'):]
    if '</think>' in txt:
        reasoning, _, content = txt.partition('</think>')
    else:
        reasoning, content = txt, ''
    calls, ok = [], True
    if '<TOOLCALL>' in content:
        try:
            raw = N9_TOOLCALL.findall(content)[0].strip()
            if not raw.startswith('['):
                raw = '[' + raw
            if not raw.endswith(']'):
                raw = ']' + raw
            for tc in json.loads(raw):
                try:
                    name, args = tc['name'], tc['arguments']
                    args = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else args
                    if not isinstance(name, str) or not isinstance(args, str):
                        raise ValueError('not a tool call')
                    calls.append({'type': 'function', 'function': {'name': name, 'arguments': args}})
                except Exception:
                    continue
            content = content[:content.rfind('<TOOLCALL>')]
        except Exception:
            calls, ok = [], False
    msg = {'role': 'assistant', 'content': content.strip()}
    if reasoning.strip('\n'):
        msg['reasoning_content'] = reasoning.strip('\n')
    if calls:
        msg['tool_calls'] = calls
    if not calls and not msg['content']:
        ok = False
    return msg, ok


PARSE_URL = None  # Nemotron 3 Nano: the parse service (parse_service.py, vLLM 0.29's parser); set by rollout.py (parse_url)


def parse_assistant_n3(text, tools=None):
    """Nemotron 3 Nano: completion text after the generation prompt ("...<|im_start|>assistant\n<think>\n") -> (chat
    message, ok), parsed by vLLM 0.29's NemotronV3Parser itself (reasoning before </think>, qwen3_coder XML tool calls,
    parameter values typed by the tool schema), the parser the τ² / BFCL evaluation servers run, through parse_service.py
    (the controller's vLLM 0.25.1 has an older copy). As parse_assistant: a reply with neither content nor a call, a
    <tool_call> left in the content (no call parsed from it) or a parse failure is ok False. Arguments become dicts (its
    chat template renders tool_call.arguments|items, as vLLM's chat path converts them)."""
    import requests
    txt = text.split('<|im_end|>')[0]
    # the service down or failing is an environment error (raised: the episode ends as one, logged as such), not an
    # unparsable turn of the model
    r = requests.post(PARSE_URL + '/parse', json={'text': txt, 'tools': tools or []}, timeout=120)
    if r.status_code != 200:
        raise RuntimeError('parse service: HTTP %d %s' % (r.status_code, r.text[:200]))
    p = r.json()
    try:
        calls = [{'type': 'function', 'function': {'name': c['name'], 'arguments': json.loads(c['arguments'] or '{}')}}
                 for c in p['tool_calls']]
        if not all(isinstance(c['function']['arguments'], dict) for c in calls):
            raise ValueError('arguments not an object')
        ok = True
    except Exception:  # a call whose arguments are not a JSON object
        calls, ok = [], False
    content = (p.get('content') or '').strip()
    if not calls and '<tool_call>' in content:
        ok = False
    msg = {'role': 'assistant', 'content': content}
    if (p.get('reasoning') or '').strip('\n'):
        msg['reasoning_content'] = p['reasoning'].strip('\n')
    if calls:
        msg['tool_calls'] = calls
    if not calls and not msg['content']:
        ok = False
    return msg, ok


PARSERS = {'qwen3': parse_assistant, 'n9': parse_assistant_n9, 'n3': parse_assistant_n3}
TOOLS_PARSERS = {'n3'}  # these take the episode's tools and run in a thread (an HTTP call)


class ReplayEnv:
    """The stored conversation split at its assistant turns: initial context, then after stored assistant turn j the
    non-assistant messages that followed it (replayed unchanged)."""

    def __init__(self, msgs):
        pos = [i for i, m in enumerate(msgs) if m.get('role') == 'assistant']
        self.initial = list(msgs[:pos[0]]) if pos else list(msgs)
        self.after = []
        for j, p in enumerate(pos):
            end = pos[j + 1] if j + 1 < len(pos) else len(msgs)
            self.after.append([m for m in msgs[p + 1:end] if m.get('role') != 'assistant'])
        self.n_turns = len(pos) + 1  # the stored assistant turns, then the final turn

    def next_messages(self, j, student_msg, conv=None):
        """messages after student turn j (conv: the conversation so far, ending with student_msg); None: the episode
        ends (after the final turn, or a tool call to nobody)"""
        if j >= self.n_turns - 1 or student_msg.get('tool_calls'):
            return None
        return self.after[j]


class ToolEnv(ReplayEnv):
    """Replayed user turns; the student's own tool calls are answered by `executor(call) -> str`."""

    def __init__(self, msgs, executor):
        super().__init__(msgs)
        self.executor = executor
        # stored tool outputs are not reused: only user / system messages are replayed
        self.after = [[m for m in seg if m.get('role') != 'tool'] for seg in self.after]

    async def next_messages(self, j, student_msg, conv=None):
        if student_msg.get('tool_calls'):
            out = []
            for c in student_msg['tool_calls']:
                r = self.executor(c)
                out.append({'role': 'tool', 'content': (await r) if inspect.isawaitable(r) else r})
            return out
        return None if j >= self.n_turns - 1 else self.after[j]


STOP, TRANSFER = '###STOP###', '###TRANSFER###'
# the agent hands the customer over: a transfer-like tool (any name) or saying so (in 26 of 120 episodes of an
# end-to-end run the simulated customer declared "###TRANSFER###" without any such sign from the agent)
TRANSFER_SIGN_TOOL = re.compile(r'transfer|escalat|hand_?off|connect_to', re.I)
TRANSFER_SIGN_TEXT = re.compile(r"transferr?(ing|ed)?\s+(you|your)|escalat\w*\s+(you|your|this|the)|connect\w*\s+you\s+(with|to)|"
                                r"hand(ing)?\s+you\s+(over|off)|being transferred", re.I)


# the agent is waiting for the customer: its last visible message asks for a detail (in a 300-episode test the customer
# stopped while the agent asked for an ID the card has in 8-14 episodes, 3 without strict_end)
AGENT_ASKS = re.compile(r"(could|can|would|will) you (please )?(share|provide|confirm|verify|send|give|tell|enter|let me know)|"
                        r"please (share|provide|confirm|verify|send|enter|give|reply)|(share|provide) (me )?(your|the|that|those|it)\b|"
                        r"what (is|are) (your|the)|may i (have|get)|i('ll| will) need (your|the)", re.I)
CLOSING_OFFER = re.compile(r"anything else|something else|any other|further (assistance|help|questions)|is there more", re.I)


# gpt-oss writes non-breaking hyphens and spaces (U+2010/2011, U+00A0/202F): 13 % of simulated customer messages had one
# inside an id or number ("AP\u20111144", "555\u2011123\u20114567"), which real users and τ²'s simulator do not write and the
# student copied into tool arguments. Every simulator output and card is written with ASCII ones.
REJECT_REPEAT = 3   # the same call answered with an error this often stops the episode
REPEAT_ANY = 5      # the same call this often, whatever the result, stops it
_PLAIN = str.maketrans({'\u2010': '-', '\u2011': '-', '\u00a0': ' ', '\u202f': ' '})


def _plain(text):
    return text.translate(_PLAIN) if isinstance(text, str) else text


def agent_waiting(conv):
    """does the agent's last visible message ask the customer for something (not a closing offer)?"""
    last = next((m for m in reversed(conv or []) if m.get('role') == 'assistant' and (m.get('content') or '').strip()), None)
    if last is None:
        return False
    qs = [x for x in re.split(r'(?<=[.!?])\s+', last['content'][-600:].strip()) if '?' in x]
    return bool(qs) and bool(AGENT_ASKS.search(qs[-1])) and not CLOSING_OFFER.search(qs[-1])


def agent_transferred(conv):
    """did the agent hand the customer over since the customer's last message?"""
    last_u = max([i for i, m in enumerate(conv or []) if m.get('role') == 'user'], default=-1)
    for m in (conv or [])[last_u + 1:]:
        if m.get('role') != 'assistant':
            continue
        if any(TRANSFER_SIGN_TOOL.search(str((c.get('function') or c).get('name') or '')) for c in m.get('tool_calls') or []):
            return True
        if TRANSFER_SIGN_TEXT.search(m.get('content') or ''):
            return True
    return False

USER_SIM = """# Instructions
- You are playing the role of the USER in a conversation with an AI assistant (for example a customer contacting a \
customer service agent).
- Between <reference> and </reference> below is a recorded conversation between this same user and a DIFFERENT \
assistant. It defines who you are: your goals, the requests you make and their order, and every fact you know \
(names, IDs, codes, numbers, dates, preferences).
- The assistant you are talking to now may behave differently from the recorded one. React naturally to what it \
actually says: answer its questions, correct it if it gets something wrong, push back if it refuses something you \
need, follow its instructions if they are reasonable. Do not copy recorded messages that no longer fit.
- Pursue the same goals as in the reference, in the same order, one step at a time: first answer what the assistant \
just asked, and do not bring up a later request before the earlier ones are dealt with. Give information only when \
it is asked for or needed. Never contradict the reference; if asked for something it does not cover, give a plausible \
answer consistent with it. Do not repeat a request the assistant has already answered.
- Write one message at a time, as the user. Output only the text of your message.
- When all your goals have been handled (done, or definitively refused) output only '###STOP###'. If the assistant \
has transferred you to a human agent output only '###TRANSFER###'.

<reference>
{reference}
</reference>"""

TOOL_SIM = """# Instructions
- Your task is to produce the result of executing a tool that an AI assistant called during a conversation with a \
user.
- The assistant's instructions (its system prompt, including any policy) are between <system> and </system>; the \
tool definitions are between <tools> and </tools>.
- Between <reference> and </reference> is a recorded conversation of the same user with a different assistant, \
including the tool calls it made and the results the tools returned. Treat those results as the true state of the \
backend (accounts, orders, records, codes): your result must be consistent with them and with any earlier results \
in the current conversation.
- You are the backend, not a reviewer. Judge a call only by the tool definitions and the backend's state: a call to a \
defined tool with valid arguments succeeds and returns what the backend returns for exactly these arguments, even if \
it does not fit what the user asked for or seems unhelpful. Never comment on the user's intent or on whether the call \
was a good choice, and never say that data is unavailable, simulated or a placeholder: return realistic, specific data.
- Return an error only when the backend itself would: wrongly typed or malformed arguments, a record that does not \
exist, or an action the backend's own rules reject; then return the error the backend would return. Check arguments \
against the definitions exactly as declared: a value of the declared type is valid (a numeric string for a string \
parameter, an omitted parameter that is not required), and a value outside a declared enum is not.
- Output only the execution result (usually JSON) and nothing else. Abbreviate very long raw data (base64 or binary \
content, long arrays of numbers) with '...'.

<system>
{system}
</system>

<tools>
{tools}
</tools>

<reference>
{reference}
</reference>"""


# NeMo-Gym's user simulator prompt (resources_servers/conversational_tool_use_simulation/app.py) with one added rule
# (the last bullet: use only the scenario's facts); used with a scenario card that the simulator model first writes from
# the stored conversation (SimEnv(scenario=True))
USER_SIM_CARD = """# Instructions
- You are playing the role of a customer contacting a customer service representative.
- The scenario that describes the customer that you are acting as and the tasks that are to be completed is included \
below between the <scenario> and </scenario> tags.
- Strictly follow the instructions in the scenario description.
- To start a conversation, please make a request to the customer service representative in order to complete the \
tasks described in the scenario.
- Generate one message at a time, maintaining natural conversation flow.
- Each message should be a message from a customer to the representative.
- The goal is to continue the conversation until the tasks are complete.
- If the tasks have been completed during the previous messages in the interaction with the customer service \
representative, then instead of generating a message, output just '###STOP###' to indicate that the conversation \
should end.
- If you are transferred to another agent by the customer service representative in a previous message in the \
conversation, then instead of generating a message, output just '###TRANSFER###' to indicate the transfer.
- Use only the facts given in the scenario. Never make up names, IDs, account or order numbers, codes, dates or other \
details that are not in the scenario; if the representative asks for something you do not have, say that you do not \
have it.

<scenario>
{scenario}
</scenario>

Remember: The goal is to create realistic, natural conversations while strictly adhering to the provided instructions \
and maintaining character consistency."""

# Ending rules: 5.4 % of the customer-service episodes of an end-to-end Nemotron Nano 9B v2 run hit the user-turn cap;
# in a sample of 14 of them, 3 ended on a thank-you / goodbye written instead of '###STOP###' (the student kept answering
# pleasantries) and 4 on requests the scenario never had (an Excel version of a template, a calendar text block, a price
# list), which pull the student into writing long off-task documents. strict_end (default) adds these rules to both user
# prompts and ends the episode on a closing-only customer message (closing_only). A shorter wording ("as soon as your
# last task has been handled, stop") made the customer stop while the agent was still waiting for a detail the card has
# (14 of 300 test episodes, 3 without the rules): answering is therefore spelled out as part of the tasks, and stopping
# while the agent waits is forbidden.
STRICT_END_RULES = """- Make only the requests in your instructions. Do not ask for anything they do not mention (other documents, \
templates, other formats, summaries, extra services), even if the {them} offers it. Answering the {them}'s questions \
and giving the details it asks for (from your scenario) is part of your tasks, not a new request.
- A task is handled only when the {them} has done it, definitively refused it, or said it cannot do it. While the \
{them} is waiting for something from you (a detail, a code, a confirmation), reply to it; never stop then.
- Once your last task has been handled, output only '###STOP###'. Do not write a thank-you or goodbye message instead: \
'###STOP###' is the whole message.
"""
GRANT_RULES = """Self-service steps: some steps in your instructions are actions you complete yourself on your side (\
confirming, approving, verifying something). You can do such a step only after the representative has given you access \
to it: you then see a line "[Access granted: ...]" in the representative's message. Never say you did the step, and never \
report its result, before that line has appeared for it; until then ask the representative how to proceed. Once access \
is granted, do the step and report its result as your instructions say."""


def _granted(call):
    """the self-service action a grant call hands over, in words (give_discoverable_user_tool's discoverable_tool_name)"""
    a = (call.get('function') or call).get('arguments')
    try:
        a = json.loads(a) if isinstance(a, str) else (a or {})
    except ValueError:
        a = {}
    n = str(a.get('discoverable_tool_name') or a.get('tool_name') or a.get('name') or 'the action')
    return n.replace('_', ' ')


USER_SIM_CARD_STRICT = USER_SIM_CARD.replace('\n\n<scenario>', '\n' + STRICT_END_RULES.format(them='representative').rstrip('\n') + '\n\n<scenario>', 1)
USER_SIM_STRICT = USER_SIM.replace('\n\n<reference>', '\n' + STRICT_END_RULES.format(them='assistant').rstrip('\n') + '\n\n<reference>', 1)
assert USER_SIM_CARD_STRICT != USER_SIM_CARD and USER_SIM_STRICT != USER_SIM

# a customer message that only closes the conversation (thanks, goodbye, nothing more needed) and asks for nothing;
# conservative: any question mark or request phrase keeps the conversation going
CLOSING = re.compile(r"\b(thanks?|thank you|appreciate|good ?bye|bye|have a (great|good|nice|wonderful) (day|one|evening|weekend)|"
                     r"that'?s (all|everything)|that is (all|everything)|that will be all|no further|nothing (else|more|further)|"
                     r"all set|no (more|other) questions|no additional)\b", re.I)
REQUEST = re.compile(r"\?|\b(could|can|would|will) you\b|\bplease\b|\b(i'?d|i would) (like|need|want|prefer)\b|\bi need\b|"
                     r"\b(what|how|when|where|why|which|whether)\b|\b(go ahead|proceed|confirm|yes|send|provide|check|update|cancel|"
                     r"change|book|schedule|add|remove|let me know|tell me|share|need|want|still|also|another|instead|but|however|"
                     r"my \w+(?: \w+)? (?:is|are|was))\b|\d|@", re.I)  # giving a detail (a number, an e-mail, "my name is") is not closing
END_MARK = re.compile(r"###\s*(stop|end)\b|^\W*end of (the )?conversation\W*$", re.I | re.M)


def closing_only(text):
    """a customer message that closes the conversation without asking for anything (or a misspelt end marker)"""
    t = (text or '').strip().replace('\u2019', "'").replace('\u2018', "'")
    if END_MARK.search(t):
        return True
    own = re.sub(r"\b(i|we)('ll| will|'m going to| am going to| are going to)\s+\w+", ' ', t, flags=re.I)  # "I'll send it myself"
    return 0 < len(t) <= 600 and bool(CLOSING.search(t)) and not REQUEST.search(own)


SCENARIO_FROM_CONV = """Below, between <conversation> and </conversation>, is a recorded conversation between a USER (a \
customer) and an AI assistant (the representative), preceded by the assistant's system prompt. A user simulator will \
play this same CUSTOMER with a DIFFERENT assistant, seeing only a scenario card. Write that card. The card describes \
the CUSTOMER as the recording shows them: what they want, what they know and what they do. Never describe the \
representative's job (verifying identity, calling tools, explaining policy): the simulator is the customer.

Describe only what the recording shows. Do not add hypothetical contingencies ("if the representative refuses, ask for \
a human") unless the customer actually reacts that way in the recording, and do not add steps after the point where \
the recording stops.

Return a JSON object with these string fields:
- "customer_persona": who the customer is and how they write (tone, length, style), 1-3 sentences.
- "representative_domain": what the assistant is (for example "customer service agent of an online grocery store").
- "reason_for_contact": what the customer wants overall, 1-2 sentences.
- "customer_details": every concrete fact the customer knows and gives at some point (names, IDs, account numbers, \
order numbers, codes, dates, amounts, preferences), one per line, exactly as written in the conversation.
- "unknown_info": only what the customer does not know at the start and has to ask for or have looked up (for \
example their order status, a fee, a policy rule). Never list something the customer already knows or states.
- "task_instructions": numbered steps addressed to the customer in the second person, in the order the customer acts \
in the recording: what you ask for, what you provide when asked, how you react to the obstacles that actually appear \
in the recording (verification requests, refusals, alternatives offered), and how the conversation ends in the \
recording (done and thanked, transferred to a human, or still open where it stops: then you are done once your last \
request is handled). Refer to the assistant as "the representative"; never write "the user" or "the customer".
  Right: "1. Ask for a refund for the crushed cereal box from order FM2025-5550. 2. When the representative asks you to \
verify your identity, give your phone number (555) 123-4567, then the code 1234 when asked. 3. When the refund is \
confirmed, thank the representative and end the conversation."
  Wrong: "Ask the user to verify their identity." / "Call the refund tool." / "If the representative refuses, ask for a \
supervisor." (when the recorded customer never did that)
Do not mention the assistant's specific replies or tool results. Output only the JSON object.

<conversation>
{conversation}
</conversation>"""


def scenario_string(c):
    """NeMo-Gym's CustomerScenario.create_string(include_domain=True)"""
    ind = lambda x: '\n'.join('\t' + l for l in str(x or '').splitlines())
    out = ['Customer persona:', ind(c.get('customer_persona')),
           'Domain of representative: %s' % c.get('representative_domain'),
           'Reason for contacting representative:', ind(c.get('reason_for_contact')),
           'Customer details:', ind(c.get('customer_details'))]
    if c.get('unknown_info'):
        out += ['Unknown info:', ind(c.get('unknown_info'))]
    out += ['Task instructions:', ind(c.get('task_instructions'))]
    return '\n'.join(out)


def _clip(x, n):
    x = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)
    return x if len(x) <= n else x[:n // 2] + ' [...] ' + x[-n // 2:]


def _norm_val(v):
    """argument value for matching: strings without case, spaces and punctuation (phone / ID formatting), numbers as
    floats, lists and dicts recursively"""
    if isinstance(v, str):
        x = re.sub(r'[\s\-_(),#/+]', '', v.lower())  # keeps '.': "1.5" is not 15, a.b@x is not ab@x
        try:
            return float(x)
        except ValueError:
            return x
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    if isinstance(v, list):
        return [_norm_val(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm_val(x) for k, x in sorted(v.items()) if x not in (None, '', [], {})}
    return v


def _norm_key(c):
    name, args = _call_key(c)
    try:
        a = json.loads(args)
    except Exception:
        a = args
    return name, json.dumps(_norm_val(a), sort_keys=True, ensure_ascii=False)


def _call_key(c):
    f = c.get('function', c)
    a = f.get('arguments')
    if isinstance(a, str):
        try:
            a = json.loads(a)
        except Exception:
            pass
    return f.get('name'), json.dumps(a, sort_keys=True, ensure_ascii=False)


def _ntxt(x):
    """text for id lookups: lower case, letters / digits / @ only (formatting differences do not count)"""
    return re.sub(r'[^a-z0-9@]', '', (x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)).lower())


_IDTOK = re.compile(r'[A-Za-z0-9#@._+-]*\d[A-Za-z0-9#@._+-]*')


def id_tokens(text):
    """id-like tokens of a text: letters and digits (>= 5 characters), e-mail addresses, or >= 6 digits"""
    out = set()
    for t in _IDTOK.findall(text or ''):
        t = t.strip('.,;:!?()[]"\'*')
        n = _ntxt(t)
        if (len(n) >= 5 and re.search(r'[a-z@]', n)) or len(re.sub(r'\D', '', t)) >= 6:
            out.add(t)
    return out


# identifier arguments, including plural lists (camera_ids), identifier, sku and phone (without them made-up ids such as
# CAMERA_00x were accepted in a 400-conversation test)
ID_ARG = re.compile(r'(^ids?$|_ids?$|(?-i:[a-z0-9]Ids?$)|_number$|_code$|^code$|token$|^email$|_email$|^account$|'
                    r'_account$|^identifier$|^sku$|_sku$|phone)', re.I)  # camelCase Id only: with re.I a bare "Ids?$"
# matched amount_paid / max_bid / liquid and rejected amounts as unknown ids


def id_args(args):
    """(key, value) of the identifier arguments of a call: a key like *_id / *_number / *_code / token / email whose
    value has a digit or an @ (names, dates under other keys and free text are not checked)"""
    out = []
    def walk(x, key):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, k)
        elif isinstance(x, list):
            for v in x:
                walk(v, key)
        elif isinstance(x, (str, int)) and not isinstance(x, bool) and ID_ARG.search(key or ''):
            v = str(x).strip()
            if len(v) >= 3 and (re.search(r'\d', v) or '@' in v):
                out.append((key, v))
    walk(args, '')
    return out


def _transcript(msgs, tools_too, n_msg=1500):
    """a conversation as text for the simulators: USER / ASSISTANT lines, with tool calls and results if tools_too"""
    out = []
    for m in msgs:
        r = m.get('role')
        if r == 'user':
            out.append('USER: ' + _clip(m.get('content') or '', n_msg))
        elif r == 'assistant':
            if (m.get('content') or '').strip():
                out.append('ASSISTANT: ' + _clip(m['content'], n_msg))
            if tools_too:
                for c in m.get('tool_calls') or []:
                    k = _call_key(c)
                    out.append('ASSISTANT CALLS TOOL: %s(%s)' % (k[0], _clip(k[1], n_msg)))
        elif r == 'tool' and tools_too:
            out.append('TOOL RESULT: ' + _clip(m.get('content') or '', n_msg))
    return '\n\n'.join(out)


class SimEnv:
    """An LLM plays the user and the tools (see the module docstring)."""

    def __init__(self, msgs, tools, sim, *, max_user_turns=None, max_tokens=2048, n_ref=24000, n_ref_tool=160000, scenario=False,
                 card=None, tool_max_tokens=8192, closed_world=True, strict_end=True, schema_check=True, read_extra=None,
                 grant_tool=None):
        """scenario: the simulator first writes a scenario card from the stored conversation (one extra call) and the
        user simulator then sees only the card, with NeMo-Gym's prompt (it cannot copy later stored requests early);
        False: the user simulator sees the stored conversation itself (USER_SIM)"""
        self.sim, self.max_tokens, self.tool_max_tokens = sim, max_tokens, tool_max_tokens
        # read_extra (regex on the tool name): more read-like tools for a set whose read tools READ_TOOL does not know
        # (e.g. inspect_* / semantically_find_* / grep_* / shell), so they are not counted as writes; grant_tool: the
        # agent's tool that hands the customer a self-service action (give_discoverable_user_tool): the customer simulator
        # sees "[Access granted: ...]" after a successful call and may do / report the step only then. Both off by default
        self.read_extra = re.compile(read_extra) if read_extra else None
        self.grant_tool = grant_tool
        # closed world (customer service): an identifier argument whose value appears nowhere in what the backend knows
        # (system prompt, tool definitions, the recorded conversation, the card, this episode's tool results) is a record
        # that does not exist -> "Error: ... not found", as τ²'s backend answers; without it the simulator accepted 14 of
        # 15 made-up ids (user123, PAT123456, a wrong 6-digit code) and the agent never met a tool error
        self.closed_world = closed_world
        self.strict_end = strict_end  # STRICT_END_RULES in the user prompt + closing-only messages end the episode
        # schema check: a call to a tool that is not offered, or one that leaves out a required parameter, gets the
        # backend's error without the simulator (on the function-calling set the simulator had returned a success for
        # 7 of 8 calls missing a required parameter and for 1 of 2 calls to unknown functions)
        self.schema = {}
        if schema_check:
            for t in tools or []:
                f = t.get('function', t) if isinstance(t, dict) else {}
                if f.get('name'):
                    p = f.get('parameters') if isinstance(f.get('parameters'), dict) else {}
                    self.schema[f['name']] = {str(x).strip().rstrip(':').strip() for x in (p.get('required') or [])
                                              if isinstance(x, str)}
        self.known = _ntxt(json.dumps([msgs, tools or []], ensure_ascii=False)) + _ntxt(card or '')
        self.scenario, self.card = scenario or bool(card), _plain(card)  # card: a scenario card written offline
        first = next((i for i, m in enumerate(msgs) if m.get('role') == 'assistant'), len(msgs))
        self.initial = list(msgs[:first])  # system prompt and the first user message(s), as recorded
        self.system = '\n\n'.join(m.get('content') or '' for m in msgs if m.get('role') == 'system')
        self.tools = json.dumps([t.get('function', t) for t in (tools or [])], ensure_ascii=False)
        self.ref_user = _clip(_transcript(msgs, False), n_ref)
        # the tool simulator gets the recorded tool results whole (the backend's true state): clipped at 1,500 chars
        # each / 24,000 in all, 38 % of the tool-calling sets' results and 85 % of their conversations lost data
        # (median 43K chars, max 197K); 160K chars + tools + the current conversation stay within 131K tokens
        self.ref_tool = _clip(_transcript(msgs, True, n_msg=40000), n_ref_tool)
        # recorded results and the episode cache are keyed by (call, the writes made so far): a read after a state change
        # must not return the state from before it (keyed by the call alone, get_order after the student's cancel_order
        # returned the recorded pre-cancel order). The key is the sequence of (normalized) write calls, not their
        # count: a different write of the student's must not pick up the recording's state after its own write. A
        # write is a non-read-like call that did not fail.
        self.recorded = {}  # ((name, args), writes before it) -> recorded result
        self.recorded_norm = {}  # the same under normalized arguments (formatting differences)
        self.cache = {}  # (normalized call, writes) -> result already returned in this episode (same call, same answer)
        self.by_tool = {}   # name -> recorded results (format / size examples for the tool simulator)
        self.writes = ()
        w = ()
        for i, m in enumerate(msgs):
            outs = [x for x in msgs[i + 1:i + 1 + len(m.get('tool_calls') or [])] if x.get('role') == 'tool']
            for c, o in zip(m.get('tool_calls') or [], outs):
                if not str(o.get('content') or '').strip():  # an empty recorded result (0.2 % of the function-calling
                    continue                                 # set's) is not reused: the student retried one 62 times
                self.recorded.setdefault((_call_key(c), w), o.get('content'))
                self.recorded_norm.setdefault((_norm_key(c), w), o.get('content'))
                self.by_tool.setdefault(_call_key(c)[0], []).append(o.get('content') or '')
                if self._write(c, o.get('content')):
                    w += (_norm_key(c),)
        n_user = sum(m.get('role') == 'user' for m in msgs)
        self.max_user_turns = max_user_turns or n_user + 3
        self.n_turns = None  # open-ended: the user simulator decides
        self.counts = dict(user_sim=0, tool_sim=0, tool_recorded=0, user_turns=sum(m.get('role') == 'user' for m in self.initial),
                           cards=0, card_failed=0)
        self.why = None
        # the same rejected call again and again (in a test run one episode repeated a call missing three required
        # parameters 56 times until max_turns, another a call the simulator had rejected 62 times): the episode stops at
        # the REJECT_REPEAT-th identical call that got an error
        self.rejected = collections.Counter()
        self.called = collections.Counter()  # every call: the REPEAT_ANY-th identical one stops the episode (a student
        self.stuck = False                   # retrying an empty result 41-62 times until max_turns / the window)
        self.stuck_why = None
        self.ref_card = _clip(self.system, 8000) + '\n\n' + self.ref_user

    async def _card(self):
        if self.card is None:
            self.card = ''
            txt = await self.sim([{'role': 'user', 'content': SCENARIO_FROM_CONV.format(conversation=self.ref_card)}], 2048)
            txt = re.sub(r'^```(?:json)?\s*|\s*```$', '', txt.strip())
            try:
                c = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])
                c = {k: (v if isinstance(v, str) else '\n'.join(map(str, v)) if isinstance(v, list) else json.dumps(v))
                     for k, v in c.items()}
                self.card = _plain(scenario_string(c))
                self.counts['cards'] += 1
            except Exception:
                self.counts['card_failed'] += 1  # fall back to the stored conversation as the scenario
        return self.card

    async def _user(self, conv):
        card = (await self._card()) if self.scenario else ''
        if card:
            m = [{'role': 'system', 'content': (USER_SIM_CARD_STRICT if self.strict_end else USER_SIM_CARD).format(scenario=card)}]
            them, me = 'Representative: ', 'Customer: '
        else:
            m = [{'role': 'system', 'content': (USER_SIM_STRICT if self.strict_end else USER_SIM).format(reference=self.ref_user)}]
            them = me = ''
        if self.grant_tool:
            m[0]['content'] += '\n\n' + GRANT_RULES
        for i, x in enumerate(conv):  # roles swapped: the simulator writes the user's turns
            if x.get('role') == 'user':
                m.append({'role': 'assistant', 'content': me + (x.get('content') or '')})
            elif x.get('role') == 'assistant':
                txt = (x.get('content') or '').strip()
                if self.grant_tool:
                    calls = x.get('tool_calls') or []
                    outs = [y for y in conv[i + 1:i + 1 + len(calls)] if y.get('role') == 'tool']
                    for c, o in zip(calls, outs):
                        if (c.get('function') or c).get('name') == self.grant_tool and not _is_error(o.get('content')):
                            txt += '\n[Access granted: you can now do this yourself: %s]' % _granted(c)
                if txt.strip():
                    m.append({'role': 'user', 'content': them + txt.strip()})
        if m[-1]['role'] != 'user':  # the student's turn had no visible text
            m.append({'role': 'user', 'content': them + '(no reply)'})
        self.counts['user_sim'] += 1
        out = _plain((await self.sim(m, self.max_tokens)).strip())
        if card:  # a made-up id (3.5 % of customer messages stated one): resample once, then keep
            seen = _ntxt(card) + _ntxt([x.get('content') or '' for x in conv])
            if any(_ntxt(t) not in seen for t in id_tokens(out)):
                self.counts['user_resampled'] = self.counts.get('user_resampled', 0) + 1
                out = _plain((await self.sim(m, self.max_tokens)).strip())
        return out

    async def _tool(self, conv, call):
        """recorded result for an identical call (also after normalizing argument formatting); the result already
        returned for the same call in this episode; else the simulator, told to agree with every earlier result"""
        out = await self._tool_result(conv, call)
        nk = _norm_key(call)
        self.called[(nk, self.writes)] += 1  # the same call on the same state (a read after each write is normal)
        if self.called[(nk, self.writes)] >= REPEAT_ANY:
            self.stuck, self.stuck_why = True, self.stuck_why or 'stopped: repeated call'
        if _is_error(out):  # a rejection (schema check, closed world) or an error result of the simulator, cached for the call
            self.rejected[nk] += 1
            if self.rejected[nk] >= REJECT_REPEAT:
                self.stuck, self.stuck_why = True, self.stuck_why or 'stopped: repeated rejected call'
        if self._write(call, out):
            self.writes += (_norm_key(call),)
        return out

    def _write(self, call, result):
        if self.read_extra and self.read_extra.search(str((call.get('function') or call).get('name') or '')):
            return False
        return _is_write(call, result)

    async def _tool_result(self, conv, call):
        k = (_call_key(call), self.writes)
        nk = (_norm_key(call), self.writes)
        if k in self.recorded or nk in self.recorded_norm:
            self.counts['tool_recorded'] += 1
            out = self.recorded.get(k, self.recorded_norm.get(nk))
            self.cache.setdefault(nk, out)
            return out
        if nk in self.cache:
            self.counts['tool_cached'] = self.counts.get('tool_cached', 0) + 1
            return self.cache[nk]
        k = k[0]
        if self.schema:
            out = None
            if k[0] not in self.schema:
                out = 'Error: unknown function %s' % k[0]
            else:
                try:
                    args = json.loads(k[1]) if k[1] else {}
                except Exception:
                    args = None
                args = {} if args is None else args
                if not isinstance(args, dict):
                    out = 'Error: arguments of %s are not a JSON object' % k[0]
                else:
                    miss = sorted(self.schema[k[0]] - {str(x).strip().rstrip(':').strip() for x in args})
                    if miss:
                        out = 'Error: missing required parameter%s %s' % ('s' if len(miss) > 1 else '', ', '.join(miss))
            if out:
                self.counts['tool_schema_rejected'] = self.counts.get('tool_schema_rejected', 0) + 1
                self.cache[nk] = out
                return out
        if self.closed_world:
            seen = self.known + _ntxt([m.get('content') or '' for m in (conv or []) if m.get('role') == 'tool'])
            bad = [(a, v) for a, v in id_args(json.loads(k[1]) if k[1] else {}) if _ntxt(v) not in seen]
            if bad:
                a, v = bad[0]
                out = 'Error: invalid %s' % a if re.search(r'code|token', a, re.I) else 'Error: %s %s not found' % (a, v)
                self.counts['tool_rejected'] = self.counts.get('tool_rejected', 0) + 1
                self.cache[nk] = out
                return out
        ex = self.by_tool.get(k[0]) or []
        ex_txt = ''
        if ex:  # format / size examples (without the explicit length the long results came out
            n = sum(len(x) for x in ex) // len(ex)  # shorter, mean 0.86 of the recorded size; with it 0.97)
            ex_txt = ('\n\nRecorded results of this same tool in the reference conversation, complete:\n'
                      + '\n---\n'.join(_clip(x, 12000) for x in ex[:2])
                      + '\n\nLength and shape: the recorded results of this tool average about %d characters. Return a '
                        'result of about the same length, with the same fields, nesting and number of items (lists as long '
                        'as in the recorded results), adapted to these arguments. Do not summarize or shorten.' % n)
        m = [{'role': 'system', 'content': TOOL_SIM.format(system=_clip(self.system, 16000), tools=_clip(self.tools, 60000),
                                                         reference=self.ref_tool)},
             {'role': 'user', 'content': 'Current conversation:\n\n%s%s\n\nConsistency: every record (user, account, order, '
                                        'item, reservation, ...) that already appears in the reference results or earlier in '
                                        'the current conversation must come back with exactly the same values; only what has '
                                        'not been shown yet may be new.\n\nTool to execute: %s\nArguments: %s'
                                        % (_clip(_transcript(conv, True, n_msg=20000), 100000), ex_txt, k[0], k[1])}]
        self.counts['tool_sim'] += 1
        # room for a result as long as the recorded ones (~3 chars / token) plus reasoning: 1,416 of the 25K rows have a
        # tool whose recorded results average > 24K chars, beyond the 8,192 default
        n_ex = sum(len(x) for x in ex) // len(ex) if ex else 0
        # a result cut off at the budget is kept (as a long real output would be clipped) instead of failing the episode:
        # 3 of 4 cut-off episodes in an end-to-end run were tools returning base64 images / big tables
        out = _plain((await self.sim(m, min(24576, max(self.tool_max_tokens, n_ex // 3 + 2048)), accept_cut=True)).strip())
        fence = re.fullmatch(r'```[\w-]*\s*(.*?)\s*```', out, re.S)
        if fence:  # 5 % of simulated results came wrapped in a ```json fence; a backend returns bare JSON
            out = fence.group(1)
            self.counts['tool_unfenced'] = self.counts.get('tool_unfenced', 0) + 1
        self.cache[nk] = out
        self.counts['tool_sim_chars'] = self.counts.get('tool_sim_chars', 0) + len(out)
        if ex:
            self.counts['tool_ref_chars'] = self.counts.get('tool_ref_chars', 0) + sum(len(x) for x in ex) // len(ex)
        return out

    async def next_messages(self, j, student_msg, conv=None):
        if student_msg.get('tool_calls'):
            out = [{'role': 'tool', 'content': await self._tool(conv, c)} for c in student_msg['tool_calls']]
            if self.stuck:
                self.why = self.stuck_why
                return None
            return out
        if self.counts['user_turns'] >= self.max_user_turns:
            self.why = 'max_user_turns'
            return None
        u = await self._user(conv)
        if TRANSFER in u and not agent_transferred(conv):  # declared without a transfer: resample once, then end
            self.counts['user_transfer_unprompted'] = self.counts.get('user_transfer_unprompted', 0) + 1
            u = await self._user(conv)
            if TRANSFER in u and not agent_transferred(conv):
                self.why = 'complete'
                return None
        if self.strict_end and STOP in u and agent_waiting(conv):  # stopping while the agent waits: resample once
            self.counts['user_stop_waiting'] = self.counts.get('user_stop_waiting', 0) + 1
            u = await self._user(conv)
        if STOP in u or TRANSFER in u or not u:
            self.why = 'transfer' if TRANSFER in u else 'complete'
            return None
        if self.strict_end and closing_only(u):  # a thank-you / goodbye instead of ###STOP###
            self.counts['user_closing'] = self.counts.get('user_closing', 0) + 1
            self.why = 'complete'
            return None
        self.counts['user_turns'] += 1
        return [{'role': 'user', 'content': re.sub(r'^\s*(USER|User|Customer|CUSTOMER)\s*:\s*', '', u)}]


USER_ADAPT = """You are the USER in a conversation with an AI assistant. Your next message was written in advance (below, \
between <planned> and </planned>), but the assistant's last reply may differ from what was expected. Rewrite the planned \
message only as much as needed so it reads naturally after the assistant's actual reply. You are the USER: never answer \
your own request, never give the information or results you are asking the assistant for, never write as the \
assistant. Do not thank it for something it \
did not do or say, do not refer to results it did not give, and if it asked you a question whose answer is in the planned \
message or in what you already said, answer it. Keep every request, fact, name, ID and number of the planned message \
unchanged and add no new request. If the planned message already fits, return it unchanged. Output only the message.

<conversation so far>
{conv}
</conversation so far>

<planned>
{planned}
</planned>"""


class ReplayUserEnv:
    """Tool-calling conversations: the stored user messages, one each time the student ends its turn with a text reply,
    each first adapted to the student's actual reply (USER_ADAPT: same requests and facts, wording that fits; replayed
    verbatim, 23 % of them did not fit the student's reply vs 7.5 % in the stored conversations); the student's
    tool calls go to SimEnv's tool layer (recorded result, episode cache, simulator)."""

    def __init__(self, msgs, tools, sim, adapt_users=True, **sim_kw):
        sim_kw.pop('scenario', None); sim_kw.pop('card', None)
        self.adapt = adapt_users and sim is not None
        self.sim = sim
        self.tool_env = SimEnv(msgs, tools, sim, closed_world=False, **sim_kw)  # general APIs: no closed world
        first = next((i for i, m in enumerate(msgs) if m.get('role') == 'assistant'), len(msgs))
        self.initial = list(msgs[:first])
        self.users = [m for m in msgs[first:] if m.get('role') == 'user']
        self.k = 0
        self.n_turns = None
        self.counts = self.tool_env.counts
        self.card, self.why = None, None

    async def next_messages(self, j, student_msg, conv=None):
        if student_msg.get('tool_calls'):
            out = [{'role': 'tool', 'content': await self.tool_env._tool(conv, c)} for c in student_msg['tool_calls']]
            if self.tool_env.stuck:
                self.why = self.tool_env.stuck_why
                return None
            return out
        if self.k >= len(self.users):
            self.why = 'complete'
            return None
        u = self.users[self.k]
        self.k += 1
        self.counts['user_turns'] += 1
        planned = u.get('content') or ''
        if self.adapt:
            # the reply budget holds gpt-oss's reasoning and the rewrite (planned messages reach 7.5K chars); a failed
            # or cut-off rewrite falls back to the planned message instead of ending the episode
            try:
                out = (await self.sim([{'role': 'user', 'content': USER_ADAPT.format(
                    conv=_transcript(conv, False, n_msg=3000)[-12000:], planned=planned)}], 1024 + len(planned) // 2)).strip()
                out = _plain(out)
            except Exception:
                out = ''
                self.counts['user_adapt_failed'] = self.counts.get('user_adapt_failed', 0) + 1
            self.counts['user_adapted'] = self.counts.get('user_adapted', 0) + 1
            # keep the rewrite only if it is still the user's message with the same facts: much longer = it answered its
            # own request as the assistant (9 of 94 rewrites), much shorter = it dropped the request (7 of 94), or an id
            # of the planned message lost / a new one made up -> the planned message verbatim
            p = planned.strip()
            ctx = _ntxt(p) + _ntxt([x.get('content') or '' for x in conv])
            if out and out != p and not (len(out) > 1.5 * len(p) + 150 or len(out) < 0.6 * len(p)
                                         or any(_ntxt(t) not in _ntxt(out) for t in id_tokens(p))
                                         or any(_ntxt(t) not in ctx for t in id_tokens(out))):
                self.counts['user_adapt_changed'] = self.counts.get('user_adapt_changed', 0) + 1
                planned = out
            elif out and out != p:
                self.counts['user_adapt_rejected'] = self.counts.get('user_adapt_rejected', 0) + 1
        return [{'role': 'user', 'content': _plain(planned)}]


def spans_of(tok, text, fam='qwen3'):
    """token spans [a, b) of the assistant turns inside a rendered prompt (as prompts.SubsetSource computes them)"""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    st = np.fromiter((o[0] for o in enc['offset_mapping']), np.int64, len(enc['input_ids']))
    sp = [[int(np.searchsorted(st, c0)), int(np.searchsorted(st, c1))] for c0, c1 in hist_char_spans(text, *HIST_MARK[fam])]
    return enc['input_ids'], [s for s in sp if s[1] > s[0]]


def progress_of(msgs, k):
    """the stored conversation's position before its k-th assistant message (0-based; k = the number of assistant
    messages means the stored end): (user messages so far, assistant messages since the last user message)"""
    pos = [i for i, m in enumerate(msgs) if m.get('role') == 'assistant']
    p = pos[k] if k < len(pos) else len(msgs)
    users = [i for i, m in enumerate(msgs[:p]) if m.get('role') == 'user']
    last_u = users[-1] if users else -1
    return len(users), sum(1 for m in msgs[last_u + 1:p] if m.get('role') == 'assistant')


class StepDeadline(Exception):
    """the rollout step's deadline passed while an episode was still running"""


async def run_episode(gen, tok, msgs, tools, *, window, cap, tkw=None, mode='last', env='replay', executor=None,
                      min_gen=1024, max_turns=64, fam='qwen3', sim=None, sim_kw=None, rng=None, cut_at=None,
                      pre_cap=8192, retries=1, turn_cap=8192, deadline=None):
    """gen(ids, max_tokens) -> (response ids, finish, mean logp). Returns (samples, info): samples are dicts with
    ids (prompt), resp, finish, lp, hist_spans; info has the per-turn record and why the episode ended.
    mode cut: the trained turn is the first student turn whose progress (user messages so far, student turns since the
    last user message) reaches that of the stored conversation before its cut_at-th assistant message (cut_at: stored by
    sets with a fixed cut, e.g. the τ² cut sets, else drawn with rng over the stored assistant messages + 1). The student
    may need fewer or more turns than the recorded assistant did, so counting turns would drift; progress keeps the
    decision point.
    Turns before the trained one get at most pre_cap tokens (normal turns: p99 ~3K, > 8K in 0.1 %): a turn that hits it
    (a loop) or calls a tool unparsably is resampled `retries` times, then the episode is dropped (no sample) -- with
    ~3 % loops per turn, keeping such episodes made 14 % of the samples 32K-token loops (in a test run). The trained
    turn itself gets the full cap and is kept even if truncated, like single-turn data.
    mode each (end to end, every turn trained): every turn gets turn_cap tokens; a turn that hits it or calls a tool
    unparsably is resampled `retries` times, then the episode stops there and keeps the turns before it.
    Also: a text reply that closes the target's user turn before the target step (the student needed fewer tool calls)
    is the trained turn -- otherwise the cut slid to the next user turn (20 % of cuts in a test run); a turn that calls
    a transfer-to-human tool is trained and ends the episode (nothing after a transfer is trained, as the τ² cut sets
    never cut after one); a simulator error drops the episode.
    Failures: deadline (time.time() value, the rollout passes step start + deadline_s) stops an episode
    still running then, cancelling a pending generation / simulator call (an API outage cannot stall the step); a
    generation or simulator error ends it too. In mode each the finished turns are kept (valid states, already
    generated); in cut / last the episode gives no sample."""

    async def within(coro):  # await coro, but not past the deadline (a timeout of the call itself stays an error)
        if deadline is None:
            return await coro
        left = deadline - time.time()
        if left <= 0:
            coro.close()
            raise StepDeadline
        try:
            return await asyncio.wait_for(coro, left)
        except TimeoutError:
            if time.time() >= deadline - 0.05:
                raise StepDeadline
            raise

    tkw = {'enable_thinking': True} if tkw is None else tkw
    if env in ('sim', 'replay_tools') and fam not in PARSERS:
        raise NotImplementedError('tool-call parsing for %s: only %s (PARSERS)' % (fam, sorted(PARSERS)))
    parse = PARSERS.get(fam, parse_assistant)
    if env == 'sim':
        E = SimEnv(msgs, tools, sim, **(sim_kw or {}))
    elif env == 'replay_tools':
        E = ReplayUserEnv(msgs, tools, sim, **(sim_kw or {}))
    else:
        E = ToolEnv(msgs, executor) if env == 'tools' else ReplayEnv(msgs)
    target = None
    if mode != 'cut':
        cut_at = None
    else:
        if cut_at is None:
            import random as _random
            n_ref = sum(m.get('role') == 'assistant' for m in msgs) + 1
            cut_at = (rng or _random).randrange(n_ref)
        target = progress_of(msgs, cut_at)
    hist, turns, samples, why = list(E.initial), [], [], 'complete'
    last, resampled = None, 0
    j = 0
    while True:
        text = tok.apply_chat_template(hist, tools=tools, tokenize=False, add_generation_prompt=True, **tkw)
        if mode in ('last', 'cut') and j > 0:
            ids, spans = spans_of(tok, text, fam)
        else:
            ids, spans = tok(text, add_special_tokens=False)['input_ids'], None
        is_target = target is not None and progress_of(hist + [{'role': 'assistant'}], len([m for m in hist if m.get('role') == 'assistant'])) >= target
        pre = target is not None and not is_target
        e2e = mode == 'each' and bool(turn_cap)
        guard = pre or e2e
        mt = min(pre_cap if pre else turn_cap if e2e else cap, window - len(ids))
        if mt < min_gen:
            why = 'window'
            break
        try:
            for attempt in range(retries + 1):
                resp, fin, lp = await within(gen(ids, mt))
                dec = tok.decode(resp, skip_special_tokens=False)
                msg, ok = (await asyncio.to_thread(parse, dec, tools)) if fam in TOOLS_PARSERS else parse(dec)
                if not guard or (fin != 'length' and ok):
                    break
                resampled += 1
        except StepDeadline:
            why, last = 'stopped: step deadline', None
            break
        except Exception as e:  # the student's server kept failing (rollout.generate retries 5 times)
            why, last = 'dropped: generation error: %s' % repr(e)[:120], None
            break
        turns.append(dict(turn=j, prompt=len(ids), response=len(resp), finish=fin, tool_calls=len(msg.get('tool_calls') or []),
                          target=is_target))
        cur = dict(ids=ids, resp=resp, finish=fin, lp=lp, hist_spans=spans if mode in ('last', 'cut') else None, turn=j)
        if pre and (fin == 'length' or not ok):  # still a loop / unparsable after resampling: drop the episode
            why = 'dropped: %s before the cut' % ('loop' if fin == 'length' else 'unparsable or empty reply')
            last = None
            break
        if e2e and (fin == 'length' or not ok):  # end to end: stop here, keep the turns before this one
            why = 'stopped: %s' % ('loop' if fin == 'length' else 'unparsable or empty reply')
            break
        if mode == 'each':
            samples.append(cur)
        last = cur
        if fin == 'length':
            why = 'truncated'
            break
        if not ok:
            why = 'unparsable or empty reply'
            break
        if any(TRANSFER_TOOL.search((c.get('function') or {}).get('name') or '') for c in msg.get('tool_calls') or []):
            why = 'transfer call'  # the decision to transfer is trained; what follows it is not
            break
        if is_target:
            why = 'cut'
            break
        if target is not None and not msg.get('tool_calls'):
            u_now, s_now = progress_of(hist + [{'role': 'assistant'}], len([m for m in hist if m.get('role') == 'assistant']))
            if u_now == target[0] and s_now < target[1]:  # closed the target's user turn early: that reply is the cut
                why = 'cut (turn closed early)'
                break
        hist.append(msg)
        try:
            nxt = E.next_messages(j, msg, hist)
            if inspect.isawaitable(nxt):
                nxt = await within(nxt)
        except StepDeadline:
            why, last = 'stopped: step deadline', None
            break
        except Exception as e:  # a simulator that keeps failing: drop the episode (its last turn is not the cut)
            why = 'dropped: env error: %s' % repr(e)[:120]
            last = None
            break
        if nxt is None:
            if env in ('sim', 'replay_tools'):
                why = E.why or 'complete'
            else:
                why = 'complete' if j >= E.n_turns - 1 else 'tool call without a responder'
            break
        hist.extend(nxt)
        j += 1
        if j >= max_turns:
            why = 'max_turns'
            break
    if mode in ('last', 'cut'):
        samples = [last] if last is not None else []
    info = dict(turns=turns, stored_turns=E.n_turns, ended=why, cut_at=cut_at, target=target, resampled=resampled)
    if env in ('sim', 'replay_tools'):
        info['sim'] = dict(E.counts)
        info['card'] = E.card
        info['conversation'] = hist
    return samples, info
