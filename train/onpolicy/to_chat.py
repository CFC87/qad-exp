"""Stored prompts (the source datasets' original schema) -> chat messages and tool definitions that the Qwen chat
templates accept. Used both for the prompt lengths and for building rollout requests, so both see the same text.

    msgs, tools = to_chat(prompt_json, tools_json)

- Responses-format items: developer -> system; message content lists -> text; function_call -> assistant with
  tool_calls; function_call_output -> tool. Consecutive function_call items become one assistant turn.
- tool_calls stored as JSON strings are parsed; function arguments stored as JSON strings are parsed to dicts
  (newer Qwen templates iterate over them); arguments that are not valid JSON stay strings.
- Tool definitions in Responses form ({type, name, description, parameters}) become {type: function, function: {...}}.
- reasoning_content None is dropped; content None becomes ''.
"""
import json


def _text(c):
    if c is None:
        return ''
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return '\n'.join((p.get('text') or p.get('output_text') or '') if isinstance(p, dict) else str(p) for p in c)
    return str(c)


def _args(a):
    if isinstance(a, str):
        try:
            v = json.loads(a)
            return v if isinstance(v, dict) else a
        except Exception:
            return a
    return a if a is not None else {}


def _call(tc):
    if isinstance(tc, dict):
        f = tc.get('function') if isinstance(tc.get('function'), dict) else tc
        out = {'type': 'function', 'function': {'name': f.get('name'), 'arguments': _args(f.get('arguments'))}}
        if tc.get('id'):
            out['id'] = tc['id']
        return out
    return tc


def _tools(tools):
    if not tools:
        return None
    out = []
    for t in tools:
        if isinstance(t, dict) and 'function' not in t and t.get('name'):
            out.append({'type': 'function', 'function': {k: v for k, v in t.items() if k != 'type'}})
        else:
            out.append(t)
    return out


def to_chat(prompt, tools=None):
    items = json.loads(prompt) if isinstance(prompt, str) else prompt
    tools = json.loads(tools) if isinstance(tools, str) else tools
    msgs = []
    for it in items:
        if not isinstance(it, dict):
            msgs.append({'role': 'user', 'content': str(it)}); continue
        typ, role = it.get('type'), it.get('role')
        if typ == 'function_call':
            call = _call({'function': {'name': it.get('name'), 'arguments': it.get('arguments')}, 'id': it.get('call_id')})
            if msgs and msgs[-1]['role'] == 'assistant' and msgs[-1].get('tool_calls') is not None and not msgs[-1].get('_closed'):
                msgs[-1]['tool_calls'].append(call)
            else:
                msgs.append({'role': 'assistant', 'content': '', 'tool_calls': [call]})
            continue
        if typ == 'function_call_output':
            msgs.append({'role': 'tool', 'content': _text(it.get('output'))}); continue
        if typ == 'reasoning':
            continue  # Responses reasoning items: the templates only render reasoning_content on assistant turns
        role = {'developer': 'system'}.get(role, role) or 'user'
        m = {'role': role, 'content': _text(it.get('content'))}
        tc = it.get('tool_calls')
        if isinstance(tc, str):
            try:
                tc = json.loads(tc)
            except Exception:
                tc = None
        if tc:
            m['tool_calls'] = [_call(t) for t in tc]
        if it.get('reasoning_content'):
            m['reasoning_content'] = it['reasoning_content']
        if role == 'tool' and it.get('name'):
            m['name'] = it['name']
        msgs.append(m)
    for m in msgs:
        m.pop('_closed', None)
    return msgs, _tools(tools)
