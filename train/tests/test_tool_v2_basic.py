"""tool_v2 basic checks: with tool_v2 off the environment behaves as the multiturn.py before tool_v2 (the reference:
a file given as the first argument, else that file at commit BASE of this repository), then the tool_v2 paths.
    python -B train/tests/test_tool_v2_basic.py [OLD_MULTITURN.py]"""
import asyncio, json, os, subprocess, sys, tempfile, importlib.util
from pathlib import Path
ONP = Path(__file__).resolve().parent.parent / 'onpolicy'  # train/onpolicy
BASE = '9e4a805'  # the last commit before tool_v2
def load(path, name):
    sys.path.insert(0, str(ONP))
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def old_path():
    if len(sys.argv) > 1:
        return sys.argv[1]
    src = subprocess.run(['git', '-C', str(ONP), 'show', BASE + ':train/onpolicy/multiturn.py'], capture_output=True, text=True, check=True).stdout
    f = os.path.join(tempfile.mkdtemp(), 'multiturn.py'); open(f, 'w').write(src); return f
new = load(str(ONP / 'multiturn.py'), 'mt_new')
old = load(old_path(), 'mt_old')
tools = [{'type': 'function', 'function': {'name': 'get_price', 'parameters': {'type': 'object', 'properties': {'s': {'type': 'string'}}, 'required': ['s']}}}]
msgs = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'price of BTC and ETH'},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'get_price', 'arguments': '{"s": "BTC"}'}}]},
        {'role': 'tool', 'content': "{'s': 'BTC', 'p': 1}"},
        {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'get_price', 'arguments': '{"s": "ETH"}'}}]},
        {'role': 'tool', 'content': '{"s": "ETH", "p": 2}'}, {'role': 'assistant', 'content': 'BTC 1, ETH 2'}]
call = {'function': {'name': 'get_price', 'arguments': '{"s": "SOL"}'}}
conv = msgs[:2] + [{'role': 'assistant', 'content': '', 'tool_calls': [call]}]
def fake(replies, reviews=()):
    # Generation traces retain the legacy assertions; review/progress calls are
    # routed separately, with their full cost/effort tested in test_tool_v2_port.py.
    seen = []
    verdicts = iter(reviews)
    async def sim(m, mt, accept_cut=False):
        if 'Return JSON: {"ok"' in m[0]['content']:
            return json.dumps({'ok': next(verdicts, True), 'reason': 'Incorrect backend result.'})
        if m[0]['content'].startswith('Decide whether the USER'):
            return '{"advance": true}'
        seen.append((m, mt)); return replies[min(len(seen) - 1, len(replies) - 1)]
    return sim, seen
async def main():
    so, a = fake(['{"s": "SOL", "p": 3}']); sn, b = fake(['{"s": "SOL", "p": 3}'])
    ro = await old.SimEnv(msgs, tools, so, closed_world=False)._tool_result(conv, call)
    rn = await new.SimEnv(msgs, tools, sn, closed_world=False)._tool_result(conv, call)
    assert a == b and ro == rn, 'default path changed'
    # Cover legacy schema/cache, customer/card, effort and replay paths as well.
    for card in (None, 'Task instructions: Ask for the balance.'):
        so, a = fake(['Please check my balance.']); sn, b = fake(['Please check my balance.'])
        eo = old.SimEnv(msgs, tools, so, card=card)
        en = new.SimEnv(msgs, tools, sn, card=card, tool_v2=False)
        assert await eo._user(conv) == await en._user(conv) and a == b
        assert eo.counts == en.counts
    for args in ('{}', '{"s":"BTC"}', '{"s":"SOL"}', '{"api_key":"YOUR_KEY"}'):
        so, a = fake(['{"s":"SOL","p":3}']); sn, b = fake(['{"s":"SOL","p":3}'])
        eo = old.SimEnv(msgs, tools, so, closed_world=False)
        en = new.SimEnv(msgs, tools, sn, closed_world=False, tool_v2=False)
        c0 = {'function': {'name': 'get_price', 'arguments': args}}
        for _ in range(2):
            assert await eo._tool(conv, c0) == await en._tool(conv, c0)
        assert a == b and eo.counts == en.counts
    planned_msgs = msgs + [{'role': 'user', 'content': 'Please check BTC again.'}]
    so, a = fake(['Please check BTC once more.']); sn, b = fake(['Please check BTC once more.'])
    eo = old.ReplayUserEnv(planned_msgs, tools, so)
    en = new.ReplayUserEnv(planned_msgs, tools, sn, tool_v2=False)
    assert await eo.next_messages(0, msgs[-1], msgs) == await en.next_messages(0, msgs[-1], msgs)
    assert a == b and eo.counts == en.counts
    print('default path identical (tool/cache/schema, customer/card, replay prompts, budgets, results, counters)')
    s2, c = fake(['<|start|>assistant<|channel|>final <|constrain|>17.5<|return|>', '{"s": "SOL", "p": 3}'], reviews=(False, True))
    e = new.SimEnv(msgs, tools, s2, closed_world=False, tool_v2=True)
    r = await e._tool_result(conv, call)
    assert len(c) == 2 and r == '{"s": "SOL", "p": 3}' and e.counts['tool_resampled'] == 1, (r, len(c), e.counts)
    s3, d = fake(['<|start|>assistant<|channel|>final <|constrain|>17.5<|return|>'])
    e = new.SimEnv(msgs, tools, s3, closed_world=False, tool_v2=True)
    r = await e._tool_result(conv, call)
    assert r == '17.5' and e.counts['tool_unleaked'] == 1, r
    s4, f = fake(['{"s": "SOL", "p": 3}'])
    e = new.SimEnv(msgs, tools, s4, closed_world=False, tool_v2=True)
    r = await e._tool_result(conv, call)
    u = f[0][0][1]['content']
    assert len(f) == 1 and '<recorded_result 1>' in u and '<recorded_result 2>' in u and '---' not in u and 'Known values' not in u and '<report' not in u, u[-800:]
    s5, h = fake(['{"x": 1}', '{"s": "SOL", "p": 3}'], reviews=(False, True))
    e = new.SimEnv(msgs, tools, s5, closed_world=False, tool_v2=True)
    r = await e._tool_result(conv, call)
    assert len(h) == 2 and r == '{"s": "SOL", "p": 3}' and e.counts['tool_reshaped'] == 1 and 'Validation issue:' in h[1][0][1]['content'], (r, e.counts)
    s6, q = fake(['{"error": "no such symbol"}'])
    e = new.SimEnv(msgs, tools, s6, closed_world=False, tool_v2=True)
    r = await e._tool_result(conv, call)
    assert len(q) == 1, 'an error reply is not reshaped'
    print('v2: off-shape reply resampled with a reminder, errors kept')
    # error / write rules
    env_err = '{"type":"text","text":"Error executing tool analyse_poker_cards: 1 validation error","annotations":null}'
    assert new._is_error_v2(env_err) and not new._is_error(env_err) and new._is_error_v2('[{"type": "text", "text": "\u521b\u5efa\u6587\u4ef6 thresh \u5931\u8d25"}]')
    assert not new._is_error_v2('{"type":"text","text":"Found 3 errors in your code"}')
    W = lambda n: new._is_write_v2({'function': {'name': n, 'arguments': '{}'}}, '{"ok": 1}')
    for n in ('add', 'create_room', 'book_flight', 'cancel_order', 'send_email', 'drawing_fillRectangle', 'updateUser', 'write_file', 'connect'):
        assert W(n), n
    for n in ('airbnb_search', 'sequentialthinking', 'GetChartImgLink', 'get_weather', 'multiply', 'lotuswisdom', 'resolve-library-id', 'analyse_poker_cards'):
        assert not W(n), n
    # repeat stop on a failing non-read call (57-62 repeats never stopped in one run)
    tools2 = [{'type': 'function', 'function': {'name': 'connect', 'parameters': {'type': 'object', 'properties': {'u': {'type': 'string'}}}}}]
    msgs2 = [{'role': 'user', 'content': 'connect me'}]
    c2 = {'function': {'name': 'connect', 'arguments': '{"u": "x"}'}}
    s7, _ = fake([env_err])
    for v2 in (False, True):
        e = new.SimEnv(msgs2, tools2, s7, closed_world=False, tool_v2=v2)
        n = 0
        while not e.stuck and n < 30:
            await e._tool(msgs2 + [{'role': 'assistant', 'tool_calls': [c2]}], c2); n += 1
        print('repeat stop, tool_v2=%s: stopped after %d identical failing calls (%s)' % (v2, n, e.stuck_why))
    # user adaptation: an item from a recorded result the student never showed may be dropped
    msgs3 = [{'role': 'system', 'content': 'sys'}, {'role': 'user', 'content': 'find flights to Mexico City'},
             {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'search_flights', 'arguments': '{"to": "MEX"}'}}]},
             {'role': 'tool', 'content': '{"flights": [{"id": "MX123", "price": 349.98}]}'},
             {'role': 'assistant', 'content': 'MX123 at 349.98'}, {'role': 'user', 'content': 'Book MX123 ($349.98) for 4 people please.'}]
    tools3 = [{'type': 'function', 'function': {'name': 'search_flights', 'parameters': {'type': 'object', 'properties': {'to': {'type': 'string'}}}}}]
    rewrite = 'Could you find a direct flight to Mexico City for 4 people and book it?'
    s8, g = fake([rewrite])
    r = new.ReplayUserEnv(msgs3, tools3, s8, tool_v2=True)
    conv3 = msgs3[:2] + [{'role': 'assistant', 'content': 'I could not find flights.'}]
    out = await r.next_messages(0, {'role': 'assistant', 'content': 'I could not find flights.'}, conv3)
    assert out[0]['content'] == rewrite and 'MX123' in g[0][0][0]['content'].split('<not shown>')[-1] and r.counts.get('user_unseen_rewritten') == 1, (out, r.counts)
    s9, g9 = fake([rewrite])
    r = new.ReplayUserEnv(msgs3, tools3, s9)
    out = await r.next_messages(0, {'role': 'assistant', 'content': 'I could not find flights.'}, conv3)
    assert out[0]['content'] == 'Book MX123 ($349.98) for 4 people please.', out
    # copy guard: a verbatim copy of the BTC result for SOL is resampled with the differing keys named
    s10, h10 = fake(["{'s': 'BTC', 'p': 1}", '{"s": "SOL", "p": 3}'], reviews=(False, True))
    e = new.SimEnv(msgs, tools, s10, closed_world=False, tool_v2=True)
    r = await e._tool_result(conv, call)
    assert len(h10) == 2 and r == '{"s": "SOL", "p": 3}' and e.counts.get('tool_uncopied') == 1 and 'Validation issue:' in h10[1][0][1]['content'], (r, e.counts)
    # per-role effort reaches the simulator only when set
    eff = []
    async def sim_e(m, mt, accept_cut=False, effort=None):
        eff.append(effort)
        return '{"ok": true}' if 'Return JSON:' in m[0]['content'] else '{"s": "SOL", "p": 3}'
    e = new.SimEnv(msgs, tools, sim_e, closed_world=False, tool_v2=True, tool_effort='high', user_effort='medium')
    await e._tool_result(conv, call)
    assert eff == ['high', 'high'], eff
    print('v2: invalid copy resampled after review; tool_effort reaches both calls')
    # an error is reused only for the exactly same call (limit "3" vs 3)
    tools4 = [{'type': 'function', 'function': {'name': 'tags', 'parameters': {'type': 'object', 'properties': {'limit': {'type': 'integer'}}}}}]
    err = "{'error': 'Invalid tool call', 'details': {'invalid': {'limit': 'Expected integer, got string'}}}"
    msgs4 = [{'role': 'user', 'content': 'tags'}, {'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'tags', 'arguments': '{"limit": "3"}'}}]},
             {'role': 'tool', 'content': err}]
    s11, h11 = fake(['{"tags": ["a", "b", "c"]}'])
    for v2, want in ((False, err), (True, '{"tags": ["a", "b", "c"]}')):
        e = new.SimEnv(msgs4, tools4, s11, closed_world=False, tool_v2=v2)
        r = await e._tool_result(msgs4[:1], {'function': {'name': 'tags', 'arguments': '{"limit": 3}'}})
        assert r == want, (v2, r)
        r2 = await e._tool_result(msgs4[:1], {'function': {'name': 'tags', 'arguments': '{"limit": "3"}'}})
        assert (r2.startswith('Error: invalid arguments:') if v2 else r2 == err), (v2, r2)
    # a cached simulated error is not served to the corrected call either
    s12, h12 = fake(['{"tags": ["x"]}'])
    e = new.SimEnv(msgs4[:1], tools4, s12, closed_world=False, tool_v2=True)
    r = await e._tool(msgs4[:1], {'function': {'name': 'tags', 'arguments': '{"limit": "5"}'}})
    r2 = await e._tool(msgs4[:1], {'function': {'name': 'tags', 'arguments': '{"limit": 5}'}})
    assert r.startswith('Error: invalid arguments:') and r2 == '{"tags": ["x"]}', (r, r2)
    print('v2: errors reused only for the exact call (recorded and cached)')
    # a rewrite in the assistant's voice is not used
    s13, g13 = fake(["I'm not able to create or export files, but here is the summary you asked for."])
    r = new.ReplayUserEnv(msgs3, tools3, s13, tool_v2=True)
    out = await r.next_messages(0, {'role': 'assistant', 'content': 'Done.'}, conv3)
    assert out[0]['content'].startswith('Book MX123') and r.counts.get('user_fallback_planned') == 1, (out, r.counts)
    print('user v2: assistant-voice rewrite rejected')
    # review fixes
    assert new._is_error_v2('{"content": [{"type": "text", "text": "Error: boom"}], "isError": false}') and new._is_error_v2("{'success': False, 'data': None}")
    ok = '{"ok": 1}'
    W2 = lambda n: new._is_write_v2({'function': {'name': n}}, ok)
    assert not W2('api.get_order') and W2('authenticate_customer') and W2('api.create_order')
    assert not new._auth_tool('validate_zip_code') and not new._send_code_tool('generate_qr_code') and new._send_code_tool('send_verification_code')
    # batch: the second call of a batch sees the first call's result
    seen_ctx = []
    async def sim_b(m, mt, accept_cut=False):
        if 'Return JSON:' in m[0]['content']:
            return '{"ok": true}'
        seen_ctx.append(m[1]['content']); return '{"id": "R%d"}' % len(seen_ctx)
    e = new.SimEnv(msgs, tools, sim_b, closed_world=False, tool_v2=True)
    c1 = {'function': {'name': 'get_price', 'arguments': '{"s": "SOL"}'}}; c2 = {'function': {'name': 'get_price', 'arguments': '{"s": "ADA"}'}}
    outb = await e._batch(msgs[:2] + [{'role': 'assistant', 'content': '', 'tool_calls': [c1, c2]}], [c1, c2])
    sol = [x for x in seen_ctx if x.rstrip().endswith('{"s": "SOL"}') or 'Arguments: {"s": "SOL"}' in x]
    ada = [x for x in seen_ctx if 'Arguments: {"s": "ADA"}' in x]
    assert ada and all('TOOL RESULT:' in x for x in ada) and sol and not any('TOOL RESULT:' in x for x in sol), len(seen_ctx)
    # code sent only counts when the send succeeded
    convf = [{'role': 'assistant', 'content': '', 'tool_calls': [{'function': {'name': 'send_verification_code', 'arguments': '{}'}}]},
             {'role': 'tool', 'content': 'Error: user not found'}]
    e = new.SimEnv(msgs, tools, sim_b, card='x', tool_v2=True)
    assert not e._code_sent(convf) and e._code_sent(convf[:1] + [{'role': 'tool', 'content': '{"sent": true}'}])
    print('v2: nested MCP / success False errors, method-name writes, auth writes, zip/qr not auth/send, batch context, successful send')
    print('user v2: unseen MX123 listed and the rewrite without it kept; v1 keeps the planned message')
    print('v2: resample on leak / unstructured, unleak, tagged examples, known values; budget', f[0][1])
    print(u[u.index('Recorded results'):][:900])
asyncio.run(main())
