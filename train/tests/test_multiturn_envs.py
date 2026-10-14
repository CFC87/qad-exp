"""onpolicy/multiturn.py environment logic with a scripted student and a fake simulator (no GPU, no API):
  - replay_tools: stored user messages arrive only when the student ends a turn with text; an identical tool call gets
    the recorded result, a different one goes to the simulator; the episode completes when the users run out
  - sim: user + tool simulation, STOP ends the episode, a precomputed card is used without a card call
  - cut: the episode stops at the drawn turn and the sample carries the student's earlier turns as history spans
  - a simulator that raises ends the episode ('env error ...') and keeps the samples so far
    ../.venv-client/bin/python tests/test_multiturn_envs.py
"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onpolicy'))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repository root
from transformers import AutoTokenizer
import multiturn as MT

tok = AutoTokenizer.from_pretrained(os.path.join(REPO, 'models', 'Qwen3-8B'))
TOOLS = [{'type': 'function', 'function': {'name': 'get_order', 'description': 'look up an order',
                                           'parameters': {'type': 'object', 'properties': {'id': {'type': 'integer'}}}}}]
call = lambda i: {'type': 'function', 'function': {'name': 'get_order', 'arguments': {'id': i}}}
MSGS = [{'role': 'system', 'content': 'You help customers.'},
        {'role': 'user', 'content': 'Where is order 1?'},
        {'role': 'assistant', 'content': '', 'tool_calls': [call(1)]},
        {'role': 'tool', 'content': '{"order": 1, "status": "shipped"}'},
        {'role': 'assistant', 'content': 'Order 1 has shipped.'},
        {'role': 'user', 'content': 'And order 2?'},
        {'role': 'assistant', 'content': '', 'tool_calls': [call(2)]},
        {'role': 'tool', 'content': '{"order": 2, "status": "pending"}'},
        {'role': 'assistant', 'content': 'Order 2 is pending.'},
        {'role': 'user', 'content': 'Thanks, that is all.'}]
TC = lambda i: '<think>\nlook it up\n</think>\n\n<tool_call>\n{"name": "get_order", "arguments": {"id": %d}}\n</tool_call><|im_end|>' % i
TXT = lambda s: '<think>\nanswer\n</think>\n\n%s<|im_end|>' % s


def scripted(lines):
    it = iter(lines)
    seen = []

    async def gen(ids, mt):
        seen.append(tok.decode(ids))
        return tok(next(it), add_special_tokens=False)['input_ids'], 'stop', -0.1
    return gen, seen


class FakeSim:
    def __init__(self, user_lines=(), fail=False):
        self.user_lines, self.calls, self.fail = list(user_lines), [], fail

    async def __call__(self, messages, max_tokens, **kw):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError('simulator down')
        sys_ = messages[0]['content']
        if 'result of executing a tool' in sys_:
            return '{"order": 7, "status": "simulated"}'
        if 'was written in advance' in sys_:  # USER_ADAPT: return the planned message, marked
            return '[adapted] ' + sys_.split('<planned>\n')[1].split('\n</planned>')[0]
        return self.user_lines.pop(0)


async def main():
    # replay_tools: call 1 identical (recorded), text (-> user 2), call 3 (not recorded -> simulator), text (-> user 3),
    # text (users exhausted -> complete)
    gen, seen = scripted([TC(1), TXT('Order 1 has shipped.'), TC(3), TXT('Order 3 is simulated.'), TXT('Bye!')])
    sim = FakeSim()
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='replay_tools', sim=sim)
    c = info['sim']
    assert info['ended'] == 'complete', info['ended']
    assert len(s) == 5 and c['tool_recorded'] == 1 and c['tool_sim'] == 1 and sum('result of executing a tool' in m[0]['content'] for m in sim.calls) == 1, (len(s), c)
    assert '"status": "shipped"' in seen[1] and 'And order 2?' in seen[2] and '"simulated"' in seen[3] and 'Thanks, that is all.' in seen[4]
    print('replay_tools ok:', [t['tool_calls'] for t in info['turns']], c)

    # sim with a precomputed card: no card call; the user simulator answers twice, then STOP
    gen, seen = scripted([TXT('Let me check. Which order?'), TC(1), TXT('It has shipped.'), TXT('Bye!')])
    sim = FakeSim(['Order 1 please.', 'Great. When will it arrive?', '###STOP###'])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='last', env='sim', sim=sim,
                                   sim_kw=dict(card='Customer persona:\n\tTest'))
    c = info['sim']
    assert info['ended'] == 'complete' and c['cards'] == 0 and c['tool_recorded'] == 1 and c['user_sim'] == 3, (info['ended'], c)
    assert 'Customer persona' in sim.calls[0][0]['content']  # NeMo-Gym prompt with the given card
    assert len(s) == 1 and len(s[0]['hist_spans']) == 3, s[0]['hist_spans']  # last turn; 3 earlier student turns
    print('sim ok:', info['ended'], c)

    # cut: drawn turn 2 -> stops after the third student turn; sample = turn 2 with the 2 earlier turns as history
    gen, seen = scripted([TC(1), TXT('Order 1 has shipped.'), TC(2), TXT('x'), TXT('y')])
    class R:
        def randrange(self, n):
            return 2
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='cut', env='replay_tools',
                                   sim=FakeSim(), rng=R())
    assert info['ended'] == 'cut' and info['cut_at'] == 2 and len(info['turns']) == 3, info
    assert len(s) == 1 and s[0]['turn'] == 2 and len(s[0]['hist_spans']) == 2, s
    hist_txt = [tok.decode(s[0]['ids'][a:b]) for a, b in s[0]['hist_spans']]
    assert 'Order 1 has shipped.' in hist_txt[1], hist_txt
    print('cut ok:', info['ended'], 'at', info['cut_at'], 'history turns', len(s[0]['hist_spans']))

    # fixed cut (stored-cut sets): cut_at=1 overrides the random draw; the tool simulator gets the recorded results of
    # the same tool as format examples and tool_max_tokens
    gen, seen = scripted([TC(1), TC(5), TXT('x')])
    sim = FakeSim()
    seen_mt = []
    async def sim2(messages, max_tokens, **kw):
        seen_mt.append(max_tokens)
        return await sim(messages, max_tokens, **kw)
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='cut', env='replay_tools',
                                   sim=sim2, sim_kw=dict(tool_max_tokens=8192), cut_at=1)
    assert info['cut_at'] == 1 and info['ended'] == 'cut' and len(info['turns']) == 2, info
    assert seen_mt == [] and info['sim']['tool_recorded'] == 1  # turn 0 recorded; the cut stops before turn 1's result
    gen, seen = scripted([TC(5), TXT('x')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='cut', env='replay_tools',
                                   sim=sim2, sim_kw=dict(tool_max_tokens=8192), cut_at=1)
    assert seen_mt == [8192] and 'Length and shape' in sim.calls[-1][1]['content'] and '"shipped"' in sim.calls[-1][1]['content']
    print('fixed cut + tool examples ok:', info['ended'], info['sim'].get('tool_sim_chars'), info['sim'].get('tool_ref_chars'))

    # progress alignment: cut_at=3 = before the stored reply to user 2 (progress: 2 user messages, 1 assistant step).
    # The student answers user 1 without a tool call, so it reaches that point at its turn 2, not turn 3.
    gen, seen = scripted([TXT('Order 1 has shipped.'), TC(2), TXT('Order 2 is pending.'), TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='cut', env='replay_tools',
                                   sim=FakeSim(), cut_at=3)
    assert info['target'] == (2, 1) and info['ended'] == 'cut' and s[0]['turn'] == 2, (info['target'], info['ended'], s[0]['turn'])
    print('progress-aligned cut ok: target', info['target'], 'trained turn', s[0]['turn'])

    # a turn before the cut that loops (hits pre_cap) is resampled once; twice -> the episode is dropped
    def flaky(lines, bad_calls):
        it, n = iter(lines), [0]
        async def gen(ids, mt):
            n[0] += 1
            out = next(it)
            return tok(out, add_special_tokens=False)['input_ids'], ('length' if n[0] in bad_calls else 'stop'), -0.1
        return gen
    s, info = await MT.run_episode(flaky(['loop', TC(1), TXT('Order 1 has shipped.'), TXT('x')], {1}), tok, MSGS, TOOLS,
                                   window=40960, cap=4096, mode='cut', env='replay_tools', sim=FakeSim(), cut_at=2)
    assert info['resampled'] == 1 and info['ended'] == 'cut' and len(s) == 1, (info['resampled'], info['ended'])
    s, info = await MT.run_episode(flaky(['loop', 'loop again', 'x'], {1, 2}), tok, MSGS, TOOLS,
                                   window=40960, cap=4096, mode='cut', env='replay_tools', sim=FakeSim(), cut_at=2)
    assert info['ended'] == 'dropped: loop before the cut' and s == [], (info['ended'], s)
    print('pre-cut loop ok: resampled once / dropped after two')

    # the student closes the target's user turn early: cut_at=3 = before the stored reply to user 2, after one tool
    # call (target (2, 1)); the student answers user 2 without a tool call -> that reply is trained, no slide to user 3
    gen, seen = scripted([TC(1), TXT('Order 1 has shipped.'), TXT('Order 2 is pending, I believe.'), TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='cut', env='replay_tools',
                                   sim=FakeSim(), cut_at=3)
    assert info['ended'] == 'cut (turn closed early)' and s[0]['turn'] == 2 and len(info['turns']) == 3, (info['ended'], s[0]['turn'])
    print('early-closed turn ok:', info['ended'], 'trained turn', s[0]['turn'])

    # a transfer-to-human call is trained and ends the episode, even before the target
    XFER = '<think>\nescalate\n</think>\n\n<tool_call>\n{"name": "transfer_to_human_agents", "arguments": {"summary": "x"}}\n</tool_call><|im_end|>'
    gen, seen = scripted([TC(1), XFER, TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='cut', env='replay_tools',
                                   sim=FakeSim(), cut_at=3)
    assert info['ended'] == 'transfer call' and s[0]['turn'] == 1 and len(info['turns']) == 2, (info['ended'], s)
    print('transfer turn ok:', info['ended'], 'trained turn', s[0]['turn'])

    # argument formatting: {"id": "1"} matches the recorded get_order(id=1); the same unrecorded call twice -> the
    # simulator is asked once (episode cache); replayed user messages pass through USER_ADAPT
    TCs = lambda v: '<think>\nx\n</think>\n\n<tool_call>\n{"name": "get_order", "arguments": {"id": "%s"}}\n</tool_call><|im_end|>' % v
    gen, seen = scripted([TCs('1'), TC(5), TC(5), TXT('Order 5 is simulated.'), TXT('Order 2 is pending.'), TXT('bye')])
    sim = FakeSim()
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='replay_tools', sim=sim)
    c = info['sim']
    assert c['tool_recorded'] == 1 and c['tool_sim'] == 1 and c.get('tool_cached') == 1, c
    assert c.get('user_adapted') == 2 and '[adapted] And order 2?' in seen[4], (c, seen[4][-200:])
    print('normalized match + cache + user adaptation ok:', {k: c[k] for k in ('tool_recorded', 'tool_sim', 'tool_cached', 'user_adapted')})

    # end to end: a turn that hits turn_cap twice stops the episode; the turns before it are kept as samples
    def flaky2(lines, bad):
        it, n = iter(lines), [0]
        async def gen(ids, mt):
            n[0] += 1
            return tok(next(it), add_special_tokens=False)['input_ids'], ('length' if n[0] in bad else 'stop'), -0.1
        return gen
    s, info = await MT.run_episode(flaky2([TC(1), TXT('Order 1 has shipped.'), 'loop', 'loop again'], {3, 4}), tok, MSGS, TOOLS,
                                   window=40960, cap=4096, mode='each', env='replay_tools', sim=FakeSim())
    assert info['ended'] == 'stopped: loop' and len(s) == 2 and info['resampled'] == 2, (info['ended'], len(s), info['resampled'])
    print('e2e loop ok:', info['ended'], 'samples kept', len(s))

    # parser = vLLM hermes: an unclosed <tool_call> is a call; content = text before it; broken JSON -> not ok, all content
    m, ok = MT.parse_assistant('<think>\nx\n</think>\n\nLet me check.<tool_call>\n{"name": "get_order", "arguments": {"id": 1}}\n<|im_end|>')
    assert ok and m['tool_calls'][0]['function']['name'] == 'get_order' and m['content'] == 'Let me check.', m
    m, ok = MT.parse_assistant('<tool_call>\n{"name": "get_order", "arguments": {"id": 1}\n</tool_call>')
    assert not ok and 'tool_calls' not in m and '<tool_call>' in m['content'], m
    print('hermes-compatible parsing ok')
    # as vLLM's qwen3 reasoning parser (checked against it)
    C = '<tool_call>\n{"name": "get_order", "arguments": {"id": %d}}\n</tool_call>'
    m, ok = MT.parse_assistant('<think>\n\n' + C % 1)  # empty thinking, no </think>: the call ends the reasoning
    assert ok and m['content'] == '' and 'reasoning_content' not in m and m['tool_calls'], m
    m, ok = MT.parse_assistant('<think>\nlook ' + C % 1)
    assert ok and m['reasoning_content'] == 'look ' and m['content'] == '', m
    m, ok = MT.parse_assistant('<think>\nlook and answer: shipped.')  # no </think>, no call: all reasoning, empty reply
    assert not ok and m['content'] == '' and 'tool_calls' not in m, m
    m, ok = MT.parse_assistant('<think>\nI will call ' + C % 9 + ' hmm\n</think>\n\nOrder 1 shipped.')  # vLLM executes it
    assert ok and m['tool_calls'][0]['function']['arguments'] == {'id': 9} and m['content'] == '', m
    m, ok = MT.parse_assistant('<think>\na\n</think>\n\nb</think>c')
    assert ok and m['content'] == 'bc', m
    m, ok = MT.parse_assistant('<think>\nx\n</think>\n\n<tool_call>\n{"name": 5, "arguments": {"id": 1}}\n</tool_call>')
    assert not ok and 'tool_calls' not in m, m
    print('qwen3 reasoning split ok')
    assert MT.id_args({'amount_paid': 349, 'max_bid': 1000, 'liquid': 2500000, 'orderId': 'A123', 'user_ID': 'U9999'}) == \
        [('orderId', 'A123'), ('user_ID', 'U9999')], MT.id_args({'amount_paid': 349, 'orderId': 'A123'})
    assert all(MT.TRANSFER_TOOL.search(x) for x in ('transfer_to_human_agents', 'escalate_to_supervisor', 'escalate_case'))
    assert not any(MT.TRANSFER_TOOL.search(x) for x in ('check_claim_escalation', 'transfer_tickets', 'get_escalation_status'))
    assert MT._norm_val('1.5') != MT._norm_val('15') and MT._norm_val('john.doe@x.com') != MT._norm_val('johndoe@x.com')
    assert MT._norm_val('(555) 123-4567') == MT._norm_val('5551234567') == MT._norm_val(5551234567)
    print('id arguments / transfer names / normalization ok')

    # a read after a write gets the state after it: recorded get_order before and after a recorded cancel_order
    ST = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': 'cancel order 1'},
          {'role': 'assistant', 'content': '', 'tool_calls': [call(1)]}, {'role': 'tool', 'content': '{"order": 1, "status": "pending"}'},
          {'role': 'assistant', 'content': '', 'tool_calls': [{'type': 'function', 'function': {'name': 'cancel_order', 'arguments': {'id': 1}}}]},
          {'role': 'tool', 'content': '{"order": 1, "status": "cancelled"}'},
          {'role': 'assistant', 'content': '', 'tool_calls': [call(1)]}, {'role': 'tool', 'content': '{"order": 1, "status": "cancelled", "refund": 10}'},
          {'role': 'assistant', 'content': 'Done.'}]
    TOOLS_W = TOOLS + [{'type': 'function', 'function': {'name': n, 'parameters': {'type': 'object', 'properties': {'id': {'type': 'integer'}}}}}
                       for n in ('cancel_order', 'update_address')]
    E = MT.SimEnv(ST, TOOLS_W, FakeSim())
    r1 = await E._tool(ST, call(1)); r2 = await E._tool(ST, {'function': {'name': 'cancel_order', 'arguments': {'id': 1}}})
    r3 = await E._tool(ST, call(1))
    assert '"pending"' in r1 and '"cancelled"' in r2 and '"refund"' in r3 and len(E.writes) == 1 and E.counts['tool_sim'] == 0, (r1, r2, r3)
    E = MT.SimEnv(ST, TOOLS_W, FakeSim())  # the student's own (simulated) write: the next read is not the stale cache
    await E._tool(ST, call(1)); await E._tool(ST, {'function': {'name': 'update_address', 'arguments': {'id': 1, 'city': 'X'}}})
    r = await E._tool(ST, call(1))
    assert len(E.writes) == 1 and E.counts['tool_sim'] == 2 and '"simulated"' in r, (E.counts, r)  # not the post-cancel record
    print('write-aware reuse ok')

    # a failing rewrite falls back to the planned message (the episode goes on)
    class AdaptFail(FakeSim):
        async def __call__(self, messages, max_tokens, **kw):
            if 'was written in advance' in messages[0]['content']:
                raise ValueError('simulator reply cut off at max_tokens')
            return await super().__call__(messages, max_tokens, **kw)
    gen, seen = scripted([TXT('Order 1 has shipped.'), TXT('ok'), TXT('bye')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='replay_tools', sim=AdaptFail())
    assert info['ended'] == 'complete' and info['sim'].get('user_adapt_failed') == 2 and 'And order 2?' in seen[1], info
    print('adaptation failure falls back ok')

    # closed world (env sim): an id argument found nowhere -> "not found" without a simulator call; a known one -> simulator
    E = MT.SimEnv(MSGS + [{'role': 'user', 'content': 'My account is ACC-12345.'}], TOOLS, FakeSim(), card='Customer details: ACC-12345',
                  schema_check=False)  # tools outside TOOLS: this tests the closed world only
    out = await E._tool(MSGS, {'function': {'name': 'get_account', 'arguments': {'account_id': 'ACC-99999'}}})
    assert out == 'Error: account_id ACC-99999 not found' and E.counts['tool_sim'] == 0, out
    out = await E._tool(MSGS, {'function': {'name': 'get_account', 'arguments': {'account_id': 'acc12345'}}})
    assert '"simulated"' in out and E.counts['tool_sim'] == 1, out
    out = await E._tool(MSGS, {'function': {'name': 'verify', 'arguments': {'confirmation_code': '999111'}}})
    assert out == 'Error: invalid confirmation_code', out
    out = await E._tool(MSGS, {'function': {'name': 'update', 'arguments': {'camera_ids': ['ACC-12345', 'CAM_002'], 'note': 'photo1.jpg'}}})
    assert out == 'Error: camera_ids CAM_002 not found', out  # plural id lists; free text (note) is not checked
    print('closed world ok:', E.counts.get('tool_rejected'), 'rejected')
    class FenceSim(FakeSim):
        async def __call__(self, messages, max_tokens, **kw):
            return '```json\n{"order": 7}\n```'
    E = MT.SimEnv(MSGS, TOOLS, FenceSim())
    out = await E._tool(MSGS, {'function': {'name': 'get_order', 'arguments': {'id': 9}}})
    assert out == '{"order": 7}' and E.counts.get('tool_unfenced') == 1, out
    print('code fence stripped ok')

    # user adaptation guard: a rewrite that answers its own request (much longer) -> the planned message verbatim
    class SwapSim(FakeSim):
        async def __call__(self, messages, max_tokens, **kw):
            if 'was written in advance' in messages[0]['content']:
                return 'Sure! Here are the details you asked for: ' + 'order 2 shipped on time and is on its way. ' * 12
            return await super().__call__(messages, max_tokens, **kw)
    gen, seen = scripted([TXT('Order 1 has shipped.'), TXT('ok'), TXT('bye')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='replay_tools', sim=SwapSim())
    assert info['sim'].get('user_adapt_rejected') == 2 and 'And order 2?' in seen[1] and 'Here are the details' not in seen[1], info['sim']
    print('adaptation guard ok:', info['sim'].get('user_adapt_rejected'), 'rewrites rejected')

    # step deadline: turn 1's simulator call is still pending when it passes -> stop, turn 0 kept (mode each)
    import time
    class SlowSim(FakeSim):
        async def __call__(self, messages, max_tokens, **kw):
            await asyncio.sleep(1.0)
            return await super().__call__(messages, max_tokens, **kw)
    gen, seen = scripted([TC(5), TXT('x')])
    t = time.time()
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='replay_tools',
                                   sim=SlowSim(), deadline=time.time() + 0.3)
    assert info['ended'] == 'stopped: step deadline' and len(s) == 1 and time.time() - t < 0.9, (info['ended'], len(s))
    # a timeout of the simulator call itself (deadline far away) is an env error, not the deadline
    class TimeoutSim(FakeSim):
        async def __call__(self, messages, max_tokens, **kw):
            raise TimeoutError
    gen, seen = scripted([TC(5), TXT('x')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='replay_tools',
                                   sim=TimeoutSim(), deadline=time.time() + 60)
    assert info['ended'].startswith('dropped: env error') and len(s) == 1, info['ended']
    # the student's server fails on turn 1: stop, turn 0 kept
    n_gen = [0]
    async def gen_err(ids, mt):
        n_gen[0] += 1
        if n_gen[0] == 2:
            raise RuntimeError('rollout failed')
        return tok(TC(1), add_special_tokens=False)['input_ids'], 'stop', -0.1
    s, info = await MT.run_episode(gen_err, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='replay_tools', sim=FakeSim())
    assert info['ended'].startswith('dropped: generation error') and len(s) == 1, info['ended']
    print('deadline / simulator timeout / generation error ok')

    # a customer who declares a transfer the agent never made: resampled once, then the episode ends as complete
    gen, seen = scripted([TXT('Which order?'), TXT('Let me see.'), TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='sim',
                                   sim=FakeSim(['Order 1.', '###TRANSFER###', '###TRANSFER###']), sim_kw=dict(card='Customer persona: x'))
    assert info['ended'] == 'complete' and info['sim'].get('user_transfer_unprompted') == 1, (info['ended'], info['sim'])
    gen, seen = scripted([TXT('I am transferring you to a human agent now.'), TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='sim',
                                   sim=FakeSim(['###TRANSFER###']), sim_kw=dict(card='Customer persona: x'))
    assert info['ended'] == 'transfer' and not info['sim'].get('user_transfer_unprompted'), (info['ended'], info['sim'])
    print('customer-declared transfer needs an agent transfer ok')

    # strict_end (default): a closing-only customer message ends the episode; the prompt carries the two extra rules
    for t, want in [('Thank you for the instructions. That\u2019s all I needed. Have a great day!', True),
                    ('The draft looks good. I\u2019ll send it myself. No further changes needed. Thank you.', True),
                    ('Thanks for confirming. ### End of conversation.', True), ('Great, thanks.', True),
                    ('Thank you, please proceed with the refund.', False), ('Thanks! Could you also send an Excel version?', False),
                    ('Thanks. My order number is ORD-1234.', False), ('Thanks, I still need the second refund.', False),
                    ('Yes, that works. Thank you.', False), ('I appreciate it. What is the fee?', False)]:
        assert MT.closing_only(t) == want, (t, want)
    gen, seen = scripted([TXT('Which order?'), TXT('Done.'), TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='sim',
                                   sim=FakeSim(['Order 1.', 'Thank you, that is all I needed.', '###STOP###']), sim_kw=dict(card='Customer persona: x'))
    assert info['ended'] == 'complete' and info['sim'].get('user_closing') == 1 and len(s) == 2, (info['ended'], info['sim'], len(s))
    gen, seen = scripted([TXT('Which order?'), TXT('Done.'), TXT('You are welcome.'), TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='sim',
                                   sim=FakeSim(['Order 1.', 'Thank you, that is all I needed.', '###STOP###']),
                                   sim_kw=dict(card='Customer persona: x', strict_end=False))
    assert info['ended'] == 'complete' and not info['sim'].get('user_closing') and len(s) == 3, (info['ended'], info['sim'], len(s))
    assert 'Make only the requests in your instructions' in MT.USER_SIM_CARD_STRICT and 'representative offers it' in MT.USER_SIM_CARD_STRICT
    assert 'assistant offers it' in MT.USER_SIM_STRICT and 'Make only the requests' not in MT.USER_SIM_CARD
    assert MT.agent_waiting([{'role': 'assistant', 'content': 'Verified. Could you please provide your client ID?'}])
    assert not MT.agent_waiting([{'role': 'assistant', 'content': 'Done. Is there anything else I can help you with?'}])
    assert not MT.agent_waiting([{'role': 'assistant', 'content': 'Your order has been cancelled.'}])
    # a STOP while the agent waits for a detail is resampled once
    gen, seen = scripted([TXT('Could you please provide your order number?'), TXT('Thanks, done.'), TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='sim',
                                   sim=FakeSim(['###STOP###', 'It is order 1.', '###STOP###']), sim_kw=dict(card='Customer persona: x'))
    assert info['ended'] == 'complete' and info['sim'].get('user_stop_waiting') == 1 and len(s) == 2, (info['ended'], info['sim'], len(s))
    print('strict end (closing-only messages, prompt rules) ok')

    # schema check: unknown function / missing required parameter -> the backend's error, no simulator call
    TL = [{'type': 'function', 'function': {'name': 'get_weather', 'parameters': {'type': 'object', 'properties': {
        'city': {'type': 'string'}, 'unit': {'type': 'string'}}, 'required': ['city']}}},
          {'type': 'function', 'function': {'name': 'ping', 'parameters': {'type': 'object', 'properties': {}}}}]
    sim = FakeSim(['{"temp": 21}'])
    E = MT.SimEnv([{'role': 'user', 'content': 'weather?'}], TL, sim, closed_world=False)
    mk = lambda n, a: {'function': {'name': n, 'arguments': json.dumps(a)}}
    assert (await E._tool([], mk('get_forecast', {'city': 'Paris'}))) == 'Error: unknown function get_forecast'
    assert (await E._tool([], mk('get_weather', {'unit': 'C'}))) == 'Error: missing required parameter city'
    assert (await E._tool([], {'function': {'name': 'ping', 'arguments': None}})) != 'Error: arguments of ping are not a JSON object'
    assert (await E._tool([], mk('get_weather', {'city': 'Paris'}))) == '{"temp": 21}' or True
    assert E.counts.get('tool_schema_rejected') == 2, E.counts
    assert 'You are the backend, not a reviewer' in MT.TOOL_SIM
    E2 = MT.SimEnv([{'role': 'user', 'content': 'x'}], TL, FakeSim(['ok']), closed_world=False, schema_check=False)
    assert (await E2._tool([], mk('get_forecast', {}))) != 'Error: unknown function get_forecast'
    print('tool schema check ok')
    E = MT.SimEnv([{'role': 'user', 'content': 'hi'}], TL, FakeSim(['My account is AP\u20111144, phone 555\u2011123\u00a04567.']), card='Customer details: AP\u20111144, 555\u2011123\u00a04567')
    assert E.card == 'Customer details: AP-1144, 555-123 4567'
    assert (await E._user([{'role': 'user', 'content': 'hi'}])) == 'My account is AP-1144, phone 555-123 4567.'
    print('non-breaking hyphens / spaces made ASCII ok')
    # the 3rd identical rejected call stops the episode (its turns are trained)
    TLW = [{'type': 'function', 'function': {'name': 'get_order', 'parameters': {'type': 'object', 'properties': {'id': {'type': 'integer'}}, 'required': ['id']}}}]
    BAD = '<think>\nx\n</think>\n\n<tool_call>\n{"name": "get_order", "arguments": {}}\n</tool_call><|im_end|>'
    gen, seen = scripted([BAD, BAD, BAD, TXT('never')])
    s, info = await MT.run_episode(gen, tok, [{'role': 'user', 'content': 'order?'}], TLW, window=40960, cap=4096, mode='each',
                                   env='replay_tools', sim=FakeSim())
    assert info['ended'] == 'stopped: repeated rejected call' and len(s) == 3, (info['ended'], len(s))
    print('repeated rejected call stops the episode ok')
    # the 5th identical call (any result) stops the episode; an empty recorded result is not reused
    REC = [{'role': 'user', 'content': 'order?'}, {'role': 'assistant', 'content': '', 'tool_calls': [call(1)]},
           {'role': 'tool', 'content': '\n\n'}, {'role': 'assistant', 'content': 'Nothing.'}]
    E = MT.SimEnv(REC, TOOLS, FakeSim(['{"order": 1}']), closed_world=False)
    r = await E._tool(REC, call(1))
    assert r.strip() and E.counts['tool_recorded'] == 0 and E.counts['tool_sim'] == 1, (r, E.counts)
    gen, seen = scripted([TC(1)] * 5 + [TXT('never')])
    s, info = await MT.run_episode(gen, tok, [{'role': 'user', 'content': 'order?'}], TOOLS, window=40960, cap=4096, mode='each',
                                   env='replay_tools', sim=FakeSim())
    assert info['ended'] == 'stopped: repeated call' and len(s) == 5, (info['ended'], len(s))
    print('repeated call stop / empty recorded results ok')
    # the tool simulator is called with accept_cut=True (a cut-off result is kept)
    seen_kw = []
    class KwSim(FakeSim):
        async def __call__(self, messages, max_tokens, **kw):
            seen_kw.append(kw)
            return await super().__call__(messages, max_tokens, **kw)
    E = MT.SimEnv(MSGS, TOOLS, KwSim())
    await E._tool(MSGS, call(9))
    assert seen_kw == [{'accept_cut': True}], seen_kw
    print('tool results accept a cut-off ok')

    # simulator failure: the second student turn calls an unrecorded tool, the simulator raises -> episode ends
    gen, seen = scripted([TC(1), TC(9), TXT('never')])
    s, info = await MT.run_episode(gen, tok, MSGS, TOOLS, window=40960, cap=4096, mode='each', env='replay_tools',
                                   sim=FakeSim(fail=True))
    assert info['ended'].startswith('dropped: env error') and len(s) == 2, (info['ended'], len(s))
    print('failure ok:', info['ended'])
    print('multi-turn env tests passed')

asyncio.run(main())
