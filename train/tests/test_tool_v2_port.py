"""Deterministic contracts for the hand-ported v2 simulator paths."""
import asyncio
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

ONP = Path(__file__).resolve().parent.parent / 'onpolicy'  # train/onpolicy
sys.path.insert(0, str(ONP))
import multiturn as mt

OK = '{"ok": true}'
INITIAL = [{'role': 'user', 'content': 'Help with my account.'}]


def call(name='get_item', **args):
    return {'function': {'name': name, 'arguments': json.dumps(args)}}


def tool(name='get_item', properties=None, required=(), description=''):
    return {'function': {'name': name, 'description': description, 'parameters': {
        'type': 'object', 'properties': properties or {}, 'required': list(required)}}}


def exchange(c, out):
    return [{'role': 'assistant', 'tool_calls': [c]}, {'role': 'tool', 'content': out}]


class Fake:
    def __init__(self, *samples, reviews=(), decisions=(), audits=('YES',)):
        self.samples = list(samples)
        self.reviews = list(reviews)
        self.decisions = list(decisions)
        self.audits = list(audits)
        self.calls = []

    async def __call__(self, messages, budget, **kwargs):
        self.calls.append((copy.deepcopy(messages), budget, kwargs))
        prompt = messages[0]['content']
        if 'Return JSON: {"ok"' in prompt:
            verdict = self.reviews.pop(0) if self.reviews else True
            return json.dumps({'ok': verdict, 'reason': '' if verdict else 'Contradicts the actual request/state.'})
        if prompt.startswith('Decide whether the USER'):
            if not self.decisions:
                raise AssertionError('unexpected progress call')
            return json.dumps(self.decisions.pop(0))
        if prompt.startswith('Audit whether this'):
            return self.audits.pop(0) if self.audits else 'YES'
        if not self.samples:
            raise AssertionError('unexpected generation call')
        return self.samples.pop(0)


class Backend(unittest.IsolatedAsyncioTestCase):
    def env(self, fake, msgs=None, tools=None, **kw):
        return mt.SimEnv(INITIAL if msgs is None else msgs, tools or [tool()], fake,
                         tool_v2=True, **kw)

    async def test_placeholder_credentials_both_environments(self):
        for replay in (False, True):
            for key in ('key', 'api_key', 'apikey', 'apiKey', 'access_token', 'accessToken',
                        'password', 'clientSecret', 'credential', 'monkey'):
                fake = Fake()
                cls = mt.ReplayUserEnv if replay else mt.SimEnv
                env = cls(INITIAL, [tool()], fake, tool_v2=True)
                backend = env.tool_env if replay else env
                result = await backend._tool_result(INITIAL, call(**{key: 'YOUR_TOKEN_HERE'}))
                self.assertEqual(result, 'Error: invalid ' + key)
                self.assertEqual(backend.counts['tool_placeholder_rejected'], 1)
                self.assertFalse(fake.calls)

    async def test_nonstring_placeholder_values(self):
        for value in (None, [], ['YOUR_TOKEN_HERE']):
            fake = Fake()
            e = self.env(fake, closed_world=False)
            self.assertEqual(await e._tool_result(INITIAL, call(api_key=value)), 'Error: invalid api_key')
            self.assertFalse(fake.calls)

    async def test_exemptions_exact_sources_and_nested_keys(self):
        value = 'PEORIA_RESTAURANT_RESEARCH_KEY'
        for closed in (False, True):
            for role in ('user', 'system', 'card', 'live_user', 'live_system'):
                msgs = INITIAL + ([{'role': role, 'content': 'Use ' + value}] if role in ('user', 'system') else [])
                conv = INITIAL + ([{'role': role[5:], 'content': value}] if role.startswith('live_') else [])
                fake = Fake(OK)
                env = self.env(fake, msgs, closed_world=closed, card=value if role == 'card' else None)
                self.assertEqual(await env._tool_result(conv, call(nested={'accessToken': value})), OK)
                self.assertEqual(env.counts['tool_placeholder_exempted'], 1)
                self.assertEqual(len(fake.calls), 2)
        for role in ('assistant', 'tool', 'schema', 'prefix', 'case'):
            msgs = INITIAL + ([{'role': role, 'content': value}] if role in ('assistant', 'tool') else [])
            if role == 'prefix':
                msgs += [{'role': 'user', 'content': value + '_OTHER'}]
            if role == 'case':
                msgs += [{'role': 'user', 'content': value.lower()}]
            fake = Fake()
            env = self.env(fake, msgs, [tool(description=value if role == 'schema' else '')], closed_world=False)
            self.assertEqual(await env._tool_result(msgs, call(api_key=value)), 'Error: invalid api_key')
            self.assertFalse(fake.calls)

    async def test_tau_placeholder_exemption_and_preflight_before_reuse(self):
        c = call(user_id='USER_ID')
        for source, expected in (('user', OK), ('assistant', 'Error: invalid user_id')):
            msgs = [{'role': source, 'content': 'USER_ID'}] + exchange(c, OK)
            e = self.env(Fake(), msgs)
            self.assertEqual(await e._tool_result(INITIAL, c), expected)
        c = call(api_key='YOUR_API_KEY')
        e = self.env(Fake(), INITIAL + exchange(c, OK), closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, c), 'Error: invalid api_key')
        e.cache[(e._key(c), ())] = OK
        self.assertEqual(await e._tool_result(INITIAL, c), 'Error: invalid api_key')

    async def test_schema_and_typed_keys(self):
        spec = tool(properties={'limit': {'type': 'integer', 'minimum': 0}, 'mode': {'enum': ['a', 'b']}}, required=['limit'])
        for args in ({'limit': '3'}, {'limit': -1}, {'limit:': 3}, {'limit': 3, 'mode': 'c'}):
            c = call(**args)
            fake = Fake()
            e = self.env(fake, INITIAL + exchange(call(limit=2), OK), [spec], closed_world=False)
            self.assertTrue((await e._tool_result(INITIAL, c)).startswith('Error:'))
            self.assertEqual(e.counts['tool_schema_rejected'], 1)
            self.assertFalse(fake.calls)
            # the recording's backend accepted this very value: the recording wins over the schema (e.g. numeric
            # lat/lon rejected by a 'string' schema although the recorded backend accepts only numbers)
            e = self.env(Fake(), INITIAL + exchange(c, OK), [spec], closed_world=False)
            self.assertEqual(await e._tool_result(INITIAL, c), OK)
            self.assertFalse(e.counts.get('tool_schema_rejected'))
        fake = Fake('one', 'two')
        e = self.env(fake, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call(limit='3')), 'one')
        self.assertEqual(await e._tool_result(INITIAL, call(limit=3)), 'two')
        self.assertEqual(await e._tool_result(INITIAL, call(limit='3')), 'one')
        self.assertEqual(e.counts['tool_cached'], 1)

    async def test_documented_alternatives_not_recorded_requirements(self):
        desc = 'Verify client identity using Booking ID + email or Contract Number + Bride/Groom Last Name.'
        spec = tool('verify_client', description=desc)
        fake = Fake(OK)
        recorded = call('verify_client', booking_id='B12345', client_email='a@example.com')
        e = self.env(fake, INITIAL + exchange(recorded, OK), [spec], closed_world=False)
        self.assertTrue((await e._tool_result(INITIAL, call('verify_client', booking_id='B12345'))).startswith('Error:'))
        self.assertEqual(await e._tool_result(INITIAL, call('verify_client', contract_number='C67890', last_name='Jones')), OK)
        # No inferred required credentials when the declared contract permits omission.
        e = self.env(Fake(OK), INITIAL + exchange(call('authenticate', password='abc'), OK), [tool('authenticate')])
        self.assertEqual(await e._tool_result(INITIAL, call('authenticate')), OK)

    async def test_auth_false_and_scoped_session(self):
        self.assertFalse(mt._auth_ok('{"verified": false}'))
        self.assertFalse(mt._is_write_v2(call('authenticate'), '{"authenticated": false}'))
        auth = call('authenticate', user_id='U12345', password='Blue-Bird!')
        spec = tool(description='Authentication required for the current account.')
        e = self.env(Fake(OK), INITIAL + exchange(auth, OK), [spec], card='Users U12345 and U67890')
        self.assertIn('authentication required', await e._tool_result(INITIAL, call()))
        self.assertTrue(mt._is_error_v2(await e._tool_result(INITIAL + exchange(auth, OK), call(user_id='U67890'))))
        self.assertEqual(await e._tool_result(INITIAL + exchange(auth, OK), call(user_id='U12345')), OK)

    async def test_local_arithmetic_echo_file(self):
        examples = [('evaluate_expression', {'expression': '1+1'}, '2'),
                    ('sequentialthinking', {'thought': 'first', 'thoughtNumber': 1, 'totalThoughts': 1, 'nextThoughtNeeded': False},
                     '{"thought":"first","thoughtNumber":1,"totalThoughts":1,"nextThoughtNeeded":false}'),
                    ('read_memory_bank_file', {'filename': 'a'}, '{"type":"text","text":"old"}')]
        for name, args, result in examples:
            fake = Fake()
            e = self.env(fake, INITIAL + exchange(call(name, **args), result), [tool(name)], closed_world=False)
            if name == 'evaluate_expression':
                out = await e._tool_result(INITIAL, call(name, expression='(3 + 5) * 2 ** 3'))
                self.assertEqual(out, '64')
            elif name == 'sequentialthinking':
                args['thought'] = 'second'
                self.assertEqual(json.loads(await e._tool_result(INITIAL, call(name, **args))), args)
            else:
                conv = INITIAL + exchange(call('write_memory_bank_file', filename='a', content='new'), OK)
                self.assertEqual(json.loads(await e._tool_result(conv, call(name, filename='a')))['text'], 'new')
            self.assertEqual(e.counts['tool_exact'], 1)
            self.assertFalse(fake.calls)

    async def test_limits_defaults_and_review_effort(self):
        spec = tool(properties={'limit': {'type': 'integer', 'default': 2}})
        fake = Fake('{"items":[1,2,3]}')
        e = self.env(fake, tools=[spec], closed_world=False, tool_effort='high')
        self.assertEqual(json.loads(await e._tool_result(INITIAL, call()))['items'], [1, 2])
        self.assertEqual(e.counts['tool_limited'], 1)
        self.assertEqual(e.counts['tool_reviews'], 1)
        self.assertTrue(all(k['effort'] == 'high' for _, _, k in fake.calls))

    async def test_review_retry_and_error_never_regenerated(self):
        fake = Fake('{"x":1}', '{"x":2}', reviews=(False, True))
        e = self.env(fake, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call()), '{"x":2}')
        self.assertEqual(len(fake.calls), 4)
        self.assertEqual(e.counts['tool_resampled'], 1)
        for out in ('Error: no item', '<|start|>assistant<|channel|>final<|message|>Error: no item<|return|>', '{"x":1}'):
            fake = Fake(out, out, reviews=(False, False))
            e = self.env(fake, closed_world=False)
            await e._tool_result(INITIAL, call())  # kept, the episode goes on (fail soft)
            self.assertEqual(e.counts['tool_samples'], 1 if 'Error:' in out else 2)
            self.assertEqual(e.counts['tool_invalid'], 1)
            self.assertEqual(e.counts['tool_kept_after_review'], 1)

    async def test_reused_leaks_and_context_fail_closed(self):
        c = call()
        e = self.env(Fake(), INITIAL + exchange(c, '<|start|>assistant<|channel|>final<|message|>{"x":1}<|return|>'))
        self.assertEqual(await e._tool_result(INITIAL, c), '{"x":1}')
        e = self.env(Fake(), [{'role': 'system', 'content': 'x' * 20000}] + INITIAL, sim_ctx=4096, closed_world=False)
        with self.assertRaisesRegex(RuntimeError, 'context'):
            await e._tool_result(INITIAL, c)
        self.assertEqual(e.counts['sim_context_rejected'], 1)


class Customer(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_markers_one_retry(self):
        for marker in ('###STOP###', '###TRANSFER###'):
            fake = Fake(marker, 'Yes, please check it.')
            e = mt.SimEnv(INITIAL, [], fake, card='Task instructions: 1. Check my account.', tool_v2=True,
                          user_effort='medium')
            conv = INITIAL + [{'role': 'assistant', 'content': 'May I check your account?'}]
            out = await e.next_messages(0, conv[-1], conv)
            self.assertEqual(out[0]['content'], 'Yes, please check it.')
            self.assertEqual(e.counts['user_end_waiting'], 1)
            self.assertEqual(e.counts['user_resampled'], 1)
            self.assertTrue(all(k['effort'] == 'medium' for _, _, k in fake.calls))
        e = mt.SimEnv(INITIAL, [], Fake('###STOP###', '###STOP###'), tool_v2=True)
        out = await e.next_messages(0, conv[-1], conv)  # after one retry the stop is accepted (as before)
        self.assertEqual(e.counts['user_samples'], 2)
        self.assertIsNone(out)

    async def test_stop_goal_audit_and_role_retry(self):
        conv = INITIAL + [{'role': 'assistant', 'content': 'Anything else I can help with?'}]
        self.assertFalse(mt.agent_waiting_v2(conv))
        fake = Fake('###STOP###', 'Please check my balance.', audits=('NO',))
        e = mt.SimEnv(INITIAL, [], fake, tool_v2=True, card='Task instructions: 1. Check balance. 2. Explain fees.')
        out = await e.next_messages(0, conv[-1], conv)
        self.assertEqual(out[0]['content'], 'Please check my balance.')
        self.assertEqual(e.counts['user_goal_checks'], 1)
        e = mt.SimEnv(INITIAL, [], Fake('###STOP###'), tool_v2=True)
        self.assertIsNone(await e.next_messages(0, conv[-1], conv))
        self.assertEqual(e.why, 'complete')
        e = mt.SimEnv(INITIAL, [], Fake('Assistant: Here is your balance.', 'Customer: Please check it.'), tool_v2=True)
        self.assertEqual((await e.next_messages(0, conv[-1], conv))[0]['content'], 'Please check it.')
        self.assertEqual(e.counts['user_role_slip'], 1)

    async def test_candidate_validation_failure_stops(self):
        fake = Fake('Made up facts.', 'More made up facts.', reviews=(False, False))
        e = mt.SimEnv(INITIAL, [], fake, tool_v2=True)
        conv = INITIAL + [{'role': 'assistant', 'content': 'What do you need?'}]
        out = await e.next_messages(0, conv[-1], conv)  # the second candidate is kept (as before)
        self.assertEqual(out[0]['content'], 'More made up facts.')
        self.assertEqual(e.counts['user_reviews'], 2)
        self.assertEqual(e.counts['user_invalid'], 1)


class Replay(unittest.IsolatedAsyncioTestCase):
    def env(self, fake):
        msgs = [{'role': 'user', 'content': 'Send the report to Alice.'}, {'role': 'assistant', 'content': 'Sent.'},
                {'role': 'user', 'content': 'Now archive the report.'}]
        return mt.ReplayUserEnv(msgs, [], fake, tool_v2=True, user_effort='medium')

    async def test_cut_progress_ignores_clarification_bridges(self):
        class CharTokenizer:
            def apply_chat_template(self, hist, **kw):
                return json.dumps(hist)
            def __call__(self, text, **kw):
                return {'input_ids': list(map(ord, text)), 'offset_mapping': [(i, i + 1) for i in range(len(text))]}
            def decode(self, ids, **kw):
                return ''.join(map(chr, ids))
        outputs = iter(('Should I send it?', 'Sent to Alice.', 'Archived.'))
        async def gen(ids, budget):
            return list(map(ord, '</think>' + next(outputs))), 'stop', 0.0
        fake = Fake('Now archive the report.', decisions=(
            {'advance': False, 'reply': 'Yes, send it to Alice.'}, {'advance': True}))
        msgs = self.env(Fake()).initial + [{'role': 'assistant', 'content': 'Sent.'},
                                         {'role': 'user', 'content': 'Now archive the report.'},
                                         {'role': 'assistant', 'content': 'Archived.'}]
        samples, info = await mt.run_episode(gen, CharTokenizer(), msgs, [], window=20000, cap=2048,
            mode='cut', cut_at=1, env='replay_tools', sim=fake, sim_kw={'tool_v2': True})
        self.assertEqual(info['ended'], 'cut')
        self.assertEqual(len(info['turns']), 3)
        self.assertEqual(samples[0]['turn'], 2)
        self.assertEqual(info['sim']['user_pending'], 1)

    async def test_pending_preserved_then_advances(self):
        fake = Fake('Now archive the report.', decisions=(
            {'advance': False, 'reply': 'Yes, send it to Alice.'}, {'advance': True}))
        e = self.env(fake)
        conv = e.initial + [{'role': 'assistant', 'content': 'Should I send it to Alice?'}]
        out = await e.next_messages(0, conv[-1], conv)
        self.assertEqual(e.k, 0)
        self.assertEqual(e.pending, 'Send the report to Alice.')
        conv += out + [{'role': 'assistant', 'content': 'Sent to Alice.'}]
        out = await e.next_messages(1, conv[-1], conv)
        self.assertEqual(e.k, 1)
        self.assertEqual(e.pending, 'Now archive the report.')
        self.assertEqual(e.counts['user_pending'], 1)
        self.assertEqual(len(fake.calls), 6)
        self.assertTrue(all(k['effort'] == 'medium' for _, _, k in fake.calls))

    async def test_final_goal_held_and_repair_limit(self):
        fake = Fake('Now archive the report.', decisions=({'advance': True},) +
                    ({'advance': False, 'reply': 'Please archive the report.'},) * 3)
        e = self.env(fake)
        conv = e.initial + [{'role': 'assistant', 'content': 'Sent.'}]
        conv += await e.next_messages(0, conv[-1], conv)
        for i in range(3):
            conv += [{'role': 'assistant', 'content': 'I will archive it later.'}]
            out = await e.next_messages(i + 1, conv[-1], conv)
            if i < 2:
                self.assertIsNotNone(out)
                conv += out
            else:  # bound reached: go on with the planned messages; none left here
                self.assertIsNone(out)
                self.assertEqual(e.counts['user_pending_limit'], 1)

    async def test_invalid_progress_never_advances(self):
        fake = Fake(decisions=({'advance': False, 'reply': 'Send to Bob12345.'},) * 2)
        e = self.env(fake)
        out = await e.next_messages(0, {'role': 'assistant', 'content': 'Where?'}, e.initial)
        self.assertEqual(e.counts['user_progress_failed'], 1)  # fell back to the planned message (as before)
        self.assertEqual(out[0]['content'], 'Now archive the report.')

    async def test_invalid_adaptation_no_fallback(self):
        fake = Fake('I have sent it.', 'I have archived it.', decisions=({'advance': True},), reviews=(True, False, False))
        e = self.env(fake)
        out = await e.next_messages(0, {'role': 'assistant', 'content': 'Sent.'}, e.initial)
        self.assertEqual(out[0]['content'], 'Now archive the report.')  # planned message (as before)
        self.assertEqual(e.counts['user_invalid'], 1)
        self.assertEqual(e.counts['user_fallback_planned'], 1)


class ReadingFixes(unittest.IsolatedAsyncioTestCase):
    """fixes after a two-reader reading"""

    async def test_recorded_retry_sequence(self):
        c = call(q='Leon Edwards')
        e = mt.SimEnv(INITIAL + exchange(c, '{"error": "timeout"}') + exchange(c, OK), [tool()], Fake(), tool_v2=True,
                      closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, c), '{"error": "timeout"}')
        self.assertEqual(await e._tool_result(INITIAL, c), OK)
        self.assertEqual(await e._tool_result(INITIAL, c), OK)

    async def test_catalog_read_needs_no_auth(self):
        msgs = INITIAL + exchange(call('authenticate_user', email='a@shop.com', code='123456'), '{"verified": true}') + \
            exchange(call('get_order_details', order_id='ORD-123'), OK) + exchange(call('get_product_details', sku='SKU-123'), OK)
        tools = [tool('authenticate_user'), tool('get_order_details'), tool('get_product_details')]
        e = mt.SimEnv(msgs, tools, Fake(OK), tool_v2=True)  # recorded after the auth write: simulated here
        self.assertEqual(await e._tool_result(INITIAL, call('get_product_details', sku='SKU-123')), OK)
        self.assertEqual(await e._tool_result(INITIAL, call('get_order_details', order_id='ORD-123')),
                         'Error: authentication required for this customer')

    async def test_contact_placeholder_and_unknown_email(self):
        for value, expected in (('user_email', 'Error: invalid contact_info'),
                                ('user@example.com', 'Error: invalid contact_info'),
                                ('nobody@nowhere.com', 'Error: contact_info nobody@nowhere.com not found')):
            e = mt.SimEnv(INITIAL, [tool()], Fake(), tool_v2=True)
            self.assertEqual(await e._tool_result(INITIAL, call(contact_info=value)), expected)
        msgs = [{'role': 'user', 'content': 'My e-mail is jo@home.net.'}]
        e = mt.SimEnv(msgs, [tool()], Fake(OK), tool_v2=True)
        self.assertEqual(await e._tool_result(msgs, call(contact_info='jo@home.net')), OK)


    async def test_keywords_and_set_secrets(self):
        e = mt.SimEnv(INITIAL, [tool()], Fake(OK), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call(keywords=['laparoscopic nephrectomy'])), OK)
        e = mt.SimEnv(INITIAL, [tool('registeruser'), tool('modifypassword')], Fake(OK, OK), tool_v2=True, closed_world=False)
        reg = call('registeruser', username='ann', password='ArtHub123!')
        self.assertEqual(await e._tool_result(INITIAL, reg), OK)
        conv = INITIAL + exchange(reg, OK)
        self.assertEqual(await e._tool_result(conv, call('modifypassword', old_password='ArtHub123!',
                                                          new_password='ArtHub@2023!')), OK)
        self.assertEqual(await e._tool_result(conv, call('modifypassword', old_password='Guess123!', new_password='x1')),
                         'Error: invalid old_password')


    async def test_recorded_type_comma_enum_whole_email(self):
        spec = tool('get_news', properties={'pageSize': {'type': 'string'}, 'q': {'type': 'string'}})
        err = '{"error": "Invalid tool call: pageSize is expected to be an integer but got a string"}'
        e = mt.SimEnv(INITIAL + exchange(call('get_news', q='x', pageSize='10'), err), [spec], Fake(OK), tool_v2=True,
                      closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('get_news', q='y', pageSize=10)), OK)
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertTrue((await e._tool_result(INITIAL, call('get_news', q='y', pageSize=10))).startswith('Error:'))
        s2 = tool('hotels', properties={'amenities': {'type': 'string', 'enum': ['POOL', 'WIFI', 'SPA'],
                                                      'description': 'Amenities, multiple values separated by commas'}})
        e = mt.SimEnv(INITIAL, [s2], Fake(OK), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('hotels', amenities='POOL,WIFI')), OK)
        self.assertTrue((await e._tool_result(INITIAL, call('hotels', amenities='POOL,GYM'))).startswith('Error:'))
        s3 = tool('mortgage', properties={'rate': {'type': 'integer', 'description': 'Annual rate in percent, e.g. 4.5'},
                                          'years': {'type': 'integer', 'description': 'Term in years'}})
        e = mt.SimEnv(INITIAL, [s3], Fake(OK), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('mortgage', rate=0.5)), OK)
        self.assertTrue((await e._tool_result(INITIAL, call('mortgage', years=2.5))).startswith('Error:'))
        msgs = [{'role': 'user', 'content': 'my registered email is marketing@[brand-domain].com'}]
        e = mt.SimEnv(msgs, [tool('auth_x')], Fake(OK), tool_v2=True)
        self.assertEqual(await e._tool_result(msgs, call('auth_x', email='marketing@[brand-domain].com')), OK)


class UserCard(unittest.IsolatedAsyncioTestCase):
    """user_kind 'user': a card-driven user of a general AI assistant (function-calling sets)"""

    async def test_user_prompt_roles_and_label(self):
        card = 'Task instructions:\n\t1. Ask for the weather in Austin.\n\t2. Then ask for a 3-day forecast there.'
        fake = Fake('Me: Thanks. Could you also get the 3-day forecast for Austin?')
        e = mt.SimEnv(INITIAL, [], fake, card=card, tool_v2=True, closed_world=False, user_kind='user')
        conv = INITIAL + [{'role': 'assistant', 'content': 'It is 72F and sunny in Austin.'}]
        out = await e.next_messages(0, conv[-1], conv)
        self.assertEqual(out[0]['content'], 'Thanks. Could you also get the 3-day forecast for Austin?')
        m = [c[0] for c in fake.calls if c[0][0]['content'].startswith('# Instructions')][0]
        self.assertTrue(m[0]['content'].startswith('# Instructions\n- You are playing the role of a person (the user)'))
        self.assertIn("Reply to the assistant's latest message", m[0]['content'])
        self.assertNotIn('representative', m[0]['content'].split('<scenario>')[0].lower())
        self.assertEqual((m[1]['role'], m[1]['content']), ('assistant', 'Me: Help with my account.'))
        self.assertTrue(m[2]['content'].startswith('AI assistant: It is 72F'))
        self.assertIn('as the user', m[2]['content'])

    async def test_review_sees_plain_transcript(self):
        # on the role-swapped prompt the reviewer took the user's candidate for the assistant's reply
        card = 'Task instructions:\n\t1. Ask for the weather in Austin.\n\t2. Then ask for a 3-day forecast there.'
        fake = Fake('Could you also get the 3-day forecast for Austin?')
        e = mt.SimEnv(INITIAL, [], fake, card=card, tool_v2=True, closed_world=False, user_kind='user')
        conv = INITIAL + [{'role': 'assistant', 'content': 'It is 72F and sunny in Austin.'}]
        await e.next_messages(0, conv[-1], conv)
        rev = [c[0] for c in fake.calls if 'Return JSON: {"ok"' in c[0][0]['content']]
        self.assertEqual(len(rev), 1)
        self.assertTrue(rev[0][0]['content'].startswith('You review the next message of the simulated USER'))
        ctx = json.loads(rev[0][1]['content'])['context']
        self.assertEqual(len(ctx), 1)
        self.assertIn('Instructions given to the simulated USER:', ctx[0]['content'])
        self.assertIn('USER: Help with my account.\n\nASSISTANT: It is 72F and sunny in Austin.', ctx[0]['content'])
        self.assertNotIn('Me: ', ctx[0]['content'].split('The conversation so far')[1])

    async def test_hints_for_unfinished_stop_and_unknown_detail(self):
        card = 'Task instructions:\n\t1. Ask for the weather in Boston.\n\t2. Then ask for a 3-day forecast there.'
        e = mt.SimEnv(INITIAL, [], Fake(audits=('NO: ask for the 3-day forecast',)), card=card, tool_v2=True,
                      closed_world=False, user_kind='user')
        conv = INITIAL + [{'role': 'assistant', 'content': 'Could you give me the coordinates of Boston?'}]
        self.assertEqual(await e._goal_status_v2(conv), (False, 'ask for the 3-day forecast'))
        audit = [c[0][0]['content'] for c in e.sim.calls if c[0][0]['content'].startswith('Audit whether')][0]
        self.assertIn('cannot carry out with its tools, explained as such, is answered', audit)
        state = {'done': False, 'open': 'ask for the 3-day forecast'}
        issue = await e._customer_issue_v2(conv, '###STOP###', state)
        self.assertIn('still open: ask for the 3-day forecast', issue)
        issue = await e._customer_issue_v2(conv, 'Latitude 42.3601, longitude -71.0589.', state)
        self.assertIn('say that you do not have it', issue)
        self.assertIn('42.3601', issue)
        c = mt.SimEnv(INITIAL, [], Fake(audits=('NO',)), card=card, tool_v2=True)  # customer: unchanged
        self.assertEqual(await c._customer_issue_v2(conv, '###STOP###', {}),
                         'The representative awaits your answer. Answer using the card.')
        self.assertEqual(c.sim.calls, [])  # the customer is told to answer without an audit, as before

    async def test_status_goes_to_user_and_reviewer(self):
        # the audit refused the user's STOP while the reviewer refused anything but STOP
        card = 'Task instructions:\n\t1. Ask for the weather in Austin.\n\t2. Then ask for a 3-day forecast there.'
        fake = Fake('Thanks. Could you also get the 3-day forecast for Austin?', audits=('NO: ask for a 3-day forecast',))
        e = mt.SimEnv(INITIAL, [], fake, card=card, tool_v2=True, closed_world=False, user_kind='user')
        conv = INITIAL + [{'role': 'assistant', 'content': 'It is 72F and sunny in Austin. Anything else?'}]
        out = await e.next_messages(0, conv[-1], conv)
        self.assertEqual(out[0]['content'], 'Thanks. Could you also get the 3-day forecast for Austin?')
        draft = [c[0] for c in fake.calls if c[0][0]['content'].startswith('# Instructions')][0]
        self.assertIn('Still open: ask for a 3-day forecast', draft[-1]['content'])
        rev = [c[0] for c in fake.calls if 'Return JSON: {"ok"' in c[0][0]['content']][0]
        self.assertIn('Still open: ask for a 3-day forecast', rev[1]['content'])
        self.assertEqual(e.counts.get('user_goal_checks'), 1)

    async def test_customer_default_unchanged(self):
        fake = Fake('Please check it.')
        e = mt.SimEnv(INITIAL, [], fake, card='Task instructions: 1. Check my account.', tool_v2=True)
        conv = INITIAL + [{'role': 'assistant', 'content': 'May I check your account?'}]
        await e.next_messages(0, conv[-1], conv)
        m = fake.calls[0][0]
        self.assertIn('customer service representative', m[0]['content'])
        self.assertEqual(m[1]['content'], 'Customer: Help with my account.')
        self.assertEqual(m[2]['content'], 'Representative: May I check your account?')

    def test_card_maker_user_facts_and_leaks(self):
        spec = importlib.util.spec_from_file_location('msc', str(ONP / 'make_scenario_cards.py'))
        sys.argv, argv = ['x'], sys.argv
        src = open(spec.origin).read().replace('asyncio.run(main())', '')
        ns = {'__file__': spec.origin, '__name__': 'msc'}
        exec(compile(src, spec.origin, 'exec'), ns)
        sys.argv = argv
        msgs = [{'role': 'user', 'content': 'My key is KEY-4471. Find NBA player stats for the Knicks guard.'},
                {'role': 'assistant', 'content': 'Jalen Brunson has player ID 1629629.'},
                {'role': 'tool', 'content': '{"player": "Jalen Brunson", "team": "New York Knicks", "id": 1629629}'},
                {'role': 'user', 'content': 'Use player ID 1629629 and compare his home and away stats.'}]
        self.assertEqual(ns['user_first_facts'](msgs), ['KEY-4471'])
        vals = ns['assistant_values'](msgs)
        self.assertIn('1629629', vals)
        self.assertIn('Jalen Brunson', vals)
        self.assertEqual(ns['leaks']('Ask for stats of player 1629629 (Jalen Brunson).', msgs, vals), ['1629629', 'Jalen Brunson'])
        self.assertEqual(ns['leaks']('After the assistant names the guard, compare his home and away stats.', msgs, vals), [])
        # a review: short recorded numbers / prices, invented keys, whole-token provenance
        m2 = [{'role': 'user', 'content': 'Find the cheapest rate.'}, {'role': 'tool', 'content': '{"rate": 49.99, "n": 42}'},
              {'role': 'assistant', 'content': 'The rate is 49.99 (42 offers).'}, {'role': 'user', 'content': 'Book it.'}]
        v2 = ns['assistant_values'](m2)
        self.assertEqual(ns['leaks']('1. Ask for the cheapest rate.\n2. Then book the 49.99 rate.', m2, v2), ['49.99'])
        self.assertEqual(ns['leaks']('1. Ask for the cheapest rate.\n2. Then book it.', m2, v2), [])
        self.assertEqual(ns['invented']('Your API key is KEY-919191.', m2), ['KEY-919191'])
        m3 = [{'role': 'tool', 'content': '{"account": "AB12345"}'}, {'role': 'user', 'content': 'My own account is AB123.'}]
        self.assertEqual(ns['user_first_facts'](m3), ['AB123'])
        # lower-case common phrases and values inside a longer token are no leak; the redaction replaces
        # what the check found, in any case or spelling, so a card with only leaks left becomes ok
        m4 = [{'role': 'user', 'content': 'Find a hostel and list the S3 sub-resources.'},
              {'role': 'tool', 'content': '{"room": "Private room", "stock": "In Stock", "res": "s3_bucket", "id": "HX-4471", '
                                          '"team": "McLaren Mercedes"}'}]
        v4 = ns['assistant_values'](m4)
        self.assertEqual(ns['leaks']('Ask for a private room if it is in stock; list s3_bucket_versioning.', m4, v4), [])
        self.assertEqual(ns['leaks']('Ask for the store job at the Home Depot.', [{'role': 'tool', 'content': '{"s": "The Home Depot"}'}]),
                         ['The Home Depot'])
        card = 'Book hx 4471 (the McLaren Mercedes one) and quote hx-4471 again.'
        self.assertEqual(ns['leaks'](card, m4, v4), ['4471', 'HX-4471', 'McLaren Mercedes'])
        self.assertEqual(ns['redact'](card, m4, v4), 'Book (the one the assistant gives) (the (the one the assistant gives) one) '
                                                     'and quote (the one the assistant gives) again.')
        self.assertEqual(ns['leaks']('Use 2024-10-15_snapshot.json.', [{'role': 'tool', 'content': '{"d": "2024-10"}'}]), [])
        self.assertEqual(ns['leaks']('10. Ask again.\n11. Then stop after 10 tries.', m2, {'#10'}), ['10'])
        self.assertEqual(ns['leaks']('10. Ask again.\n11. Then stop.', m2, {'#10'}), [])
        # "25,000" is one number (the redaction made "25,(the one the assistant gives) BTC"); a year is
        # the user's own knowledge ("first big holiday of 2025" became "of (the one the assistant gives)")
        m5 = [{'role': 'user', 'content': 'Watch BTC open interest.'},
              {'role': 'assistant', 'content': 'Typical spikes reach 25,000 BTC; holidays of 2025 are listed.'}]
        v5 = ns['assistant_values'](m5)
        self.assertIn('#25000', v5)
        self.assertNotIn('#2025', v5)
        self.assertNotIn('#000', v5)
        card = 'Alert me above 25,000 BTC and list the holidays of 2025.'
        self.assertEqual(ns['leaks'](card, m5, v5), ['25000'])
        self.assertEqual(ns['redact'](card, m5, v5), 'Alert me above (the one the assistant gives) BTC and list the holidays of 2025.')
        m6 = [{'role': 'user', 'content': 'Alert me above 25,000 BTC.'}, {'role': 'assistant', 'content': 'Set at 25,000.'}]
        self.assertEqual(ns['leaks']('Alert me above 25,000 BTC.', m6), [])


class ReviewFindings(unittest.IsolatedAsyncioTestCase):
    """review findings"""

    async def test_contact_enum_and_search_email(self):
        spec = tool('update_preferences', properties={'contact_method': {'type': 'string', 'enum': ['email', 'sms', 'none']}})
        e = mt.SimEnv(INITIAL, [spec], Fake(OK), tool_v2=True)
        self.assertEqual(await e._tool_result(INITIAL, call('update_preferences', contact_method='none')), OK)
        e = mt.SimEnv(INITIAL, [tool('search_kb', properties={'query': {'type': 'string'}})], Fake(OK), tool_v2=True)
        self.assertEqual(await e._tool_result(INITIAL, call('search_kb', query='billing@acme.com')), OK)

    async def test_compound_schema_branch_and_integer(self):
        spec = tool('get_item', properties={'value': {'anyOf': [{'type': 'string'}, {'type': 'null'}]}})
        c = call('get_item', value=23.5)
        e = mt.SimEnv(INITIAL + exchange(c, OK), [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, c), OK)
        spec = tool('get_item', properties={'limit': {'type': 'integer'}})
        e = mt.SimEnv(INITIAL + exchange(call('get_item', limit=2), OK), [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertTrue((await e._tool_result(INITIAL, call('get_item', limit=2.5))).startswith('Error:'))
        desc = ('Authenticate user identity using either email + security code or phone number + PIN. Required before '
                'accessing account functions.')
        spec = tool('authenticate_user', description=desc, properties={'method': {'type': 'string'}, 'email': {'type': 'string'},
                    'security_code': {'type': 'string'}, 'phone': {'type': 'string'}, 'pin': {'type': 'string'}})
        rec = call('authenticate_user', method='phone_pin', phone='5551234567', pin='9173')
        e = mt.SimEnv(INITIAL + exchange(rec, '{"authenticated": true}'), [spec], Fake(), tool_v2=True)
        self.assertTrue((await e._tool_result(INITIAL, call('authenticate_user', method='email_code'))).startswith(
            'Error: missing required parameter'))

    async def test_credentials(self):
        for key in ('search_api_key', 'primary_api_key', 'password_value'):
            e = mt.SimEnv(INITIAL, [tool()], Fake(), tool_v2=True, closed_world=False)
            self.assertTrue((await e._tool_result(INITIAL, call(**{key: 'YOUR_API_KEY'}))).startswith('Error: invalid'))
        e = mt.SimEnv(INITIAL, [tool('register_device')], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('register_device', api_key='sk-fabricated987', device='desk')),
                         'Error: invalid api_key')
        e = mt.SimEnv(INITIAL, [tool('get_metrics')], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('get_metrics', api_keys=['sk-fabricated987'])), 'Error: invalid api_keys')
        reg = call('registeruser', username='ann', password='ArtHub123!')
        e = mt.SimEnv(INITIAL, [tool('registeruser'), tool('get_balance')], Fake(OK), tool_v2=True, closed_world=False)
        conv = INITIAL + exchange(reg, OK)
        self.assertEqual(await e._tool_result(conv, call('get_balance', api_key='ArtHub123!')), 'Error: invalid api_key')

    async def test_private_service_needs_auth(self):
        msgs = INITIAL + exchange(call('authenticate_user', email='a@shop.com', code='123456'), '{"verified": true}') + \
            exchange(call('get_service_details', service_id='SVC-123'), OK)
        tools = [tool('authenticate_user'), tool('get_service_details',
                      description="Return the customer's active subscription and billing details")]
        e = mt.SimEnv(msgs, tools, Fake(), tool_v2=True)
        self.assertEqual(await e._tool_result(INITIAL, call('get_service_details', service_id='SVC-123')),
                         'Error: authentication required for this customer')

    def test_cut_labels(self):
        self.assertEqual(mt._cut_labels_v3('Me: Please book it.\nAI assistant: Booked.\nMe: Thanks.'), 'Please book it.')

    async def test_card_user_never_ends_on_failed_stop(self):
        card = 'Task instructions:\n\t1. Check my balance.\n\t2. Then ask for my transaction history.'
        fake = Fake('###STOP###', '###STOP###', '###STOP###', audits=('NO', 'NO', 'NO'))
        e = mt.SimEnv(INITIAL, [], fake, card=card, tool_v2=True, closed_world=False, user_kind='user')
        conv = INITIAL + [{'role': 'assistant', 'content': 'Your balance is $20.'}]
        self.assertIsNone(await e.next_messages(0, conv[-1], conv))
        self.assertEqual(e.why, 'user_invalid')
        e = mt.SimEnv(INITIAL, [], Fake(), tool_v2=True, closed_world=False, user_kind='user')
        with self.assertRaises(RuntimeError):
            await e.next_messages(0, conv[-1], conv)


class ReReviewFindings(unittest.IsolatedAsyncioTestCase):
    """re-review findings"""

    async def test_metadata_new_secrets_and_accounts(self):
        spec = tool('get_secret', properties={'secret_name': {'type': 'string', 'enum': ['production-db', 'service-prod']}})
        e = mt.SimEnv(INITIAL, [spec], Fake(OK), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('get_secret', secret_name='production-db')), OK)
        e = mt.SimEnv(INITIAL, [tool('set_webhook_secret')], Fake(OK), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('set_webhook_secret', new_secret='Signing-Abc987!')), OK)
        e = mt.SimEnv(INITIAL, [tool('get_metrics')], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('get_metrics', tokens=['sk-fabricated987'])), 'Error: invalid tokens')
        dev = tool('register_device', properties={'password': {'type': 'string', 'description': 'Existing administrator password'}})
        e = mt.SimEnv(INITIAL, [dev], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('register_device', password='Fabricated987!')), 'Error: invalid password')
        # two accounts keep their passwords; a changed password stops working; a set password is no bank password
        tools = [tool('registeruser'), tool('modifypassword'), tool('login'), tool('get_balance')]
        e = mt.SimEnv(INITIAL, tools, Fake(OK, OK, OK), tool_v2=True, closed_world=False)
        conv = INITIAL + exchange(call('registeruser', username='ann', password='Alpha456!'), OK) + \
            exchange(call('registeruser', username='bob', password='Beta678!'), OK)
        self.assertEqual(await e._tool_result(conv, call('modifypassword', old_password='Alpha456!', new_password='Gamma9!')), OK)
        conv += exchange(call('modifypassword', old_password='Alpha456!', new_password='Gamma9!'), OK)
        self.assertEqual(await e._tool_result(conv, call('modifypassword', old_password='Beta678!', new_password='Delta1!')), OK)
        self.assertEqual(await e._tool_result(conv, call('modifypassword', old_password='Alpha456!', new_password='X1!')),
                         'Error: invalid old_password')
        self.assertEqual(await e._tool_result(conv, call('get_balance', bank_password='Gamma9!')), 'Error: invalid bank_password')

    async def test_public_catalog_and_owner_ids(self):
        auth = exchange(call('authenticate_user', email='a@shop.com', code='123456'), '{"verified": true}')
        for desc, args, expect_ok in (('Return public products in display order.', {'product_id': 'CAT-123'}, True),
                                      ('Get public catalog service details', {'service_id': 'CAT-123'}, True),
                                      ('', {'customer_event_id': 'EVT-123'}, False)):
            msgs = INITIAL + auth + exchange(call('get_details', **args), OK)
            e = mt.SimEnv(msgs, [tool('authenticate_user'), tool('get_details', description=desc)], Fake(OK), tool_v2=True)
            out = await e._tool_result(INITIAL, call('get_details', **args))
            self.assertEqual(out == OK, expect_ok, (desc, args, out))

    async def test_schema_override_regressions(self):
        spec = tool('get_item', properties={'value': {'type': 'object', 'properties': {'a': {'type': 'string'},
                    'b': {'type': 'string'}}, 'required': ['a', 'b']}, 'limit': {'type': 'integer'}})
        e = mt.SimEnv(INITIAL + exchange(call('get_item', value={'a': 'x'}, limit=1), OK), [spec], Fake(OK), tool_v2=True,
                      closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('get_item', value={'a': 'x'}, limit=2)), OK)
        spec = tool('get_weather', properties={'latitude': {'type': 'string'}})
        e = mt.SimEnv(INITIAL + exchange(call('get_weather', latitude=51.5), OK), [spec], Fake(OK), tool_v2=True,
                      closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('get_weather', latitude=52)), OK)
        spec = tool('get_item', properties={'value': {'anyOf': [{'type': 'integer', 'minimum': 0}, {'type': 'null'}]}})
        e = mt.SimEnv(INITIAL + exchange(call('get_item', value=2), OK), [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertTrue((await e._tool_result(INITIAL, call('get_item', value=-1))).startswith('Error:'))
        spec = tool('get_item', properties={'limit': {'type': 'integer'}})
        e = mt.SimEnv(INITIAL + exchange(call('get_item', limit=1), OK), [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertTrue((await e._tool_result(INITIAL, call('get_item', limit=True))).startswith('Error:'))

    def test_card_numbers_from_values(self):
        spec = importlib.util.spec_from_file_location('msc', str(ONP / 'make_scenario_cards.py'))
        src = open(spec.origin).read().replace('asyncio.run(main())', '')
        ns = {'__file__': spec.origin, '__name__': 'msc'}
        exec(compile(src, spec.origin, 'exec'), ns)
        msgs = [{'role': 'user', 'content': 'Compare current and historical readings.'},
                {'role': 'tool', 'content': '{"10": {"value": 1}, "20": {"value": 2}}'},
                {'role': 'assistant', 'content': 'Current 1, historical 2.'},
                {'role': 'user', 'content': 'Repeat the comparison for the next 10 days.'}]
        self.assertEqual(ns['leaks']('1. Compare readings.\n2. Repeat it for the next 10 days.', msgs), [])


class ThirdReview(unittest.IsolatedAsyncioTestCase):
    """a third review and a reading"""

    async def test_shape_error_withdrawn_and_random(self):
        # a recorded failure with the same keys is no evidence for another call (a review: v1_exercises
        # with a non-empty name got the recorded name="" error); the call goes to the simulator
        bad_c = call('v1_exercises', type='strength', title='')
        ok_c = call('v1_exercises', type='strength', title='curl', muscle='biceps')
        msgs = INITIAL + exchange(bad_c, '{"error": "Expected non-empty string"}') + exchange(ok_c, OK)
        e = mt.SimEnv(msgs, [tool('v1_exercises')], Fake('[]'), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('v1_exercises', type='cardio', title='hike')), '[]')
        self.assertFalse(mt._is_write_v2(call('get_coin_info', symbol='SKL'), OK))
        self.assertTrue(mt._is_write_v2(call('score_solution', solutionId='s1', score=0.9), OK))
        rc = call('random_fruit')
        e = mt.SimEnv(INITIAL + exchange(rc, '{"fruit": "cherry"}'), [tool('random_fruit')], Fake('{"fruit": "kiwi"}'),
                      tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, rc), '{"fruit": "cherry"}')
        self.assertEqual(await e._tool_result(INITIAL, rc), '{"fruit": "kiwi"}')

    async def test_sequentialthinking_official_shape(self):
        rec = {'thoughtNumber': 1, 'totalThoughts': 9, 'nextThoughtNeeded': True, 'branches': [], 'thoughtHistoryLength': 1}
        c0 = call('sequentialthinking', thought='plan', thoughtNumber=1, totalThoughts=9, nextThoughtNeeded=True)
        e = mt.SimEnv(INITIAL + exchange(c0, json.dumps(rec)), [tool('sequentialthinking')], Fake(), tool_v2=True,
                      closed_world=False)
        c1 = call('sequentialthinking', thought='first', thoughtNumber=1, totalThoughts=10, nextThoughtNeeded=True)
        out = json.loads(await e._tool_result(INITIAL + [{'role': 'assistant', 'tool_calls': [c1]}], c1))
        self.assertEqual((out['thoughtNumber'], out['totalThoughts'], out['thoughtHistoryLength']), (1, 10, 1))
        # a review: a rejected thought and a not yet answered one in the same batch are no history
        bad = call('sequentialthinking', thought='x', thoughtNumber=1, totalThoughts=3, nextThoughtNeeded=True, revisesThought=0)
        c2 = call('sequentialthinking', thought='second', thoughtNumber=2, totalThoughts=3, nextThoughtNeeded=True)
        c3 = call('sequentialthinking', thought='third', thoughtNumber=3, totalThoughts=3, nextThoughtNeeded=False)
        conv = INITIAL + exchange(bad, 'Error: revisesThought must be >= 1') + exchange(c1, json.dumps(rec)) + \
            [{'role': 'assistant', 'tool_calls': [c2, c3]}]
        out = json.loads(await e._tool_result(conv, c2))
        self.assertEqual(out['thoughtHistoryLength'], 2)

    async def test_shared_password_two_accounts_and_public(self):
        tools = [tool('registeruser'), tool('modifypassword'), tool('login')]
        e = mt.SimEnv(INITIAL, tools, Fake(OK), tool_v2=True, closed_world=False)
        conv = INITIAL + exchange(call('registeruser', username='ann', password='Shared456!'), OK) + \
            exchange(call('registeruser', username='bob', password='Shared456!'), OK) + \
            exchange(call('modifypassword', username='ann', old_password='Shared456!', new_password='Updated789!'), OK)
        self.assertEqual(await e._tool_result(conv, call('modifypassword', username='bob', old_password='Shared456!',
                                                          new_password='Other1!')), OK)
        auth = exchange(call('authenticate_user', email='a@shop.com', code='123456'), '{"verified": true}')
        msgs = INITIAL + auth + exchange(call('get_details', event_id='EVT-123'), OK)
        e = mt.SimEnv(msgs, [tool('authenticate_user'), tool('get_details', description='Return publicly available events for users')],
                      Fake(OK), tool_v2=True)
        self.assertEqual(await e._tool_result(INITIAL, call('get_details', event_id='EVT-123')), OK)

    async def test_card_user_ends_when_done_despite_offer(self):
        card = 'Task instructions:\n\t1. Ask for the weather in Austin.'
        fake = Fake('###STOP###', audits=('YES',))
        e = mt.SimEnv(INITIAL, [], fake, card=card, tool_v2=True, closed_world=False, user_kind='user')
        conv = INITIAL + [{'role': 'assistant', 'content': 'It is 72F in Austin. Could you tell me if you want a forecast too?'}]
        self.assertIsNone(await e.next_messages(0, conv[-1], conv))
        self.assertEqual(e.why, 'complete')


class UserSwap(unittest.IsolatedAsyncioTestCase):
    def env(self, fake):
        msgs = [{'role': 'user', 'content': 'Send the report to Alice.'}, {'role': 'assistant', 'content': 'Sent.'},
                {'role': 'user', 'content': 'Now archive report 12.'}]
        return mt.ReplayUserEnv(msgs, [], fake, tool_v2=True, user_swap=True, user_effort='medium')

    async def test_next_message_written_over_swapped_roles(self):
        fake = Fake('Thanks. Now archive report 12, please.', decisions=({'advance': True},))
        e = self.env(fake)
        conv = e.initial + [{'role': 'assistant', 'content': 'Sent it to Alice.'}]
        out = await e.next_messages(0, conv[-1], conv)
        self.assertEqual(out[0]['content'], 'Thanks. Now archive report 12, please.')
        self.assertEqual(e.k, 1)
        writer = next(m for m, _, _ in fake.calls if m[0]['content'].startswith('# You are the USER'))
        self.assertEqual([x['role'] for x in writer], ['system', 'assistant', 'user'])
        self.assertEqual(writer[1]['content'], 'Me: Send the report to Alice.')
        self.assertTrue(writer[2]['content'].startswith('AI assistant: Sent it to Alice.'))
        self.assertIn('Now archive report 12.', writer[0]['content'])
        decision = next(m for m, _, _ in fake.calls if m[0]['content'].startswith('Decide whether the USER'))
        self.assertNotIn('"reply"', decision[0]['content'])

    async def test_hold_reply_and_fallbacks(self):
        fake = Fake('Yes, Alice is right.', decisions=({'advance': False},))
        e = self.env(fake)
        conv = e.initial + [{'role': 'assistant', 'content': 'Should I send it to Alice?'}]
        out = await e.next_messages(0, conv[-1], conv)
        self.assertEqual((out[0]['content'], e.k, e.counts['user_pending']), ('Yes, Alice is right.', 0, 1))
        # the writer fails twice: a hold goes on with the next planned message, an adaptation keeps the planned one
        voice = "I can't access that. Let me know if you need anything else."
        fake = Fake(voice, voice, voice, 'Now archive report 12.', decisions=({'advance': False},))
        e = self.env(fake)
        out = await e.next_messages(0, conv[-1], conv)
        self.assertEqual((out[0]['content'], e.k), ('Now archive report 12.', 1))
        self.assertEqual(e.counts['user_hold_failed'], 1)
        fake = Fake(voice, voice, voice, decisions=({'advance': True},))
        e = self.env(fake)
        out = await e.next_messages(0, conv[-1], conv)
        self.assertEqual((out[0]['content'], e.counts['user_fallback_planned']), ('Now archive report 12.', 1))


class DisprovedErrors(unittest.IsolatedAsyncioTestCase):
    """a reading: simulated errors that the call itself disproves"""

    def env(self, fake, msgs=None):
        t = tool('plan', {'stage': {'type': 'string'}, 'analysis': {'type': 'string'}, 'date': {'type': 'string'}},
                 required=['stage'])
        return mt.SimEnv(msgs or INITIAL, [t], fake, tool_v2=True, closed_world=False)

    def test_false_error_cases(self):
        e = self.env(Fake())
        a = json.dumps({'stage': 'ideation', 'analysis': 'x', 'date': '2024/1/1'})
        flagged = ["Error: Missing required property: stage", "Error: missing required parameter 'stage'",
                   "Error: 'stage' is required.", "Error: additional property 'analysis' is not allowed",
                   "Error: 'analysis' is not in the plan schema"]
        for out in flagged:
            self.assertTrue(e._false_error_v2('plan', a, out), out)
        kept = ["Error: Invalid parameter 'date': must be YYYY-MM-DD", "Error: 'stage' is required to be one of a, b",
                "Error: additional property 'mood' is not allowed", "Error: plan not found"]
        for out in kept:
            self.assertEqual(e._false_error_v2('plan', a, out), '', out)
        self.assertEqual(e._false_error_v2('plan', json.dumps({'stage': ''}), 'Error: missing required property stage'), '')
        rec = INITIAL + exchange(call('plan', stage='x', analysis='y'), "Error: additional property 'analysis' is not allowed")
        e2 = self.env(Fake(), rec)
        self.assertEqual(e2._false_error_v2('plan', a, "Error: additional property 'analysis' is not allowed"), '')

    async def test_false_error_regenerated(self):
        fake = Fake('Error: Missing required property: stage', '{"status": "ok", "stage": "ideation"}')
        e = self.env(fake)
        out = await e._tool_result(INITIAL, call('plan', stage='ideation'))
        self.assertEqual(out, '{"status": "ok", "stage": "ideation"}')
        self.assertEqual(e.counts.get('tool_false_error'), 1)


class RejectionAudit(unittest.IsolatedAsyncioTestCase):
    """a rejection audit: program rejections the real backend would not make"""

    async def test_extra_keys_pass_unless_recorded_rejection(self):
        spec = tool('getButtons', {'q': {'type': 'string'}})
        spec['function']['parameters']['additionalProperties'] = False
        e = mt.SimEnv(INITIAL, [spec], Fake(OK), tool_v2=True, closed_world=False)
        self.assertIsNone(e._preflight_v2(INITIAL, call('getButtons', type='button')))
        rec = INITIAL + exchange(call('getButtons', active_only=True), 'Error: unexpected parameter active_only')
        e = mt.SimEnv(rec, [spec], Fake(OK), tool_v2=True, closed_world=False)
        self.assertIn('Additional properties', e._preflight_v2(rec, call('getButtons', type='button')))
        for wording in ('Argument `active_only` present in tool call arguments but not defined in tool schema.',
                        'Parameter not supported by this tool', 'Unrecognized or unsupported fields: [active_only]'):
            rec = INITIAL + exchange(call('getButtons', active_only=True), 'Error: ' + wording)
            e = mt.SimEnv(rec, [spec], Fake(OK), tool_v2=True, closed_world=False)
            self.assertIn('Additional properties', e._preflight_v2(rec, call('getButtons', type='button')))

    async def test_documented_default_credential(self):
        spec = tool('aes_encrypt', {'content': {'type': 'string'},
                                    'key': {'type': 'string', 'description': 'encrypt key, default is your-key-0123456'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(OK), tool_v2=True, closed_world=False)
        self.assertIsNone(e._preflight_v2(INITIAL, call('aes_encrypt', content='x', key='your-key-0123456')))
        self.assertEqual(e._preflight_v2(INITIAL, call('aes_encrypt', content='x', key='your-key-999999')),
                         'Error: invalid key')
        spec = tool('get_balance', {'api_key': {'type': 'string', 'description': 'API credential. Default is Secret.AbC/987'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(OK), tool_v2=True, closed_world=False)
        self.assertIsNone(e._preflight_v2(INITIAL, call('get_balance', api_key='Secret.AbC/987')))
        for wrong in ('Secret', 'secret.AbC/987'):
            self.assertEqual(e._preflight_v2(INITIAL, call('get_balance', api_key=wrong)), 'Error: invalid api_key')

    def test_type_error_against_schema(self):
        spec = tool('search_recipes_by_ingredients', {'ingredients': {'type': 'string'}, 'number': {'type': 'string'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        err = '{"error": "Invalid tool call: parameter \'number\' is expected to be an integer but got a string"}'
        a = json.dumps({'ingredients': 'lime', 'number': '5'})
        self.assertIn('declares number as string', e._false_error_v2('search_recipes_by_ingredients', a, err))
        self.assertEqual(e._false_error_v2('search_recipes_by_ingredients', json.dumps({'number': 5}), err), '')
        rec = INITIAL + exchange(call('search_recipes_by_ingredients', number='3'), err)
        e = mt.SimEnv(rec, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(e._false_error_v2('search_recipes_by_ingredients', a, err), '')

    def test_type_error_against_recorded_backend(self):
        # pageSize declared string, the recorded backend demands an integer; an integer then gets "expected string"
        spec = tool('get_news_everything', {'q': {'type': 'string'}, 'pageSize': {'type': 'string'}})
        err = '{"error": "Invalid tool call: pageSize is expected to be an integer but got a string"}'
        rec = INITIAL + exchange(call('get_news_everything', q='a', pageSize='10'), err)
        e = mt.SimEnv(rec, [spec], Fake(), tool_v2=True, closed_world=False)
        wrong = '{"error": "Invalid tool call: pageSize is expected to be a string but got an integer"}'
        self.assertIn('recorded backend asks for pageSize as integer',
                      e._false_error_v2('get_news_everything', json.dumps({'q': 'b', 'pageSize': 10}), wrong))
        self.assertEqual(e._false_error_v2('get_news_everything', json.dumps({'q': 'b', 'pageSize': '10'}), err), '')
        # the same loop with other wording ("expects a string")
        self.assertIn('recorded backend asks for pageSize as integer', e._false_error_v2(
            'get_news_everything', json.dumps({'q': 'b', 'pageSize': 10}),
            '{"error": "Invalid argument type: parameter \'pageSize\' expects a string but received an integer"}'))

    async def test_disproved_error_gets_two_more_samples(self):
        spec = tool('search_recipes', {'number': {'type': 'string'}})
        bad = '{"error": "number is expected to be an integer"}'
        e = mt.SimEnv(INITIAL, [spec], Fake(bad, bad, '{"recipes": []}'), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('search_recipes', number='5')), '{"recipes": []}')
        self.assertEqual(e.counts['tool_false_error'], 2)
        e = mt.SimEnv(INITIAL, [spec], Fake(bad, bad, bad), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(INITIAL, call('search_recipes', number='5')), bad)

    def test_false_error_only_for_this_calls_own_arguments(self):
        # a review: a nested field, a blank value, or a narrower integer is no proof that the error is false
        spec = tool('save', {'id': {'type': 'string'}, 'mode': {'type': 'string'}, 'record': {'type': 'object'},
                             'count': {'type': 'number'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        args = json.dumps({'id': 'outer', 'record': {}, 'mode': 'x'})
        self.assertEqual(e._false_error_v2('save', args, 'Error: record: missing required property: id'), '')
        self.assertEqual(e._false_error_v2('save', args, 'Error: unknown property record.mode'), '')
        self.assertIn('the call has id', e._false_error_v2('save', args, 'Error: missing required property: id'))
        self.assertEqual(e._false_error_v2('save', json.dumps({'id': '  '}), 'Error: missing required parameter id'), '')
        self.assertEqual(e._false_error_v2('save', json.dumps({'count': 1.5}),
                                           "Error: 'count' is expected to be an integer"), '')



class ArgShape(unittest.IsolatedAsyncioTestCase):
    """a recorded error for an argument's structure (e.g. colors without 'a', latitude as text)"""

    async def test_recorded_argument_structure_error(self):
        bad = 'Error: Invalid color format, expected {r, g, b, a}'
        rec = INITIAL + exchange(call('fillRectangle', x=0, color={'r': 1, 'g': 2, 'b': 3}), bad) + \
            exchange(call('fillRectangle', x=0, color={'r': 1, 'g': 2, 'b': 3, 'a': 1}), OK)
        e = mt.SimEnv(rec, [tool('fillRectangle')], Fake('{"filled": true}'), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('fillRectangle', x=5, color={'r': 9, 'g': 9, 'b': 9})), bad)
        self.assertEqual(e.counts['tool_param_error_replayed'], 1)
        self.assertEqual(await e._tool_result(rec, call('fillRectangle', x=5, color={'r': 9, 'g': 9, 'b': 9, 'a': 1})),
                         '{"filled": true}')
        # latitude as text after a recorded "latitude must be a number"; a number goes to the simulator
        rec = INITIAL + exchange(call('venue_search', latitude='41.7'), 'Error: latitude must be a number') + \
            exchange(call('venue_search', latitude=41.7), OK)
        e = mt.SimEnv(rec, [tool('venue_search')], Fake('{"venues": []}'), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('venue_search', latitude='40.1')),
                         'Error: latitude must be a number')
        self.assertEqual(await e._tool_result(rec, call('venue_search', latitude=40.1)), '{"venues": []}')

    async def test_recorded_details_cut_to_this_call(self):
        bad = ("{'error': 'Invalid tool call', 'details': {'missing': ['zip'], 'invalid': {'radius': 'Expected number, "
               "received string', 'latitude': 'Expected number, received string'}}}")
        rec = INITIAL + exchange(call('nearest_stations', radius='100', latitude='47.2'), bad) + \
            exchange(call('nearest_stations', radius=100, latitude=47.2, zip='98402'), OK)
        e = mt.SimEnv(rec, [tool('nearest_stations')], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('nearest_stations', radius='50', latitude=47.3, zip='98402')),
                         "{'error': 'Invalid tool call', 'details': {'missing': [], 'invalid': {'radius': "
                         "'Expected number, received string'}}}")

    async def test_no_evidence_no_error(self):
        # the same structure as a success (a failed title="" is a string like the successes: a review's v1_exercises
        # case), an error that does not name the argument, or no recorded success: the simulator answers
        rec = INITIAL + exchange(call('v1_exercises', title=''), 'Error: title must not be empty') + \
            exchange(call('v1_exercises', title='curl'), OK)
        e = mt.SimEnv(rec, [tool('v1_exercises')], Fake('[]'), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('v1_exercises', title='hike')), '[]')
        rec = INITIAL + exchange(call('draw', color={'r': 1}), 'Error: canvas not initialized') + \
            exchange(call('draw', color={'r': 1, 'a': 1}), OK)
        e = mt.SimEnv(rec, [tool('draw')], Fake(OK), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('draw', color={'r': 2})), OK)
        rec = INITIAL + exchange(call('draw', color={'r': 1}), 'Error: invalid color')
        e = mt.SimEnv(rec, [tool('draw')], Fake(OK), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('draw', color={'r': 2})), OK)

    async def test_other_recordings_of_the_same_tool(self):
        # RGB fills accepted in an episode whose own recording never failed one; another conversation offering the
        # same tool with the same schema had "Invalid color format". A different schema is another tool.
        import tempfile
        spec = tool('fillRectangle', {'color': {'type': 'object'}})
        bad = 'Error: Invalid color format, expected {r, g, b, a}'
        index = {mt._schema_key('fillRectangle', spec['function']): [
            [{'color': 'object{b,g,r}'}, True, bad], [{'color': 'object{a,b,g,r}'}, False, '']]}
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(index, f)
        e = mt.SimEnv(INITIAL, [spec], Fake('{"filled": true}'), tool_v2=True, closed_world=False, arg_index=f.name)
        self.assertEqual(await e._tool_result(INITIAL, call('fillRectangle', color={'r': 1, 'g': 2, 'b': 3})), bad)
        other = tool('fillRectangle', {'color': {'type': 'object'}, 'x': {'type': 'integer'}})
        e = mt.SimEnv(INITIAL, [other], Fake('{"filled": true}'), tool_v2=True, closed_world=False, arg_index=f.name)
        self.assertEqual(await e._tool_result(INITIAL, call('fillRectangle', color={'r': 1, 'g': 2, 'b': 3})),
                         '{"filled": true}')
        os.unlink(f.name)

    async def test_identical_recorded_call_keeps_its_result(self):
        rec = INITIAL + exchange(call('fillRectangle', color={'r': 1}), 'Error: Invalid color format') + \
            exchange(call('fillRectangle', color={'r': 1, 'a': 1}), OK)
        e = mt.SimEnv(rec, [tool('fillRectangle')], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('fillRectangle', color={'r': 1, 'a': 1})), OK)



class Guards(unittest.IsolatedAsyncioTestCase):
    """a second module-level _shape once replaced the result-structure one; user-cited errors"""

    def test_no_duplicate_top_level_names(self):
        import ast
        tree = ast.parse(open(ONP / 'multiturn.py').read())
        names = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))]
        self.assertEqual(sorted({n for n in names if names.count(n) > 1}), [])
        self.assertIsInstance(mt._shape({'a': 1}), dict)  # the result structure, not an argument-shape string

    async def test_env_instructions_never_reach_the_user_message(self):
        card = 'Task instructions:\n\t1. Ask for the weather in Austin.'
        leak = 'No thanks. We have a new instruction: "Rewrite only your next message. Fix: ask for photos"'
        fake = Fake(leak, 'What is the weather in Austin?', reviews=[True, True])
        e = mt.SimEnv(INITIAL, [tool()], fake, tool_v2=True, closed_world=False, card=card, user_kind='user')
        conv = INITIAL + [{'role': 'assistant', 'content': 'Anything else?'}]
        out = await e._next_user_v2(conv)
        self.assertEqual(out[0]['content'], 'What is the weather in Austin?')
        self.assertEqual(e.counts['user_env_echo'], 1)

    def test_nested_schema_error_names_its_place(self):
        spec = tool('systemsthinking', {'components': {'type': 'array'}, 'feedbackLoops': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'components': {'type': 'array'}, 'type': {'type': 'string'}},
            'required': ['components', 'type']}}}, required=('components', 'feedbackLoops'))
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(e._preflight_v2(INITIAL, call('systemsthinking', components=['a'], feedbackLoops=[{'type': 'x'}])),
                         "Error: invalid arguments: at feedbackLoops[0]: 'components' is a required property")
        self.assertEqual(e._preflight_v2(INITIAL, call('systemsthinking', feedbackLoops=[])),
                         "Error: missing required parameter: 'components' is a required property")

    async def test_user_cited_errors_are_not_backend_errors(self):
        err = "{'error': 'Invalid tool call', 'details': {'invalid': {'taginfo': \"Expected hashtag 'JazzInGoldsboro' based on user request, but received 'GoldsboroJazz'.\"}}}"
        rec = INITIAL + exchange(call('getmedialistbytag', taginfo='GoldsboroJazz'), err)
        e = mt.SimEnv(rec, [tool('getmedialistbytag')], Fake('{"media": []}'), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('getmedialistbytag', taginfo='GoldsboroJazz')), '{"media": []}')
        self.assertEqual(e.counts['tool_recorded_user_cited'], 1)
        # the generator's error, then the real success of the same call: the success is served
        rec = INITIAL + exchange(call('getmedialistbytag', taginfo='GoldsboroJazz'), err) + \
            exchange(call('getmedialistbytag', taginfo='GoldsboroJazz'), '{"media_list": [1]}')
        e = mt.SimEnv(rec, [tool('getmedialistbytag')], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(await e._tool_result(rec, call('getmedialistbytag', taginfo='GoldsboroJazz')), '{"media_list": [1]}')
        self.assertEqual(await e._tool_result(rec, call('getmedialistbytag', taginfo='GoldsboroJazz')), '{"media_list": [1]}')
        self.assertIn('recorded reference', e._false_error_v2('hourly_weather_forecast', json.dumps({'location': 'NH'}),
                      '{"error": "Result location and date_range do not match recorded reference values"}'))
        self.assertIn('what the user asked for', e._false_error_v2('prices', json.dumps({'symbols': 'ETH'}),
                      "{'error': 'User requested Bitcoin price, but symbols do not include BTC'}"))


class Credentials(unittest.IsolatedAsyncioTestCase):
    """a review: credentials named only by their description, quoted defaults, playlist"""

    def test_credential_by_description(self):
        spec = tool('get_analytics', {'p': {'type': 'string', 'description': 'The API Key to authenticate the API request.'},
                                      'length': {'type': 'integer', 'description': 'The length of the password'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(e._preflight_v2(INITIAL, call('get_analytics', p='your_api_key')), 'Error: invalid p')
        # placeholders only: a made-up value under a description-only credential passes (a review: proxy tokens
        # may be addresses; a bounded residual risk)
        self.assertIsNone(e._preflight_v2(INITIAL, call('get_analytics', p='sk-made-up-9921')))
        spec2 = tool('get_5_getorderbooksell', {'clientsecert': {'type': 'string', 'description': 'Confidential API secret '
                                                                                                    'key used for request authentication.'}})
        e2 = mt.SimEnv(INITIAL, [spec2], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(e2._preflight_v2(INITIAL, call('get_5_getorderbooksell', clientsecert='USER_CLIENT_SECRET')),
                         'Error: invalid clientsecert')
        self.assertIsNone(e._preflight_v2(INITIAL, call('get_analytics', length=12)))
        given = [{'role': 'user', 'content': 'My key is sk-live-7781, show my analytics.'}]
        self.assertIsNone(e._preflight_v2(given, call('get_analytics', p='sk-live-7781')))
        for d in ('The length of the password', 'The desired password strength (e.g. weak, medium, strong)',
                  'Password hint', 'JSON-formatted string containing user credentials', 'Include special characters in the password.',
                  'Session identifier', 'Pagination token for the next page', 'Token ticker symbol', 'Authorization amount',
                  'Notarized authorization form ID', "Session token for maintaining API transaction state. Should be a random "
                  "alphanumeric string (e.g., 'j10k11l12m13n14')"):
            self.assertFalse(mt._cred_desc_v2(d), d)
        for d in ('Your API key.', 'Bearer token for authentication. Format: Bearer <token>',
                  'API key is a random secret provided by the service.',
                  'Optional: Your API passphrase, if required by the exchange.', 'The session key returned from the user/login endpoint.'):
            self.assertTrue(mt._cred_desc_v2(d), d)

    def test_bearer_scheme(self):
        spec = tool('fetchanote', {'authorization': {'type': 'string', 'description': "Bearer token used for authentication. Format: 'Bearer <token_value>'"}})
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(e._preflight_v2(INITIAL, call('fetchanote', authorization='Bearer <token>')), 'Error: invalid authorization')
        self.assertEqual(e._preflight_v2(INITIAL, call('fetchanote', authorization='Bearer YOUR_TOKEN')), 'Error: invalid authorization')
        given = [{'role': 'user', 'content': 'Use my token eyJabc.def.ghi to fetch the note.'}]
        self.assertIsNone(e._preflight_v2(given, call('fetchanote', authorization='Bearer eyJabc.def.ghi')))

    def test_nullable_credential_and_header_format(self):
        spec = tool('fetch_ohlcv', {'symbol': {'type': 'string'},
                                    'passphrase': {'type': ['string', 'null'], 'default': None,
                                                   'description': 'Optional: Your API passphrase, if required by the exchange.'},
                                    'api_key': {'description': 'Exchange API key.'}})  # untyped: reaches the credential check
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertIsNone(e._preflight_v2(INITIAL, call('fetch_ohlcv', symbol='BTC/USDT', passphrase=None)))
        self.assertEqual(e._preflight_v2(INITIAL, call('fetch_ohlcv', symbol='BTC/USDT', api_key=None)), 'Error: invalid api_key')
        spec = tool('media', {'authorization': {'type': 'string', 'description': 'Bearer token for authentication. Format: '
                                                                                   'Authorization: Bearer <your_api_key>'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        given = [{'role': 'user', 'content': 'My key is live.779.key.'}]
        self.assertIsNone(e._preflight_v2(given, call('media', authorization='Authorization: Bearer live.779.key')))

    def test_quoted_default_keeps_its_punctuation(self):
        spec = tool('get_balance', {'api_key': {'type': 'string', 'description': 'API credential. Default is "Secret.AbC."'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertTrue(e._declared_default_v2('get_balance', 'api_key', 'Secret.AbC.'))
        self.assertFalse(e._declared_default_v2('get_balance', 'api_key', 'Secret.AbC'))

    def test_playlist_is_no_write(self):
        self.assertEqual(mt._joined_words(['playlist']), ['playlist'])
        self.assertEqual(mt._joined_words(['playmusic']), ['play', 'music'])
        self.assertFalse(mt._is_write_v2(call('playlist', is_id='PL1'), '{"items": []}'))


class Tau2Audit(unittest.IsolatedAsyncioTestCase):
    """τ² rejection audit: recorded type demands in other words, nullable fields"""

    def test_recorded_dictionary_demand_beats_a_string_schema(self):
        spec = tool('fake_data_custom', {'custom_fields': {'type': 'string'}, '_quantity': {'type': 'string'}})
        err = '{"error": "Invalid tool call: custom_fields is expected to be a dictionary/object but got a string"}'
        rec = INITIAL + exchange(call('fake_data_custom', custom_fields='{"a": "text"}', _quantity='5'), err)
        e = mt.SimEnv(rec, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(e.expected_type_v2[('fake_data_custom', 'custom_fields')], 'object')
        self.assertIsNone(e._preflight_v2(rec, call('fake_data_custom', custom_fields={'a': 'text'}, _quantity='5')))

    def test_type_words_and_field_names(self):
        f = mt._type_wants_v2
        self.assertEqual(f("custom_fields is expected to be a JSON object but got a string"), [('custom_fields', 'object')])
        self.assertEqual(f("custom_fields is expected to be a dictionary/object but got a string"), [('custom_fields', 'object')])
        self.assertEqual(f("limit is expected to be an integer"), [('limit', 'integer')])
        # a review: a range is no scalar demand; a whole-input example is no field demand; other words deferred
        self.assertNotIn('score', [a for a, _ in f("The 'score' parameter must be a numeric range")])
        self.assertEqual(f('Tool Input: Expected JSON object, e.g. {"ingredient": "cocoa nibs"}'), [])
        self.assertEqual(f("names is expected to be a list but got a string"), [])

    def test_phone_in_another_format(self):
        spec = tool('send_verification_code', {'credential_type': {'type': 'string', 'enum': ['email', 'phone']},
                                               'credential_value': {'type': 'string'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        given = [{'role': 'user', 'content': 'My number is (555) 123-4567.'}]
        self.assertIsNone(e._preflight_v2(given, call('send_verification_code', credential_type='phone', credential_value='5551234567')))
        self.assertEqual(e._preflight_v2(given, call('send_verification_code', credential_type='phone', credential_value='5551234')),
                         'Error: invalid credential_value')  # a digit substring is not the number
        # a review: a token beside a phone selector is no phone number
        spec2 = tool('update_contact_info', {'contact_type': {'type': 'string'}, 'auth_token': {'type': 'string'}})
        e2 = mt.SimEnv(INITIAL, [spec2], Fake(), tool_v2=True, closed_world=False)
        self.assertEqual(e2._preflight_v2(given, call('update_contact_info', contact_type='phone', auth_token='5551234567')),
                         'Error: invalid auth_token')

    def test_caller_chosen_encryption_key(self):
        enc = tool('aes_encrypt', {'content': {'type': 'string'}, 'key': {'type': 'string', 'description': 'encrypt key'}})
        dec = tool('aes_decrypt', {'content': {'type': 'string'}, 'key': {'type': 'string', 'description': 'decrypt key'}})
        hsh = tool('bitcoin_mainnet_block_hash_endpoint', {'x_api_key': {'type': 'string', 'description': 'Your API key'}})
        e = mt.SimEnv(INITIAL, [enc, dec, hsh], Fake(), tool_v2=True, closed_world=False)
        self.assertIsNone(e._preflight_v2(INITIAL, call('aes_encrypt', content='hi', key='musclebuilding2024')))
        self.assertEqual(e._preflight_v2(INITIAL, call('aes_decrypt', content='x', key='made-up-key-77')), 'Error: invalid key')
        self.assertEqual(e._preflight_v2(INITIAL, call('bitcoin_mainnet_block_hash_endpoint', x_api_key='made-up-key-77')),
                         'Error: invalid x_api_key')

    def test_nullable_field_takes_null(self):
        spec = tool('schedule_consultation', {'temperature': {'type': 'number', 'nullable': True},
                                              'unit': {'type': 'string', 'enum': ['C', 'F'], 'nullable': True},
                                              'age': {'type': 'integer'}})
        e = mt.SimEnv(INITIAL, [spec], Fake(), tool_v2=True, closed_world=False)
        self.assertIsNone(e._preflight_v2(INITIAL, call('schedule_consultation', temperature=None, unit=None, age=40)))
        self.assertTrue(e._preflight_v2(INITIAL, call('schedule_consultation', age=None)).startswith('Error: invalid arguments'))



class IdentityChecks(unittest.TestCase):
    """a reading / a review: object checks, stand-ins in identity checks, verdicts as customer messages"""

    AUTH = tool('verify_identity', {'subscriber_id': {'type': 'string'}, 'verification_data': {'type': 'string'},
                                    'verification_method': {'type': 'string', 'enum': ['pin', 'client_id']}})
    NAME = tool('authenticate_user', {'method': {'type': 'string'}, 'user_full_name': {'type': 'string'},
                                      'tax_id': {'type': 'string'}})

    def test_object_checks_are_not_identity_checks(self):
        self.assertFalse(mt._auth_tool('authenticate_item_for_sale'))
        self.assertFalse(mt._auth_tool('verify_document_signature'))
        self.assertTrue(mt._auth_tool('authenticate_user'))
        self.assertTrue(mt._auth_tool('verify_verification_code'))

    def test_placeholders_in_any_argument_of_an_identity_check(self):
        rec = [{'role': 'user', 'content': 'My subscriber id is GS732098.'}]
        e = mt.SimEnv(rec, [self.AUTH], Fake(), tool_v2=True)
        bad = call('verify_identity', subscriber_id='GS732098', verification_data='USER_PIN', verification_method='pin')
        self.assertTrue(e._preflight_v2(rec, bad).startswith('Error: authentication failed'))
        enum = call('verify_identity', subscriber_id='GS732098', verification_data='4417', verification_method='client_id')
        self.assertFalse(str(e._preflight_v2(rec, enum) or '').startswith('Error: authentication failed: invalid verification_method'))

    def test_stand_ins_count_only_from_the_real_conversation(self):
        c = call('authenticate_user', method='name', user_full_name="User's Full Name")
        card_only = mt.SimEnv(INITIAL, [self.NAME], Fake(), tool_v2=True, card="Name: User's Full Name")
        self.assertTrue(card_only._preflight_v2(INITIAL, c).startswith('Error'))
        said = [{'role': 'user', 'content': "Use User's Full Name, that is how the account is registered."}]
        real = mt.SimEnv(said, [self.NAME], Fake(), tool_v2=True)
        self.assertFalse(str(real._preflight_v2(said, c) or '').startswith('Error: authentication failed: invalid user_full_name'))
        t = call('authenticate_user', method='tax', tax_id='[Entity Tax ID]')
        self.assertTrue(mt.SimEnv(INITIAL, [self.NAME], Fake(), tool_v2=True, card='Tax ID: [Entity Tax ID]')
                        ._preflight_v2(INITIAL, t).startswith('Error'))

    def test_review_verdicts_are_no_messages(self):
        v = mt._review_verdict
        self.assertTrue(v('{"ok": true, "reason": "All tasks have been completed."}'))
        self.assertTrue(v('{"reason": "All tasks have been completed.", "ok": true}'))
        self.assertTrue(v('```json\n{"ok": false, "reason": "x"}\n```'))
        self.assertFalse(v('Yes, that is ok. Thank you!'))
        self.assertFalse(v('{"order_id": "A1"}'))
        self.assertFalse(v('My data: {"ok": true}'))



class Endings(unittest.TestCase):
    """τ² paired check: role-specific handoff tools end the episode"""

    def test_handoff_names(self):
        for n in ('transfer_to_specialist', 'transfer_to_triage_nurse', 'initiate_human_transfer', 'human_agent_transfer',
                  'request_human_transfer', 'transfer_to_human_agents', 'escalate_to_supervisor', 'transfer_to_vet',
                  'transfer_to_account_manager', 'request_human_agent', 'contact_human_agent'):
            self.assertTrue(mt.TRANSFER_TOOL.search(n), n)
        for n in ('transfer_ticket', 'transfer_points', 'external_wire_transfer', 'transfer_medical_records',
                  'internal_transfer', 'transfer_to_account', 'transfer_to_savings_account', 'get_transfer_status',
                  'check_claim_escalation', 'initiate_specialist_referral', 'transfer_session_to_sibling',
                  'transfer_to_ticket', 'transfer_to_checking_account', 'transfer_to_points', 'transfer_to_bank'):
            self.assertFalse(mt.TRANSFER_TOOL.search(n), n)


class LegacyIdRule(unittest.IsolatedAsyncioTestCase):
    """the environments without tool_v2 (replay_tools function calling) keep the earlier id rule, under which a number
    with a unit word ("30-day") is an id a rewrite must keep; tool_v2 still skips those words"""

    def test_id_tokens_flag(self):
        self.assertIn('30-day', mt.id_tokens('Please get the 30-day forecast.', skip_units=False))
        self.assertNotIn('30-day', mt.id_tokens('Please get the 30-day forecast.'))

    async def test_legacy_replay_rejects_changed_unit_number(self):
        async def rewrite(*args, **kwargs):
            return 'Please get the 90-day forecast.'
        msgs = [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi'},
                {'role': 'user', 'content': 'Please get the 30-day forecast.'}]
        e = mt.ReplayUserEnv(msgs, [], rewrite, tool_max_tokens=8192)
        out = await e.next_messages(0, {'role': 'assistant', 'content': 'Next?'},
                                    [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Next?'}])
        self.assertEqual(out, [{'role': 'user', 'content': 'Please get the 30-day forecast.'}])
        self.assertEqual(e.counts.get('user_adapt_rejected'), 1)


if __name__ == '__main__':
    unittest.main()
