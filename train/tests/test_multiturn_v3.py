"""multiturn.py v3 switches (close_v3, code_v3, log_v3, fc_typed_keys, fc_write_v2, fc_exact, fc_cred) with a scripted
simulator (no GPU, no API); each switch off must leave the old behaviour.
    Run from train/:  ../.venv-client/bin/python tests/test_multiturn_v3.py
"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onpolicy'))
import multiturn as MT

fn = lambda name, args: {'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
A = lambda text='', calls=None: dict(role='assistant', content=text, **({'tool_calls': calls} if calls else {}))
U = lambda text: {'role': 'user', 'content': text}
T = lambda text: {'role': 'tool', 'content': text}
tool = lambda name, props: {'type': 'function', 'function': {'name': name, 'description': name, 'parameters': {
    'type': 'object', 'properties': {p: {'type': 'string'} for p in props}}}}
ok = []


def check(cond, what):
    assert cond, what
    ok.append(what)


class Sim:
    """answers the goal audit with `audit` and anything else with `other`; records the prompts"""
    def __init__(self, audit='YES', other='{"result": "simulated"}'):
        self.audit, self.other, self.calls = audit, other, []

    async def __call__(self, messages, max_tokens, **kw):
        self.calls.append(messages)
        if 'Audit whether this customer may end' in messages[0]['content']:
            return self.audit
        return self.other


# ---- closing_only / agent_offer_only_v3
check(not MT.closing_only("No, I don't need anything else.") and MT.closing_only("No, I don't need anything else.", True),
      'no-need closing only with v3')
check(not MT.closing_only("I don't need any adjustments, but can you change my address?", True), 'a request is no closing')
check(not MT.closing_only("No, I don't need any adjustments to the order.", True), '"any adjustments" is not "anything else"')
offer = [U('cancel my order'), A('Your order is cancelled. Would you like me to email you a confirmation?')]
need = [U('cancel my order'), A('I can do that. Could you please provide your order number?')]
mixed = [U('cancel my order'), A('Done. Would you like a receipt? Also, please confirm your email address.')]
check(MT.agent_waiting_v2(offer) and MT.agent_offer_only_v3(offer), 'an optional offer is offer-only')
check(MT.agent_waiting_v2(need) and not MT.agent_offer_only_v3(need), 'a request for a detail is not')
check(not MT.agent_offer_only_v3(mixed), 'an offer plus a request is not offer-only')
check(MT.agent_offer_only_v3([U('x'), A('All set. Is there anything else you would like help with today?')]),
      '"anything else you would like help with" is an offer')


# ---- close_v3 in _customer_issue_v2 / _next_user_v2
CS_MSGS = [{'role': 'system', 'content': 'You are a support agent.'}, U('Please cancel order 12345.'),
           A('', [fn('cancel_order', {'order_id': '12345'})]), T('{"status": "cancelled"}'), A('It is cancelled.')]
CARD = 'Reason for contacting representative:\n\tCancel order 12345.\nTask instructions:\n\t1. Ask to cancel order 12345.'


def cs_env(sim, **kw):
    return MT.SimEnv(CS_MSGS, [tool('cancel_order', ['order_id'])], sim, tool_v2=True, card=CARD, scenario=True, **kw)


async def close_tests():
    e = cs_env(Sim('YES'))
    check(await e._customer_issue_v2(offer, '###STOP###', {}) == 'The representative awaits your answer. Answer using the card.',
          'flag off: a stop on an offer is refused as before')
    e = cs_env(Sim('YES'), close_v3=True)
    st = {}
    check(await e._customer_issue_v2(offer, '###STOP###', st) == '' and e._closing_ok_v3 == '###STOP###',
          'close_v3: a stop on an offer passes when the audit says done')
    e = cs_env(Sim('YES'), close_v3=True)
    check(await e._customer_issue_v2(offer, 'No thanks, that is all.', {}) == '' and e._closing_ok_v3 == 'No thanks, that is all.',
          'close_v3: "No thanks, that is all" to an offer closes')
    e = cs_env(Sim('NO: confirm the refund amount'), close_v3=True)
    issue = await e._customer_issue_v2(offer, '###STOP###', {})
    check('still open: confirm the refund amount' in issue, 'close_v3: audit NO on an offer refuses and names the open goal')
    sim = Sim('YES')
    e = cs_env(sim, close_v3=True)
    check((await e._customer_issue_v2(mixed, '###STOP###', {})).startswith('The representative awaits'),
          'close_v3: an offer joined with a request is no offer (no end, no audit)')
    check(not any('Audit whether' in m[0]['content'] for m in sim.calls), 'close_v3: mixed request: no audit')
    joined = [U('refund'), A('Your refund is pending. Would you like a receipt, and what is your bank account number?')]
    check(not MT.agent_offer_only_v3(joined), 'close_v3: an offer joined with a wh-question about the customer is no offer')
    for t in ('Your order is cancelled. Would you like me to explain how refunds work?',
              'Done. Would you like a receipt showing what was refunded?'):
        check(MT.agent_offer_only_v3([U('x'), A(t)]), 'close_v3: an optional explanation is an offer: ' + t[:40])
    consent = [U('pay my bill'), A('Your balance is $40. Would you like to confirm the payment?')]
    check(not MT.agent_offer_only_v3(consent), 'close_v3: consent to the customer\'s own action is no offer')
    e = cs_env(Sim('NO: confirm the refund amount'), close_v3=True)
    issue = await e._customer_issue_v2([U('cancel'), A('It is cancelled.')], '###STOP###', {})
    check('confirm the refund amount' in issue, 'close_v3: a refused end names the open goal')
    sim = Sim('YES')
    e = cs_env(sim, close_v3=True)
    check((await e._customer_issue_v2(need, '###STOP###', {})).startswith('The representative awaits'), 'close_v3: a needed detail still blocks the end')
    check(not any('Audit whether' in m[0]['content'] for m in sim.calls), 'close_v3: no audit while a detail is needed')
    # _next_user_v2 ends on the allowed closing (same decision as the check), not on a refused one
    class E2(MT.SimEnv):
        async def _user(self, conv):
            self._closing_ok_v3 = 'No thanks, that is all.'
            return 'No thanks, that is all.'
    e = E2(CS_MSGS, [], Sim(), tool_v2=True, card=CARD, scenario=True, close_v3=True)
    check(await e._next_user_v2(offer) is None and e.why == 'complete', 'close_v3: the allowed closing ends the episode')
    e = E2(CS_MSGS, [], Sim(), tool_v2=True, card=CARD, scenario=True)
    check(await e._next_user_v2(offer) is not None, 'flag off: a closing on an offer does not end the episode')
    # an end delivered after two refusals (kept candidate) is not a completed task under close_v3
    class E3(MT.SimEnv):
        reply = '###STOP###'
        async def _user(self, conv):
            self._closing_ok_v3 = None
            return self.reply
    done = [U('cancel'), A('It is cancelled.')]
    e = E3(CS_MSGS, [], Sim(), tool_v2=True, card=CARD, scenario=True, close_v3=True)
    check(await e._next_user_v2(done) is None and e.why == 'user_invalid', 'close_v3: an unapproved STOP ends as user_invalid')
    e = E3(CS_MSGS, [], Sim(), tool_v2=True, card=CARD, scenario=True)
    check(await e._next_user_v2(done) is None and e.why == 'complete', 'flag off: an unapproved STOP ends as complete (as before)')
    E3.reply = "No, I don't need anything else."
    e = E3(CS_MSGS, [], Sim(), tool_v2=True, card=CARD, scenario=True, close_v3=True)
    check(await e._next_user_v2(done) is None and e.why == 'user_invalid', 'close_v3: an unapproved closing ends as user_invalid')
    e = E3(CS_MSGS, [], Sim(), tool_v2=True, card=CARD, scenario=True, close_v3=True)
    check(await e._next_user_v2(need) is not None, 'close_v3: an unapproved closing while a detail is needed is delivered (as before)')
    # the whole _user path: audit NO twice -> the kept STOP does not end as complete
    class S3(Sim):
        async def __call__(self, messages, max_tokens, **kw):
            self.calls.append(messages)
            return 'NO: the refund' if 'Audit whether' in messages[0]['content'] else '###STOP###'
    e = cs_env(S3(), close_v3=True)
    check(await e._next_user_v2(done) is None and e.why == 'user_invalid', 'close_v3: two refused STOPs end as user_invalid')
    # an approved transfer ends as transfer
    xfer = [U('cancel'), A('', [fn('transfer_to_human_agents', {'summary': 'x'})]), T('{"status": "transferred"}'),
            A('I have transferred you to a human agent.')]
    e = cs_env(Sim(other='###TRANSFER###'), close_v3=True)
    check(await e._next_user_v2(xfer) is None and e.why == 'transfer', 'close_v3: an approved transfer ends as transfer')

asyncio.run(close_tests())


# ---- code_v3
CODE_MSGS = [{'role': 'system', 'content': 'Support agent.'}, U('Hi, my user id is U-77 and I need a refund.'),
             A('', [fn('send_verification_code', {'user_id': 'U-77'})]), T('{"code_id": "CODE-123456", "status": "sent"}'),
             A('I sent a code to your email. Please share it.'), U('The code is 123456.'),
             A('', [fn('verify_code', {'code_id': 'CODE-123456', 'user_code': '123456'})]), T('{"verified": true}'),
             A('Verified.')]
CODE_CARD = ('Customer details:\n\tUser id U-77\n\tCode 123456\nTask instructions:\n\t1. Ask for a refund.\n\t2. When the code '
             'is sent, give 123456.')
CODE_TOOLS = [tool('send_verification_code', ['user_id']), tool('verify_code', ['code_id', 'user_code'])]


async def code_tests():
    e = MT.SimEnv(CODE_MSGS, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    check(e.recv_code_v3 == '123456' and e.recv_send_v3 == {'user_id': 'U-77'}, 'code_v3: the code value, not the code id')
    before = [U('refund please'), A('Please give me the code from your email.')]
    sent = before[:1] + [A('', [fn('send_verification_code', {'user_id': 'U-77'})]), T('{"status": "sent"}'),
                         A('Sent. Please give me the code.')]
    failed = before[:1] + [A('', [fn('send_verification_code', {'user_id': 'U-77'})]), T('Error: user not found'),
                           A('Please give me the code.')]
    other = before[:1] + [A('', [fn('send_verification_code', {'user_id': 'U-99'})]), T('{"status": "sent"}'),
                          A('Please give me the code.')]
    check(e._code_delivered_v3(before) is None and e._code_delivered_v3(failed) is None, 'code_v3: no code before a successful send')
    check(e._code_delivered_v3(other) is None, 'code_v3: a send to another user delivers nothing')
    check(e._code_delivered_v3(sent) == '123456', 'code_v3: the code after a successful send')
    own = before[:1] + [U('my email is pat@example.com'), A('', [fn('send_verification_code', {'email': 'pat@example.com'})]),
                        T('{"status": "sent"}'), A('Please give me the code.')]
    check(e._code_delivered_v3(own) == '123456', 'code_v3: a send to the customer\'s own e-mail (other key) delivers')
    shop = before[:1] + [U('I wrote to billing@shop.com about this'), A('', [fn('send_verification_code', {'email': 'billing@shop.com'})]),
                         T('{"status": "sent"}'), A('Please give me the code.')]
    check(e._code_delivered_v3(shop) is None, 'code_v3: a merchant address the customer only mentions is not its own')
    conflict = before[:1] + [A('', [fn('send_verification_code', {'user_id': 'U-99', 'email': 'pat@example.com'})]),
                             T('{"status": "sent"}'), A('Please give me the code.')]
    check(e._code_delivered_v3(conflict) is None, 'code_v3: the same argument with another value is another recipient')
    bare_same = before[:1] + [A('', [fn('send_verification_code', {})]), T('{"status": "sent"}'), A('Please give me the code.')]
    bare_other = before[:1] + [A('', [fn('request_otp_code', {})]), T('{"status": "sent"}'), A('Please give me the code.')]
    check(e._code_delivered_v3(bare_same) is None and e._code_delivered_v3(bare_other) is None,
          'code_v3: a send without a destination releases nothing')
    rec3 = [dict(m) for m in CODE_MSGS]
    rec3[2] = A('', [fn('send_verification_code', {'user_id': 'U-77', 'contact_info': 'pat@example.com'})])
    e7 = MT.SimEnv(rec3, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    alias = before[:1] + [A('', [fn('send_verification_code', {'user_id': 'U-77', 'email': 'evil@example.com'})]),
                          T('{"status": "sent"}'), A('Code?')]
    check(e7._code_delivered_v3(alias) is None, 'code_v3: a matching id with a conflicting e-mail under another name releases nothing')
    merch = before[:1] + [U("My merchant's email is vendor@example.com"), A('', [fn('send_verification_code', {'email': 'vendor@example.com'})]),
                          T('{"status": "sent"}'), A('Code?')]
    check(e._code_delivered_v3(merch) is None, 'code_v3: "my merchant\'s email" is not the customer\'s own')
    mine = before[:1] + [U('My registered email address is pat@example.com'), A('', [fn('send_verification_code', {'email': 'pat@example.com'})]),
                         T('{"status": "sent"}'), A('Code?')]
    check(e._code_delivered_v3(mine) == '123456', 'code_v3: "my registered email address is ..." is the customer\'s own')
    newmail = before[:1] + [U('Please use my new email new@example.com'), A('', [fn('send_verification_code', {'contact_info': 'new@example.com'})]),
                            T('{"status": "sent"}'), A('Code?')]
    colon = before[:1] + [U('My email is: pat@example.com.'), A('', [fn('send_verification_code', {'email': 'pat@example.com'})]),
                          T('{"status": "sent"}'), A('Code?')]
    check(e._code_delivered_v3(colon) == '123456', 'code_v3: "My email is: ..." is the customer\'s own')
    idvendor = before[:1] + [A('', [fn('send_verification_code', {'user_id': 'U-77', 'email': 'vendor@example.com'})]),
                             T('{"status": "sent"}'), A('Code?')]
    check(e._code_delivered_v3(idvendor) is None, 'code_v3: the recorded id with an added foreign e-mail releases nothing')
    rec6 = [dict(m) for m in CODE_MSGS]
    rec6[2] = A('', [fn('send_verification_code', {'user_id': 'U-77', 'contact_info': 'a@example.com'})])
    e9 = MT.SimEnv(rec6, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    idown = before[:1] + [U('my new email is new@example.com'), A('', [fn('send_verification_code', {'user_id': 'U-77', 'contact_info': 'new@example.com'})]),
                          T('{"status": "sent"}'), A('Code?')]
    check(e9._code_delivered_v3(idown) == '123456', 'code_v3: the recorded id with the customer\'s own new e-mail releases the code')
    rec7 = [dict(m) for m in CODE_MSGS]
    rec7[2] = A('', [fn('send_verification_code', {'contact_method': 'pat@example.com'})])
    e10 = MT.SimEnv(rec7, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    viam = before[:1] + [A('', [fn('send_verification_code', {'contact_method': 'pat@example.com'})]), T('{"status": "sent"}'), A('Code?')]
    check(e10.recv_code_v3 == '123456' and e10._code_delivered_v3(viam) == '123456', 'code_v3: an address under contact_method is a recipient')
    rec8 = [dict(m) for m in CODE_MSGS]
    rec8[2] = A('', [fn('send_verification_code', {'contact_method': 'email'})])
    e11 = MT.SimEnv(rec8, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    check(e11.recv_code_v3 == '123456' and e11._code_delivered_v3(bare_same) == '123456' and e11._code_delivered_v3(bare_other) is None
          and e11._code_delivered_v3(before) is None, 'code_v3: a recorded send without a recipient: released by the same send tool without one')
    check('123456' not in e11._mask_code_v3(CODE_CARD), 'code_v3: ... and the code stays hidden until then')
    rec9 = [dict(m) for m in CODE_MSGS]
    rec9[2] = A('', [fn('send_verification_code', {'email': 'pat@example.com', 'expires_at': 1792972800})])
    e12 = MT.SimEnv(rec9, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    stamp = before[:1] + [A('', [fn('send_verification_code', {'expires_at': 1792972800})]), T('{"status": "sent"}'), A('Code?')]
    check(e12._code_delivered_v3(stamp) is None, 'code_v3: a matching timestamp is no recipient')
    rec5 = [dict(m) for m in CODE_MSGS]
    rec5[2] = A('', [fn('send_verification_code', {'contact_info': 'a@example.com'})])
    e8 = MT.SimEnv(rec5, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    check(e8._code_delivered_v3(newmail) == '123456', 'code_v3: the customer\'s own new e-mail under the recorded argument name delivers')
    # resend after a failed check: the stored pair is the last send before the successful check
    resend = [{'role': 'system', 'content': 's'}, U('my id is U-77, email old@example.com'),
              A('', [fn('send_verification_code', {'contact_info': 'old@example.com'})]), T('{"status": "sent"}'), U('code 111111'),
              A('', [fn('verify_code', {'user_code': '111111'})]), T('{"verified": false}'), U('try my work email new@example.com'),
              A('', [fn('send_verification_code', {'contact_info': 'new@example.com'})]), T('{"status": "sent"}'), U('code 222222'),
              A('', [fn('verify_code', {'user_code': '222222'})]), T('{"verified": true}')]
    e6 = MT.SimEnv(resend, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    to_old = before[:1] + [A('', [fn('send_verification_code', {'contact_info': 'old@example.com'})]), T('{"status": "sent"}'), A('Code?')]
    to_new = before[:1] + [A('', [fn('send_verification_code', {'contact_info': 'new@example.com'})]), T('{"status": "sent"}'), A('Code?')]
    check(e6.recv_code_v3 == '222222' and e6._code_delivered_v3(to_old) is None and e6._code_delivered_v3(to_new) == '222222',
          'code_v3: a resend: the code of the successful check, released only for its own recipient')
    rec2 = [dict(m) for m in CODE_MSGS]
    rec2[2] = A('', [fn('send_verification_code', {'contact_info': 'a@example.com'})])
    e4 = MT.SimEnv(rec2, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    wrong = before[:1] + [A('', [fn('send_verification_code', {'contact_info': 'b@example.com'})]), T('{"status": "sent"}'),
                          A('Please give me the code.')]
    right = before[:1] + [A('', [fn('send_verification_code', {'contact_info': 'A@example.com'})]), T('{"status": "sent"}'),
                          A('Please give me the code.')]
    check(e4._code_delivered_v3(wrong) is None and e4._code_delivered_v3(right) == '123456',
          'code_v3: a send to another contact delivers nothing, to the recorded contact it does')
    check(e._mask_code_v3('code 123456. mail 123456@example.com id CODE-123456 n 1234567') ==
          'code (the code you receive once the representative has sent it). mail 123456@example.com id CODE-123456 n 1234567',
          'code_v3: masking takes the code as a whole token only')
    # static PIN typed before any send is not taken as the received code; no send in the recording -> no code_v3
    pin = [{'role': 'system', 'content': 's'}, U('my pin is 4321'), A('', [fn('verify_pin', {'pin': '4321'})]), T('{"verified": true}')]
    check(MT.SimEnv(pin, [], Sim(), tool_v2=True, code_v3=True).recv_code_v3 is None, 'code_v3: a static PIN without a send is untouched')
    sim = Sim(other='I have not received a code yet.')
    e.sim = sim
    gen = lambda: [m for m in sim.calls if m[0]['content'].startswith('# Instructions\n- You are playing the role of a customer')][-1][0]['content']
    await e._user(before)
    sys_ = gen()
    check('123456' not in sys_ and 'the code you receive once' in sys_, 'code_v3: the code is hidden on the card before a send')
    await e._user(sent)
    sys_ = gen()
    check('The one-time code you received: 123456' in sys_, 'code_v3: the customer gets the code after the send')
    check((await e._customer_issue_v2(before, 'Here it is: 123456', {})).startswith('No code has reached you'),
          'code_v3: a code before any send is refused')
    check((await e._customer_issue_v2(before, 'The verification code is (the code you receive once the representative has sent it).', {})
           ).startswith('No code has reached you'), 'code_v3: the mask given as a code is refused')
    check(not (await e._customer_issue_v2(before, 'Are you asking for the code you receive once you sign in?', {})).startswith('No code'),
          'code_v3: a question that only resembles the mask is not refused')
    check(not (await e._customer_issue_v2(before, 'Do you mean the code you receive once the representative has sent it? I have not received it yet.', {})).startswith('No code'),
          'code_v3: asking about the mask is not giving it')
    check((await e._customer_issue_v2(before, 'My code is: the code you receive once the representative has sent it', {})).startswith('No code'),
          'code_v3: the mask after "is:" is giving it')
    check(await e._customer_issue_v2(sent, 'The code is 123456.', {}) == '', 'code_v3: the delivered code passes the fact check')
    # a card that lost the code to the redaction: the delivered code is a known fact
    e2 = MT.SimEnv(CODE_MSGS, CODE_TOOLS, Sim(), tool_v2=True, card=CODE_CARD.replace('123456', '(the one the representative gives)'),
                   scenario=True, code_v3=True)
    check(await e2._customer_issue_v2(sent, 'The code is 123456.', {}) == '', 'code_v3: a redacted card still lets the delivered code through')
    class S4(Sim):
        async def __call__(self, messages, max_tokens, **kw):
            self.calls.append(messages)
            return 'My code is 123456.'
    e5 = MT.SimEnv(CODE_MSGS, CODE_TOOLS, S4(), tool_v2=True, card=CODE_CARD, scenario=True, code_v3=True)
    check(await e5._user(before) is None and e5.counts.get('user_code_early_end') == 1,
          'code_v3: an early code refused twice is not delivered (the episode ends)')
    e3 = MT.SimEnv(CODE_MSGS, CODE_TOOLS, Sim(other='ok'), tool_v2=True, card=CODE_CARD, scenario=True)
    await e3._user(before)
    check('Code 123456' in e3.sim.calls[0][0]['content'] and e3.recv_code_v3 is None, 'flag off: the card is shown as before')

asyncio.run(code_tests())


# ---- log_v3
async def log_tests():
    class S2(Sim):
        async def __call__(self, messages, max_tokens, **kw):
            self.calls.append(messages)
            return 'My code is 999999.'  # an invented code every time: the fact check refuses both candidates
    e = cs_env(S2(), log_v3=True)
    out = await e._user([U('cancel order 12345'), A('Please tell me what else you need.')])
    check(len(e.v3_log) == 3 and e.v3_log[0]['by'] == 'check' and e.v3_log[1]['by'] == 'check' and e.v3_log[-1].get('kept'), 'log_v3: rejected and kept candidates logged')

asyncio.run(log_tests())


# ---- fc_* (ReplayUserEnv tool layer, no tool_v2)
FC_TOOLS = [{'type': 'function', 'function': {'name': 'search', 'description': 'web search', 'parameters': {
    'type': 'object', 'properties': {'q': {'type': 'string'}, 'api_key': {'type': 'string'}, 'maxPrice': {'type': 'number'},
                                     'keyInsights': {'type': 'array'}}}}},
            {'type': 'function', 'function': {'name': 'legacy', 'description': 'old api', 'parameters': {
                'type': 'object', 'properties': {'q': {'type': 'string'}, 'apikey': {'type': 'string'}}}}},
            {'type': 'function', 'function': {'name': 'get_book_cover', 'description': 'cover', 'parameters': {
                'type': 'object', 'properties': {'key': {'type': 'string'}, 'value': {'type': 'string'}}}}},
            {'type': 'function', 'function': {'name': 'sequentialthinking', 'description': 'think', 'parameters': {
                'type': 'object', 'properties': {}}}}]
think = {'thought': 'a', 'thoughtNumber': 1, 'totalThoughts': 2, 'nextThoughtNeeded': True}
FC_MSGS = [{'role': 'system', 'content': 'Assistant.'}, U('find cheap wine'),
           A('', [fn('search', {'q': 'wine', 'maxPrice': '30'})]), T('Error: maxPrice expected a number, got a string'),
           A('', [fn('legacy', {'q': 'x', 'apikey': 'test'})]), T('{"ok": true}'),
           A('', [fn('sequentialthinking', think)]), T(json.dumps(think)),
           A('Here you go.'), U('thanks')]


async def fc_tests():
    off = MT.ReplayUserEnv(FC_MSGS, FC_TOOLS, Sim(), adapt_users=False)
    on = MT.ReplayUserEnv(FC_MSGS, FC_TOOLS, Sim(), adapt_users=False, fc_typed_keys=True, fc_write_v2=True, fc_exact=True,
                          fc_cred=True)
    t_off, t_on = off.tool_env, on.tool_env
    ph = fn('search', {'q': 'wine', 'api_key': 'your_api_key_here'})
    check(not (await t_off._tool_result([], ph)).startswith('Error'), 'flag off: a placeholder key is simulated as before')
    check(await t_on._tool_result([], ph) == 'Error: invalid api_key', 'fc_cred: a placeholder key is rejected')
    check(not (await t_on._tool_result([], fn('legacy', {'q': 'y', 'apikey': 'YOUR_API_KEY'}))).startswith('Error'),
          'fc_cred: a tool whose recorded backend took a placeholder accepts one')
    check(not (await t_on._tool_result([], fn('search', {'q': 'w', 'keyInsights': []}))).startswith('Error'), 'fc_cred: keyInsights is no key')
    check(not (await t_on._tool_result([], fn('get_book_cover', {'key': 'ID', 'value': '9780'}))).startswith('Error'),
          'fc_cred: a key / value pair is no credential')
    sup = [U('use the key your_api_key_here please')]
    check((await t_on._tool_result(sup, ph)).startswith('Error'), 'fc_cred: a placeholder only an episode (replayed) user message has is not exempt')
    t_sup = MT.ReplayUserEnv([{'role': 'system', 'content': 's'}, U('use the key your_api_key_here please'), A('ok'), U('bye')],
                             FC_TOOLS, Sim(), adapt_users=False, fc_cred=True).tool_env
    check(not (await t_sup._tool_result([], ph)).startswith('Error'), 'fc_cred: a placeholder the recorded user gave is exempt')
    # typed keys: the recorded string-type error is not served for the corrected number call
    corrected = fn('search', {'q': 'wine', 'maxPrice': 30})
    check((await t_off._tool_result([], corrected)).startswith('Error: maxPrice'), 'flag off: "30" and 30 share the recorded error')
    check(not (await t_on._tool_result([], corrected)).startswith('Error'), 'fc_typed_keys: the corrected call is not given the error')
    check((await t_on._tool_result([], fn('search', {'q': 'wine', 'maxPrice': '30'}))).startswith('Error: maxPrice'),
          'fc_typed_keys: the identical call still gets its recorded error')
    # rejection counter: the string and the number call are counted apart
    for _ in range(2):
        await t_on._tool([], fn('search', {'q': 'z', 'maxPrice': '5', 'api_key': 'YOUR_API_KEY'}))
    await t_on._tool([], fn('search', {'q': 'z', 'maxPrice': 5, 'api_key': 'YOUR_API_KEY'}))
    check(not t_on.stuck, 'fc_typed_keys: two rejections of one call and one of another do not stop the episode')
    # exact echo
    th = dict(think, thought='my own new thought', thoughtNumber=2)
    check(json.loads(await t_on._tool_result([], fn('sequentialthinking', th)))['thought'] == 'my own new thought',
          'fc_exact: sequentialthinking echoes the call')
    nul = {'type': 'function', 'function': {'name': 'sequentialthinking', 'arguments': 'null'}}
    check(t_on._exact_v2([], nul) is None, 'fc_exact: null arguments are not computed (no crash)')
    # nested documented default, password fields, api secrets
    nested = [{'type': 'function', 'function': {'name': 'svc', 'description': 'x', 'parameters': {'type': 'object', 'properties': {
        'auth': {'type': 'object', 'properties': {'api_key': {'type': 'string', 'description': 'Default is YOUR_API_KEY.'}}},
        'password': {'type': 'string'}, 'amadeus_api_secret': {'type': 'string'}}}}}]
    t_n = MT.ReplayUserEnv([{'role': 'system', 'content': 's'}, U('hi'), A('ok'), U('bye')], nested, Sim(), adapt_users=False,
                           fc_cred=True).tool_env
    check(t_n._cred_gate_v3([], fn('svc', {'auth': {'api_key': 'YOUR_API_KEY'}})) is None, 'fc_cred: a nested documented default passes')
    check(t_n._cred_gate_v3([], fn('svc', {'auth': {'api_key': 'your_api_key_here'}})) == 'Error: invalid api_key', 'fc_cred: a nested placeholder is rejected')
    check(t_n._cred_gate_v3([], fn('svc', {'password': 'my_password'})) is None, 'fc_cred: a password (maybe a new secret) is not gated')
    check(t_n._cred_gate_v3([], fn('svc', {'amadeus_api_secret': 'YOUR_AMADEUS_API_SECRET'})) is not None, 'fc_cred: an API secret placeholder is rejected')
    aes = [{'type': 'function', 'function': {'name': 'aes_encrypt', 'description': 'AES encrypt', 'parameters': {'type': 'object', 'properties': {
        'text': {'type': 'string'}, 'key': {'type': 'string', 'description': 'Encryption key, 16 chars; default is your-key-0123456'}}}}}]
    t_a = MT.ReplayUserEnv([{'role': 'system', 'content': 's'}, U('hi'), A('ok'), U('bye')], aes, Sim(), adapt_users=False, fc_cred=True).tool_env
    check(t_a._cred_gate_v3([], fn('aes_encrypt', {'text': 't', 'key': 'your-key-7654321'})) is None, 'fc_cred: a chosen encryption key is no API credential')
    # write rule: an execution tool stays a write, a pairing tool becomes a read
    check(t_on._write(fn('shell-exec', {'cmd': 'echo new > f'}), 'ok'), 'fc_write_v2: shell-exec stays a write')
    check(not t_on._write(fn('dish_pairing_for_wine', {'wine': 'x'}), '{"pairings": []}') and t_off._write(fn('dish_pairing_for_wine', {'wine': 'x'}), '{"pairings": []}'),
          'fc_write_v2: dish_pairing_for_wine is no longer a write')
    check(t_on._write(fn('browser_navigate', {'url': 'u'}), 'ok') and t_on._write(fn('python_exec', {'code': 'x'}), 'ok'),
          'fc_write_v2: navigation and python execution stay writes')
    check(all(t_on._write(fn(n, {'x': 'y'}), 'ok') for n in ('ensure_directory', 'appointmentregistration', 'download_arxiv',
                                                           'sequentialthinking_tools', 'pinterest_search_and_download')),
          'fc_write_v2: tools that are not established reads stay writes')
    check(not t_on._write(fn('walmart_search', {'q': 'x'}), 'ok') and not t_on._write(fn('wine_recommendation', {'q': 'x'}), 'ok'),
          'fc_write_v2: searches and recommendations are reads')
    bad = dict(think, thought={}, nextThoughtNeeded='false')
    check(t_on._exact_v2([], fn('sequentialthinking', bad)) is None, 'fc_exact: wrong field types are not computed')
    mb_tools = [{'type': 'function', 'function': {'name': n, 'description': n, 'parameters': {'type': 'object', 'properties': {}}}}
                for n in ('read_memory_bank_file', 'write_memory_bank_file')]
    mb_rec = [{'role': 'system', 'content': 's'}, U('read'), A('', [fn('read_memory_bank_file', {'filename': 'a.md'})]),
              T(json.dumps({'type': 'text', 'text': 'old'})), A('ok'), U('bye')]
    t_mb = MT.ReplayUserEnv(mb_rec, mb_tools, Sim(), adapt_users=False, fc_exact=True).tool_env
    conv_mb = [A('', [fn('write_memory_bank_file', {'filename': 'a.md'})]), T('ok')]
    check(t_mb._exact_v2(conv_mb, fn('read_memory_bank_file', {'filename': 'a.md'})) is None, 'fc_exact: a write without content is not modelled (no crash)')
    conv_mb2 = [A('', [fn('write_memory_bank_file', {'filename': 'a.md', 'content': 'new'})]), T('ok')]
    check(json.loads(t_mb._exact_v2(conv_mb2, fn('read_memory_bank_file', {'filename': 'a.md'})))['text'] == 'new', 'fc_exact: the file read returns what was written')
    # same-batch context: two sequentialthinking calls in one message, history-shaped recording
    st_keys = {'thoughtNumber': 1, 'totalThoughts': 2, 'nextThoughtNeeded': True, 'branches': [], 'thoughtHistoryLength': 1}
    msgs = [{'role': 'system', 'content': 's'}, U('think'), A('', [fn('sequentialthinking', dict(think))]), T(json.dumps(st_keys)),
            A('done'), U('ok')]
    tb = MT.ReplayUserEnv(msgs, FC_TOOLS, Sim(), adapt_users=False, fc_exact=True).tool_env
    two = A('', [fn('sequentialthinking', dict(think, thought='p')), fn('sequentialthinking', dict(think, thought='q', thoughtNumber=2))])
    res = await tb._batch([U('think'), two], two['tool_calls'])
    check([json.loads(r['content'])['thoughtHistoryLength'] for r in res] == [1, 2], 'fc_exact: same-batch calls see each other')

asyncio.run(fc_tests())



# ---- replay_v3, user_repair_v3, fc_cred holes
class RSim:
    """replayed-user simulator: progression always advances; the adaptation returns `adapt`; the reviewer accepts"""
    def __init__(self, adapt):
        self.adapt, self.calls = adapt, []

    async def __call__(self, messages, max_tokens, **kw):
        self.calls.append(messages)
        t = messages[0]['content']
        if t.startswith('Decide whether the USER'):
            return '{"advance": true}'
        if 'Return JSON: {"ok": true' in t:
            return '{"ok": true, "reason": ""}'
        if 'result of executing a tool' in t:
            return '{"result": "simulated"}'
        return self.adapt


R_MSGS = [{'role': 'system', 'content': 'Assistant with tools.'}, U('find wine for salmon'),
          A('', [fn('search', {'q': 'wine salmon'})]), T('{"top": "Domaine Lumiere Pinot 2019"}'),
          A('I recommend the "Domaine Lumiere Pinot 2019".'), U('Great, order the Domaine Lumiere Pinot 2019 for me.')]


async def replay_tests():
    e = MT.ReplayUserEnv(R_MSGS, FC_TOOLS, RSim('Great, order the Domaine Lumiere Pinot 2019 for me.'), replay_v3=True)
    check(e.v2 and not e.tool_env.tool_v2 and e.replay_v3, 'replay_v3: the v2 user path on the legacy tool layer')
    check(any('Domaine Lumiere Pinot 2019' in v for v in e.learned[0]), 'replay_v3: values of the recorded assistant text count as learned')
    conv = [U('find wine for salmon'), A('Sorry, the search failed; could you tell me your budget?')]
    out = await e.next_messages(1, conv[-1], conv)
    check(out is None and e.why == 'user_invalid', 'replay_v3: a fitted message that keeps an unseen item twice ends the episode')
    e2 = MT.ReplayUserEnv(R_MSGS, FC_TOOLS, RSim('My budget is open. Please find a wine for salmon and order it for me.'), replay_v3=True)
    out = await e2.next_messages(1, conv[-1], conv)
    check(out and 'Domaine' not in out[0]['content'], 'replay_v3: a fitted message without the unseen item is delivered')
    e3 = MT.ReplayUserEnv(R_MSGS, FC_TOOLS, RSim('x'), adapt_users=True)
    check(not e3.v2, 'flag off: the legacy replayed user')
    class Boom(RSim):
        async def __call__(self, messages, max_tokens, **kw):
            if messages[0]['content'].startswith('Decide whether the USER'):
                return '{"advance": true}'
            if 'Return JSON: {"ok": true' in messages[0]['content']:
                return '{"ok": true, "reason": ""}'
            raise RuntimeError('simulator down')  # the adaptation call fails
    e4 = MT.ReplayUserEnv(R_MSGS, FC_TOOLS, Boom('x'), replay_v3=True)
    check(await e4.next_messages(1, conv[-1], conv) is None and e4.why == 'user_invalid', 'replay_v3: a failing adaptation never delivers the planned message')
    class Hold(RSim):
        async def __call__(self, messages, max_tokens, **kw):
            t = messages[0]['content']
            if t.startswith('Decide whether the USER'):
                return 'not json'
            return await RSim.__call__(self, messages, max_tokens, **kw)
    e5 = MT.ReplayUserEnv(R_MSGS, FC_TOOLS, Hold('x'), replay_v3=True)
    check(await e5.next_messages(1, conv[-1], conv) is None and e5.why == 'user_invalid', 'replay_v3: a progression that fails twice ends the episode, not complete')
    LM = [{'role': 'system', 'content': 's'}, U('hi'), A('Hello'), U('1) Order one bottle. 2) Send it to my home.')]
    e6 = MT.ReplayUserEnv(LM, FC_TOOLS, RSim('Please order one bottle and send it to my home.'), replay_v3=True)
    out = await e6.next_messages(0, A('What can I do for you?'), [U('hi'), A('What can I do for you?')])
    check(out and 'order one bottle' in out[0]['content'], 'replay_v3: list markers are not numbers to keep')

asyncio.run(replay_tests())


async def repair_tests():
    class S5(Sim):
        async def __call__(self, messages, max_tokens, **kw):
            self.calls.append(messages)
            return 'My code is 999999.'  # invented: refused twice
    e = cs_env(S5(), user_repair_v3=True)
    check(await e._next_user_v2([U('cancel order 12345'), A('Please tell me what else you need.')]) is None and e.why == 'user_invalid',
          'user_repair_v3: a customer candidate refused twice is not delivered; the episode ends')
    e = cs_env(S5())
    check(await e._next_user_v2([U('cancel order 12345'), A('Please tell me what else you need.')]) is not None,
          'flag off: the refused candidate is delivered as before')

asyncio.run(repair_tests())


def cred_holes_tests():
    tools = FC_TOOLS + [{'type': 'function', 'function': {'name': 'getTokenPrice', 'description': 'crypto price', 'parameters': {
        'type': 'object', 'properties': {'token': {'type': 'string', 'description': 'Token symbol, e.g. btc'}}}}}]
    rec = [{'role': 'system', 'content': 's'}, U('find cheap wine; my key is k-REAL-123'),
           A('', [fn('search', {'q': 'wine', 'api_key': 'k-REAL-123'})]), T('{"ok": true, "next_token": "tok-ISSUED-9"}'),
           A('done'), U('thanks')]
    t = MT.ReplayUserEnv(rec, tools, Sim(), adapt_users=False, fc_cred=True).tool_env
    replayed = [U('use YOUR_API_KEY_HERE as my key')]
    check(t._cred_gate_v3(replayed, fn('search', {'q': 'x', 'api_key': 'YOUR_API_KEY_HERE'})) == 'Error: invalid api_key',
          'fc_cred: a placeholder only a replayed user message has is not supplied')
    check(t._cred_gate_v3([], fn('search', {'q': 'x', 'api_key': 'k-REAL-123'})) is None, 'fc_cred: the recorded user key passes')
    check(t._cred_gate_v3([], fn('search', {'q': 'x', 'api_key': 'AIzaSyINVENTED000'})) == 'Error: invalid api_key',
          'fc_cred: an invented real-looking key is rejected')
    live = [A('', [fn('search', {'q': 'y'})]), T('{"session_api_key": "sk-LIVE-77"}')]
    check(t._cred_gate_v3(live, fn('search', {'q': 'x', 'api_key': 'sk-LIVE-77'})) is None, 'fc_cred: a key issued in this episode passes')
    check(t._cred_gate_v3([], fn('getTokenPrice', {'token': 'sol'})) is None, 'fc_cred: a token symbol is no credential')
    err = [A('', [fn('search', {'q': 'y', 'api_key': 'sk-BAD-77'})]), T('Error: invalid API key sk-BAD-77')]
    check(t._cred_gate_v3(err, fn('search', {'q': 'x', 'api_key': 'sk-BAD-77'})) == 'Error: invalid api_key',
          'fc_cred: a key quoted in an error message is no evidence')
    bearer_tools = tools + [{'type': 'function', 'function': {'name': 'svc2', 'description': 'x', 'parameters': {'type': 'object',
        'properties': {'auth_token': {'type': 'string', 'description': 'Bearer <token>'}, 'p': {'type': 'string',
        'description': 'The API key to authenticate the API request.'}}}}}]
    t2 = MT.ReplayUserEnv(rec, bearer_tools, Sim(), adapt_users=False, fc_cred=True).tool_env
    issued = [A('', [fn('search', {'q': 'y'})]), T('{"access_token": "sk-LIVE-77"}')]
    check(t2._cred_gate_v3(issued, fn('svc2', {'auth_token': 'Bearer sk-LIVE-77'})) is None, 'fc_cred: "Bearer <issued token>" passes')
    check(t2._cred_gate_v3([], fn('svc2', {'p': 'DEMO_KEY_12345'})) == 'Error: invalid p', 'fc_cred: a field described as the API key is checked')


cred_holes_tests()
print('PASS %d checks' % len(ok))
