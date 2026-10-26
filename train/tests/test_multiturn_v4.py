"""multiturn.py checks_v4 / close_v4 (checker audit) with a scripted simulator (no GPU, no API);
off by default, so the flag off must leave the old behaviour.
    Run from train/:  ../.venv-client/bin/python tests/test_multiturn_v4.py
"""
import asyncio, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'onpolicy'))
import multiturn as MT

fn = lambda name, args: {'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}
A = lambda text='', calls=None: dict(role='assistant', content=text, **({'tool_calls': calls} if calls else {}))
U = lambda text: {'role': 'user', 'content': text}
T = lambda text: {'role': 'tool', 'content': text}
tool = lambda name, props, desc=None: {'type': 'function', 'function': {'name': name, 'description': desc or name, 'parameters': {
    'type': 'object', 'properties': {p: {'type': 'string'} for p in props}}}}
ok = []


def check(cond, what):
    assert cond, what
    ok.append(what)


class Sim:
    """answers the goal audit with `audit`, reviews with `review`, anything else with `other`; records the prompts"""
    def __init__(self, audit='YES', other='{"result": "simulated"}', review='{"ok": false, "reason": "write ###STOP###"}'):
        self.audit, self.other, self.review, self.calls = audit, other, review, []

    async def __call__(self, messages, max_tokens, **kw):
        self.calls.append(messages)
        c = messages[0]['content']
        if 'Audit whether this customer may end' in c:
            return self.audit
        if 'Return JSON: ' in c:
            return self.review
        return self.other

    def n(self, text):
        return sum(text in m[0]['content'] for m in self.calls)


CS_MSGS = [{'role': 'system', 'content': 'You are a support agent.'}, U('Please cancel order 12345.'),
           A('', [fn('cancel_order', {'order_id': '12345'})]), T('{"status": "cancelled"}'), A('It is cancelled.')]
CARD = ('Reason for contacting representative:\n\tCancel order 12345 before November 2, 2024.\nTask instructions:\n\t1. Ask '
        'to cancel order 12345.')
offer = [U('cancel my order 12345'), A('Your order is cancelled. Would you like me to email you a confirmation?')]
need = [U('cancel my order'), A('I can do that. Could you please provide your order number?')]


def cs_env(sim, **kw):
    return MT.SimEnv(CS_MSGS, [tool('cancel_order', ['order_id'])], sim, tool_v2=True, card=CARD, scenario=True, **kw)


async def customer_tests():
    # waiting gate: off -> refused without audit; v4 -> the narrowed audit decides
    sim = Sim('YES')
    check((await cs_env(sim)._customer_issue_v2(offer, '###STOP###', {})).startswith('The representative awaits'),
          'flag off: a stop while the representative asks is refused')
    sim = Sim('YES'); e = cs_env(sim, checks_v4=True)
    check(await e._customer_issue_v2(offer, '###STOP###', {}) == '' and e._closing_ok_v3 == '###STOP###',
          'v4: a stop on an offer passes when the audit allows it')
    check(sim.n('Answer NO only if you can name one card goal') == 1, 'v4: the narrowed audit prompt is used')
    e = cs_env(Sim('YES'), checks_v4=True)
    check(await e._customer_issue_v2(offer, "No, I don't need anything else, thanks.", {}) == '' and
          e._closing_ok_v3 == "No, I don't need anything else, thanks.", 'v4: a no-need closing goes to the audit and passes')
    e = cs_env(Sim('NO: ask for the refund'), checks_v4=True)
    issue = await e._customer_issue_v2(need, '###STOP###', {})
    check('still open: ask for the refund' in issue, 'v4: audit NO refuses and names the open goal')
    e = cs_env(Sim('YES'), checks_v4=True)
    check(await e._customer_issue_v2(offer, 'Thanks ###STOP###', {}) == 'Do not mix an end marker with a customer reply.',
          'v4: a marker mixed into a reply is still refused')
    e = cs_env(Sim('YES'), checks_v4=True)
    check((await e._customer_issue_v2(offer, '###TRANSFER###', {})).startswith('No actual handoff'),
          'v4: a transfer without a handoff is still refused')
    # fact check: a date by calendar day; an invented code still refused
    conv = [U('cancel'), A('Your pickup is on 10/15/2023. Anything else?')]
    for t in ('Please make sure it is cancelled before 2024-11-02.', 'Yes, 2023-10-15 works.'):
        check(await cs_env(Sim(), checks_v4=True)._customer_issue_v2(conv, t, {}) == '', 'v4: a known date passes: ' + t)
        check((await cs_env(Sim())._customer_issue_v2(conv, t, {})).startswith('Use known facts'), 'flag off: refused: ' + t)
    check((await cs_env(Sim(), checks_v4=True)._customer_issue_v2(conv, 'My code is 839274.', {})).startswith('Use known facts'),
          'v4: an invented code is still refused')
    check((await cs_env(Sim(), checks_v4=True)._customer_issue_v2(conv, 'Move it to 2026-11-02.', {})).startswith('Use known facts'),
          'v4: a date nobody wrote is still refused')
    check(not MT._date_seen_v4('2099-11-02', 'on November 2 please') and MT._date_seen_v4('2024-11-02', 'November 2, 2024') and
          MT._date_seen_v4('2024-11-02', 'November 2 ... in 2024') and not MT._date_seen_v4('2024-02-30', '2024-02-30') and
          MT._date_seen_v4('05/20/1990', 'born 1990-05-20') and not MT._date_seen_v4('2024-06-05', '05/06/2024'),
          'v4: dates: the year needs evidence, real days only, numeric dates month first')
    # no customer reviewer
    sim = Sim(other='I would like to cancel order 12345.'); e = cs_env(sim, checks_v4=True)
    out = await e._next_user_v2(need)
    check(out and out[0]['content'] == 'I would like to cancel order 12345.' and sim.n('Return JSON: ') == 0,
          'v4: no customer reviewer')
    sim = Sim(other='I would like to cancel order 12345.'); e = cs_env(sim)
    await e._next_user_v2(need)
    check(sim.n('Return JSON: ') >= 1, 'flag off: the customer reviewer runs')
    # ends: exactly when allowed
    sim = Sim('YES', other="No, that's all, thank you."); e = cs_env(sim, checks_v4=True)
    check(await e._next_user_v2(offer) is None and e.why == 'complete', 'v4: an allowed closing on an offer ends complete')
    sim = Sim('YES', other="No, that's all, thank you."); e = cs_env(sim)
    r = await e._next_user_v2(offer)
    check(r is not None, 'flag off: the same closing is delivered (the no_close loop)')
    sim = Sim('NO: the refund', other='###STOP###'); e = cs_env(sim, checks_v4=True)
    check(await e._next_user_v2(offer) is None and e.why == 'user_invalid', 'v4: a STOP refused twice ends as user_invalid')
    sim = Sim('NO: the refund', other='Thanks, goodbye.'); e = cs_env(sim, checks_v4=True)
    r = await e._next_user_v2(offer)
    check(r is not None and r[0]['content'] == 'Thanks, goodbye.', 'v4: a closing refused twice is delivered, the talk goes on')
    xfer = [U('cancel'), A('', [fn('transfer_to_human_agents', {'summary': 'x'})]), T('{"status": "transferred"}'),
            A('I have transferred you to a human agent.')]
    e = cs_env(Sim(other='###TRANSFER###'), checks_v4=True)
    check(await e._next_user_v2(xfer) is None and e.why == 'transfer', 'v4: an approved transfer ends as transfer')
    # the card user (user_kind='user') keeps its own path: same answers with the flag on and off
    for cand in ('###STOP###', "No, that's all, thank you.", 'My code is 839274.', 'Please do it before 2024-11-02.'):
        got = []
        for flag in (False, True):
            e = MT.SimEnv(CS_MSGS, [], Sim('NO: x'), tool_v2=True, card=CARD, scenario=True, user_kind='user', checks_v4=flag)
            got.append(await e._customer_issue_v2(offer, cand, {}))
        check(got[0] == got[1], 'v4: the card user is unchanged: ' + cand)
    # an end is approved only after every check passed; a transfer by its marker only
    e = cs_env(Sim('YES'), checks_v4=True)
    issue = await e._customer_issue_v2(offer, 'End of conversation\n839274', {})
    check(issue.startswith('Use known facts') and e._closing_ok_v3 is None, 'v4: an end with an invented code is not approved')
    sim = Sim('YES', other='End of conversation\n839274'); e = cs_env(sim, checks_v4=True)
    r = await e._next_user_v2(offer)
    check(r is not None or e.why != 'complete', 'v4: an end refused twice by the fact check does not end complete')
    sim = Sim('YES', other='Thanks for the transfer, goodbye.'); e = cs_env(sim, checks_v4=True)
    check(await e._next_user_v2(offer) is None and e.why == 'complete', 'v4: "thanks for the transfer" is a closing, not a transfer')
    e = cs_env(Sim(), checks_v4=True)
    check((await e._customer_issue_v2(conv, 'My verification code is 2024-11-02.', {})).startswith('Use known facts'),
          'v4: a date given as a code is refused')
    check((await e._customer_issue_v2(conv, '2024-11-02 is my verification code.', {})).startswith('Use known facts'),
          'v4: a date given as a code (label after) is refused')
    ask = [U('cancel'), A('Please tell me the verification code we sent you.')]
    check((await e._customer_issue_v2(ask, '2024-11-02', {})).startswith('Use known facts'),
          'v4: a bare date in reply to a code question is refused')
    check(await e._customer_issue_v2(conv, 'The ticket for 2024-11-02 is the one I mean.', {}) == '',
          'v4: a date assertion after "ticket" passes')
    check((await e._customer_issue_v2(conv, 'The appointment date is 2024-11-02. My verification code is 2024-11-02.', {}))
          .startswith('Use known facts'), 'v4: every occurrence of a date must be given as a date')
    # with close_v3 on too, no approval before the fact check
    sim = Sim('YES', other='End of conversation\n839274'); e = cs_env(sim, checks_v4=True, close_v3=True)
    r = await e._next_user_v2(offer)
    check(r is not None or e.why != 'complete', 'v4 + close_v3: an end with an invented code does not end complete')

asyncio.run(customer_tests())


# ---- tools
def env(tools, msgs, card='', **kw):
    return MT.SimEnv(msgs, tools, Sim(), tool_v2=True, card=card, scenario=False, **kw)


def tool_tests():
    # order-inferred authentication: kept only where documented or with a token argument
    rec = [{'role': 'system', 'content': 's'}, U('I am C-441, cancel order O-778'),
           A('', [fn('authenticate_customer', {'customer_id': 'C-441', 'pin': '4321'})]), T('{"authenticated": true}'),
           A('', [fn('cancel_order', {'order_id': 'O-778'})]), T('{"status": "cancelled"}'), A('done')]
    plain = [tool('authenticate_customer', ['customer_id', 'pin'], 'Authenticate a customer by id and PIN.'),
             tool('cancel_order', ['order_id'])]
    doc = [tool('authenticate_customer', ['customer_id', 'pin'], 'Authenticate a customer. Required before any action.'),
           tool('cancel_order', ['order_id'])]
    call = fn('cancel_order', {'order_id': 'O-778'})
    msg = 'Error: authentication required for this customer'
    check(env(plain, rec)._preflight_v2([], call) == msg, 'flag off: the order alone requires authentication')
    check(env(plain, rec, checks_v4=True)._preflight_v2([], call) is None, 'v4: the order alone is no requirement')
    check(env(doc, rec, checks_v4=True)._preflight_v2([], call) == msg, 'v4: a documented requirement keeps the order rule')
    tok = [tool('authenticate_customer', ['customer_id', 'pin'], 'Authenticate a customer by id and PIN.'),
           tool('cancel_order', ['order_id', 'auth_token'])]
    rec_t = rec[:4] + [A('', [fn('cancel_order', {'order_id': 'O-778', 'auth_token': 'tk-1'})]), T('{"status": "cancelled"}'), A('done')]
    check(env(tok, rec_t, checks_v4=True)._preflight_v2([], fn('cancel_order', {'order_id': 'O-778', 'auth_token': 'tk-1'})) == msg,
          'v4: a token argument keeps the order rule')
    # code check: a static PIN login is no code check
    check(not MT._check_code_tool_v4('authenticate_user', {'account_id': 'A1', 'access_pin': '8391'}) and
          MT._check_code_tool('authenticate_user', {'account_id': 'A1', 'access_pin': '8391'}), 'v4: a PIN login is no code check')
    check(MT._check_code_tool_v4('authenticate_user', {'email': 'a@b.com', 'verification_pin': '123456'}),
          'v4: a verification PIN is a code check')
    check(MT._check_code_tool_v4('verify_temporary_pin', {'user_id': 'U1', 'pin': '738462'}) and
          MT._check_code_tool_v4('authenticate_user', {'user_id': 'U1', 'pin': '738462'}, 'Checks the one-time PIN sent by SMS.')
          and not MT._check_code_tool_v4('authenticate_user', {'user_id': 'U1', 'pin': '7384'}, 'The 4-digit PIN the customer set.'),
          'v4: temporary / sent PINs by name or description are code checks, a set PIN is not')
    check(MT._auth_doc_v4('Required to access customer records.') and MT._auth_doc_v4('Identity must be verified to access an account.')
          and not MT._auth_doc_v4('Authentication is not required before account access.') and
          not MT._auth_doc_v4('The PIN must be 4 digits.'), 'v4: documented requirement: affirmative, negation-aware')
    check(not MT._auth_token_arg_v4('next_page_token') and MT._auth_token_arg_v4('auth_token') and
          MT._auth_token_arg_v4('token', 'Session token issued at login') and not MT._auth_token_arg_v4('token', 'Page cursor')
          and MT._auth_token_arg_v4('authentication_level') and
          not MT._auth_token_arg_v4('token', 'Continuation token issued by the previous page'), 'v4: auth tokens, not pagination tokens')
    check(MT._auth_doc_v4('Authentication is optional for public data but required before any account change.') and
          MT._auth_doc_v4('Authentication is optional for public data BUT required before any account change.'),
          'v4: negation scoped to its clause (any case)')
    check(not MT._auth_token_arg_v4('session_token', 'Continuation token for the next page of results'),
          'v4: a session-named pagination token is no auth token')
    check(MT._check_code_tool_v4('verify_temporary_pin', {'user_id': 'U441', 'value': '738462'}),
          'v4: a temporary-PIN tool is a code check whatever its value argument is called')
    check(MT._check_code_tool_v4('verify_code', {'user_id': 'U1', 'code': '123456'}) and
          MT._check_code_tool_v4('authenticate_user', {'email': 'a@b.com', 'otp': '123456'}), 'v4: one-time code checks stay')
    # credentials written another way
    rec2 = [{'role': 'system', 'content': 's'}, U('I am John Doe, born January 1, 2000; email maria@globimports.com; '
            'school ID APD_12345, student email jd@university.edu; answers: Lowe\'s hardware store, Wolverines'),
            A('done')]
    tl = [tool('verify_identity', ['full_name', 'date_of_birth', 'email_domain', 'credentials', 'security_answers'],
               'Verify identity. Institutional: School ID and educational domain.')]
    tl[0]['function']['parameters']['properties']['security_answers'] = {'type': 'array', 'items': {'type': 'string'}}
    e0, e4 = env(tl, rec2), env(tl, rec2, checks_v4=True)
    cases = [({'full_name': 'John Doe', 'date_of_birth': '2000-01-01'}, True),
             ({'full_name': 'John Doe', 'date_of_birth': '01/01/2000'}, True),
             ({'full_name': 'John Doe', 'date_of_birth': '1985-04-12'}, False),
             ({'email_domain': 'globimports.com'}, True),
             ({'email_domain': 'example.com'}, False),
             ({'credentials': 'APD_12345 university.edu'}, False),
             ({'credentials': 'jd@university.edu'}, False),
             ({'credentials': 'user@university.edu'}, False),
             ({'security_answers': ["Lowe's hardware store", 'Wolverines']}, False),
             ({'security_answers': ["Home Depot", 'Wolverines']}, False)]
    for a, good in cases:
        r4 = e4._preflight_v2([], fn('verify_identity', a))
        check((r4 is None) == good, 'v4: credential %s -> %s (%s)' % (a, 'pass' if good else 'refused', r4))
        if good:
            check(e0._preflight_v2([], fn('verify_identity', a)) is not None, 'flag off: refused before: %s' % a)
    rec3 = [{'role': 'system', 'content': 's'}, U('Born January 1, 2000. Cancel before November 2, 2024, email jo@acme.org.'), A('ok')]
    e6 = env([tool('authenticate', ['date_of_birth'], 'Authenticate.')], rec3, checks_v4=True)
    check(e6._preflight_v2([], fn('authenticate', {'date_of_birth': '2000-01-01'})) is None and
          e6._preflight_v2([], fn('authenticate', {'date_of_birth': '2024-11-02'})) is not None,
          'v4: a birth date only by birth-date evidence (not a deadline on the card)')
    rec4 = [{'role': 'system', 'content': 's'}, U('DOB unknown; cancel by 11/02/2024.'), A('ok')]
    e7 = env([tool('authenticate', ['date_of_birth'], 'Authenticate.')], rec4, checks_v4=True)
    check(e7._preflight_v2([], fn('authenticate', {'date_of_birth': '2024-11-02'})) is not None,
          'v4: a birth keyword does not reach a date in the next clause')
    rec5 = [{'role': 'system', 'content': 's'}, U('verify me'),
            A('', [fn('authenticate', {'date_of_birth': 'January 1, 2000'})]), T('{"verified": true}'), A('ok')]
    e8 = env([tool('authenticate', ['date_of_birth'], 'Authenticate.')], rec5, checks_v4=True)
    check(e8._preflight_v2([], fn('authenticate', {'date_of_birth': '2000-01-01'})) is None,
          'v4: a recorded date_of_birth field is birth-date evidence')
    for text, good in (('Born Jan. 1, 2000.', True), ('D.O.B. January 1, 2000.', True), ('My dob is 01/01/2000.', True),
                       ('date_of_birth: unknown, due: 11/02/2024', False), ('Osborn delivery 11/02/2024.', False),
                       ('DOB unknown; cancel by 11/02/2024.', False), ('dateOfBirth: 2000-01-01', True),
                       ("{'date_of_birth': 'January 1, 2000'}", True), ("My birth date is 'January 1, 2000'.", True),
                       ('{"date_of_birth2": "January 1, 2000"}', True)):
        want = '2000-01-01' if good else '2024-11-02'
        ex = env([tool('authenticate', ['date_of_birth'], 'Authenticate.')],
                 [{'role': 'system', 'content': 's'}, U(text), A('ok')], checks_v4=True)
        check((ex._preflight_v2([], fn('authenticate', {'date_of_birth': want})) is None) == good,
              'v4: birth evidence %r -> %s %s' % (text, want, 'pass' if good else 'refused'))
    tl2 = [tool('authenticate', ['pin', 'date_of_birth', 'email_domain'], 'Authenticate.')]
    e5 = env(tl2, rec2, checks_v4=True)
    for a in ({'pin': 'globimports.com'}, {'date_of_birth': 'globimports.com'}, {'email_domain': 'acme.org'}):
        check(e5._preflight_v2([], fn('authenticate', a)) is not None, 'v4: no normalization across kinds: %s' % a)
    # placeholders still refused
    check(e4._preflight_v2([], fn('verify_identity', {'credentials': 'YOUR_SCHOOL_ID'})) is not None, 'v4: a placeholder is refused')


tool_tests()


async def review_tests():
    rec = [{'role': 'system', 'content': 's'}, U('status of order 12345'), A('', [fn('get_order', {'order_id': '12345'})]),
           T('{"status": "shipped"}'), A('Shipped.')]
    for flag, want in ((False, True), (True, False)):
        sim = Sim(other='{"status": "processing"}', review='{"ok": true, "reason": ""}')
        e = MT.SimEnv(rec, [tool('get_order', ['order_id'])], sim, tool_v2=True, card='', scenario=False, checks_v4=flag)
        await e._batch([U('status of order 12345')], [fn('get_order', {'order_id': '12345', 'verbose': 'yes'})])
        check((sim.n('Validate the backend result') > 0) == want, 'tool reviewer runs: %s with checks_v4=%s' % (want, flag))

asyncio.run(review_tests())
print('PASS %d checks' % len(ok))


# ---- close_v4: only the customer end path of checks_v4
async def close4_tests():
    sim = Sim('YES'); e = cs_env(sim, close_v4=True)
    check(await e._customer_issue_v2(offer, '###STOP###', {}) == '' and e._closing_ok_v3 == '###STOP###',
          'close_v4: a stop on an offer passes when the narrowed audit allows it')
    check(sim.n('Answer NO only if you can name one card goal') == 1, 'close_v4: the narrowed audit prompt is used')
    e = cs_env(Sim('NO: ask for the refund'), close_v4=True)
    check('still open: ask for the refund' in await e._customer_issue_v2(need, '###STOP###', {}), 'close_v4: audit NO refuses')
    e = cs_env(Sim('YES'), close_v4=True)
    issue = await e._customer_issue_v2(offer, 'End of conversation\n839274', {})
    check(issue.startswith('Use known facts') and e._closing_ok_v3 is None, 'close_v4: an end with an invented code is not approved')
    sim = Sim('YES', other="No, that's all, thank you."); e = cs_env(sim, close_v4=True)
    check(await e._next_user_v2(offer) is None and e.why == 'complete' and sim.n('Return JSON: ') == 0,
          'close_v4: an allowed closing ends complete without the reviewer')
    sim = Sim(other='I would like to cancel order 12345.', review='{"ok": true, "reason": ""}'); e = cs_env(sim, close_v4=True)
    out = await e._next_user_v2(need)
    check(out and sim.n('Return JSON: ') >= 1, 'close_v4: other customer messages are still reviewed')
    sim = Sim('NO: the refund', other='###STOP###'); e = cs_env(sim, close_v4=True)
    check(await e._next_user_v2(offer) is None and e.why == 'user_invalid', 'close_v4: a STOP refused twice ends as user_invalid')
    conv = [U('cancel'), A('Your pickup is on 10/15/2023. Anything else?')]
    check((await cs_env(Sim(), close_v4=True)._customer_issue_v2(conv, 'Yes, 2023-10-15 works.', {})).startswith('Use known facts'),
          'close_v4: the fact check is unchanged (no date normalization)')
    for cand in ('###STOP###', "No, that's all, thank you."):
        got = []
        for flag in (False, True):
            e = MT.SimEnv(CS_MSGS, [], Sim('NO: x'), tool_v2=True, card=CARD, scenario=True, user_kind='user', close_v4=flag)
            got.append(await e._customer_issue_v2(offer, cand, {}))
        check(got[0] == got[1], 'close_v4: the card user is unchanged: ' + cand)
    # tool side unchanged
    rec = [{'role': 'system', 'content': 's'}, U('I am C-441, cancel order O-778'),
           A('', [fn('authenticate_customer', {'customer_id': 'C-441', 'pin': '4321'})]), T('{"authenticated": true}'),
           A('', [fn('cancel_order', {'order_id': 'O-778'})]), T('{"status": "cancelled"}'), A('done')]
    plain = [tool('authenticate_customer', ['customer_id', 'pin'], 'Authenticate a customer by id and PIN.'), tool('cancel_order', ['order_id'])]
    check(env(plain, rec, close_v4=True)._preflight_v2([], fn('cancel_order', {'order_id': 'O-778'})) ==
          'Error: authentication required for this customer', 'close_v4: tool checks unchanged (order rule still refuses)')
    for flag, want in ((False, True), (True, True)):
        sim = Sim(other='{"status": "processing"}', review='{"ok": true, "reason": ""}')
        e = MT.SimEnv([{'role': 'system', 'content': 's'}, U('status of order 12345'), A('', [fn('get_order', {'order_id': '12345'})]),
                       T('{"status": "shipped"}'), A('Shipped.')], [tool('get_order', ['order_id'])], sim, tool_v2=True, card='',
                      scenario=False, close_v4=flag)
        await e._batch([U('status of order 12345')], [fn('get_order', {'order_id': '12345', 'verbose': 'yes'})])
        check((sim.n('Validate the backend result') > 0) == want, 'close_v4: the tool reviewer still runs (close_v4=%s)' % flag)

asyncio.run(close4_tests())
print('PASS %d checks (with close_v4)' % len(ok))
