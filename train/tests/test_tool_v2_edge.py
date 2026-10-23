#!/usr/bin/env python3
"""Deterministic, pytest-free edge tests for the local multiturn.py, including the v2 ports.

Run: python -B train/tests/test_tool_v2_edge.py
Optional: --filter SUBSTRING, --verbose, --report (write the full Markdown report).

No network, model, repository mutation, monkeypatches, or expected-failure markers.
Every case has an independent environment and an async simulator with copied prompts.
The runner continues after failures and exits 1 if any assertion fails, 2 for test
infrastructure errors. Source locations/fix suggestions accompany failing cases.
Retries are deliberately given a different second reply where accepting/rejecting
the first reply incorrectly would change observable behavior. V2 candidates now fail after one retry. Review/progress responses are explicit
protocol fixtures; calls records generation only, auxiliary_calls records the new calls.
"""
import sys

sys.dont_write_bytecode = True  # Includes imports from the read-only repository.

import argparse
import asyncio
import collections
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import traceback

SOURCE = Path(__file__).resolve().parent.parent / 'onpolicy/multiturn.py'
SOURCE_BYTES = SOURCE.read_bytes()
SOURCE_SHA256 = hashlib.sha256(SOURCE_BYTES).hexdigest()
SOURCE_LINES = SOURCE_BYTES.decode().splitlines()
sys.path.insert(0, str(SOURCE.parent))
spec = importlib.util.spec_from_file_location('multiturn_edge_target', SOURCE)
mt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mt)

CASES = []
OK = '{"ok": true}'
ERR = '{"error": "not found"}'
INITIAL = [{'role': 'user', 'content': 'Please help with my request.'}]

# Reported line numbers -> source anchors: a failure is located in the version actually loaded.
ANCHORS = {
    84: '_ERR_TEXT = re.compile',
    88: 'if _is_error(txt):',
    90: "v = _parsed(txt or '')",
    152: '_PLACEHOLDER = re.compile',
    158: 'def _placeholder(v):',
    641: r"t = re.sub(r'^\s*\**\s*(customer|user)",
    642: r"if re.match(r'^\s*\**\s*(agent|representative|assistant)",
    900: "self.rec_cred_keys.setdefault(f.get('name'), set()).update(",
    913: "if _send_code_tool(f.get('name')):",
    1036: "if _auth_tool((c.get('function') or c).get('name')) and not _is_error_v2(o.get('content')):",
    1049: "if _send_code_tool((c.get('function') or c).get('name')) and not _is_error_v2(o.get('content')):",
    1058: ('async def _batch(self, conv, calls):', 5),
    1077: 'rk = (nk, self.writes)',
    1105: ('self.cache_src.setdefault(nk, k)', 1),
    1106: 'if nk in self.cache and (self.cache_src.get(nk) == k or not _is_error_v2(self.cache[nk])):',
    1132: 'miss = sorted(self.schema[k[0]] -',
    1145: 'ph = [a for a, v in args.items()',
    1154: "out = 'Error: authentication required:",
    1158: 'for a, v in args.items():',
    1161: 'if isinstance(v, (str, int)) and not isinstance(v, bool) and str(v).strip()',
    1162: 'set(_name_words(a)) & CRED_WORDS and _ntxt(str(v)) not in cred_seen:',
    1213: 'budget = min(24576, max(self.tool_max_tokens, n_ex // 2 + 4096)',
    1216: 'want_struct = self.tool_v2 and bool(ex)',
    1229: 'off_shape = got is not None',
    1233: 'copied = [(n, a) for n, a in self.rec_calls.get(out.strip(), [])',
    1239: 'differ.append([x for x in sorted(set(mine) | set(theirs))',
    1318: '_ASSISTANT_VOICE = re.compile',
    1359: 'self.learned.append({v for v in vals',
    1382: "seen = _wtxt(' '.join(str(x.get('content') or '') for x in conv or []))",
    1390: "ctx = _ntxt(p) + _ntxt([x.get('content') or '' for x in conv])",
    1419: 'voice = self.v2 and bool(_ASSISTANT_VOICE.search(out))',
    1422: 'if out and out != base and not voice',
    1423: 'or any(_ntxt(t) not in _ntxt(out) for t in keep)',
    1424: 'or any(_ntxt(t) not in ctx for t in id_tokens(out))):',
}


def source_location(original_line):
    anchor = ANCHORS.get(original_line)
    if anchor is None:
        return str(original_line)
    needle, offset = anchor if isinstance(anchor, tuple) else (anchor, 0)
    matches = [i + 1 + offset for i, line in enumerate(SOURCE_LINES) if needle in line]
    if len(matches) != 1:
        return '?'
    return str(matches[0])


def case(name, line, fix):
    def register(fn):
        CASES.append((name, fn, line, fix))
        return fn
    return register


def check(condition, detail):
    if not condition:
        raise AssertionError(detail)


def equal(actual, expected):
    check(actual == expected, 'expected %r; got %r' % (expected, actual))


class Fake:
    def __init__(self, *replies, reviews=()):
        self.replies = replies or (OK,)
        self.calls = []
        self.auxiliary_calls = []
        self.verdicts = iter(reviews)

    async def __call__(self, messages, max_tokens, **kwargs):
        if 'Return JSON: {"ok"' in messages[0]['content']:
            self.auxiliary_calls.append((copy.deepcopy(messages), max_tokens, dict(kwargs)))
            return json.dumps({'ok': next(self.verdicts, True), 'reason': 'Backend candidate violates the test contract.'})
        if messages[0]['content'].startswith('Decide whether the USER'):
            # These historical cases isolate adaptation; pending-goal decisions
            # and invalid advancement are tested in test_tool_v2_port.py.
            self.auxiliary_calls.append((copy.deepcopy(messages), max_tokens, dict(kwargs)))
            return '{"advance": true}'
        self.calls.append((copy.deepcopy(messages), max_tokens, dict(kwargs)))
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        if isinstance(reply, Exception):
            raise reply
        if callable(reply):
            reply = reply(messages, max_tokens, kwargs)
        await asyncio.sleep(0)
        return reply


def call(name='get_item', args=None):
    return {'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args or {})}}


def rawcall(name, args):
    return {'function': {'name': name, 'arguments': args}}


def tool(name='get_item', required=(), properties=None):
    return {'type': 'function', 'function': {'name': name, 'parameters': {
        'type': 'object', 'properties': properties or {}, 'required': list(required)}}}


def exchange(c, out):
    return [{'role': 'assistant', 'content': '', 'tool_calls': [c]}, {'role': 'tool', 'content': out}]


def conversation(*pairs):
    return copy.deepcopy(INITIAL) + [m for c, out in pairs for m in exchange(c, out)]


def env(msgs=None, tools=None, replies=(), closed=False, reviews=(), **kwargs):
    fake = Fake(*replies, reviews=reviews)
    e = mt.SimEnv(INITIAL if msgs is None else msgs, tools or [], fake,
                  closed_world=closed, tool_v2=True, **kwargs)
    return e, fake


async def expect_blocked(e, fake, c, conv=None, contains=None):
    n = len(fake.calls)
    out = await e._tool_result(INITIAL if conv is None else conv, c)
    check(mt._is_error_v2(out), 'bad call accepted: %r' % out)
    equal(len(fake.calls), n)
    if contains:
        check(contains in out, 'wrong rejection: %r' % out)
    return out


async def expect_fallback(e, ctx):
    """an invalid replayed-user rewrite falls back to the planned message (no dropped episode)"""
    before = e.tool_env.counts.get('user_fallback_planned', 0)
    out = await adapt(e, ctx)
    check(e.tool_env.counts.get('user_fallback_planned', 0) == before + 1, 'no planned-message fallback: %r' % out)
    return out


async def expect_invalid(coro):
    try:
        await coro
    except RuntimeError:
        return
    raise AssertionError('invalid simulator candidate should stop the episode')


async def expect_ok(e, c, conv=None, expected=OK):
    out = await e._tool_result(INITIAL if conv is None else conv, c)
    equal(out, expected)
    return out


# Recorded results, episode cache, schema checks, and exact-call error reuse.
@case('reuse.exact_recorded', 1097, 'Preserve exact recorded replay.')
async def _():
    c = call(args={'id': 'A12345'})
    e, s = env(conversation((c, OK)), [tool()])
    await expect_ok(e, c)
    equal(len(s.calls), 0)
    equal(e.counts['tool_recorded'], 1)


@case('reuse.normalized_success', 1097, 'Different exact arguments must be simulated even if normalized forms coincide.')
async def _():
    e, s = env(conversation((call(args={'id': 'A-12345'}), OK)), [tool()])
    await expect_ok(e, call(args={'id': 'a12345'}))
    equal(len(s.calls), 1)


@case('reuse.exact_cached_error', 1106, 'Preserve exact-call cached errors.')
async def _():
    e, s = env(tools=[tool()], replies=(ERR,))
    c = call(args={'limit': '3'})
    equal(await e._tool_result(INITIAL, c), ERR)
    equal(await e._tool_result(INITIAL, c), ERR)
    equal(len(s.calls), 1)


for origin in ('recorded', 'cached'):
    @case('reuse.corrected_type_' + origin, 1098, 'Do not normalize error reuse across argument types.')
    async def _(origin=origin):
        bad, good = call(args={'limit': '3'}), call(args={'limit': 3})
        msgs = conversation((bad, ERR)) if origin == 'recorded' else INITIAL
        e, s = env(msgs, [tool()], replies=(OK,) if origin == 'recorded' else (ERR, OK))
        equal(await e._tool_result(INITIAL, bad), ERR)
        await expect_ok(e, good)
        # Error-only reference examples currently cause an independent shape retry;
        # this case isolates reuse, not the shape policy (tested below).
        check(len(s.calls) >= 1, 'corrected call never reached the simulator')
        equal(e.counts.get('tool_recorded'), 1 if origin == 'recorded' else 0)


@case('reuse.bad_type_after_correction', 1106,
      'Keep errors under exact keys; do not reuse normalized success across schema-incompatible types.')
async def _():
    bad, good = call(args={'limit': '3'}), call(args={'limit': 3})
    e, s = env(tools=[tool(properties={'limit': {'type': 'integer'}})], replies=(OK,))
    check(mt._is_error_v2(await e._tool_result(INITIAL, bad)), 'bad type accepted')
    await expect_ok(e, good)
    check(mt._is_error_v2(await e._tool_result(INITIAL, bad)), 'bad type accepted')


@case('reuse.empty_recording_simulated', 943, 'Keep skipping empty recorded outputs.')
async def _():
    c = call()
    e, s = env(conversation((c, '  ')), [tool()])
    await expect_ok(e, c)
    equal(len(s.calls), 1)


@case('reuse.read_cache_invalidated_by_write', 1068, 'Retain write-dependent cache keys.')
async def _():
    read, write = call('get_item'), call('update_item')
    e, s = env(tools=[tool('get_item'), tool('update_item')], replies=('before', OK, 'after'))
    equal(await e._tool(INITIAL, read), 'before')
    equal(await e._tool(INITIAL, read), 'before')
    equal(await e._tool(INITIAL, write), OK)
    equal(await e._tool(INITIAL, read), 'after')
    equal(len(s.calls), 3)


@case('reuse.recorded_write_sequence', 945, 'Replay recorded reads only after the matching writes.')
async def _():
    write, read = call('update_item', {'value': 2}), call('get_item')
    e, s = env(conversation((write, OK), (read, '{"value": 2}')), [tool('update_item'), tool('get_item')])
    equal(await e._tool(INITIAL, write), OK)
    equal(await e._tool(INITIAL, read), '{"value": 2}')
    equal(len(s.calls), 0)


for label, c, required, fragment in (
    ('unknown_tool', call('missing'), (), 'unknown function'),
    ('missing_required', call(), ('id',), 'missing required'),
    ('array_args', rawcall('get_item', '[]'), (), 'not a JSON object'),
    ('malformed_json', rawcall('get_item', '{bad'), (), 'not a JSON object'),
):
    @case('schema.' + label, 1119, 'Keep rejecting unknown functions, missing fields, and non-object arguments.')
    async def _(c=c, required=required, fragment=fragment):
        e, s = env(tools=[tool(required=required)])
        await expect_blocked(e, s, c, contains=fragment)


@case('schema.punctuated_key_is_not_required_key', 1132,
      'Normalize malformed schema declarations if necessary, but compare actual argument keys literally.')
async def _():
    e, s = env(tools=[tool(required=('id',))])
    await expect_blocked(e, s, call(args={'id:': 'A12345'}))


# Closed world: whole identifiers, placeholders, credential values and prerequisites.
for value in ('USER_ID', 'AUTH_TOKEN', 'YOUR_TOKEN_HERE', 'N/A', 'n/a', '<user_id>', '{{id}}', 'xxx'):
    @case('closed.placeholder_' + value, 152,
          'Match generic placeholders such as N/A case-insensitively; retain the uppercase template-name rule separately.')
    async def _(value=value):
        e, s = env(tools=[tool()], closed=True)
        await expect_blocked(e, s, call(args={'user_id': value}), contains='invalid')


for label, args in (
    ('nested', {'customer': {'user_id': 'USER_ID'}}),
    ('list', {'user_ids': ['USER_ID']}),
    ('digit_template', {'user_id': '<USER_ID_1>'}),
):
    @case('closed.placeholder_' + label, 158 if label == 'digit_template' else 1145,
          'Reject placeholder delimiters before the digit exemption.' if label == 'digit_template' else
          'Recursively walk objects/lists and check placeholders under each identifier/credential key.')
    async def _(args=args):
        # Supplied card templates are now exempt; unsupplied templates still fail.
        e, s = env(tools=[tool()], closed=True, card='No template value supplied.')
        await expect_blocked(e, s, call(args=args))


@case('closed.free_text_placeholder_allowed', 1145, 'Restrict placeholder checks to identity/credential fields.')
async def _():
    e, s = env(tools=[tool()], closed=True)
    await expect_ok(e, call(args={'description': 'YOUR_NAME_HERE'}))


for label, value, valid in (
    ('whole_known_id', 'U123456789', True),
    ('formatted_known_id', 'U-123456789', True),
    ('substring_unknown_id', 'U-12345', False),
):
    @case('closed.' + label, 1174, 'Match whole normalized identifiers, not substrings.')
    async def _(value=value, valid=valid):
        e, s = env(tools=[tool()], closed=True, card='Account U123456789')
        c = call(args={'user_id': value})
        if valid:
            await expect_ok(e, c)
        else:
            await expect_blocked(e, s, c, contains='not found')


@case('closed.live_tool_result_knows_id', 1173, 'Include actual episode tool results in known identifiers.')
async def _():
    e, s = env(tools=[tool()], closed=True)
    await expect_ok(e, call(args={'id': 'NEW12345'}), INITIAL + exchange(call('search'), '{"id": "NEW12345"}'))


@case('closed.student_assertion_does_not_create_id', 1173, 'Do not trust student text/call arguments as backend facts.')
async def _():
    c = call(args={'id': 'FAKE12345'})
    e, s = env(tools=[tool()], closed=True)
    ctx = INITIAL + [{'role': 'assistant', 'content': 'The ID is FAKE12345.', 'tool_calls': [c]}]
    await expect_blocked(e, s, c, ctx)


@case('closed.new_read_knowledge_rechecks_cached_rejection', 1106,
      'Revalidate closed-world rejections when tool-result knowledge expands, even without a write.')
async def _():
    c = call(args={'id': 'NEW12345'})
    e, s = env(tools=[tool(), tool('search_items')], closed=True, replies=('{"id": "NEW12345"}', OK))
    await expect_blocked(e, s, c)
    search = call('search_items')
    result = await e._tool(INITIAL, search)
    await expect_ok(e, c, INITIAL + exchange(search, result))


@case('closed.formatted_phone_is_whole_credential', 1162,
      'Normalize complete structured credential values, rather than splitting phone numbers into tokens.')
async def _():
    e, s = env(tools=[tool('authenticate_customer')], closed=True, card='Phone: (555) 123-4567')
    await expect_ok(e, call('authenticate_customer', {'phone': '5551234567'}))


@case('closed.multiword_security_answer', 1162,
      'Compare normalized whole credential values; a multiword answer is not a single lexical token.')
async def _():
    msgs = [{'role': 'user', 'content': 'My security answer is New York.'}]
    e, s = env(msgs, [tool('authenticate_customer')], closed=True)
    await expect_ok(e, call('authenticate_customer', {'security_answer': 'New York'}))


@case('closed.credential_ignores_schema_and_policy', 888, 'Keep credential evidence separate from schema/policy text.')
async def _():
    msgs = [{'role': 'system', 'content': 'Password may be SECRET.'}] + INITIAL
    e, s = env(msgs, [tool('authenticate_customer')], closed=True)
    await expect_blocked(e, s, call('authenticate_customer', {'password': 'SECRET'}), contains='does not match')


@case('closed.known_credential_allowed', 1162, 'Accept the known credential.')
async def _():
    e, s = env(tools=[tool('authenticate_customer')], closed=True, card='Password: bluebird')
    await expect_ok(e, call('authenticate_customer', {'password': 'bluebird'}))


@case('closed.wrong_credential_blocked', 1162, 'Reject unknown credentials.')
async def _():
    e, s = env(tools=[tool('authenticate_customer')], closed=True, card='Password: bluebird')
    await expect_blocked(e, s, call('authenticate_customer', {'password': 'redbird'}), contains='does not match')


@case('closed.password_case_and_punctuation_matter', 1162,
      'Compare passwords/tokens exactly; reserve formatting normalization for phone/identifier fields.')
async def _():
    e, s = env(tools=[tool('authenticate_customer')], closed=True, card='Password: Blue-Bird!')
    await expect_blocked(e, s, call('authenticate_customer', {'password': 'bluebird'}))


@case('closed.unrelated_fact_is_not_a_password', 1162,
      'Bind credentials to their field and customer instead of accepting any token in the reference.')
async def _():
    e, s = env(tools=[tool('authenticate_customer')], closed=True,
               card='Password: bluebird. Postal code: 90210.')
    await expect_blocked(e, s, call('authenticate_customer', {'password': '90210'}))


@case('closed.nested_wrong_credential', 1158, 'Recursively validate credential fields in nested objects.')
async def _():
    e, s = env(tools=[tool('authenticate_customer')], closed=True, card='Password: bluebird')
    await expect_blocked(e, s, call('authenticate_customer', {'credentials': {'password': 'redbird'}}))


def credential_env(replies=(OK,)):
    auth = call('authenticate_customer', {'password': 'bluebird'})
    return env(conversation((auth, OK)), [tool('authenticate_customer', required=('password',), properties={'password': {'type': 'string', 'minLength': 1}})], closed=True, replies=replies)


@case('closed.missing_recorded_credential', 1147, 'Enforce declared required credentials, never infer requirements from a recording.')
async def _():
    e, s = credential_env()
    await expect_blocked(e, s, call('authenticate_customer'), contains='required')


for label, value in (('empty', ''), ('null', None), ('boolean', False), ('array', []), ('object', {})):
    @case('closed.invalid_credential_' + label, 1161,
          'Require a nonempty scalar of the declared credential type before comparing its value.')
    async def _(value=value):
        e, s = credential_env()
        await expect_blocked(e, s, call('authenticate_customer', {'password': value}))


@case('closed.alternative_credential_sets', 900,
      'Keep successful credential sets as alternatives instead of unioning every recorded key.')
async def _():
    a = call('authenticate_customer', {'password': 'bluebird'})
    b = call('authenticate_customer', {'security_answer': 'robin'})
    definition = tool('authenticate_customer', properties={
        'password': {'type': 'string'}, 'security_answer': {'type': 'string'}, 'remember': {'type': 'boolean'}})
    definition['function']['parameters']['oneOf'] = [
        {'required': ['password']}, {'required': ['security_answer']}]
    e, s = env(conversation((a, OK), (b, OK)), [definition], closed=True)
    await expect_ok(e, call('authenticate_customer', {'password': 'bluebird', 'remember': True}))


def auth_env():
    a = call('authenticate_customer', {'user_id': 'U12345', 'password': 'bluebird'})
    r = call('get_account', {'user_id': 'U12345'})
    e, s = env(conversation((a, OK), (r, '{"balance": 100}'),
                            (call('get_current_account'), '{"balance": 100}')),
               [tool('authenticate_customer'), tool('get_account'), tool('get_current_account'),
                tool('send_verification_code'), tool('find_customer')],
               closed=True, card='Customers U12345 and U67890; password bluebird; email a@example.com')
    for name in ('get_account', 'get_current_account'):
        e.specs_v2[name]['description'] = 'Authentication required for the current account.'
    return e, s, a, r


@case('auth.required_before_protected_read', 1154, 'Preserve authentication prerequisites.')
async def _():
    e, s, a, r = auth_env()
    await expect_blocked(e, s, r, contains='authentication required')


@case('auth.success_allows_protected_read', 1036, 'Recognize a successful identity check in episode history.')
async def _():
    e, s, a, r = auth_env()
    await expect_ok(e, r, INITIAL + exchange(a, OK))


@case('auth.failure_does_not_authorize', 1036, 'Keep failed authentication from authorizing reads.')
async def _():
    e, s, a, r = auth_env()
    await expect_blocked(e, s, r, INITIAL + exchange(a, '{"isError": true}'))


@case('auth.sending_code_is_not_authentication', 1036,
      'Exclude _send_code_tool from passed-auth detection and recorded auth inference.')
async def _():
    e, s, a, r = auth_env()
    send = call('send_verification_code', {'user_id': 'U12345'})
    await expect_blocked(e, s, r, INITIAL + exchange(send, '{"sent": true}'))


@case('auth.customer_scope', 1036, 'Track which customer authenticated and require that identity for protected calls.')
async def _():
    e, s, a, r = auth_env()
    await expect_blocked(e, s, call('get_account', {'user_id': 'U67890'}), INITIAL + exchange(a, OK))


@case('auth.implicit_current_account', 1154,
      'Apply authentication requirements to protected session tools even when no ID argument is supplied.')
async def _():
    e, s, a, r = auth_env()
    await expect_blocked(e, s, call('get_current_account'))


@case('auth.preauth_lookup_remains_usable', 1154,
      'Infer prerequisites per operation; exempt lookups demonstrated before authentication.')
async def _():
    lookup = call('find_customer', {'email': 'a@example.com'})
    auth = call('authenticate_customer', {'user_id': 'U12345', 'password': 'bluebird'})
    read = call('get_account', {'user_id': 'U12345'})
    e, s = env(conversation((lookup, '{"id": "U12345"}'), (auth, OK), (read, OK)),
               [tool('find_customer'), tool('authenticate_customer'), tool('get_account')], closed=True,
               replies=('{"id": "U12345"}',))
    await expect_ok(e, call('find_customer', {'email': 'a@example.com', 'include_name': True}),
                    expected='{"id": "U12345"}')


def code_env(send_result=OK):
    send = call('send_verification_code', {'user_id': 'U12345'})
    verify = call('verify_customer_code', {'user_id': 'U12345', 'code': '654321'})
    e, s = env(conversation((send, send_result), (verify, OK)),
               [tool('send_verification_code'), tool('verify_customer_code')], closed=True,
               card='User U12345, verification code 654321')
    return e, s, send, verify


@case('code.must_send_first', 1150, 'Retain the recorded send-before-verify prerequisite.')
async def _():
    e, s, send, verify = code_env()
    await expect_blocked(e, s, verify, contains='no verification code')


@case('code.successful_send_allows_verification', 1049, 'Recognize successful sends.')
async def _():
    e, s, send, verify = code_env()
    await expect_ok(e, verify, INITIAL + exchange(send, OK))


@case('code.failed_send_does_not_count', 1049, 'Keep failed sends from unlocking verification.')
async def _():
    e, s, send, verify = code_env()
    await expect_blocked(e, s, verify, INITIAL + exchange(send, ERR), contains='no verification code')


@case('code.inference_requires_successful_recorded_send', 913,
      'Set the recorded sent flag only after the matching send result succeeds.')
async def _():
    e, s, send, verify = code_env(ERR)
    # The recording's code check succeeds using the preexisting code in the card,
    # despite the failed resend. An equivalent check must not require a new send.
    verify['function']['arguments'] = json.dumps({'user_id': 'U12345', 'code': '654321', 'purpose': 'identity'})
    await expect_ok(e, verify)


@case('code.send_is_scoped_to_recipient', 1049,
      'Match successful sends to the recipient of the verification call.')
async def _():
    e, s, send, verify = code_env()
    other = call('send_verification_code', {'user_id': 'U67890'})
    await expect_blocked(e, s, verify, INITIAL + exchange(other, OK))


for name, expected in (('validate_zip_code', False), ('verify_email_format', False),
                       ('authenticate_customer', True), ('verify_customer_code', True)):
    @case('auth.classify_' + name, 137, 'Preserve identity-tool classification.')
    def _(name=name, expected=expected):
        equal(mt._auth_tool(name), expected)


for name, expected in (('generate_qr_code', False), ('send_coupon_code', False), ('send_verification_code', True)):
    @case('code.classify_' + name, 161, 'Preserve verification-code sender classification.')
    def _(name=name, expected=expected):
        equal(mt._send_code_tool(name), expected)


# Tool simulator output handling: copies, shape, error exemptions, leaks, budget.
def shape_env(*replies, reviews=(), recorded='{"items": [{"id": "A12345", "price": 1}]}'):
    ref = call('search_items', {'q': 'old'})
    e, s = env(conversation((ref, recorded)), [tool('search_items')], replies=replies, reviews=reviews)
    return e, s, call('search_items', {'q': 'new'})


@case('guard.shape_retry_and_reminder', 1253, 'Keep one shape retry with an explicit reminder.')
async def _():
    good = '{"items": [{"id": "B12345", "price": 2}]}'
    e, s, c = shape_env('{"wrong": 1}', good, reviews=(False, True))
    await expect_ok(e, c, expected=good)
    equal(len(s.calls), 2)
    equal(e.counts.get('tool_reshaped'), 1)
    check('Validation issue:' in s.calls[1][0][1]['content'], 'missing shape reminder')


@case('guard.copy_retry_and_differing_key', 1233, 'Keep one copy retry naming the changed argument.')
async def _():
    old, good = '{"price": 1}', '{"price": 2}'
    e, s, c = shape_env(old, good, recorded=old, reviews=(False, True))
    await expect_ok(e, c, expected=good)
    equal(len(s.calls), 2)
    check('Validation issue:' in s.calls[1][0][1]['content'], 'missing argument difference')


@case('guard.copy_numeric_decimal_difference', 1239,
      'Compare typed argument values for copies; _ntxt conflates 1.5 and 15 (and negative/positive values).')
async def _():
    ref = call('compute_double', {'n': 1.5})
    e, s = env(conversation((ref, '{"value": 3}')), [tool('compute_double')],
               replies=('{"value": 3}', '{"value": 30}'), reviews=(False, True))
    await expect_ok(e, call('compute_double', {'n': 15}), expected='{"value": 30}')
    equal(len(s.calls), 2)


@case('guard.cross_tool_equal_result_not_copy', 1233,
      'Restrict copy detection to recorded calls of the current tool.')
async def _():
    ref = call('get_room_status', {'room': 'R1'})
    e, s = env(conversation((ref, '{"status": "available"}')),
               [tool('get_room_status'), tool('get_car_status')],
               replies=('{"status": "available"}', '{"status": "unavailable"}'))
    await expect_ok(e, call('get_car_status', {'car': 'C1'}), expected='{"status": "available"}')
    equal(len(s.calls), 1)


@case('guard.generic_error_not_a_bad_copy', 1233,
      'Exempt genuine errors from the copy guard: different bad calls may return the same error.')
async def _():
    e, s, c = shape_env(ERR, OK, recorded=ERR)
    await expect_ok(e, c, expected=ERR)
    equal(len(s.calls), 1)


for label, error in (
    ('top_level_error', ERR),
    ('mcp_error', '{"content": [{"type": "text", "text": "Error: no such item"}], "isError": true}'),
    ('mcp_text_error', '{"type": "text", "text": "Error: no such item"}'),
    ('mcp_nested_text_error', '{"content": [{"type": "text", "text": "Error: no such item"}]}'),
    ('mcp_list_error', '[{"type": "text", "text": "Error: no such item"}]'),
    ('success_false', '{"success": false}'),
    ('failed_status', '{"status": "failed"}'),
    ('plain_error', 'Error: no such item'),
):
    @case('guard.preserve_' + label, 1229,
          'Use _is_error_v2(out) to exempt errors from both shape and structured-output retries.')
    async def _(error=error):
        e, s, c = shape_env(error, '{"items": [{"id": "B12345", "price": 2}]}')
        await expect_ok(e, c, expected=error)
        equal(len(s.calls), 1)


@case('guard.empty_search_result_is_valid_shape', 1229,
      'Treat empty arrays as compatible with recorded nonempty arrays; item count depends on arguments.')
async def _():
    e, s, c = shape_env('{"items": []}', '{"items": [{"id": "B12345", "price": 2}]}')
    await expect_ok(e, c, expected='{"items": []}')
    equal(len(s.calls), 1)


@case('guard.success_after_error_only_examples', 1216,
      'Build expected success shapes only from non-error reference results; an error schema is not a success schema.')
async def _():
    good = '{"items": [{"id": "B12345", "price": 2}]}'
    e, s, c = shape_env(good, ERR, recorded=ERR)
    await expect_ok(e, c, expected=good)
    equal(len(s.calls), 1)


@case('guard.error_named_field_does_not_exempt_success', 1229,
      'Exempt actual errors via _is_error_v2(out), not every object containing an error-named key.')
async def _():
    good = '{"items": [{"id": "B12345", "price": 2}]}'
    e, s, c = shape_env('{"error_count": 0, "unrelated": true}', good, reviews=(False, True))
    await expect_ok(e, c, expected=good)
    equal(len(s.calls), 2)


@case('guard.bounded_best_effort_after_two_bad_shapes', 1220, 'Stop after one retry when both candidates fail review.')
async def _():
    e, s, c = shape_env('{"wrong": 1}', 'still malformed', reviews=(False, False))
    await e._tool_result(INITIAL, c)  # kept after one retry: the episode goes on (a check run dropped half the episodes)
    equal(e.counts.get('tool_invalid'), 1)
    equal(e.counts.get('tool_kept_after_review'), 1)
    equal(len(s.calls), 2)


@case('guard.python_repr_and_number_types', 1219, 'Treat Python reprs as structured and int/float shapes alike.')
async def _():
    e, s, c = shape_env("{'items': [{'id': 'B12345', 'price': 2.5}]}")
    await expect_ok(e, c, expected="{'items': [{'id': 'B12345', 'price': 2.5}]}")
    equal(len(s.calls), 1)


@case('guard.unfence_and_plain_unicode', 1224, 'Strip a whole-output code fence and normalize nonbreaking punctuation.')
async def _():
    e, s = env(tools=[tool()], replies=('```json\n{"id": "AP\u20111144"}\n```',))
    await expect_ok(e, call(), expected='{"id": "AP-1144"}')
    equal(e.counts.get('tool_unfenced'), 1)


LEAK = '<|start|>assistant<|channel|>final<|message|>{"value": 7}<|return|>'


@case('guard.leak_retries', 1241, 'Retry leaked model formatting once.')
async def _():
    e, s = env(tools=[tool()], replies=(LEAK, '{"value": 8}'), reviews=(False, True))
    await expect_ok(e, call(), expected='{"value": 8}')
    equal(len(s.calls), 2)


@case('guard.persistent_leak_unwrapped', 1257, 'Unwrap the final retry when model markers persist.')
async def _():
    e, s = env(tools=[tool()], replies=(LEAK,))
    await expect_ok(e, call(), expected='{"value": 7}')
    equal(len(s.calls), 1)
    equal(e.counts.get('tool_unleaked'), 1)


@case('guard.recorded_leak_not_exposed', 1105,
      'Apply result sanitization to replay/cache fast paths as well as simulator responses.')
async def _():
    c = call()
    e, s = env(conversation((c, LEAK)), [tool()])
    result = await e._tool_result(INITIAL, c)
    check(not mt._LEAK.search(result), 'recorded format tokens exposed: %r' % result)


@case('guard.prompt_tags_effort_and_accept_cut', 1192, 'Preserve tagged examples, role effort, and accept_cut.')
async def _():
    e, s, c = shape_env('{"items": [{"id": "B12345", "price": 2}]}')
    e.tool_effort = 'high'
    await e._tool_result(INITIAL, c)
    m, budget, kw = s.calls[0]
    check('<recorded_result 1>' in m[1]['content'], 'missing example delimiters')
    check('Known values' not in m[1]['content'] and '<report' not in m[1]['content'], 'obsolete prompt material')
    equal(kw, {'accept_cut': True, 'effort': 'high'})
    check(2048 <= budget <= 24576, 'invalid tool budget')


@case('guard.large_reference_grows_budget', 1213, 'Grow the result budget with reference size, within the configured ceiling.')
async def _():
    recorded = json.dumps({'text': 'a' * 30000})
    good = json.dumps({'text': 'b' * 30000})
    e, s, c = shape_env(good, recorded=recorded)
    await expect_ok(e, c, expected=good)
    check(8192 < s.calls[0][1] <= 24576, 'large output did not receive a larger bounded budget')


@case('guard.small_context_budget', 1213,
      'Clip the prompt before sampling and cap output by remaining space; remove the unconditional 2048-token floor.')
async def _():
    recorded = json.dumps({'text': '1 ' * 20000})
    e, s, c = shape_env('{"text": "empty"}', recorded=recorded)
    e.sim_ctx = 4096
    await expect_invalid(e._tool_result(INITIAL, c))
    equal(e.counts.get('sim_context_rejected'), 1)
    equal(len(s.calls), 0)
    for messages, budget, _ in s.calls:
        estimate = sum(len(m['content']) for m in messages) // 3
        check(estimate + budget + 1024 <= e.sim_ctx,
              'prompt estimate %d + budget %d + reserve 1024 > sim_ctx %d' % (estimate, budget, e.sim_ctx))


# Errors and writes. Each row is independently reported.
for label, text, expected in (
    ('plain_error', 'Error: connection failed', True),
    ('json_error', '{"error":"bad request"}', True),
    ('python_failure', "{'success': False, 'data': None}", True),
    ('mcp_list', '[{"type":"text","text":"Error executing tool: boom"}]', True),
    ('mcp_nested', '{"content":[{"type":"text","text":"Error: boom"}],"isError":false}', True),
    ('explicit_is_error', '{"isError":true}', True),
    ('empty_error', '{"error":"", "success":true}', False),
    ('normal_diagnostic', '{"type":"text","text":"Found 3 errors in your code"}', False),
    ('plain_failed', 'Failed: connection refused', True),
    ('plain_exception', 'Exception: connection refused', True),
    ('plain_failure', 'Failure: connection refused', True),
    ('json_string_error', '"Error: connection refused"', True),
    ('zero_error_metric', 'Error count: 0', False),
    ('chinese_failure', '[{"type":"text","text":"\u521b\u5efa\u6587\u4ef6\u5931\u8d25"}]', True),
    ('chinese_zero_failures', '{"type":"text","text":"\u5931\u8d25\u6b21\u6570\uff1a0\uff1b\u4efb\u52a1\u6210\u529f"}', False),
):
    error_line = 84 if label == 'chinese_zero_failures' else 88 if label == 'zero_error_metric' else 90
    error_fix = ('Exclude diagnostic count/rate headings before calling _is_error.' if label == 'zero_error_metric' else
                 'Exclude zero-failure diagnostic counts from the unanchored Chinese failure match.'
                 if label == 'chinese_zero_failures' else
                 'Apply the failure-text predicate to raw text and parsed strings before walking dict/list envelopes.')
    @case('error.' + label, error_line, error_fix)
    def _(text=text, expected=expected):
        equal(mt._is_error_v2(text), expected)


for name, expected in (
    ('create_room', True), ('book_flight', True), ('send_email', True), ('drawing_fillRectangle', True),
    ('updateUser', True), ('api.create_order', True), ('authenticate_customer', True), ('connect', True),
    ('api.get_order', False), ('get_order', False), ('get_update_status', False), ('airbnb_search', False),
    ('sequentialthinking', False), ('GetChartImgLink', False), ('multiply', False),
    ('validate_zip_code', False), ('transfer_to_human', False),
):
    @case('write.' + name, 180, 'Preserve method-name write verbs and read/auth/transfer exceptions.')
    def _(name=name, expected=expected):
        equal(mt._is_write_v2(call(name), OK), expected)


@case('write.failed_write_does_not_change_state', 1081, 'Do not append failed writes to the state sequence.')
async def _():
    e, s = env(tools=[tool('connect')], replies=('{"isError":true}',))
    await e._tool(INITIAL, call('connect'))
    equal(e.writes, ())


@case('write.read_extra_override', 1086, 'Keep read_extra as an explicit write-classification override.')
async def _():
    e, s = env(tools=[tool('run_query')], read_extra='^run_query$')
    await e._tool(INITIAL, call('run_query'))
    equal(e.writes, ())


# Batch ordering, stop thresholds and state resets.
@case('batch.later_prompt_sees_earlier_result', 1060, 'Keep sequential context and leave input conversation untouched.')
async def _():
    a, b = call(args={'q': 'a'}), call(args={'q': 'b'})
    ctx = INITIAL + [{'role': 'assistant', 'tool_calls': [a, b]}]
    before = copy.deepcopy(ctx)
    e, s = env(tools=[tool()], replies=('{"id":"R12345"}', '{"id":"R67890"}'))
    results = await e._batch(ctx, [a, b])
    equal([r['content'] for r in results], ['{"id":"R12345"}', '{"id":"R67890"}'])
    equal(ctx, before)
    check('TOOL RESULT:' not in s.calls[0][0][1]['content'], 'first call saw a future result')
    check('TOOL RESULT: {"id":"R12345"}' in s.calls[1][0][1]['content'], 'second call missed preceding result')


@case('batch.same_batch_auth_unlocks_read', 1060, 'Recognize successful auth from earlier calls of the same batch.')
async def _():
    e, s, auth, read = auth_env()
    ctx = INITIAL + [{'role': 'assistant', 'tool_calls': [auth, read]}]
    out = await e._batch(ctx, [auth, read])
    equal(out[0]['content'], OK)
    equal(out[1]['content'], '{"balance": 100}')
    equal(len(s.calls), 0)


@case('batch.same_batch_send_unlocks_code', 1060, 'Recognize successful code sending within the current batch.')
async def _():
    e, s, send, verify = code_env()
    out = await e._batch(INITIAL + [{'role': 'assistant', 'tool_calls': [send, verify]}], [send, verify])
    equal([r['content'] for r in out], [OK, OK])


for label, name, reply, limit, why in (
    ('read', 'get_item', OK, mt.REPEAT_ANY, 'stopped: repeated call'),
    ('write', 'send_email', OK, mt.REPEAT_ANY, 'stopped: repeated call'),
    ('rejection', 'connect', '{"type":"text","text":"Error: refused"}', mt.REJECT_REPEAT,
     'stopped: repeated rejected call'),
):
    @case('stop.' + label, 1072, 'Preserve repeat thresholds, including successful writes and MCP errors.')
    async def _(name=name, reply=reply, limit=limit, why=why):
        e, s = env(tools=[tool(name)], replies=(reply,))
        for n in range(1, limit + 1):
            await e._tool(INITIAL, call(name))
            equal(e.stuck, n == limit)
        equal(e.stuck_why, why)


@case('stop.rejection_count_resets_after_state_change', 1077, 'Count rejections separately after successful writes.')
async def _():
    e, s = env(tools=[tool('get_item'), tool('update_item')], replies=(ERR, OK, ERR))
    for _ in range(2):
        await e._tool(INITIAL, call())
    await e._tool(INITIAL, call('update_item'))
    for _ in range(2):
        await e._tool(INITIAL, call())
    equal(e.stuck, False)


@case('stop.distinct_rejected_queries_do_not_merge', 1077,
      'Use exact argument keys for rejected-call counters, not formatting normalization of arbitrary strings.')
async def _():
    e, s = env(tools=[tool()], replies=(ERR,))
    for query in ('A+B', 'AB', 'A B'):
        await e._tool(INITIAL, call(args={'query': query}))
    equal(len(s.calls), 3)
    equal(e.stuck, False)


@case('stop.batch_does_not_execute_after_stop', 1058,
      'Break the batch as soon as _tool sets stuck; do not execute later writes.')
async def _():
    e, s = env(tools=[tool(), tool('send_email')])
    for _ in range(mt.REPEAT_ANY - 1):
        await e._tool(INITIAL, call())
    n = len(s.calls)
    calls = [call(), call('send_email', {'body': 'Should never be sent'})]
    await e.next_messages(0, {'role': 'assistant', 'tool_calls': calls},
                          INITIAL + [{'role': 'assistant', 'tool_calls': calls}])
    equal(e.why, 'stopped: repeated call')
    equal(len(s.calls), n)
    equal(e.writes, ())


# Replay adaptation: the user only knows what the assistant actually showed.
FLIGHT = '{"flights": [{"id": "MX123", "price": 349.98}]}'
PLANNED = 'Book MX123 ($349.98) for 4 people please.'
REWRITE = 'Please find and book a flight for 4 people.'
NO_FLIGHTS = {'role': 'assistant', 'content': 'I could not find flights.'}


def replay(*replies, planned=PLANNED, more_users=(), **kwargs):
    c = call('search_flights', {'to': 'MEX'})
    msgs = [{'role': 'user', 'content': 'Find flights to Mexico City.'}] + exchange(c, FLIGHT)
    msgs += [{'role': 'assistant', 'content': 'MX123 at 349.98'}, {'role': 'user', 'content': planned}]
    msgs += [m for u in more_users for m in ({'role': 'assistant', 'content': 'Okay.'}, {'role': 'user', 'content': u})]
    s = Fake(*replies)
    e = mt.ReplayUserEnv(msgs, [tool('search_flights')], s, tool_v2=True, **kwargs)
    return e, s, e.initial + [NO_FLIGHTS]


async def adapt(e, ctx):
    return (await e.next_messages(0, ctx[-1], ctx))[0]['content']


@case('replay.unseen_can_be_removed', 1384, 'Keep allowing removal of unseen result facts while preserving the request.')
async def _():
    e, s, ctx = replay(REWRITE)
    equal(await adapt(e, ctx), REWRITE)
    equal(e.counts.get('user_unseen_rewritten'), 1)
    check('MX123' in s.calls[0][0][0]['content'].split('\n<not shown>\n')[1].split('</not shown>')[0], 'missing unseen id')


@case('replay.shown_id_must_stay', 1394, 'Retain IDs the assistant actually showed.')
async def _():
    e, s, ctx = replay(REWRITE)
    ctx[-1] = {'role': 'assistant', 'content': 'Flight MX123 costs 349.98.'}
    await expect_fallback(e, ctx)
    equal(e.counts.get('user_unseen', 0), 0)


@case('replay.tool_only_id_is_still_unseen', 1382,
      'Compute user-visible seen values from user/system/assistant text, excluding tool result content.')
async def _():
    e, s, ctx = replay(REWRITE)
    ctx = e.initial + exchange(call('search_flights', {'to': 'MEX'}), FLIGHT) + [NO_FLIGHTS]
    equal(await adapt(e, ctx), REWRITE)
    equal(e.counts.get('user_unseen'), 1)


@case('replay.call_arguments_do_not_show_id', 1382, 'Keep excluding assistant call arguments from user-visible facts.')
async def _():
    e, s, ctx = replay(REWRITE)
    ctx.insert(1, {'role': 'assistant', 'content': '', 'tool_calls': [call('search_flights', {'id': 'MX123'})]})
    equal(await adapt(e, ctx), REWRITE)


@case('replay.retries_unchanged_unseen', 1400, 'Retry an unchanged rewrite when it still presumes unseen results.')
async def _():
    e, s, ctx = replay(PLANNED, REWRITE)
    equal(await adapt(e, ctx), REWRITE)
    equal(len(s.calls), 2)


@case('replay.changed_text_still_assumes_unseen_id', 1422,
      'Reject rewrites that still presume unseen results; rewording alone does not resolve unseen facts.')
async def _():
    bad = 'Please book MX123 ($349.98) for 4 people.'
    e, s, ctx = replay(bad, REWRITE)
    equal(await adapt(e, ctx), REWRITE)
    equal(len(s.calls), 2)


@case('replay.unseen_survives_previous_adaptation', 1359,
      'Determine learned/known facts from actual replay history; a dropped recorded user ID stays unseen later.')
async def _():
    second = 'Please confirm MX123 ($349.98) for 4 people.'
    next_rewrite = 'Please find and confirm a flight for 4 people.'
    e, s, ctx = replay(REWRITE, next_rewrite, more_users=(second,))
    first = await adapt(e, ctx)
    equal(first, REWRITE)
    ctx += [{'role': 'user', 'content': first}, NO_FLIGHTS]
    equal(await adapt(e, ctx), next_rewrite)


@case('replay.new_identifier_rejected', 1424, 'Reject genuinely new identifiers.')
async def _():
    planned = 'Please check order AB12345 for me.'
    e, s, ctx = replay('Please check order ZZ99999 for me.', planned=planned)
    await expect_fallback(e, ctx)


@case('replay.new_identifier_substring_rejected', 1424,
      'Match whole identifiers for newly introduced facts instead of substring membership in _ntxt(ctx).')
async def _():
    planned = 'Please check order AB12345 for me.'
    e, s, ctx = replay('Please check order AB12345 and order AB123.', planned=planned)
    await expect_fallback(e, ctx)


@case('replay.required_identifier_substring_not_enough', 1423,
      'Use whole identifier matching when checking retained facts.')
async def _():
    planned = 'Please check both AB12345 and AB123 for me.'
    e, s, ctx = replay('Please check AB12345 for me.', planned=planned)
    await expect_fallback(e, ctx)


@case('replay.small_numbers_preserved', 1424,
      'Check all explicit numeric facts, not just long ID-like tokens.')
async def _():
    planned = 'Please book a flight for 4 people.'
    e, s, ctx = replay('Please book a flight for 40 people.', planned=planned)
    await expect_fallback(e, ctx)


@case('replay.assistant_voice_rejected', 1419, 'Preserve rejection of clear assistant-voice rewrites.')
async def _():
    e, s, ctx = replay("I'm not able to create or export files, but here is your summary.")
    await expect_fallback(e, ctx)
    equal(e.counts.get('user_adapt_voice'), 2)


@case('replay.user_access_problem_is_user_voice', 1318,
      'Do not classify first-person account/access problems as assistant voice solely from "I cannot access".')
async def _():
    planned = 'Please help me log into my account by resetting my password.'
    rewrite = 'I cannot access my account; please reset my password.'
    e, s, ctx = replay(rewrite, planned=planned)
    equal(await adapt(e, ctx), rewrite)


for label, planned, rewrite in (
    ('here_is_answer', 'Please summarize the itinerary for my trip to Mexico City.',
     'Here is the summary you requested for your trip to Mexico City.'),
    ('curly_apostrophe', 'Please create a summary of the travel itinerary for my Mexico City trip.',
     'I\u2019m not able to create that summary, but here is some general travel advice.'),
    ('planned_voice_disables_guard', "I cannot access my account; please create a replacement for me.",
     "I've created the account you requested. Let me know if you need anything else."),
):
    voice_fix = {
        'here_is_answer': 'Recognize direct answer introductions such as "Here is the summary you requested".',
        'curly_apostrophe': 'Normalize curly apostrophes before applying the assistant-voice regex.',
        'planned_voice_disables_guard': 'Check newly introduced assistant answer cues even if planned text contains a user access complaint.',
    }[label]
    @case('replay.assistant_voice_' + label, 1419 if label == 'planned_voice_disables_guard' else 1318, voice_fix)
    async def _(planned=planned, rewrite=rewrite):
        e, s, ctx = replay(rewrite, planned=planned)
        await expect_fallback(e, ctx)


@case('replay.adaptation_exception_falls_back', 1412, 'Fail the episode on adaptation errors; never inject an unvalidated planned user.')
async def _():
    e, s, ctx = replay(RuntimeError('fake outage'))
    await expect_fallback(e, ctx)
    equal(e.counts.get('user_adapt_failed'), 1)


@case('replay.adaptation_disabled', 1340, 'Do not call the simulator when adaptation is disabled.')
async def _():
    e, s, ctx = replay(REWRITE, adapt_users=False)
    equal(await adapt(e, ctx), PLANNED)
    equal(len(s.calls), 0)


@case('replay.absent_optional_conversation', 1390,
      'Normalize conv=None to an empty/initial history at next_messages entry.')
async def _():
    e, s, ctx = replay(REWRITE)
    try:
        out = await e.next_messages(0, NO_FLIGHTS)
    except TypeError as exc:
        raise AssertionError('optional conv=None crashes: %s' % exc) from exc
    check(out and out[0]['role'] == 'user', 'no replayed user')


@case('replay.tool_turn_does_not_consume_user', 1367, 'Only consume planned users on assistant text turns.')
async def _():
    e, s, ctx = replay(FLIGHT)
    c = call('search_flights', {'to': 'MEX'})
    student = {'role': 'assistant', 'tool_calls': [c]}
    out = await e.next_messages(0, student, e.initial + [student])
    equal(out[0]['role'], 'tool')
    equal(e.k, 0)


@case('replay.completes_after_last_user', 1373, 'End when planned users are exhausted.')
async def _():
    e, s, ctx = replay(REWRITE)
    await adapt(e, ctx)
    equal(await e.next_messages(1, NO_FLIGHTS, ctx), None)
    equal(e.why, 'complete')


@case('replay.user_effort_forwarded', 1409, 'Forward role-specific user effort.')
async def _():
    e, s, ctx = replay(REWRITE, user_effort='medium')
    await adapt(e, ctx)
    equal(s.calls[0][2], {'effort': 'medium'})


# Single-turn cleanup, including markdown prefixes and role-slip integration.
for label, text, expected, split in (
    ('plain', 'Please help me.', 'Please help me.', 0),
    ('customer', 'Customer: Please help me.', 'Please help me.', 0),
    ('case_and_spaces', '  uSeR :  Please help me.', 'Please help me.', 0),
    ('bold_word', '**Customer**: Please help me.', 'Please help me.', 0),
    ('cut_agent', 'Customer: Please help me.\nAgent: Sure.\nCustomer: Thanks.', 'Please help me.', 1),
    ('cut_second_user', 'User: First request.\nUser: Second request.', 'First request.', 1),
    ('ordinary_colon', 'My note: please help me.', 'My note: please help me.', 0),
    ('empty', '', '', 0),
    ('bold_label', '**Customer:** Please help me.', 'Please help me.', 0),
):
    @case('one_turn.' + label, 641, 'Support markdown bold around the entire role label, including its colon.')
    def _(text=text, expected=expected, split=split):
        counts = {}
        equal(mt._one_turn(text, counts), expected)
        equal(counts.get('user_split', 0), split)


@case('one_turn.assistant_role_not_delivered_as_user', 642,
      'On a leading assistant/agent role, reject and resample in _user instead of merely counting the slip.')
async def _():
    e, s = env(replies=('Assistant: I can help you with that.', 'Customer: Please check my balance.'), strict_end=False)
    ctx = INITIAL + [{'role': 'assistant', 'content': 'How can I help?'}]
    out = await e.next_messages(0, ctx[-1], ctx)
    check(out and out[0]['content'] == 'Please check my balance.', 'assistant role delivered to user: %r' % out)


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--filter', default='', help='Run case names containing this substring')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--report', action='store_true', help='Write a Markdown report beside this script')
    options = parser.parse_args()
    selected = [c for c in CASES if options.filter in c[0]]
    check(bool(selected), 'filter selected no cases')
    check(len({c[0] for c in CASES}) == len(CASES), 'duplicate case name')
    before = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    check(before == SOURCE_SHA256, 'target source changed while loading; rerun the script')
    passed, failed, errors, findings = [], [], [], []
    for name, fn, line, fix in selected:
        try:
            result = fn()
            if inspect.isawaitable(result):
                await result
        except AssertionError as exc:
            failed.append(name)
            located = source_location(line)
            findings.append((name, located, str(exc), fix))
            print('FAIL %s | multiturn.py:%s | %s\n  fix: %s' % (name, located, exc, fix))
        except Exception:
            errors.append(name)
            print('TEST ERROR ' + name + '\n' + traceback.format_exc())
        else:
            passed.append(name)
            if options.verbose:
                print('PASS ' + name)
    check(hashlib.sha256(SOURCE.read_bytes()).hexdigest() == before, 'target source changed during test run')
    if not options.verbose:
        groups = collections.defaultdict(list)
        for name in passed:
            group, suffix = name.split('.', 1)
            groups[group].append(suffix)
        for group, names in groups.items():
            print('PASS %s (%d): %s' % (group, len(names), ', '.join(names)))
    print('\n%d cases: %d PASS, %d FAIL, %d TEST ERROR. Source SHA256: %s' %
          (len(selected), len(passed), len(failed), len(errors), before))
    if options.report:
        def cell(s):
            return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('|', '&#124;').replace('\n', '<br>')
        report = [
            '**%d cases: %d PASS, %d FAIL, %d test errors.**' % (len(selected), len(passed), len(failed), len(errors)),
            '', 'Run: `%s -B test_tool_v2_edge.py --report` (exit %d).' % (sys.executable, 2 if errors else 1 if failed else 0),
            '', 'Target: [%s](%s).' % (SOURCE.name, SOURCE),
            'SHA256: `%s`.' % before,
            '', 'The target changed externally during the review. These findings and source lines refer only to the final tested SHA above. '
            'The tests do not modify the target and disable bytecode writes.',
            '', 'Every failed assertion is listed below; related cases may share one underlying bug. Fixes are suggestions, not applied patches.',
            '', '| Failing case | Source line | Observed failure | Minimal fix |', '|---|---|---|---|',
        ]
        for name, line, detail, fix in findings:
            report.append('| `%s` | [%s](%s:%s) | %s | %s |' %
                          (cell(name), line, SOURCE, line, cell(detail), cell(fix)))
        report += ['', 'Passing cases (briefly):', '']
        groups = collections.defaultdict(list)
        for name in passed:
            group, suffix = name.split('.', 1)
            groups[group].append(suffix)
        for group, names in groups.items():
            report.append('- **%s (%d):** %s.' % (group, len(names), ', '.join('`%s`' % n for n in names)))
        if errors:
            report += ['', 'Test infrastructure errors: ' + ', '.join(errors) + '. See the run log for tracebacks.']
        Path(__file__).with_suffix('.report.md').write_text('\n'.join(report) + '\n')
    return 2 if errors else 1 if failed else 0


if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
