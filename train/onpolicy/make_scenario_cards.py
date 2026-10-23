"""Offline scenario cards for multi-turn `sim` episodes (multiturn.py): one NeMo-Gym style customer scenario per stored
conversation, written by an API model (default gpt-oss-120b) through local rate-limiting proxies (sim_throttle.py).

    python make_scenario_cards.py SUBSET_DIR OUT.parquet \\
        --only SFT-Agentic-v2/interactive_agent,Agentic-v1/interactive_agent,RL-Agentic-Conversational-Tool-Use-Pivot-v1 \\
        --urls http://127.0.0.1:8290,http://127.0.0.1:8291 [--limit N]

Check: every concrete fact the user gives in the stored conversation (e-mail addresses, and tokens containing a digit:
IDs, order / account / phone numbers, codes) must appear in the card (compared without case, spaces and punctuation);
a card missing some, or whose task instructions describe the representative's job (role slip), is rewritten
once with the problems named; `ok` = nothing missing and no role slip. Resumable: prompt ids
already in OUT are skipped. rollout.py uses cards with ok (SCENARIO_CARDS); the others fall back to an online card.
"""
import argparse, asyncio, json, os, re, sys, time
import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiturn as MT  # noqa: E402
from to_chat import to_chat  # noqa: E402

EMAIL = re.compile(r'[\w.+-]+@[\w-]+(?:\.[\w-]+)+')
DIGITY = re.compile(r'[#(]?[A-Za-z0-9][A-Za-z0-9_\-#./()]*\d[A-Za-z0-9_\-#./()]*')
norm = lambda x: re.sub(r'[^a-z0-9]', '', x.lower())


ORDINAL_OR_SPAN = re.compile(r'^(\d+(st|nd|rd|th)|\d+([-.]\d+)*-?(day|days|hour|hours|minute|minutes|week|weeks|month|months|year|years|y|m|d|h|k|K|gb|GB|mb|MB|%)s?|[\d,.]+)$')


def user_facts(msgs):
    """concrete identifying facts the customer gives: e-mail addresses and ID-like tokens (letters and digits, or long
    numbers such as phone / account / order numbers). Not facts (false alarms): ordinals and dates (25th),
    bare short numbers (100, 000), durations / sizes (14-day, 9.6GB), and long base64 blobs (attachments)."""
    facts = set()
    for m in msgs:
        if m.get('role') != 'user':
            continue
        c = m.get('content') or ''
        facts.update(EMAIL.findall(c))
        for t in DIGITY.findall(EMAIL.sub(' ', c)):
            t = t.strip('.,;:()#')
            n = norm(t)
            if len(n) < 3 or n.isalpha() or len(t) > 40:
                continue
            if ORDINAL_OR_SPAN.match(t) and not (t.replace(',', '').replace('.', '').isdigit() and len(n) >= 6):
                continue  # keep long pure numbers (phone, account, order numbers); drop short numbers and spans
            facts.add(t)
    return sorted(facts)


def user_first_facts(msgs):
    """--kind user: the user's own facts, i.e. user_facts() minus the values an assistant message or a tool result had
    before the user said them (those the recorded assistant gave: a card must not know them in advance). Compared as
    whole tokens (a review: the user's own AB123 was dropped because a tool had returned AB12345)"""
    out, said = [], set()
    for m in msgs:
        if m.get('role') == 'user':
            for f in user_facts([m]):
                if norm(f) not in said and f not in out:
                    out.append(f)
        if m.get('role') in ('assistant', 'tool'):
            txt = (m.get('content') or '') + ' ' + json.dumps(m.get('tool_calls') or [], ensure_ascii=False)
            said |= {norm(t) for t in re.findall(r'[\w@.+-]+', txt)} | {norm(t) for t in user_facts([{'role': 'user', 'content': txt}])}
    return sorted(out)


def assistant_values(msgs):
    """--kind user: specific values that only the assistant's messages or the tool results contain (not the system prompt
    or a user message before them): IDs and values with digits, names (multi-word or mixed case) from tool results, and
    quoted multi-word titles in the assistant's text; not JSON keys or plain words. A card that names one presents a
    recorded answer as known (e.g. a card's "player ID 1629629" from a recorded lookup)"""
    vals, user_said = set(), ''
    for m in msgs:
        c = m.get('content') or ''
        if m.get('role') in ('system', 'user'):
            user_said += ' ' + c
            continue
        if m.get('role') == 'tool':
            cand = MT._result_values(c)
        elif m.get('role') == 'assistant':
            cand = set(user_facts([{'role': 'user', 'content': c}])) | {
                q for q in re.findall(r'"([^"\n]{8,80})"', c) if len(q.split()) >= 2 and re.search(r'[A-Z0-9]', q)}
        else:
            continue
        for v in cand:
            n = norm(v)
            if len(n) >= 5 and not re.fullmatch(r'[A-Za-z][a-z]+', v.strip()) and n not in norm(user_said):
                vals.add(v)
        # numbers the assistant or a tool gave: prices, amounts, computed answers (two digits or more, or decimals;
        # a review: a recorded rate 49.99 and an answer 42 passed the leak check); a number any user message or
        # the system prompt has is the user's (first stricter cards: "$20-30", "12 guests" counted as leaks because the
        # assistant had said 20 / 12 earlier)
        # numbers compared normalized (100.0 = 100); a short integer any user message has is the user's (coincidences
        # such as "$20-30" vs "20-25 minutes"), a decimal or longer number only if the user had it before (e.g.
        # "Book the 49.99 rate" after the assistant found 49.99)
        user_nums = _all_nums(' '.join(x.get('content') or '' for x in msgs if x.get('role') in ('system', 'user')))
        before_nums = _all_nums(user_said)
        if m.get('role') == 'tool' and MT._parsed(c) is not None:
            # values of the parsed result only, not its keys (a review: {"10": {...}} made "10 days" a leak)
            nums = []
            def walk(x):
                if isinstance(x, dict):
                    for v in x.values():
                        walk(v)
                elif isinstance(x, list):
                    for v in x:
                        walk(v)
                elif isinstance(x, (int, float)) and not isinstance(x, bool):
                    nums.append(('%s' % x).rstrip('0').rstrip('.') if isinstance(x, float) else str(x))
                elif isinstance(x, str):
                    nums.extend(re.findall(NUM, x))
            walk(MT._parsed(c))
        else:
            nums = re.findall(NUM, c)
        for num in map(_numnorm, nums):
            # a calendar year is the user's own knowledge, not an answer (a card: "first big holiday of 2025"
            # became "of (the one the assistant gives)")
            if re.fullmatch(r'(?:19|20)\d\d', num):
                continue
            if len(num.replace('.', '')) >= 2 or '.' in num:
                long_ = '.' in num or len(num) >= 3
                if num not in (before_nums if long_ else user_nums):
                    vals.add('#' + num)
    return vals


# a number with its thousands separators ("25,000" is one number, not 25 and 000: a card said "above
# 25,(the one the assistant gives) BTC" and the simulated user asked for 25 BTC)
NUM = r'(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?![\w])(?!,\d)'


def _all_nums(text):
    """every number of a text, also inside words ("20GB"), thousands separators joined"""
    t = text or ''
    return {_numnorm(n) for n in re.findall(r'\d+(?:\.\d+)?', t)} | \
        {_numnorm(n) for n in re.findall(r'\d{1,3}(?:,\d{3})+(?:\.\d+)?', t)}


def _numnorm(n):
    """a number as text without thousands separators and trailing decimal zeros: 25,000 -> 25000, 100.0 -> 100"""
    n = str(n).replace(',', '')
    return n.rstrip('0').rstrip('.') if '.' in n else n


def _value_re(v):
    """a recorded value as a pattern over the card: its letters and digits in order with any spaces or punctuation
    between them (as norm() compares), the whole token only, not inside a longer word, ID or file name"""
    parts = re.findall(r'[A-Za-z0-9]+', v)
    return re.compile(r'(?<![A-Za-z0-9_])(?<![A-Za-z0-9_][-./:])' + r'[^A-Za-z0-9]*'.join(map(re.escape, parts))
                      + r'(?![A-Za-z0-9_])(?![-./:][A-Za-z0-9_])', re.I)


def _list_marker(card, m):
    """a match that is the "1." / "2)" marker of a step line"""
    line = card[card.rfind('\n', 0, m.start()) + 1:m.start()]
    return len(m[0]) <= 2 and not line.strip() and bool(re.match(r'[.)]\s', card[m.end():m.end() + 2]))


def leak_spans(card, msgs, vals=None):
    """{value: [(start, end), ...]}: where the card holds an assistant-only value. A value with digits counts in any
    case or spelling; a value of words only when the card writes it as the recording does (exact, or a name written in
    capitals, the first letter in either case: "McLaren Mercedes", "the Home Depot"), not a common phrase in lower case ("private room", "in stock", "code examples" as
    the user's own words; in a first set of training cards 2/3 of the failed cards were such phrases or values inside a longer token).
    Numbers '#n' as whole numbers, not inside a date, range, ID or file name, list markers "1." of the card's steps
    not counted."""
    out = {}
    for v in (vals if vals is not None else assistant_values(msgs)):
        if v.startswith('#'):
            spans = [m.span() for m in re.finditer(r'(?<![A-Za-z0-9_][-./:])' + NUM + r'(?![-./:][A-Za-z0-9_])', card)
                     if _numnorm(m[0]) == v[1:] and not _list_marker(card, m)]
            if spans:
                out[v[1:]] = spans
            continue
        if len(norm(v)) < 4:
            continue
        digits = re.search(r'\d', v)
        caps = len(re.findall(r'[A-Z]', v)) >= 2
        letters = re.sub(r'[^A-Za-z0-9]', '', v)
        spans = [m.span() for m in _value_re(v).finditer(card)
                 if digits or m[0] == v or caps and re.sub(r'[^A-Za-z0-9]', '', m[0])[1:] == letters[1:]]
        if spans:
            out[v] = spans
    return out


def leaks(card, msgs, vals=None):
    """the assistant-only values a user card contains (see leak_spans)"""
    return sorted(leak_spans(card, msgs, vals))


def redact(card, msgs, vals=None, who='assistant'):
    """the card with every leaked value replaced by a relative reference, at the places leak_spans found"""
    spans = sorted(sp for v in leak_spans(card, msgs, vals).values() for sp in v)
    out, end = [], 0
    for a, b in spans:
        if a >= end:
            out.append(card[end:a] + '(the one the %s gives)' % who)
            end = b
    return ''.join(out) + card[end:]


def invented(card, msgs):
    """ID-like values, e-mails and key-like strings of a card that the recorded conversation never contains (e.g.
    a card adding "Your API key is KEY-919191" was ok)"""
    allt = norm(json.dumps(msgs, ensure_ascii=False))
    toks = set(user_facts([{'role': 'user', 'content': card}])) | set(EMAIL.findall(card))
    return sorted(t for t in toks if norm(t) and norm(t) not in allt)


CARD_REVIEW = """Below are the USER messages of a recorded conversation (in order) and two parts of a scenario card written
for a simulator that will play this user: "Your details" and "Task instructions". Check them against the user messages:
- "missing": a request of the user messages that no task step covers at all, or a concrete detail the user gives (a
  name, ID, number, date, key) that neither part contains. Paraphrases, merged steps and steps described relative to
  the assistant's answers are fine; do not list wording differences.
- "invented": a concrete fact, value or request in the two parts that no user message or the system prompt states.
  Instructions on how to react (answer when asked, say you do not have a detail) are not facts.
- The values under <assistant values> came from the recorded assistant's answers, not from the user: a card must refer
  to them relatively ("the cheapest rate the assistant found"), so never list them as missing. In the card,
  "(the one the assistant gives)" stands for such a value: it is neither missing nor invented.
Return JSON only: {{"missing": ["..."], "invented": ["..."]}} with empty lists when there is none.

<assistant values>
{avals}
</assistant values>

<system prompt>
{system}
</system prompt>
<user messages>
{users}
</user messages>
<card>
{card}
</card>"""


# --kind customer --rules v3 (mixed-training env): the customer card without the recorded representative's
# answers. A two-reader audit of 100 mixed customer cards (v2 rules) found 39-66 % with a material leak: ticket and case
# IDs before the request was filed, computed fees, looked-up availability picked in advance, rebooked flights, outcomes;
# v2 asked for "every concrete fact the customer knows and gives at some point", which includes the answers the customer
# repeated. only the customer's own facts, later requests relative to the answers, and the user cards' checks (leaks,
# invented values, redaction, fidelity review).
SCENARIO_FROM_CONV_V3 = MT.SCENARIO_FROM_CONV.replace(
    '- "customer_details": every concrete fact the customer knows and gives at some point (names, IDs, account numbers, '
    'order numbers, codes, dates, amounts, preferences), one per line, exactly as written in the conversation.',
    '- "customer_details": every concrete fact the customer has before the representative tells them anything (their '
    'names, their own IDs, account and order numbers, codes they already hold, dates, amounts, preferences), one per '
    'line, exactly as the customer writes it. Never a value that only the representative or a tool gave (a record or ID '
    'it looked up or created, a computed amount or fee, an availability, an outcome), even if the customer repeats it '
    'later.').replace(
    "Do not mention the assistant's specific replies or tool results. Output only the JSON object.",
    "Do not mention the assistant's specific replies or tool results. A request that builds on an answer is written "
    'relative to it ("when the representative tells you the fee, ask to pay it"), never with the answer\'s value, and '
    'no step states an outcome before it happens. Output only the JSON object.')
assert SCENARIO_FROM_CONV_V3.count('never with the answer') == 1 and SCENARIO_FROM_CONV_V3.count('before the representative tells') == 1, 'v3 prompt'
CARD_REVIEW_CUSTOMER = CARD_REVIEW.replace('USER messages', 'CUSTOMER messages').replace('this user', 'this customer').replace(
    '"Your details"', '"Customer details"').replace('the user messages', 'the customer messages').replace(
    'the user gives', 'the customer gives').replace("the user's", "the customer's").replace('no user message', 'no customer message').replace(
    "recorded assistant's answers, not from the user", "recorded representative's answers, not from the customer").replace(
    '<user messages>', '<customer messages>').replace('</user messages>', '</customer messages>')


ROLE_SLIP = re.compile(r'\b(ask|tell|inform|verify|authenticate|confirm with|help|assist|offer|remind|thank|greet|apologi[sz]e to|'
                       r'transfer|escalate) the (customer|user|caller)\b(?!\s*(service|id|ids|relations|support|account|portal|line|team))'
                       r'|\b(use|call|trigger|invoke|run) the [\w -]{0,30}\b(tool|function)\b', re.I)


def role_slips(c):
    """task_instructions that describe the representative's job instead of the customer's (a common failure); nouns such
    as 'customer service', 'user ID' are not slips (all 8 inspected flags of the old rule were such nouns)"""
    return [m.group(0).strip() for m in ROLE_SLIP.finditer(c.get('task_instructions') or '')]


# --self-service (knowledge_agent rows, τ²): the representative grants the customer self-service actions
# (give_discoverable_user_tool) that the customer completes and reports; the pre-check cards named them by tool name
# (25 / 98: "execute the `confirm_..._approval` action ... tell the representative it came back successful"), so the
# simulated customer asked for the tool by name or claimed it done without access. The rule asks for plain words, and a
# card naming any tool of the conversation is rewritten once (then not ok).
SELF_SERVICE_CARD = """In this recording the representative gives the customer access to self-service actions that the \
customer then completes on their own side (confirming, approving, verifying something) and reports back. In the card, \
describe each such step in plain words (for example "confirm on your side that you approve the repair work"), never by \
a tool or function name or any identifier with underscores, and never tell the customer to ask for a tool by name. \
Write such a step as: after the representative gives you access to <the step in words>, do it and report <the result \
the recording shows>."""


def tool_names(msgs, tools):
    """tool names of the conversation: the offered tools, the called ones and the discoverable tools named in arguments"""
    out = {str((t.get('function') or t).get('name') or '') for t in tools or []}
    for m in msgs:
        for tc in m.get('tool_calls') or []:
            f = tc.get('function') or tc
            out.add(str(f.get('name') or ''))
            a = f.get('arguments')
            try:
                a = json.loads(a) if isinstance(a, str) else (a or {})
            except ValueError:
                a = {}
            for k in ('discoverable_tool_name', 'agent_tool_name', 'tool_name'):
                if isinstance(a, dict) and a.get(k):
                    out.add(str(a[k]))
    return {n for n in out if '_' in n}


def named_tools(card, names):
    low = card.lower()
    return sorted(n for n in names if re.search(r'(?<![a-z0-9_])%s(?![a-z0-9_])' % re.escape(n.lower()), low))


def parse_card(txt):
    txt = re.sub(r'^```(?:json)?\s*|\s*```$', '', (txt or '').strip())
    c = json.loads(txt[txt.index('{'):txt.rindex('}') + 1])
    return {k: (v if isinstance(v, str) else '\n'.join(map(str, v)) if isinstance(v, list) else json.dumps(v))
            for k, v in c.items()}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset_dir'); ap.add_argument('out')
    ap.add_argument('--only', required=True, help='comma list of dataset/subset or dataset')
    ap.add_argument('--urls', default='http://127.0.0.1:8290,http://127.0.0.1:8291')
    ap.add_argument('--model', default='nvcf/openai/gpt-oss-120b')
    ap.add_argument('--effort', default='medium')
    ap.add_argument('--concurrency', type=int, default=8)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--ranks', default=None, help='lo,hi: only rows with lo <= rank < hi, written in rank order')
    ap.add_argument('--retry-failed', action='store_true', help='rewrite the cards in OUT that are not ok (new sample)')
    ap.add_argument('--self-service', action='store_true', help='SELF_SERVICE_CARD rule + no tool names in the card (knowledge_agent)')
    ap.add_argument('--kind', default='customer', choices=['customer', 'user'],
                    help="user: a person using a general AI assistant (function-calling sets, SCENARIO_FROM_CONV_USER): "
                         "only the user's own facts are required and assistant-only values are leaks")
    ap.add_argument('--ids', default=None, help='file with one prompt_id per line: only these rows')
    ap.add_argument('--rules', default='v2', choices=['v2', 'v3'],
                    help="v3 (--kind customer): SCENARIO_FROM_CONV_V3 and the user cards' checks (own facts only, leaks, "
                         "invented values, redaction, fidelity review); v2: as before")
    a = ap.parse_args()
    only = set(a.only.split(','))
    t = pq.read_table(os.path.join(os.path.expanduser(a.subset_dir), 'subset.parquet'),
                      columns=['prompt_id', 'dataset', 'subset', 'prompt', 'tools', 'rank']).to_pylist()
    rows = [r for r in t if r['dataset'] in only or '%s/%s' % (r['dataset'], r['subset']) in only]
    if a.ranks:
        lo, hi = map(int, a.ranks.split(','))
        rows = [r for r in rows if lo <= r['rank'] < hi]
    if a.ids:
        keep = {l.strip() for l in open(a.ids) if l.strip()}
        rows = [r for r in rows if r['prompt_id'] in keep]
    rows.sort(key=lambda r: r['rank'])  # the run draws prompts in rank order: write the earliest first
    done = {}
    other = {}  # rows of another card kind: kept in the file unless rewritten now, never reused (a review)
    if os.path.exists(a.out):
        for r in pq.read_table(a.out).to_pylist():
            if (r.get('kind') or 'customer') != a.kind:
                other[r['prompt_id']] = r
                continue
            done[r['prompt_id']] = r
    if a.retry_failed:  # a fresh attempt for the failed ones (and other-kind ones); a new ok card replaces the old entry
        bad = {k for k, v in done.items() if not v['ok']} | set(other)
        todo = [r for r in rows if r['prompt_id'] in bad]
    else:
        todo = [r for r in rows if r['prompt_id'] not in done]
    if a.limit:
        todo = todo[:a.limit]
    if a.retry_failed:  # only the rows of this call (--ranks / --ids / --limit): failed entries outside stay in the file
        for r in todo:
            done.pop(r['prompt_id'], None)
    print('%d conversations selected, %d done, %d to write' % (len(rows), len(done), len(todo)), flush=True)
    urls = a.urls.split(',')
    sem = asyncio.Semaphore(a.concurrency)
    new, t0 = [], time.time()

    async def ask(s, url, messages):
        body = dict(model=a.model, messages=messages, max_tokens=4096, temperature=0.3, reasoning_effort=a.effort)
        for attempt in range(3):
            try:
                async with s.post(url + '/v1/chat/completions', json=body, timeout=aiohttp.ClientTimeout(total=3600)) as r:
                    j = await r.json()
                return j['choices'][0]['message']['content'] or ''
            except Exception:
                if attempt == 2:
                    raise
                await asyncio.sleep(10)

    async def one(s, i, r):
        async with sem:
            msgs, tools = to_chat(r['prompt'], r['tools'])
            E = MT.SimEnv(msgs, tools, None)
            user = a.kind == 'user'
            strict = user or a.rules == 'v3'  # the user cards' checks
            facts = user_first_facts(msgs) if strict else user_facts(msgs)
            avals = assistant_values(msgs) if strict else set()
            prompt = (MT.SCENARIO_FROM_CONV_USER if user else SCENARIO_FROM_CONV_V3 if strict
                      else MT.SCENARIO_FROM_CONV).format(conversation=E.ref_card)
            names = tool_names(msgs, tools) if a.self_service else set()
            if a.self_service:
                prompt = prompt.replace('\n\n<conversation>\n', '\n\n' + SELF_SERVICE_CARD + '\n\n<conversation>\n', 1)
            async def review_card(parts):
                txt_r = await ask(s, urls[i % len(urls)], [{'role': 'user', 'content': (CARD_REVIEW if user else CARD_REVIEW_CUSTOMER).format(
                    avals='\n'.join(sorted(v.lstrip('#') for v in avals))[:4000] or '(none)',
                    system=MT._clip(E.system, 6000), card=parts, users=json.dumps(
                        [m.get('content') or '' for m in msgs if m.get('role') == 'user'], ensure_ascii=False))}])
                # a verdict without both lists is no clean review
                rv = MT._parsed(re.sub(r'^```(?:json)?\s*|\s*```$', '', (txt_r or '').strip()))
                if not (isinstance(rv, dict) and isinstance(rv.get('missing'), list) and isinstance(rv.get('invented'), list)
                        and all(isinstance(x, str) for x in rv['missing'] + rv['invented'])):
                    return {'missing': ['(the review did not return both lists)']}
                return {k: [str(x) for x in rv[k] if x] for k in ('missing', 'invented') if any(rv[k])}

            messages = [{'role': 'user', 'content': prompt}]
            card, missing, err, named, leaked, review = None, facts, None, [], [], {}
            for attempt in range(3 if strict else 2):
                err = None
                try:
                    txt = await ask(s, urls[i % len(urls)], messages)
                    c = parse_card(txt)
                    card = (MT.scenario_string_user if user else MT.scenario_string)(c)
                    nc = norm(card)
                    missing = [f for f in facts if norm(f) not in nc]
                    slips = role_slips(c)
                    named = named_tools(card, names)
                    leaked = leaks(card, msgs, avals) if strict else []
                    made_up = invented(card, msgs) if strict else []
                    if strict and not c.get('task_instructions'):
                        slips = slips + ['(no task_instructions)']
                    review = {}
                    if strict and not missing and not slips and not leaked and not made_up:
                        review = await review_card('%s:\n%s\n\nTask instructions:\n%s' % (
                            'Your details' if user else 'Customer details',
                            c.get('user_details' if user else 'customer_details') or '', c.get('task_instructions') or ''))
                    if not missing and not slips and not named and not leaked and not made_up and not review:
                        break
                    fix = []
                    if made_up:
                        fix.append('it contains values the conversation never has (%s); use only what the %s said'
                                   % (', '.join(made_up[:8]), 'user' if user else 'customer'))
                    if review.get('missing'):
                        fix.append('its task instructions leave out these requests or details of the user: %s'
                                   % '; '.join(map(str, review['missing'][:8])))
                    if review.get('invented'):
                        fix.append('it adds facts or requests the user never stated: %s; remove them'
                                   % '; '.join(map(str, review['invented'][:8])))
                    if leaked:
                        fix.append('it contains values that only the %s gave in the recording (%s); the %s does '
                                   'not know them in advance: describe those requests relative to the %s\'s answer '
                                   'instead' % ('assistant' if user else 'representative', ', '.join(leaked[:8]),
                                                'user' if user else 'customer', 'assistant' if user else 'representative'))
                    if named:
                        fix.append('it names tools or actions by their identifiers (%s); describe every step in plain words, '
                                   'without any identifier' % ', '.join(named[:5]))
                    if missing:
                        fix.append('it leaves out facts the %s gives in the conversation: %s; put every one of them in '
                                   '"%s" exactly as written' % ('user' if user else 'customer', ', '.join(missing),
                                                                'user_details' if user else 'customer_details'))
                    if slips:
                        fix.append('"task_instructions" must be addressed to the customer in the second person and describe only '
                                   'what the customer does; it now contains %s' % '; '.join(repr(x) for x in slips[:5]))
                    messages = messages[:1] + [{'role': 'assistant', 'content': txt},
                                               {'role': 'user', 'content': 'Fix the card: ' + '; and '.join(fix) + '. Return the whole JSON object.'}]
                except Exception as e:
                    err = repr(e)[:200]
            slips = (role_slips(c) + (['(no task_instructions)'] if strict and not c.get('task_instructions') else [])) if card else []
            named = named_tools(card, names) if card else []
            leaked = leaks(card, msgs, avals) if card and strict else []
            redacted = []
            if strict and card and leaked and not missing and not slips and not named and err is None \
                    and not (invented(card, msgs)) and not review:
                # only leaks left after the attempts: the leaked values become a relative reference, so the row keeps the
                # card-driven user instead of falling back to the replayed one (in one set 61 of 250 cards failed on leaks alone)
                card = redact(card, msgs, avals, who='assistant' if user else 'representative')
                redacted, leaked = leaked, leaks(card, msgs, avals)
                # the redacted card is reviewed like any other (41 cards of one set were redacted and ok without
                # the fidelity review, some losing the user's own values, e.g. a candidate's listed skills)
                try:
                    review = await review_card(card)
                except Exception as e:
                    err = repr(e)[:200]
                # (an earlier filter dropped review "missing" items naming a replaced name or id; withdrawn: a whole item was dropped
                # when it merely contained such a name, also with a separate lost requirement in it,
                # and the filter never took effect (it overwrote `named`, so its cards stayed refused: 46 cards in one set))
            made_up = invented(card, msgs) if card and strict else []
            fidelity = review if strict else {}
            new.append(dict(prompt_id=r['prompt_id'], dataset=r['dataset'], subset=r['subset'], card=card or '',
                            ok=bool(card) and not missing and not slips and not named and not leaked and not made_up
                            and not fidelity and err is None if strict else bool(card) and not missing and not slips and not named,
                            n_facts=len(facts), missing=json.dumps(missing), role_slips=json.dumps(slips), error=err,
                            named_tools=json.dumps(named), kind=a.kind, rules=a.rules, leaks=json.dumps(leaked), redacted=json.dumps(redacted),
                            invented=json.dumps(made_up), review=json.dumps(fidelity, ensure_ascii=False)))
            if len(new) % 100 == 0:
                ok = sum(x['ok'] for x in new)
                print('%d / %d written (%.0f s), ok %d (%.1f%%)' % (len(new), len(todo), time.time() - t0, ok, 100 * ok / len(new)), flush=True)
            if len(new) % 500 == 0:
                save()

    def save():
        written = {x['prompt_id'] for x in new}
        allr = [x for x in other.values() if x['prompt_id'] not in written and x['prompt_id'] not in done] + \
            [x for x in done.values() if x['prompt_id'] not in written] + new
        keys = set().union(*(r.keys() for r in allr)) if allr else set()
        full = []
        for r in allr:  # every row with every field, a legacy row's kind = customer
            d = {k: None for k in keys}; d.update(r); d['kind'] = r.get('kind') or 'customer'; full.append(d)
        allr = full
        pq.write_table(pa.Table.from_pylist(allr), a.out + '.tmp'); os.replace(a.out + '.tmp', a.out)

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0)) as s:
        await asyncio.gather(*[one(s, i, r) for i, r in enumerate(todo)])
    save()
    ok = sum(x['ok'] for x in new)
    print('done: %d cards, ok %d (%.1f%%), %.0f s' % (len(new), ok, 100 * ok / max(1, len(new)), time.time() - t0), flush=True)

asyncio.run(main())
