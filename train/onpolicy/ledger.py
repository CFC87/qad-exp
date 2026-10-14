"""The controller's logs as the record of one training lineage (resume and WSD branches).

rollout.py appends to <log_dir>/rollouts.jsonl (one line per rollout), steps.jsonl (one line per step and route) and
episodes.jsonl (multi-turn conversations). A resume restarts the controller at an earlier step, so a log that went
through a resume holds some steps twice: the lines of the attempt that crashed and those of the attempt that trained.
Every line names its controller start ('run'), and each (route, step) keeps the lines of its latest start only.
Lines without 'run' (older logs): a new attempt starts where a training step is written again (a step number below the
previous one, or the same step in a new block; a resume right after a step was written is not seen -- the reason for
'run').

  used_ranks(log_dir, start)       the prompts the lineage consumed before step `start`: per step, steps.jsonl
                                   'drawn_ranks' (drawn from the source and not given back); steps logged without
                                   it: the ranks with a rollout in the step minus 'requeued_ranks' (given back)
  carried_totals(log_dir, start)   the 'cumulative' totals after step start - 1
  branch(parent, child, start, val_every)
                                   the parent's lines before `start` (training steps < start, validations
                                   e < start // val_every: the trainer evaluated those before its checkpoint at start)
                                   into a new log dir, for a run that continues the parent's checkpoint at `start`

    python ledger.py branch PARENT_LOG_DIR CHILD_LOG_DIR START VAL_EVERY
"""
import json, os, sys

FILES = ('rollouts.jsonl', 'steps.jsonl', 'episodes.jsonl')


def _route(r):
    return r.get('route', 'pack')


def read(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def latest_attempt(lines):
    """The lines of each (route, step)'s latest attempt, in file order."""
    att, cur, last_step, prev_pack_step, prev_run = [], 0, None, None, None
    for r in lines:
        run = r.get('run')
        if run is not None:  # the controller start the line names
            if run != prev_run and (prev_run is not None or att):
                cur += 1
            prev_run, last_step, prev_pack_step = run, None, None
        elif _route(r) == 'pack':
            s = r['step']
            if last_step is not None and (s < last_step or (s == last_step and prev_pack_step != s)):
                cur += 1
            last_step, prev_pack_step = s, s
        else:
            prev_pack_step = None  # a pack block after other lines is a new block
        att.append(cur)
    best = {}
    for r, a in zip(lines, att):
        k = (_route(r), r['step'])
        best[k] = max(best.get(k, -1), a)
    return [r for r, a in zip(lines, att) if best[(_route(r), r['step'])] == a]


def before(lines, start, val_every):
    """Training steps < start; validations e < start // val_every."""
    n_val = start // val_every if val_every else 0
    return [r for r in lines if (r['step'] < start if _route(r) == 'pack' else r['step'] < n_val)]


def used_ranks(log_dir, start):
    roll = [r for r in latest_attempt(read(os.path.join(log_dir, 'rollouts.jsonl'))) if _route(r) == 'pack' and r['step'] < start]
    steps = {r['step']: r for r in latest_attempt(read(os.path.join(log_dir, 'steps.jsonl'))) if _route(r) == 'pack'}
    by_step = {}
    for r in roll:
        if r.get('rank') is not None:
            by_step.setdefault(r['step'], []).append(r['rank'])
    used = set()
    for s in sorted(set(by_step) | {s for s in steps if s < start}):
        st = steps.get(s, {})
        if st.get('drawn_ranks') is not None:
            used |= set(st['drawn_ranks'])
        else:
            used |= set(by_step.get(s, ()))
            used -= set(st.get('requeued_ranks') or ())
    return used


def carried_totals(log_dir, start):
    for r in latest_attempt(read(os.path.join(log_dir, 'steps.jsonl'))):
        if _route(r) == 'pack' and r['step'] == start - 1 and r.get('cumulative'):
            return r['cumulative']
    return None


def branch(parent, child, start, val_every, force=False):
    assert start > 0, 'a branch starts after step 0'
    steps = [r for r in latest_attempt(read(os.path.join(parent, 'steps.jsonl'))) if _route(r) == 'pack']
    have = {r['step'] for r in steps}
    missing = [s for s in range(start) if s not in have]
    if missing:
        raise SystemExit('%s: training steps %s are not in steps.jsonl (the parent has not reached step %d)' % (
            parent, missing[:10], start))
    os.makedirs(child, exist_ok=True)
    clash = [f for f in FILES if os.path.exists(os.path.join(child, f))]
    if clash and not force:
        raise SystemExit('%s already has %s: not overwritten' % (child, ', '.join(clash)))
    counts = {}
    for f in FILES:
        src = read(os.path.join(parent, f))
        keep = before(latest_attempt(src), start, val_every)
        with open(os.path.join(child, f), 'w') as out:
            out.writelines(json.dumps(r) + '\n' for r in keep)
        counts[f] = dict(parent_lines=len(src), kept=len(keep))
    info = dict(parent=os.path.abspath(parent), start_step=start, val_every=val_every, files=counts,
                used_prompts=len(used_ranks(child, start)), cumulative=carried_totals(child, start))
    json.dump(info, open(os.path.join(child, 'BRANCH.json'), 'w'), indent=1)
    return info


if __name__ == '__main__':
    if len(sys.argv) >= 6 and sys.argv[1] == 'branch':
        info = branch(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), force='--force' in sys.argv)
        print('ledger branch at step %d: %s; %d prompts used before it' % (
            info['start_step'], ', '.join('%s %d of %d lines' % (f, c['kept'], c['parent_lines']) for f, c in info['files'].items()),
            info['used_prompts']))
    else:
        print(__doc__)
        sys.exit(2)
