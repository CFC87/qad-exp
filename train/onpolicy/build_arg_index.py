"""The recorded calls of a list's conversations by tool (name + parameter schema), for the env's recorded argument
errors across conversations (multiturn.SimEnv arg_index; sim_kw "arg_index": OUT): a tool whose recorded backend
failed a call with an error naming an argument of a structure no recorded success gave it fails such a call in every
conversation that offers the same tool with the same schema (e.g. RGB fills accepted 10 times in an episode
whose own recording never failed one). Only tools with a recorded error and a recorded success are kept; arguments are
kept as their structure (multiturn._arg_shapes), errors with their text (first 1,200 characters).

  build_arg_index.py SUBSET_DIR[,SUBSET_DIR...] OUT.json [--only dataset/subset,...] [--train-only]

Several lists: their recordings are pooled before the filter (a key kept only when the pool has an error and a success;
a review: filtering each list first and merging the outputs loses one list's successes as counterevidence),
each source conversation once (a review: a decision-point cut "<parent>@<n>" and its full conversation are one
source; the row with the most tool calls is kept); --train-only skips each list's validation ranks (SUBSET.json
train_ranks). A tool defined twice with different parameters in one row is skipped there, and a key whose recordings
come with different tool descriptions (possibly another backend under the same name and parameters) is left out.
OUT.manifest.json records the sources (file hashes, ranks), the skipped duplicates, the excluded keys and, per key, the
(prompt_id, message index) of every call."""
import argparse
import collections
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import multiturn as MT  # noqa: E402
from to_chat import to_chat  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('subset_dir'); ap.add_argument('out')
    ap.add_argument('--only', default=None, help='comma list of dataset/subset or dataset')
    ap.add_argument('--train-only', action='store_true', help='skip the validation ranks of each list')
    a = ap.parse_args()
    only = set(a.only.split(',')) if a.only else None
    best, sources = {}, []
    for d in a.subset_dir.split(','):
        d = os.path.expanduser(d)
        hi = None
        if a.train_only:
            hi = json.load(open(os.path.join(d, 'SUBSET.json')))['train_ranks'][1]
        sources.append(dict(dir=d, train_hi=hi, sha256=hashlib.sha256(open(os.path.join(d, 'subset.parquet'), 'rb').read()).hexdigest()))
        for r in pq.read_table(os.path.join(d, 'subset.parquet'),
                               columns=['prompt_id', 'rank', 'dataset', 'subset', 'prompt', 'tools']).to_pylist():
            if hi is not None and r['rank'] >= hi:
                continue
            if only and r['dataset'] not in only and '%s/%s' % (r['dataset'], r['subset']) not in only:
                continue
            msgs, tools = to_chat(r['prompt'], r['tools'])
            n_calls = sum(len(m.get('tool_calls') or []) for m in msgs if m.get('role') == 'assistant')
            parent = str(r['prompt_id']).split('@')[0]
            if parent not in best or n_calls > best[parent][0]:
                best[parent] = (n_calls, r['prompt_id'], msgs, tools)
    idx, where, descs = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(set)
    rows, dup_names = 0, 0
    for parent, (_, pid, msgs, tools) in sorted(best.items()):
        specs, twice = {}, set()
        for x in tools or []:
            f = x.get('function') or x
            n = f.get('name')
            if n in specs and json.dumps(specs[n].get('parameters'), sort_keys=True) != json.dumps(f.get('parameters'), sort_keys=True):
                twice.add(n)
            specs[n] = f
        dup_names += bool(twice)
        rows += 1
        for i, m in enumerate(msgs):
            calls = m.get('tool_calls') or [] if m.get('role') == 'assistant' else []
            outs = [o for o in msgs[i + 1:i + 1 + len(calls)] if o.get('role') == 'tool']
            for c, o in zip(calls, outs):
                f = c.get('function') or c
                if f.get('name') not in specs or f.get('name') in twice:
                    continue
                out = o.get('content') or ''
                err = MT._is_error_v2(out)
                if err and MT._user_cited(out):  # the data generator's, not the backend's
                    continue
                key = MT._schema_key(f.get('name'), specs[f.get('name')])
                idx[key].append([MT._arg_shapes(MT._args_dict(f.get('arguments'))), err, out[:1200] if err else ''])
                where[key].append([pid, i])
                descs[key].add(hashlib.sha1(str(specs[f.get('name')].get('description') or '').encode()).hexdigest()[:12])
    ambiguous = sorted(k for k in idx if len(descs[k]) > 1)
    keep = {k: v for k, v in idx.items() if any(x[1] for x in v) and not all(x[1] for x in v) and k not in ambiguous}
    with open(os.path.expanduser(a.out) + '.manifest.json', 'w') as f:
        json.dump(dict(sources=sources, source_conversations=rows, rows_with_conflicting_duplicate_names=dup_names,
                       keys_left_out_for_differing_descriptions=[k for k in ambiguous if k in idx and any(x[1] for x in idx[k])
                                                                 and not all(x[1] for x in idx[k])],
                       calls={k: where[k] for k in keep}), f)
    with open(os.path.expanduser(a.out), 'w') as f:
        json.dump(keep, f, ensure_ascii=False)
    print('conversations %d, tools %d, kept (an error and a success) %d with %d calls'
          % (rows, len(idx), len(keep), sum(len(v) for v in keep.values())))


if __name__ == '__main__':
    main()
