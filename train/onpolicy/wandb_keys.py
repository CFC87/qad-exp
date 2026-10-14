"""What goes to wandb, under plain descriptive names; the section is the part before the first '/'.

  key/                  ~20 curated metrics for a one-glance view of the run (copies of the ones below)
  kl_train/, kl_val/    KL, top-1 agreement, log-prob gap, NLLs, tail shares over all loss tokens
  kl_by_dataset/<dataset>, loss_tokens_by_dataset/, top1_by_dataset/          (validation: val_kl_by_dataset/)
  kl_by_position/<0-1k|1k-4k|4k-16k|16k+>, kl_by_segment/<thinking|answer|history>, loss_tokens_by_position/, ..._by_segment/
  history_kl_by_dataset/<dataset>, history_loss_tokens_by_dataset/   (history supervision; val: val_history_...)
  rollout/, val_rollout/    overall rollout statistics
  response_len_by_dataset/, truncated_share_by_dataset/, count_by_dataset/     rollout per dataset
  time/, cumulative/
Everything, including the fields left out here, still goes to metrics.jsonl on the last stage (original names).
rollout/sampling_time_logprob is vLLM's log p of each sampled token at sampling time: biased high (the token was
picked partly for that computation's noise), so never compare it with the trainer's
kl_*/student_nll, which recompute the same tokens.
"""

KD_FIELD = {'kl': 'kl', 'top1': 'top1_agreement', 'logratio': 'logprob_gap_teacher_minus_student',
            'nll_s': 'student_nll', 'nll_t': 'teacher_nll', 'kl_gt0.1': 'share_tokens_kl_over_0.1',
            'kl_gt1': 'share_tokens_kl_over_1', 'kl_max': 'max_token_kl', 'tokens': 'loss_tokens'}
POS = {'0-1k': '0-1k', '1-4k': '1k-4k', '4-16k': '4k-16k', '16k+': '16k+'}
SEG = {'think': 'thinking', 'answer': 'answer', 'history': 'history'}
ROLLOUT = {'finished': 'finished', 'submitted': 'submitted', 'truncated': 'truncated_at_32k',
           'gen_tokens': 'generated_tokens', 'prompt_tokens': 'prompt_tokens', 'packed_tokens': 'packed_tokens',
           'loss_tokens': 'loss_tokens', 'hist_loss_tokens': 'history_loss_tokens', 'packed_samples': 'packed_samples', 'dropped_samples': 'dropped_samples_no_fit',
           'dropped_tokens': 'dropped_tokens_no_fit', 'requeued': 'requeued_samples', 'empty_packs': 'empty_packs',
           'fill': 'pack_fill', 'seconds': 'seconds', 'trunc_rate': 'truncated_share_32k',
           'no_think_rate': 'no_think_end_share', 'rep_mean': 'repetition_32gram', 'lp_mean': 'sampling_time_logprob',
           'packs': 'packs_with_data',
           'resp_len_p50': 'response_len_median', 'resp_len_p90': 'response_len_p90',
           'resp_len_p99': 'response_len_p99', 'resp_len_max': 'response_len_max', 'resp_len_mean': 'response_len_mean'}
CUM = {'prompts': 'prompts_used', 'prompt_tokens': 'prompt_tokens', 'gen_tokens': 'generated_tokens',
       'packed_tokens': 'packed_tokens', 'loss_tokens': 'loss_tokens', 'dropped_tokens': 'dropped_tokens',
       'subset_share': 'subset_used_share'}
TIME = {'step_s': 'step_seconds', 'push_s': 'weight_push_seconds'}


def wandb_view(m, split, lr=None, grad_norm=None):
    out = {}
    val = split != 'train'
    vp = 'val_' if val else ''
    for k, v in m.items():
        p = k.split('/')
        if p[0] == 'kd' and len(p) >= 4:
            grp, field = p[2], p[-1]
            if grp == 'all':
                out['kl_%s/%s' % ('val' if val else 'train', KD_FIELD.get(field, field))] = v
            elif grp == 'sup' and field in ('kl', 'tokens', 'top1'):  # response + history (= the loss)
                out['%skl_supervised_all/%s' % (vp, KD_FIELD.get(field, field))] = v
            elif grp == 'ds' and field in ('kl', 'tokens', 'top1'):
                sec = {'kl': 'kl', 'tokens': 'loss_tokens', 'top1': 'top1'}[field]
                out['%s%s_by_dataset/%s' % (vp, sec, p[3])] = v
            elif grp == 'hist_ds' and field in ('kl', 'tokens'):
                out['%shistory_%s_by_dataset/%s' % (vp, {'kl': 'kl', 'tokens': 'loss_tokens'}[field], p[3])] = v
            elif grp in ('pos', 'seg') and field in ('kl', 'tokens'):
                sec = {'kl': 'kl', 'tokens': 'loss_tokens'}[field]
                label = POS.get(p[3], p[3]) if grp == 'pos' else SEG.get(p[3], p[3])
                out['%s%s_by_%s/%s' % (vp, sec, 'position' if grp == 'pos' else 'segment', label)] = v
        elif p[0] == 'rollout_ds' and len(p) == 3:
            sec = {'resp_mean': 'response_len_by_dataset', 'trunc_rate': 'truncated_share_by_dataset',
                   'n': 'count_by_dataset'}.get(p[2])
            if sec:
                out['%s/%s' % (sec, p[1])] = v
        elif p[0] in ('rollout', 'val_rollout') and len(p) == 2 and p[1] in ROLLOUT:
            out['%s/%s' % (p[0], ROLLOUT[p[1]])] = v
        elif p[0] == 'cumulative' and p[1] in CUM:
            out['cumulative/%s' % CUM[p[1]]] = v
        elif p[0] == 'time':
            if p[1] in TIME:
                out['time/%s' % TIME[p[1]]] = v
            elif p[1] == 'wait_for_packs_s' and len(p) == 3:
                out['time/wait_for_packs_seconds_%s' % p[2]] = v
    g = m.get
    if not val:
        key = {'train_kl': g('kd/train/all/kl'), 'top1_agreement': g('kd/train/all/top1'),
               'answer_kl': g('kd/train/seg/answer/kl'), 'thinking_kl': g('kd/train/seg/think/kl'),
               'history_kl': g('kd/train/seg/history/kl'), 'supervised_kl': g('kd/train/sup/kl'),
               'history_loss_tokens': g('rollout/hist_loss_tokens'),
               'share_tokens_kl_over_1': g('kd/train/all/kl_gt1'),
               'logprob_gap_teacher_minus_student': g('kd/train/all/logratio'),
               'learning_rate': lr, 'grad_norm': grad_norm, 'response_len_mean': g('rollout/resp_len_mean'),
               'truncated_share_32k': g('rollout/trunc_rate'), 'no_think_end_share': g('rollout/no_think_rate'),
               'repetition_32gram': g('rollout/rep_mean'), 'pack_fill': g('rollout/fill'),
               'packs_with_data': g('rollout/packs'),
               'weight_push_seconds': g('time/push_s')}
        if g('time/step_s'):
            key['step_minutes'] = g('time/step_s') / 60
            if g('time/wait_for_packs_s/stage0') is not None:
                key['trainer_idle_share'] = g('time/wait_for_packs_s/stage0') / g('time/step_s')
        gen, prm, drop = g('rollout/gen_tokens'), g('rollout/prompt_tokens'), g('rollout/dropped_tokens')
        if gen and drop is not None:
            key['rollout_wasted_share'] = drop / (gen + (prm or 0))
        if g('cumulative/packed_tokens') is not None:
            key['tokens_seen_billions'] = g('cumulative/packed_tokens') / 1e9
        if g('cumulative/subset_share') is not None:
            key['subset_used_percent'] = 100 * g('cumulative/subset_share')
    else:
        key = {'val_kl': g('kd/val/all/kl'), 'val_top1_agreement': g('kd/val/all/top1'),
               'val_answer_kl': g('kd/val/seg/answer/kl'), 'val_history_kl': g('kd/val/seg/history/kl'),
               'val_supervised_kl': g('kd/val/sup/kl'), 'val_truncated_share_32k': g('val_rollout/trunc_rate'),
               # these ranked the checkpoints of a Nemotron Nano 9B V2 run like the evaluations did (val_kl did not:
               # repetition loops lower it)
               'val_kl_first_1k': g('kd/val/pos/0-1k/kl'), 'val_repetition_32gram': g('val_rollout/rep_mean')}
    out.update({'key/' + k: v for k, v in key.items() if v is not None})
    return out
