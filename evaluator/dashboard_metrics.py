"""Metrics for report presentation; preserve raw evaluation records."""
import math

ACTION_PREDICTIONS = {'pass_no_review_needed': False, 'review_needed': True}


def decision_metrics(rows, source='classifier'):
    if source not in ('classifier', 'llm'):
        raise ValueError('Choose classifier or llm metrics.')
    pairs = []
    for row in rows:
        if row.get('fraudulent') not in (0, 1):
            continue
        if source == 'classifier':
            predicted = row.get('classifier_prediction')
        else:
            # A safe application fallback is not a completed LLM prediction.
            status = row.get('agent_status', (row.get('case') or {}).get('status'))
            predicted = ACTION_PREDICTIONS.get(row.get('recommendation')) if status == 'completed' else None
        if type(predicted) is bool:
            pairs.append((row['fraudulent'], predicted))
    tp = sum(label == 1 and predicted for label, predicted in pairs)
    fp = sum(label == 0 and predicted for label, predicted in pairs)
    fn = sum(label == 1 and not predicted for label, predicted in pairs)
    tn = len(pairs)-tp-fp-fn
    return dict(evaluated=len(pairs), total=len(rows), true_positive=tp, false_positive=fp,
                false_negative=fn, true_negative=tn,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)


def score_distribution(rows, bins=10):
    if bins < 1:
        raise ValueError('Provide a positive bin count.')
    values = [dict(low=i/bins, high=(i+1)/bins, below=0, flagged=0, unknown=0) for i in range(bins)]
    cutoffs = set()
    for row in rows:
        case = row.get('case') or {}
        event = next((t for t in reversed(case.get('tool_trace', []))
                      if t.get('name') == 'score_posting' and t.get('status') == 'success'), None)
        if not event:
            continue
        result = event.get('output') or {}
        fixture = result.get('is_test_fixture', result.get('is_dummy'))
        score = result.get('model_score')
        if (fixture is not False or isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score) or not 0 <= score <= 1):
            continue
        bucket = values[min(int(score*bins), bins-1)]
        key = 'flagged' if result.get('flagged') is True else 'below' if result.get('flagged') is False else 'unknown'
        bucket[key] += 1
        cutoff = result.get('cutoff')
        if isinstance(cutoff, (int, float)) and not isinstance(cutoff, bool) and math.isfinite(cutoff) and 0 <= cutoff <= 1:
            cutoffs.add(cutoff)
    return dict(bins=values, count=sum(sum(b[k] for k in ('below','flagged','unknown')) for b in values),
                cutoffs=sorted(cutoffs))
