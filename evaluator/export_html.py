"""Standalone evaluation dashboard rendered from saved results; no model calls."""
import argparse
from html import escape
import json
from pathlib import Path

from evaluator.dashboard_metrics import decision_metrics, score_distribution
from evaluator.schemas import DIMENSIONS

HERE = Path(__file__).resolve().parent
NAMES = dict(evidence_grounding="Evidence grounding", context_understanding="Context understanding",
             recommendation_consistency="Recommendation consistency", explanation_clarity="Explanation clarity")
ACTIONS = dict(pass_no_review_needed="Pass - no review needed", review_needed="Review needed",
               propose_rejection="Rejection proposed", escalate_to_review="Review recommended")


def e(value):
    return escape(str(value if value is not None else "Unavailable"), quote=True)


def pretty(value):
    return '<pre>'+escape(json.dumps(value, indent=2, ensure_ascii=False))+'</pre>'


def dropdown(title, body, css="section", extra=""):
    return f'<details class="{css}" {extra}><summary>{title}</summary><div class="details-body">{body}</div></details>'


def badge(label, tone="neutral"):
    return f'<span class="badge {tone}">{e(label)}</span>'


def state(value, yes="Passed", no="Failed"):
    return badge(yes, "good") if value is True else badge(no, "bad") if value is False else badge("Unavailable")


def action(value):
    return badge(ACTIONS.get(value, "Unavailable"), "good" if value == 'pass_no_review_needed' else "review")


def number(value, digits=4):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) and not isinstance(value, bool) else 'N/A'


def pct(numerator, denominator):
    return f'{100*numerator/denominator:.0f}%' if denominator else 'N/A'


def metric(label, value, caption, css=""):
    return f'<div class="metric {css}"><span class="metric-label">{e(label)}</span><strong>{e(value)}</strong><span class="metric-caption">{e(caption)}</span></div>'


def tool(case, name):
    events = [t for t in case.get('tool_trace', []) if t['name'] == name]
    return next((t for t in reversed(events) if t['status'] == 'success'), events[-1] if events else None)


def effective_scores(row):
    # Do not promote raw judge ratings when code excludes a score.
    return (row.get('scoring') or {}).get('effective_scores') or {}


def score_text(row):
    values = [v for v in effective_scores(row).values() if isinstance(v, (float, int))]
    return f'{sum(values)/len(values):.2f}/4' if values else 'N/A'


def bullets(values):
    return '<ul>'+''.join('<li>'+e(v)+'</li>' for v in values)+'</ul>' if values else '<p class="muted">None returned.</p>'


def prf_cards(metrics):
    tp, fp, fn = [metrics[k] for k in ('true_positive', 'false_positive', 'false_negative')]
    cards = metric('Precision', pct(tp,tp+fp), f'{tp} correct fake predictions / {tp+fp} predicted fake' if tp+fp else 'No fake predictions')
    cards += metric('Recall', pct(tp,tp+fn), f'{tp} caught / {tp+fn} actual fake' if tp+fn else 'No fake ads in this selection')
    denominator = 2*tp+fp+fn
    cards += metric('F1 score', pct(2*tp,denominator), 'Balance of precision and recall' if denominator else 'No actual or predicted fake ads')
    return '<div class="prf-grid">'+cards+'</div>'


def confusion_matrix(metrics, source):
    n = metrics['evaluated']
    if not n:
        return '<section class="panel"><h2>Confusion matrix</h2><p>No labelled, valid '+source+' predictions are available.</p></section>'
    tn, fp, fn, tp = [metrics[k] for k in ('true_negative','false_positive','false_negative','true_positive')]
    first, second = ('Pass', 'Review needed') if source == 'llm' else ('Below cutoff','Flagged')
    title = 'LLM recommendation confusion matrix' if source == 'llm' else 'Confusion matrix'
    return f'''<section class="panel matrix-panel"><div class="panel-heading"><h2>{title}</h2><span class="subtle">{n} scored postings</span></div>
<p class="axis-caption">Actual dataset label &darr; &nbsp; / &nbsp; {'LLM recommendation' if source == 'llm' else 'Classifier prediction'} &rarr;</p>
<table class="matrix" data-source="{source}"><thead><tr><th scope="col"></th><th scope="col">Genuine<br><small>{first}</small></th><th scope="col">Fake<br><small>{second}</small></th></tr></thead><tbody>
<tr><th scope="row">Genuine <small>{tn+fp} actual</small></th><td class="cell-good"><strong>{tn}</strong><span>True negatives</span></td><td class="cell-warn"><strong>{fp}</strong><span>False positives</span></td></tr>
<tr><th scope="row">Fake <small>{tp+fn} actual</small></th><td class="{'cell-bad' if fn else 'cell-empty'}"><strong>{fn}</strong><span>False negatives</span></td><td class="{'cell-good' if tp+fn else 'cell-empty'}"><strong>{tp}</strong><span>True positives</span></td></tr></tbody></table></section>'''


def distribution_chart(rows):
    distribution = score_distribution(rows)
    bins = distribution['bins']
    if not distribution['count']:
        return '<section class="panel distribution-panel"><h2>Distribution of classifier scores</h2><p>No valid real classifier scores are available.</p></section>'
    peak = max(sum(b[k] for k in ('below','flagged','unknown')) for b in bins)
    # Fixed 0..1 score axis with ten equal-width bins; counts stay visible.
    left, top, width, height = 43, 40, 474, 176
    svg = [f'<svg class="histogram" viewBox="0 0 540 278" role="img" aria-labelledby="histogram-title"><title id="histogram-title">Distribution of classifier scores for {distribution["count"]} postings</title>']
    ticks = sorted({0, max(1,peak//2), peak})
    for tick in ticks:
        y = top+height-height*tick/peak
        svg.append(f'<line class="chart-grid" x1="{left}" x2="{left+width}" y1="{y}" y2="{y}"/><text class="chart-label" x="{left-9}" y="{y+4}" text-anchor="end">{tick}</text>')
    svg.append('<text class="chart-axis-title" x="43" y="20">Posting count</text>')
    for i,b in enumerate(bins):
        x = left+i*width/len(bins)+4
        total = sum(b[k] for k in ('below','flagged','unknown'))
        base = top+height
        for key,colour in [('below','#307d6e'),('flagged','#d99a36'),('unknown','#a0adb7')]:
            count = b[key]
            bar_height = height*count/peak
            base -= bar_height
            svg.append(f'<rect x="{x:.2f}" y="{base:.2f}" width="{width/len(bins)-8:.2f}" height="{bar_height:.2f}" rx="3" fill="{colour}"><title>Scores {b["low"]:.1f} to {b["high"]:.1f}: {count} {key} postings</title></rect>')
        if total:
            svg.append(f'<text class="chart-count" x="{x+(width/len(bins)-8)/2:.2f}" y="{base-7:.2f}" text-anchor="middle">{total}</text>')
    for i in range(6):
        value = i/5
        x = left+width*value
        svg.append(f'<text class="chart-label" x="{x:.2f}" y="{top+height+21}" text-anchor="middle">{value:.1f}</text>')
    for cutoff in distribution['cutoffs']:
        x = left+width*cutoff
        svg.append(f'<line class="cutoff-line" x1="{x:.2f}" x2="{x:.2f}" y1="{top}" y2="{top+height}"><title>Cutoff {cutoff:.6f}</title></line>')
    svg.append('<text class="chart-axis-title" x="280" y="266" text-anchor="middle">Classifier score</text></svg>')
    cutoff_text = ', '.join(number(v) for v in distribution['cutoffs'])
    subtitle = f'{distribution["count"]} scores / 10 bins'
    legend = '<div class="chart-legend"><span><i class="legend-below"></i>Below cutoff</span><span><i class="legend-flagged"></i>Flagged</span>'
    if cutoff_text:
        legend += '<span><i class="legend-cutoff"></i>Cutoff '+e(cutoff_text)+'</span>'
    legend += '</div>'
    return '<section class="panel distribution-panel"><div class="panel-heading"><h2>Distribution of classifier scores</h2><span class="subtle">'+subtitle+'</span></div>'+''.join(svg)+legend+'</section>'


def classifier_panel(rows):
    metrics = decision_metrics(rows, 'classifier')
    return '<section class="result-section" id="classifier-results"><p class="eyebrow">01 / Classifier KPIs</p><h2>Classifier results</h2>'+prf_cards(metrics)+'<div class="classification-grid">'+confusion_matrix(metrics,'classifier')+distribution_chart(rows)+'</div></section>'


def llm_panel(rows, judge):
    metrics = decision_metrics(rows,'llm')
    mapping = f'Positive class: fake. Review needed = predicted fake; pass = predicted genuine. Completed, labelled recommendations: {metrics["evaluated"]}/{len(rows)}.'
    return '<section class="result-section" id="llm-results"><p class="eyebrow">02 / LLM result KPIs</p><h2>LLM results</h2><p class="mapping-caption">'+mapping+'</p>'+prf_cards(metrics)+'<div class="llm-analysis-grid">'+confusion_matrix(metrics,'llm')+judge_panel(rows,judge)+'</div></section>'


def workflow_panel(rows):
    n = len(rows)
    passed = sum(r.get('recommendation') == 'pass_no_review_needed' for r in rows)
    review = sum(r.get('recommendation') in ('review_needed','escalate_to_review','propose_rejection') for r in rows)
    completed = sum(r.get('agent_status') == 'completed' for r in rows)
    code = sum((r.get('code') or {}).get('all_passed') is True for r in rows)
    judged = sum(r.get('judge_status') == 'completed' for r in rows)
    recovered = sum((r.get('case') or {}).get('recovery_attempted',False) for r in rows)
    cards = [metric('Pass recommended', str(passed), f'{pct(passed,n)} of postings', 'accent-good'),
             metric('Review recommended', str(review), f'{pct(review,n)} of postings', 'accent-review'),
             metric('Agent completed', pct(completed,n), f'{completed} / {n} completed'),
             metric('Code checks passed', pct(code,n), f'{code} / {n} evaluations'),
             metric('Judge coverage', pct(judged,n), f'{judged} / {n} responses'),
             metric('Answer correction needed', pct(recovered,n), f'{recovered} / {n} used a recovery call')]
    return dropdown('Execution metrics', '<div class="kpi-grid">'+''.join(cards)+'</div>', 'workflow panel')


def judge_panel(rows, judge):
    cards = []
    for key in DIMENSIONS:
        values = [effective_scores(r).get(key) for r in rows if r.get('judge_status') == 'completed']
        values = [v for v in values if isinstance(v, (int,float))]
        mean = sum(values)/len(values) if values else None
        width = 100*mean/4 if mean is not None else 0
        cards.append('<div class="judge-card"><span>'+e(NAMES[key])+'</span><strong>'+number(mean,1)+'<small> / 4</small></strong>'
                     +f'<div class="score-track"><div style="width:{width:.2f}%"></div></div><small>{len(values)} scored evaluations</small></div>')
    return '<section class="panel judge-panel"><div class="panel-heading"><h2>LLM-as-judge ratings</h2><span class="subtle">'+e((judge or {}).get('model','Model not recorded'))+'</span></div><div class="judge-grid">'+''.join(cards)+'</div></section>'


def posting_input(posting):
    parts = []
    for field, value in posting.items():
        if field == 'posting_id':
            continue
        content = '<p class="posting-text">'+e(value)+'</p>' if value is not None else '<p class="muted">Not supplied</p>'
        parts.append('<div class="input-field"><h4>'+e(field.replace('_',' ').capitalize())+'</h4>'+content+'</div>')
    return ''.join(parts)+dropdown('Input JSON',pretty(posting), 'raw')


def classifier_output(event):
    if not event:
        return '<p class="muted">No classifier response recorded.</p>'
    out = event.get('output') or {}
    if event.get('status') != 'success':
        return '<p>Classifier tool did not complete.</p>'+pretty(event)
    flag = out.get('flagged')
    flag_label = badge('Flagged', 'review') if flag is True else badge('Below cutoff','good') if flag is False else badge('Unavailable')
    content = '<div class="tool-metrics">'+metric('Score',number(out.get('model_score')), 'Model output')+metric('Cutoff',number(out.get('cutoff')), 'Review threshold')
    content += '<div class="metric"><span class="metric-label">Classifier result</span>'+flag_label+'</div></div>'
    if 'passed' in out:
        content += '<p>Tool check '+state(out.get('passed'))+'</p>'
    if out.get('review_reasons'):
        content += '<h4>Review reasons</h4>'+bullets(out['review_reasons'])
    if out.get('missing_input_fields'):
        content += '<h4>Inputs not supplied</h4><p class="muted">'+e(', '.join(out['missing_input_fields']))+'</p>'
    return content+dropdown('Complete classifier response',pretty(event), 'raw')


def lexicon_output(event):
    if not event:
        return '<p class="muted">No lexicon response recorded.</p>'
    out = event.get('output') or {}
    content = '<div class="inline-status"><span>Tool check '+state(out.get('passed'))+'</span><span>Context review '+badge(out.get('context_status','Unavailable'), 'good' if out.get('context_status') == 'completed' else 'review')+'</span></div>'
    if out.get('review_reasons'):
        content += '<h4>Review reasons</h4>'+bullets(out['review_reasons'])
    if out.get('context_error'):
        content += '<p class="error-text">'+e(out['context_error'])+'</p>'
    evidence = out.get('evidence') or []
    groups = [('Risk wording',[v for v in evidence if v.get('source') in ('risk_lexicon','llm_context')]),
              ('Posting observations',[v for v in evidence if v.get('source') == 'posting']),
              ('Background matches',[v for v in evidence if v.get('source') == 'dataset_lexicon'])]
    for title, found in groups:
        cards = []
        for v in found:
            body = '<div class="evidence-meta">'+badge(v.get('evidence_id'))+' '+badge(v.get('field'))+' '+badge(v.get('context'))+'</div>'
            if v.get('quote'):
                body += '<blockquote>'+e(v['quote'])+'</blockquote>'
            body += '<p>'+e(v.get('observation'))+'</p>'
            cards.append('<div class="evidence-card">'+body+'</div>')
        content += dropdown(e(title)+f' <span class="count">{len(found)}</span>', ''.join(cards) if cards else '<p class="muted">No findings returned.</p>', 'subsection')
    return content+dropdown('Complete lexicon response',pretty(event), 'raw')


def llm_answer(case):
    a = case.get('assessment') or {}
    content = '<p>'+action(a.get('recommendation'))+'</p><blockquote class="rationale">'+e(a.get('rationale'))+'</blockquote>'
    content += '<div class="answer-grid"><div><h4>Limitations</h4>'+bullets(a.get('limitations'))+'</div><div><h4>Suggested next steps</h4>'+bullets(a.get('next_steps'))+'</div></div>'
    content += '<p class="muted">Cited evidence: '+e(', '.join(a.get('evidence_ids') or []) or 'None')+'</p>'
    content += '<p>Final-answer correction: '+badge('Used' if case.get('recovery_attempted') else 'Not needed', 'review' if case.get('recovery_attempted') else 'neutral')+'</p>'
    return content+dropdown('Complete answer JSON',pretty(a), 'raw')


def evaluation_output(row):
    code = row.get('code') or {}
    checks = code.get('checks') or {}
    content = '<div class="evaluation-heading"><h4>Code checks</h4>'+state(code.get('all_passed'))+'</div>'
    content += '<div class="check-grid">'+''.join('<div><span>'+e(k.replace('_',' ').capitalize())+'</span>'+state(v)+'</div>' for k,v in checks.items())+'</div>'
    content += '<div class="evaluation-heading"><h4>LLM judge</h4>'+badge(row.get('judge_status'), 'good' if row.get('judge_status') == 'completed' else 'review')+'</div>'
    if row.get('judge_error'):
        content += '<p class="error-text">'+e(row['judge_error'])+'</p>'
    judgment = row.get('judgment') or {}
    scores = effective_scores(row)
    for key in DIMENSIONS:
        rating = judgment.get(key) or {}
        text = '<p>'+e(rating.get('reason','No judgment returned.'))+'</p>'
        text += ''.join('<blockquote>'+e(q)+'</blockquote>' for q in rating.get('answer_quotes') or [])
        if scores.get(key) != rating.get('score'):
            text += '<p class="muted">Raw judge score: '+number(rating.get('score'),0)+'/4; effective score: '+number(scores.get(key),0)+'</p>'
        content += dropdown(e(NAMES[key])+' <span class="rating">'+number(scores.get(key),0)+'/4</span>',text,'subsection')
    claims = judgment.get('unsupported_claims') or []
    if claims:
        content += dropdown('Judge-reported unsupported claims '+badge(len(claims),'review'), ''.join('<blockquote>'+e(c['quote'])+'</blockquote><p>'+e(c['reason'])+'</p>' for c in claims),'subsection')
    adjustments = (row.get('scoring') or {}).get('adjustments') or []
    if adjustments:
        content += '<h4>Score adjustments</h4>'+bullets(adjustments)
    return content+dropdown('Complete evaluation JSON',pretty({k:row.get(k) for k in ('code','judge_status','judge_error','judgment','scoring','agent_error','input_error')}),'raw')


def render(report):
    rows = report.get('rows') or []
    css = (HERE/'report.css').read_text(encoding='utf-8')
    script = (HERE/'report.js').read_text(encoding='utf-8')
    models = sorted({(r.get('case') or {}).get('model_name','') for r in rows} - {''})
    source = str(report.get('source') or '').replace('\\','/').split('/')[-1]
    chips = ''.join('<span>'+e(v)+'</span>' for v in (f'{len(rows)} evaluations',source,*models) if v)
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Job posting evaluation</title><style>'+css+'</style></head><body><main>',
             '<header><div><p class="eyebrow">Evaluation report</p><h1>Job posting review</h1><div class="meta">'+chips+'</div></div><button class="print-button" onclick="window.print()">Print / save PDF</button></header>',
             classifier_panel(rows),llm_panel(rows,report.get('judge')),workflow_panel(rows),
             '<section class="postings"><p class="eyebrow">03 / Posting breakdown</p><div class="panel-heading"><h2>All postings</h2><span id="visible-count" class="subtle" aria-live="polite">'+str(len(rows))+' of '+str(len(rows))+' shown</span></div>',
             '<div class="toolbar"><label class="search-label">Find a posting<input id="search" type="search" placeholder="Search title or ID"></label><label>Recommendation<select id="filter"><option value="all">All recommendations</option><option value="pass_no_review_needed">Pass - no review needed</option><option value="review_needed">Review needed</option></select></label><button id="expand-all">Expand all postings</button><button id="collapse-all">Collapse all</button></div>',
             '<div class="table-wrap"><table class="posting-table"><thead><tr><th>ID / title</th><th>Dataset label</th><th>Classifier</th><th>Lexicon</th><th>Recommendation</th><th>Code checks</th><th>Judge avg.</th></tr></thead><tbody>']
    case_blocks = []
    for i,row in enumerate(rows):
        case = row.get('case') or {}
        posting = case.get('posting') or {}
        title = posting.get('title') or 'Input unavailable'
        cid = row['case_id']
        anchor = 'posting-'+str(i)
        decision = row.get('recommendation') or (case.get('assessment') or {}).get('recommendation')
        attrs = f'data-search="{e((str(cid)+" "+title).lower())}" data-action="{e(decision)}"'
        score_event, lex_event = tool(case,'score_posting'), tool(case,'match_lexicon')
        score = (score_event or {}).get('output') or {}
        lex = (lex_event or {}).get('output') or {}
        flag = score.get('flagged')
        score_cell = '<strong class="score-number">'+number(score.get('model_score'))+'</strong><small>'+('Flagged' if flag is True else 'Below cutoff' if flag is False else 'Unavailable')+'</small>'
        gold = {0:'Genuine',1:'Fake'}.get(row.get('fraudulent'),'Unlabelled')
        parts.append(f'<tr class="posting-row" {attrs}><td><a class="posting-link" href="#{anchor}" data-open="{anchor}"><span class="row-id">ID {e(cid)}</span>{e(title)}</a></td><td>'+badge(gold)+'</td><td>'+score_cell+'</td><td>'+state(lex.get('passed'))+'</td><td>'+action(decision)+'</td><td>'+state((row.get('code') or {}).get('all_passed'))+'</td><td class="judge-average">'+score_text(row)+'</td></tr>')
        head = '<div class="case-title"><span class="row-id">Posting '+e(cid)+'</span><strong>'+e(title)+'</strong></div><div class="case-badges">'+action(decision)+'</div>'
        sections = dropdown('<span class="step">1</span> Input posting',posting_input(posting) if posting else '<p>No posting snapshot was returned.</p>')
        sections += dropdown('<span class="step">2</span> Classifier tool',classifier_output(score_event))
        sections += dropdown('<span class="step">3</span> Lexicon tool',lexicon_output(lex_event))
        sections += dropdown('<span class="step">4</span> LLM answer',llm_answer(case) if case else '<p>No agent answer was returned.</p>')
        sections += dropdown('<span class="step">5</span> Evaluation',evaluation_output(row))
        sections += dropdown('Recorded model tool calls',pretty(case.get('llm_tool_calls') or []),'raw')
        case_blocks.append(dropdown(head,sections,'case',f'id="{anchor}" {attrs}'))
    parts += ['</tbody></table></div><p id="no-results" hidden>No postings match your search.</p><div class="case-list">',*case_blocks,'</div></section>',
              '<footer>Generated from saved evaluation results. Judge ratings use a 0-4 scale.</footer></main><script>'+script+'</script></body></html>']
    return '\n'.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding='utf-8-sig'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(report), encoding='utf-8')
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
