"""Render saved inputs, both tool results, agent answers and evaluation as HTML."""
import argparse
from html import escape
import json
from pathlib import Path


def pretty(value):
    return '<pre>'+escape(json.dumps(value, indent=2, ensure_ascii=False))+'</pre>'


def render(report):
    parts = ['''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Two-tool job-ad review results</title><style>
body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 22px;color:#1e293b}
h1,h2,h3{line-height:1.3}article{border-top:2px solid #cbd5e1;margin-top:38px;padding-top:18px}
table{border-collapse:collapse;width:100%;margin:18px 0}th,td{text-align:left;border-bottom:1px solid #cbd5e1;padding:9px;vertical-align:top}
pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f9;padding:16px;border-radius:6px;font-size:13px}
summary{cursor:pointer;font-weight:600;margin:12px 0}.note{background:#fff7ed;padding:16px;border-left:4px solid #ea580c}
blockquote{border-left:4px solid #94a3b8;padding-left:16px;margin-left:0}
</style><body><h1>Real LightGBM + lexicon/context tool</h1>
<p>Inputs, actual tool outputs, the LLM answer and the automated evaluation are shown separately.</p>
<div class="note">Read the run's input source and selection before interpreting results. Scores are not calibrated fraud probabilities. Pass means both current checks passed, not verified employer legitimacy. All outputs are recommendations only. Judge ratings are automated opinions and may share the agent's mistakes.</div>''']
    parts.append('<h2>Run summary</h2>'+pretty(dict(rubric=report.get('rubric_version'), judge=report.get('judge'), summary=report['summary'])))
    for row in report['rows']:
        parts.append('<article><h2>'+escape(row['case_id'])+'</h2>')
        case = row.get('case')
        if not case:
            parts.append(pretty(row)+'</article>')
            continue
        parts.append('<h3>1. Input</h3>'+pretty(case['posting']))
        parts.append('<p>Reference label (hidden from the agent and judge): <strong>'
                     +escape(str(row.get('fraudulent')))+'</strong></p>')
        for number, name in enumerate(('score_posting', 'match_lexicon'), 2):
            event = next((t for t in case['tool_trace'] if t['name'] == name), None)
            parts.append(f'<h3>{number}. Tool: {escape(name)}</h3>')
            if not event:
                parts.append('<p>No tool output was recorded.</p>')
                continue
            output = event['output']
            if name == 'score_posting':
                selected = {k: output.get(k) for k in ('model_family','model_version','model_score','cutoff','flagged','is_test_fixture','passed','review_reasons','calibrated_probability','missing_input_fields')}
                parts.append(pretty(selected))
            else:
                parts.append('<p>Context review: '+escape(str(output.get('context_status')))+'</p>')
                parts.append(pretty({k: output.get(k) for k in ('passed', 'review_reasons')}))
                parts.append('<table><tr><th>Finding</th><th>Exact text</th><th>Context</th><th>Explanation</th></tr>')
                for evidence in output.get('evidence', []):
                    parts.append('<tr>'+''.join('<td>'+escape(str(v if v is not None else ''))+'</td>' for v in
                        (evidence['kind'], evidence.get('quote'), evidence.get('context'), evidence['observation']))+'</tr>')
                parts.append('</table>')
            parts.append('<details><summary>Full actual tool result</summary>'+pretty(event)+'</details>')
        assessment = case['assessment']
        parts.append('<h3>4. LLM answer</h3><p>Agent status: <strong>'+escape(case['status'])
                     +'</strong>; recovery attempted: '+escape(str(case.get('recovery_attempted', False)))+'</p>')
        parts.append('<p>Recommendation: <strong>'+escape(assessment['recommendation'])+'</strong></p><blockquote>'
                     +escape(assessment['rationale'])+'</blockquote>')
        parts.append('<details open><summary>Full answer, limitations and next steps</summary>'+pretty(assessment)+'</details>')
        parts.append('<h3>5. Evaluation</h3>'+pretty(dict(code=row.get('code'), judge_status=row['judge_status'],
                     judge_error=row.get('judge_error'), scoring=row.get('scoring'))))
        parts.append('<details open><summary>Judge scores, reasons and quoted answer passages</summary>'+pretty(row.get('judgment'))+'</details></article>')
    parts.append('</body></html>')
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
