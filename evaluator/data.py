"""Strict, allowlisted inputs. Gold annotations never enter agent inputs."""
import csv
import json
from pathlib import Path
from agent.core.schemas import Posting, ReviewCase
from evaluator.schemas import Example

POSTING_FIELDS = tuple(f for f in Posting.model_fields if f != 'posting_id')


def records(path):
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        if path.suffix.lower() == '.csv':
            return list(csv.DictReader(stream))
        if path.suffix.lower() == '.jsonl':
            return [json.loads(line) for line in stream if line.strip()]
        value = json.load(stream)
    return value if isinstance(value, list) else [value]


def label(value):
    if value is None or value == '':
        return None
    if type(value) is int and value in (0, 1):
        return value
    if isinstance(value, str) and value.strip() in ('0', '1'):
        return int(value.strip())
    raise ValueError('fraudulent must be 0, 1, or absent.')


def load_examples(path, replay=False):
    examples, seen = [], set()
    for position, record in enumerate(records(path), 1):
        raw_case = record.get('case') if replay else None
        if replay and raw_case is None and 'assessment' in record:
            raw_case = record  # Direct ReviewCase JSON is also supported.
        if replay and raw_case is None:
            raise ValueError(f'Record {position} has no saved case.')
        case = ReviewCase.model_validate(raw_case) if replay else None
        source = record.get('posting', record)
        if replay:
            source = record.get('posting', case.posting.model_dump())
        case_id = str(record.get('case_id') or source.get('posting_id')
                      or source.get('job_id') or f'row_{position}')
        if case_id in seen:
            raise ValueError('Case IDs must be unique; no records are silently skipped.')
        seen.add(case_id)
        fields = {f: source.get(f) or None for f in POSTING_FIELDS}
        for field in ('telecommuting', 'has_company_logo', 'has_questions'):
            value = source.get(field)
            fields[field] = int(value) if str(value).strip() in ('0', '1') else value or None
        fields['posting_id'] = str(source.get('posting_id') or case_id)
        try:
            posting = Posting.model_validate(fields)
            input_error = None
        except ValueError:
            posting, input_error = None, 'invalid_posting_fields'
        examples.append(Example(case_id=case_id, posting=posting,
            input_error=input_error,
            fraudulent=label(record.get('fraudulent')),
            expected_action=record.get('expected_action') or None, case=case))
    if not examples:
        raise ValueError('Input contains no examples.')
    return examples
