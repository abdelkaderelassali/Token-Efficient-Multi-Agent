"""Reproducible live requirement check; preserve all blocked and measured attempts."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

from adaptive_workflow import run_pair, PROTOCOL, scenarios
from custom_scenarios import FIELDS, normalize_custom
from evaluation_suite import summarize
from context_integrity_audit import audit_experiment


def inputs():
    data = {key: copy.deepcopy(scenarios()[2][key]) for key in FIELDS}
    data['title'] = 'Requirement test: refrigerated international shipment'
    data['brief'] += ' Refrigerated transport is mandatory. Only D6 declares this capability.'
    answers = {}
    for quote in data['options']:
        supplied = quote['id'] == 'D6'
        quote['terms'] += ' Refrigerated transport is ' + ('included.' if supplied else 'not available.')
        answers[quote['id']] = {'refrigerated_transport': supplied}
    data['requirement_review'] = {'confirmed':True, 'required_capabilities':['refrigerated_transport'],
        'other_requirements':[], 'quote_capabilities':answers}
    unknown = copy.deepcopy(data)
    unknown['title'] = 'Requirement test: missing refrigeration declaration'
    unknown['requirement_review']['quote_capabilities']['D6']['refrigerated_transport'] = None
    # Remove the declaration from the unknown-source fixture as well.
    unknown['options'][-1]['terms'] = unknown['options'][-1]['terms'].replace(' Refrigerated transport is included.', '')
    unknown['brief'] = unknown['brief'].replace(' Only D6 declares this capability.', '')
    unsupported = copy.deepcopy(data)
    unsupported['title'] = 'Requirement test: unsupported temperature range'
    unsupported['brief'] += ' Maintain a temperature of 2–8°C throughout transport.'
    unsupported['requirement_review']['other_requirements'] = ['Maintain 2–8°C throughout transport']
    return [data, unknown, unsupported]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.output.with_suffix('.summary.json'), args.output.with_suffix('.audit.json')):
        if path.exists():
            parser.error('Use new output paths')
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for data in inputs():
        print(data['title'], flush=True)
        result = run_pair(normalize_custom(data), seed=130)
        records.append(result)
        args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(json.dumps({'status':result['status'], 'comparison':result.get('comparison'),
                          'requirements':result.get('requirement_validation')}), flush=True)
    summary = {'protocol':PROTOCOL, 'phase':'pilot', 'created_at':datetime.now(timezone.utc).isoformat(),
               'models':summarize(records), 'source_files':[str(args.output)],
               'scope':'Requirement safety check: one live three-branch Llama 3 comparison and two pre-inference '
                       'Needs review cases. Blocked cases are not successful decisions or token-saving measurements. '
                       'Requirements and supplier declarations were explicitly authored in these fictional fixtures.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    args.output.with_suffix('.audit.json').write_text(json.dumps([audit_experiment(r) for r in records], indent=2), encoding='utf-8')
    valid = records[0]['status'] == 'validated' and audit_experiment(records[0])['all_checks_passed']
    valid = valid and all(records[0][mode]['evaluation']['decision']['option_id'] == 'D6'
                         for mode in ('baseline', 'compression_only', 'compressed'))
    valid = valid and all(r['status'] == 'needs_review' and r['measured_tokens'] == 0 for r in records[1:])
    return 0 if valid else 1


if __name__ == '__main__':
    raise SystemExit(main())
