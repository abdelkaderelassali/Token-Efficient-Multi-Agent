"""Reproduce adaptive comparisons with source audits and all three branch costs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from adaptive_workflow import PROTOCOL, run_pair, scenarios
from custom_scenarios import normalize_custom
from evaluation_suite import summarize, cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--custom-file', type=Path, help='JSON array of explicit custom quote inputs')
    parser.add_argument('--case', action='append', help='Demo/edge scenario ID; may be repeated')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--seed', type=int, default=110)
    parser.add_argument('--protocol', choices=(PROTOCOL, 'validated-v12-tools', 'validated-v11-adaptive'), default=PROTOCOL)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.summary.json').exists() or args.repeats < 1:
        parser.error('Use new output paths and at least one repetition')
    selected = cases() if not args.case else [s for s in cases() if s['id'] in args.case]
    if args.case and set(args.case) - {s['id'] for s in selected}:
        parser.error('Unknown scenario ID')
    if args.custom_file:
        selected += [normalize_custom(p) for p in json.loads(args.custom_file.read_text(encoding='utf-8'))]
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for repeat in range(args.repeats):
        for scenario in selected:
            print(f"{scenario['id']} repeat={repeat + 1}", flush=True)
            result = run_pair(scenario, protocol=args.protocol, seed=args.seed + repeat, compressed_first=bool(repeat % 2))
            records.append(result)
            args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
            print(json.dumps({'status': result['status'], 'error': result.get('error'),
                              'comparison': result.get('comparison')}), flush=True)
    summary = {'protocol': args.protocol, 'phase': 'evaluate' if args.repeats >= 2 else 'pilot',
               'created_at': datetime.now(timezone.utc).isoformat(), 'models': summarize(records),
               'scope': 'Local Llama 3 quotation scenarios. Three branches measured; all supplied terms reach '
                        'Risk and Decision. Savings include compression, routing and output variation. '
                        'Arbitrary narrative constraints are not certified; there is no guaranteed minimum saving.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
