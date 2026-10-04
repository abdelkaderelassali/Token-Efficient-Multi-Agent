"""Model comparison benchmark.

Runs one paired experiment per model on an identical scenario, with the same
seed, temperature, routing and output-limit table. Saves results to a JSON
file that the evaluation endpoint and dashboard can load.

Usage
-----
  python benchmark_models.py --scenario detailed --repeats 2 --output verification/model-bench.json
  python benchmark_models.py --scenario all     --repeats 2 --output verification/model-bench.json
  python benchmark_models.py --scenario detailed --models llama3,phi3 --output verification/bench.json

Notes
-----
* All three models use identical LIMITS from validated_workflow.
* Branch order alternates each repeat so order bias averages out.
* The same compression mode is used for every model and repeat. The default
  'always' measures compressor behavior directly; 'auto' evaluates the policy.
* Failed runs are retained in the output so denominators are correct.
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from evaluation_suite import summarize, cases, MODELS
from graph_selective_workflow import run_pair, scenarios, PROTOCOL, ROOT


def run_benchmark(model_list, scenario_list, repeats, output_path, compression_mode):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        datefmt='%H:%M:%S')
    logging.getLogger('httpx').setLevel(logging.WARNING)

    records = []
    jobs = [
        (model, scenario, 42 + repeat, bool(repeat % 2), compression_mode)
        for repeat in range(repeats)
        for scenario in scenario_list
        for model in (model_list[repeat % len(model_list):] + model_list[:repeat % len(model_list)])
    ]

    print(f"Running {len(jobs)} paired experiments  "
          f"({len(model_list)} models x {len(scenario_list)} scenarios x {repeats} repeats)")
    print("Identical settings: seed per repeat, temperature=0, num_ctx=8192, output limits from LIMITS dict")
    print()

    for index, (model, scenario, seed, reverse, cmode) in enumerate(jobs):
        print(f"[{index+1}/{len(jobs)}] model={model}  scenario={scenario['id']}  "
              f"seed={seed}  branch_order={'compressed_first' if reverse else 'baseline_first'}  "
              f"compression={cmode}", flush=True)
        result = run_pair(scenario, model=model, seed=seed, compressed_first=reverse,
                          compression_mode=cmode, profile={} if cmode == 'always' else None)
        result['evaluation_phase'] = 'benchmark'
        records.append(result)
        output_path.write_text(json.dumps(records, indent=2), encoding='utf-8')

        comp = result.get('comparison') or {}
        wall = comp.get('experiment_wall_seconds', result.get('experiment_wall_seconds', '?'))
        wall_str = f'{wall:.0f}s' if isinstance(wall, (int, float)) else '?'
        print(f"  -> status={result['status']}  "
              f"saved={comp.get('saved_tokens', '?')} tokens  "
              f"attribution={comp.get('difference_attribution', '?')}  "
              f"wall={wall_str}", flush=True)
    return records


def print_table(summary):
    print()
    print(f"{'Model':<20} {'Attempts':>8} {'Validated':>10} {'Decision%':>10} "
          f"{'Claim%':>8} {'Applied':>8} {'Negatives':>10} {'Mean+/-SD%':>14} {'Time':>8}")
    print("-" * 110)
    for model, s in summary.items():
        neg = s.get('negative_savings_pairs', 0)
        mean = s['mean_applied_reduction_percent']
        sd   = s.get('sd_applied_reduction_percent')
        acc  = s['decision_accuracy']
        claim = s['factual_claim_accuracy']
        t    = s['mean_wall_seconds']
        mean_sd = (f"{mean:.1f}%+/-{sd:.1f}%" if sd is not None else
                   f"{mean:.1f}%" if mean is not None else "---")
        acc_str  = f"{acc*100:.0f}%" if acc is not None else "---"
        claim_str = f"{claim*100:.1f}%" if claim is not None else "---"
        time_str = f"{t:.0f}s" if t is not None else "---"
        print(f"{model:<20} {s['attempted_pairs']:>8} {s['validated_decisions']:>10} "
              f"{acc_str:>10} {claim_str:>8} {s['compression_applied_pairs']:>8} "
              f"{neg:>10} {mean_sd:>14} {time_str:>8}")


def main():
    parser = argparse.ArgumentParser(
        description='Fair model comparison: identical scenarios, prompts, routing, output limits.')
    parser.add_argument('--scenario', choices=['short', 'medium', 'detailed', 'all', 'cases'],
                        default='detailed',
                        help='Which scenarios to run. "all" = 3 base; "cases" = all + difficult.')
    parser.add_argument('--models', default=','.join(MODELS),
                        help='Comma-separated model names (default: all three).')
    parser.add_argument('--repeats', type=int, default=2,
                        help='Repetitions per model per scenario (min 2 for SD).')
    parser.add_argument('--compression', choices=['auto', 'always', 'never'], default='always',
                        help='Compression mode for every model and repeat (default: always).')
    parser.add_argument('--output', type=Path, required=True,
                        help='Path to save the raw records JSON.')
    args = parser.parse_args()

    if args.output.exists():
        parser.error(f'Output file already exists: {args.output}. Use a new path.')
    if args.repeats < 2:
        parser.error('Use --repeats >= 2 to get standard deviation and avoid single-run bias.')

    model_list = [m.strip() for m in args.models.split(',') if m.strip()]
    unknown = set(model_list) - set(MODELS)
    if unknown:
        parser.error(f'Unknown model(s): {unknown}. Known: {MODELS}')

    base = scenarios()
    scenario_map = {'short': [base[0]], 'medium': [base[1]], 'detailed': [base[2]],
                    'all': base, 'cases': cases()}
    scenario_list = scenario_map[args.scenario]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    records = run_benchmark(model_list, scenario_list, args.repeats, args.output, args.compression)
    summary = {model: stats for model, stats in summarize(records).items() if model in model_list}
    print_table(summary)

    summary_record = {
        'protocol': PROTOCOL, 'phase': 'benchmark',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'models': summary,
        'scope': (
            'Fair model comparison. Identical scenario, prompt text, routing, output limits. '
            'Branch order alternates each repeat. Compression cost always included. '
            'Failed and negative-saving runs retained in denominators. '
            'Small fictional-quote sample; results are not general model rankings.'
        ),
    }
    summary_path = args.output.with_suffix('.summary.json')
    summary_path.write_text(json.dumps(summary_record, indent=2), encoding='utf-8')
    print(f"\nSummary saved to {summary_path}")
    print(f"Records saved to {args.output}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
