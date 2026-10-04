"""Reproducible, sequential multi-model pilots and held-out paired evaluation."""
import argparse
import copy
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import statistics

from graph_selective_workflow import run_pair, scenarios, PROTOCOL, ROOT
from measured_policy import fit_profile

MODELS = ['llama3', 'phi3', 'qwen2.5:7b']


def cases():
    original = scenarios()
    no_offer = copy.deepcopy(original[0])
    no_offer.update(id='eval_no_eligible', title='No eligible quotation', budget_usd=100)
    no_offer['brief'] = 'Select the lowest-cost eligible offer, or hold. The authoritative budget is USD 100.'
    tie = copy.deepcopy(original[0])
    tie.update(id='eval_tie', title='Equal-price eligible offers')
    tie['options'][1] = dict(tie['options'][0], id='S0', carrier='Tie Carrier')
    return original + [no_offer, tie]


def summarize(records):
    output = {}
    for model in MODELS:
        runs = [r for r in records if r['configuration']['model'] == model]
        complete = [r for r in runs if 'comparison' in r]
        applied = [r for r in complete if r['comparison']['compression_applied']]
        branch_names = lambda r: ('baseline', 'compression_only', 'compressed') if r['configuration'].get('pipeline_version') in ('validated-v11-adaptive', 'validated-v12-tools', 'validated-v13-requirements') else ('baseline', 'compressed')
        branches = [r[m] for r in complete for m in branch_names(r)]
        planned_decisions = sum(len(branch_names(r)) for r in runs)
        savings = [r['comparison']['reduction_percent'] for r in applied]
        claim_total = sum(b['claim_metrics']['claims_total'] for b in branches)
        inferred = [r for r in runs if r['status'] != 'needs_review']
        # Shared claims are charged per deployment branch, consistently with token accounting.
        output[model] = {
            'attempted_pairs': len(runs), 'completed_pairs': len(complete),
            'execution_failures': sum(r['status'] == 'execution_failed' for r in runs),
            'needs_review_pairs': sum(r['status'] == 'needs_review' for r in runs),
            'failed_validation_pairs': sum(r['status'] == 'failed_validation' for r in complete),
            'compression_applied_pairs': len(applied), 'skipped_pairs': len(complete) - len(applied),
            'validated_decisions': sum(b['status'] == 'validated' for b in branches), 'total_decisions': len(branches),
            'decision_accuracy': sum(b['status'] == 'validated' for b in branches) / planned_decisions if runs else None,
            'attempted_decisions': planned_decisions,
            'completed_decision_accuracy': sum(b['status'] == 'validated' for b in branches) / len(branches) if branches else None,
            'factual_claim_accuracy': sum(b['claim_metrics']['claims_correct'] for b in branches) / claim_total if claim_total else None,
            'claims_evaluated': claim_total, 'schema_failures': sum(b['claim_metrics']['schema_failures'] for b in branches),
            'rejected_items': sum(b['claim_metrics']['rejected_items'] for b in branches),
            'mean_applied_reduction_percent': statistics.mean(savings) if savings else None,
            'sd_applied_reduction_percent': statistics.stdev(savings) if len(savings) > 1 else None,
            'min_applied_reduction_percent': min(savings) if savings else None,
            'negative_savings_pairs': sum(r['comparison']['saved_tokens'] < 0 for r in applied),
            'quality_preserving_savings_pairs': sum(r['comparison']['quality_preserving_savings'] for r in applied),
            'measured_physical_tokens': sum(r['comparison']['total_experiment_tokens'] if 'comparison' in r else r['measured_tokens'] for r in runs),
            'mean_wall_seconds': statistics.mean(r['comparison']['experiment_wall_seconds'] if 'comparison' in r else r['experiment_wall_seconds'] for r in inferred) if inferred else None,
            'prediction_mae_tokens': statistics.mean(abs(r['comparison']['prediction_error_tokens']) for r in applied if r['comparison']['prediction_error_tokens'] is not None) if any(r['comparison']['prediction_error_tokens'] is not None for r in applied) else None,
        }
    return output


def main():
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logging.getLogger('httpx').setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=('pilot', 'evaluate', 'stress'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 2:
        parser.error('Use a new output path and at least two repetitions')
    records = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(model, scenarios()[2], 31, False) for model in MODELS] if args.phase == 'pilot' else [
        (model, scenario, 42 + repeat, bool(repeat % 2))
        for repeat in range(args.repeats) for scenario in cases()
        for model in (MODELS[repeat % 3:] + MODELS[:repeat % 3])]
    if args.phase == 'stress':
        jobs = [(model, scenarios()[0], 80, False) for model in MODELS]
    for index, (model, scenario, seed, reverse) in enumerate(jobs):
        print(f"{index+1}/{len(jobs)} {model} {scenario['id']} seed={seed}", flush=True)
        result = run_pair(scenario, model=model, seed=seed, compressed_first=reverse,
                          compression_mode='always' if args.phase in ('pilot', 'stress') else 'auto',
                          profile={} if args.phase in ('pilot', 'stress') else None)
        result['evaluation_phase'] = args.phase
        records.append(result)
        args.output.write_text(json.dumps(records, indent=2), encoding='utf-8')
        print(json.dumps({'status': result['status'], 'error': result.get('error'),
                          'comparison': result.get('comparison')}), flush=True)
    if args.phase == 'pilot':
        calibration = {m: fit_profile(records, m, PROTOCOL) for m in MODELS}
        (ROOT / 'verification' / 'compression-calibration-graph.json').write_text(json.dumps(calibration, indent=2), encoding='utf-8')
    summary = {'protocol': PROTOCOL, 'phase': args.phase, 'created_at': datetime.now(timezone.utc).isoformat(),
               'models': summarize(records), 'scope': 'Small local fictional-quote sample. Claim tuple accuracy excludes unverified prose; rejected prose is blocked. Failed and negative runs retained.'}
    args.output.with_suffix('.summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
