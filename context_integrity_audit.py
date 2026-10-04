"""Audit saved paired experiments for exact source-backed context routing.

This checks factual provenance and explicit coverage. It does not certify
semantic equivalence of omitted supplier prose.
"""
import argparse
import json
from pathlib import Path

from claim_guard import caveat_catalog
from compact_projection import render_review_context
from lossless_projection import render_lossless_context, render_verified_lossless_context
from paired_benchmark import source_text
from protected_facts import ProtectedFacts
from quote_calculations import calculate_quotes


def audit_experiment(experiment):
    if experiment.get('status') == 'needs_review':
        return {'experiment_id': experiment['experiment_id'], 'scenario_id': experiment['scenario']['id'],
                'protocol': experiment['configuration']['pipeline_version'], 'status': 'needs_review',
                'all_checks_passed': False, 'scope': 'No inference or decision was validated; requirements need review.'}
    if experiment['configuration']['pipeline_version'] in ('validated-v11-adaptive', 'validated-v12-tools', 'validated-v13-requirements'):
        return audit_adaptive(experiment)
    scenario = experiment['scenario']
    record = ProtectedFacts.from_scenario(scenario)
    python_audit = calculate_quotes(record)
    branch = experiment['compressed']
    state = branch['langgraph_state']
    applied = branch['compression']['applied']
    catalog_rows = caveat_catalog(scenario)
    catalog = {row['id']: f"{row['quote_id']}: {row['detail']}" for row in catalog_rows}
    selected = branch['compression'].get('selected_caveat_ids') or []
    protocol = experiment['configuration']['pipeline_version']
    lossless = protocol in (
        'validated-v7-lossless', 'validated-v8-lossless',
        'validated-v9-lossless-verified', 'validated-v10-lossless-verified')
    all_agents = protocol in ('validated-v8-lossless', 'validated-v9-lossless-verified',
                              'validated-v10-lossless-verified')
    eligible = set(python_audit['eligible_ids'])
    required = {row['id'] for row in catalog_rows
                if not eligible or row['quote_id'] in eligible}
    source = source_text(scenario)
    if applied:
        if protocol in ('validated-v9-lossless-verified', 'validated-v10-lossless-verified'):
            exact_projection = render_verified_lossless_context(record, python_audit)
        elif lossless:
            exact_projection = render_lossless_context(record)
        elif not all(item in catalog for item in selected):
            exact_projection = None
        elif experiment['configuration']['pipeline_version'] == 'validated-v6-projection':
            exact_projection = render_review_context(record, python_audit, catalog, selected)
        else:
            exact_projection = (
                record.compact_text() + '\nSOURCE CAVEATS:\n'
                + '\n'.join(catalog[item] for item in selected))
    else:
        exact_projection = source
    context = state['active_context']
    downstream = ('Logistics', 'Finance', 'Risk', 'Compliance', 'Decision') if all_agents else (
        'Logistics', 'Finance', 'Risk', 'Decision')
    by_role = {call['role']: call for call in branch['calls']}
    checked_calls = [call for call in branch['calls'] if 'claim_validation' in call]
    checks = {
        'source_unchanged': state['source_context'] == source,
        'protected_record_unchanged': (
            state['protected_facts'] == record.to_dict()
            == experiment['shared']['protected_facts']),
        'python_audit_unchanged': (
            python_audit == experiment['shared']['constraint_audit']),
        'context_exactly_reconstructed_from_source': context == exact_projection,
        'all_required_caveats_selected': (
            not applied or all(row['detail'] in context for row in catalog_rows
                               if row['id'] in required)
            if lossless else not applied or required.issubset(selected)),
        'selected_caveats_are_source_exact': (
            not applied or all(row['detail'] in context for row in catalog_rows)
            if lossless else not applied or all(catalog[item] in context for item in selected
                               if item in catalog)),
        'downstream_prompts_use_active_context': all(
            context in by_role[role]['input'] for role in downstream),
        'ingestion_prompt_uses_full_source': (
            not all_agents or source in experiment['shared']['calls'][0]['input']),
        'python_checks_in_compressed_context': (
            protocol not in ('validated-v9-lossless-verified', 'validated-v10-lossless-verified') or not applied or
            all(f"{row['quote_id']}|" in context and all(
                f"{name}={'P' if row[name] else 'F'}" in context
                for name in ('within_budget', 'on_time',
                             'enough_capacity', 'insured_as_required'))
                for row in python_audit['rows'])),
        'agent_reports_forward_only_guarded_text': all(
            call['forwarded_output'] == call['claim_validation']['safe_text']
            for call in checked_calls),
        'both_final_decisions_validated': experiment['comparison']['both_checklists_passed'],
    }
    omitted = [row for row in catalog_rows if row['detail'] not in context] if applied else []
    return {
        'experiment_id': experiment['experiment_id'],
        'scenario_id': scenario['id'],
        'protocol': experiment['configuration']['pipeline_version'],
        'compression_applied': applied,
        'saved_tokens': experiment['comparison']['saved_tokens'],
        'reduction_percent': experiment['comparison']['reduction_percent'],
        'checks': checks,
        'all_checks_passed': all(checks.values()),
        'selected_caveat_ids': selected,
        'omitted_catalog_caveats': omitted,
        'full_brief_in_active_context': record.brief in context,
        'carrier_and_method_names_in_active_context': all(
            quote.carrier in context and quote.method in context
            for quote in record.quotes),
        'every_eligible_full_terms_in_active_context': all(
            quote.terms in context for quote in record.quotes if quote.id in eligible),
        'every_full_terms_in_active_context': all(
            quote.terms in context for quote in record.quotes),
        'scope': (
            'Every original source field and supplier term copied exactly; prompt routing, '
            'guarded claims and final supplied-fact checks verified. Generated prose quality '
            'is not certified.' if lossless else
            'Exact source-backed structured facts, selected caveats, prompt routing, '
            'guarded claims, and final supplied-fact checks. Omitted prose is not '
            'certified semantically equivalent.'),
    }


def audit_adaptive(experiment):
    from adaptive_context import build_plan, python_report
    if experiment['configuration']['pipeline_version'] in ('validated-v12-tools', 'validated-v13-requirements'):
        from tool_routing import build_plan
    from claim_guard import guard_report, decision_reason
    from paired_benchmark import evaluate_decision
    scenario = experiment['scenario']
    record = ProtectedFacts.from_scenario(scenario)
    audit = calculate_quotes(record)
    source = source_text(scenario)
    plan = build_plan(record, audit, source)
    checks = {'plan_matches_source': plan == experiment['configuration']['adaptive_plan'],
              'shared_source_and_audit_intact': source + '\n' + audit['text'] == experiment['shared']['context'],
              'shared_protected_record_intact': record.to_dict() == experiment['shared']['protected_facts'],
              'shared_calculations_intact': audit == experiment['shared']['constraint_audit'],
              'ingestion_prompt_intact': len(experiment['shared']['calls']) == 1 and
                  experiment['shared']['calls'][0]['role'] == 'Ingestion' and
                  experiment['shared']['calls'][0]['input'] == source + '\n' + audit['text']}
    if experiment['configuration']['pipeline_version'] == 'validated-v13-requirements':
        from requirement_review import assess_requirements
        review = assess_requirements(scenario)
        checks['requirements_ready_and_recorded'] = (review['status'] == 'ready' and
            experiment.get('requirement_validation') == review)
    roles = ('Logistics', 'Finance', 'Risk', 'Compliance', 'Decision')
    for mode in ('baseline', 'compression_only', 'compressed'):
        branch = experiment[mode]
        state = branch['langgraph_state']
        applied = bool(branch.get('compression') and branch['compression']['applied'])
        routed = mode == 'compressed' and applied
        context = plan['full_context'] if applied else source
        calls = {call['role']: call for call in branch['calls']}
        expected_roles = set(roles) - (set(plan['skipped_roles']) if routed else set())
        checks[f'{mode}_roles'] = set(calls) == expected_roles
        checks[f'{mode}_source_intact'] = state['source_context'] == source and state['protected_facts'] == record.to_dict()
        checks[f'{mode}_active_context'] = state['active_context'] == context
        checks[f'{mode}_prompt_coverage'] = all(
            (plan['role_contexts'].get(role, context) if routed else context) in calls.get(role, {}).get('input', '')
            for role in expected_roles)
        checks[f'{mode}_full_terms_in_risk_and_decision'] = all(
            record.brief in calls.get(role, {}).get('input', '') and all(
                quote.terms in calls.get(role, {}).get('input', '') for quote in record.quotes)
            for role in ('Risk', 'Decision'))
        checks[f'{mode}_guarded_handoff'] = all(
            call['forwarded_output'] == guard_report(call['output'], call['role'], audit,
                caveat_catalog(scenario), allow_terminal_period_omission=True)['safe_text']
            for call in experiment['shared']['calls'] + branch['calls'] if 'claim_validation' in call)
        safe_reports = {}
        for role in roles[:-1]:
            if routed and role in plan['skipped_roles']:
                safe_reports[role] = python_report(role, audit)
            else:
                safe_reports[role] = guard_report(calls.get(role, {}).get('output', ''), role, audit,
                    caveat_catalog(scenario), allow_terminal_period_omission=True)['safe_text']
        checks[f'{mode}_reports_match_validated_outputs'] = all(
            branch['reports'].get(role) == safe for role, safe in safe_reports.items())
        suffix = '' if applied else '\n' + audit['text']
        expected_inputs = {
            role: (plan['role_contexts'].get(role, context) if routed else context) + suffix
            for role in ('Logistics', 'Finance', 'Compliance')}
        compact_handoff = routed and plan.get('compact_handoff', False)
        full = context
        for role in ('Logistics', 'Finance'):
            if not (compact_handoff and role in plan['skipped_roles']):
                full += '\n' + role.upper() + ':\n' + safe_reports[role]
        expected_inputs['Risk'] = full + suffix
        compliance = '' if compact_handoff and 'Compliance' in plan['skipped_roles'] else (
            '\nCOMPLIANCE VERIFIED:\n' + safe_reports['Compliance'])
        expected_inputs['Decision'] = (full + '\nRISK VERIFIED:\n' + safe_reports['Risk']
            + compliance + suffix + '\nPYTHON FIELDS:\n' + json.dumps(audit['recommendation'])
            + '\nVERIFIED REASON:\n' + decision_reason(audit))
        checks[f'{mode}_prompts_exactly_reconstructed'] = all(
            calls.get(role, {}).get('input') == expected_inputs[role] for role in expected_roles)
        if routed and plan['skipped_roles']:
            checks['python_replacement_reports_exact'] = all(
                branch['reports'][role] == python_report(role, audit) for role in plan['skipped_roles'])
            if plan.get('compact_handoff'):
                checks['python_checks_reach_decision'] = plan['full_context'] in calls['Decision']['input']
            else:
                checks['python_reports_reach_decision'] = all(
                    python_report(role, audit) in calls['Decision']['input'] for role in plan['skipped_roles'])
        final = evaluate_decision(calls.get('Decision', {}).get('output', ''), scenario)
        checks[f'{mode}_decision'] = (final['all_passed'] and
            final['decision'].get('reasoning') == decision_reason(audit) and
            final['decision'].get('conditions') == [])
    return {'experiment_id': experiment['experiment_id'], 'scenario_id': scenario['id'],
            'protocol': experiment['configuration']['pipeline_version'],
            'checks': checks, 'all_checks_passed': all(checks.values()),
            'saved_tokens': experiment['comparison']['saved_tokens'],
            'reduction_percent': experiment['comparison']['reduction_percent'],
            'scope': plan['scope']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('paths', nargs='+', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    reports = []
    for path in args.paths:
        payload = json.loads(path.read_text(encoding='utf-8'))
        for experiment in payload if isinstance(payload, list) else [payload]:
            reports.append({'file': str(path), **audit_experiment(experiment)})
    text = json.dumps(reports, indent=2)
    if args.output:
        args.output.write_text(text, encoding='utf-8')
    print(text)
    return 0 if all(report['all_checks_passed'] for report in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
