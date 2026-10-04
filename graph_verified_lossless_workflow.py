"""Lossless source projection with one compact authoritative Python audit."""
from datetime import datetime, timezone
import hashlib
import copy
import json
import logging
from pathlib import Path
import threading
import time
import uuid

from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from paired_benchmark import source_text, evaluate_decision, scenarios, MODEL
from protected_facts import ProtectedFacts
from quote_calculations import calculate_quotes
from structured_claims import SCHEMA, instructions
from claim_guard import caveat_catalog, guard_report, decision_reason
from langgraph_orchestration import run_branch
from lossless_projection import render_verified_lossless_context

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'validated-v10-lossless-verified'
LIMITS = {'Ingestion': 256, 'Logistics': 384, 'Finance': 384, 'Risk': 384,
          'Compliance': 384, 'Compressor': 192, 'Decision': 384}


class Runner:
    def __init__(self, model, seed, context_window=8192, preflight=False):
        self.model, self.seed = model, seed
        self.context_window, self.preflight = context_window, preflight
        self.calls, self.lock = [], threading.Lock()

    def call(self, role, system, text, schema):
        start = time.perf_counter()
        call = {'role': role, 'system_prompt': system, 'input': text, 'output': '',
                'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'settings': {'model': self.model, 'seed': self.seed, 'temperature': 0,
                             'num_ctx': self.context_window, 'max_output_tokens': LIMITS[role]},
                'input_tokens': None, 'output_tokens': None, 'total_tokens': None}
        try:
            # UTF-8 bytes give a conservative upper bound for this Llama
            # tokenizer, with headroom for chat delimiters and output.
            if self.preflight and len((system + text).encode('utf-8')) + LIMITS[role] + 512 >= self.context_window:
                raise ValueError('Prompt exceeds conservative context-window bound; shorten the supplied documents')
            response = ChatOllama(model=self.model, seed=self.seed, temperature=0, num_ctx=self.context_window,
                                  num_predict=LIMITS[role], format=schema).invoke(
                                      [SystemMessage(content=system), HumanMessage(content=text)])
            meta = response.response_metadata
            call.update(output=response.content, raw_output=response.content, finish_reason=meta.get('done_reason'))
            for key in ('prompt_eval_count', 'eval_count'):
                if type(meta.get(key)) is not int or meta[key] < 0:
                    raise ValueError('Missing measured token usage')
            call.update(input_tokens=meta['prompt_eval_count'], output_tokens=meta['eval_count'],
                        total_tokens=meta['prompt_eval_count'] + meta['eval_count'])
            if not isinstance(response.content, str) or not response.content.strip():
                raise ValueError('Empty model response')
            if meta['prompt_eval_count'] + LIMITS[role] >= self.context_window:
                raise ValueError('Context window safety margin exceeded')
        except Exception as exc:
            call['error'] = str(exc)
            raise
        finally:
            call['seconds'] = time.perf_counter() - start
            with self.lock:
                self.calls.append(call)
            logging.info('%s %s: %s tokens, %.1fs', self.model, role, call.get('total_tokens'), call['seconds'])
        return call


def sum_tokens(calls):
    return sum(c['total_tokens'] for c in calls if c.get('total_tokens') is not None)


def run_pair(scenario, model=MODEL, seed=42, target_ratio=.28, compressed_first=False,
             runner=None, compression_mode='auto', profile=None, compressor_strategy='python_only',
             adaptive=False, adaptive_protocol='validated-v11-adaptive'):
    if compression_mode not in ('auto', 'always', 'never'):
        raise ValueError('Invalid compression mode')
    if compressor_strategy != 'python_only':
        raise ValueError('Lossless protocol requires deterministic source projection')
    if adaptive and adaptive_protocol not in ('validated-v11-adaptive', 'validated-v12-tools', 'validated-v13-requirements'):
        raise ValueError('Unknown adaptive protocol')
    requirement_validation = None
    if adaptive and adaptive_protocol == 'validated-v13-requirements':
        from requirement_review import assess_requirements, blocked_result
        requirement_validation = assess_requirements(scenario)
        if requirement_validation['status'] != 'ready':
            return blocked_result(scenario, requirement_validation, adaptive_protocol, model, seed)
    started = time.perf_counter()
    runner = runner or Runner(model, seed, context_window=16384 if adaptive else 8192, preflight=adaptive)
    record = ProtectedFacts.from_scenario(scenario)
    audit = calculate_quotes(record)
    caveats = caveat_catalog(scenario)
    catalog = {c['id']: f"{c['quote_id']}: {c['detail']}" for c in caveats}
    # Every caveat for a feasible offer is mandatory. Ineligible offers are
    # explained by the protected numeric/boolean audit; their narrative
    # caveats are optional. With no feasible offer, preserve all caveats.
    eligible = set(audit['eligible_ids'])
    required_ids = [c['id'] for c in caveats if not eligible or c['quote_id'] in eligible]
    optional_limit = min(3, len(catalog) - len(required_ids))
    protocol = PROTOCOL
    branch_order = ['compressed', 'baseline'] if compressed_first else ['baseline', 'compressed']
    if adaptive:
        protocol = adaptive_protocol
        branch_order = ['compressed', 'compression_only', 'baseline'] if compressed_first else ['baseline', 'compression_only', 'compressed']
    result = {'schema_version': 1, 'experiment_id': str(uuid.uuid4()),
              'created_at': datetime.now(timezone.utc).isoformat(), 'scenario': scenario,
              'configuration': {'pipeline_version': protocol, 'model': model, 'seed': seed,
                                'temperature': 0, 'num_ctx': 16384 if adaptive else 8192, 'limits': LIMITS,
                                'compression_mode': compression_mode,
                                'compressor_strategy': compressor_strategy,
                                'branch_order': branch_order,
                                'validation_scope': 'All source fields and supplier terms copied exactly into the downstream projection; Python-supported claims and final decision checked. Free-form agent prose quality is not certified.'}}

    if requirement_validation is not None:
        result['requirement_validation'] = requirement_validation

    def agent(role, context):
        schema = copy.deepcopy(SCHEMA)
        schema['properties']['claims']['maxItems'] = 2
        schema['properties']['uncertainties']['maxItems'] = 2
        example_id = record.quotes[0].id
        example_check = 'within_budget'
        example_result = 'PASS' if audit['rows'][0][example_check] else 'FAIL'
        system = (f'You are the {role} quotation reviewer. Return at most TWO relevant claims. '
                  f'For example, quote_id="{example_id}", constraint="{example_check}" requires explanation="{example_id}: {example_check}={example_result}." '
                  'Substitute the actual quote ID, constraint and uppercase result from each claim. Never output placeholder names. '
                  'Use only Python comparisons. Do not add numeric prose. Prefer uncertainties=[]; '
                  'otherwise copy a protected source caveat exactly.' + instructions(role))
        call = runner.call(role, system, context, schema)
        guard = guard_report(call['output'], role, audit, caveats,
                             allow_terminal_period_omission=True)
        call['claim_validation'] = guard
        call['forwarded_output'] = guard['safe_text']
        return call

    try:
        shared_started = time.perf_counter()
        source = source_text(scenario)
        authoritative = audit['text']
        ingestion = agent('Ingestion', source + '\n' + authoritative)
        upstream = [ingestion]
        shared_hash = hashlib.sha256((source + '\n' + authoritative).encode()).hexdigest()
        result['shared'] = {'calls': upstream, 'reports': {'Ingestion': ingestion['forwarded_output']},
                            'context': source + '\n' + authoritative,
                            'context_sha256': shared_hash, 'protected_facts': record.to_dict(),
                            'constraint_audit': audit, 'protected_caveats': caveats,
                            'token_usage': sum_tokens(upstream), 'seconds': time.perf_counter() - shared_started}
        # The strict projection includes the entire brief and every quote term.
        # This removes formatting repetition without dropping source semantics.
        compressor_input = json.dumps(catalog, separators=(',', ':'))
        compact_renderer = lambda _ids: render_verified_lossless_context(record, audit)
        compact = compact_renderer([])
        adaptation = None
        if adaptive:
            if protocol in ('validated-v12-tools', 'validated-v13-requirements'):
                from tool_routing import build_plan
            else:
                from adaptive_context import build_plan
            adaptation = build_plan(record, audit, source)
            result['configuration']['validation_scope'] = adaptation['scope']
            result['configuration']['adaptive_plan'] = adaptation
        selection_schema = {'type': 'object', 'additionalProperties': False, 'required': ['keep'],
                            'properties': {'keep': {'type': 'array', 'minItems': len(required_ids),
                                                    'maxItems': len(required_ids) + optional_limit, 'uniqueItems': True,
                                                    'items': {'type': 'string', 'enum': list(catalog) or ['NONE']}}}}
        canonical_reason = decision_reason(audit)
        for mode in result['configuration']['branch_order']:
            branch_start = time.perf_counter()
            state = run_branch(
                mode='baseline' if mode == 'baseline' else 'compressed', source=source, record=record, audit=audit,
                caveats=caveats, catalog=catalog, required_ids=required_ids,
                optional_limit=optional_limit, compressor_input=compressor_input,
                compact=compact, selection_schema=selection_schema,
                canonical_reason=canonical_reason, model=model, protocol=protocol,
                compression_mode=compression_mode,
                compressor_strategy=compressor_strategy, profile=profile,
                runner=runner, agent=agent, scenario=scenario, ingestion=ingestion,
                full_source_for_all_agents=True,
                compact_renderer=compact_renderer,
                adaptation=adaptation if mode == 'compressed' else None)
            order = ('Compressor', 'Logistics', 'Finance', 'Risk', 'Compliance', 'Decision')
            calls = sorted(state['calls'], key=lambda call: order.index(call['role']))
            context, compression = state['active_context'], state['compression']
            logistics, finance = state['logistics_call'], state['finance_call']
            risk, compliance = state['risk_call'], state['compliance_call']
            decision, evaluation = state['decision_call'], state['decision_evaluation']
            reports = {'Ingestion': ingestion['forwarded_output'],
                       'Logistics': logistics['forwarded_output'],
                       'Finance': finance['forwarded_output']}
            status = 'validated' if evaluation['all_passed'] else 'failed_validation'
            reports_branch = dict(reports, Risk=risk['forwarded_output'], Compliance=compliance['forwarded_output'],
                                  Decision=decision['output'] if evaluation['all_passed'] else 'FAILED VALIDATION: ' + ', '.join(evaluation['failed_checks']),
                                  Compressor=context if compression and compression['applied'] else '')
            guards = [c['claim_validation'] for c in upstream + calls if 'claim_validation' in c]
            claim_metrics = {'claims_total': sum(g['claims_total'] for g in guards),
                             'claims_correct': sum(g['claims_correct'] for g in guards),
                             'rejected_items': sum(len(g['rejected']) for g in guards),
                             'rejected_reports': sum(not g['valid'] for g in guards),
                             'schema_failures': sum(not g['schema_valid'] for g in guards)}
            result[mode] = {'mode': mode, 'status': status, 'reports': reports_branch, 'calls': calls,
                            'evaluation': evaluation, 'claim_metrics': claim_metrics,
                            'report_validation': {c['role']: c['claim_validation'] for c in upstream + calls if 'claim_validation' in c},
                            'constraint_audit': audit, 'compression': compression,
                            'langgraph_state': {'source_context': state['source_context'],
                                                'active_context': context,
                                                'compressed_context': state['compressed_context'],
                                                'protected_facts': state['protected_facts'],
                                                'raw_history': state['raw_history']},
                            'shared_context_sha256': shared_hash, 'token_usage': sum_tokens(upstream + calls),
                            'execution_time_seconds': result['shared']['seconds'] + time.perf_counter() - branch_start}
        base, comp = result['baseline'], result['compressed']
        by_role = lambda b, role: next((c for c in b['calls'] if c['role'] == role), {'input_tokens': 0, 'output_tokens': 0})
        net = base['token_usage'] - comp['token_usage']
        cost = sum(c['total_tokens'] for c in comp['calls'] if c['role'] == 'Compressor')
        compared_roles = ('Logistics', 'Finance', 'Risk', 'Compliance', 'Decision')
        input_saved = sum(by_role(base, role)['input_tokens'] - by_role(comp, role)['input_tokens'] for role in compared_roles)
        output_saved = sum(by_role(base, role)['output_tokens'] - by_role(comp, role)['output_tokens'] for role in compared_roles)
        passed = base['evaluation']['all_passed'] and comp['evaluation']['all_passed']
        compression = comp['compression']
        result['status'] = 'validated' if passed else 'failed_validation'
        prediction = compression['estimated_net_tokens_saved']
        result['comparison'] = {'saved_tokens': net, 'reduction_percent': 100 * net / base['token_usage'],
            'compressor_tokens': cost, 'downstream_input_tokens_saved': input_saved, 'downstream_output_tokens_saved': output_saved,
            'review_input_reduction_percent': 100 * (by_role(base, 'Risk')['input_tokens'] - by_role(comp, 'Risk')['input_tokens']) / by_role(base, 'Risk')['input_tokens'],
            'downstream_input_reduction_percent': 100 * input_saved / sum(by_role(base, role)['input_tokens'] for role in compared_roles),
            'both_checklists_passed': passed, 'compression_applied': compression['applied'],
            'difference_attribution': 'paired_compression_and_output_variation' if compression['applied'] else
                                      'rejected_compression_cost_and_output_variation' if compression['compressor_called'] else
                                      'run_to_run_variation_no_compression',
            'quality_preserving_savings': bool(passed and compression['applied'] and net > 0),
            'predicted_net_tokens_saved': prediction,
            'prediction_error_tokens': net - prediction if prediction is not None and compression['applied'] else None,
            'total_experiment_tokens': sum_tokens(upstream + base['calls'] + comp['calls']),
            'experiment_wall_seconds': time.perf_counter() - started,
            'truncated_calls': [c['role'] for c in upstream + base['calls'] + comp['calls'] if c.get('finish_reason') == 'length'],
            'accounting': 'All measured input and output charged, including rejected reports and compressor calls. Shared cost charged to each branch; physically executed once.'}
        if adaptive:
            ablation = result['compression_only']
            applied = compression['applied']
            compression_gain = base['token_usage'] - ablation['token_usage']
            routing_gain = ablation['token_usage'] - comp['token_usage']
            passed = passed and ablation['evaluation']['all_passed']
            result['status'] = 'validated' if passed else 'failed_validation'
            result['comparison'].update(
                all_branches_validated=passed,
                compression_only_saved_tokens=compression_gain,
                compression_only_reduction_percent=100 * compression_gain / base['token_usage'],
                routing_saved_tokens=routing_gain,
                routing_input_tokens_saved=sum(by_role(ablation, r)['input_tokens'] - by_role(comp, r)['input_tokens'] for r in compared_roles),
                routing_output_tokens_saved=sum(by_role(ablation, r)['output_tokens'] - by_role(comp, r)['output_tokens'] for r in compared_roles),
                compression_input_tokens_saved=sum(by_role(base, r)['input_tokens'] - by_role(ablation, r)['input_tokens'] for r in compared_roles),
                compression_output_tokens_saved=sum(by_role(base, r)['output_tokens'] - by_role(ablation, r)['output_tokens'] for r in compared_roles),
                adaptive_path=adaptation['path'] if applied else 'disabled',
                skipped_llm_roles=adaptation['skipped_roles'] if applied else [],
                difference_attribution='compression_routing_and_output_variation' if applied else 'run_to_run_variation_no_compression',
                quality_preserving_savings=bool(passed and applied and net > 0),
                total_experiment_tokens=sum_tokens(upstream + base['calls'] + ablation['calls'] + comp['calls']),
                truncated_calls=[c['role'] for c in upstream + base['calls'] + ablation['calls'] + comp['calls'] if c.get('finish_reason') == 'length'],
                accounting='Three measured branches: fixed six-agent baseline, compression only, and adaptive. '
                           'Shared Ingestion charged to each deployment branch, executed once. '
                           'All three branches are included in physical experiment cost. '
                           'Attribution differences include measured output variation; skipping LLM roles is not compressor-only saving.')
            from context_integrity_audit import audit_experiment
            result['context_audit'] = audit_experiment(result)
            if not result['context_audit']['all_checks_passed']:
                result['status'] = 'failed_validation'
                result['compressed']['status'] = 'failed_validation'
                result['compressed']['reports']['Decision'] = 'FAILED VALIDATION: context integrity audit failed'
                result['comparison']['quality_preserving_savings'] = False
        return result
    except Exception as exc:
        result.update(status='execution_failed', error=str(exc), calls=runner.calls,
                      measured_tokens=sum_tokens(runner.calls), usage_complete=all(c.get('total_tokens') is not None for c in runner.calls),
                      experiment_wall_seconds=time.perf_counter() - started)
        return result

