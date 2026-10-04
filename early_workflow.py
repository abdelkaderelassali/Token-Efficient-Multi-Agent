"""Early-compression paired experiment with the same protected-facts guard."""
from concurrent.futures import ThreadPoolExecutor
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
from measured_policy import choose_policy

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'validated-v3-early'
LIMITS = {'Ingestion': 256, 'Logistics': 384, 'Finance': 384, 'Risk': 384,
          'Compliance': 384, 'Compressor': 192, 'Decision': 384}


class Runner:
    def __init__(self, model, seed):
        self.model, self.seed = model, seed
        self.calls, self.lock = [], threading.Lock()

    def call(self, role, system, text, schema):
        start = time.perf_counter()
        call = {'role': role, 'system_prompt': system, 'input': text, 'output': '',
                'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
                'settings': {'model': self.model, 'seed': self.seed, 'temperature': 0,
                             'num_ctx': 8192, 'max_output_tokens': LIMITS[role]},
                'input_tokens': None, 'output_tokens': None, 'total_tokens': None}
        try:
            response = ChatOllama(model=self.model, seed=self.seed, temperature=0, num_ctx=8192,
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
            if meta['prompt_eval_count'] + LIMITS[role] >= 8192:
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
             runner=None, compression_mode='auto', profile=None, compressor_strategy='llm'):
    if compression_mode not in ('auto', 'always', 'never'):
        raise ValueError('Invalid compression mode')
    if compressor_strategy not in ('llm', 'python_only'):
        raise ValueError('Invalid compressor strategy')
    started = time.perf_counter()
    runner = runner or Runner(model, seed)
    record = ProtectedFacts.from_scenario(scenario)
    audit = calculate_quotes(record)
    caveats = caveat_catalog(scenario)
    catalog = {c['id']: f"{c['quote_id']}: {c['detail']}" for c in caveats}
    if profile is None:
        path = ROOT / 'verification' / 'compression-calibration-early.json'
        if path.exists():
            profile = json.loads(path.read_text(encoding='utf-8')).get(model)
    result = {'schema_version': 1, 'experiment_id': str(uuid.uuid4()),
              'created_at': datetime.now(timezone.utc).isoformat(), 'scenario': scenario,
              'configuration': {'pipeline_version': PROTOCOL, 'model': model, 'seed': seed,
                                'temperature': 0, 'num_ctx': 8192, 'limits': LIMITS,
                                'compression_mode': compression_mode,
                                'compressor_strategy': compressor_strategy,
                                'branch_order': ['compressed', 'baseline'] if compressed_first else ['baseline', 'compressed'],
                                'validation_scope': 'Exact Python-supported claims and canonical decision reasoning. Original narrative quality is not certified.'}}

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
        guard = guard_report(call['output'], role, audit, caveats)
        call['claim_validation'] = guard
        call['forwarded_output'] = guard['safe_text']
        return call

    try:
        shared_started = time.perf_counter()
        source = source_text(scenario)
        authoritative = audit['text']
        ingestion = agent('Ingestion', record.compact_text() + '\n' + authoritative)
        upstream = [ingestion]
        shared_hash = hashlib.sha256((source + '\n' + authoritative).encode()).hexdigest()
        result['shared'] = {'calls': upstream, 'reports': {'Ingestion': ingestion['forwarded_output']},
                            'context': source + '\n' + authoritative,
                            'context_sha256': shared_hash, 'protected_facts': record.to_dict(),
                            'constraint_audit': audit, 'protected_caveats': caveats,
                            'token_usage': sum_tokens(upstream), 'seconds': time.perf_counter() - shared_started}
        # Only source-backed caveats reach the compressor. Protected numeric facts bypass it.
        compressor_input = json.dumps(catalog, separators=(',', ':'))
        compact = record.compact_text() + '\nSOURCE CAVEATS:\n' + '\n'.join(catalog.values())
        selection_schema = {'type': 'object', 'additionalProperties': False, 'required': ['keep'],
                            'properties': {'keep': {'type': 'array', 'minItems': len(catalog), 'maxItems': len(catalog), 'uniqueItems': True,
                                                    'items': {'type': 'string', 'enum': list(catalog) or ['NONE']}}}}
        canonical_reason = decision_reason(audit)
        for mode in result['configuration']['branch_order']:
            branch_start = time.perf_counter()
            calls, context, compression = [], source, None
            if mode == 'compressed':
                # The compact context is reused by four reviewers. The policy
                # estimates all four opportunities before paying for the LLM.
                # The Python audit is identical in both branches, so only the
                # replaced source text belongs in the savings prediction.
                full_contexts = [source] * 4
                compression = choose_policy(compression_mode, model, PROTOCOL,
                                            full_contexts, compact, compressor_input, profile)
                if compression['should_compress']:
                    if compressor_strategy == 'python_only':
                        context = compact
                        compression.update(applied=True, compressor_called=False,
                                           caveats_preserved=True,
                                           reason='Deterministic Python compact context; LLM compressor omitted')
                    else:
                        call = runner.call('Compressor',
                        'Compress the supplied verified caveat catalog into an ordered list of reference IDs. '
                        'Return JSON {"keep":[...]}. Keep every caveat exactly once; no prose or numeric facts. '
                        'Place constraints and unresolved issues first. Empty catalog means keep=[].',
                        compressor_input, selection_schema)
                        calls.append(call)
                        try:
                            selection = json.loads(call['output'])
                            ids = selection['keep']
                            valid = isinstance(ids, list) and all(isinstance(i, str) for i in ids) and len(ids) == len(catalog) and set(ids) == set(catalog) and set(selection) == {'keep'}
                        except (ValueError, TypeError, KeyError):
                            valid, ids = False, []
                        call['compression_validation'] = {'valid': valid, 'required_caveat_ids': list(catalog)}
                        compression.update(compressor_called=True, caveats_preserved=valid)
                        if valid and (compression_mode == 'always' or len(compact) < len(source)):
                            context = record.compact_text() + '\nSOURCE CAVEATS:\n' + '\n'.join(catalog[i] for i in ids)
                            compression['applied'] = True
                        else:
                            compression['reason'] = 'Full context retained: invalid caveat coverage or no size reduction; compressor cost counted'
            with ThreadPoolExecutor(max_workers=2) as pool:
                lf = pool.submit(agent, 'Logistics', context + '\n' + authoritative)
                ff = pool.submit(agent, 'Finance', context + '\n' + authoritative)
                logistics, finance = lf.result(), ff.result()
            calls.extend([logistics, finance])
            reports = {'Ingestion': ingestion['forwarded_output'],
                       'Logistics': logistics['forwarded_output'],
                       'Finance': finance['forwarded_output']}
            full = context + '\nLOGISTICS:\n' + reports['Logistics'] + '\nFINANCE:\n' + reports['Finance']
            with ThreadPoolExecutor(max_workers=2) as pool:
                rf = pool.submit(agent, 'Risk', full + '\n' + authoritative)
                cf = pool.submit(agent, 'Compliance', record.compact_text() + '\n' + authoritative)
                risk, compliance = rf.result(), cf.result()
            calls.extend([risk, compliance])
            decision_schema = {'type': 'object', 'additionalProperties': False,
                'required': ['verdict', 'option_id', 'total_cost_usd', 'arrival_date', 'quantity', 'budget_usd', 'deadline', 'reasoning', 'conditions'],
                'properties': {'verdict': {'type': 'string', 'enum': ['proceed', 'hold']},
                    'option_id': {'type': ['string', 'null']}, 'total_cost_usd': {'type': ['number', 'null']},
                    'arrival_date': {'type': ['string', 'null']}, 'quantity': {'type': 'number'},
                    'budget_usd': {'type': 'number'}, 'deadline': {'type': 'string'},
                    'reasoning': {'type': 'string'}, 'conditions': {'type': 'array', 'maxItems': 0, 'items': {'type': 'string'}}}}
            decision = runner.call('Decision',
                'Return only JSON: verdict, option_id, total_cost_usd, arrival_date, quantity, budget_usd, deadline, reasoning, conditions. '
                'Copy Python recommendation fields exactly. reasoning must EXACTLY equal the provided VERIFIED REASON. '
                'Use conditions=[]. Do not invent or add explanatory text.',
                full + '\nRISK VERIFIED:\n' + risk['forwarded_output'] + '\nCOMPLIANCE VERIFIED:\n' + compliance['forwarded_output']
                + '\n' + authoritative + '\nPYTHON FIELDS:\n' + json.dumps(audit['recommendation']) + '\nVERIFIED REASON:\n' + canonical_reason,
                decision_schema)
            calls.append(decision)
            evaluation = evaluate_decision(decision['output'], scenario)
            parsed = evaluation['decision']
            evaluation['checks']['verified_explanation'] = (parsed.get('reasoning') == canonical_reason
                                                           and parsed.get('option_id') == audit['recommendation']['option_id'])
            evaluation['checks']['no_unverified_conditions'] = parsed.get('conditions') == []
            evaluation.update(passed=sum(evaluation['checks'].values()), total=len(evaluation['checks']),
                              all_passed=all(evaluation['checks'].values()),
                              failed_checks=[k for k, v in evaluation['checks'].items() if not v],
                              scope='Selection and exact verified explanation; failed outputs are blocked, never corrected silently.')
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
                            'shared_context_sha256': shared_hash, 'token_usage': sum_tokens(upstream + calls),
                            'execution_time_seconds': result['shared']['seconds'] + time.perf_counter() - branch_start}
        base, comp = result['baseline'], result['compressed']
        by_role = lambda b, role: next(c for c in b['calls'] if c['role'] == role)
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
        return result
    except Exception as exc:
        result.update(status='execution_failed', error=str(exc), calls=runner.calls,
                      measured_tokens=sum_tokens(runner.calls), usage_complete=all(c.get('total_tokens') is not None for c in runner.calls),
                      experiment_wall_seconds=time.perf_counter() - started)
        return result
