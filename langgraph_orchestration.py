"""LangGraph state orchestration for the validated selective workflow.

Raw history remains available for audit in state. Only active_context and
guarded reports are passed to model prompts.
"""
import json
import operator
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END

from measured_policy import choose_policy
from paired_benchmark import evaluate_decision


class BranchState(TypedDict, total=False):
    source_context: str
    active_context: str
    compressed_context: str
    protected_facts: dict
    constraint_audit: dict
    raw_history: Annotated[list, operator.add]
    calls: Annotated[list, operator.add]
    compression: dict | None
    logistics_call: dict
    finance_call: dict
    risk_call: dict
    compliance_call: dict
    decision_call: dict
    decision_evaluation: dict


def run_branch(*, mode, source, record, audit, caveats, catalog, required_ids,
               optional_limit, compressor_input, compact, selection_schema,
               canonical_reason, model, protocol, compression_mode,
               compressor_strategy, profile, runner, agent, scenario,
               ingestion, compact_renderer=None, full_source_for_all_agents=False,
               adaptation=None):
    """Execute the agent dependency graph and return its full final state."""
    authoritative = audit['text']

    def compressor(state: BranchState):
        if compressor_strategy == 'python_only':
            # A self-contained projection also replaces the separately
            # appended Python audit, so compare the complete reviewer input.
            original_length = len(source) + (len(authoritative) if compact_renderer else 0)
            shorter = len(compact) < original_length
            apply = compression_mode == 'always' or (compression_mode == 'auto' and shorter)
            policy = {
                'mode': compression_mode, 'applied': False,
                'should_compress': apply, 'compressor_called': False,
                'estimated_net_tokens_saved': None,
                'estimated_compressor_tokens': 0,
                'calibration': None,
                'reason': ('Forced deterministic pilot' if compression_mode == 'always'
                           else 'Disabled' if compression_mode == 'never'
                           else 'Deterministic projection is shorter than source' if shorter
                           else 'Deterministic projection is not shorter'),
            }
        else:
            policy = choose_policy(compression_mode, model, protocol,
                                   [source] * 4, compact, compressor_input, profile)
        update = {'compression': policy, 'compressed_context': ''}
        if not policy['should_compress']:
            return update
        if compressor_strategy == 'python_only':
            policy.update(applied=True, compressor_called=False,
                          caveats_preserved=True,
                          reason='Deterministic Python source projection; no LLM compressor cost')
            selected_context = compact_renderer(list(catalog)) if compact_renderer else compact
            update.update(active_context=selected_context, compressed_context=selected_context)
            return update
        call = runner.call(
            'Compressor',
            'Select the source caveats still useful after the protected fact table and Python audit. '
            f'Return JSON {{"keep":[...]}}. Required IDs: {json.dumps(required_ids)}; include every required ID exactly once. '
            f'Choose at most {optional_limit} other IDs only for unresolved issues not already covered by Python. '
            'Do not copy prose or facts. Empty catalog means keep=[].',
            compressor_input, selection_schema)
        try:
            selection = json.loads(call['output'])
            ids = selection['keep']
            valid = (isinstance(ids, list)
                     and all(isinstance(i, str) and i in catalog for i in ids)
                     and len(ids) == len(set(ids))
                     and set(required_ids).issubset(ids)
                     and len(ids) <= len(required_ids) + optional_limit
                     and set(selection) == {'keep'})
        except (ValueError, TypeError, KeyError):
            valid, ids = False, []
        call['compression_validation'] = {
            'valid': valid, 'required_caveat_ids': required_ids,
            'selected_ids': ids,
            'optional_omitted_ids': [i for i in catalog if i not in ids and i not in required_ids] if valid else [],
        }
        policy.update(compressor_called=True, caveats_preserved=valid,
                      selected_caveat_ids=ids if valid else None,
                      mandatory_caveat_ids=required_ids)
        update.update(calls=[call], raw_history=[{'role': 'Compressor', 'raw_output': call['output']}])
        if valid and (compression_mode == 'always' or len(compact) < len(source)):
            selected_context = (compact_renderer(ids) if compact_renderer else
                                record.compact_text() + '\nSOURCE CAVEATS:\n' + '\n'.join(catalog[i] for i in ids))
            policy['applied'] = True
            update.update(active_context=selected_context, compressed_context=selected_context)
        else:
            policy['reason'] = 'Full context retained: invalid caveat coverage or no size reduction; compressor cost counted'
        return update

    def reviewer(role, input_text):
        def node(state: BranchState):
            if adaptation and state.get('compression', {}).get('applied') and role in adaptation['skipped_roles']:
                from adaptive_context import python_report
                text = python_report(role, audit)
                return {role.lower() + '_call': {'role': role, 'output': text, 'forwarded_output': text,
                                                'source': 'python'},
                        'raw_history': [{'role': role, 'raw_output': text, 'source': 'python'}]}
            call = agent(role, input_text(state))
            return {role.lower() + '_call': call, 'calls': [call],
                    'raw_history': [{'role': role, 'raw_output': call['output']}]}
        return node

    def audit_suffix(state):
        return '' if compact_renderer and state['compression'] and state['compression']['applied'] else '\n' + authoritative

    def role_context(state, role):
        if adaptation and state.get('compression', {}).get('applied'):
            return adaptation['role_contexts'].get(role, state['active_context'])
        return state['active_context']

    logistics = reviewer('Logistics', lambda s: role_context(s, 'Logistics') + audit_suffix(s))
    finance = reviewer('Finance', lambda s: role_context(s, 'Finance') + audit_suffix(s))

    def full_context(state):
        context = state['active_context']
        for role in ('Logistics', 'Finance'):
            call = state[role.lower() + '_call']
            # Tool results already occur in the authoritative compact table.
            # Keep their separate reports in history without repeating the table.
            if adaptation and adaptation.get('compact_handoff') and call.get('source') == 'python':
                continue
            context += '\n' + role.upper() + ':\n' + call['forwarded_output']
        return context

    risk = reviewer('Risk', lambda s: full_context(s) + audit_suffix(s))
    compliance = reviewer(
        'Compliance',
        (lambda s: role_context(s, 'Compliance') + audit_suffix(s))
        if full_source_for_all_agents else
        (lambda s: record.compact_text() + '\n' + authoritative))

    decision_schema = {
        'type': 'object', 'additionalProperties': False,
        'required': ['verdict', 'option_id', 'total_cost_usd', 'arrival_date',
                     'quantity', 'budget_usd', 'deadline', 'reasoning', 'conditions'],
        'properties': {
            'verdict': {'type': 'string', 'enum': ['proceed', 'hold']},
            'option_id': {'type': ['string', 'null']},
            'total_cost_usd': {'type': ['number', 'null']},
            'arrival_date': {'type': ['string', 'null']},
            'quantity': {'type': 'number'},
            'budget_usd': {'type': 'number'},
            'deadline': {'type': 'string'},
            'reasoning': {'type': 'string'},
            'conditions': {'type': 'array', 'maxItems': 0, 'items': {'type': 'string'}},
        },
    }

    def decide(state: BranchState):
        compliance_report = state['compliance_call']
        compliance_context = ('\nCOMPLIANCE VERIFIED:\n' + compliance_report['forwarded_output'])
        if adaptation and adaptation.get('compact_handoff') and compliance_report.get('source') == 'python':
            compliance_context = ''
        call = runner.call(
            'Decision',
            'Return only JSON: verdict, option_id, total_cost_usd, arrival_date, quantity, budget_usd, deadline, reasoning, conditions. '
            'Copy Python recommendation fields exactly. reasoning must EXACTLY equal the provided VERIFIED REASON. '
            'Use conditions=[]. Do not invent or add explanatory text.',
            full_context(state) + '\nRISK VERIFIED:\n' + state['risk_call']['forwarded_output']
            + compliance_context
            + audit_suffix(state) + '\nPYTHON FIELDS:\n'
            + json.dumps(audit['recommendation']) + '\nVERIFIED REASON:\n'
            + canonical_reason, decision_schema)
        evaluation = evaluate_decision(call['output'], scenario)
        parsed = evaluation['decision']
        evaluation['checks']['verified_explanation'] = (
            parsed.get('reasoning') == canonical_reason
            and parsed.get('option_id') == audit['recommendation']['option_id'])
        evaluation['checks']['no_unverified_conditions'] = parsed.get('conditions') == []
        evaluation.update(
            passed=sum(evaluation['checks'].values()),
            total=len(evaluation['checks']),
            all_passed=all(evaluation['checks'].values()),
            failed_checks=[k for k, v in evaluation['checks'].items() if not v],
            scope='Selection and exact verified explanation; failed outputs are blocked, never corrected silently.')
        return {'decision_call': call, 'decision_evaluation': evaluation,
                'calls': [call],
                'raw_history': [{'role': 'Decision', 'raw_output': call['output']}]}

    graph = StateGraph(BranchState)
    if mode == 'compressed':
        graph.add_node('Compressor', compressor)
    graph.add_node('Logistics', logistics)
    graph.add_node('Finance', finance)
    graph.add_node('Risk', risk)
    graph.add_node('Compliance', compliance)
    graph.add_node('Decision', decide)
    first = 'Compressor' if mode == 'compressed' else START
    if mode == 'compressed':
        graph.add_edge(START, 'Compressor')
    graph.add_edge(first, 'Logistics')
    graph.add_edge(first, 'Finance')
    graph.add_edge(['Logistics', 'Finance'], 'Risk')
    graph.add_edge(['Logistics', 'Finance'], 'Compliance')
    graph.add_edge(['Risk', 'Compliance'], 'Decision')
    graph.add_edge('Decision', END)
    initial: BranchState = {
        'source_context': source,
        'active_context': source,
        'compressed_context': '',
        'protected_facts': record.to_dict(),
        'constraint_audit': audit,
        'raw_history': [
            {'role': 'Source', 'raw_output': source},
            {'role': 'Ingestion', 'raw_output': ingestion['output']},
        ],
        'calls': [],
        'compression': None,
    }
    return graph.compile().invoke(initial)
