import copy
import unittest

from adaptive_workflow import run_pair, scenarios, PROTOCOL
from context_integrity_audit import audit_experiment
from tests.test_guarded_workflow import FakeGuardedRunner


class AdaptiveWorkflowTests(unittest.TestCase):
    def run_case(self, scenario, **kwargs):
        if scenario not in scenarios() and 'requirement_review' not in scenario:
            scenario = dict(scenario, requirement_review={'confirmed':True, 'required_capabilities':[],
                'other_requirements':[], 'quote_capabilities':{}})
        result = run_pair(scenario, runner=FakeGuardedRunner(scenario), **kwargs)
        self.assertEqual(result['status'], 'validated', result.get('error', result.get('context_audit')))
        return result

    def test_short_replaces_two_calls_and_accounts_for_three_branches(self):
        r = self.run_case(scenarios()[0])
        self.assertEqual(r['comparison']['skipped_llm_roles'], ['Finance', 'Compliance'])
        self.assertEqual(len(r['compressed']['calls']), 3)
        self.assertEqual(len(r['compression_only']['calls']), 5)
        self.assertEqual(r['comparison']['saved_tokens'],
                         r['comparison']['compression_only_saved_tokens'] + r['comparison']['routing_saved_tokens'])
        physical = sum(c['total_tokens'] for branch in ('baseline', 'compression_only', 'compressed') for c in r[branch]['calls'])
        self.assertEqual(r['comparison']['total_experiment_tokens'], physical + r['shared']['token_usage'])
        self.assertTrue(r['context_audit']['all_checks_passed'])
        from evaluation_suite import summarize
        stats = summarize([r])['llama3']
        self.assertEqual(stats['validated_decisions'], 3)
        self.assertEqual(stats['attempted_decisions'], 3)

    def test_long_uses_source_spans_and_keeps_every_term_in_final_review(self):
        scenario = scenarios()[2]
        r = self.run_case(scenario, protocol='validated-v11-adaptive')
        self.assertEqual(r['comparison']['adaptive_path'], 'long')
        plan = r['configuration']['adaptive_plan']
        terms = {q['id']: q['terms'] for q in scenario['options']}
        for chunk in plan['passages']:
            self.assertEqual(chunk['text'], terms[chunk['quote_id']][chunk['start']:chunk['end']])
        for role in ('Risk', 'Decision'):
            call = next(c for c in r['compressed']['calls'] if c['role'] == role)
            for text in terms.values():
                self.assertIn(text, call['input'])
        r['compressed']['calls'][-1]['input'] = 'source accidentally lost'
        self.assertFalse(audit_experiment(r)['all_checks_passed'])

    def test_long_tools_replace_three_calls_without_repeating_checks(self):
        scenario = scenarios()[2]
        r = self.run_case(scenario)
        self.assertEqual(r['configuration']['pipeline_version'], PROTOCOL)
        self.assertEqual(r['comparison']['skipped_llm_roles'], ['Logistics', 'Finance', 'Compliance'])
        self.assertEqual([c['role'] for c in r['compressed']['calls']], ['Risk', 'Decision'])
        self.assertEqual(len(r['baseline']['calls']), 5)
        self.assertEqual(len(r['compression_only']['calls']), 5)
        full = r['configuration']['adaptive_plan']['full_context']
        for call in r['compressed']['calls']:
            self.assertEqual(call['input'].count(full), 1)
            self.assertIn(scenario['brief'], call['input'])
            for quote in scenario['options']:
                self.assertIn(quote['terms'], call['input'])
            for section in ('LOGISTICS:', 'FINANCE:', 'COMPLIANCE VERIFIED:'):
                self.assertNotIn(section, call['input'])
        history = r['compressed']['langgraph_state']['raw_history']
        self.assertEqual({h['role'] for h in history if h.get('source') == 'python'},
                         {'Logistics', 'Finance', 'Compliance'})
        self.assertIn('D1: on_time=FAIL.', r['compressed']['reports']['Logistics'])
        self.assertIn('D5: enough_capacity=FAIL.', r['compressed']['reports']['Logistics'])
        self.assertTrue(r['context_audit']['all_checks_passed'])
        r['compressed']['reports']['Logistics'] = 'D1: on_time=PASS.'
        self.assertFalse(audit_experiment(r)['all_checks_passed'])

    def test_long_disabled_retains_all_llm_roles(self):
        r = self.run_case(scenarios()[2], compression_mode='never')
        self.assertEqual(len(r['compressed']['calls']), 5)
        self.assertEqual(r['comparison']['skipped_llm_roles'], [])
        self.assertEqual(r['comparison']['saved_tokens'], 0)

    def test_long_tools_handle_no_eligible_quote_and_ties(self):
        case = copy.deepcopy(scenarios()[2])
        case['budget_usd'] = 1
        r = self.run_case(case)
        self.assertEqual(r['compressed']['evaluation']['decision']['verdict'], 'hold')
        case = copy.deepcopy(scenarios()[2])
        case['options'][0] = dict(case['options'][1], id='D0')
        r = self.run_case(case)
        self.assertEqual(r['compressed']['evaluation']['decision']['option_id'], 'D0')

    def test_tool_role_must_be_explicit(self):
        from adaptive_context import python_report
        with self.assertRaises(KeyError):
            python_report('Unknown', {})

    def test_audit_blocks_added_prose_even_when_source_is_still_present(self):
        r = self.run_case(scenarios()[2])
        r['compressed']['calls'][-1]['input'] += '\nD2: $12800 exceeds $15000.'
        audit = audit_experiment(r)
        self.assertTrue(audit['checks']['compressed_full_terms_in_risk_and_decision'])
        self.assertFalse(audit['checks']['compressed_prompts_exactly_reconstructed'])
        self.assertFalse(audit['all_checks_passed'])

    def test_audit_checks_actual_ingestion_prompt(self):
        r = self.run_case(scenarios()[2])
        r['shared']['calls'][0]['input'] = 'Missing the original source'
        self.assertFalse(audit_experiment(r)['checks']['ingestion_prompt_intact'])

    def test_disabled_has_no_routing_or_avoided_calls(self):
        r = self.run_case(scenarios()[0], compression_mode='never')
        self.assertEqual(len(r['compressed']['calls']), 5)
        self.assertEqual(r['comparison']['saved_tokens'], 0)
        self.assertEqual(r['comparison']['adaptive_path'], 'disabled')

    def test_hold_and_tie_decisions_remain_valid(self):
        case = copy.deepcopy(scenarios()[0])
        case['budget_usd'] = 1
        self.run_case(case)
        case = copy.deepcopy(scenarios()[0])
        case['options'][1] = dict(case['options'][0], id='TIE')
        self.run_case(case)

    def test_false_claim_and_invalid_final_are_blocked(self):
        case = scenarios()[2]
        r = run_pair(case, runner=FakeGuardedRunner(case, bad_claim=True))
        for mode in ('baseline', 'compression_only', 'compressed'):
            for call in r[mode]['calls']:
                self.assertNotIn('$12800 exceeds $15000', call['input'])
        r = run_pair(case, runner=FakeGuardedRunner(case, bad_reason=True))
        self.assertEqual(r['status'], 'failed_validation')
        self.assertFalse(r['comparison']['quality_preserving_savings'])
