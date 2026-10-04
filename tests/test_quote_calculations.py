import copy
import unittest

from paired_benchmark import scenarios, run_pair, evaluate_decision
from protected_facts import ProtectedFacts
from quote_calculations import calculate_quotes
from test_paired_benchmark import FakeRunner
import json


class QuoteCalculationsTests(unittest.TestCase):
    def calculate(self, scenario):
        return calculate_quotes(ProtectedFacts.from_scenario(scenario))

    def test_known_scenarios_match_independent_expected_answers(self):
        expected = [('S1', 720), ('M3', 5100), ('D2', 12800)]
        for scenario, (quote, price) in zip(scenarios(), expected):
            result = self.calculate(scenario)
            self.assertEqual(result['recommendation']['option_id'], quote)
            self.assertEqual(result['lowest_eligible_total_usd'], price)
        result = self.calculate(scenarios()[2])
        details = {r['quote_id']: r for r in result['comparisons']}
        self.assertEqual(details['D2']['budget_remaining_usd'], '2200')
        self.assertEqual(details['D1']['days_before_deadline'], -3)
        self.assertEqual(details['D5']['capacity_surplus_units'], -150)
        self.assertEqual(details['D4']['failed_constraints'], ['insured_as_required'])

    def test_exact_limits_pass_and_one_unit_violations_fail(self):
        source = copy.deepcopy(scenarios()[0])
        source['options'] = [source['options'][0]]
        offer = source['options'][0]
        offer.update(quoted_total_usd=source['budget_usd'], arrival_date=source['deadline'], capacity_units=source['quantity'])
        self.assertEqual(self.calculate(source)['eligible_ids'], ['S1'])
        for key, value, failure in [('quoted_total_usd', 901, 'within_budget'),
                                    ('arrival_date', '2026-10-09', 'on_time'),
                                    ('capacity_units', 39, 'enough_capacity'),
                                    ('insurance_included', False, 'insured_as_required')]:
            changed = copy.deepcopy(source)
            changed['options'][0][key] = value
            result = self.calculate(changed)
            self.assertEqual(result['comparisons'][0]['failed_constraints'], [failure])
            self.assertEqual(result['recommendation']['verdict'], 'hold')
            for field in ('option_id', 'total_cost_usd', 'arrival_date'):
                self.assertIsNone(result['recommendation'][field])

    def test_optional_insurance_does_not_exclude_uninsured_quote(self):
        source = copy.deepcopy(scenarios()[0])
        source['require_insurance'] = False
        source['options'][0]['insurance_included'] = False
        self.assertEqual(self.calculate(source)['recommendation']['option_id'], 'S1')

    def test_ties_are_order_independent_and_all_minima_are_accepted(self):
        source = copy.deepcopy(scenarios()[0])
        second = dict(source['options'][0], id='S0')
        source['options'] = [source['options'][0], second]
        result = self.calculate(source)
        self.assertEqual(result['lowest_cost_eligible_ids'], ['S0', 'S1'])
        self.assertEqual(result['recommendation']['option_id'], 'S0')
        source['options'].reverse()
        self.assertEqual(self.calculate(source)['recommendation'], result['recommendation'])
        for quote in ('S0', 'S1'):
            decision = dict(result['recommendation'], option_id=quote, reasoning='Tied minimum.', conditions=[])
            self.assertTrue(evaluate_decision(json.dumps(decision), source)['all_passed'])

    def test_decimal_budget_balance_does_not_use_binary_float_subtraction(self):
        source = copy.deepcopy(scenarios()[0])
        source['budget_usd'] = 0.3
        source['options'][0]['quoted_total_usd'] = 0.1
        self.assertEqual(self.calculate(source)['comparisons'][0]['budget_remaining_usd'], '0.2')

    def test_all_agents_receive_same_authority_and_model_output_is_not_replaced(self):
        source = scenarios()[0]
        runner = FakeRunner({'verdict': 'hold'})
        result = run_pair(source, runner=runner, compression_mode='always')
        authority = result['shared']['constraint_audit']
        for role, _, context, _, _ in runner.inputs:
            self.assertIn(authority['text'], context, role)
        for mode in ('baseline', 'compressed'):
            self.assertEqual(result[mode]['constraint_audit'], authority)
            self.assertEqual(result[mode]['evaluation']['decision'], {'verdict': 'hold'})
            self.assertFalse(result[mode]['evaluation']['all_passed'])


if __name__ == '__main__':
    unittest.main()
