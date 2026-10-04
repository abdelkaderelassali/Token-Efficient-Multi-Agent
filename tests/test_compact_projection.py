import unittest

from compact_projection import render_review_context
from graph_compact_workflow import run_pair, scenarios
from protected_facts import ProtectedFacts
from quote_calculations import calculate_quotes
from tests.test_selective_workflow import SelectiveRunner


class CompactProjectionTests(unittest.TestCase):
    def setUp(self):
        self.scenario = scenarios()[2]
        self.record = ProtectedFacts.from_scenario(self.scenario)
        self.audit = calculate_quotes(self.record)

    def test_projection_retains_every_structured_quote_fact_and_python_check(self):
        context = render_review_context(self.record, self.audit, {}, [])
        for value in (self.record.route, str(self.record.quantity), str(self.record.budget_usd),
                      self.record.deadline, str(self.record.require_insurance).lower()):
            self.assertIn(value, context)
        for quote, row in zip(self.record.quotes, self.audit['rows']):
            for value in (quote.id, quote.carrier, quote.method,
                          str(quote.quoted_total_usd), quote.arrival_date,
                          str(quote.capacity_units), str(quote.insurance_included).lower()):
                self.assertIn(value, context)
            for name in ('within_budget', 'on_time', 'enough_capacity', 'insured_as_required'):
                self.assertIn(f"{name}={'P' if row[name] else 'F'}", context)

    def test_graph_projection_routes_compact_context_without_duplicate_audit(self):
        result = run_pair(self.scenario, runner=SelectiveRunner(self.scenario),
                          compression_mode='always', profile={})
        self.assertEqual(result['status'], 'validated', result.get('error'))
        state = result['compressed']['langgraph_state']
        logistics = next(c for c in result['compressed']['calls'] if c['role'] == 'Logistics')
        self.assertEqual(logistics['input'], state['active_context'])
        self.assertIn('PYTHON CHECKS: P=PASS, F=FAIL', logistics['input'])
        self.assertNotIn('PYTHON CONSTRAINT CHECKS (authoritative calculations', logistics['input'])
        self.assertEqual(len(state['raw_history']), 8)

    def test_invalid_caveat_selection_restores_full_prompt(self):
        result = run_pair(self.scenario,
                          runner=SelectiveRunner(self.scenario, omit_required=True),
                          compression_mode='always', profile={})
        self.assertEqual(result['status'], 'validated', result.get('error'))
        self.assertFalse(result['comparison']['compression_applied'])
        for role in ('Logistics', 'Finance', 'Risk', 'Decision'):
            baseline = next(c for c in result['baseline']['calls'] if c['role'] == role)
            compressed = next(c for c in result['compressed']['calls'] if c['role'] == role)
            self.assertEqual(compressed['input'], baseline['input'])


if __name__ == '__main__':
    unittest.main()
