import unittest

from context_integrity_audit import audit_experiment
from graph_lossless_workflow import run_pair, scenarios
from lossless_projection import render_lossless_context
from protected_facts import ProtectedFacts
from tests.test_selective_workflow import SelectiveRunner


class LosslessWorkflowTests(unittest.TestCase):
    def test_projection_contains_every_original_field_and_supplier_term(self):
        for scenario in scenarios():
            record = ProtectedFacts.from_scenario(scenario)
            context = render_lossless_context(record)
            for value in (record.brief, record.route, str(record.quantity),
                          str(record.budget_usd), record.deadline,
                          str(record.require_insurance).lower()):
                self.assertIn(value, context)
            for quote in record.quotes:
                for value in (quote.id, quote.carrier, quote.method,
                              str(quote.quoted_total_usd), quote.arrival_date,
                              str(quote.capacity_units),
                              str(quote.insurance_included).lower(), quote.terms):
                    self.assertIn(value, context)

    def test_graph_routes_full_terms_without_llm_compressor_cost(self):
        scenario = scenarios()[2]
        result = run_pair(scenario, runner=SelectiveRunner(scenario),
                          compression_mode='auto')
        self.assertEqual(result['status'], 'validated', result.get('error'))
        self.assertTrue(result['comparison']['compression_applied'])
        self.assertEqual(result['comparison']['compressor_tokens'], 0)
        self.assertTrue(audit_experiment(result)['all_checks_passed'])
        state = result['compressed']['langgraph_state']
        self.assertIn(state['source_context'], result['shared']['calls'][0]['input'])
        for role in ('Logistics', 'Finance', 'Risk', 'Compliance', 'Decision'):
            call = next(c for c in result['compressed']['calls'] if c['role'] == role)
            self.assertIn(state['active_context'], call['input'])
        self.assertFalse(any(c['role'] == 'Compressor' for c in result['compressed']['calls']))

    def test_never_mode_keeps_source_context(self):
        scenario = scenarios()[0]
        result = run_pair(scenario, runner=SelectiveRunner(scenario),
                          compression_mode='never')
        self.assertEqual(result['status'], 'validated', result.get('error'))
        self.assertFalse(result['comparison']['compression_applied'])
        state = result['compressed']['langgraph_state']
        self.assertEqual(state['active_context'], state['source_context'])


if __name__ == '__main__':
    unittest.main()
