import unittest

from context_integrity_audit import audit_experiment
from graph_verified_lossless_workflow import run_pair, scenarios
from lossless_projection import render_verified_lossless_context
from protected_facts import ProtectedFacts
from quote_calculations import calculate_quotes
from tests.test_selective_workflow import SelectiveRunner


class VerifiedLosslessWorkflowTests(unittest.TestCase):
    def test_projection_keeps_every_source_field_and_python_check(self):
        for scenario in scenarios():
            record = ProtectedFacts.from_scenario(scenario)
            audit = calculate_quotes(record)
            context = render_verified_lossless_context(record, audit)
            for value in (record.brief, record.route, str(record.quantity),
                          str(record.budget_usd), record.deadline,
                          str(record.require_insurance).lower()):
                self.assertIn(value, context)
            for quote in record.quotes:
                for value in (quote.id, quote.carrier, quote.method,
                              str(quote.quoted_total_usd), quote.arrival_date,
                              str(quote.capacity_units),
                              str(quote.insurance_included).lower(),
                              quote.terms):
                    self.assertIn(value, context)
            for row in audit['rows']:
                for name in ('within_budget', 'on_time',
                             'enough_capacity', 'insured_as_required'):
                    self.assertIn(
                        f"{name}={'P' if row[name] else 'F'}", context)

    def test_every_agent_gets_full_source_with_one_verified_audit(self):
        scenario = scenarios()[2]
        result = run_pair(scenario, runner=SelectiveRunner(scenario),
                          compression_mode='auto')
        self.assertEqual(result['status'], 'validated', result.get('error'))
        self.assertTrue(result['comparison']['compression_applied'])
        self.assertEqual(result['comparison']['compressor_tokens'], 0)
        report = audit_experiment(result)
        self.assertTrue(report['all_checks_passed'], report['checks'])
        self.assertTrue(report['every_full_terms_in_active_context'])
        self.assertIn(result['baseline']['langgraph_state']['source_context'],
                      result['shared']['calls'][0]['input'])
        active = result['compressed']['langgraph_state']['active_context']
        for role in ('Logistics', 'Finance', 'Risk', 'Compliance', 'Decision'):
            call = next(c for c in result['compressed']['calls'] if c['role'] == role)
            self.assertIn(active, call['input'])
            self.assertNotIn('PYTHON CONSTRAINT CHECKS (authoritative', call['input'])


if __name__ == '__main__':
    unittest.main()
