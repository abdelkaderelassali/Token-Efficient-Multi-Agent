import unittest

import paired_benchmark as paired
from context_routing import finance_uncertainties, route_context
from test_paired_benchmark import FakeRunner


class ContextRoutingTests(unittest.TestCase):
    def test_finance_filter_preserves_uncertainties_and_falls_back_safely(self):
        report = '## Costs\n- Cost details.\n## Budget\n- Budget details.\n## Unknowns\n- Customs amount disputed.'
        selected = finance_uncertainties(report)
        self.assertIn('Customs amount disputed', selected)
        self.assertNotIn('Cost details', selected)
        self.assertEqual(finance_uncertainties('Unstructured report'), 'Unstructured report')
        self.assertEqual(finance_uncertainties('## Costs\nMissing sections'), '## Costs\nMissing sections')

    def test_source_constraints_are_retained_for_every_role(self):
        reports = {'Logistics': 'Logistics finding', 'Finance': 'Finance uncertainty'}
        for role in ('Risk', 'Compliance', 'Decision'):
            context, _ = route_context(role, 'AUTHORITATIVE budget=900 deadline=2026-10-08', reports)
            self.assertIn('budget=900', context)
            self.assertIn('deadline=2026-10-08', context)
        with self.assertRaises(ValueError):
            route_context('unknown', 'source', reports)

    def test_compliance_receives_identical_source_and_audit_without_narratives(self):
        runner = FakeRunner({})
        result = paired.run_pair(paired.scenarios()[0], runner=runner)
        contexts = [call[2] for call in runner.inputs if call[0] == 'Compliance']
        self.assertEqual(contexts[0], contexts[1])
        self.assertIn('PYTHON CONSTRAINT CHECKS', contexts[0])
        self.assertNotIn('Logistics findings', contexts[0])
        self.assertNotIn('Finance findings', contexts[0])
        self.assertEqual(result['compressed']['context_routing']['Compliance'], ['source', 'python_constraint_audit'])

    def test_specialists_do_not_receive_redundant_generated_briefing(self):
        runner = FakeRunner({})
        paired.run_pair(paired.scenarios()[0], runner=runner)
        for role, _, context, _, _ in runner.inputs:
            if role in ('Logistics', 'Finance'):
                self.assertNotIn('Ingestion findings', context)
                self.assertIn('QUOTE S1', context)
        for role, _, context, _, _ in runner.inputs:
            if role == 'Decision':
                self.assertIn('Risk findings', context)
                self.assertIn('Compliance findings', context)


if __name__ == '__main__':
    unittest.main()
