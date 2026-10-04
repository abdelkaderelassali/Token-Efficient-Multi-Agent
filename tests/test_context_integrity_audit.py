import copy
import unittest

from context_integrity_audit import audit_experiment
from graph_selective_workflow import run_pair, scenarios
from tests.test_selective_workflow import SelectiveRunner


class ContextIntegrityAuditTests(unittest.TestCase):
    def setUp(self):
        scenario = scenarios()[2]
        self.result = run_pair(scenario, runner=SelectiveRunner(scenario),
                               compression_mode='always', profile={})

    def test_source_backed_context_passes_audit(self):
        report = audit_experiment(self.result)
        self.assertTrue(report['all_checks_passed'], report['checks'])
        self.assertFalse(report['full_brief_in_active_context'])
        self.assertFalse(report['carrier_and_method_names_in_active_context'])

    def test_altered_context_fails_audit(self):
        changed = copy.deepcopy(self.result)
        changed['compressed']['langgraph_state']['active_context'] += '\nInvented discount.'
        report = audit_experiment(changed)
        self.assertFalse(report['all_checks_passed'])
        self.assertFalse(report['checks']['context_exactly_reconstructed_from_source'])


if __name__ == '__main__':
    unittest.main()
