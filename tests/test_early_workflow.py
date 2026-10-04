import unittest

from early_workflow import PROTOCOL, run_pair, scenarios
from tests.test_guarded_workflow import FakeGuardedRunner


class EarlyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.scenario = scenarios()[2]

    def test_early_compression_reaches_four_reused_prompts_and_counts_cost(self):
        result = run_pair(self.scenario, runner=FakeGuardedRunner(self.scenario),
                          compression_mode='always', profile={})
        self.assertEqual(result['status'], 'validated')
        self.assertEqual(result['configuration']['pipeline_version'], PROTOCOL)
        self.assertEqual([c['role'] for c in result['shared']['calls']], ['Ingestion'])
        self.assertTrue(result['comparison']['compression_applied'])
        self.assertGreater(result['comparison']['compressor_tokens'], 0)
        self.assertGreater(result['comparison']['saved_tokens'], 0)
        self.assertEqual(result['comparison']['saved_tokens'],
                         result['comparison']['downstream_input_tokens_saved']
                         + result['comparison']['downstream_output_tokens_saved']
                         - result['comparison']['compressor_tokens'])
        for role in ('Logistics', 'Finance', 'Risk', 'Decision'):
            baseline = next(c for c in result['baseline']['calls'] if c['role'] == role)
            compressed = next(c for c in result['compressed']['calls'] if c['role'] == role)
            self.assertLess(compressed['input_tokens'], baseline['input_tokens'], role)

    def test_missing_caveat_rejects_early_compression(self):
        result = run_pair(self.scenario,
                          runner=FakeGuardedRunner(self.scenario, drop_caveat=True),
                          compression_mode='always', profile={})
        self.assertEqual(result['status'], 'validated')
        self.assertFalse(result['comparison']['compression_applied'])
        self.assertLess(result['comparison']['saved_tokens'], 0)
        for role in ('Logistics', 'Finance', 'Risk', 'Decision'):
            baseline = next(c for c in result['baseline']['calls'] if c['role'] == role)
            compressed = next(c for c in result['compressed']['calls'] if c['role'] == role)
            self.assertEqual(compressed['input'], baseline['input'], role)

    def test_early_branches_block_false_claim_prose_and_final_reason(self):
        false_claim = '$12800 exceeds $15000'
        runner = FakeGuardedRunner(self.scenario, bad_claim=True, bad_reason=True)
        result = run_pair(self.scenario, runner=runner, compression_mode='always', profile={})
        self.assertEqual(result['status'], 'failed_validation')
        self.assertTrue(any(false_claim in call['raw_output'] for call in runner.calls))
        self.assertTrue(all(false_claim not in call['input'] for call in runner.calls))
        for mode in ('baseline', 'compressed'):
            self.assertEqual(result[mode]['status'], 'failed_validation')
            self.assertTrue(result[mode]['reports']['Decision'].startswith('FAILED VALIDATION'))

    def test_python_only_ablation_omits_llm_cost_without_losing_caveats(self):
        result = run_pair(self.scenario, runner=FakeGuardedRunner(self.scenario),
                          compression_mode='always', profile={}, compressor_strategy='python_only')
        self.assertEqual(result['status'], 'validated')
        self.assertTrue(result['compressed']['compression']['caveats_preserved'])
        self.assertFalse(result['compressed']['compression']['compressor_called'])
        self.assertEqual(result['comparison']['compressor_tokens'], 0)
        self.assertFalse(any(c['role'] == 'Compressor' for c in result['compressed']['calls']))


if __name__ == '__main__':
    unittest.main()
