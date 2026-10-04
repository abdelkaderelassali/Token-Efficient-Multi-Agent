import unittest
from unittest.mock import patch

import benchmark
import paired_benchmark as paired
from compression_policy import compression_policy
from test_paired_benchmark import FakeRunner
from test_workflow import RecordingModel


class CompressionPolicyTests(unittest.TestCase):
    def test_small_context_skips_and_large_context_can_qualify(self):
        self.assertFalse(compression_policy('short context')['should_compress'])
        self.assertTrue(compression_policy('source ' * 900, compact_chars=600)['should_compress'])
        self.assertFalse(compression_policy('source ' * 900, compact_chars=6200)['should_compress'])
        self.assertTrue(compression_policy('short', mode='always')['should_compress'])
        self.assertFalse(compression_policy('source ' * 900, mode='never')['should_compress'])

    def test_skip_has_no_compressor_cost_and_same_downstream_inputs(self):
        runner = FakeRunner({})
        result = paired.run_pair(paired.scenarios()[0], runner=runner)
        self.assertFalse(result['compressed']['compression']['applied'])
        self.assertFalse(result['compressed']['compression']['compressor_called'])
        self.assertEqual(result['comparison']['compressor_tokens'], 0)
        self.assertEqual(result['baseline']['token_usage'], result['compressed']['token_usage'])
        self.assertEqual(result['compressed']['reports']['Compressor'], '')
        for role in ('Risk', 'Compliance', 'Decision'):
            inputs = [c[2] for c in runner.inputs if c[0] == role]
            self.assertEqual(inputs[0], inputs[1])

    def test_large_context_calls_compressor(self):
        result = paired.run_pair(paired.scenarios()[2], runner=FakeRunner({}))
        self.assertTrue(result['compressed']['compression']['applied'])
        self.assertGreater(result['comparison']['compressor_tokens'], 0)

    def test_oversized_result_falls_back_and_still_counts_cost(self):
        runner = FakeRunner({})
        original = runner.call
        def oversized(*args, **kwargs):
            call = original(*args, **kwargs)
            if args[0] == 'Compressor':
                call['output'] = 'Uncertainty. ' * 1200
            return call
        runner.call = oversized
        result = paired.run_pair(paired.scenarios()[2], runner=runner)
        policy = result['compressed']['compression']
        self.assertFalse(policy['applied'])
        self.assertTrue(policy['compressor_called'])
        self.assertEqual(result['comparison']['compressor_tokens'], 120)
        self.assertEqual(result['comparison']['saved_tokens'], -120)

    def test_legacy_skip_keeps_reports_and_completes_reviews(self):
        model = RecordingModel()
        with patch.object(benchmark, 'llm', model):
            result = benchmark.build_multi_agent_graph(True).invoke(benchmark.initial_state('Ship goods'))
        self.assertNotIn('Compressor', [role for role, _ in model.calls])
        self.assertFalse(result['compression']['applied'])
        self.assertEqual(result['decision_report'], model.OUTPUTS['Decision'])
        self.assertEqual(result['logistics_report'], model.OUTPUTS['Logistics'])

    def test_invalid_mode_fails_before_inference(self):
        runner = FakeRunner({})
        with self.assertRaises(ValueError):
            paired.run_pair(paired.scenarios()[0], runner=runner, compression_mode='invalid')
        self.assertEqual(runner.inputs, [])


if __name__ == '__main__':
    unittest.main()
