import unittest

from graph_selective_workflow import run_pair, scenarios
from tests.test_selective_workflow import SelectiveRunner


class LangGraphWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.scenario = scenarios()[2]

    def test_graph_keeps_raw_history_but_sends_selected_context(self):
        result = run_pair(self.scenario, runner=SelectiveRunner(self.scenario),
                          compression_mode='always', profile={})
        self.assertEqual(result['status'], 'validated', result.get('error'))
        self.assertTrue(result['comparison']['compression_applied'])
        state = result['compressed']['langgraph_state']
        self.assertEqual(state['source_context'], result['baseline']['langgraph_state']['source_context'])
        self.assertTrue(state['compressed_context'])
        self.assertEqual(state['active_context'], state['compressed_context'])
        self.assertGreater(len(state['source_context']), len(state['active_context']))
        self.assertEqual(len(state['raw_history']), 8)
        logistics = next(c for c in result['compressed']['calls'] if c['role'] == 'Logistics')
        self.assertIn(state['active_context'], logistics['input'])
        self.assertNotIn(state['source_context'], logistics['input'])
        self.assertEqual(result['comparison']['saved_tokens'],
                         result['baseline']['token_usage'] - result['compressed']['token_usage'])

    def test_invalid_selection_falls_back_to_source_and_charges_compressor(self):
        result = run_pair(self.scenario,
                          runner=SelectiveRunner(self.scenario, omit_required=True),
                          compression_mode='always', profile={})
        self.assertEqual(result['status'], 'validated', result.get('error'))
        self.assertFalse(result['comparison']['compression_applied'])
        self.assertGreater(result['comparison']['compressor_tokens'], 0)
        self.assertEqual(result['compressed']['langgraph_state']['active_context'],
                         result['compressed']['langgraph_state']['source_context'])


if __name__ == '__main__':
    unittest.main()
