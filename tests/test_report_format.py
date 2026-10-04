import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage

import paired_benchmark as paired
from report_format import report_diagnostics
from structured_claims import instructions


class ReportFormatTests(unittest.TestCase):
    def test_sections_and_word_target_are_checked_without_rewriting(self):
        output = '## Risks\n- Late arrival.\n## Impact\n- Deadline missed.\n## Actions\n- Verify quote S2.'
        result = report_diagnostics('Risk', output)
        self.assertTrue(result['sections_match'])
        self.assertTrue(result['within_word_target'])
        invalid = report_diagnostics('Risk', 'Unstructured ' * 100)
        self.assertFalse(invalid['sections_match'])
        self.assertFalse(invalid['within_word_target'])

    def test_model_receives_contract_and_raw_output_is_preserved(self):
        with patch.object(paired, 'ChatOllama') as model:
            model.return_value.invoke.return_value = AIMessage(content='{"claims":[],"uncertainties":[]}', response_metadata={
                'prompt_eval_count': 50, 'eval_count': 5})
            result = paired.ModelRunner().call('Finance', 'Analyze costs.', 'Source')
            self.assertIn(instructions('Finance'), result['system_prompt'])
            self.assertEqual(result['raw_output'], '{"claims":[],"uncertainties":[]}')
            self.assertEqual(result['structured_report'], {'claims': [], 'uncertainties': []})
            self.assertTrue(result['claim_validation']['valid'])
            self.assertEqual(result['total_tokens'], 55)

    def test_final_json_contract_is_not_overridden_by_markdown(self):
        with patch.object(paired, 'ChatOllama') as model:
            model.return_value.invoke.return_value = AIMessage(content='{}', response_metadata={
                'prompt_eval_count': 50, 'eval_count': 5})
            result = paired.ModelRunner().call('Decision', 'Return JSON.', 'Source', json_output=True)
            self.assertEqual(result['system_prompt'], 'Return JSON.')
            self.assertIsNone(result['report_format'])


if __name__ == '__main__':
    unittest.main()
