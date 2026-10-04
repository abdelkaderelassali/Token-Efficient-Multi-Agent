import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from langchain_core.messages import AIMessage

import paired_benchmark as paired
from structured_claims import parse_report, render_report, SCHEMA


class StructuredClaimsTests(unittest.TestCase):
    def setUp(self):
        self.data = {'claims': [{'quote_id': 'D2', 'constraint': 'within_budget', 'result': 'pass',
                                'explanation': 'USD 12800 fits the USD 15000 budget.'}],
                     'uncertainties': [{'quote_id': None, 'detail': 'Confirm booking availability.'}]}

    def test_roundtrip_and_readable_render_preserve_fields(self):
        parsed = parse_report(json.dumps(self.data), 'Finance')
        self.assertEqual(parsed, self.data)
        rendered = render_report(parsed)
        self.assertIn('D2 | within_budget | pass', rendered)
        self.assertIn(self.data['claims'][0]['explanation'], rendered)
        self.assertIn('Confirm booking availability.', rendered)

    def test_missing_extra_invalid_duplicate_and_overlong_claims_fail(self):
        variants = []
        for field, value in [('quote_id', None), ('constraint', 'invented'), ('result', True),
                             ('explanation', ''), ('explanation', 'word ' * 31)]:
            data = copy.deepcopy(self.data)
            data['claims'][0][field] = value
            variants.append(data)
        data = copy.deepcopy(self.data); data['claims'].append(data['claims'][0]); variants.append(data)
        variants.extend([{}, {'claims': [], 'uncertainties': [], 'extra': 1},
                         {'claims': 'text', 'uncertainties': []}])
        for data in variants:
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_report(json.dumps(data), 'Finance')

    def test_duplicate_json_keys_and_nonfinite_numbers_fail(self):
        for raw in ('{"claims":[],"claims":[],"uncertainties":[]}',
                    '{"claims":NaN,"uncertainties":[]}', 'not JSON'):
            with self.assertRaises(ValueError):
                parse_report(raw, 'Risk')

    def test_compressor_only_accepts_uncertainties(self):
        with self.assertRaises(ValueError):
            parse_report(json.dumps(self.data), 'Compressor')
        data = {'claims': [], 'uncertainties': self.data['uncertainties']}
        self.assertEqual(parse_report(json.dumps(data), 'Compressor'), data)

    def test_false_but_well_formed_claim_is_not_silently_corrected(self):
        self.data['claims'][0]['result'] = 'fail'
        self.assertEqual(parse_report(json.dumps(self.data), 'Finance'), self.data)

    def test_invalid_model_response_is_saved_and_not_forwarded(self):
        with TemporaryDirectory() as directory, patch.object(paired, 'ROOT', Path(directory)), patch.object(paired, 'ChatOllama') as model:
            model.return_value.invoke.return_value = AIMessage(content='broken JSON', response_metadata={
                'prompt_eval_count': 80, 'eval_count': 20, 'done_reason': 'length'})
            with self.assertRaisesRegex(ValueError, 'invalid structured report'):
                paired.ModelRunner().call('Finance', 'Analyze.', 'Source')
            self.assertEqual(model.call_args.kwargs['format'], SCHEMA)
            files = list((Path(directory) / 'verification').glob('invalid-report-*.json'))
            self.assertEqual(len(files), 1)
            call = json.loads(files[0].read_text())['call']
            self.assertEqual(call['raw_output'], 'broken JSON')
            self.assertEqual(call['total_tokens'], 100)
            self.assertFalse(call['claim_validation']['valid'])

    def test_parsed_claims_reach_saved_branch_results(self):
        report = {'claims': [], 'uncertainties': []}
        with patch.object(paired, 'ChatOllama') as model:
            def reply(messages):
                content = '{}' if 'Return only JSON with keys: verdict' in messages[0].content else json.dumps(report)
                return AIMessage(content=content, response_metadata={'prompt_eval_count': 80, 'eval_count': 20})
            model.return_value.invoke.side_effect = reply
            result = paired.run_pair(paired.scenarios()[0], compression_mode='always')
        for mode in ('baseline', 'compressed'):
            self.assertEqual(result[mode]['structured_reports']['Finance'], report)
            self.assertTrue(result[mode]['report_validation']['Finance']['valid'])
        self.assertEqual(result['compressed']['structured_reports']['Compressor'], report)


if __name__ == '__main__':
    unittest.main()
