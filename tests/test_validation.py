import copy
import unittest
from unittest.mock import patch

import paired_benchmark as paired
from protected_facts import ProtectedFacts
from validation import validate_decision_structure, validate_compact_facts, validate_report
from test_paired_benchmark import FakeRunner


class ValidationTests(unittest.TestCase):
    def setUp(self):
        # These regression cases explicitly exercise the compression path.
        original_policy = paired.compression_policy
        forced = patch.object(paired, 'compression_policy', side_effect=lambda *a, **kw: original_policy(*a, **dict(kw, mode='always')))
        forced.start()
        self.addCleanup(forced.stop)
        self.source = paired.scenarios()[0]
        self.answer = {"verdict": "proceed", "option_id": "S1", "total_cost_usd": 720,
                       "arrival_date": "2026-10-07", "quantity": 40, "budget_usd": 900,
                       "deadline": "2026-10-08", "reasoning": "Lowest feasible quote", "conditions": []}

    def test_valid_decision_and_hold(self):
        self.assertTrue(validate_decision_structure(self.answer)['valid'])
        hold = dict(self.answer, verdict='hold', option_id=None, total_cost_usd=None, arrival_date=None)
        self.assertTrue(validate_decision_structure(hold)['valid'])

    def test_bad_dates_types_ranges_and_extra_fields_have_errors(self):
        for change in ({'deadline': '2026-02-30'}, {'quantity': 1.5}, {'quantity': True},
                       {'budget_usd': float('nan')}, {'total_cost_usd': -1},
                       {'verdict': 'hold'}, {'option_id': ''}, {'unexpected': 'field'}):
            with self.subTest(change=change):
                result = validate_decision_structure(dict(self.answer, **change))
                self.assertFalse(result['valid'])
                self.assertTrue(result['errors'])

    def test_compact_source_facts_are_verified_not_generated_notes(self):
        record = ProtectedFacts.from_scenario(self.source)
        context = 'AUTHORITATIVE COMPACT FACTS:\n' + record.compact_text() + '\n\nCOMPRESSED ANALYSIS:\nUnknown.'
        self.assertTrue(validate_compact_facts(context, record)['valid'])
        with self.assertRaisesRegex(ValueError, 'protected-facts validation'):
            validate_compact_facts(context.replace('720', '700'), record)

    def test_corrupted_compression_stops_before_downstream_calls(self):
        runner = FakeRunner(self.answer)
        shared = paired.shared_reports(self.source, runner)
        before = len(runner.inputs)
        with patch.object(paired, 'compress_context', return_value=('wrong facts', {}, {})):
            with self.assertRaisesRegex(ValueError, 'protected-facts validation'):
                paired.downstream(self.source, shared, runner, True, 0.28)
        self.assertEqual(len(runner.inputs), before)

    def test_report_format_is_advisory_but_empty_report_is_error(self):
        result = validate_report('Risk', 'Unstructured but available text')
        self.assertFalse(result['valid'])
        self.assertTrue(result['warnings'])
        with self.assertRaises(ValueError):
            validate_report('Risk', ' ')

    def test_validation_is_saved_without_changing_generated_decision(self):
        result = paired.run_pair(self.source, runner=FakeRunner(self.answer))
        for mode in ('baseline', 'compressed'):
            branch = result[mode]
            self.assertTrue(branch['evaluation']['structure_validation']['valid'])
            self.assertIn('Finance', branch['report_validation'])
            self.assertEqual(branch['evaluation']['decision'], self.answer)
        self.assertTrue(result['compressed']['compression']['validation']['valid'])


if __name__ == '__main__':
    unittest.main()
