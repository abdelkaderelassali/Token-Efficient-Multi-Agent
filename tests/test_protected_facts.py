import copy
from dataclasses import FrozenInstanceError
import json
import unittest
from unittest.mock import patch

import paired_benchmark as paired
from protected_facts import ProtectedFacts
from test_paired_benchmark import FakeRunner


class ProtectedFactsTests(unittest.TestCase):
    def setUp(self):
        # These regression cases explicitly exercise the compression path.
        original_policy = paired.compression_policy
        forced = patch.object(paired, 'compression_policy', side_effect=lambda *a, **kw: original_policy(*a, **dict(kw, mode='always')))
        forced.start()
        self.addCleanup(forced.stop)
        self.source = copy.deepcopy(paired.scenarios()[0])

    def test_all_source_fields_survive_json_roundtrip(self):
        record = ProtectedFacts.from_scenario(self.source)
        saved = json.loads(json.dumps(record.to_dict()))
        for key in ('brief', 'route', 'quantity', 'budget_usd', 'deadline', 'require_insurance'):
            self.assertEqual(saved[key], self.source[key])
        self.assertEqual(saved['quotes'], self.source['options'])
        self.assertNotIn('expected_option_ids', saved)
        self.assertEqual(saved, ProtectedFacts.from_scenario(self.source).to_dict())

    def test_source_and_export_mutations_cannot_change_record(self):
        record = ProtectedFacts.from_scenario(self.source)
        before = record.to_dict()
        self.source['options'][0]['quoted_total_usd'] = 1
        record.to_dict()['quotes'][0]['quoted_total_usd'] = 2
        self.assertEqual(record.to_dict(), before)
        with self.assertRaises(FrozenInstanceError):
            record.quotes[0].quoted_total_usd = 3

    def test_invalid_constraints_and_duplicate_ids_are_rejected(self):
        for key, value in [('quantity', True), ('quantity', 0), ('deadline', '2026-02-30'),
                           ('budget_usd', -1), ('budget_usd', float('inf')), ('require_insurance', 'yes')]:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                ProtectedFacts.from_scenario(dict(self.source, **{key: value}))
        self.source['options'][1]['id'] = self.source['options'][0]['id']
        with self.assertRaisesRegex(ValueError, 'unique'):
            ProtectedFacts.from_scenario(self.source)

    def test_both_branches_reference_same_record_and_summary_cannot_rewrite_it(self):
        result = paired.run_pair(self.source, runner=FakeRunner({}))
        record = result['shared']['protected_facts']
        for mode in ('baseline', 'compressed'):
            self.assertEqual(result[mode]['protected_facts_sha256'], record['record_sha256'])
        self.assertEqual(record['quotes'], self.source['options'])
        self.assertIn('S1 | 720 | 2026-10-07', result['compressed']['reports']['Compressor'])

    def test_changed_source_is_rejected_before_downstream_model_calls(self):
        runner = FakeRunner({})
        shared = paired.shared_reports(self.source, runner)
        count = len(runner.inputs)
        self.source['budget_usd'] += 1
        for compressed in (False, True):
            with self.assertRaisesRegex(ValueError, 'facts changed'):
                paired.downstream(self.source, shared, runner, compressed, 0.28)
        self.assertEqual(len(runner.inputs), count)


if __name__ == '__main__':
    unittest.main()
