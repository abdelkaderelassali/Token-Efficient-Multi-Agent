import json
import unittest

from claim_guard import caveat_catalog
from selective_workflow import run_pair, scenarios
from tests.test_guarded_workflow import FakeGuardedRunner


class SelectiveRunner(FakeGuardedRunner):
    def __init__(self, source, omit_required=False):
        super().__init__(source)
        self.omit_required = omit_required

    def call(self, role, system, text, schema):
        call = super().call(role, system, text, schema)
        if role == 'Compressor':
            eligible = set(self.audit['eligible_ids'])
            caveats = caveat_catalog(self.source)
            required = [c['id'] for c in caveats if not eligible or c['quote_id'] in eligible]
            optional = [c['id'] for c in caveats if c['id'] not in required]
            keep = required[1:] if self.omit_required else required + optional[:2]
            call['output'] = call['raw_output'] = json.dumps({'keep': keep})
        return call


class SelectiveWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.scenario = scenarios()[2]

    def test_required_caveats_survive_and_optional_caveats_can_be_removed(self):
        result = run_pair(self.scenario, runner=SelectiveRunner(self.scenario),
                          compression_mode='always', profile={})
        self.assertEqual(result['status'], 'validated')
        self.assertTrue(result['comparison']['compression_applied'])
        compressor = next(c for c in result['compressed']['calls'] if c['role'] == 'Compressor')
        validation = compressor['compression_validation']
        self.assertTrue(validation['valid'])
        self.assertTrue(validation['optional_omitted_ids'])
        for caveat in caveat_catalog(self.scenario):
            if caveat['id'] in validation['required_caveat_ids']:
                self.assertIn(caveat['detail'], result['compressed']['reports']['Compressor'])
        self.assertEqual(len(result['shared']['protected_caveats']), len(caveat_catalog(self.scenario)))

    def test_missing_required_caveat_restores_full_context_and_charges_call(self):
        result = run_pair(self.scenario, runner=SelectiveRunner(self.scenario, omit_required=True),
                          compression_mode='always', profile={})
        self.assertFalse(result['comparison']['compression_applied'])
        self.assertGreater(result['comparison']['compressor_tokens'], 0)
        for role in ('Logistics', 'Finance', 'Risk', 'Decision'):
            baseline = next(c for c in result['baseline']['calls'] if c['role'] == role)
            compressed = next(c for c in result['compressed']['calls'] if c['role'] == role)
            self.assertEqual(compressed['input'], baseline['input'])


if __name__ == '__main__':
    unittest.main()
