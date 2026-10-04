import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient

import api
from custom_scenarios import FIELDS, normalize_custom
from paired_benchmark import scenarios
from adaptive_workflow import run_pair
from tests.test_guarded_workflow import FakeGuardedRunner


class CustomScenarioTests(unittest.TestCase):
    def setUp(self):
        self.payload = {key: copy.deepcopy(scenarios()[0][key]) for key in FIELDS}
        self.client = TestClient(api.app)

    def test_exact_source_and_identity_are_preserved(self):
        scenario = normalize_custom(self.payload)
        for key in FIELDS:
            self.assertEqual(scenario[key], self.payload[key])
        self.assertEqual(normalize_custom(self.payload)['id'], scenario['id'])
        self.payload['options'][0]['terms'] += ' A signature is required.'
        self.assertNotEqual(normalize_custom(self.payload)['id'], scenario['id'])

    def test_bad_source_is_rejected_before_any_model_call(self):
        for field, value in [('quantity', True), ('budget_usd', float('inf')), ('deadline', 'tomorrow'), ('new_requirement', 'unknown')]:
            bad = copy.deepcopy(self.payload); bad[field] = value
            with self.assertRaises((ValueError, TypeError)):
                normalize_custom(bad)
        self.payload['options'][1]['id'] = self.payload['options'][0]['id']
        with patch.object(api, 'run_adaptive_pair') as run:
            response = self.client.post('/api/compare', json={'scenario_id': 'custom', 'custom_scenario': self.payload, 'fidelity': 'adaptive'})
            self.assertEqual(response.status_code, 422)
            run.assert_not_called()

    def test_custom_comparison_saved_and_audited(self):
        self.payload['requirement_review'] = {'confirmed': True, 'required_capabilities': [],
                                              'other_requirements': [], 'quote_capabilities': {}}
        def fake_run(scenario, **kwargs):
            return run_pair(scenario, runner=FakeGuardedRunner(scenario), **kwargs)
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)), patch.object(api, 'run_adaptive_pair', side_effect=fake_run):
            response = self.client.post('/api/compare', json={'scenario_id': 'custom', 'custom_scenario': self.payload, 'fidelity': 'adaptive'})
            self.assertEqual(response.status_code, 200, response.text)
            result = response.json()
            self.assertEqual(result['status'], 'validated')
            self.assertTrue(result['context_audit']['all_checks_passed'])
            saved = self.client.get('/api/compare/latest/' + result['scenario']['id'] + '?fidelity=adaptive')
            self.assertEqual(saved.json()['experiment_id'], result['experiment_id'])
            self.assertTrue(list((Path(directory) / 'verification').glob('paired-*.json')))

    def test_template_and_no_silent_mixed_inputs(self):
        self.assertEqual(self.client.get('/api/compare/template/demo_short').json(), self.payload)
        self.assertEqual(self.client.post('/api/compare', json={'scenario_id': 'demo_short', 'custom_scenario': self.payload}).status_code, 422)
        self.assertEqual(self.client.post('/api/compare', json={'scenario_id': 'custom'}).status_code, 422)
