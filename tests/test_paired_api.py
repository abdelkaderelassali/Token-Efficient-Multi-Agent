import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import api


class PairedApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_scenario_menu_exposes_demo_and_existing_scenarios(self):
        result = self.client.get('/api/scenarios').json()
        self.assertEqual(result[0]['id'], 'demo_detailed')
        self.assertEqual(result[0]['kind'], 'paired')
        self.assertIn('scenario_1', [item['id'] for item in result])

    def test_unknown_scenario_does_not_invoke_the_model(self):
        with patch.object(api, 'run_lossless_pair') as run:
            response = self.client.post('/api/compare', json={'scenario_id':'unknown'})
            self.assertEqual(response.status_code, 404)
            run.assert_not_called()

    def test_success_is_saved_and_can_be_loaded(self):
        result = {'experiment_id':'test', 'schema_version':1, 'created_at':'2026-09-23T00:00:00Z',
                  'configuration': {'pipeline_version': api.LOSSLESS_PROTOCOL},
                  'scenario':{'id':'demo_short'}}
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)), patch.object(api, 'run_lossless_pair', return_value=result):
            response = self.client.post('/api/compare', json={'scenario_id':'demo_short'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(json.loads((Path(directory)/'verification'/'paired-test.json').read_text()), [result])
            self.assertEqual(self.client.get('/api/compare/latest/demo_short').json(), result)

    def test_missing_or_partial_saved_file_is_handled(self):
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)):
            root = Path(directory)/'verification'; root.mkdir()
            (root/'paired-partial.json').write_text('[')
            self.assertEqual(self.client.get('/api/compare/latest/demo_short').status_code, 404)

    def test_latest_ignores_newer_historical_protocol(self):
        current = {'schema_version': 1, 'created_at': '2026-09-23T00:00:00Z',
                   'configuration': {'pipeline_version': api.LOSSLESS_PROTOCOL}, 'scenario': {'id': 'demo_short'}}
        old = {'schema_version': 1, 'created_at': '2026-09-24T00:00:00Z',
               'configuration': {'pipeline_version': 'validated-v1'}, 'scenario': {'id': 'demo_short'}}
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)):
            root = Path(directory) / 'verification'; root.mkdir()
            (root / 'paired-pilot.json').write_text(json.dumps([current]))
            (root / 'paired-old.json').write_text(json.dumps([old]))
            self.assertEqual(self.client.get('/api/compare/latest/demo_short').json(), current)

    def test_latest_uses_model_configured_for_dashboard(self):
        llama = {'schema_version': 1, 'created_at': '2026-09-23T00:00:00Z',
                 'configuration': {'pipeline_version': api.LOSSLESS_PROTOCOL, 'model': api.PAIRED_MODEL},
                 'scenario': {'id': 'demo_short'}}
        qwen = {'schema_version': 1, 'created_at': '2026-09-24T00:00:00Z',
                'configuration': {'pipeline_version': api.LOSSLESS_PROTOCOL, 'model': 'qwen2.5:7b'},
                'scenario': {'id': 'demo_short'}}
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)):
            root = Path(directory) / 'verification'; root.mkdir()
            (root / 'paired-llama.json').write_text(json.dumps([llama]))
            (root / 'paired-qwen.json').write_text(json.dumps([qwen]))
            self.assertEqual(self.client.get('/api/compare/latest/demo_short').json(), llama)
            self.assertEqual(self.client.get('/api/compare/latest/demo_short?model=qwen2.5:7b').json(), qwen)

    def test_latest_can_load_cli_evaluation_record(self):
        record = {'schema_version': 1, 'created_at': '2026-09-25T00:00:00Z',
                  'configuration': {'pipeline_version': api.LOSSLESS_PROTOCOL, 'model': api.PAIRED_MODEL},
                  'scenario': {'id': 'demo_short'}}
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)):
            root = Path(directory) / 'verification'; root.mkdir()
            (root / 'model-benchmark.json').write_text(json.dumps([record]))
            self.assertEqual(self.client.get('/api/compare/latest/demo_short').json(), record)

    def test_model_results_show_pilot_until_evaluation_exists(self):
        pilot = {'protocol': api.LOSSLESS_PROTOCOL, 'phase': 'pilot', 'created_at': '2026-09-23T00:00:00Z', 'models': {}}
        evaluation = {**pilot, 'phase': 'evaluate', 'created_at': '2026-09-22T00:00:00Z'}
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)):
            root = Path(directory) / 'verification'; root.mkdir()
            (root / 'pilot.summary.json').write_text(json.dumps(pilot))
            self.assertEqual(self.client.get('/api/evaluation/latest').json(), pilot)
            (root / 'eval.summary.json').write_text(json.dumps(evaluation))
            self.assertEqual(self.client.get('/api/evaluation/latest').json(), evaluation)

    def test_model_failure_releases_lock_and_does_not_save_success(self):
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)), patch.object(api, 'run_lossless_pair', side_effect=RuntimeError('Ollama offline')):
            response = self.client.post('/api/compare', json={'scenario_id':'demo_short'})
            self.assertEqual(response.status_code, 502)
            self.assertIn('Ollama offline', response.json()['detail'])
            self.assertFalse(api.comparison_lock.locked())
            self.assertEqual(list(Path(directory).rglob('*.json')), [])

    def test_fidelity_routes_and_loads_protocols_separately(self):
        strict = {'experiment_id': 'strict', 'schema_version': 1,
                  'created_at': '2026-09-29T00:00:00Z',
                  'configuration': {'pipeline_version': api.LOSSLESS_PROTOCOL},
                  'scenario': {'id': 'demo_short'}}
        selective = {'experiment_id': 'selective', 'schema_version': 1,
                     'created_at': '2026-09-29T01:00:00Z',
                     'configuration': {'pipeline_version': api.SELECTIVE_PROTOCOL},
                     'scenario': {'id': 'demo_short'}}
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)), \
             patch.object(api, 'run_lossless_pair', return_value=strict) as lossless, \
             patch.object(api, 'run_selective_pair', return_value=selective) as decision:
            self.assertEqual(self.client.post('/api/compare', json={
                'scenario_id': 'demo_short'}).json(), strict)
            self.assertEqual(self.client.post('/api/compare', json={
                'scenario_id': 'demo_short', 'fidelity': 'decision'}).json(), selective)
            lossless.assert_called_once()
            decision.assert_called_once()
            self.assertEqual(self.client.get('/api/compare/latest/demo_short').json(), strict)
            self.assertEqual(self.client.get(
                '/api/compare/latest/demo_short?fidelity=decision').json(), selective)


if __name__ == '__main__':
    unittest.main()
