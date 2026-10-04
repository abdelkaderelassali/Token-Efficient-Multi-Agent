import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import api
from adaptive_workflow import run_pair
from custom_scenarios import FIELDS, normalize_custom
from context_integrity_audit import audit_experiment
from evaluation_suite import summarize
from paired_benchmark import scenarios, evaluate_decision
from protected_facts import ProtectedFacts
from quote_calculations import calculate_quotes
from requirement_review import assess_requirements, CAPABILITIES
from tests.test_guarded_workflow import FakeGuardedRunner


def example():
    source = copy.deepcopy(scenarios()[0])
    payload = {key: source[key] for key in FIELDS}
    payload['options'][1]['arrival_date'] = payload['deadline']
    payload['brief'] += ' Refrigerated transport is mandatory.'
    payload['options'][0]['terms'] += ' Refrigerated transport is included.'
    payload['options'][1]['terms'] += ' Refrigerated transport is not available.'
    payload['requirement_review'] = {'confirmed':True, 'required_capabilities':['refrigerated_transport'],
        'other_requirements':[], 'quote_capabilities':{'S1':{'refrigerated_transport':True},
                                                     'S2':{'refrigerated_transport':False}}}
    return payload


class RequirementReviewTests(unittest.TestCase):
    def test_capability_changes_cheapest_selection_and_survives_context(self):
        scenario = normalize_custom(example())
        result = run_pair(scenario, runner=FakeGuardedRunner(scenario))
        self.assertEqual(result['status'], 'validated', result)
        for mode in ('baseline', 'compression_only', 'compressed'):
            self.assertEqual(result[mode]['evaluation']['decision']['option_id'], 'S1')
            self.assertTrue(result[mode]['evaluation']['checks']['additional_requirements_met'])
        self.assertIn('meets_refrigerated_transport=FAIL', result['compressed']['reports']['Compliance'])
        record = ProtectedFacts.from_scenario(scenario)
        self.assertEqual(record.to_dict()['requirement_review'], scenario['requirement_review'])
        for call in result['compressed']['calls']:
            self.assertIn(record.requirement_review_json, call['input'])
        self.assertTrue(audit_experiment(result)['all_checks_passed'])
        original = record.to_dict()
        scenario['requirement_review']['quote_capabilities']['S2']['refrigerated_transport'] = True
        self.assertEqual(record.to_dict(), original)
        self.assertFalse(audit_experiment(result)['all_checks_passed'])

    def test_no_eligible_capability_returns_valid_hold_not_needs_review(self):
        data = example()
        data['requirement_review']['quote_capabilities']['S1']['refrigerated_transport'] = False
        scenario = normalize_custom(data)
        result = run_pair(scenario, runner=FakeGuardedRunner(scenario))
        self.assertEqual(result['status'], 'validated')
        self.assertEqual(result['compressed']['evaluation']['decision']['verdict'], 'hold')

    def test_each_supported_capability_excludes_a_known_no(self):
        for key in CAPABILITIES:
            data = example()
            data['requirement_review'].update(required_capabilities=[key],
                quote_capabilities={'S1':{key:False}, 'S2':{key:True}})
            audit = calculate_quotes(ProtectedFacts.from_scenario(normalize_custom(data)))
            self.assertEqual(audit['eligible_ids'], ['S2'])
            self.assertFalse(audit['rows'][0]['meets_' + key])

    def test_missing_confirmation_unsupported_and_unknown_never_call_model(self):
        variations = []
        data = example(); data.pop('requirement_review'); variations.append(data)
        data = example(); data['requirement_review']['confirmed'] = False; variations.append(data)
        data = example(); data['requirement_review']['other_requirements'] = ['Maintain 2–8°C']; variations.append(data)
        data = example(); data['requirement_review']['required_capabilities'].append('temperature_2_8'); variations.append(data)
        for answer in (None, 'missing'):
            data = example()
            if answer is None:
                data['requirement_review']['quote_capabilities']['S2']['refrigerated_transport'] = None
            else:
                data['requirement_review']['quote_capabilities'].pop('S2')
            variations.append(data)
        for data in variations:
            with self.subTest(data=data):
                scenario = normalize_custom(data)
                runner = FakeGuardedRunner(scenario)
                result = run_pair(scenario, runner=runner)
                self.assertEqual(result['status'], 'needs_review')
                self.assertEqual(runner.calls, [])
                self.assertNotIn('comparison', result)
                self.assertEqual(result['measured_tokens'], 0)
                self.assertFalse(audit_experiment(result)['all_checks_passed'])
                stats = summarize([result])['llama3']
                self.assertEqual(stats['needs_review_pairs'], 1)
                self.assertEqual(stats['validated_decisions'], 0)
                self.assertEqual(stats['decision_accuracy'], 0)
                self.assertIsNone(stats['mean_applied_reduction_percent'])
                self.assertIsNone(stats['mean_wall_seconds'])
                self.assertEqual(stats['execution_failures'], 0)

    def test_modified_demo_cannot_inherit_curated_scope(self):
        scenario = copy.deepcopy(scenarios()[0])
        scenario['brief'] += ' Refrigeration is mandatory.'
        self.assertEqual(assess_requirements(scenario)['status'], 'needs_review')

    def test_no_extra_rules_still_needs_user_review_for_custom(self):
        data = example()
        data['requirement_review'].update(required_capabilities=[], quote_capabilities={})
        self.assertEqual(assess_requirements(normalize_custom(data))['status'], 'ready')
        data['requirement_review']['confirmed'] = False
        self.assertEqual(assess_requirements(normalize_custom(data))['status'], 'needs_review')

    def test_api_blocks_and_saves_all_modes_without_inference(self):
        client = TestClient(api.app)
        data = example(); data['requirement_review']['confirmed'] = False
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)), \
                patch.object(api, 'run_adaptive_pair') as adaptive, patch.object(api, 'run_lossless_pair') as complete, \
                patch.object(api, 'run_selective_pair') as selective:
            for mode in ('adaptive', 'complete', 'decision'):
                result = client.post('/api/compare', json={'scenario_id':'custom', 'fidelity':mode, 'custom_scenario':data})
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['status'], 'needs_review')
            for runner in (adaptive, complete, selective):
                runner.assert_not_called()
            self.assertEqual(len(list((Path(directory) / 'verification').glob('paired-*.json'))), 3)

    def test_capability_types_and_unknown_quote_ids_are_rejected(self):
        for value in ('yes', 1, [], {}):
            data = example()
            data['requirement_review']['quote_capabilities']['S1']['refrigerated_transport'] = value
            with self.assertRaises(ValueError):
                normalize_custom(data)
        data = example(); data['requirement_review']['quote_capabilities']['UNKNOWN'] = {}
        with self.assertRaises(ValueError):
            normalize_custom(data)

    def test_extra_capabilities_require_adaptive_mode(self):
        client = TestClient(api.app)
        with TemporaryDirectory() as directory, patch.object(api, 'ROOT', Path(directory)), \
                patch.object(api, 'run_lossless_pair') as runner:
            response = client.post('/api/compare', json={'scenario_id':'custom', 'fidelity':'complete', 'custom_scenario':example()})
            self.assertEqual(response.json()['status'], 'needs_review')
            self.assertIn('Adaptive', ' '.join(response.json()['requirement_validation']['issues']))
            runner.assert_not_called()

    def test_unsupported_requirements_cannot_pass_standalone_decision_validation(self):
        scenario = normalize_custom(example())
        valid = run_pair(scenario, runner=FakeGuardedRunner(scenario))
        scenario['requirement_review']['other_requirements'] = ['Temperature must remain 2–8°C']
        evaluation = evaluate_decision(valid['compressed']['calls'][-1]['output'], scenario)
        self.assertFalse(evaluation['all_passed'])
        self.assertFalse(evaluation['checks']['requirements_review_complete'])
