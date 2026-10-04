import copy
import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

import paired_benchmark as paired


class FakeRunner:
    def __init__(self, answer):
        self.answer = answer
        self.inputs = []

    def call(self, role, system, text, max_output=384, json_output=False):
        self.inputs.append((role, system, text, max_output, json_output))
        output = json.dumps(self.answer) if role == "Decision" else f"{role} findings"
        return {"role": role, "input_tokens": 100, "output_tokens": 20,
                "total_tokens": 120, "seconds": 0, "output": output,
                "input_sha256": "test", "finish_reason": "stop"}


class PairedBenchmarkTests(unittest.TestCase):
    def setUp(self):
        # These regression cases explicitly exercise the compression path.
        original_policy = paired.compression_policy
        forced = patch.object(paired, 'compression_policy', side_effect=lambda *a, **kw: original_policy(*a, **dict(kw, mode='always')))
        forced.start()
        self.addCleanup(forced.stop)
        self.scenario = paired.scenarios()[0]
        self.answer = {"verdict": "proceed", "option_id": "S1", "total_cost_usd": 720,
                       "arrival_date": "2026-10-07", "quantity": 40, "budget_usd": 900,
                       "deadline": "2026-10-08", "reasoning": "Lowest feasible quote", "conditions": []}

    def test_shared_work_executes_once_and_is_charged_to_both_branches(self):
        runner = FakeRunner(self.answer)
        result = paired.run_pair(self.scenario, runner=runner)
        roles = [item[0] for item in runner.inputs]
        self.assertEqual(len(roles), 10)
        for role in ("Ingestion", "Logistics", "Finance", "Compressor"):
            self.assertEqual(roles.count(role), 1)
        for role in ("Risk", "Compliance", "Decision"):
            self.assertEqual(roles.count(role), 2)
        self.assertEqual(result["shared"]["token_usage"], 360)
        self.assertEqual(result["baseline"]["token_usage"], 720)
        self.assertEqual(result["compressed"]["token_usage"], 840)
        self.assertEqual(result["comparison"]["total_experiment_tokens"], 1200)
        self.assertEqual(result["comparison"]["saved_tokens"], -120)
        self.assertLess(result["comparison"]["reduction_percent"], 0)
        comparison = result["comparison"]
        self.assertEqual(comparison["saved_tokens"], comparison["downstream_input_tokens_saved"]
                         + comparison["downstream_output_tokens_saved"] - comparison["compressor_tokens"])
        self.assertFalse(comparison["quality_preserving_savings"])
        self.assertEqual(comparison["token_breakdown"]["shared"]["input_tokens"], 300)

    def test_same_upstream_reports_and_same_downstream_prompts(self):
        runner = FakeRunner(self.answer)
        result = paired.run_pair(self.scenario, runner=runner)
        self.assertEqual(result["baseline"]["shared_context_sha256"], result["compressed"]["shared_context_sha256"])
        for role in ("Ingestion", "Logistics", "Finance"):
            self.assertEqual(result["baseline"]["reports"][role], result["compressed"]["reports"][role])
        for role in ("Risk", "Compliance", "Decision"):
            calls = [item for item in runner.inputs if item[0] == role]
            self.assertEqual(calls[0][1], calls[1][1])
            self.assertEqual(calls[0][3:], calls[1][3:])
        compressor_input = next(item[2] for item in runner.inputs if item[0] == "Compressor")
        self.assertEqual(compressor_input, result["shared"]["context"])

    def test_reversed_branch_order_still_produces_correct_accounting(self):
        result = paired.run_pair(self.scenario, runner=FakeRunner(self.answer), compressed_first=True)
        self.assertEqual(result["configuration"]["branch_order"], ["compressed", "baseline"])
        self.assertEqual(result["baseline"]["token_usage"], 720)
        self.assertEqual(result["compressed"]["token_usage"], 840)

    def test_compressor_preserves_all_quote_fields_without_injecting_answer(self):
        scenario = paired.scenarios()[2]
        before = copy.deepcopy(scenario)
        text = paired.protected_facts(scenario)
        for quote in scenario["options"]:
            self.assertIn(f"{quote['id']} | {quote['quoted_total_usd']} | {quote['arrival_date']} | {quote['capacity_units']} | {str(quote['insurance_included']).lower()}", text)
        self.assertIn(str(scenario["budget_usd"]), text)
        self.assertIn(scenario["deadline"], text)
        self.assertNotIn("expected_option", text)
        self.assertEqual(before, scenario)

    def test_truncated_summary_drops_incomplete_sentence_but_keeps_cost_and_facts(self):
        runner = FakeRunner(self.answer)
        original = runner.call
        def truncated(*args, **kwargs):
            result = original(*args, **kwargs)
            result.update(output='A complete observation. An unfinished claim that', finish_reason='length')
            return result
        runner.call = truncated
        shared = {'context':'source '*100, 'context_sha256':'known'}
        context, call, stats = paired.compress_context(self.scenario, shared, runner)
        self.assertIn('A complete observation.', context)
        self.assertNotIn('An unfinished claim', context)
        self.assertIn('S1 | 720 | 2026-10-07', context)
        self.assertEqual(call['total_tokens'], 120)
        self.assertIn('unfinished', call['output'])
        self.assertTrue(stats['summary_truncated'])

    def test_correct_answer_passes_all_checks(self):
        result = paired.evaluate_decision(json.dumps(self.answer), self.scenario)
        self.assertTrue(result["all_passed"])
        self.assertEqual(result["passed"], 12)

    def test_audit_identifies_real_constraint_failures_and_python_recommendation(self):
        audit = paired.constraint_audit(paired.scenarios()[2])
        rows = {row['quote_id']: row for row in audit['rows']}
        self.assertFalse(rows['D1']['on_time'])
        self.assertFalse(rows['D3']['within_budget'])
        self.assertFalse(rows['D4']['insured_as_required'])
        self.assertFalse(rows['D5']['enough_capacity'])
        self.assertTrue(all(value for key, value in rows['D2'].items() if key != 'quote_id'))
        self.assertTrue(all(value for key, value in rows['D6'].items() if key != 'quote_id'))
        self.assertNotIn('expected_option', audit['text'])
        self.assertNotIn('BY PRICE', audit['text'])
        self.assertNotIn('Select the first', audit['text'])
        self.assertEqual(audit['lowest_cost_eligible_ids'], ['D2'])
        self.assertEqual(audit['recommendation']['option_id'], 'D2')

    def test_wrong_price_late_quote_and_invented_quote_fail(self):
        for changed in ({"total_cost_usd": 721}, {"option_id": "S2"}, {"option_id": "made-up"},
                        {"deadline": "2027-01-01"}, {"quantity": True}):
            with self.subTest(changed=changed):
                answer = dict(self.answer, **changed)
                self.assertFalse(paired.evaluate_decision(json.dumps(answer), self.scenario)["all_passed"])

    def test_malformed_answer_is_not_a_quality_success(self):
        for text in ("not JSON", "[]", "{}", '{"verdict": "proceed"}'):
            with self.subTest(text=text):
                self.assertFalse(paired.evaluate_decision(text, self.scenario)["all_passed"])

    def test_structure_requires_reasoning_and_typed_conditions(self):
        for change in ({"reasoning": " "}, {"conditions": "none"}, {"conditions": [3]}, {"quantity": True}):
            result = paired.evaluate_decision(json.dumps(dict(self.answer, **change)), self.scenario)
            self.assertFalse(result["checks"]["structured_answer"])
            self.assertIn("structured_answer", result["failed_checks"])
        answer = dict(self.answer)
        del answer["reasoning"]
        self.assertFalse(paired.evaluate_decision(json.dumps(answer), self.scenario)["all_passed"])

    def test_duplicate_keys_and_nonfinite_json_are_rejected(self):
        for text in ('{"verdict":"hold","verdict":"proceed"}', '{"quantity":NaN}', '{"quantity":Infinity}'):
            result = paired.evaluate_decision(text, self.scenario)
            self.assertFalse(result["checks"]["structured_answer"])
            self.assertEqual(result["decision"], {})

    def test_zero_usage_has_no_undefined_percentage_or_success_claim(self):
        runner = FakeRunner(self.answer)
        original = runner.call
        def zero(*args, **kwargs):
            result = original(*args, **kwargs)
            result.update(input_tokens=0, output_tokens=0, total_tokens=0, finish_reason="length")
            return result
        runner.call = zero
        comparison = paired.run_pair(self.scenario, runner=runner)["comparison"]
        self.assertIsNone(comparison["reduction_percent"])
        self.assertIsNone(comparison["downstream_input_reduction_percent"])
        self.assertFalse(comparison["quality_preserving_savings"])
        self.assertIn("compressed/Decision", comparison["truncated_calls"])

    def test_token_measurements_reject_booleans_negative_and_fractional_counts(self):
        for value in (True, -1, 1.5):
            with patch.object(paired, "ChatOllama") as model:
                response = AIMessage(content="report", response_metadata={"prompt_eval_count": value, "eval_count": 2})
                model.return_value.invoke.return_value = response
                with self.assertRaises(ValueError):
                    paired.ModelRunner().call("Risk", "system", "context")

    def test_no_feasible_quote_requires_hold(self):
        scenario = copy.deepcopy(self.scenario)
        scenario["budget_usd"] = 100
        answer = dict(self.answer, verdict="hold", option_id=None, total_cost_usd=None,
                      arrival_date=None, budget_usd=100)
        self.assertTrue(paired.evaluate_decision(json.dumps(answer), scenario)["all_passed"])
        self.assertFalse(paired.evaluate_decision(json.dumps(self.answer), scenario)["all_passed"])

    def test_missing_usage_cannot_be_reported_as_zero_cost(self):
        with patch.object(paired, "ChatOllama") as model:
            model.return_value.invoke.return_value = AIMessage(content="report", response_metadata={})
            with self.assertRaisesRegex(ValueError, "missing Ollama token measurement"):
                paired.ModelRunner().call("Risk", "system", "context")

    def test_measured_input_and_output_tokens_are_both_counted(self):
        with patch.object(paired, "ChatOllama") as model:
            model.return_value.invoke.return_value = AIMessage(content='{"claims":[],"uncertainties":[]}', response_metadata={
                "prompt_eval_count": 101, "eval_count": 23, "done_reason": "stop"})
            call = paired.ModelRunner().call("Risk", "system", "context")
            self.assertEqual(call["total_tokens"], 124)

    def test_empty_model_report_is_rejected_even_with_valid_usage(self):
        with patch.object(paired, "ChatOllama") as model:
            model.return_value.invoke.return_value = AIMessage(content="  ", response_metadata={
                "prompt_eval_count": 101, "eval_count": 2})
            with self.assertRaisesRegex(ValueError, "Risk: model returned an empty"):
                paired.ModelRunner().call("Risk", "system", "context")


if __name__ == "__main__":
    unittest.main()
