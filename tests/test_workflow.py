"""Behavioral regression checks; all LLM responses are local test doubles."""
import threading
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import api
import benchmark


class RecordingModel:
    OUTPUTS = {
        "Ingestion": "EXTRACTED: 500 units, budget $15,000",
        "Logistics": "LOGISTICS-ORIGINAL: sea route, delivery uncertain",
        "Finance": "FINANCE-ORIGINAL: cost estimate excludes customs",
        "Compressor": "COMPACT: sea route; customs cost unknown; deadline uncertain",
        "Risk": "RISK-FINDING: delay may breach deadline",
        "Compliance": "COMPLIANCE-FINDING: customs documents missing",
        "Decision": "DECISION: HOLD until customs documents and delivery are confirmed",
    }
    COSTS = {name: index * 10 for index, name in enumerate(OUTPUTS, 1)}

    def __init__(self, fail=None):
        self.calls = []
        self.lock = threading.Lock()
        self.fail = fail

    def invoke(self, messages):
        system = messages[0].content
        role = next((name for name, marker in {
            "Ingestion": "Extract the key entities",
            "Logistics": "Logistics Expert",
            "Finance": "Finance Expert",
            "Compressor": "Context Compressor",
            "Risk": "Risk Assessor",
            "Compliance": "Compliance Officer",
            "Decision": "Decision Manager",
        }.items() if marker in system), None)
        if role is None:
            raise AssertionError(f"Unexpected model call: {system}")
        with self.lock:
            self.calls.append((role, messages[-1].content))
        if role == self.fail:
            raise RuntimeError(f"{role} unavailable")
        return AIMessage(content=self.OUTPUTS[role], response_metadata={
            "prompt_eval_count": self.COSTS[role] - 3, "eval_count": 3,
        })


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        original_policy = benchmark.compression_policy
        forced = patch.object(benchmark, 'compression_policy', side_effect=lambda *a, **kw: original_policy(*a, **dict(kw, mode='always')))
        forced.start()
        self.addCleanup(forced.stop)

    def test_missing_or_invalid_usage_is_not_silently_counted_as_zero(self):
        for metadata in ({}, {"prompt_eval_count": True, "eval_count": 3},
                         {"prompt_eval_count": 4, "eval_count": -1}):
            with self.assertRaises(ValueError):
                benchmark.count_tokens(AIMessage(content="report", response_metadata=metadata))

    def run_graph(self, compressed):
        model = RecordingModel()
        with patch.object(benchmark, "llm", model):
            result = benchmark.build_multi_agent_graph(compressed).invoke(
                benchmark.initial_state("ORIGINAL-TASK: ship 500 units under $15,000")
            )
        return model, result

    def test_all_original_reports_and_final_decision_survive_in_both_modes(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                _, result = self.run_graph(compressed)
                for role in ("Ingestion", "Logistics", "Finance", "Risk", "Compliance", "Decision"):
                    self.assertEqual(result[f"{role.lower()}_report"], RecordingModel.OUTPUTS[role])
                self.assertEqual(result["compressed_context"], RecordingModel.OUTPUTS["Compressor"] if compressed else "")

    def test_decision_runs_once_after_both_reviews_and_receives_their_findings(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                model, _ = self.run_graph(compressed)
                roles = [role for role, _ in model.calls]
                self.assertEqual(roles.count("Decision"), 1)
                self.assertEqual(roles[-1], "Decision")
                for role in ("Risk", "Compliance"):
                    self.assertLess(roles.index(role), roles.index("Decision"))
                context = dict(model.calls)["Decision"]
                for marker in ("ORIGINAL-TASK", "RISK-FINDING", "COMPLIANCE-FINDING"):
                    self.assertIn(marker, context)

    def test_compression_changes_context_not_original_reports(self):
        model, _ = self.run_graph(True)
        calls = dict(model.calls)
        for role in ("Risk", "Compliance", "Decision"):
            self.assertIn("COMPACT:", calls[role])
            self.assertNotIn("LOGISTICS-ORIGINAL", calls[role])
            self.assertNotIn("FINANCE-ORIGINAL", calls[role])
        self.assertIn("LOGISTICS-ORIGINAL", calls["Compressor"])
        self.assertIn("FINANCE-ORIGINAL", calls["Compressor"])

    def test_full_context_reaches_both_reviews_and_decision(self):
        model, _ = self.run_graph(False)
        calls = dict(model.calls)
        self.assertNotIn("Compressor", calls)
        for role in ("Risk", "Compliance", "Decision"):
            self.assertIn("LOGISTICS-ORIGINAL", calls[role])
            self.assertIn("FINANCE-ORIGINAL", calls[role])

    def test_original_request_reaches_specialists_and_reviews_in_both_modes(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                model, _ = self.run_graph(compressed)
                calls = dict(model.calls)
                for role in ("Logistics", "Finance", "Risk", "Compliance", "Decision"):
                    self.assertIn("ORIGINAL-TASK: ship 500 units under $15,000", calls[role])

    def test_empty_report_stops_workflow_before_final_decision(self):
        for role in ("Ingestion", "Logistics", "Finance", "Compressor", "Risk", "Compliance", "Decision"):
            with self.subTest(role=role):
                model = RecordingModel()
                with patch.dict(model.OUTPUTS, {role: " \n "}), patch.object(benchmark, "llm", model):
                    with self.assertRaisesRegex(ValueError, f"{role}: model returned"):
                        benchmark.build_multi_agent_graph(True).invoke(benchmark.initial_state("task"))
                if role != "Decision":
                    self.assertNotIn("Decision", [name for name, _ in model.calls])

    def test_api_rejects_invalid_requests_without_running_workflow(self):
        with TestClient(api.app) as client, patch.object(api, "build_multi_agent_graph") as build:
            for scenario, prompt, status in (("custom", "  ", 422), ("unknown", None, 404)):
                response = client.post("/api/run", json={
                    "scenario_id": scenario, "custom_prompt": prompt, "use_compression": True,
                })
                self.assertEqual(response.status_code, status)
                self.assertIn("detail", response.json())
            build.assert_not_called()

    def test_api_returns_failure_status_instead_of_completed_report(self):
        with TestClient(api.app) as client, patch.object(benchmark, "llm", RecordingModel(fail="Risk")):
            response = client.post("/api/run", json={
                "scenario_id": "custom", "custom_prompt": "Ship goods", "use_compression": True,
            })
        self.assertEqual(response.status_code, 502)
        self.assertIn("Risk unavailable", response.json()["detail"])
        self.assertNotIn("reports", response.json())

    def test_each_agent_runs_once_and_all_tokens_are_counted(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                model, result = self.run_graph(compressed)
                expected = set(RecordingModel.OUTPUTS) - (set() if compressed else {"Compressor"})
                self.assertEqual(len(model.calls), len(expected))
                self.assertEqual({role for role, _ in model.calls}, expected)
                self.assertEqual(result["token_usage"], sum(RecordingModel.COSTS[role] for role in expected))

    def test_review_failure_prevents_an_unreviewed_final_decision(self):
        for compressed in (False, True):
            for failed_role in ("Risk", "Compliance"):
                with self.subTest(compressed=compressed, failed_role=failed_role):
                    model = RecordingModel(fail=failed_role)
                    with patch.object(benchmark, "llm", model):
                        with self.assertRaisesRegex(RuntimeError, f"{failed_role} unavailable"):
                            benchmark.build_multi_agent_graph(compressed).invoke(benchmark.initial_state("task"))
                    self.assertNotIn("Decision", [role for role, _ in model.calls])

    def test_api_returns_distinct_reports_for_both_modes_and_request_types(self):
        with TestClient(api.app) as client:
            for compressed in (False, True):
                for scenario in ("scenario_1", "custom"):
                    with self.subTest(compressed=compressed, scenario=scenario):
                        with patch.object(benchmark, "llm", RecordingModel()):
                            response = client.post("/api/run", json={
                                "scenario_id": scenario, "use_compression": compressed,
                                "custom_prompt": "Ship 500 units under $15,000",
                            })
                        self.assertEqual(response.status_code, 200)
                        reports = response.json()["reports"]
                        for role in ("Ingestion", "Logistics", "Finance", "Risk", "Compliance", "Decision"):
                            self.assertEqual(reports[role], RecordingModel.OUTPUTS[role])
                        self.assertEqual(reports["Compressor"], RecordingModel.OUTPUTS["Compressor"] if compressed else "")


if __name__ == "__main__":
    unittest.main()
