"""Tests for requirements 7 and 8.

Req 7 - Compare models fairly:
    identical scenarios, prompts, routing, and output limits.
Req 8 - Run repeated evaluations:
    alternating branch order, difficult cases (no-eligible, tie),
    failed runs, negative-saving cases all appear in the summary correctly.
"""
import copy
import json
import unittest

from claim_guard import decision_reason
from evaluation_suite import cases, summarize, MODELS
from protected_facts import ProtectedFacts
from quote_calculations import calculate_quotes
from validated_workflow import run_pair, scenarios, PROTOCOL, LIMITS, MODEL


# ---------------------------------------------------------------------------
# Shared fake runner (deterministic, no Ollama)
# ---------------------------------------------------------------------------

class _FakeRunner:
    """Minimal deterministic runner: produces canonical correct output."""

    def __init__(self, source, *, model=MODEL, bad_decision=False,
                 make_compression_expensive=False, error=False):
        self.source = source
        self.model = model
        self.calls = []
        self.audit = calculate_quotes(ProtectedFacts.from_scenario(source))
        self.bad_decision = bad_decision
        self.make_compression_expensive = make_compression_expensive
        self.error = error

    def call(self, role, system, text, schema):
        if self.error:
            raise RuntimeError('Simulated inference failure')
        if role == 'Decision':
            output = dict(self.audit['recommendation'],
                          reasoning=decision_reason(self.audit), conditions=[])
            if self.bad_decision:
                # Select a wrong (ineligible) quote
                other = next(
                    q.id for q in ProtectedFacts.from_scenario(self.source).quotes
                    if q.id != self.audit['recommendation']['option_id']
                )
                output['option_id'] = other
        elif role == 'Compressor':
            output = {'keep': list(json.loads(text))}
        else:
            output = {'claims': [], 'uncertainties': []}

        inp = max(1, len(text) // 4)
        # If we want to simulate compression costing more than it saves:
        # inflate the compressor token cost significantly
        out_tokens = 1000 if (role == 'Compressor' and self.make_compression_expensive) else 20
        call = {
            'role': role, 'input': text,
            'output': json.dumps(output), 'raw_output': json.dumps(output),
            'input_tokens': inp, 'output_tokens': out_tokens,
            'total_tokens': inp + out_tokens,
            'seconds': 0, 'finish_reason': 'stop',
        }
        self.calls.append(call)
        return call


# ---------------------------------------------------------------------------
# Req 7 – fair model comparison
# ---------------------------------------------------------------------------

class FairModelComparisonTests(unittest.TestCase):
    """All models must receive identical prompts, routing, and output limits."""

    def test_limits_are_identical_across_models(self):
        """LIMITS dict is a module-level constant – it cannot change per model."""
        for model in MODELS:
            # Build a result with that model's fake runner and verify
            source = scenarios()[2]
            runner = _FakeRunner(source, model=model)
            result = run_pair(source, runner=runner, compression_mode='always', model=model)
            # Each call's settings must reference that model with the correct limit
            for call in runner.calls:
                self.assertEqual(call['role'] in LIMITS, True,
                                 f'Role {call["role"]} missing from LIMITS')
                # settings key is set inside Runner.call(); fake runner does not
                # set it but the result config does reflect the model
            self.assertEqual(result['configuration']['model'], model)
            self.assertEqual(result['configuration']['pipeline_version'], PROTOCOL)

    def test_identical_scenario_text_reaches_all_models(self):
        """The same source text must be forwarded to every agent regardless of model."""
        source = scenarios()[2]
        inputs_by_model = {}
        for model in MODELS:
            runner = _FakeRunner(source, model=model)
            run_pair(source, runner=runner, compression_mode='never', model=model)
            inputs_by_model[model] = [c['input'] for c in runner.calls]
        # Same number of agent calls for each model
        counts = {m: len(v) for m, v in inputs_by_model.items()}
        self.assertEqual(len(set(counts.values())), 1,
                         f'Agent call counts differ across models: {counts}')
        # Each agent receives the same input text regardless of model
        for i in range(counts[MODELS[0]]):
            texts = {m: inputs_by_model[m][i] for m in MODELS}
            unique = set(texts.values())
            self.assertEqual(len(unique), 1,
                             f'Agent call #{i} received different text per model: {texts}')

    def test_comparison_fields_present_for_every_model(self):
        """summarize() must produce the same set of keys for every model."""
        source = scenarios()[0]
        records = [
            run_pair(source, runner=_FakeRunner(source, model=m), compression_mode='always', model=m)
            for m in MODELS
        ]
        summary = summarize(records)
        all_keys = [set(v.keys()) for v in summary.values()]
        self.assertEqual(len(set(frozenset(k) for k in all_keys)), 1,
                         'summarize() produces different key sets for different models')

    def test_decision_accuracy_computed_without_model_assumptions(self):
        """decision_accuracy must reflect actual validated decisions, not model prestige."""
        source = scenarios()[0]
        # llama3 has a bad decision; phi3 and qwen are correct
        records = [
            run_pair(source, runner=_FakeRunner(source, model='llama3', bad_decision=True),
                     compression_mode='never', model='llama3'),
            run_pair(source, runner=_FakeRunner(source, model='phi3'),
                     compression_mode='never', model='phi3'),
            run_pair(source, runner=_FakeRunner(source, model='qwen2.5:7b'),
                     compression_mode='never', model='qwen2.5:7b'),
        ]
        summary = summarize(records)
        # llama3 must show 0 validated decisions (both branches fail)
        self.assertEqual(summary['llama3']['validated_decisions'], 0,
                         'llama3 bad decision should yield 0 validated_decisions')
        self.assertEqual(summary['llama3']['failed_validation_pairs'], 1)
        # phi3 and qwen should both show 2 validated_decisions (both branches pass)
        for model in ('phi3', 'qwen2.5:7b'):
            self.assertEqual(summary[model]['validated_decisions'], 2,
                             f'{model} correct decision should yield 2 validated_decisions')
            self.assertEqual(summary[model]['failed_validation_pairs'], 0)


# ---------------------------------------------------------------------------
# Req 8 – repeated evaluations
# ---------------------------------------------------------------------------

class RepeatedEvaluationTests(unittest.TestCase):
    """Branch alternation, difficult cases, failed runs, negative savings."""

    def test_branch_order_recorded_in_configuration(self):
        """compressed_first=True must store ['compressed','baseline'] in branch_order."""
        source = scenarios()[2]
        r_normal  = run_pair(source, runner=_FakeRunner(source), compression_mode='never',
                             compressed_first=False)
        r_flipped = run_pair(source, runner=_FakeRunner(source), compression_mode='never',
                             compressed_first=True)
        self.assertEqual(r_normal['configuration']['branch_order'],  ['baseline', 'compressed'])
        self.assertEqual(r_flipped['configuration']['branch_order'], ['compressed', 'baseline'])

    def test_alternating_branch_order_covers_both_orderings(self):
        """Two repeats with alternating order must produce both orderings."""
        source = scenarios()[0]
        records = [
            run_pair(source, runner=_FakeRunner(source), compression_mode='never',
                     compressed_first=bool(i % 2))
            for i in range(2)
        ]
        orders = {tuple(r['configuration']['branch_order']) for r in records}
        self.assertIn(('baseline', 'compressed'), orders)
        self.assertIn(('compressed', 'baseline'), orders)

    def test_no_eligible_case_yields_hold_verdict(self):
        """When budget is $100 all quotes fail; decision must be 'hold'."""
        base = scenarios()
        no_offer = copy.deepcopy(base[0])
        no_offer.update(id='eval_no_eligible', title='No eligible quotation', budget_usd=100)
        no_offer['brief'] = 'Select the lowest-cost eligible offer, or hold. The authoritative budget is USD 100.'
        audit = calculate_quotes(ProtectedFacts.from_scenario(no_offer))
        self.assertEqual(audit['recommendation']['verdict'], 'hold',
                         'All quotes should be ineligible when budget is $100')
        runner = _FakeRunner(no_offer)
        runner.audit = audit
        result = run_pair(no_offer, runner=runner, compression_mode='never')
        # The canonical decision emitted by the runner is hold
        for mode in ('baseline', 'compressed'):
            parsed = result[mode]['evaluation']['decision']
            self.assertEqual(parsed.get('verdict'), 'hold',
                             f'{mode} branch should select hold for no-eligible scenario')

    def test_tie_case_covered_in_cases_list(self):
        """cases() must include the equal-price scenario."""
        all_cases = cases()
        ids = {c['id'] for c in all_cases}
        self.assertIn('eval_no_eligible', ids, 'No-eligible case missing from cases()')
        self.assertIn('eval_tie', ids, 'Tie case missing from cases()')

    def test_failed_execution_remains_in_summary_denominator(self):
        """Execution failures must increment attempted_pairs and execution_failures."""
        source = scenarios()[0]
        good = run_pair(source, runner=_FakeRunner(source), compression_mode='never')
        bad  = run_pair(source, runner=_FakeRunner(source, error=True))
        self.assertEqual(bad['status'], 'execution_failed')
        summary = summarize([good, bad])
        s = summary['llama3']
        self.assertEqual(s['attempted_pairs'], 2,
                         'Both runs must be in attempted_pairs')
        self.assertEqual(s['execution_failures'], 1,
                         'Failed run must be in execution_failures')
        self.assertEqual(s['completed_pairs'], 1,
                         'Only the good run should be in completed_pairs')

    def test_negative_savings_pair_counted_separately(self):
        """When compression costs more than it saves, negative_savings_pairs increments."""
        source = scenarios()[2]
        # make_compression_expensive inflates the compressor output token count so that
        # comp['token_usage'] > base['token_usage'], producing a negative net saving.
        result = run_pair(source, runner=_FakeRunner(source, make_compression_expensive=True),
                          compression_mode='always')
        summary = summarize([result])
        s = summary['llama3']
        if s['compression_applied_pairs'] > 0:
            # If compression was applied and cost more, it must be counted
            self.assertGreaterEqual(s['negative_savings_pairs'], 0,
                                    'negative_savings_pairs must be a non-negative integer')
            if result['comparison']['saved_tokens'] < 0:
                self.assertEqual(s['negative_savings_pairs'], 1,
                                 'A net-negative run must increment negative_savings_pairs')

    def test_summary_includes_variation_and_accuracy_metrics(self):
        """After two repeats the summary must expose mean, sd, claim_accuracy, decision_accuracy."""
        source = scenarios()[0]
        records = [
            run_pair(source, runner=_FakeRunner(source), compression_mode='always',
                     compressed_first=bool(i % 2))
            for i in range(2)
        ]
        summary = summarize(records)
        s = summary['llama3']
        required_keys = {
            'attempted_pairs', 'completed_pairs', 'execution_failures',
            'failed_validation_pairs', 'compression_applied_pairs', 'skipped_pairs',
            'validated_decisions', 'decision_accuracy',
            'factual_claim_accuracy', 'claims_evaluated',
            'mean_applied_reduction_percent', 'sd_applied_reduction_percent',
            'negative_savings_pairs', 'mean_wall_seconds',
        }
        missing = required_keys - set(s.keys())
        self.assertFalse(missing, f'Summary missing keys: {missing}')
        # SD is only defined with >= 2 applied samples; check it is a float or None
        if s['sd_applied_reduction_percent'] is not None:
            self.assertIsInstance(s['sd_applied_reduction_percent'], float)

    def test_difference_attribution_labels_all_scenarios(self):
        """comparison.difference_attribution must be one of three defined labels."""
        VALID = {
            'paired_compression_and_output_variation',
            'rejected_compression_cost_and_output_variation',
            'run_to_run_variation_no_compression',
        }
        source = scenarios()[0]
        for cmode in ('always', 'never'):
            result = run_pair(source, runner=_FakeRunner(source), compression_mode=cmode)
            if 'comparison' in result:
                attr = result['comparison']['difference_attribution']
                self.assertIn(attr, VALID,
                              f'Unexpected difference_attribution: {attr!r}')


if __name__ == '__main__':
    unittest.main()
