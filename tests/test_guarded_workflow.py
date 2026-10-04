import copy
import json
import unittest

from claim_guard import guard_report, caveat_catalog, decision_reason
from quote_calculations import calculate_quotes
from protected_facts import ProtectedFacts
from validated_workflow import run_pair, scenarios, PROTOCOL
from measured_policy import fit_profile, choose_policy
from evaluation_suite import summarize

# The exact false prose used in FakeGuardedRunner bad-claim mode.
# D2 costs $12,800 which is within the $15,000 budget, so this sentence
# is factually contradictory and must never influence a downstream agent.
_FALSE_CLAIM_PROSE = '$12800 exceeds $15000'


class FakeGuardedRunner:
    def __init__(self, source, bad_claim=False, bad_reason=False, drop_caveat=False, error=False):
        self.source, self.calls = source, []
        self.audit = calculate_quotes(ProtectedFacts.from_scenario(source))
        self.bad_claim, self.bad_reason, self.drop_caveat, self.error = bad_claim, bad_reason, drop_caveat, error

    def call(self, role, system, text, schema):
        if self.error:
            raise RuntimeError('Inference unavailable')
        if role == 'Decision':
            output = dict(self.audit['recommendation'], reasoning=decision_reason(self.audit), conditions=[])
            if self.bad_reason:
                output['reasoning'] = '$12800 exceeds $15000.'
        elif role == 'Compressor':
            output = {'keep': [] if self.drop_caveat else list(json.loads(text))}
        else:
            output = {'claims': [], 'uncertainties': []}
            if self.bad_claim:
                output['claims'] = [{'quote_id': 'D2', 'constraint': 'within_budget', 'result': 'pass',
                                     'explanation': '$12800 exceeds $15000.'}]
        inp = max(1, len(text) // 4)
        call = {'role': role, 'input': text, 'output': json.dumps(output), 'raw_output': json.dumps(output),
                'input_tokens': inp, 'output_tokens': 20, 'total_tokens': inp + 20,
                'seconds': 0, 'finish_reason': 'stop'}
        self.calls.append(call)
        return call


class GuardedWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.source = scenarios()[2]
        self.audit = calculate_quotes(ProtectedFacts.from_scenario(self.source))

    def test_correct_flag_with_false_prose_is_blocked_and_preserved(self):
        runner = FakeGuardedRunner(self.source, bad_claim=True)
        result = run_pair(self.source, runner=runner, compression_mode='always', profile={})
        for call in runner.calls:
            self.assertNotIn('$12800 exceeds $15000', call['input'])
        finance = result['shared']['calls'][2]
        self.assertIn('$12800 exceeds $15000', finance['raw_output'])
        self.assertEqual(len(finance['claim_validation']['rejected']), 1)

    def test_incorrect_result_unknown_id_and_unverified_uncertainty_are_blocked(self):
        for quote, constraint, value in [('D2', 'within_budget', 'fail'), ('X', 'eligible', 'pass')]:
            report = {'claims': [{'quote_id': quote, 'constraint': constraint, 'result': value,
                                  'explanation': f'{quote}: {constraint}={value.upper()}.'}],
                      'uncertainties': [{'quote_id': None, 'detail': '600 is less than 500.'}]}
            guarded = guard_report(json.dumps(report), 'Risk', self.audit, [])
            self.assertFalse(guarded['valid'])
            self.assertEqual(guarded['accepted'], [])
            self.assertNotIn('600', guarded['safe_text'])

    def test_valid_canonical_claim_and_exact_source_caveat_pass(self):
        caveats = caveat_catalog(self.source)
        item = next(c for c in caveats if len(c['detail'].split()) <= 30)
        report = {'claims': [{'quote_id': 'D2', 'constraint': 'within_budget', 'result': 'pass',
                              'explanation': 'D2: within_budget=PASS.'}],
                  'uncertainties': [{'quote_id': item['quote_id'], 'detail': item['detail']}]}
        self.assertTrue(guard_report(json.dumps(report), 'Risk', self.audit, caveats)['valid'])

    def test_missing_terminal_period_is_canonicalized_without_forwarding_prose(self):
        report = {'claims': [{'quote_id': 'D2', 'constraint': 'within_budget',
                              'result': 'pass', 'explanation': 'D2: within_budget=PASS'}],
                  'uncertainties': []}
        guarded = guard_report(json.dumps(report), 'Risk', self.audit, [],
                               allow_terminal_period_omission=True)
        self.assertTrue(guarded['valid'])
        self.assertEqual(guarded['safe_text'], 'D2: within_budget=PASS.')
        report['claims'][0]['explanation'] = 'D2: within_budget=PASS because 12800 exceeds 15000'
        rejected = guard_report(json.dumps(report), 'Risk', self.audit, [],
                                allow_terminal_period_omission=True)
        self.assertFalse(rejected['valid'])
        self.assertEqual(rejected['safe_text'], 'No verified claims available.')

    def test_invalid_reason_is_blocked_not_silently_replaced(self):
        result = run_pair(self.source, runner=FakeGuardedRunner(self.source, bad_reason=True), compression_mode='never')
        self.assertEqual(result['status'], 'failed_validation')
        for mode in ('baseline', 'compressed'):
            self.assertEqual(result[mode]['status'], 'failed_validation')
            self.assertIn('FAILED VALIDATION', result[mode]['reports']['Decision'])
            self.assertIn('$12800 exceeds $15000', result[mode]['calls'][-1]['raw_output'])
        self.assertFalse(result['comparison']['quality_preserving_savings'])

    def test_invalid_selection_is_blocked_even_with_verified_reason(self):
        runner = FakeGuardedRunner(self.source)
        original = runner.call
        def wrong(*args, **kwargs):
            call = original(*args, **kwargs)
            if args[0] == 'Decision':
                decision = json.loads(call['output'])
                decision['option_id'] = 'D1'
                call['output'] = call['raw_output'] = json.dumps(decision)
            return call
        runner.call = wrong
        result = run_pair(self.source, runner=runner, compression_mode='never')
        self.assertEqual(result['status'], 'failed_validation')
        self.assertIn('lowest_cost_feasible_quote', result['baseline']['evaluation']['failed_checks'])

    def test_uncertainty_markers_remain_in_catalog(self):
        source = copy.deepcopy(self.source)
        source['options'][0]['terms'] = 'Delivery may be delayed. Approval is pending. Capacity is unconfirmed.'
        detail = ' '.join(c['detail'] for c in caveat_catalog(source))
        for phrase in ('Delivery may be delayed.', 'Approval is pending.', 'Capacity is unconfirmed.'):
            self.assertIn(phrase, detail)

    def test_caveat_loss_falls_back_and_charges_compressor(self):
        result = run_pair(self.source, runner=FakeGuardedRunner(self.source, drop_caveat=True), compression_mode='always')
        self.assertFalse(result['compressed']['compression']['applied'])
        self.assertGreater(result['comparison']['compressor_tokens'], 0)
        self.assertFalse(result['comparison']['quality_preserving_savings'])

    def test_skipped_compression_is_not_attributed_as_savings(self):
        result = run_pair(self.source, runner=FakeGuardedRunner(self.source), compression_mode='auto', profile={})
        self.assertFalse(result['comparison']['compression_applied'])
        self.assertEqual(result['comparison']['difference_attribution'], 'run_to_run_variation_no_compression')

    def test_fit_uses_only_same_model_and_protocol(self):
        result = run_pair(self.source, runner=FakeGuardedRunner(self.source), compression_mode='always', model='llama3')
        profile = fit_profile([result], 'llama3', PROTOCOL)
        self.assertIsNotNone(profile)
        self.assertIsNone(fit_profile([result], 'phi3', PROTOCOL))
        self.assertIsNone(fit_profile([result], 'llama3', 'old'))
        policy = choose_policy('auto', 'phi3', PROTOCOL, ['x' * 9000] * 2, 'short', '{}', profile)
        self.assertFalse(policy['should_compress'])

    def test_failures_remain_in_summary_denominator(self):
        failed = run_pair(self.source, runner=FakeGuardedRunner(self.source, error=True))
        self.assertEqual(failed['status'], 'execution_failed')
        summary = summarize([failed])['llama3']
        self.assertEqual(summary['attempted_pairs'], 1)
        self.assertEqual(summary['execution_failures'], 1)
        self.assertEqual(summary['completed_pairs'], 0)

    def test_cost_identity_including_rejected_reports_and_compressor(self):
        result = run_pair(self.source, runner=FakeGuardedRunner(self.source, bad_claim=True), compression_mode='always')
        c = result['comparison']
        self.assertEqual(c['saved_tokens'], c['downstream_input_tokens_saved'] + c['downstream_output_tokens_saved'] - c['compressor_tokens'])
        self.assertGreater(result['compressed']['claim_metrics']['rejected_items'], 0)

    # ------------------------------------------------------------------
    # Requirement 3 – Block contradictory claims
    # Done when: claims such as "$12,800 exceeds $15,000" cannot influence
    # another agent AND rejected reports are preserved in the experiment record.
    # ------------------------------------------------------------------

    def test_rejected_claim_prose_preserved_in_experiment_record(self):
        """The exact false prose must appear in report_validation, not just be counted."""
        result = run_pair(self.source, runner=FakeGuardedRunner(self.source, bad_claim=True),
                          compression_mode='always', profile={})
        # The shared Finance call (index 2) should have its claim_validation stored.
        finance_validation = result['shared']['calls'][2]['claim_validation']
        self.assertFalse(finance_validation['valid'],
                         'A report with a contradictory claim must be marked invalid')
        # The rejected item stores the full claim dict so the prose is auditable.
        rejected_explanations = [r['value']['explanation']
                                 for r in finance_validation['rejected']
                                 if r.get('kind') == 'claim']
        self.assertTrue(
            any(_FALSE_CLAIM_PROSE in exp for exp in rejected_explanations),
            'False prose must be preserved verbatim in the rejected list of report_validation'
        )
        # branch report_validation must also carry the Finance entry
        for mode in ('baseline', 'compressed'):
            branch_rv = result[mode]['report_validation']
            self.assertIn('Finance', branch_rv,
                          f'{mode} branch report_validation must include Finance')
            self.assertFalse(branch_rv['Finance']['valid'],
                             f'{mode} Finance report_validation must remain invalid')

    def test_false_claim_prose_never_reaches_any_downstream_input(self):
        """No agent input may contain the contradictory prose in either branch."""
        runner = FakeGuardedRunner(self.source, bad_claim=True)
        run_pair(self.source, runner=runner, compression_mode='always', profile={})
        for call in runner.calls:
            self.assertNotIn(
                _FALSE_CLAIM_PROSE, call['input'],
                f"False prose reached {call['role']}'s input"
            )

    # ------------------------------------------------------------------
    # Requirement 4 – Make the final decision dependable
    # Done when: invalid selections are clearly blocked and counted as failures.
    # ------------------------------------------------------------------

    def test_invalid_selection_status_and_report_are_explicitly_failed(self):
        """Wrong option_id must yield failed_validation status and a FAILED VALIDATION Decision report."""
        runner = FakeGuardedRunner(self.source)
        original = runner.call
        def wrong(*args, **kwargs):
            call = original(*args, **kwargs)
            if args[0] == 'Decision':
                decision = json.loads(call['output'])
                decision['option_id'] = 'D1'  # D1 is ineligible (late arrival)
                call['output'] = call['raw_output'] = json.dumps(decision)
            return call
        runner.call = wrong
        result = run_pair(self.source, runner=runner, compression_mode='never')
        # Top-level status must be failed_validation, not silently validated.
        self.assertEqual(result['status'], 'failed_validation',
                         'Top-level status must be failed_validation for an invalid selection')
        for mode in ('baseline', 'compressed'):
            branch = result[mode]
            self.assertEqual(branch['status'], 'failed_validation',
                             f'{mode} branch must show failed_validation status')
            # The Decision report entry in reports must be replaced with the failure marker.
            self.assertTrue(
                branch['reports']['Decision'].startswith('FAILED VALIDATION'),
                f'{mode} Decision report must start with FAILED VALIDATION, got: '
                f"{branch['reports']['Decision'][:60]}"
            )
            # failed_checks must name the broken constraint explicitly.
            self.assertIn(
                'lowest_cost_feasible_quote',
                branch['evaluation']['failed_checks'],
                f'{mode} evaluation must list lowest_cost_feasible_quote as a failed check'
            )

    def test_invalid_selection_is_counted_as_failure_in_summary(self):
        """Summary failed_validation_pairs must increment; the pair must never appear as validated."""
        runner = FakeGuardedRunner(self.source)
        original = runner.call
        def wrong(*args, **kwargs):
            call = original(*args, **kwargs)
            if args[0] == 'Decision':
                decision = json.loads(call['output'])
                decision['option_id'] = 'D1'
                call['output'] = call['raw_output'] = json.dumps(decision)
            return call
        runner.call = wrong
        result = run_pair(self.source, runner=runner, compression_mode='never')
        summary = summarize([result])['llama3']
        self.assertEqual(summary['attempted_pairs'], 1)
        self.assertEqual(summary['completed_pairs'], 1,
                         'A failed-validation pair is still a completed pair (execution succeeded)')
        self.assertEqual(summary['failed_validation_pairs'], 1,
                         'Invalid selection must be counted in failed_validation_pairs')
        self.assertEqual(summary['validated_decisions'], 0,
                         'No branch should be counted as validated when the selection is wrong')
        self.assertEqual(summary['decision_accuracy'], 0.0,
                         'decision_accuracy must be 0.0 when all decisions fail validation')


if __name__ == '__main__':
    unittest.main()
