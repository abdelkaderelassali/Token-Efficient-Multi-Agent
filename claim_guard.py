"""Conservative factual boundary: unverified prose never enters another prompt."""
import re
from structured_claims import parse_report


def caveat_catalog(scenario):
    catalog = []
    for offer in scenario['options']:
        for sentence in re.split(r'(?<=[.!?])\s+', offer['terms']):
            if re.search(r'\b(no|not|without|exclude\w*|requir\w*|must|only|fixed|unless|subject|conditional|may|could|pending|uncertain\w*|unconfirmed|unknown|estimated)\b', sentence, re.I):
                catalog.append({'id': f'U{len(catalog)}', 'quote_id': offer['id'], 'detail': sentence})
    return catalog


def expected_claim(audit, quote_id, constraint):
    row = next((r for r in audit['rows'] if r['quote_id'] == quote_id), None)
    if row is None:
        return None
    if constraint == 'eligible':
        return quote_id in audit['eligible_ids']
    if constraint == 'lowest_cost_eligible':
        return quote_id in audit['lowest_cost_eligible_ids']
    return row.get(constraint)


def claim_explanation(quote_id, constraint, result):
    return f'{quote_id}: {constraint}={result.upper()}.'


def guard_report(raw, role, audit, caveats, *, allow_terminal_period_omission=False):
    accepted, rejected = [], []
    try:
        report = parse_report(raw, role)
    except (ValueError, TypeError) as exc:
        return {'valid': False, 'schema_valid': False, 'accepted': [], 'rejected': [
            {'kind': 'report', 'reason': str(exc), 'value': raw}], 'claims_total': 0,
            'claims_correct': 0, 'safe_text': 'No verified claims available.'}
    correct = 0
    for claim in report['claims']:
        expected = expected_claim(audit, claim['quote_id'], claim['constraint'])
        matches = expected is not None and claim['result'] == ('pass' if expected else 'fail')
        correct += int(matches)
        canonical = claim_explanation(claim['quote_id'], claim['constraint'], claim['result'])
        allowed = (canonical, canonical[:-1]) if allow_terminal_period_omission else (canonical,)
        if matches and claim['explanation'] in allowed:
            # Punctuation is not a factual claim. Forward only our canonical
            # rendering, so downstream agents never see unverified prose.
            accepted.append({'kind': 'claim', **claim, 'explanation': canonical})
        else:
            rejected.append({'kind': 'claim', 'value': claim, 'reason':
                             'Contradicts Python or cites unknown facts' if not matches else 'Explanation is not verified canonical wording'})
    for uncertainty in report['uncertainties']:
        found = next((c for c in caveats if c['quote_id'] == uncertainty['quote_id'] and c['detail'] == uncertainty['detail']), None)
        if found:
            accepted.append({'kind': 'uncertainty', **uncertainty, 'source_id': found['id']})
        else:
            rejected.append({'kind': 'uncertainty', 'value': uncertainty, 'reason': 'Not an exact protected source caveat'})
    lines = [v['explanation'] if v['kind'] == 'claim' else f"{v['quote_id']}: {v['detail']}" for v in accepted]
    return {'valid': not rejected, 'schema_valid': True, 'accepted': accepted, 'rejected': rejected,
            'claims_total': len(report['claims']), 'claims_correct': correct,
            'safe_text': '\n'.join(lines) or 'No verified claims available.'}


def decision_reason(audit):
    rec = audit['recommendation']
    if rec['verdict'] == 'hold':
        return 'No supplied quote satisfies every mandatory constraint.'
    reason = (f"{rec['option_id']} is a lowest-cost eligible quote at USD {rec['total_cost_usd']}; "
            f"arrival {rec['arrival_date']} meets deadline {rec['deadline']}, with sufficient capacity and required insurance.")
    additional = [key.removeprefix('meets_') for key in audit['rows'][0] if key.startswith('meets_')]
    if additional:
        reason += ' Confirmed required capabilities: ' + ', '.join(additional) + '.'
    return reason
