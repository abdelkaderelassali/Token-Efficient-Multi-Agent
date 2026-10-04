"""Explicit scope review and tri-state supplier capabilities; no NLP certification."""
import json
from datetime import datetime, timezone
import uuid

PROTOCOL = 'validated-v13-requirements'
CAPABILITIES = {
    'refrigerated_transport': 'Refrigerated transport',
    'fragile_handling': 'Fragile-item handling',
    'signature_on_delivery': 'Signature on delivery',
}
REVIEW_FIELDS = {'confirmed', 'required_capabilities', 'other_requirements', 'quote_capabilities'}


def validate_review(value, quote_ids):
    if not isinstance(value, dict) or set(value) != REVIEW_FIELDS:
        raise ValueError('requirement_review needs confirmed, required_capabilities, other_requirements and quote_capabilities')
    if type(value['confirmed']) is not bool:
        raise ValueError('Requirement confirmation must be a boolean')
    required, other, suppliers = (value[k] for k in ('required_capabilities', 'other_requirements', 'quote_capabilities'))
    for name, items in [('required_capabilities', required), ('other_requirements', other)]:
        if not isinstance(items, list) or len(items) > 20 or any(
                not isinstance(s, str) or not s.strip() or len(s) > 500 for s in items):
            raise ValueError(f'{name} must contain at most 20 nonempty strings of at most 500 characters')
        if len(set(items)) != len(items):
            raise ValueError(f'{name} must not contain duplicates')
    if not isinstance(suppliers, dict) or set(suppliers) - set(quote_ids):
        raise ValueError('Supplier capability records must refer to existing quotation IDs')
    for fields in suppliers.values():
        if not isinstance(fields, dict) or set(fields) - set(CAPABILITIES):
            raise ValueError('Unknown supplier capability field')
        if any(v is not None and type(v) is not bool for v in fields.values()):
            raise ValueError('Supplier capabilities must be true, false or null (unknown)')
    return value


def assess_requirements(scenario):
    value = scenario.get('requirement_review')
    custom = scenario.get('size') == 'custom' or str(scenario.get('id', '')).startswith('custom_')
    from paired_benchmark import scenarios
    if value is None and not custom and scenario in scenarios():
        # Built-in fixtures have an explicitly scoped four-constraint objective.
        return {'status': 'ready', 'basis': 'structured_benchmark', 'issues': [],
                'scope': 'Benchmark checks cover budget, deadline, capacity and insurance only.'}
    issues = []
    if value is None:
        issues.append('Review the complete request and supplier terms, then confirm the critical requirements.')
    else:
        validate_review(value, [q['id'] for q in scenario['options']])
        if not value['confirmed']:
            issues.append('Critical requirements and supplier declarations have not been confirmed.')
        for capability in value['required_capabilities']:
            if capability not in CAPABILITIES:
                issues.append(f'Unsupported requirement: {capability}')
                continue
            for quote in scenario['options']:
                answer = value['quote_capabilities'].get(quote['id'], {}).get(capability)
                if answer is None:
                    issues.append(f"{quote['id']}: supplier information missing for {CAPABILITIES[capability]}.")
        issues.extend('Manual review required: ' + condition for condition in value['other_requirements'])
    return {'status': 'needs_review' if issues else 'ready', 'basis': 'user_confirmed_fields' if not issues else 'incomplete_review',
            'issues': issues,
            'scope': 'Python validates declared structured requirements. The user reviews completeness and supplier '
                     'declarations; arbitrary prose is not automatically understood or certified.'}


def blocked_result(scenario, review, protocol, model='llama3', seed=42):
    return {'schema_version': 1, 'experiment_id': str(uuid.uuid4()),
            'created_at': datetime.now(timezone.utc).isoformat(), 'scenario': scenario,
            'configuration': {'pipeline_version': protocol, 'model': model, 'seed': seed,
                              'validation_scope': review['scope']},
            'status': 'needs_review', 'requirement_validation': review,
            'calls': [], 'measured_tokens': 0, 'usage_complete': True,
            'experiment_wall_seconds': 0,
            'reports': {'Decision': 'NEEDS REVIEW\n' + '\n'.join(review['issues'])},
            'measurement_note': 'Stopped before inference. No decision or compression saving is claimed.'}


def review_text(record):
    return ('\nREQUIREMENT REVIEW AND SUPPLIER DECLARATIONS: ' + record.requirement_review_json
            if record.requirement_review_json is not None else '')


def capability_checks(record, quote_id):
    if record.requirement_review_json is None:
        return {}
    review = json.loads(record.requirement_review_json)
    return {'meets_' + key: review['quote_capabilities'].get(quote_id, {}).get(key)
            for key in review['required_capabilities'] if key in CAPABILITIES}
