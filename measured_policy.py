"""Model-specific policy fitted only to earlier measured paired experiments."""
import statistics


def fit_profile(records, model, protocol):
    records = [r for r in records if r.get('configuration', {}).get('model') == model
               and r['configuration'].get('pipeline_version') == protocol
               and r.get('compressed', {}).get('compression', {}).get('applied')]
    slopes, costs, ids = [], [], []
    for r in records:
        compressor = next((c for c in r['compressed']['calls'] if c['role'] == 'Compressor'), None)
        if not compressor or not compressor.get('total_tokens'):
            continue
        reused_roles = ('Logistics', 'Finance', 'Risk', 'Decision') if protocol in ('validated-v3-early', 'validated-v4-selective', 'validated-v5-graph') else ('Risk', 'Decision')
        for role in reused_roles:
            a = next(c for c in r['baseline']['calls'] if c['role'] == role)
            b = next(c for c in r['compressed']['calls'] if c['role'] == role)
            diff = len(a['input']) - len(b['input'])
            if diff > 0 and a['input_tokens'] > b['input_tokens']:
                slopes.append((a['input_tokens'] - b['input_tokens']) / diff)
        costs.append((compressor['total_tokens'], len(compressor['input'])))
        ids.append(r['experiment_id'])
    if not slopes or not costs:
        return None
    return {'model': model, 'protocol': protocol, 'sample_count': len(ids), 'experiment_ids': ids,
            'tokens_per_removed_character': statistics.median(slopes),
            'compressor_tokens': statistics.median(c[0] for c in costs),
            'compressor_input_characters': statistics.median(c[1] for c in costs),
            'safety_multiplier': 1.2, 'minimum_expected_saving': 100,
            'scope': 'Small pilot calibration; held-out results must verify generalization.'}


def choose_policy(mode, model, protocol, full_contexts, compact_context, compressor_input, profile):
    if mode not in ('auto', 'always', 'never'):
        raise ValueError('Invalid compression mode')
    usable = profile and profile.get('model') == model and profile.get('protocol') == protocol
    prediction, cost = None, None
    if usable:
        removed = sum(len(value) - len(compact_context) for value in full_contexts)
        scale = max(1, len(compressor_input) / max(1, profile['compressor_input_characters']))
        cost = profile['compressor_tokens'] * scale * profile['safety_multiplier']
        prediction = removed * profile['tokens_per_removed_character'] - cost
    apply = mode == 'always' or (mode == 'auto' and usable and prediction >= profile['minimum_expected_saving'])
    return {'mode': mode, 'applied': False, 'should_compress': bool(apply), 'compressor_called': False,
            'estimated_net_tokens_saved': round(prediction, 2) if prediction is not None else None,
            'estimated_compressor_tokens': round(cost, 2) if cost is not None else None,
            'calibration': profile if usable else None,
            'reason': 'Forced pilot' if mode == 'always' else 'Disabled' if mode == 'never' else
                      'No compatible measured calibration; skipped' if not usable else
                      'Predicted savings exceed cost and margin' if apply else 'Predicted savings below margin'}
