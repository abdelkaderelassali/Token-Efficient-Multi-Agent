"""Source-exact retrieval and deterministic routing for quotation workflows.

Specialists may see selected passages. Risk and Decision always see every term.
No semantic equivalence of omitted specialist prose is claimed.
"""
from dataclasses import replace
import re

from lossless_projection import render_verified_lossless_context

PROTOCOL = 'validated-v11-adaptive'
SHORT_SOURCE_CHARACTERS = 2600
QUERIES = {
    'Logistics': {'arrival', 'delivery', 'collection', 'capacity', 'deadline', 'transport',
                  'route', 'shipment', 'units', 'slot', 'handling'},
    'Finance': {'price', 'cost', 'quoted', 'amount', 'charges', 'customs', 'premium',
                'fee', 'total', 'includes', 'excludes'},
    'Compliance': {'insurance', 'insured', 'customs', 'requirement', 'mandatory',
                   'excludes', 'included', 'documentation'},
}


def passages(record):
    """Partition terms into verbatim spans, retaining offsets for provenance."""
    output = []
    for quote in record.quotes:
        start = 0
        boundaries = [m.end() for m in re.finditer(r'(?<=[.!?])\s+', quote.terms)]
        for end in boundaries + [len(quote.terms)]:
            if end > start:
                output.append({'id': f'{quote.id}:{start}:{end}', 'quote_id': quote.id,
                               'start': start, 'end': end, 'text': quote.terms[start:end]})
            start = end
    return output


def build_plan(record, audit, source):
    full = render_verified_lossless_context(record, audit)
    short = len(record.quotes) <= 2 and len(source) <= SHORT_SOURCE_CHARACTERS
    plan = {'path': 'short' if short else 'long',
            'policy': {'short_max_quotes': 2, 'short_max_source_characters': SHORT_SOURCE_CHARACTERS,
                       'calibration': 'Initial routing heuristic; no saving predicted from length alone.'},
            'skipped_roles': ['Finance', 'Compliance'] if short else [],
            'role_contexts': {}, 'retrieval': {}, 'full_context': full,
            'source_characters': len(source),
            'scope': 'Complete source in Risk and Decision; exact structured facts in every active role. '
                     'Specialist retrieval coverage is not a proof of semantic equivalence. '
                     'Python validates the four structured constraints, not arbitrary prose conditions.'}
    if short:
        return plan
    chunks = passages(record)
    plan['passages'] = chunks
    for role, query in QUERIES.items():
        selected = []
        for quote in record.quotes:
            candidates = [c for c in chunks if c['quote_id'] == quote.id]
            # Local keyword retrieval has no LLM inference cost. Unknown prose
            # remains complete in Risk/Decision; it is never deemed irrelevant.
            score = lambda c: len(set(re.findall(r'[a-z]+', c['text'].lower())) & query)
            ranked = sorted(candidates, key=lambda c: (-score(c), c['start']))
            selected.extend(sorted([c for c in ranked[:2] if score(c) > 0], key=lambda c: c['start']))
        selected_ids = {c['id'] for c in selected}
        quotes = tuple(replace(q, terms='\n'.join(
            f"[{c['id']}] {c['text']}" for c in selected if c['quote_id'] == q.id)
            or '(No specialist passage selected; full terms go to Risk and Decision.)') for q in record.quotes)
        candidate = ('SPECIALIST VIEW: selected verbatim passages; consult Risk and Decision for full terms.\n'
                     + render_verified_lossless_context(replace(record, quotes=quotes), audit))
        use_retrieval = len(candidate) < len(full)
        plan['role_contexts'][role] = candidate if use_retrieval else full
        plan['retrieval'][role] = {
            'applied': use_retrieval, 'selected_ids': [c['id'] for c in selected] if use_retrieval else [c['id'] for c in chunks],
            'omitted_ids': [c['id'] for c in chunks if c['id'] not in selected_ids] if use_retrieval else [],
            'reason': 'Shorter specialist view' if use_retrieval else 'Full projection retained: retrieval was not shorter'}
    return plan


def python_report(role, audit):
    fields = {'Logistics': ('on_time', 'enough_capacity'),
              'Finance': ('within_budget',),
              'Compliance': ('insured_as_required',)}[role]
    if role == 'Compliance':
        fields += tuple(key for key in audit['rows'][0] if key.startswith('meets_'))
    return '\n'.join(f"{row['quote_id']}: {field}={'PASS' if row[field] else 'FAIL'}."
                     for row in audit['rows'] for field in fields)
