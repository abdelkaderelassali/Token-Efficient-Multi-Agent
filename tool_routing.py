"""Compression plus deterministic tools for structured quotation reviews.

This reduces LLM work, not the source available to the remaining reviewers.
It does not interpret arbitrary constraints written only in supplier prose.
"""
from adaptive_context import SHORT_SOURCE_CHARACTERS
from lossless_projection import render_verified_lossless_context

PROTOCOL = 'validated-v12-tools'


def build_plan(record, audit, source):
    short = len(record.quotes) <= 2 and len(source) <= SHORT_SOURCE_CHARACTERS
    plan = {
        'path': 'short' if short else 'long',
        'policy': {
            'short_max_quotes': 2,
            'short_max_source_characters': SHORT_SOURCE_CHARACTERS,
            'calibration': 'Explicit routing heuristic; no minimum saving is predicted.',
            'tool_fields': {
                'Logistics': ['on_time', 'enough_capacity'],
                'Finance': ['within_budget'],
                'Compliance': ['insured_as_required'],
            },
        },
        'skipped_roles': ['Finance', 'Compliance'] if short else ['Logistics', 'Finance', 'Compliance'],
        'compact_handoff': not short,
        'role_contexts': {},
        'retrieval': {},
        'full_context': render_verified_lossless_context(record, audit),
        'source_characters': len(source),
        'scope': 'Complete brief, every structured quote field and all supplier terms reach every active LLM role. '
                 'Python tools replace structured constraint checks, not interpretation of arbitrary prose. '
                 'Final validation covers the four structured constraints and lowest eligible price; '
                 'it does not certify all narrative requirements or generated prose.',
    }
    if record.requirement_review_json is not None:
        plan['scope'] = ('Complete source and declared requirements reach every active LLM. '
                         'Python checks budget, deadline, capacity, insurance and required supported capabilities. '
                         'Completeness and supplier declarations require human review; arbitrary prose is not certified.')
    return plan
