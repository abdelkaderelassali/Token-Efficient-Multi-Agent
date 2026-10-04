"""Explicit custom source data: no silent LLM extraction or ignored fields."""
from copy import deepcopy
import re

from protected_facts import ProtectedFacts, digest
from paired_benchmark import source_text
from quote_calculations import calculate_quotes

FIELDS = {'title', 'brief', 'route', 'quantity', 'budget_usd', 'deadline', 'require_insurance', 'options'}
QUOTE_FIELDS = {'id', 'carrier', 'method', 'quoted_total_usd', 'arrival_date',
                'capacity_units', 'insurance_included', 'terms'}
OPTIONAL_FIELDS = {'requirement_review'}


def normalize_custom(payload):
    if not isinstance(payload, dict) or not FIELDS.issubset(payload) or set(payload) - FIELDS - OPTIONAL_FIELDS:
        raise ValueError('Custom scenario must contain exactly: ' + ', '.join(sorted(FIELDS)))
    if not isinstance(payload['title'], str) or not 1 <= len(payload['title'].strip()) <= 120:
        raise ValueError('Title must be 1–120 characters')
    options = payload['options']
    if not isinstance(options, list) or not 1 <= len(options) <= 12:
        raise ValueError('Supply between 1 and 12 quotations')
    for offer in options:
        if not isinstance(offer, dict) or set(offer) != QUOTE_FIELDS:
            raise ValueError('Every quotation needs exactly: ' + ', '.join(sorted(QUOTE_FIELDS)))
        if not isinstance(offer['id'], str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,19}', offer['id']):
            raise ValueError('Quote IDs must start with a letter and contain at most 20 letters, digits, hyphens or underscores')
    scenario = deepcopy(payload)
    if 'requirement_review' in scenario:
        from requirement_review import validate_review
        validate_review(scenario['requirement_review'], [q['id'] for q in options])
    # Identity follows the supplied data, so different custom inputs never
    # share cached or persisted comparisons by accident.
    scenario['id'] = 'custom_' + digest(payload)[:24]
    scenario['size'] = 'custom'
    record = ProtectedFacts.from_scenario(scenario)
    if len((source_text(scenario) + calculate_quotes(record)['text']).encode('utf-8')) > 11000:
        raise ValueError('Scenario is too large for the configured local comparison. Shorten the source text or use fewer quotes (11,000 UTF-8 bytes including the Python audit).')
    return scenario
