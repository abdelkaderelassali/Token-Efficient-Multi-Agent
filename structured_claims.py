"""Strict claim syntax. Factual verification is a separate validation stage."""
import json

CONSTRAINTS = ("within_budget", "on_time", "enough_capacity", "insured_as_required",
               "eligible", "lowest_cost_eligible")
SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claims", "uncertainties"],
    "properties": {
        "claims": {"type": "array", "maxItems": 12, "items": {
            "type": "object", "additionalProperties": False,
            "required": ["quote_id", "constraint", "result", "explanation"],
            "properties": {"quote_id": {"type": "string", "minLength": 1},
                           "constraint": {"type": "string", "enum": list(CONSTRAINTS)},
                           "result": {"type": "string", "enum": ["pass", "fail", "unknown"]},
                           "explanation": {"type": "string", "minLength": 1}}}},
        "uncertainties": {"type": "array", "maxItems": 6, "items": {
            "type": "object", "additionalProperties": False, "required": ["quote_id", "detail"],
            "properties": {"quote_id": {"type": ["string", "null"]},
                           "detail": {"type": "string", "minLength": 1}}}},
    },
}


def instructions(role):
    compressor = (" For Compressor, use claims=[] and record only unresolved issues in uncertainties; "
                  "do not repeat the protected quote table.") if role == "Compressor" else ""
    return (
        " Return only JSON matching the provided schema, with claims and uncertainties arrays. "
        "Every factual comparison must be a claim with the exact supplied quote_id, a constraint "
        f"from {', '.join(CONSTRAINTS)}, result pass/fail/unknown, and a short explanation. "
        "Use Python calculations as authoritative. Each explanation or uncertainty must be at most "
        "30 words. State missing evidence explicitly; never interpret unknown as pass. "
        "Use at most 12 relevant claims and 6 uncertainties. Do not duplicate a quote/constraint pair. "
        "An uncertainty has quote_id (null only for a general issue) and detail. "
        "Use empty arrays when no relevant claims or uncertainties exist. No Markdown or extra fields."
        + compressor
    )


def parse_report(raw, role):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    data = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    if not isinstance(data, dict) or set(data) != {"claims", "uncertainties"}:
        raise ValueError("Report must contain only claims and uncertainties")
    if not isinstance(data['claims'], list) or len(data['claims']) > 12:
        raise ValueError("claims must be an array of at most 12 items")
    if not isinstance(data['uncertainties'], list) or len(data['uncertainties']) > 6:
        raise ValueError("uncertainties must be an array of at most 6 items")

    def short_text(value):
        return isinstance(value, str) and bool(value.strip()) and len(value.split()) <= 30

    seen = set()
    for claim in data['claims']:
        if not isinstance(claim, dict) or set(claim) != {'quote_id', 'constraint', 'result', 'explanation'}:
            raise ValueError("Invalid claim fields")
        if not short_text(claim['quote_id']) or not short_text(claim['explanation']):
            raise ValueError("Claim ID and explanation must be nonempty, concise strings")
        if claim['constraint'] not in CONSTRAINTS or claim['result'] not in ('pass', 'fail', 'unknown'):
            raise ValueError("Invalid constraint or result")
        key = (claim['quote_id'], claim['constraint'])
        if key in seen:
            raise ValueError("Duplicate quote/constraint claim")
        seen.add(key)
    for uncertainty in data['uncertainties']:
        if not isinstance(uncertainty, dict) or set(uncertainty) != {'quote_id', 'detail'}:
            raise ValueError("Invalid uncertainty fields")
        if (uncertainty['quote_id'] is not None and not short_text(uncertainty['quote_id'])) or not short_text(uncertainty['detail']):
            raise ValueError("Uncertainty needs a valid quote ID or null and concise detail")
    if role == 'Compressor' and data['claims']:
        raise ValueError("Compressor must only report uncertainties; protected facts are supplied separately")
    return data


def render_report(data):
    lines = ['## Claims']
    lines.extend(f"- {c['quote_id']} | {c['constraint']} | {c['result']}: {c['explanation']}" for c in data['claims'])
    if not data['claims']:
        lines.append('- No claims returned.')
    lines.append('## Uncertainties')
    lines.extend(f"- {u['quote_id'] or 'General'}: {u['detail']}" for u in data['uncertainties'])
    if not data['uncertainties']:
        lines.append('- No uncertainties reported by the model.')
    return '\n'.join(lines)
