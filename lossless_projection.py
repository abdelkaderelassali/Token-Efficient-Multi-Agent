"""Exact source-field projection for downstream prompts.

Every narrative source field is copied verbatim. Only repetitive labels and
formatting are shortened. The verified projection includes compact Python
constraint checks in the same downstream context.
"""


def render_lossless_context(record):
    lines = [
        'BRIEF: ' + record.brief,
        (f'ROUTE: {record.route}; QUANTITY: {record.quantity}; '
         f'BUDGET USD: {record.budget_usd}; DEADLINE: {record.deadline}; '
         f'INSURANCE REQUIRED: {str(record.require_insurance).lower()}'),
        'QUOTES: ID | carrier | method | total USD | arrival | capacity | insured',
    ]
    for quote in record.quotes:
        lines.append(
            f'{quote.id} | {quote.carrier} | {quote.method} | '
            f'{quote.quoted_total_usd} | {quote.arrival_date} | '
            f'{quote.capacity_units} | {str(quote.insurance_included).lower()}')
        lines.append('TERMS: ' + quote.terms)
    from requirement_review import review_text
    return '\n'.join(lines) + review_text(record)


def render_verified_lossless_context(record, audit):
    """Copy every source field verbatim while serializing Python checks once.

    The complete supplier prose remains intact. Derived margins are omitted
    from the prompt because Python already enforces them; source prices,
    dates, capacity and requirements remain visible.
    """
    lines = [
        'BRIEF: ' + record.brief,
        (f'REQUIREMENTS: route={record.route}; quantity={record.quantity}; '
         f'budget_usd={record.budget_usd}; deadline={record.deadline}; '
         f'insurance_required={str(record.require_insurance).lower()}'),
        'QUOTES: ID | carrier | method | total USD | arrival | capacity | insured',
    ]
    for quote in record.quotes:
        lines.append(
            f'{quote.id}|{quote.carrier}|{quote.method}|'
            f'{quote.quoted_total_usd}|{quote.arrival_date}|'
            f'{quote.capacity_units}|{str(quote.insurance_included).lower()}')
        lines.append('TERMS: ' + quote.terms)
    lines.append('PYTHON CHECKS (P=PASS, F=FAIL; authoritative):')
    for row in audit['rows']:
        checks = ','.join(
            f"{name}={'P' if row[name] else 'F'}"
            for name in row if name != 'quote_id')
        lines.append(f"{row['quote_id']}|{checks}")
    rec = audit['recommendation']
    lines.append('ELIGIBLE: ' + (','.join(audit['eligible_ids']) or 'NONE'))
    lines.append('LOWEST-COST ELIGIBLE: ' + (
        ','.join(audit['lowest_cost_eligible_ids']) or 'NONE'))
    lines.append(
        'PYTHON RECOMMENDATION: ' +
        (f"proceed with {rec['option_id']}, USD {rec['total_cost_usd']}, "
         f"arrival {rec['arrival_date']}" if rec['option_id'] else
         'hold; no eligible quote'))
    from requirement_review import review_text
    return '\n'.join(lines) + review_text(record)
