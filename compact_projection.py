"""A single source-backed projection of the brief, quote facts and Python checks."""


def render_review_context(record, audit, catalog, selected_ids):
    """Render every structured source fact once, plus validated source caveats.

    This is a prompt projection, not the protected record. The original source,
    full quote terms and raw agent history remain in LangGraph state for audit.
    """
    lines = [
        f'TASK: Choose the lowest-cost feasible single quote for {record.quantity} units, '
        f'{record.route}; budget USD {record.budget_usd}; deadline {record.deadline}; '
        f'insurance required={str(record.require_insurance).lower()}. '
        'Fictional supplied quotes; totals all-in USD. No invented charges, prices, '
        'capacity, upgrades, carriers, or split shipments.',
        'PYTHON CHECKS: P=PASS, F=FAIL; fields below are from protected source facts.',
    ]
    for quote, checks in zip(record.quotes, audit['rows']):
        assert quote.id == checks['quote_id']
        result = ','.join(
            f"{name}={'P' if checks[name] else 'F'}"
            for name in ('within_budget', 'on_time', 'enough_capacity', 'insured_as_required'))
        lines.append(
            f'{quote.id}|{quote.carrier}|{quote.method}|USD {quote.quoted_total_usd}'
            f'|arrival {quote.arrival_date}|capacity {quote.capacity_units}'
            f'|insured {str(quote.insurance_included).lower()}|{result}')
    recommendation = audit['recommendation']
    lines.append('PYTHON ELIGIBLE IDS: ' + (', '.join(audit['eligible_ids']) or 'NONE'))
    lines.append('PYTHON LOWEST-COST ELIGIBLE IDS: ' + (', '.join(audit['lowest_cost_eligible_ids']) or 'NONE'))
    lines.append(
        'PYTHON RECOMMENDATION: ' +
        (f"proceed with {recommendation['option_id']}, USD {recommendation['total_cost_usd']}, arrival {recommendation['arrival_date']}"
         if recommendation['option_id'] else 'hold; no eligible quote'))
    lines.append('SOURCE CAVEATS:')
    lines.extend(catalog[item] for item in selected_ids)
    return '\n'.join(lines)
