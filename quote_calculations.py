"""Authoritative calculations for supplied quotes, independent of generated text."""
from datetime import date
from decimal import Decimal

from protected_facts import ProtectedFacts


def calculate_quotes(record: ProtectedFacts):
    budget = Decimal(str(record.budget_usd))
    deadline = date.fromisoformat(record.deadline)
    rows, comparisons = [], []
    for quote in record.quotes:
        price = Decimal(str(quote.quoted_total_usd))
        arrival = date.fromisoformat(quote.arrival_date)
        checks = {
            "within_budget": price <= budget,
            "on_time": arrival <= deadline,
            "enough_capacity": quote.capacity_units >= record.quantity,
            "insured_as_required": not record.require_insurance or quote.insurance_included,
        }
        from requirement_review import capability_checks
        checks.update(capability_checks(record, quote.id))
        rows.append(dict(quote_id=quote.id, **checks))
        comparisons.append({
            "quote_id": quote.id, "quoted_total_usd": quote.quoted_total_usd,
            "budget_usd": record.budget_usd,
            "budget_remaining_usd": str(budget - price),
            "arrival_date": quote.arrival_date, "deadline": record.deadline,
            "days_before_deadline": (deadline - arrival).days,
            "capacity_units": quote.capacity_units, "quantity": record.quantity,
            "capacity_surplus_units": quote.capacity_units - record.quantity,
            "insurance_included": quote.insurance_included,
            "insurance_required": record.require_insurance,
            "eligible": all(checks.values()),
            "failed_constraints": [key for key, passed in checks.items() if not passed],
        })
    eligible = [c for c in comparisons if c["eligible"]]
    minimum = min((Decimal(str(c["quoted_total_usd"])) for c in eligible), default=None)
    cheapest = sorted(c["quote_id"] for c in eligible if Decimal(str(c["quoted_total_usd"])) == minimum)
    selected = next((q for q in record.quotes if cheapest and q.id == cheapest[0]), None)
    recommendation = {
        "verdict": "proceed" if selected else "hold",
        "option_id": selected.id if selected else None,
        "total_cost_usd": selected.quoted_total_usd if selected else None,
        "arrival_date": selected.arrival_date if selected else None,
        "quantity": record.quantity, "budget_usd": record.budget_usd, "deadline": record.deadline,
    }
    eligible_ids = [c["quote_id"] for c in eligible]
    excluded = [c["quote_id"] for c in comparisons if not c["eligible"]]
    lines = ["PYTHON CONSTRAINT CHECKS (authoritative calculations from protected source fields):"]
    for row, detail in zip(rows, comparisons):
        checks_text = "; ".join(f"{key}={'UNKNOWN' if value is None else 'PASS' if value else 'FAIL'}" for key, value in row.items() if key != "quote_id")
        lines.append(f"{row['quote_id']}: {checks_text}; budget_remaining_usd={detail['budget_remaining_usd']}; "
                     f"days_before_deadline={detail['days_before_deadline']}; capacity_surplus_units={detail['capacity_surplus_units']}")
    lines.extend([
        "ELIGIBLE QUOTE IDS: " + (", ".join(eligible_ids) or "NONE"),
        "INELIGIBLE QUOTE IDS: " + (", ".join(excluded) or "NONE"),
        "LOWEST-COST ELIGIBLE QUOTE IDS: " + (", ".join(cheapest) or "NONE"),
        "PYTHON RECOMMENDATION: " + (f"proceed with {selected.id}, USD {selected.quoted_total_usd}, arrival {selected.arrival_date}" if selected else "hold; no eligible quote"),
        "Any FAIL makes a quote infeasible. Negative remaining budget, days or capacity indicate a shortfall. "
        "Explain these computed results; do not replace them with model arithmetic. "
        "Equal-price eligible quotes are tied; the recommendation uses the lexicographically smallest quote ID.",
    ])
    return {"calculation_version": "quote-calculations-v2" if record.requirement_review_json is not None else "quote-calculations-v1", "rows": rows,
            "comparisons": comparisons, "eligible_ids": eligible_ids, "ineligible_ids": excluded,
            "lowest_cost_eligible_ids": cheapest,
            "lowest_eligible_total_usd": selected.quoted_total_usd if selected else None,
            "recommendation": recommendation,
            "tie_break": "lexicographically smallest quote ID; all minimum-price eligible quotes accepted by evaluation",
            "protected_facts_sha256": record.to_dict()["record_sha256"], "text": "\n".join(lines)}
