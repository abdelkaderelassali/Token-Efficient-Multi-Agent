"""Deterministic validation; no LLM calls or automatic answer correction."""
from datetime import date
import math

from report_format import report_diagnostics


def validate_decision_structure(decision):
    errors = []
    if not isinstance(decision, dict):
        return {"valid": False, "errors": ["Decision must be a JSON object"]}
    required = {"verdict", "option_id", "total_cost_usd", "arrival_date", "quantity",
                "budget_usd", "deadline", "reasoning", "conditions"}
    errors.extend(f"Missing field: {key}" for key in sorted(required - decision.keys()))
    errors.extend(f"Unexpected field: {key}" for key in sorted(decision.keys() - required))
    verdict = decision.get("verdict")
    if verdict not in ("proceed", "hold"):
        errors.append("verdict must be proceed or hold")
    for key in ("quantity", "budget_usd", "total_cost_usd"):
        value = decision.get(key)
        if key == "total_cost_usd" and verdict == "hold" and value is None:
            continue
        numeric = type(value) is int or (type(value) is float and math.isfinite(value))
        if not numeric or value < 0:
            errors.append(f"{key} must be a finite nonnegative number")
        elif key == "quantity" and (value <= 0 or value != int(value)):
            errors.append("quantity must be a positive whole number")
    for key in ("deadline", "arrival_date"):
        value = decision.get(key)
        if key == "arrival_date" and verdict == "hold" and value is None:
            continue
        try:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError()
        except (ValueError, TypeError):
            errors.append(f"{key} must be a valid YYYY-MM-DD date")
    if verdict == "hold":
        for key in ("option_id", "total_cost_usd", "arrival_date"):
            if decision.get(key) is not None:
                errors.append(f"{key} must be null when holding")
    elif not isinstance(decision.get("option_id"), str) or not decision["option_id"].strip():
        errors.append("option_id must identify a quote when proceeding")
    reasoning = decision.get("reasoning")
    if not isinstance(reasoning, str) or not reasoning.strip():
        errors.append("reasoning must be nonempty text")
    conditions = decision.get("conditions")
    if not isinstance(conditions, list) or not all(isinstance(c, str) and c.strip() for c in conditions):
        errors.append("conditions must be an array of nonempty strings")
    return {"valid": not errors, "errors": errors}


def validate_compact_facts(context, record):
    """Require the exact deterministic fact block, separate from model notes."""
    expected = "AUTHORITATIVE COMPACT FACTS:\n" + record.compact_text() + "\n\nCOMPRESSED ANALYSIS:\n"
    if not isinstance(context, str) or not context.startswith(expected):
        raise ValueError("Compressed context failed protected-facts validation")
    return {"valid": True, "record_sha256": record.to_dict()["record_sha256"],
            "scope": "Exact protected block only; model notes are not certified factual."}


def validate_report(role, output):
    if not isinstance(output, str) or not output.strip():
        raise ValueError(f"{role}: missing report")
    diagnostics = report_diagnostics(role, output)
    warnings = []
    if not diagnostics["sections_match"]:
        warnings.append("Expected report sections are missing or out of order")
    if not diagnostics["within_word_target"]:
        warnings.append("Report exceeds its word target")
    return {"valid": not warnings, "warnings": warnings, **diagnostics}
