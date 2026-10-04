"""Immutable source facts for the quotation benchmark; no model extraction."""
from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonempty text")
    return value


def number(value, field, integer=False, positive=False):
    if type(value) not in ((int,) if integer else (int, float)):
        raise ValueError(f"{field} must be numeric; booleans are not accepted")
    if (type(value) is float and not math.isfinite(value)) or value < 0 or (positive and value == 0):
        raise ValueError(f"{field} is outside the allowed range")
    return value


def boolean(value, field):
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean")
    return value


def iso_date(value, field):
    text(value, field)
    if date.fromisoformat(value).isoformat() != value:
        raise ValueError(f"{field} must use YYYY-MM-DD")
    return value


@dataclass(frozen=True)
class QuoteFacts:
    id: str
    carrier: str
    method: str
    quoted_total_usd: int | float
    arrival_date: str
    capacity_units: int
    insurance_included: bool
    terms: str


@dataclass(frozen=True)
class ProtectedFacts:
    schema_version: int
    scenario_id: str
    source_sha256: str
    brief: str
    route: str
    quantity: int
    budget_usd: int | float
    deadline: str
    require_insurance: bool
    quotes: tuple[QuoteFacts, ...]
    requirement_review_json: str | None = None

    @classmethod
    def from_scenario(cls, scenario):
        options = scenario.get("options")
        if not isinstance(options, list) or not options:
            raise ValueError("options must be a nonempty list")
        quotes = tuple(QuoteFacts(
            id=text(o["id"], "quote.id"), carrier=text(o["carrier"], "carrier"),
            method=text(o["method"], "method"),
            quoted_total_usd=number(o["quoted_total_usd"], "quoted_total_usd"),
            arrival_date=iso_date(o["arrival_date"], "arrival_date"),
            capacity_units=number(o["capacity_units"], "capacity_units", integer=True),
            insurance_included=boolean(o["insurance_included"], "insurance_included"),
            terms=text(o["terms"], "terms"),
        ) for o in options)
        if len({q.id for q in quotes}) != len(quotes):
            raise ValueError("Quote IDs must be unique")
        review_json = None
        if 'requirement_review' in scenario:
            from requirement_review import validate_review
            review_json = json.dumps(validate_review(scenario['requirement_review'], [q.id for q in quotes]),
                                     ensure_ascii=False, sort_keys=True, separators=(',', ':'))
        return cls(
            schema_version=1, scenario_id=text(scenario["id"], "scenario.id"),
            source_sha256=digest(scenario), brief=text(scenario["brief"], "brief"),
            route=text(scenario["route"], "route"),
            quantity=number(scenario["quantity"], "quantity", integer=True, positive=True),
            budget_usd=number(scenario["budget_usd"], "budget_usd"),
            deadline=iso_date(scenario["deadline"], "deadline"),
            require_insurance=boolean(scenario["require_insurance"], "require_insurance"),
            quotes=quotes,
            requirement_review_json=review_json,
        )

    def to_dict(self):
        payload = json.loads(json.dumps(asdict(self), allow_nan=False))
        review_json = payload.pop('requirement_review_json')
        if review_json is not None:
            payload['requirement_review'] = json.loads(review_json)
        return dict(payload, record_sha256=digest(payload))

    def compact_text(self):
        """Compact projection of structured constraints; prose stays in the record."""
        lines = [
            f"Route={self.route}; quantity={self.quantity} units; budget_usd={self.budget_usd}; "
            f"deadline={self.deadline}; insurance_required={str(self.require_insurance).lower()}.",
            "Select lowest-cost feasible quote. All totals are all-in USD. "
            "All units in one offer; no invented charges, upgrades, or split shipments.",
            "Quote ID | total USD | arrival YYYY-MM-DD | capacity units | insurance included",
        ]
        lines.extend(f"{q.id} | {q.quoted_total_usd} | {q.arrival_date} | "
                     f"{q.capacity_units} | {str(q.insurance_included).lower()}" for q in self.quotes)
        return "\n".join(lines)
