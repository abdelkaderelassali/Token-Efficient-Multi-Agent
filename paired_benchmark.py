"""Paired compressor experiment with shared upstream reports and auditable costs.

Python computes the authoritative quote recommendation supplied to agents and checked by the evaluator. Both branches see the same
source quotations and model-generated reports; only their representation differs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import logging
import math
import re
from pathlib import Path
import time
import uuid

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from benchmark import report_text
from protected_facts import ProtectedFacts
from quote_calculations import calculate_quotes
from context_routing import route_context
from compression_policy import compression_policy
from structured_claims import SCHEMA as CLAIM_SCHEMA, instructions as claim_instructions, parse_report, render_report
from validation import validate_decision_structure, validate_compact_facts, validate_report

ROOT = Path(__file__).resolve().parent
MODEL = "llama3"
CONTEXT_WINDOW = 8192
LOGGER = logging.getLogger(__name__)


def scenarios():
    return json.loads((ROOT / "demo_scenarios.json").read_text(encoding="utf-8"))


def source_text(scenario):
    header = (
        f"REQUEST: {scenario['brief']}\n"
        f"Route: {scenario['route']}\nQuantity: {scenario['quantity']} units\n"
        f"Budget: USD {scenario['budget_usd']}\nDeadline: {scenario['deadline']}\n"
        f"Transit insurance required: {scenario['require_insurance']}\n"
        "Objective: lowest quoted total among offers satisfying every requirement.\n"
        "No split shipments or unquoted changes. All quoted totals are all-in.\n"
    )
    quotes = []
    for offer in scenario["options"]:
        quotes.append(
            f"QUOTE {offer['id']} — {offer['carrier']} / {offer['method']}\n"
            f"All-in total: USD {offer['quoted_total_usd']}\n"
            f"Warehouse arrival: {offer['arrival_date']}\n"
            f"Confirmed capacity: {offer['capacity_units']} units\n"
            f"Transit insurance included: {offer['insurance_included']}\n"
            f"Supplier terms: {offer['terms']}"
        )
    from requirement_review import review_text
    return header + "\n" + "\n\n".join(quotes) + review_text(ProtectedFacts.from_scenario(scenario))


def protected_facts(scenario):
    """Render validated source fields without computing the answer."""
    return ProtectedFacts.from_scenario(scenario).compact_text()


def constraint_audit(scenario):
    """Authoritative Python eligibility and cheapest-quote calculations."""
    return calculate_quotes(ProtectedFacts.from_scenario(scenario))


class ModelRunner:
    def __init__(self, model=MODEL, seed=42):
        self.model, self.seed = model, seed

    def call(self, role, system, text, max_output=384, json_output=False):
        started = time.perf_counter()
        if not json_output:
            system += claim_instructions(role)
        model = ChatOllama(
            model=self.model, temperature=0, seed=self.seed,
            num_ctx=CONTEXT_WINDOW, num_predict=max_output,
            format="json" if json_output else CLAIM_SCHEMA,
        )
        response = model.invoke([SystemMessage(content=system), HumanMessage(content=text)])
        metadata = response.response_metadata
        # Missing usage is an experiment failure, never a zero-cost call.
        for key in ("prompt_eval_count", "eval_count"):
            if type(metadata.get(key)) is not int or metadata[key] < 0:
                raise ValueError(f"{role}: missing Ollama token measurement {key}")
        if metadata["prompt_eval_count"] + max_output >= CONTEXT_WINDOW:
            raise ValueError(f"{role}: context too close to the configured window")
        LOGGER.info("%s complete: %s input + %s output tokens", role,
                    metadata['prompt_eval_count'], metadata['eval_count'])
        call = {
            "role": role, "input_tokens": metadata["prompt_eval_count"],
            "output_tokens": metadata["eval_count"],
            "total_tokens": metadata["prompt_eval_count"] + metadata["eval_count"],
            "seconds": time.perf_counter() - started,
            "output": report_text(response, role),
            "system_prompt": system, "input": text,
            "settings": {"model": self.model, "seed": self.seed, "temperature": 0,
                         "num_ctx": CONTEXT_WINDOW, "max_output_tokens": max_output,
                         "json_output": True, "structured_claims": not json_output},
            "input_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "finish_reason": metadata.get("done_reason", "unknown"),
            "report_format": None,
        }
        if not json_output:
            call["raw_output"] = call["output"]
            try:
                claims = parse_report(call["raw_output"], role)
            except (ValueError, TypeError) as exc:
                call["claim_validation"] = {"valid": False, "error": str(exc)}
                destination = ROOT / "verification" / f"invalid-report-{uuid.uuid4()}.json"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps({"status": "invalid_structured_report", "call": call}, indent=2), encoding="utf-8")
                raise ValueError(f"{role}: invalid structured report ({exc}). Raw output and measured call cost saved to {destination.name}") from exc
            call["structured_report"] = claims
            call["claim_validation"] = {"valid": True, "schema_version": "claims-v1", "scope": "Syntax only; claims are not yet fact-checked"}
            call["output"] = render_report(claims)
        return call


def shared_reports(scenario, runner):
    started = time.perf_counter()
    record = ProtectedFacts.from_scenario(scenario)
    source = source_text(scenario)
    audit = calculate_quotes(record)
    ingestion = runner.call("Ingestion", (
        "Extract the shipment objective and mandatory constraints from the source. "
        "Python calculations are authoritative; do not recompute comparisons. "
        "Do not invent facts. Keep this briefing under 80 words."
    ), f"REQUEST:\n{record.brief}\n\n{record.compact_text()}\n\n{audit['text']}", max_output=256)
    agent_input = f"SOURCE QUOTATIONS:\n{source}\n\n{audit['text']}"
    with ThreadPoolExecutor(max_workers=2) as pool:
        logistics_future = pool.submit(runner.call, "Logistics", (
            "You are a logistics analyst. Compare the supplied offers for route, arrival, "
            "capacity and insurance. Identify feasible choices and any uncertainty. "
            "Explain the supplied Python eligibility results; do not recompute them. "
            "Use only supplied facts; preserve quote IDs. The source takes precedence "
            "over the briefing. Write a useful report under 140 words, without padding."
        ), agent_input, 384)
        finance_future = pool.submit(runner.call, "Finance", (
            "You are a finance analyst. Compare supplied all-in quote totals with the "
            "budget. Do not invent prices or add included charges again. Explain the "
            "lowest-cost feasible choice and trade-offs with other offers. "
            "Use the Python budget balances and recommendation without recalculating them. Preserve IDs "
            "and amounts. The source takes precedence over the briefing. Write a useful "
            "report under 140 words, without padding."
        ), agent_input, 384)
        logistics, finance = logistics_future.result(), finance_future.result()
    reports = {"Ingestion": ingestion["output"], "Logistics": logistics["output"], "Finance": finance["output"]}
    context = (
        f"AUTHORITATIVE SOURCE:\n{source}\n\n{audit['text']}\n\n"
        f"LOGISTICS REPORT:\n{reports['Logistics']}\n\n"
        f"FINANCE REPORT:\n{reports['Finance']}"
    )
    return {
        "context": context, "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        "reports": reports, "calls": [ingestion, logistics, finance],
        "protected_facts": record.to_dict(),
        "constraint_audit": audit,
        "seconds": time.perf_counter() - started,
        "token_usage": sum(c["total_tokens"] for c in (ingestion, logistics, finance)),
    }


def compress_context(scenario, shared, runner, target_ratio=0.28):
    # The word budget is a heuristic. Actual inference tokens are measured separately.
    record = ProtectedFacts.from_scenario(scenario)
    snapshot = record.to_dict()
    if shared.get("protected_facts", snapshot) != snapshot:
        raise ValueError("Protected source facts changed after upstream generation")
    facts = record.compact_text()
    source_words = len(shared["context"].split())
    summary_words = max(20, min(60, int(source_words * target_ratio) - len(facts.split())))
    call = runner.call("Compressor", (
        "Compress the reports into decision-relevant notes. Preserve uncertainty, "
        "disagreements and qualifications. Source quotations override analyst errors. "
        "Python calculations override conflicting generated claims; never summarize a contradiction as an established fact. "
        f"Write at most {summary_words} words with no introduction. "
        "A losslessly copied quote table will accompany your notes, so do not repeat "
        "the table's prices, dates, capacities or insurance fields. Summarize only "
        "unresolved issues and disagreements; do not recommend a winning quote. "
        "Use complete short sentences. Do not invent facts."
    ), shared["context"], max_output=160)
    summary = call["output"]
    summary_truncated = call.get("finish_reason") == "length"
    if summary_truncated and not call.get("structured_report"):
        # Do not forward a sentence cut off at the generation limit. All generated
        # tokens are still charged, and the full raw output is retained in calls.
        boundaries = list(re.finditer(r"[.!?](?:\s+|$)", summary))
        summary = summary[:boundaries[-1].start() + 1] if boundaries else ""
    context = f"AUTHORITATIVE COMPACT FACTS:\n{facts}\n\nCOMPRESSED ANALYSIS:\n{summary or 'No complete analysis notes returned; refer to the source facts above.'}"
    return context, call, {
        "strategy": "LLM summary plus losslessly packed source facts",
        "target_word_ratio": target_ratio,
        "source_words": source_words, "compressed_words": len(context.split()),
        "achieved_word_ratio": len(context.split()) / source_words,
        "summary_word_budget": summary_words,
        "summary_truncated": summary_truncated,
        "source_sha256": shared["context_sha256"],
        "protected_facts": facts,
        "protected_facts_sha256": snapshot["record_sha256"],
    }


def evaluate_decision(text, scenario):
    """Score adherence to the authoritative Python calculations and source fields."""
    audit = constraint_audit(scenario)
    feasible = audit["eligible_ids"]
    best = audit["lowest_cost_eligible_ids"]
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        decision = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_object)
        if not isinstance(decision, dict):
            raise ValueError("Expected a JSON object")
    except (ValueError, TypeError):
        decision = {}
    selected = next((o for o in scenario["options"] if o["id"] == decision.get("option_id")), None)
    structure = validate_decision_structure(decision)
    selected_checks = next((row for row in audit["rows"] if row["quote_id"] == decision.get("option_id")), {})
    number_equals = lambda value, expected: type(value) in (int, float) and value == expected
    checks = {
        "structured_answer": structure["valid"],
        "correct_verdict": decision.get("verdict") == ("proceed" if feasible else "hold"),
        "lowest_cost_feasible_quote": decision.get("option_id") in best if feasible else decision.get("option_id", "missing") is None,
        "quoted_total_correct": selected is not None and number_equals(decision.get("total_cost_usd"), selected["quoted_total_usd"]) if feasible else decision.get("total_cost_usd", "missing") is None,
        "arrival_correct": selected is not None and decision.get("arrival_date") == selected["arrival_date"] if feasible else decision.get("arrival_date", "missing") is None,
        "quantity_preserved": number_equals(decision.get("quantity"), scenario["quantity"]),
        "budget_preserved": number_equals(decision.get("budget_usd"), scenario["budget_usd"]),
        "deadline_preserved": decision.get("deadline") == scenario["deadline"],
        "insurance_requirement_met": selected_checks.get("insured_as_required", False) if feasible else decision.get("verdict") == "hold",
        "delivery_requirement_met": selected_checks.get("on_time", False) if feasible else decision.get("verdict") == "hold",
        "budget_requirement_met": selected_checks.get("within_budget", False) if feasible else decision.get("verdict") == "hold",
        "capacity_requirement_met": selected_checks.get("enough_capacity", False) if feasible else decision.get("verdict") == "hold",
    }
    if scenario.get('requirement_review', {}).get('required_capabilities'):
        checks['additional_requirements_met'] = (all(value is True for key, value in selected_checks.items()
            if key.startswith('meets_')) and selected is not None) if feasible else decision.get('verdict') == 'hold'
    if 'requirement_review' in scenario:
        from requirement_review import assess_requirements
        checks['requirements_review_complete'] = assess_requirements(scenario)['status'] == 'ready'
    return {
        "passed": sum(checks.values()), "total": len(checks),
        "all_passed": all(checks.values()), "checks": checks,
        "failed_checks": [key for key, passed in checks.items() if not passed],
        "evaluator_version": "quote-checklist-v3",
        "structure_validation": structure,
        "expected_option_ids": best, "decision": decision,
        "scope": "Checks supplied quote selection and key facts; not a complete evaluation of prose or real-world feasibility.",
    }


def make_decision(runner, context, audit, risk, compliance):
    return runner.call("Decision", (
        "Explain and use the PYTHON RECOMMENDATION. Python has already computed eligibility and the cheapest eligible quotes. "
        "All INELIGIBLE QUOTE IDS are disallowed, even if a report recommends them. "
        "Python calculations override advisory reviews; do not redo arithmetic or ranking. Never invent quotes, prices or upgrades. "
        "Return only JSON with keys: verdict ('proceed' if eligible quotes exist, otherwise "
        "'hold'), option_id, total_cost_usd (number), arrival_date (YYYY-MM-DD), quantity "
        "(number), budget_usd (number), deadline (YYYY-MM-DD), reasoning (brief string), "
        "conditions (array of at most three short strings). Keep reasoning under 50 words.  If no offer is eligible, option_id, total_cost_usd "
        "and arrival_date must be null. Always retain the requested quantity, budget and deadline."
    ), f"{context}\n\nRISK REVIEW (advisory):\n{risk['output']}\n\n"
       f"COMPLIANCE REVIEW (advisory):\n{compliance['output']}\n\n{audit['text']}",
        max_output=384, json_output=True)


def downstream(scenario, shared, runner, compressed, target_ratio, compression_mode="auto"):
    started = time.perf_counter()
    record = ProtectedFacts.from_scenario(scenario).to_dict()
    if record != shared["protected_facts"]:
        raise ValueError("Protected source facts changed after upstream generation")
    calls = []
    context, compression = shared["context"], None
    use_compact = False
    if compressed:
        compression = compression_policy(context, len(ProtectedFacts.from_scenario(scenario).compact_text()) + 400,
                                         consumers=2, mode=compression_mode)
        if compression["should_compress"]:
            candidate, call, details = compress_context(scenario, shared, runner, target_ratio)
            calls.append(call)
            compression.update(details, compressor_called=True)
            compression["validation"] = validate_compact_facts(candidate, ProtectedFacts.from_scenario(scenario))
            if compression_mode == "always" or len(candidate) < len(context):
                context, use_compact = candidate, True
                compression["applied"] = True
            else:
                compression["reason"] = "Generated context was not shorter; kept full context and counted compressor cost"
    audit = shared["constraint_audit"]
    source = f"AUTHORITATIVE SOURCE:\n{source_text(scenario)}"
    routed = {role: route_context(role, source, shared["reports"], context if use_compact else None)
              for role in ("Risk", "Compliance", "Decision")}
    common = (
        "Use only supplied facts. Authoritative source facts override generated reports. "
        "Explain the authoritative Python comparisons and recommendation; do not redo arithmetic or ranking. "
        "Do not invent costs, regulations, or exceptions. Keep your review under 90 words. "
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        risk_future = pool.submit(runner.call, "Risk", common + (
            "You assess shipping risks: timing, capacity, insurance and unsupported assumptions. "
            "Identify issues relevant to the proposed choice using quote IDs."
        ), routed["Risk"][0] + "\n\n" + audit["text"], 192)
        compliance_future = pool.submit(runner.call, "Compliance", common + (
            "Check each offer against the stated budget, delivery deadline, capacity and "
            "insurance requirements. State which quote IDs satisfy all requirements. "
            "The supplied Python checks are authoritative for arithmetic and date comparisons. "
            "This is a supplied-constraints check, not legal certification."
        ), routed["Compliance"][0] + "\n\n" + audit["text"], 256)
        risk, compliance = risk_future.result(), compliance_future.result()
    calls.extend([risk, compliance])
    decision = make_decision(runner, routed["Decision"][0], audit, risk, compliance)
    calls.append(decision)
    seconds = time.perf_counter() - started
    reports = dict(shared["reports"], Risk=risk["output"], Compliance=audit['text'] + "\n\nMODEL REVIEW:\n" + compliance["output"],
                   Decision=decision["output"], Compressor=context if use_compact else "")
    return {
        "mode": "compressed" if compressed else "baseline",
        "shared_context_sha256": shared["context_sha256"],
        "token_usage": shared["token_usage"] + sum(c["total_tokens"] for c in calls),
        "execution_time_seconds": shared["seconds"] + seconds,
        "branch_seconds": seconds, "calls": calls, "reports": reports,
        "compression": compression, "evaluation": evaluate_decision(decision["output"], scenario),
        "structured_reports": {c["role"]: c["structured_report"] for c in shared["calls"] + calls if "structured_report" in c},
        "report_validation": {c["role"]: c.get("claim_validation") or validate_report(c["role"], c["output"])
                              for c in shared["calls"] + calls if c["role"] != "Decision"},
        "context_routing": {role: components + (["risk_review", "compliance_review", "python_constraint_audit"] if role == "Decision" else ["python_constraint_audit"] if role == "Risk" else [])
                            for role, (_, components) in routed.items()},
        "constraint_audit": audit,
        "protected_facts_sha256": record["record_sha256"],
    }


def run_pair(scenario, model=MODEL, seed=42, target_ratio=0.28, compressed_first=False, runner=None, compression_mode="auto"):
    if compression_mode not in ("auto", "always", "never"):
        raise ValueError("compression mode must be auto, always, or never")
    if not 0.1 <= target_ratio <= 0.8:
        raise ValueError("Target ratio must be between 0.1 and 0.8")
    runner = runner or ModelRunner(model, seed)
    started = time.perf_counter()
    shared = shared_reports(scenario, runner)
    branches = {}
    for compressed in ((True, False) if compressed_first else (False, True)):
        branch = downstream(scenario, shared, runner, compressed, target_ratio, compression_mode)
        branches[branch["mode"]] = branch
    baseline, compact = branches["baseline"], branches["compressed"]
    saved = baseline["token_usage"] - compact["token_usage"]
    # Same reviewer and system prompt: this is measured input reduction, not an estimate.
    base_risk = next(c for c in baseline["calls"] if c["role"] == "Risk")
    comp_risk = next(c for c in compact["calls"] if c["role"] == "Risk")
    def totals(calls):
        return {key: sum(call[key] for call in calls) for key in ("input_tokens", "output_tokens", "total_tokens")}
    base_downstream = totals(baseline["calls"])
    comp_downstream = totals([c for c in compact["calls"] if c["role"] != "Compressor"])
    compressor = totals([c for c in compact["calls"] if c["role"] == "Compressor"])
    both_passed = baseline["evaluation"]["all_passed"] and compact["evaluation"]["all_passed"]
    truncated = [f"shared/{c['role']}" for c in shared["calls"] if c.get("finish_reason") == "length"]
    truncated += [f"{mode}/{c['role']}" for mode, branch in branches.items()
                  for c in branch["calls"] if c.get("finish_reason") == "length"]
    percent_saved = lambda before, after: 100 * (before - after) / before if before else None
    return {
        "schema_version": 1, "experiment_id": str(uuid.uuid4()),
        "created_at": datetime.now(timezone.utc).isoformat(), "scenario": scenario,
        "configuration": {"pipeline_version": "paired-v9", "claim_schema": "claims-v1", "calculation_version": "quote-calculations-v1", "compression_mode": compression_mode, "context_routing": "role-specific-v1", "report_format": "claims-v1", "model": model, "seed": seed, "temperature": 0,
                          "num_ctx": CONTEXT_WINDOW, "target_word_ratio": target_ratio,
                          "constraint_validation": "Python computes eligibility and cheapest quote; model explains the same results in both branches",
                          "branch_order": list(branches)},
        "shared": shared, **branches,
        "comparison": {
            "saved_tokens": saved, "reduction_percent": percent_saved(baseline["token_usage"], compact["token_usage"]),
            "review_input_reduction_percent": percent_saved(base_risk["input_tokens"], comp_risk["input_tokens"]),
            "downstream_input_reduction_percent": percent_saved(base_downstream["input_tokens"], comp_downstream["input_tokens"]),
            "downstream_input_tokens_saved": base_downstream["input_tokens"] - comp_downstream["input_tokens"],
            "downstream_output_tokens_saved": base_downstream["output_tokens"] - comp_downstream["output_tokens"],
            "compressor_tokens": compressor["total_tokens"],
            "token_breakdown": {"shared": totals(shared["calls"]), "baseline_downstream": base_downstream,
                                "compressed_downstream": comp_downstream, "compressor": compressor},
            "both_checklists_passed": both_passed,
            "quality_preserving_savings": both_passed and saved > 0,
            "truncated_calls": truncated,
            "timing_scope": "Branch times include the shared stage once; parallel call durations must not be summed. Model loading and queueing are included.",
            "total_experiment_tokens": baseline["token_usage"] + compact["token_usage"] - shared["token_usage"],
            "experiment_wall_seconds": time.perf_counter() - started,
            "accounting": "Each branch includes the same upstream cost. Compression input/output tokens are included. Upstream calls were executed once physically. No retries or uncounted judge-model calls.",
        },
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="demo_detailed", help="Scenario ID or all")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--compression-mode", choices=("auto", "always", "never"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-ratio", type=float, default=0.28)
    parser.add_argument("--output", type=Path, default=ROOT / "verification" / "paired-results.json")
    args = parser.parse_args()
    selected = [s for s in scenarios() if args.scenario in ("all", s["id"])]
    if args.scenario == "all":
        selected.reverse()  # Exercise the detailed case first.
    if not selected or args.repeats < 1:
        parser.error("Choose an existing scenario and at least one repetition")
    if args.output.exists():
        parser.error("Output already exists; choose a new path to preserve previous experiments")
    results = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for scenario in selected:
        for repeat in range(args.repeats):
            print(f"Running {scenario['id']} pair {repeat + 1}/{args.repeats}", flush=True)
            result = run_pair(scenario, args.model, args.seed + repeat, args.target_ratio, repeat % 2 == 1, compression_mode=args.compression_mode)
            results.append(result)
            args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps({"scenario": scenario['id'], "baseline_tokens": result['baseline']['token_usage'],
                              "compressed_tokens": result['compressed']['token_usage'],
                              "baseline_checks": result['baseline']['evaluation']['passed'],
                              "compressed_checks": result['compressed']['evaluation']['passed'],
                              **result['comparison']}), flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
