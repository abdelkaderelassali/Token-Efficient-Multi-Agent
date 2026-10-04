"""Explicit role routing, with conservative fallback for unstructured reports."""
import re


def finance_uncertainties(report):
    # Only omit other sections when the expected structure is intact.
    headings = re.findall(r"^##\s+(.+?)\s*$", report, re.MULTILINE)
    if headings != ["Costs", "Budget", "Unknowns"]:
        return report
    return report[re.search(r"^##\s+Unknowns\s*$", report, re.MULTILINE).start():]


def route_context(role, source, reports, compressed_context=None):
    if role not in ("Risk", "Compliance", "Decision"):
        raise ValueError(f"Unsupported downstream role: {role}")
    # Compliance checks supplied constraints, not other agents' recommendations.
    if role == "Compliance":
        return source, ["source", "python_constraint_audit"]
    if compressed_context is not None:
        return compressed_context, ["protected_facts", "compressed_findings"]
    finance = reports["Finance"]
    if role == "Risk":
        finance = finance_uncertainties(finance)
    context = (f"{source}\n\nLOGISTICS REPORT:\n{reports['Logistics']}"
               f"\n\nFINANCE {'UNCERTAINTIES' if role == 'Risk' else 'REPORT'}:\n{finance}")
    return context, ["source", "logistics_report", "finance_uncertainties" if role == "Risk" else "finance_report"]
