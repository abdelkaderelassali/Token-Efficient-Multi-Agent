"""Shared concise report contracts; diagnostics preserve the original output."""
import re


REPORT_SECTIONS = {
    "Ingestion": (80, ("Objective", "Constraints", "Unknowns")),
    "Logistics": (140, ("Options", "Constraints", "Unknowns")),
    "Finance": (140, ("Costs", "Budget", "Unknowns")),
    "Risk": (90, ("Risks", "Impact", "Actions")),
    "Compliance": (90, ("Checks", "Missing evidence", "Actions")),
    "Compressor": (60, ("Findings", "Unresolved")),
    "Decision": (160, ("Decision", "Basis", "Conditions", "Next actions")),
}


def report_instructions(role):
    limit, sections = REPORT_SECTIONS[role]
    return (
        f" Return a concise Markdown report, at most {limit} words, using exactly these "
        "headings in order: " + ", ".join(f"## {heading}" for heading in sections) + ". "
        "Under each heading use short factual bullets, at most three per section. "
        "Write 'Not supplied' for missing evidence and 'None identified' only when supported. "
        "Retain relevant quote IDs, exact amounts, dates, units, caveats and disagreements. "
        "Do not repeat the request or add introductions, conclusions, generic advice or invented facts. "
        "If a tighter word limit is given, follow it."
    )


def report_diagnostics(role, output):
    limit, sections = REPORT_SECTIONS[role]
    headings = re.findall(r"^##\s+(.+?)\s*$", output, re.MULTILINE)
    words = len(output.split())
    return {"format_version": "concise-v1", "word_count": words, "target_words": limit,
            "within_word_target": words <= limit, "sections_match": headings == list(sections),
            "missing_sections": [section for section in sections if section not in headings]}
