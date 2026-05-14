"""
clean_outputs.py — Post-processing cleaner for Term2Note generated outputs.

Strips base-model formatting artifacts (markdown bold/headers/tables/code)
and replaces fabricated PII with deidentification placeholders.

Preserves legitimate MIMIC-style formatting:
  - Numbered lists (1. HCV Cirrhosis)
  - Dash bullets (- aspirin 81mg daily)
  - Star bullets (* follow up with cardiology)

Usage (standalone):
    python clean_outputs.py \
        --input outputs/generated/term2note_v3_eps4/synthetic_term2note.jsonl \
        --output outputs/generated/term2note_v3_eps4/synthetic_term2note_clean.jsonl

Usage (as module):
    from clean_outputs import clean_note, CleanStats
    stats = CleanStats()
    cleaned = clean_note(text, stats)
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, field


# ── Markdown patterns to STRIP ─────────────────────────────────────
# Each entry: (compiled regex, replacement, name for logging)
MD_PATTERNS = [
    # Bold: **text** → text (handles multiline bold spans)
    (re.compile(r"\*\*(.+?)\*\*", re.DOTALL), r"\1", "bold"),
    # H2-H6 headers: ## Header → Header:
    (re.compile(r"^#{2,6}\s+(.+)$", re.MULTILINE), r"\1:", "header"),
    # Table separators: |---|---| → remove entire line
    (re.compile(r"^\|[\s\-:]+\|.*$", re.MULTILINE), "", "table_sep"),
    # Table rows: | cell | cell | → cell, cell (require at least 2 interior pipes)
    (re.compile(r"^\|([^|\n]+\|[^|\n]+(?:\|[^|\n]*)*)\|$", re.MULTILINE), lambda m: re.sub(r"\s*\|\s*", ", ", m.group(1)).strip(), "table_row"),
    # Code blocks: ```...``` → contents only
    (re.compile(r"```\w*\n?(.*?)```", re.DOTALL), r"\1", "code_block"),
    # Inline code: `text` → text
    (re.compile(r"`([^`]+)`"), r"\1", "inline_code"),
    # Italic with single stars (not bullets): *word* → word
    # Only match mid-line italic, not line-start bullets
    (re.compile(r"(?<=\S)\*([A-Za-z][^*\n]{0,60})\*(?=[\s,.:;)\]])"), r"\1", "italic"),
    # Markdown links: [text](url) → text
    (re.compile(r"\[([^\]]+)\]\([^)]+\)"), r"\1", "link"),
    # Orphaned triple backticks (unpaired)
    (re.compile(r"```\w*"), "", "orphaned_backtick"),
    # Consecutive blank lines → single blank
    (re.compile(r"\n{3,}"), "\n\n", "excess_blanks"),
]

# ── PII patterns to REPLACE with ___ ───────────────────────────────
PII_PATTERNS = [
    # Known fabricated names — use (?<!\w) / (?!\w) instead of \b
    # so underscores (which are \w) don't block the match
    (re.compile(r"(?<![a-zA-Z])John Doe(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])Jane Doe(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])John Smith(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])Jane Smith(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])Sarah Johnson(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])Robert Johnson(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])Alex Chen(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])Michael Brown(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])Emily Davis(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])David Wilson(?![a-zA-Z])", re.I), "pii_name_known"),
    (re.compile(r"(?<![a-zA-Z])Mary Johnson(?![a-zA-Z])", re.I), "pii_name_known"),
    # Dr./Mr./Mrs./Ms. + full name or single name
    (re.compile(r"\b(?:Dr|Mr|Mrs|Ms)\.\s+[A-Z][a-z]{1,15}(?:\s+[A-Z][a-z]{1,15})?\b"), "pii_titled_name"),
    # Phone: (555)123-4567 or (123) 456-7890
    (re.compile(r"\(\d{3}\)\s*\d{3}[-.]?\d{4}"), "pii_phone"),
    # Phone: 555-1234 style
    (re.compile(r"\b555[-.]?\d{4}\b"), "pii_phone"),
    # Phone: XXX-XXX-XXXX (require separators to avoid matching lab values)
    (re.compile(r"\b\d{3}[-.]\d{3}[-.]\d{4}\b"), "pii_phone"),
    # SSN: 123-45-6789
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "pii_ssn"),
    # Email addresses
    (re.compile(r"\b[a-zA-Z][a-zA-Z0-9._]*@[a-zA-Z0-9.-]+\.\w{2,}\b"), "pii_email"),
    # Fabricated street addresses: 123 Main St, 789 Oak Ave
    (re.compile(r"\b\d{1,5}\s+[A-Z][a-z]+\s+(?:St|Ave|Rd|Blvd|Dr|Lane|Way|Court|Circle|Place)\.?\b"), "pii_address"),
    # Specific dates: March 12, 1999
    (re.compile(
        r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s*(?:19|20)\d{2}\b"
    ), "pii_date"),
    # City, State ZIP
    (re.compile(r"\b[A-Z][a-z]+,\s*[A-Z]{2}\s+\d{5}\b"), "pii_city_state_zip"),
]

# Clinical terms that look like names but aren't — whitelist to avoid false positives
CLINICAL_NOT_PII = {
    "blood pressure", "body mass", "heart rate", "blood count",
    "white blood", "red blood", "blood urea", "total cholesterol",
    "low density", "high density", "complete blood", "blood glucose",
    "medication list", "discharge summary", "discharge instructions",
    "hospital stay", "patient information", "clinical course",
    "medical history", "surgical history", "family history",
    "social history", "lab results", "vital signs",
}


@dataclass
class CleanStats:
    """Tracks per-note and aggregate cleaning statistics."""
    notes_processed: int = 0
    md_removals: dict = field(default_factory=lambda: {
        "bold": 0, "header": 0, "table_sep": 0, "table_row": 0,
        "code_block": 0, "inline_code": 0, "italic": 0, "link": 0,
        "orphaned_backtick": 0, "excess_blanks": 0,
    })
    pii_removals: dict = field(default_factory=lambda: {
        "pii_name_known": 0, "pii_titled_name": 0, "pii_phone": 0,
        "pii_ssn": 0, "pii_email": 0, "pii_address": 0,
        "pii_date": 0, "pii_city_state_zip": 0,
    })

    def note_report(self, note_md, note_pii):
        """Return a dict summarizing what was stripped from one note."""
        return {
            "markdown": {k: v for k, v in note_md.items() if v > 0},
            "pii": {k: v for k, v in note_pii.items() if v > 0},
            "total_md": sum(note_md.values()),
            "total_pii": sum(note_pii.values()),
        }

    def summary(self):
        total_md = sum(self.md_removals.values())
        total_pii = sum(self.pii_removals.values())
        return {
            "notes_processed": self.notes_processed,
            "total_md_removals": total_md,
            "total_pii_removals": total_pii,
            "md_by_type": {k: v for k, v in self.md_removals.items() if v > 0},
            "pii_by_type": {k: v for k, v in self.pii_removals.items() if v > 0},
        }


def clean_note(text, stats=None):
    """Clean a single note's text. Returns (cleaned_text, note_report)."""
    note_md = {name: 0 for _, _, name in MD_PATTERNS}
    note_pii = {name: 0 for _, name in PII_PATTERNS}

    # Strip PII first (before markdown removal might alter context)
    for pat, name in PII_PATTERNS:
        matches = pat.findall(text)
        if matches:
            count = len(matches)
            note_pii[name] += count
            if stats:
                stats.pii_removals[name] = stats.pii_removals.get(name, 0) + count
            text = pat.sub("___", text)

    # Strip markdown
    for pat, repl, name in MD_PATTERNS:
        before = text
        if callable(repl):
            text = pat.sub(repl, text)
        else:
            text = pat.sub(repl, text)
        if text != before:
            count = len(pat.findall(before))
            count = max(count, 1)
            note_md[name] += count
            if stats:
                stats.md_removals[name] = stats.md_removals.get(name, 0) + count

    # Clean up trailing/leading whitespace on lines
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)

    if stats:
        stats.notes_processed += 1

    report = {
        "markdown": {k: v for k, v in note_md.items() if v > 0},
        "pii": {k: v for k, v in note_pii.items() if v > 0},
        "total_md": sum(note_md.values()),
        "total_pii": sum(note_pii.values()),
    }
    return text, report


def clean_jsonl(input_path, output_path, report_path=None):
    """Clean an entire JSONL file. Returns aggregate stats."""
    stats = CleanStats()
    per_note_reports = []

    with open(input_path) as f_in, open(output_path, "w") as f_out:
        for i, line in enumerate(f_in):
            rec = json.loads(line.strip())
            plain = rec.get("plain_text", "")
            cleaned, report = clean_note(plain, stats)
            rec["plain_text"] = cleaned
            rec["plain_text_raw"] = plain

            # Also clean individual sections if present
            sections = rec.get("sections", [])
            for sec in sections:
                sec_text = sec.get("text", "")
                sec_cleaned, _ = clean_note(sec_text)
                sec["text"] = sec_cleaned

            # Rebuild text field from cleaned sections
            if sections:
                parts = []
                cc = rec.get("control_codes", {})
                prefix_parts = []
                for k, v in cc.items():
                    prefix_parts.append(k + ": " + str(v))
                if prefix_parts:
                    parts.append(" | ".join(prefix_parts))
                for sec in sections:
                    parts.append(sec.get("group", ""))
                    parts.append(sec.get("terms", ""))
                    parts.append(sec.get("text", ""))
                rec["text"] = "\n".join(parts)

            f_out.write(json.dumps(rec) + "\n")
            per_note_reports.append({"note_idx": i, **report})

    if report_path:
        with open(report_path, "w") as f:
            json.dump({
                "aggregate": stats.summary(),
                "per_note": per_note_reports,
            }, f, indent=2)

    return stats


def main():
    ap = argparse.ArgumentParser(description="Clean Term2Note generated outputs")
    ap.add_argument("--input", required=True, help="Input synthetic_term2note.jsonl")
    ap.add_argument("--output", required=True, help="Output cleaned JSONL")
    ap.add_argument("--report", default=None, help="Output cleaning report JSON")
    args = ap.parse_args()

    if not args.report:
        args.report = args.output.replace(".jsonl", "_clean_report.json")

    print("Cleaning " + args.input)
    stats = clean_jsonl(args.input, args.output, args.report)
    s = stats.summary()
    print("Processed " + str(s["notes_processed"]) + " notes")
    print("  MD removals: " + str(s["total_md_removals"]) + " " + str(s["md_by_type"]))
    print("  PII removals: " + str(s["total_pii_removals"]) + " " + str(s["pii_by_type"]))
    print("Report: " + args.report)
    print("Output: " + args.output)


if __name__ == "__main__":
    main()
