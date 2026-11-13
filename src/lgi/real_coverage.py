"""Presence-based coverage and field-specific warnings for opt-in real sources.

Coverage records whether a section has identifying evidence, not whether optional
prose was supplied. This policy is intentionally separate from fixture coverage.
"""

from __future__ import annotations

from datetime import date
import re

from lgi.schema import SECTIONS

MONTH = re.compile(r"\d{4}-(0[1-9]|1[0-2])\Z")
SAFE_DATA_WARNINGS = {
    "experience.start: position row skipped because the source start date was missing or unparseable",
    "activity.date: shares row skipped because the source date was missing or unparseable",
    "activity.date: comments row skipped because the source date was missing or unparseable",
    "activity.date: reactions row skipped because the source date was missing or unparseable",
    "activity.date: publication skipped because OpenAlex publication_date was missing or malformed",
}


def _supplied(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _dated_employer(job: dict) -> bool:
    start = job.get("start")
    end = job.get("end")
    return _supplied(job.get("company")) and isinstance(start, str) and bool(MONTH.fullmatch(start)) and (
        end is None or (isinstance(end, str) and bool(MONTH.fullmatch(end)))
    )


def _dated_activity(item: dict) -> bool:
    if not _supplied(item.get("post_id")):
        return False
    try:
        return date.fromisoformat(str(item.get("date", ""))).isoformat() == item.get("date")
    except ValueError:
        return False


def apply_real_coverage(profile: dict) -> dict:
    """Recompute real-source coverage after mapping or anonymisation."""
    basic = profile["basic"]
    coverage = {
        "basic": _supplied(profile["id"]) and _supplied(basic["name"]),
        "experience": any(_dated_employer(job) for job in profile["experience"]),
        "education": any(_supplied(row.get("school")) for row in profile["education"]),
        "skills": any(_supplied(skill) for skill in profile["skills"]),
        "interests": any(_supplied(interest) for interest in profile["interests"]),
        "activity": any(_dated_activity(item) for item in profile["activity"]),
    }
    profile["coverage"] = {section: bool(coverage[section]) for section in SECTIONS}
    source = profile["source"]
    origin = "OpenAlex" if source == "openalex" else "the export"
    warnings: list[str] = [
        warning for warning in profile["warnings"] if warning in SAFE_DATA_WARNINGS
    ]
    if not coverage["basic"]:
        warnings.append(f"basic.name: name not provided by {origin}")
    if not coverage["experience"]:
        if any(_supplied(job.get("company")) for job in profile["experience"]):
            reason = "employment dates not provided for connection" if source == "export" else f"dated affiliation not provided by {origin}"
            warnings.append(f"experience.start: {reason}")
        else:
            warnings.append(f"experience.company: employer or institution not provided by {origin}")
    if source == "openalex" and profile["experience"]:
        warnings.extend([
            "experience.title: job title not provided by OpenAlex",
            "experience.start/end: affiliation years are publication observations, not employment dates",
        ])
    if not coverage["education"]:
        warnings.append(f"education.school: education not provided by {origin}")
    if not coverage["skills"]:
        warnings.append(f"skills: no skills or topics provided by {origin}")
    elif source == "openalex":
        warnings.append("skills: values derived from OpenAlex topics, not declared skills")
    if not coverage["interests"]:
        warnings.append(f"interests: interests not provided by {origin}")
    if not coverage["activity"]:
        if not profile["activity"]:
            warnings.append(f"activity: no activity provided by {origin}")
        elif any(_supplied(item.get("post_id")) for item in profile["activity"]):
            warnings.append(f"activity.date: valid activity date not provided by {origin}")
        else:
            warnings.append(f"activity.post_id: activity identifier not provided by {origin}")
    elif source == "openalex":
        warnings.append("activity.type: publications are not social posts; citations are not reactions")
    profile["warnings"] = list(dict.fromkeys(warnings))
    return profile
