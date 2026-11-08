"""Deterministic career, education, and skill-coverage summaries.

These measures describe the observed profiles. A missing title or skill is never
interpreted as evidence that a person lacks the underlying ability.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import math
import re
from typing import Any


# Rules are checked in order, so a title such as "Senior Engineering Manager"
# is classified by its management responsibility rather than its seniority word.
SENIORITY_RULES: tuple[tuple[str, str], ...] = (
    ("manager/director", r"\b(manager|director|head of|vp|vice president)\b"),
    ("lead", r"\b(lead|principal|staff|architect)\b"),
    ("senior", r"\b(senior|sr\.?|experienced)\b"),
    ("junior", r"\b(junior|jr\.?|associate|entry.level)\b"),
    ("intern", r"\b(intern|internship|trainee)\b"),
    ("mid", r"\b(engineer|scientist|analyst|developer|designer|specialist|researcher)\b"),
)


def normalise_seniority(title: str) -> str:
    """Use explicit title cues and leave unfamiliar titles unclassified."""
    for label, pattern in SENIORITY_RULES:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return label
    return "unknown"


def _month_index(value: str | None) -> int | None:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", value):
        return None
    year, month = (int(part) for part in value.split("-"))
    return year * 12 + month - 1


def _clean_skills(profile: dict[str, Any]) -> set[str]:
    return {str(skill).strip().casefold() for skill in profile.get("skills", []) if str(skill).strip()}


def _transition_rows(profiles: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter[str]]:
    """Count only dated, nonoverlapping moves between different companies."""
    movements: dict[tuple[str, str], dict[str, Any]] = {}
    outgoing_people: dict[str, set[str]] = defaultdict(set)
    outgoing_count: Counter[str] = Counter()
    person_count: Counter[str] = Counter()
    for profile in profiles:
        dated_jobs = []
        for job in profile.get("experience", []):
            start = _month_index(job.get("start"))
            company = str(job.get("company", "")).strip()
            if start is not None and company:
                dated_jobs.append((start, company, _month_index(job.get("end"))))
        dated_jobs.sort(key=lambda row: (row[0], row[1]))
        for previous, current in zip(dated_jobs, dated_jobs[1:]):
            _, source, previous_end = previous
            current_start, target, _ = current
            if source == target or previous_end is None or previous_end > current_start:
                continue
            key = (source, target)
            row = movements.setdefault(key, {"source": source, "target": target, "count": 0, "person_ids": set()})
            row["count"] += 1
            row["person_ids"].add(str(profile["id"]))
            outgoing_people[source].add(str(profile["id"]))
            outgoing_count[source] += 1
            person_count[str(profile["id"])] += 1
    transitions = [
        {**row, "person_ids": sorted(row["person_ids"]), "people_count": len(row["person_ids"])}
        for _, row in sorted(movements.items())
    ]
    sources = [
        {"company": company, "outbound_people": len(outgoing_people[company]), "outbound_transitions": count}
        for company, count in outgoing_count.items()
    ]
    sources.sort(key=lambda row: (-row["outbound_people"], -row["outbound_transitions"], row["company"]))
    return transitions, sources, person_count


def analyze_career(profiles: list[dict[str, Any]], reference_month: str) -> dict[str, Any]:
    """Use the configured reference month for every ongoing tenure."""
    ordered = sorted(profiles, key=lambda profile: str(profile["id"]))
    observed_month = _month_index(reference_month)
    if observed_month is None:
        raise ValueError("reference_month must be YYYY-MM")
    role_rows: list[dict[str, Any]] = []
    person_rows: list[dict[str, Any]] = []
    school_counts: Counter[str] = Counter()
    field_counts: Counter[str] = Counter()
    skill_holders: dict[str, set[str]] = defaultdict(set)
    skill_pairs: Counter[tuple[str, str]] = Counter()

    for profile in ordered:
        person_id = str(profile["id"])
        jobs = profile.get("experience", [])
        tenures: list[int] = []
        for job in jobs:
            start = _month_index(job.get("start"))
            end = _month_index(job.get("end"))
            ongoing = job.get("end") is None
            effective_end = observed_month if ongoing else end
            tenure = None
            if start is not None and effective_end is not None and effective_end >= start:
                tenure = effective_end - start + 1
                tenures.append(tenure)
            role_rows.append({
                "id": person_id,
                "company": str(job.get("company", "")),
                "title": str(job.get("title", "")),
                "seniority": normalise_seniority(str(job.get("title", ""))),
                "start": job.get("start"),
                "end": job.get("end"),
                "ongoing": ongoing,
                "tenure_months": tenure,
            })
        ranked_jobs = sorted(jobs, key=lambda job: (_month_index(job.get("start")) or -1, job.get("end") is None))
        latest_title = str(ranked_jobs[-1].get("title", "")) if ranked_jobs else ""
        education = profile.get("education", [])
        person_schools: set[str] = set()
        person_fields: set[str] = set()
        for item in education:
            school = str(item.get("school", "")).strip()
            field = str(item.get("field", "")).strip()
            if school:
                person_schools.add(school)
            if field:
                person_fields.add(field)
        school_counts.update(person_schools)
        field_counts.update(person_fields)
        skills = sorted(_clean_skills(profile))
        for skill in skills:
            skill_holders[skill].add(person_id)
        skill_pairs.update(combinations(skills, 2))
        person_rows.append({
            "id": person_id,
            "latest_seniority": normalise_seniority(latest_title),
            "observed_role_months": sum(tenures),
            "mean_role_tenure_months": round(sum(tenures) / len(tenures), 2) if tenures else None,
            "dated_role_count": len(tenures),
            "education_count": len(education),
        })

    transitions, talent_sources, person_transitions = _transition_rows(ordered)
    for row in person_rows:
        row["transition_count"] = person_transitions[row["id"]]
    skill_rows = [
        {"skill": skill, "people_count": len(holders), "person_ids": sorted(holders),
         "coverage_fraction": round(len(holders) / len(ordered), 6) if ordered else 0.0}
        for skill, holders in sorted(skill_holders.items())
    ]
    bus_factor = [
        {"skill": row["skill"], "person_id": row["person_ids"][0]}
        for row in skill_rows if row["people_count"] == 1
    ]
    # These are candidates for review, not claims about required capabilities.
    common_min_people = max(2, math.ceil(len(ordered) / 2))
    common_skills = {row["skill"] for row in skill_rows if row["people_count"] >= common_min_people}
    gap_candidates = [
        {"id": str(profile["id"]), "missing_common_skills": sorted(common_skills - _clean_skills(profile))}
        for profile in ordered
    ]
    return {
        "as_of_month": reference_month,
        "seniority_mapping": [{"category": label, "pattern": pattern} for label, pattern in SENIORITY_RULES],
        "roles": role_rows,
        "people": person_rows,
        "movement_graph": {"directed": True, "edges": transitions},
        "talent_sources": talent_sources,
        "education_schools": [{"school": name, "profile_count": count} for name, count in sorted(school_counts.items())],
        "education_fields": [{"field": name, "profile_count": count} for name, count in sorted(field_counts.items())],
        "skill_cooccurrence": [
            {"skill_a": pair[0], "skill_b": pair[1], "people_count": count}
            for pair, count in sorted(skill_pairs.items(), key=lambda item: (-item[1], item[0]))
        ],
        "skill_holders": skill_rows,
        "bus_factor_one_skills": bus_factor,
        "skill_gap_candidates": gap_candidates,
        "common_skill_min_people": common_min_people,
        "assumptions": [
            "Ongoing tenure ends at the configured reference month.",
            "A company move requires dated, nonoverlapping successive jobs; concurrent roles are excluded.",
            "Skill gaps are missing common recorded skills, not proven skill deficiencies.",
        ],
    }
