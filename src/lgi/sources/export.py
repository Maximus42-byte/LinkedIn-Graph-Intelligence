"""Read a user's official account export without inferring absent sections."""

import csv
from pathlib import Path
from typing import Iterable

from lgi.schema import empty_profile, validate_profile


def _rows(path: Path, header_hints: tuple[str, ...]) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    header_index = next((i for i, line in enumerate(lines) if any(h in line.casefold() for h in header_hints) and ("," in line or line.strip().casefold() in header_hints)), None)
    if header_index is None:
        return []
    reader = csv.DictReader(lines[header_index:])
    return [{str(k).strip().casefold(): (v or "").strip() for k, v in row.items() if k is not None} for row in reader]


def _get(row: dict[str, str], *names: str) -> str:
    for name in names:
        if row.get(name):
            return row[name]
    return ""


def _month(raw: str) -> str:
    """Official exports may use YYYY-MM-DD, YYYY-MM, or month names."""
    from datetime import datetime
    raw = raw.strip()
    if not raw or raw.casefold() == "present":
        return ""
    for fmt in ("%Y-%m-%d", "%Y-%m", "%b %Y", "%B %Y", "%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m")
        except ValueError:
            pass
    return ""


class ExportAdapter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def iter_profiles(self) -> Iterable[dict]:
        profile_rows = _rows(self.directory / "Profile.csv", ("first name", "firstname", "headline"))
        if not profile_rows:
            return
        owner = empty_profile("p01", "export")
        row = profile_rows[0]
        basic = owner["basic"]
        basic["name"] = " ".join(x for x in (_get(row, "first name", "firstname"), _get(row, "last name", "lastname")) if x)
        basic["headline"] = _get(row, "headline", "title")
        basic["location"] = _get(row, "location", "geo location")
        basic["summary"] = _get(row, "summary", "about")
        basic["profile_url"] = _get(row, "profile url", "public profile url", "url")
        owner["coverage"]["basic"] = all(basic.values())
        position_skipped = False
        for row in _rows(self.directory / "Positions.csv", ("company", "title")):
            start = _month(_get(row, "started on", "start date", "start"))
            if not start:
                owner["warnings"].append("Skipped position without a parseable start date")
                position_skipped = True
                continue
            owner["experience"].append({
                "company": _get(row, "company name", "company"),
                "title": _get(row, "title", "position"),
                "start": start,
                "end": _month(_get(row, "finished on", "end date", "end")) or None,
                "location": _get(row, "location"),
                "description": _get(row, "description"),
            })
        owner["coverage"]["experience"] = bool(owner["experience"]) and not position_skipped and all(all(job[key] for key in ("company", "title", "start", "location", "description")) for job in owner["experience"])
        for row in _rows(self.directory / "Education.csv", ("school", "degree")):
            try:
                start = int(_get(row, "start date", "start year", "started on")[:4] or 0)
                end = int(_get(row, "end date", "end year", "finished on")[:4] or 0)
            except ValueError:
                start = end = 0
            owner["education"].append({
                "school": _get(row, "school name", "school"),
                "degree": _get(row, "degree name", "degree"),
                "field": _get(row, "field of study", "field"),
                "start_year": start,
                "end_year": end,
            })
        owner["coverage"]["education"] = bool(owner["education"]) and all(all(item[key] for key in ("school", "degree", "field", "start_year", "end_year")) for item in owner["education"])
        unique_skills = {}
        for row in _rows(self.directory / "Skills.csv", ("name", "skill")):
            skill = _get(row, "name", "skill").strip()
            if skill:
                unique_skills.setdefault(skill.casefold(), skill)
        owner["skills"] = sorted(unique_skills.values(), key=str.casefold)
        owner["coverage"]["skills"] = bool(owner["skills"])
        for section in ("basic", "experience", "education", "skills", "interests", "activity"):
            if not owner["coverage"][section]:
                owner["warnings"].append(f"{section} incomplete or unavailable in export")
        yield validate_profile(owner)
        connections = _rows(self.directory / "Connections.csv", ("first name", "firstname", "company"))
        for index, row in enumerate(connections, 2):
            stub = empty_profile(f"p{index:02d}", "export")
            name = " ".join(x for x in (_get(row, "first name", "firstname"), _get(row, "last name", "lastname")) if x)
            stub["basic"]["name"] = name
            company = _get(row, "company", "company name")
            title = _get(row, "position", "title")
            stub["basic"]["headline"] = " at ".join(x for x in (title, company) if x)
            stub["warnings"].append("Connection stub: only name, company, and position supplied")
            yield validate_profile(stub)
