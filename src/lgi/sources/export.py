"""Read an owner's downloaded account export; never infer absent fields."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable

from lgi.real_coverage import apply_real_coverage
from lgi.schema import empty_profile, validate_profile


HEADERS = {
    "profile": {"first name", "firstname", "last name", "lastname", "headline"},
    "positions": {"company", "company name", "title", "started on"},
    "education": {"school", "school name", "degree", "degree name"},
    "skills": {"name", "skill"},
    "connections": {"first name", "firstname", "last name", "lastname", "connected on"},
    "interests": {"name", "interest"},
    "shares": {"date", "shared date", "content"},
    "comments": {"date", "comment date", "content"},
    "reactions": {"date", "reaction date", "content"},
}


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    for index, line in enumerate(lines):
        fields = next(csv.reader([line]), [])
        columns = [field.strip().casefold() for field in fields]
        recognized = set(columns) & set().union(*HEADERS.values())
        if len(recognized) < 2 and not ({"name", "skill", "interest"} & set(columns)):
            continue
        reader = csv.DictReader(lines[index:])
        rows = [
            {str(key).strip().casefold(): (value or "").strip() for key, value in row.items() if key}
            for row in reader
        ]
        return rows, columns
    return [], []


def _kind(path: Path, columns: list[str]) -> str | None:
    stem = path.stem.casefold()
    for kind in HEADERS:
        if kind in stem or stem.rstrip("s") == kind.rstrip("s"):
            return kind
    names = set(columns)
    if {"first name", "last name", "connected on"} <= names:
        return "connections"
    if {"first name", "last name"} <= names:
        return "profile"
    if {"company", "title"} <= names or {"company name", "title"} <= names:
        return "positions"
    if "school name" in names or "school" in names:
        return "education"
    if "skill" in names:
        return "skills"
    return None


def _get(row: dict[str, str], *names: str) -> str:
    return next((row[name] for name in names if row.get(name)), "")


def _month(raw: str) -> str:
    raw = raw.strip()
    if not raw or raw.casefold() == "present":
        return ""
    for fmt in ("%Y-%m-%d", "%Y-%m", "%b %Y", "%B %Y", "%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m")
        except ValueError:
            pass
    return ""


def _day(raw: str) -> str:
    raw = raw.strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%b %d, %Y", "%B %d, %Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return ""


class ExportAdapter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.discovered: dict[str, dict[str, object]] = {}

    def _tables(self) -> dict[str, list[dict[str, str]]]:
        tables: dict[str, list[dict[str, str]]] = {}
        for path in sorted(self.directory.glob("*.csv")):
            rows, columns = _read_csv(path)
            kind = _kind(path, columns)
            print(f"Export file {path.name}: columns {', '.join(columns) if columns else '(none found)'}; mapped {kind or 'none'}")
            self.discovered[path.name] = {"columns": columns, "kind": kind, "rows": len(rows)}
            if kind:
                tables.setdefault(kind, []).extend(rows)
        return tables

    def iter_profiles(self) -> Iterable[dict]:
        tables = self._tables()
        profile_rows = tables.get("profile", [])
        if not profile_rows:
            raise ValueError("No owner profile row found in export CSV files")
        owner = empty_profile("p01", "export")
        row = profile_rows[0]
        basic = owner["basic"]
        basic["name"] = " ".join(x for x in (_get(row, "first name", "firstname"), _get(row, "last name", "lastname")) if x)
        basic["headline"] = _get(row, "headline", "title")
        basic["location"] = _get(row, "location", "geo location")
        basic["summary"] = _get(row, "summary", "about")
        basic["profile_url"] = _get(row, "profile url", "public profile url", "url")
        for row in tables.get("positions", []):
            start = _month(_get(row, "started on", "start date", "start"))
            if not start:
                owner["warnings"].append("experience.start: position row skipped because the source start date was missing or unparseable")
                continue
            owner["experience"].append({
                "company": _get(row, "company name", "company"),
                "title": _get(row, "title", "position"),
                "start": start,
                "end": _month(_get(row, "finished on", "end date", "end")) or None,
                "location": _get(row, "location"),
                "description": _get(row, "description"),
            })
        for row in tables.get("education", []):
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
        owner["skills"] = sorted({
            _get(row, "name", "skill").strip()
            for row in tables.get("skills", []) if _get(row, "name", "skill").strip()
        }, key=str.casefold)
        owner["interests"] = sorted({
            _get(row, "name", "interest").strip()
            for row in tables.get("interests", []) if _get(row, "name", "interest").strip()
        }, key=str.casefold)
        for kind, activity_kind in (("shares", "share"), ("comments", "comment"), ("reactions", "reaction")):
            for index, row in enumerate(tables.get(kind, [])):
                published = _day(_get(row, "date", "shared date", "comment date", "reaction date", "created date"))
                if not published:
                    owner["warnings"].append(f"activity.date: {kind} row skipped because the source date was missing or unparseable")
                    continue
                raw_count = _get(row, "reaction count", "likes")
                owner["activity"].append({
                    "post_id": _get(row, "post id", "update url", "url") or f"{kind}_{index + 1}",
                    "date": published,
                    "type": activity_kind,
                    "text": _get(row, "content", "text", "comment"),
                    "reaction_count": int(raw_count) if raw_count.isdigit() else 0,
                    "engaged_ids": [],
                })
        yield validate_profile(apply_real_coverage(owner))
        for index, row in enumerate(tables.get("connections", []), 2):
            stub = empty_profile(f"p{index:02d}", "export")
            name = " ".join(x for x in (_get(row, "first name", "firstname"), _get(row, "last name", "lastname")) if x)
            stub["basic"]["name"] = name
            company = _get(row, "company", "company name")
            title = _get(row, "position", "title")
            stub["basic"]["headline"] = " at ".join(x for x in (title, company) if x)
            if company or title:
                stub["experience"].append({
                    "company": company, "title": title, "start": "", "end": None,
                    "location": "", "description": "",
                })
            stub["connected_date"] = _day(_get(row, "connected on", "connected date", "date"))
            yield validate_profile(apply_real_coverage(stub))
