"""A single validated wire format for every source adapter."""

from __future__ import annotations

from datetime import date
import re
from typing import Any

SECTIONS = ("basic", "experience", "education", "skills", "interests", "activity")
SOURCES = {"fixture", "export", "manual", "live"}
BASIC_KEYS = {"name", "headline", "location", "summary", "profile_url"}
EXPERIENCE_KEYS = {"company", "title", "start", "end", "location", "description"}
EDUCATION_KEYS = {"school", "degree", "field", "start_year", "end_year"}
ACTIVITY_KEYS = {"post_id", "date", "type", "text", "reaction_count", "engaged_ids"}
MONTH = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def empty_profile(profile_id: str, source: str) -> dict[str, Any]:
    """Missing data stay empty so coverage cannot imply a guessed value."""
    return {
        "id": profile_id,
        "source": source,
        "basic": {key: "" for key in BASIC_KEYS},
        "experience": [],
        "education": [],
        "skills": [],
        "interests": [],
        "activity": [],
        "coverage": {key: False for key in SECTIONS},
        "warnings": [],
    }


def _keys(value: Any, expected: set[str], where: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{where} must contain exactly {sorted(expected)}")


def _string(value: Any, where: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{where} must be a string")


def _month(value: Any, where: str) -> None:
    if not isinstance(value, str) or not MONTH.fullmatch(value):
        raise ValueError(f"{where} must be YYYY-MM")


def validate_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Reject malformed values before they can corrupt graph calculations."""
    _keys(profile, {"id", "source", *SECTIONS, "coverage", "warnings"}, "profile")
    _string(profile["id"], "id")
    if not profile["id"]:
        raise ValueError("id cannot be empty")
    if profile["source"] not in SOURCES:
        raise ValueError("invalid source")
    _keys(profile["basic"], BASIC_KEYS, "basic")
    for key, value in profile["basic"].items():
        _string(value, f"basic.{key}")
    for section in ("experience", "education", "skills", "interests", "activity", "warnings"):
        if not isinstance(profile[section], list):
            raise ValueError(f"{section} must be a list")
    for row in profile["experience"]:
        _keys(row, EXPERIENCE_KEYS, "experience row")
        for key in EXPERIENCE_KEYS - {"end"}:
            _string(row[key], f"experience.{key}")
        _month(row["start"], "experience.start")
        if row["end"] is not None:
            _month(row["end"], "experience.end")
            if row["end"] < row["start"]:
                raise ValueError("experience end precedes start")
    for row in profile["education"]:
        _keys(row, EDUCATION_KEYS, "education row")
        for key in ("school", "degree", "field"):
            _string(row[key], f"education.{key}")
        for key in ("start_year", "end_year"):
            if type(row[key]) is not int or row[key] < 0:
                raise ValueError(f"education.{key} must be a nonnegative year")
        if row["end_year"] and row["start_year"] > row["end_year"]:
            raise ValueError("education end precedes start")
    for section in ("skills", "interests", "warnings"):
        for value in profile[section]:
            _string(value, section)
    if len({value.casefold() for value in profile["skills"]}) != len(profile["skills"]):
        raise ValueError("duplicate skills")
    for row in profile["activity"]:
        _keys(row, ACTIVITY_KEYS, "activity row")
        for key in ("post_id", "date", "type", "text"):
            _string(row[key], f"activity.{key}")
        try:
            date.fromisoformat(row["date"])
        except ValueError as error:
            raise ValueError("activity.date must be YYYY-MM-DD") from error
        if row["type"] not in {"post", "share", "comment"}:
            raise ValueError("invalid activity type")
        if type(row["reaction_count"]) is not int or row["reaction_count"] < 0:
            raise ValueError("reaction_count must be nonnegative")
        if not isinstance(row["engaged_ids"], list) or not all(isinstance(x, str) for x in row["engaged_ids"]):
            raise ValueError("engaged_ids must be strings")
    _keys(profile["coverage"], set(SECTIONS), "coverage")
    if not all(type(value) is bool for value in profile["coverage"].values()):
        raise ValueError("coverage values must be booleans")
    return profile
