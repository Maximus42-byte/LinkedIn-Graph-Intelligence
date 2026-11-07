"""Parse visible semantic HTML and public JSON-LD into the shared schema."""

from __future__ import annotations

from bs4 import BeautifulSoup, Tag
from datetime import datetime
import json
import re

from lgi.schema import SECTIONS, empty_profile, validate_profile


def clean(value: str | None) -> str:
    return " ".join((value or "").split())


def _text(node: Tag | None, selector: str) -> str:
    found = node.select_one(selector) if node else None
    return clean(found.get_text(" ", strip=True)) if found else ""


def _month(value: str | None) -> str:
    value = clean(value)
    for fmt in ("%Y-%m", "%Y-%m-%d", "%B %Y", "%b %Y", "%Y"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m")
        except ValueError:
            pass
    return ""


def _year(value: str | None) -> int:
    match = re.search(r"\b\d{4}\b", clean(value))
    return int(match.group()) if match else 0


def _json_ld_person(soup: BeautifulSoup) -> dict:
    for script in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(script.string or script.get_text())
        except (ValueError, TypeError):
            continue
        candidates = data if isinstance(data, list) else [data]
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("@type") == "Person":
                return candidate
    return {}


def parse_html(html: str, source: str = "fixture", profile_id: str = "p01") -> dict:
    """Only explicit page content becomes a field; missing sections remain uncovered."""
    soup = BeautifulSoup(html, "lxml")
    profile = empty_profile(profile_id, source)
    main = soup.select_one("main[data-profile-id], main") or soup
    basic_node = main.select_one("section#basic") or main
    ld = _json_ld_person(soup)
    basic = profile["basic"]
    basic["name"] = _text(basic_node, "h1, .name") or clean(ld.get("name"))
    basic["headline"] = _text(basic_node, ".headline, [data-field='headline']") or clean(ld.get("jobTitle"))
    basic["location"] = _text(basic_node, ".location, [data-field='location']")
    basic["summary"] = _text(main, "section#about p") or _text(main, "section#about") or clean(ld.get("description"))
    link = basic_node.select_one("a.profile-url, a[rel='canonical']")
    basic["profile_url"] = clean(link.get("href")) if link else clean(ld.get("url"))
    profile["coverage"]["basic"] = all(basic.values())

    experience_complete = True
    for item in main.select("section#experience .experience-item, section#experience article, section#experience li"):
        times = item.select("time")
        start = _month(item.get("data-start") or (times[0].get("datetime") if times else "") or _text(item, ".start"))
        end_raw = item.get("data-end") or (times[1].get("datetime") if len(times) > 1 else "") or _text(item, ".end")
        end = None if clean(end_raw).casefold() in ("present", "current", "") else _month(end_raw)
        if not start or (end_raw and end is None and clean(end_raw).casefold() not in ("present", "current")):
            profile["warnings"].append("Skipped experience with missing or invalid date")
            experience_complete = False
            continue
        if not end_raw:
            experience_complete = False
            profile["warnings"].append("Experience end date missing; present status unverified")
        profile["experience"].append({
            "company": clean(item.get("data-company")) or _text(item, ".company, [data-field='company']"),
            "title": clean(item.get("data-title")) or _text(item, "h3, .title, [data-field='title']"),
            "start": start,
            "end": end,
            "location": clean(item.get("data-location")) or _text(item, ".location, [data-field='location']"),
            "description": _text(item, ".description, p"),
        })
    profile["coverage"]["experience"] = bool(profile["experience"]) and experience_complete and all(all(row[key] for key in ("company", "title", "start", "location", "description")) for row in profile["experience"])

    for item in main.select("section#education .education-item, section#education article, section#education li"):
        times = item.select("time")
        start = _year(item.get("data-start-year") or (times[0].get("datetime") if times else "") or _text(item, ".start"))
        end = _year(item.get("data-end-year") or (times[1].get("datetime") if len(times) > 1 else "") or _text(item, ".end"))
        profile["education"].append({
            "school": clean(item.get("data-school")) or _text(item, ".school, h3"),
            "degree": clean(item.get("data-degree")) or _text(item, ".degree"),
            "field": clean(item.get("data-field")) or _text(item, ".field"),
            "start_year": start,
            "end_year": end,
        })
    profile["coverage"]["education"] = bool(profile["education"]) and all(all(row[key] for key in ("school", "degree", "field", "start_year", "end_year")) for row in profile["education"])

    seen = set()
    for item in main.select("section#skills .skill, section#skills li"):
        skill = clean(item.get_text(" ", strip=True))
        if skill and skill.casefold() not in seen:
            profile["skills"].append(skill)
            seen.add(skill.casefold())
    profile["coverage"]["skills"] = bool(profile["skills"])
    profile["interests"] = list(dict.fromkeys(clean(item.get_text(" ", strip=True)) for item in main.select("section#interests .interest, section#interests li") if clean(item.get_text(" ", strip=True))))
    profile["coverage"]["interests"] = bool(profile["interests"])

    activity_complete = True
    for item in main.select("section#activity .activity-item, section#activity article, section#activity li"):
        time = item.select_one("time")
        raw_date = clean(item.get("data-date") or (time.get("datetime") if time else ""))
        try:
            date = datetime.fromisoformat(raw_date).date().isoformat()
        except ValueError:
            profile["warnings"].append("Skipped activity with invalid date")
            activity_complete = False
            continue
        raw_engaged = clean(item.get("data-engaged-ids"))
        engaged = [value.strip() for value in raw_engaged.split(",") if value.strip()]
        reaction_raw = item.get("data-reaction-count") or _text(item, ".reaction-count")
        if not reaction_raw:
            activity_complete = False
            profile["warnings"].append("Activity reaction count missing")
        try:
            reactions = int(reaction_raw or 0)
            if reactions < 0:
                raise ValueError("negative reaction count")
        except ValueError:
            reactions = 0
            activity_complete = False
            profile["warnings"].append("Activity reaction count invalid")
        activity_type = clean(item.get("data-type"))
        if activity_type not in {"post", "share", "comment"}:
            activity_complete = False
            profile["warnings"].append("Skipped activity with missing or invalid type")
            continue
        profile["activity"].append({
            "post_id": clean(item.get("data-post-id")) or clean(item.get("id")),
            "date": date,
            "type": activity_type,
            "text": _text(item, ".post-text, .text, p") or clean(item.get_text(" ", strip=True)),
            "reaction_count": reactions,
            "engaged_ids": engaged,
        })
    profile["coverage"]["activity"] = bool(profile["activity"]) and activity_complete and all(row["post_id"] and row["text"] for row in profile["activity"])
    if profile["activity"] and not any(row["engaged_ids"] for row in profile["activity"]):
        profile["warnings"].append("Engager identities unavailable; activity graph layer skipped")
    for section in SECTIONS:
        if not profile["coverage"][section]:
            profile["warnings"].append(f"{section} incomplete or missing in visible page")
    return validate_profile(profile)
