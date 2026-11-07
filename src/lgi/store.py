"""Deterministic JSON and flat CSV exports for inspection or analysis."""

import csv
import json
from pathlib import Path

from lgi.schema import validate_profile


def write_profiles(profiles: list[dict], json_path: str | Path, flat_dir: str | Path) -> None:
    """Sorted rows and keys make two runs byte-for-byte comparable."""
    profiles = sorted((validate_profile(p) for p in profiles), key=lambda p: p["id"])
    json_path = Path(json_path)
    flat_dir = Path(flat_dir)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    flat_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(profiles, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tables: dict[str, list[dict]] = {name: [] for name in ("basic", "experience", "education", "skills", "interests", "activity")}
    for profile in profiles:
        pid = profile["id"]
        tables["basic"].append({"id": pid, "source": profile["source"], **profile["basic"], **{f"coverage_{key}": value for key, value in profile["coverage"].items()}})
        for name in ("experience", "education", "activity"):
            for row in profile[name]:
                copy = {"id": pid, **row}
                if name == "activity":
                    copy["engaged_ids"] = ";".join(copy["engaged_ids"])
                tables[name].append(copy)
        for name in ("skills", "interests"):
            for value in profile[name]:
                tables[name].append({"id": pid, name[:-1] if name == "skills" else "interest": value})
    fields = {
        "basic": ["id", "source", "name", "headline", "location", "summary", "profile_url", *[f"coverage_{name}" for name in ("basic", "experience", "education", "skills", "interests", "activity")]],
        "experience": ["id", "company", "title", "start", "end", "location", "description"],
        "education": ["id", "school", "degree", "field", "start_year", "end_year"],
        "skills": ["id", "skill"],
        "interests": ["id", "interest"],
        "activity": ["id", "post_id", "date", "type", "text", "reaction_count", "engaged_ids"],
    }
    for name, rows in tables.items():
        with (flat_dir / f"{name}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields[name])
            writer.writeheader()
            writer.writerows(rows)
