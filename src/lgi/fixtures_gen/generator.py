"""Generate deterministic, visibly synthetic profile pages for offline analysis.

The ground-truth file is kept separate from the HTML. Downstream extraction must
learn only from the page content; validation may later read the intended labels.
"""

from __future__ import annotations

import argparse
from html import escape
import json
from pathlib import Path
import random
from typing import Any

from . import vocab


def _assign_teams(n: int, seed: int) -> list[str]:
    """Preserve the designed small example and balance larger examples."""
    designed = ["ML"] * 3 + ["Platform"] * 3 + ["Security"] * 3 + ["Bridge"]
    teams = designed[:n]
    counts = {team: teams.count(team) for team in vocab.TEAMS}
    rng = random.Random(seed)
    while len(teams) < n:
        minimum = min(counts.values())
        choices = [team for team in vocab.TEAMS if counts[team] == minimum]
        team = rng.choice(choices)
        teams.append(team)
        counts[team] += 1
    return teams


def _title(team: str, team_index: int) -> str:
    if team == "Bridge":
        return "Cross-team Integration Lead"
    titles = vocab.TITLES[team]
    return titles[team_index % len(titles)]


def _experience(team: str, current_title: str, index: int) -> list[dict[str, Any]]:
    """Past jobs create known date-overlap edges and real career transitions."""
    if team == "Bridge":
        return [
            {
                "company": vocab.ORG,
                "title": current_title,
                "start": "2022-01",
                "end": None,
                "location": vocab.LOCATION,
                "description": "Connects the fictional model and platform teams through shared architecture reviews.",
            },
            {
                "company": vocab.PAST_EMPLOYERS["Platform"],
                "title": "Platform Engineer",
                "start": "2020-01",
                "end": "2021-12",
                "location": vocab.LOCATION,
                "description": "Built synthetic service reliability and observability exercises.",
            },
            {
                "company": vocab.PAST_EMPLOYERS["ML"],
                "title": "ML Engineer",
                "start": "2018-01",
                "end": "2019-12",
                "location": vocab.LOCATION,
                "description": "Documented fictional model evaluation and data pipeline work.",
            },
        ]
    current_year = 2021 + (index % 3)
    return [
        {
            "company": vocab.ORG,
            "title": current_title,
            "start": f"{current_year}-03",
            "end": None,
            "location": vocab.LOCATION,
            "description": f"Works on synthetic {team.lower()} projects in the fictional organisation.",
        },
        {
            "company": vocab.PAST_EMPLOYERS[team],
            "title": vocab.PAST_TITLES[team],
            "start": "2018-01",
            "end": f"{current_year}-02",
            "location": vocab.LOCATION,
            "description": f"Completed fictional {team.lower()} projects with overlapping team tenure.",
        },
    ]


def _education(team: str, index: int) -> list[dict[str, Any]]:
    school = vocab.SCHOOLS["ML"] if team == "Bridge" else vocab.SCHOOLS[team]
    if index in (3, 4):
        school = vocab.SHARED_SCHOOL
    field = {
        "ML": "Imaginary Computer Science",
        "Platform": "Fictional Systems Engineering",
        "Security": "Imaginary Information Security",
        "Bridge": "Fictional Computing",
    }[team]
    start_year = 2013 + (index % 4)
    return [{
        "school": school,
        "degree": "BSc" if index % 3 else "MSc",
        "field": field,
        "start_year": start_year,
        "end_year": start_year + 4,
    }]


def _skills(team: str, index: int) -> list[str]:
    if team == "Bridge":
        return [*vocab.COMMON_SKILLS, "Model evaluation", "Data pipelines",
                "Distributed systems", "Observability"]
    team_skills = vocab.TEAM_SKILLS[team]
    selected = [team_skills[(index + offset) % len(team_skills)] for offset in range(4)]
    skills = [*vocab.COMMON_SKILLS, *selected]
    if index == 9:
        skills.append(vocab.RARE_SKILL)
    return skills


def _engaged_ids(profile_index: int, teams: list[str], post_number: int) -> list[str]:
    """Interactions mostly follow teams, with a few visible cross-team ties."""
    team = teams[profile_index - 1]
    if team == "Bridge":
        colleagues = [1, 4, 7] if post_number == 1 else [3, 6, 9]
        colleagues = [number for number in colleagues if number <= len(teams)]
    else:
        colleagues = [number for number, other_team in enumerate(teams, 1)
                      if other_team == team and number != profile_index]
        if colleagues:
            offset = (profile_index + post_number) % len(colleagues)
            colleagues = (colleagues[offset:] + colleagues[:offset])[:2]
        if len(teams) >= 10 and (profile_index in (3, 4, 7) or profile_index % 11 == 0):
            colleagues.append(10)
        if len(teams) > 10 and profile_index % 13 == 0:
            other = next((number for number, other_team in enumerate(teams, 1)
                          if other_team not in (team, "Bridge")), None)
            if other is not None:
                colleagues.append(other)
    return [f"p{number:02d}" for number in dict.fromkeys(colleagues)]


def _activity(team: str, index: int, teams: list[str], rng: random.Random) -> list[dict[str, Any]]:
    templates = vocab.POSTS["Mismatch"] if index == 9 else vocab.POSTS[team]
    rows = []
    for post_number in (1, 2):
        engaged_ids = _engaged_ids(index, teams, post_number)
        rows.append({
            "post_id": f"p{index:02d}-a{post_number}",
            "date": f"2025-0{post_number + 2}-{(index % 28) + 1:02d}",
            "type": "post" if post_number == 1 else ("share" if index % 2 else "comment"),
            "text": templates[(index + post_number - 2) % len(templates)],
            "reaction_count": len(engaged_ids) + rng.randint(2, 10),
            "engaged_ids": engaged_ids,
        })
    return rows


def _profile(index: int, team: str, team_index: int, teams: list[str], seed: int) -> dict[str, Any]:
    """Build the schema shape first so HTML rendering cannot omit a section."""
    rng = random.Random(seed * 1009 + index * 9176)
    profile_id = f"p{index:02d}"
    title = _title(team, team_index)
    focus = "cross-team model and platform integration" if team == "Bridge" else f"{team.lower()} practice"
    return {
        "id": profile_id,
        "source": "fixture",
        "basic": {
            "name": f"Synthetic Persona {index:02d}",
            "headline": f"{title} at {vocab.ORG}",
            "location": vocab.LOCATION,
            "summary": f"This fictional profile represents {focus} at {vocab.ORG}. All work and activity here are synthetic.",
            "profile_url": f"https://example.invalid/synthetic/{profile_id}",
        },
        "experience": _experience(team, title, index),
        "education": _education(team, index),
        "skills": _skills(team, index),
        "interests": list(vocab.INTERESTS[team]),
        "activity": _activity(team, index, teams, rng),
        "coverage": {section: True for section in
                     ("basic", "experience", "education", "skills", "interests", "activity")},
        "warnings": [],
    }


def build_profiles(n: int = 10, seed: int = 73) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return fixture records and separate labels without reading external data."""
    if n < 1:
        raise ValueError("n must be positive")
    teams = _assign_teams(n, seed)
    profiles: list[dict[str, Any]] = []
    labels: dict[str, dict[str, str]] = {}
    team_counts = {team: 0 for team in (*vocab.TEAMS, "Bridge")}
    for index, team in enumerate(teams, 1):
        team_index = team_counts[team]
        team_counts[team] += 1
        profile = _profile(index, team, team_index, teams, seed)
        profiles.append(profile)
        labels[profile["id"]] = {"team": team, "role": profile["experience"][0]["title"]}
    ground_truth = {
        "seed": seed,
        "n": n,
        "profiles": labels,
        "post_skill_mismatch_ids": ["p09"] if n >= 9 else [],
    }
    return profiles, ground_truth


def _render_experience(rows: list[dict[str, Any]]) -> str:
    items = []
    for row in rows:
        end = '<span class="end">Present</span>' if row["end"] is None else (
            f'<time class="end" datetime="{row["end"]}">{row["end"]}</time>')
        items.append(
            '      <li class="experience-item">\n'
            f'        <h3 class="title">{escape(row["title"])}</h3>\n'
            f'        <span class="company">{escape(row["company"])}</span>\n'
            f'        <time class="start" datetime="{row["start"]}">{row["start"]}</time> '
            f'to {end}\n'
            f'        <span class="location">{escape(row["location"])}</span>\n'
            f'        <p class="description">{escape(row["description"])}</p>\n'
            '      </li>'
        )
    return "\n".join(items)


def _render_education(rows: list[dict[str, Any]]) -> str:
    items = []
    for row in rows:
        items.append(
            '      <li class="education-item">\n'
            f'        <h3 class="school">{escape(row["school"])}</h3>\n'
            f'        <span class="degree">{escape(row["degree"])}</span>\n'
            f'        <span class="field">{escape(row["field"])}</span>\n'
            f'        <time class="start-year" datetime="{row["start_year"]}">{row["start_year"]}</time> '
            f'to <time class="end-year" datetime="{row["end_year"]}">{row["end_year"]}</time>\n'
            '      </li>'
        )
    return "\n".join(items)


def _render_activity(rows: list[dict[str, Any]]) -> str:
    items = []
    for row in rows:
        ids = ",".join(row["engaged_ids"])
        items.append(
            f'      <li class="activity-item" data-post-id="{row["post_id"]}" '
            f'data-type="{row["type"]}" data-engaged-ids="{ids}">\n'
            f'        <time class="date" datetime="{row["date"]}">{row["date"]}</time>\n'
            f'        <span class="type">{row["type"]}</span>\n'
            f'        <p class="text">{escape(row["text"])}</p>\n'
            f'        <span class="reaction-count">{row["reaction_count"]}</span> reactions\n'
            f'        <span class="engaged-ids">{escape(", ".join(row["engaged_ids"]))}</span>\n'
            '      </li>'
        )
    return "\n".join(items)


def render_html(profile: dict[str, Any]) -> str:
    """Use ordinary semantic sections and explicit source-provided interactions."""
    basic = profile["basic"]
    ld = json.dumps({
        "@context": "https://schema.org",
        "@type": "Person",
        "name": basic["name"],
        "jobTitle": profile["experience"][0]["title"],
        "url": basic["profile_url"],
        "description": basic["summary"],
    }, sort_keys=True, ensure_ascii=False).replace("<", "\\u003c")
    skills = list(profile["skills"])
    if profile["id"] == "p02":
        skills.append("python")  # Deliberate case variant exercises parser deduplication.
    skill_html = "\n".join(f'      <li class="skill">{escape(skill)}</li>' for skill in skills)
    interest_html = "\n".join(
        f'      <li class="interest">{escape(interest)}</li>' for interest in profile["interests"])
    return (
        '<!doctype html>\n'
        '<html lang="en">\n'
        '<head>\n'
        '  <meta charset="utf-8">\n'
        '  <meta name="synthetic" content="true">\n'
        f'  <title>{escape(basic["name"])} | Synthetic Profile</title>\n'
        f'  <script type="application/ld+json">{ld}</script>\n'
        '</head>\n'
        '<body>\n'
        '  <aside class="synthetic-banner">SYNTHETIC DATA — fictional people, organisations, schools, and activity.</aside>\n'
        f'  <main data-profile-id="{profile["id"]}">\n'
        '    <section id="basic" aria-label="Basic information">\n'
        f'      <h1 class="name">{escape(basic["name"])}</h1>\n'
        f'      <p class="headline">{escape(basic["headline"])}</p>\n'
        f'      <p class="location">{escape(basic["location"])}</p>\n'
        f'      <a class="profile-url" href="{escape(basic["profile_url"], quote=True)}">Profile URL</a>\n'
        '    </section>\n'
        '    <section id="about" aria-label="About">\n'
        '      <h2>About</h2>\n'
        f'      <p class="summary">{escape(basic["summary"])}</p>\n'
        '    </section>\n'
        '    <section id="experience" aria-label="Experience">\n'
        '      <h2>Experience</h2>\n'
        '      <ul>\n'
        f'{_render_experience(profile["experience"])}\n'
        '      </ul>\n'
        '    </section>\n'
        '    <section id="education" aria-label="Education">\n'
        '      <h2>Education</h2>\n'
        '      <ul>\n'
        f'{_render_education(profile["education"])}\n'
        '      </ul>\n'
        '    </section>\n'
        '    <section id="skills" aria-label="Skills">\n'
        '      <h2>Skills</h2>\n'
        '      <ul>\n'
        f'{skill_html}\n'
        '      </ul>\n'
        '    </section>\n'
        '    <section id="interests" aria-label="Interests">\n'
        '      <h2>Interests</h2>\n'
        '      <ul>\n'
        f'{interest_html}\n'
        '      </ul>\n'
        '    </section>\n'
        '    <section id="activity" aria-label="Activity">\n'
        '      <h2>Activity</h2>\n'
        '      <ul>\n'
        f'{_render_activity(profile["activity"])}\n'
        '      </ul>\n'
        '    </section>\n'
        '  </main>\n'
        '</body>\n'
        '</html>\n'
    )


def generate_fixtures(n: int = 10, seed: int = 73, root: Path = Path(".")) -> list[dict[str, Any]]:
    """Write an exact fixture set, removing stale pages from a prior larger run."""
    root = Path(root)
    fixture_dir = root / "data" / "fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    profiles, ground_truth = build_profiles(n, seed)
    for path in fixture_dir.glob("profile_*.html"):
        path.unlink()
    for profile in profiles:
        index = int(profile["id"][1:])
        (fixture_dir / f"profile_{index:02d}.html").write_text(
            render_html(profile), encoding="utf-8")
    truth_path = root / "data" / "ground_truth.json"
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(json.dumps(ground_truth, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline synthetic profile pages")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.seed is None:
        config_path = args.root / "config.yaml"
        if config_path.exists():
            from lgi.config import load_config
            seed = int(load_config(config_path)["seed"])
        else:
            seed = 73
    else:
        seed = args.seed
    profiles = generate_fixtures(n=args.n, seed=seed, root=args.root)
    print(f"Generated {len(profiles)} synthetic profiles in {args.root / 'data' / 'fixtures'}")


if __name__ == "__main__":
    main()
