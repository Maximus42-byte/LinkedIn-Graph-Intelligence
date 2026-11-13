"""Official OpenAlex API adapter for institution co-authorship studies.

No request occurs until a non-placeholder owner-supplied key is loaded by the caller.
Raw responses are cached locally using keys that exclude the API credential.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from itertools import combinations
from pathlib import Path
import re
import time
from typing import Iterable
from urllib.parse import quote

import requests

from lgi.schema import empty_profile, validate_profile


API = "https://api.openalex.org"
ID_PATTERN = re.compile(r"(?:https?://openalex[.]org/)?(I\d+)$", re.IGNORECASE)


def _id(value: str) -> str:
    return value.rsplit("/", 1)[-1]


class OpenAlexAdapter:
    def __init__(
        self,
        institution: str,
        api_key: str,
        cache_dir: str | Path,
        *,
        year_from: int = 2020,
        max_authors: int = 30,
        max_requests: int = 20,
        refresh: bool = False,
        session: requests.Session | None = None,
    ):
        if not api_key.strip() or api_key == "PASTE_YOUR_OPENALEX_API_KEY_HERE":
            raise ValueError("OpenAlex requires a non-placeholder API key in the secrets file")
        if not institution.strip():
            raise ValueError("OpenAlex requires an institution name or ID")
        if max_authors < 1 or max_requests < 1 or year_from < 1800:
            raise ValueError("max-authors and max-requests must be positive; year-from must be valid")
        self.institution = institution
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.year_from = year_from
        self.max_authors = max_authors
        self.max_requests = max_requests
        self.refresh = refresh
        self.session = session or requests.Session()
        self.session.headers.update({
            "User-Agent": "LinkedIn-Graph-Intelligence/1.0 (OpenAlex research client; contact: project maintainer)",
            "Accept": "application/json",
        })
        self.session.trust_env = False
        self.request_count = 0
        self.coauthor_counts: dict[tuple[str, str], int] = {}
        self.institution_name = ""
        self.selected_ids: list[str] = []

    def _redact(self, message: str) -> str:
        message = message.replace(self.api_key, "[REDACTED]")
        message = message.replace(quote(self.api_key, safe=""), "[REDACTED]")
        return re.sub(r"(api_key=)[^&\s]+", r"\1[REDACTED]", message, flags=re.IGNORECASE)

    def _get(self, path: str, params: dict[str, object] | None = None) -> dict:
        public_params = dict(params or {})
        identity = json.dumps({"path": path, "params": public_params}, sort_keys=True)
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        cached = self.cache_dir / f"{digest}.json"
        if cached.exists() and not self.refresh:
            return json.loads(cached.read_text(encoding="utf-8"))
        for attempt in range(3):
            if self.request_count >= self.max_requests:
                raise RuntimeError(f"OpenAlex request budget reached ({self.max_requests}); increase --max-requests or rerun from cache")
            self.request_count += 1
            try:
                response = self.session.get(
                    f"{API}{path}",
                    params={**public_params, "api_key": self.api_key},
                    timeout=15,
                    allow_redirects=False,
                )
                if (response.status_code == 429 or 500 <= response.status_code < 600) and attempt < 2:
                    retry_after = response.headers.get("Retry-After", "")
                    try:
                        delay = max(0.0, float(retry_after))
                    except ValueError:
                        delay = float(2 ** attempt)
                    time.sleep(delay)
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("OpenAlex returned a non-object response")
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                serialised = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)
                cached.write_text(serialised.replace(self.api_key, "[REDACTED]").replace(quote(self.api_key, safe=""), "[REDACTED]") + "\n", encoding="utf-8")
                return data
            except Exception as error:
                raise RuntimeError("OpenAlex request failed: " + self._redact(str(error))) from None
        raise RuntimeError("OpenAlex request failed after retries")

    def _institution_id(self) -> str:
        match = ID_PATTERN.fullmatch(self.institution.strip())
        if match:
            institution = self._get(f"/institutions/{match.group(1).upper()}")
        else:
            found = self._get("/institutions", {"search": self.institution, "per_page": 100})
            results = found.get("results", [])
            exact = [row for row in results if row.get("display_name", "").casefold() == self.institution.casefold()]
            institution = sorted(exact or results, key=lambda row: str(row.get("id", "")))[0] if results else {}
        if not institution.get("id"):
            raise ValueError("No matching OpenAlex institution found")
        self.institution_name = str(institution.get("display_name") or _id(str(institution["id"])))
        return _id(str(institution["id"]))

    def _authors(self, institution_id: str) -> list[dict]:
        candidates: list[dict] = []
        page = 1
        while len(candidates) < self.max_authors and page <= 100:
            data = self._get("/authors", {
                "filter": f"last_known_institutions.id:{institution_id}",
                "sort": "works_count:desc",
                "per_page": 100,
                "page": page,
            })
            rows = data.get("results", [])
            candidates.extend(rows)
            if len(rows) < 100:
                break
            page += 1
        selected = sorted(candidates, key=lambda row: (-int(row.get("works_count") or 0), _id(str(row.get("id", "")))))[:self.max_authors]
        self.selected_ids = [_id(str(row["id"])) for row in selected if row.get("id")]
        return selected

    def _works(self) -> list[dict]:
        if not self.selected_ids:
            return []
        works: dict[str, dict] = {}
        page = 1
        author_filter = "|".join(self.selected_ids)
        while page <= 100:
            data = self._get("/works", {
                "filter": f"authorships.author.id:{author_filter},from_publication_date:{self.year_from}-01-01",
                "per_page": 100,
                "page": page,
            })
            rows = data.get("results", [])
            for row in rows:
                if row.get("id"):
                    works[_id(str(row["id"]))] = row
            if len(rows) < 100:
                break
            page += 1
        return [works[key] for key in sorted(works)]

    def iter_profiles(self) -> Iterable[dict]:
        institution_id = self._institution_id()
        authors = self._authors(institution_id)
        works = self._works()
        selected = set(self.selected_ids)
        counts: Counter[tuple[str, str]] = Counter()
        by_author: dict[str, list[dict]] = {author: [] for author in selected}
        for work in works:
            members = sorted({
                _id(str(item.get("author", {}).get("id", "")))
                for item in work.get("authorships", [])
            } & selected)
            for author in members:
                by_author[author].append(work)
            for pair in combinations(members, 2):
                counts[pair] += 1
        self.coauthor_counts = dict(sorted(counts.items()))
        for author in authors:
            aid = _id(str(author["id"]))
            profile = empty_profile(aid, "openalex")
            profile["basic"]["name"] = str(author.get("display_name") or "")
            profile["basic"]["profile_url"] = str(author["id"])
            profile["coverage"]["basic"] = bool(profile["basic"]["name"])
            profile["warnings"].append("OpenAlex does not provide job titles; experience title is unavailable")
            for affiliation in author.get("affiliations", []):
                institution = affiliation.get("institution") or {}
                years = sorted(int(year) for year in affiliation.get("years", []) if str(year).isdigit())
                if not institution.get("display_name") or not years:
                    continue
                profile["experience"].append({
                    "company": str(institution["display_name"]),
                    "title": "",
                    "start": f"{years[0]:04d}-01",
                    "end": f"{years[-1]:04d}-12",
                    "location": "",
                    "description": "",
                })
            profile["warnings"].append("Affiliation years are annual observations, not employment dates")
            profile["skills"] = sorted({
                str(item["display_name"])
                for item in author.get("topics", [])
                if item.get("display_name")
            }, key=str.casefold)
            if profile["skills"]:
                profile["coverage"]["skills"] = True
                profile["warnings"].append("Skills are derived from OpenAlex topics, not declared skills")
            for work in by_author.get(aid, []):
                date = str(work.get("publication_date") or "")
                if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                    continue
                profile["activity"].append({
                    "post_id": _id(str(work["id"])),
                    "date": date,
                    "type": "publication",
                    "text": str(work.get("title") or work.get("display_name") or ""),
                    "reaction_count": 0,
                    "engaged_ids": [],
                })
            profile["activity"].sort(key=lambda row: (row["date"], row["post_id"]))
            profile["coverage"]["activity"] = bool(profile["activity"])
            profile["warnings"].append("Publications are not social posts; citations are not reactions")
            for section in ("education", "interests"):
                profile["warnings"].append(f"{section} unavailable in OpenAlex")
            yield validate_profile(profile)
