"""Manually supplied schema JSON, with deterministic pseudonymisation."""

import hashlib
import json
from pathlib import Path
from typing import Iterable

from lgi.schema import validate_profile


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16] if value else ""


class ManualAdapter:
    def __init__(self, directory: str | Path, anonymise: bool = False):
        self.directory = Path(directory)
        self.anonymise = anonymise

    def iter_profiles(self) -> Iterable[dict]:
        for path in sorted(self.directory.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            profiles = data if isinstance(data, list) else [data]
            for profile in profiles:
                profile = validate_profile(profile)
                profile["source"] = "manual"
                if self.anonymise:
                    profile["basic"]["name"] = f"anon-{_hash(profile['basic']['name'])}"
                    profile["basic"]["profile_url"] = f"hash:{_hash(profile['basic']['profile_url'])}"
                yield validate_profile(profile)
