"""Offline HTML fixtures exercise the same parser as public HTML."""

from pathlib import Path
from typing import Iterable

from lgi.parse import parse_html


class FixtureAdapter:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def iter_profiles(self) -> Iterable[dict]:
        for path in sorted(self.directory.glob("profile_*.html")):
            yield parse_html(path.read_text(encoding="utf-8"), source="fixture", profile_id=path.stem.replace("profile_", "p"))
