"""One downstream contract for every profile source."""

from typing import Iterable, Protocol


class SourceAdapter(Protocol):
    def iter_profiles(self) -> Iterable[dict]:
        """Yield schema-validated profiles in stable order."""
        ...
