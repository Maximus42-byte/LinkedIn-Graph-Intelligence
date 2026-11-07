import copy
import pytest

from lgi.schema import empty_profile, validate_profile


def test_schema_accepts_empty_but_honest_profile() -> None:
    profile = empty_profile("p01", "manual")
    assert validate_profile(profile) is profile
    assert not any(profile["coverage"].values())


@pytest.mark.parametrize("change", [
    lambda p: p["coverage"].pop("skills"),
    lambda p: p["skills"].extend(["Python", "python"]),
    lambda p: p["experience"].append({"company": "X"}),
    lambda p: p.update(source="unknown"),
])
def test_schema_rejects_broken_profiles(change) -> None:
    profile = copy.deepcopy(empty_profile("p01", "manual"))
    change(profile)
    with pytest.raises(ValueError):
        validate_profile(profile)
