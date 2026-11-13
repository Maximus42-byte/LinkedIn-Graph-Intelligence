"""Load local-only credentials and a stable HMAC salt for real-data runs."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import shutil
import stat
import subprocess


PLACEHOLDER_KEY = "PASTE_YOUR_OPENALEX_API_KEY_HERE"


def _require_ignored(path: Path) -> None:
    """Refuse sensitive files that Git could include in a commit."""
    if shutil.which("git") is None:
        return
    result = subprocess.run(
        ["git", "-C", str(path.parent), "check-ignore", "-q", str(path.resolve())],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Sensitive file is not git-ignored: {path}. Add it to .gitignore before running.")


def load_secrets(path: str | Path, require_key: bool = False) -> tuple[str, bytes]:
    """Return the API key and a persistent salt without echoing either secret."""
    path = Path(path)
    if path.exists():
        _require_ignored(path)
        mode = stat.S_IMODE(path.stat().st_mode)
        if os.name == "posix" and mode & 0o044:
            print(f"Warning: {path} is readable by other users; run chmod 600 {path}")
        try:
            content = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ValueError("Secrets file could not be parsed") from None
        if not isinstance(content, dict):
            raise ValueError("Secrets file must be a JSON object")
    else:
        if require_key:
            raise ValueError(f"OpenAlex requires a local secrets file at {path}; copy secrets.example.json first")
        content = {}
    key = content.get("openalex_api_key", "")
    if not isinstance(key, str):
        raise ValueError("openalex_api_key must be a string")
    if require_key and (not key.strip() or key.strip() == PLACEHOLDER_KEY):
        raise ValueError("OpenAlex API key is empty or still the placeholder in the secrets file")
    raw_salt = content.get("anon_salt", "")
    if raw_salt and not isinstance(raw_salt, str):
        raise ValueError("anon_salt must be a string")
    salt_path = path.parent / "anon_salt"
    if raw_salt:
        salt = raw_salt.encode("utf-8")
    else:
        _require_ignored(salt_path)
        if salt_path.exists():
            salt = salt_path.read_bytes().strip()
        else:
            salt = secrets.token_hex(32).encode("ascii")
            salt_path.parent.mkdir(parents=True, exist_ok=True)
            salt_path.write_bytes(salt + b"\n")
            if os.name == "posix":
                salt_path.chmod(0o600)
    return key, salt


def pseudonym(salt: bytes, category: str, value: str) -> str:
    digest = hmac.new(salt, (category + "\x00" + value.casefold().strip()).encode("utf-8"), hashlib.sha256).hexdigest()[:16]
    return f"{category}_{digest}"


def anonymise_profile(profile: dict, salt: bytes) -> dict:
    """Keep equality of categorical identifiers while removing direct identifiers and prose."""
    import copy

    p = copy.deepcopy(profile)
    p["id"] = pseudonym(salt, "person", profile["id"])
    basic = p["basic"]
    basic["name"] = pseudonym(salt, "person", basic["name"] or profile["id"])
    basic["profile_url"] = pseudonym(salt, "url", basic["profile_url"]) if basic["profile_url"] else ""
    basic["headline"] = ""
    basic["summary"] = ""
    basic["location"] = ""
    for job in p["experience"]:
        job["company"] = pseudonym(salt, "employer", job["company"]) if job["company"] else ""
        job["title"] = ""
        job["location"] = ""
        job["description"] = ""
    for school in p["education"]:
        school["school"] = pseudonym(salt, "institution", school["school"]) if school["school"] else ""
        school["degree"] = ""
        school["field"] = ""
    p["skills"] = [pseudonym(salt, "skill", value) for value in p["skills"]]
    p["interests"] = []
    for post in p["activity"]:
        post["post_id"] = pseudonym(salt, "work", post["post_id"])
        post["text"] = ""
        post["engaged_ids"] = [pseudonym(salt, "person", value) for value in post["engaged_ids"]]
    for key in ("basic", "experience", "education", "interests", "activity"):
        p["coverage"][key] = False
    p["warnings"] = [
        "Anonymised: identifiers replaced with keyed hashes and free text removed",
        *[f"{section} incomplete or unavailable" for section, covered in p["coverage"].items() if not covered],
    ]
    return p
