"""Opt-in, public-page fetcher. Never use this adapter in the default pipeline.

The caller must supply a URL file and acknowledge the terms-of-service warning.
All unsuccessful URLs still yield a profile with false coverage and a reason, so
downstream reports can account for missing data instead of silently dropping it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
import random
import sys
import time
from typing import Callable, Iterable, Mapping
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import requests

from lgi.schema import empty_profile, validate_profile


class FetchError(RuntimeError):
    """A public URL could not be fetched under the adapter's safety rules."""


class AuthWallError(FetchError):
    """The server required authentication or presented an authwall."""


class RobotsDisallowed(FetchError):
    """Robots rules forbid the URL or could not be checked safely."""


@dataclass(frozen=True)
class LiveSettings:
    """Conservative defaults for user-initiated public-page requests."""

    user_agent: str = "LinkedIn-Graph-Intelligence/1.0 (public profile research; no login)"
    accept_language: str = "en-US,en;q=0.8"
    timeout_seconds: float = 10.0
    min_delay_seconds: float = 2.0
    jitter_seconds: float = 0.5
    backoff_seconds: float = 2.0
    max_attempts: int = 3
    seed: int = 42

    @classmethod
    def from_config(cls, config: Mapping[str, object]) -> "LiveSettings":
        """Read the live subsection while allowing a small standalone config."""
        nested = config.get("live", config)
        values = nested if isinstance(nested, Mapping) else {}
        return cls(
            user_agent=str(values.get("user_agent", cls.user_agent)),
            accept_language=str(values.get("accept_language", cls.accept_language)),
            timeout_seconds=float(values.get("timeout_seconds", cls.timeout_seconds)),
            min_delay_seconds=float(values.get("min_delay_seconds", cls.min_delay_seconds)),
            jitter_seconds=float(values.get("jitter_seconds", cls.jitter_seconds)),
            backoff_seconds=float(values.get("backoff_seconds", cls.backoff_seconds)),
            max_attempts=int(values.get("max_attempts", cls.max_attempts)),
            seed=int(config.get("seed", cls.seed)),
        )

    def __post_init__(self) -> None:
        if "LinkedIn-Graph-Intelligence" not in self.user_agent or "mozilla" in self.user_agent.lower():
            raise ValueError("Use an honest project-identifying User-Agent")
        if self.timeout_seconds <= 0 or self.max_attempts < 1:
            raise ValueError("Timeout and max_attempts must be positive")
        if min(self.min_delay_seconds, self.jitter_seconds, self.backoff_seconds) < 0:
            raise ValueError("Delay and backoff settings cannot be negative")


def _empty_profile(profile_id: str, url: str, reason: str) -> dict:
    """Keep failed fetches visible without inventing any profile section."""
    profile = empty_profile(profile_id, "live")
    profile["basic"]["profile_url"] = url
    profile["warnings"].append(reason)
    return validate_profile(profile)


def _is_authwall_html(html: str) -> bool:
    """Identify login forms without treating a normal link to login as a wall."""
    soup = BeautifulSoup(html, "html.parser")
    for form in soup.find_all("form"):
        action = str(form.get("action", "")).lower()
        if any(token in action for token in ("login", "signin", "authwall", "checkpoint")):
            return True
        if form.find("input", attrs={"type": "password"}):
            return True
    return False


def _retry_after_seconds(value: str | None) -> float:
    """HTTP permits both delay seconds and an absolute date in Retry-After."""
    if not value:
        return 0.0
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            deadline = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return 0.0
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=timezone.utc)
        return max(0.0, (deadline - datetime.now(timezone.utc)).total_seconds())


class LiveAdapter:
    """Read only publicly visible pages after a cached robots check per host."""

    def __init__(
        self,
        urls_path: Path | str,
        settings: LiveSettings | None = None,
        *,
        acknowledged: bool = False,
        sleep_fn: Callable[[float], None] = time.sleep,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not acknowledged:
            raise ValueError("Live use requires --i-understand-tos")
        self.urls_path = Path(urls_path)
        self.settings = settings or LiveSettings()
        self._sleep = sleep_fn
        self._clock = clock_fn
        self._rng = random.Random(self.settings.seed)
        self._session = requests.Session()
        self._session.trust_env = False  # Avoid ambient proxies and credentials.
        self._session.auth = None
        self._session.headers.clear()
        self._session.headers.update(
            {
                "User-Agent": self.settings.user_agent,
                "Accept": "text/html,application/xhtml+xml;q=0.9,text/plain;q=0.8",
                "Accept-Language": self.settings.accept_language,
            }
        )
        self._robots: dict[str, RobotFileParser | bool | str] = {}
        self._last_request: dict[str, float] = {}

    def _throttle(self, host: str) -> None:
        """Apply the same per-host limit to robots and profile requests."""
        previous = self._last_request.get(host)
        if previous is not None:
            minimum = self.settings.min_delay_seconds
            jitter = self._rng.uniform(0.0, self.settings.jitter_seconds)
            wait = max(0.0, previous + minimum + jitter - self._clock())
            if wait:
                self._sleep(wait)
        self._last_request[host] = self._clock()

    def _get_text(self, url: str, *, allow_not_found: bool = False) -> tuple[int, str]:
        """Retry only transient responses; never follow redirects or send cookies."""
        host = urlsplit(url).netloc.lower()
        for attempt in range(self.settings.max_attempts):
            self._throttle(host)
            self._session.cookies.clear()
            response = None
            try:
                response = self._session.get(
                    url,
                    timeout=self.settings.timeout_seconds,
                    allow_redirects=False,
                )
                status = response.status_code
                target = str(response.headers.get("Location", ""))
                if status == 999:
                    raise AuthWallError("HTTP 999 authwall")
                if 300 <= status < 400:
                    if any(token in target.lower() for token in ("login", "signin", "authwall", "checkpoint")):
                        raise AuthWallError("redirect to login/authwall")
                    raise FetchError(f"redirect refused (HTTP {status})")
                if status == 429 or 500 <= status <= 599:
                    if attempt + 1 == self.settings.max_attempts:
                        raise FetchError(f"HTTP {status} after {self.settings.max_attempts} attempts")
                    backoff = self.settings.backoff_seconds * (2**attempt)
                    backoff += self._rng.uniform(0.0, self.settings.jitter_seconds)
                    retry_after = _retry_after_seconds(response.headers.get("Retry-After"))
                    self._sleep(max(backoff, retry_after))
                    continue
                if status == 404 and allow_not_found:
                    return status, ""
                if status >= 400:
                    raise FetchError(f"HTTP {status}")
                body = response.text
                content_type = str(response.headers.get("Content-Type", ""))
                if ("html" in content_type.lower() or "<form" in body.lower()) and _is_authwall_html(body):
                    raise AuthWallError("login form/authwall in body")
                return status, body
            except requests.RequestException as exc:
                if attempt + 1 == self.settings.max_attempts:
                    raise FetchError(f"network error: {type(exc).__name__}") from exc
                backoff = self.settings.backoff_seconds * (2**attempt)
                self._sleep(backoff + self._rng.uniform(0.0, self.settings.jitter_seconds))
            finally:
                if response is not None:
                    response.close()
                self._session.cookies.clear()
        raise FetchError("request attempts exhausted")

    def _check_robots(self, url: str) -> None:
        """Fail closed if robots.txt cannot be fetched or interpreted safely."""
        parts = urlsplit(url)
        host = parts.netloc.lower()
        cached = self._robots.get(host)
        if cached is None:
            robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
            try:
                status, body = self._get_text(robots_url, allow_not_found=True)
                if status == 404:
                    cached = True
                else:
                    parser = RobotFileParser()
                    parser.set_url(robots_url)
                    parser.parse(body.splitlines())
                    cached = parser
            except FetchError as exc:
                cached = f"robots.txt unavailable: {exc}"
            self._robots[host] = cached
        if cached is True:
            return
        if isinstance(cached, str):
            raise RobotsDisallowed(cached)
        if not cached.can_fetch(self.settings.user_agent, url):
            raise RobotsDisallowed("robots.txt disallows this URL")

    @staticmethod
    def _validate_url(url: str) -> None:
        """Reject local paths, embedded credentials, and non-HTTPS targets."""
        parts = urlsplit(url)
        if parts.scheme.lower() != "https" or not parts.hostname:
            raise FetchError("only HTTPS public profile URLs are accepted")
        if parts.username or parts.password:
            raise FetchError("URLs with embedded credentials are rejected")
        if parts.hostname != "linkedin.com" and not parts.hostname.endswith(".linkedin.com"):
            raise FetchError("only LinkedIn profile hosts are accepted")
        try:
            port = parts.port
        except ValueError as exc:
            raise FetchError("invalid URL port") from exc
        if port not in (None, 443):
            raise FetchError("nonstandard URL ports are rejected")

    def iter_profiles(self) -> Iterable[dict]:
        """Continue after a per-URL failure, preserving the reason in warnings."""
        from lgi.parse import parse_html

        print("Warning: fetching public LinkedIn pages may violate the site terms of service.", file=sys.stderr)
        try:
            urls = [line.strip() for line in self.urls_path.read_text(encoding="utf-8").splitlines()]
            urls = [line for line in urls if line and not line.startswith("#")]
            for index, url in enumerate(urls, start=1):
                profile_id = f"p{index:02d}"
                try:
                    self._validate_url(url)
                    self._check_robots(url)
                    _, html = self._get_text(url)
                    profile = parse_html(html, source="live", profile_id=profile_id)
                    profile["id"] = profile_id
                    profile["source"] = "live"
                    profile["basic"]["profile_url"] = url
                    yield validate_profile(profile)
                except (FetchError, ValueError) as exc:
                    yield _empty_profile(profile_id, url, str(exc))
        finally:
            self._session.close()
