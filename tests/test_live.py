"""All live-adapter HTTP tests use a fake Session; no network is reachable."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import sys
from types import ModuleType
from unittest.mock import patch

import pytest

from lgi.sources.live import LiveAdapter, LiveSettings


@dataclass
class FakeResponse:
    status_code: int
    text: str = ""
    headers: dict[str, str] = field(default_factory=dict)

    def close(self) -> None:
        pass


class FakeCookieJar(dict):
    def clear(self) -> None:
        super().clear()


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict, dict]] = []
        self.headers: dict[str, str] = {}
        self.cookies = FakeCookieJar({"ambient_cookie": "should never be sent"})
        self.auth = None
        self.trust_env = True
        self.closed = False

    def get(self, url: str, **kwargs: object) -> FakeResponse:
        self.calls.append((url, dict(kwargs), dict(self.cookies)))
        response = self.responses.pop(0)
        if "Set-Cookie" in response.headers:
            self.cookies["server_cookie"] = "should also never be sent"
        return response

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep this suite about HTTP behavior rather than the separate parser."""
    module = ModuleType("lgi.parse")

    def parse_html(html: str, *, source: str, profile_id: str) -> dict:
        return {
            "id": profile_id,
            "source": source,
            "basic": {"name": "Synthetic Person", "headline": "", "location": "", "summary": "", "profile_url": ""},
            "experience": [],
            "education": [],
            "skills": [],
            "interests": [],
            "activity": [],
            "coverage": {"basic": True, "experience": False, "education": False, "skills": False, "interests": False, "activity": False},
            "warnings": [],
        }

    module.parse_html = parse_html
    monkeypatch.setitem(sys.modules, "lgi.parse", module)


def url_file(tmp_path: Path, *urls: str) -> Path:
    path = tmp_path / "urls.txt"
    path.write_text("\n".join(urls) + "\n", encoding="utf-8")
    return path


def settings() -> LiveSettings:
    return LiveSettings(min_delay_seconds=0, jitter_seconds=0, backoff_seconds=0.1)


def test_explicit_acknowledgement_required(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="i-understand-tos"):
        LiveAdapter(url_file(tmp_path, "https://www.linkedin.com/in/p01"))


def test_authwall_body_and_http_999_are_recorded_and_next_url_processed(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(200, "User-agent: *\nAllow: /"),
            FakeResponse(200, '<form action="/login"><input type="password"></form>', {"Content-Type": "text/html"}),
            FakeResponse(999, ""),
        ]
    )
    urls = url_file(tmp_path, "https://www.linkedin.com/in/p01", "https://www.linkedin.com/in/p02")
    with patch("lgi.sources.live.requests.Session", return_value=session):
        profiles = list(LiveAdapter(urls, settings(), acknowledged=True, sleep_fn=lambda _: None).iter_profiles())

    assert len(profiles) == 2
    assert all(not any(profile["coverage"].values()) for profile in profiles)
    assert "authwall" in profiles[0]["warnings"][0]
    assert "999" in profiles[1]["warnings"][0]
    assert len(session.calls) == 3
    assert session.closed


def test_retry_after_and_cookie_clearing(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(200, "User-agent: *\nAllow: /", {"Set-Cookie": "x=1"}),
            FakeResponse(429, "", {"Retry-After": "3"}),
            FakeResponse(200, "<html><h1>Public profile</h1></html>", {"Content-Type": "text/html"}),
        ]
    )
    slept: list[float] = []
    urls = url_file(tmp_path, "https://www.linkedin.com/in/p01")
    with patch("lgi.sources.live.requests.Session", return_value=session):
        profiles = list(LiveAdapter(urls, settings(), acknowledged=True, sleep_fn=slept.append).iter_profiles())

    assert profiles[0]["basic"]["profile_url"].endswith("/p01")
    assert len(session.calls) == 3
    assert all(not cookies for _, _, cookies in session.calls)
    assert all(kwargs["allow_redirects"] is False and kwargs["timeout"] > 0 for _, kwargs, _ in session.calls)
    assert "Cookie" not in session.headers
    assert "Mozilla" not in session.headers["User-Agent"]
    assert session.trust_env is False
    assert any(delay >= 3 for delay in slept)


def test_robots_disallow_is_cached_and_profile_is_not_fetched(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(200, "User-agent: *\nDisallow: /in/")])
    urls = url_file(tmp_path, "https://www.linkedin.com/in/p01", "https://www.linkedin.com/in/p02")
    with patch("lgi.sources.live.requests.Session", return_value=session):
        profiles = list(LiveAdapter(urls, settings(), acknowledged=True, sleep_fn=lambda _: None).iter_profiles())

    assert len(session.calls) == 1
    assert len(profiles) == 2
    assert all("robots.txt disallows" in profile["warnings"][0] for profile in profiles)
    assert all(not any(profile["coverage"].values()) for profile in profiles)


def test_login_redirect_is_not_followed(tmp_path: Path) -> None:
    session = FakeSession(
        [
            FakeResponse(200, "User-agent: *\nAllow: /"),
            FakeResponse(302, "", {"Location": "/authwall?target=profile"}),
        ]
    )
    with patch("lgi.sources.live.requests.Session", return_value=session):
        profiles = list(
            LiveAdapter(
                url_file(tmp_path, "https://www.linkedin.com/in/p01"),
                settings(),
                acknowledged=True,
                sleep_fn=lambda _: None,
            ).iter_profiles()
        )

    assert "redirect to login/authwall" in profiles[0]["warnings"][0]
    assert len(session.calls) == 2


def test_robots_404_means_no_rules(tmp_path: Path) -> None:
    session = FakeSession([FakeResponse(404), FakeResponse(200, "<html>Public</html>")])
    with patch("lgi.sources.live.requests.Session", return_value=session):
        profiles = list(
            LiveAdapter(
                url_file(tmp_path, "https://www.linkedin.com/in/p01"),
                settings(),
                acknowledged=True,
                sleep_fn=lambda _: None,
            ).iter_profiles()
        )
    assert profiles[0]["coverage"]["basic"] is True
    assert len(session.calls) == 2
