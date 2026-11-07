from __future__ import annotations

from lgi.ingest_demo import IngestDemo, idempotency_key


def test_duplicate_url_is_processed_once() -> None:
    demo = IngestDemo[str]()
    url = "https://example.invalid/in/synthetic"
    assert demo.enqueue(url)
    assert not demo.enqueue(url)
    called: list[str] = []

    def process(value: str) -> str:
        called.append(value)
        return "parsed"

    demo.run_worker(process)
    assert called == [url]
    assert demo.results[idempotency_key(url)] == "parsed"
    assert not demo.enqueue(url)


def test_retry_then_success_and_dead_letter() -> None:
    delays: list[float] = []
    demo = IngestDemo[str](max_attempts=3, backoff_seconds=0.5, sleep_fn=delays.append)
    good = "https://example.invalid/good"
    bad = "https://example.invalid/bad"
    demo.enqueue(good)
    demo.enqueue(bad)
    attempts: dict[str, int] = {}

    def process(url: str) -> str:
        attempts[url] = attempts.get(url, 0) + 1
        if url == bad or attempts[url] == 1:
            raise RuntimeError("temporary failure")
        return "parsed"

    demo.run_worker(process)
    assert attempts == {good: 2, bad: 3}
    assert demo.results == {idempotency_key(good): "parsed"}
    assert len(demo.dead_letters) == 1
    assert demo.dead_letters[0].job.key == idempotency_key(bad)
    assert demo.dead_letters[0].reason == "temporary failure"
    assert delays == [0.5, 0.5, 1.0]
