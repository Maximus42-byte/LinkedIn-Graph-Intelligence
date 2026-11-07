"""Small stdlib model of at-least-once URL ingestion; no service is started.

The queue's idempotency key is a stable URL hash. The worker writes results by
that key, so duplicate deliveries replace the same record. A real Kafka or
Redis deployment would need durable key storage and host-based partitions.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from queue import Empty, Queue
import time
from typing import Callable, Generic, TypeVar
from urllib.parse import urlsplit


T = TypeVar("T")


def idempotency_key(url: str) -> str:
    """Use a stable key that works across processes and repeated deliveries."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Job:
    url: str
    key: str
    host: str
    attempts: int = 0


@dataclass(frozen=True)
class DeadLetter:
    job: Job
    reason: str


class IngestDemo(Generic[T]):
    """A single-worker queue with retries, deduplication and dead letters."""

    def __init__(
        self,
        *,
        max_attempts: int = 3,
        backoff_seconds: float = 0.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if max_attempts < 1 or backoff_seconds < 0:
            raise ValueError("max_attempts must be positive and backoff nonnegative")
        self.max_attempts = max_attempts
        self.backoff_seconds = backoff_seconds
        self._sleep = sleep_fn
        self.queue: Queue[Job] = Queue()
        self._seen: set[str] = set()
        self.results: dict[str, T] = {}
        self.dead_letters: list[DeadLetter] = []

    def enqueue(self, url: str) -> bool:
        """Ignore duplicate URL submissions before they reach a worker."""
        normalized = url.strip()
        if not normalized:
            raise ValueError("URL cannot be empty")
        key = idempotency_key(normalized)
        if key in self._seen:
            return False
        self._seen.add(key)
        self.queue.put(Job(url=normalized, key=key, host=urlsplit(normalized).netloc.lower()))
        return True

    def run_worker(self, process: Callable[[str], T]) -> None:
        """Retry failed jobs with exponential delay, then put them in a dead letter list."""
        while True:
            try:
                job = self.queue.get_nowait()
            except Empty:
                break
            try:
                if job.key not in self.results:
                    result = process(job.url)
                    self.results[job.key] = result
            except Exception as exc:
                next_attempt = job.attempts + 1
                if next_attempt >= self.max_attempts:
                    self.dead_letters.append(DeadLetter(job=job, reason=str(exc)))
                else:
                    delay = self.backoff_seconds * (2 ** job.attempts)
                    if delay:
                        self._sleep(delay)
                    self.queue.put(
                        Job(url=job.url, key=job.key, host=job.host, attempts=next_attempt)
                    )
            finally:
                self.queue.task_done()
