"""Shared operational state: per-IP rate limiting and the global daily budget.

Two backends behind one interface:

- InMemoryState (default): same semantics as the original in-app dicts —
  correct for a SINGLE worker process only.
- RedisState (set REDIS_URL): shared, atomic state so multiple stateless
  replicas enforce one truth. Uses fixed-window counters (INCR + EXPIRE),
  which are cheap and race-free; the window boundary approximation is
  acceptable for abuse control.

The backend is chosen once per process by create_state(); limits are passed
per call so callers (and tests) control them.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from src.config import REDIS_URL

logger = logging.getLogger(__name__)


class InMemoryState:
    """Single-process backend (the pre-Redis behavior, extracted)."""

    _MAX_IPS = 10_000  # evict oldest IPs beyond this

    def __init__(self) -> None:
        self._rate: dict[str, list[float]] = defaultdict(list)
        self._daily_count = 0
        self._daily_date = ""

    def rate_limit_allow(self, ip: str, max_requests: int, window_seconds: float) -> bool:
        now = time.time()
        window_start = now - window_seconds
        self._rate[ip] = [t for t in self._rate[ip] if t > window_start]
        if len(self._rate[ip]) >= max_requests:
            return False
        self._rate[ip].append(now)
        if len(self._rate) > self._MAX_IPS:
            oldest = sorted(self._rate, key=lambda k: max(self._rate[k], default=0))
            for k in oldest[: self._MAX_IPS // 10]:
                del self._rate[k]
        return True

    def daily_take(self, cap: int) -> bool:
        """Consume one slot of the global daily budget; False when exhausted."""
        if cap <= 0:
            return True
        today = time.strftime("%Y-%m-%d")
        if today != self._daily_date:
            self._daily_date = today
            self._daily_count = 0
        if self._daily_count >= cap:
            return False
        self._daily_count += 1
        return True

    def daily_refund(self) -> None:
        self._daily_count = max(0, self._daily_count - 1)


class RedisState:
    """Shared backend for multi-replica deployments."""

    def __init__(self, client) -> None:
        self._r = client

    def rate_limit_allow(self, ip: str, max_requests: int, window_seconds: float) -> bool:
        window = int(time.time() // max(window_seconds, 1))
        key = f"gg:rl:{ip}:{window}"
        count = self._r.incr(key)
        if count == 1:
            self._r.expire(key, int(window_seconds) + 1)
        return count <= max_requests

    def daily_take(self, cap: int) -> bool:
        if cap <= 0:
            return True
        key = f"gg:budget:{time.strftime('%Y-%m-%d')}"
        count = self._r.incr(key)
        if count == 1:
            self._r.expire(key, 2 * 24 * 3600)
        if count > cap:
            # Leave the counter — refunding an over-cap take would let
            # hammering clients consume real budget at the boundary.
            return False
        return True

    def daily_refund(self) -> None:
        key = f"gg:budget:{time.strftime('%Y-%m-%d')}"
        try:
            if int(self._r.decr(key)) < 0:
                self._r.incr(key)
        except Exception:  # noqa: BLE001 — refund is best-effort
            pass


def create_state():
    """RedisState when REDIS_URL is configured and reachable, else in-memory."""
    if REDIS_URL:
        try:
            import redis
            client = redis.Redis.from_url(REDIS_URL, socket_timeout=2)
            client.ping()
            logger.info("State backend: Redis (%s)", REDIS_URL.split("@")[-1])
            return RedisState(client)
        except Exception as exc:  # noqa: BLE001 — degrade loudly but keep serving
            logger.error(
                "REDIS_URL set but unusable (%s) — falling back to in-memory "
                "state. Rate limits and budget are PER-PROCESS until fixed.", exc,
            )
    return InMemoryState()
