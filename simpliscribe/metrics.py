from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

START_TIME = time.time()
_request_counts: dict[str, int] = defaultdict(int)
_analysis_counts: dict[str, int] = defaultdict(int)
_cache_counts: dict[str, int] = defaultdict(int)


def record_http_request(method: str, status_code: int) -> None:
    key = f"{method.upper()}_{status_code}"
    _request_counts[key] += 1


def record_analysis_event(status: str) -> None:
    _analysis_counts[status] += 1


def record_cache_lookup(hit: bool) -> None:
    key = "hit" if hit else "miss"
    _cache_counts[key] += 1


def get_metrics_snapshot() -> dict[str, Any]:
    uptime = time.time() - START_TIME
    return {
        "uptime_seconds": round(uptime, 2),
        "http_requests": dict(_request_counts),
        "analyses": dict(_analysis_counts),
        "vector_cache": dict(_cache_counts),
    }


def generate_prometheus_metrics() -> str:
    uptime = time.time() - START_TIME
    lines: list[str] = [
        "# HELP simpliscribe_uptime_seconds Application uptime in seconds.",
        "# TYPE simpliscribe_uptime_seconds gauge",
        f"simpliscribe_uptime_seconds {uptime:.2f}",
        "",
        "# HELP simpliscribe_http_requests_total Total number of HTTP requests processed.",
        "# TYPE simpliscribe_http_requests_total counter",
    ]
    for key, count in sorted(_request_counts.items()):
        parts = key.split("_", 1)
        method = parts[0] if len(parts) > 1 else "UNKNOWN"
        status = parts[1] if len(parts) > 1 else "200"
        lines.append(f'simpliscribe_http_requests_total{{method="{method}",status="{status}"}} {count}')

    lines.extend([
        "",
        "# HELP simpliscribe_analyses_total Total prescription analyses performed.",
        "# TYPE simpliscribe_analyses_total counter",
    ])
    for status, count in sorted(_analysis_counts.items()):
        lines.append(f'simpliscribe_analyses_total{{status="{status}"}} {count}')

    lines.extend([
        "",
        "# HELP simpliscribe_vector_cache_lookups_total Total semantic vector cache queries.",
        "# TYPE simpliscribe_vector_cache_lookups_total counter",
    ])
    for result, count in sorted(_cache_counts.items()):
        lines.append(f'simpliscribe_vector_cache_lookups_total{{result="{result}"}} {count}')

    return "\n".join(lines) + "\n"
