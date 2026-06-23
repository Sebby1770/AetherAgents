"""Lightweight, optional tracing.

By default :func:`span` is a no-op context manager so the framework has zero
runtime cost and no hard dependency on OpenTelemetry. Call
:func:`configure_tracing` (after installing ``aetheragents[telemetry]``) to emit
real spans.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

_tracer: Any = None


def configure_tracing(service_name: str = "aetheragents") -> bool:
    """Enable OpenTelemetry tracing if the SDK is installed.

    Returns ``True`` if tracing was enabled, ``False`` otherwise.
    """
    global _tracer
    try:
        from opentelemetry import trace
    except ImportError:
        return False
    _tracer = trace.get_tracer(service_name)
    return True


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[None]:
    """Open a tracing span (no-op unless tracing is configured)."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as sp:  # pragma: no cover - otel optional
        for key, value in attributes.items():
            sp.set_attribute(key, value)
        yield
