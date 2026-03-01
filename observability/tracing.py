"""
OpenTelemetry distributed tracing for PPM.

Traces every significant operation with spans:
  ppm.session.turn          - full turn (parent span)
  ppm.observer.observe      - embedding + intent extraction
  ppm.predictor.predict     - trajectory prediction
  ppm.staging.prefetch      - background prefetch per slot
  ppm.staging.inject        - knapsack injection planning
  ppm.store.vector_search   - Qdrant search
  ppm.feedback.evaluate     - hit/miss detection
  ppm.write.distill         - memory distillation
  ppm.write.store           - memory persistence

Usage:
  Spans are automatically exported to any OTel-compatible backend
  (Jaeger, Honeycomb, Datadog, etc.) via OTLP exporter.
  Set PPM_OTEL_ENDPOINT env var to enable (disabled by default).
"""

import os
import functools
import time
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Callable

_OTEL_AVAILABLE = False
_tracer = None

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.resources import Resource
    _OTEL_AVAILABLE = True
except ImportError:
    pass


def init_tracing(
    service_name: str = "ppm",
    endpoint: str | None = None,
) -> None:
    """
    Initialize OpenTelemetry tracing.
    No-op if opentelemetry-sdk is not installed or endpoint not set.
    """
    global _tracer

    otel_endpoint = endpoint or os.getenv("PPM_OTEL_ENDPOINT")
    if not _OTEL_AVAILABLE or not otel_endpoint:
        _tracer = _NoOpTracer()
        return

    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        resource = Resource.create({"service.name": service_name, "service.version": "0.1.0"})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("ppm")
    except Exception as e:
        _tracer = _NoOpTracer()


def get_tracer():
    global _tracer
    if _tracer is None:
        _tracer = _NoOpTracer()
    return _tracer


@contextmanager
def span(name: str, attributes: dict | None = None):
    """Sync context manager for a trace span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        if attributes and hasattr(s, "set_attribute"):
            for k, v in attributes.items():
                try:
                    s.set_attribute(k, v)
                except Exception:
                    pass
        yield s


@asynccontextmanager
async def async_span(name: str, attributes: dict | None = None):
    """Async context manager for a trace span."""
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as s:
        if attributes and hasattr(s, "set_attribute"):
            for k, v in (attributes or {}).items():
                try:
                    s.set_attribute(k, v)
                except Exception:
                    pass
        yield s


def traced(span_name: str):
    """Decorator for async functions — wraps in a trace span."""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            async with async_span(span_name):
                return await fn(*args, **kwargs)
        return wrapper
    return decorator


class _NoOpTracer:
    """Null tracer — used when OTel is not configured."""

    def start_as_current_span(self, name: str):
        return _NoOpSpan()


class _NoOpSpan:
    def __enter__(self): return self
    def __exit__(self, *_): pass
    def set_attribute(self, *_): pass
    def record_exception(self, *_): pass
    def set_status(self, *_): pass

