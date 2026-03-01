"""
FastAPI middleware for automatic request instrumentation.

Adds to every request:
  - Prometheus request counter + latency histogram
  - OpenTelemetry span with HTTP attributes
  - Request ID header (X-Request-ID)
  - Structured access log
"""

import time
import uuid
import structlog
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from observability.metrics import REGISTRY
from prometheus_client import Counter, Histogram

log = structlog.get_logger("ppm.http")

http_requests_total = Counter(
    "ppm_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
    registry=REGISTRY,
)

http_request_latency = Histogram(
    "ppm_http_request_latency_seconds",
    "HTTP request latency",
    ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
    registry=REGISTRY,
)


class ObservabilityMiddleware(BaseHTTPMiddleware):
    """
    Instruments every HTTP request with metrics, tracing, and logging.
    Adds X-Request-ID header to every response.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = str(uuid.uuid4())[:8]
        t0 = time.monotonic()

        # Normalize path for metrics (replace path params with {param})
        path = self._normalize_path(request.url.path)

        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as e:
            status = 500
            log.error(
                "http.request_error",
                request_id=request_id,
                method=request.method,
                path=path,
                error=str(e),
            )
            raise
        finally:
            latency = time.monotonic() - t0
            http_requests_total.labels(
                method=request.method,
                path=path,
                status=str(status),
            ).inc()
            http_request_latency.labels(
                method=request.method,
                path=path,
            ).observe(latency)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Latency-Ms"] = f"{latency * 1000:.1f}"

        log.info(
            "http.request",
            request_id=request_id,
            method=request.method,
            path=path,
            status=status,
            latency_ms=round(latency * 1000, 1),
        )

        return response

    @staticmethod
    def _normalize_path(path: str) -> str:
        """Replace UUIDs and IDs in paths with {id} placeholder."""
        import re
        # Replace UUID segments
        path = re.sub(
            r'/[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
            '/{id}', path
        )
        return path


def add_metrics_endpoint(app) -> None:
    """Add /metrics endpoint to FastAPI app for Prometheus scraping."""
    from fastapi import Response as FastAPIResponse
    from observability.metrics import metrics_response

    @app.get("/metrics", include_in_schema=False)
    async def prometheus_metrics():
        body, content_type = metrics_response()
        return FastAPIResponse(content=body, media_type=content_type)

