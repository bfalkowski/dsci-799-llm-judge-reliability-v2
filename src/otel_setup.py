"""OpenTelemetry setup for LLM-as-a-Judge reliability validation.

Provides trace context (trace_id, span_id) and structured metadata per judgment
for correlation and token-based reliability metrics.
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.resources import Resource

SERVICE_NAME = "llm-judge"
SERVICE_VERSION = "1.0"


class _NoOpSpanExporter(SpanExporter):
    """Exporter that drops spans (we embed trace_id/span_id in our JSONL output)."""

    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


def setup_tracer() -> trace.Tracer:
    """Initialize OTEL and return a tracer for the judge pipeline."""
    resource = Resource.create({
        "service.name": SERVICE_NAME,
        "service.version": SERVICE_VERSION,
        "deployment.environment": "experiment",
    })
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(SimpleSpanProcessor(_NoOpSpanExporter()))
    trace.set_tracer_provider(provider)
    return trace.get_tracer(SERVICE_NAME, SERVICE_VERSION)


def get_trace_context() -> tuple[str, str]:
    """Get trace_id and span_id from the current span context, formatted for JSONL."""
    ctx = trace.get_current_span().get_span_context()
    trace_id = format(ctx.trace_id, "032x")  # 32-char hex
    span_id = format(ctx.span_id, "016x")    # 16-char hex
    return trace_id, span_id
