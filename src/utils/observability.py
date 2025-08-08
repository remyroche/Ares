from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def init_sentry() -> None:
    """Initialize Sentry if SENTRY_DSN is provided.

    Set SENTRY_ENV and SENTRY_TRACES_SAMPLE_RATE as needed.
    """
    dsn = os.getenv("SENTRY_DSN")
    if not dsn:
        return

    try:
        import sentry_sdk
        from sentry_sdk.integrations.logging import LoggingIntegration
        from sentry_sdk.integrations.aiohttp import AioHttpIntegration
        from sentry_sdk.integrations.fastapi import FastApiIntegration

        sentry_logging = LoggingIntegration(
            level=logging.INFO,
            event_level=logging.ERROR,
        )
        sentry_sdk.init(
            dsn=dsn,
            environment=os.getenv("SENTRY_ENV", "production"),
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.0")),
            profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.0")),
            integrations=[sentry_logging, AioHttpIntegration(), FastApiIntegration()],
            send_default_pii=False,
        )
        logger.info("Sentry initialized")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to initialize Sentry: {exc}")


def init_otlp_logging() -> None:
    """Initialize OpenTelemetry logging exporter if OTLP endpoint is provided."""
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return

    try:
        # Minimal setup for OTLP logging exporter
        from opentelemetry import _logs as otel_logs
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "ares-bot")})
        provider = LoggerProvider(resource=resource)
        exporter = OTLPLogExporter()
        provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
        otel_logs.set_logger_provider(provider)
        logger.info("OpenTelemetry logging exporter initialized")
    except Exception as exc:  # pragma: no cover
        logger.warning(f"Failed to initialize OTLP logging: {exc}")


def init_observability(_: dict[str, Any] | None = None) -> None:
    """Initialize production observability hooks: Sentry and OTLP if configured."""
    init_sentry()
    init_otlp_logging()