from __future__ import annotations
"""
HTTP error handler middleware.

Provides middleware for various HTTP frameworks to handle
AppError instances and convert them to appropriate responses.
"""

import json
import logging
from collections.abc import Callable
from typing import Any

from ..base import AppError
from ..mapping import error_mapper

logger = logging.getLogger(__name__)


def create_flask_error_handler():
    """
    Create error handler for Flask applications.

    Example:
        from flask import Flask
        from src.core.errors.handlers.http import create_flask_error_handler

        app = Flask(__name__)
        error_handler = create_flask_error_handler()
        app.register_error_handler(AppError, error_handler)
    """
    def handle_app_error(error: AppError):
        from flask import jsonify

        response_data = error.to_dict()
        return jsonify(response_data), error.status_code

    return handle_app_error


def create_fastapi_exception_handler():
    """
    Create exception handler for FastAPI applications.

    Example:
        from fastapi import FastAPI
        from src.core.errors.base import AppError
        from src.core.errors.handlers.http import create_fastapi_exception_handler

        app = FastAPI()
        exception_handler = create_fastapi_exception_handler()
        app.add_exception_handler(AppError, exception_handler)
    """
    async def handle_app_error(request, error: AppError):
        from fastapi.responses import JSONResponse

        response_data = error.to_dict()
        return JSONResponse(
            status_code=error.status_code,
            content=response_data,
        )

    return handle_app_error


def create_django_middleware():
    """
    Create middleware for Django applications.

    Example:
        # In Django settings.py:
        MIDDLEWARE = [
            # ... other middleware ...
            'src.core.errors.handlers.http.DjangoErrorMiddleware',
        ]
    """
    from django.http import JsonResponse
    from django.utils.deprecation import MiddlewareMixin

    class DjangoErrorMiddleware(MiddlewareMixin):
        def process_exception(self, request, exception):
            if isinstance(exception, AppError):
                response_data = exception.to_dict()
                return JsonResponse(
                    response_data,
                    status=exception.status_code,
                )

            # Map other exceptions
            app_error = error_mapper.map_exception(exception)
            response_data = app_error.to_dict()

            # Log the original exception
            logger.error(
                f"Unhandled exception mapped to {app_error.code}",
                exc_info=exception,
                extra={
                    "request_path": request.path,
                    "request_method": request.method,
                },
            )

            return JsonResponse(
                response_data,
                status=app_error.status_code,
            )

    return DjangoErrorMiddleware


def create_aiohttp_middleware():
    """
    Create middleware for aiohttp applications.

    Example:
        from aiohttp import web
        from src.core.errors.handlers.http import create_aiohttp_middleware

        app = web.Application(middlewares=[create_aiohttp_middleware()])
    """
    from aiohttp import web
    @web.middleware
    async def error_middleware(request, handler):
        try:
            return await handler(request)
        except AppError as error:
            response_data = error.to_dict()
            return web.json_response(
                response_data,
                status=error.status_code,
            )
        except Exception as exc:
            # Map other exceptions
            app_error = error_mapper.map_exception(exc)
            response_data = app_error.to_dict()

            # Log the original exception
            logger.exception(
                f"Unhandled exception mapped to {app_error.code}",
                exc_info=exc,
                extra={
                    "request_path": request.path,
                    "request_method": request.method,
                },
            )

            return web.json_response(
                response_data,
                status=app_error.status_code,
            )

    return error_middleware


def create_generic_wsgi_middleware(app: Callable):
    """
    Create generic WSGI middleware for error handling.

    Args:
        app: WSGI application

    Returns:
        Wrapped WSGI application
    """
    def middleware(environ: dict[str, Any], start_response: Callable):
        try:
            return app(environ, start_response)
        except AppError as error:
            response_data = error.to_dict()
            response_body = json.dumps(response_data).encode("utf-8")

            status = f"{error.status_code} {_get_status_text(error.status_code)}"
            headers = [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body))),
            ]

            start_response(status, headers)
            return [response_body]
        except Exception as exc:
            # Map other exceptions
            app_error = error_mapper.map_exception(exc)
            response_data = app_error.to_dict()
            response_body = json.dumps(response_data).encode("utf-8")

            logger.exception(
                f"Unhandled exception mapped to {app_error.code}",
                exc_info=exc,
            )

            status = f"{app_error.status_code} {_get_status_text(app_error.status_code)}"
            headers = [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(response_body))),
            ]

            start_response(status, headers)
            return [response_body]

    return middleware


def _get_status_text(status_code: int) -> str:
    """Get status text for HTTP status code."""
    status_texts = {
        400: "Bad Request",
        401: "Unauthorized",
        403: "Forbidden",
        404: "Not Found",
        409: "Conflict",
        422: "Unprocessable Entity",
        429: "Too Many Requests",
        500: "Internal Server Error",
        503: "Service Unavailable",
        504: "Gateway Timeout",
    }
    return status_texts.get(status_code, "Unknown")
