"""Logging configuration utilities."""

import logging

def get_logger(name: str = None):
    """Get a logger instance."""
    return logging.getLogger(name or __name__)
