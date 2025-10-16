#!/usr/bin/env python3
"""Convenience wrapper for src.launcher.ares_launcher CLI."""

import asyncio

from src.launcher.ares_launcher import main as _ares_main


def main() -> None:
    """Dispatch to the async launcher entrypoint."""
    asyncio.run(_ares_main())


if __name__ == "__main__":
    main()
