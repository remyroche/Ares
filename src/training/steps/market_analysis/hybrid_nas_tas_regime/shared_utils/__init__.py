"""Deprecated hybrid shared utilities shim.

The shared NAS/TAS utilities now live in ``src.utils.nas_tas.shared_utils``.
This package re-exports everything from the new location for backwards compatibility.
"""

from src.utils.nas_tas.shared_utils import *  # noqa: F401,F403
