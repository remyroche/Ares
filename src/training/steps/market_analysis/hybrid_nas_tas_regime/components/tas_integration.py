"""Stub TAS integration component for testing purposes."""

from typing import Any, Dict

class TASIntegrationComponent:
    """Lightweight stub for TAS integration used in tests."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def analyze(self, *args, **kwargs) -> Dict[str, Any]:
        """Return a minimal analysis payload."""
        return {
            "success": True,
            "analysis": {},
        }
