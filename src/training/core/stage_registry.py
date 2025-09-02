"""Stage registry for the modular training pipeline.
This module provides stage registration and management functionality.
"""


from typing import Any, Dict, List, Optional
from datetime import datetime


class StageRegistry:
    """Simple stage registry for pipeline stages."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the stage registry."""
        self.config = config
        self.registered_stages: Dict[str, Any] = {}
        self.stage_history: List[Dict[str, Any]] = []
        self.is_initialized = False

    async def initialize(self) -> bool:
        """Initialize the stage registry."""
        try:
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Error initializing stage registry: {e}")
            return False


    def register_stage(self, stage_name: str, stage_class: Any) -> bool:
        """Register a new stage."""
        try:
            self.registered_stages[stage_name] = stage_class
            self.stage_history.append({
                "action": "register",
                "stage_name": stage_name,
                "timestamp": datetime.now().isoformat()
            })
            return True
        except Exception as e:
            print(f"Error registering stage {stage_name}: {e}")
            return False

    def get_stage(self, stage_name: str) -> Optional[Any]:
        """Get a registered stage."""
        return self.registered_stages.get(stage_name)

    def list_stages(self) -> List[str]:
        """List all registered stages."""
        return list(self.registered_stages.keys())

    def unregister_stage(self, stage_name: str) -> bool:
        """Unregister a stage."""
        try:
            if stage_name in self.registered_stages:
                del self.registered_stages[stage_name]
                self.stage_history.append({
                    "action": "unregister",
                    "stage_name": stage_name,
                    "timestamp": datetime.now().isoformat()
                })
                return True
            return False
        except Exception as e:
            print(f"Error unregistering stage {stage_name}: {e}")
            return False

    def get_registry_status(self) -> Dict[str, Any]:
        """Get registry status."""
        return {
            "is_initialized": self.is_initialized,
            "total_stages": len(self.registered_stages),
            "registered_stages": list(self.registered_stages.keys()),
            "history_count": len(self.stage_history)
        }

