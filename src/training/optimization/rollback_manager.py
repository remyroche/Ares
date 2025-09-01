import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors, handle_specific_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import error, initialization_error


@dataclass
class RollbackPoint:
    """Rollback point for parameter configuration."""

    timestamp: datetime
    description: str
    config_snapshot: Dict[str, Any]
    pipeline_state: Dict[str, Any]
    performance_metrics: Optional[Dict[str, Any]] = None
    optimization_results: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None


@dataclass
class RollbackOperation:
    """Rollback operation details."""

    timestamp: datetime
    from_point: str
    to_point: str
    parameters_changed: List[str]
    success: bool
    error_message: Optional[str] = None


class RollbackManager:
    """Manages rollback points and allows manual reversion to previous parameter configurations."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize rollback manager.

        Args:
            config: Configuration dictionary

        """
        self.config = config
        self.logger = system_logger.getChild("RollbackManager")

        # Rollback storage
        self.rollback_points: Dict[str, RollbackPoint] = {}
        self.rollback_history: List[RollbackOperation] = []

        # Storage configuration
        self.storage_config = {
            "rollback_directory": "data/optimization/rollbacks",
            "max_rollback_points": 50,
            "auto_cleanup_days": 30,
        }

        # Initialize storage
        self._initialize_storage()

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="storage initialization",
    )
    def _initialize_storage(self) -> None:
        """Initialize rollback storage directory."""
        try:
            rollback_dir = Path(self.storage_config["rollback_directory"])
            rollback_dir.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"📁 Rollback storage initialized at: {rollback_dir}")

        except Exception as e:
            self.logger.error(
                initialization_error(f"Error initializing rollback storage: {e}")
            )

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid rollback point data"),
            AttributeError: (False, "Missing rollback point parameters"),
            KeyError: (False, "Missing required rollback data"),
        },
        default_return=False,
        context="rollback point creation",
    )
    def create_rollback_point(
        self,
        description: str,
        pipeline_state: Dict[str, Any],
        performance_metrics: Optional[Dict[str, Any]] = None,
        optimization_results: Optional[Dict[str, Any]] = None,
        notes: Optional[str] = None,
    ) -> bool:
        """Create a rollback point with current configuration.

        Args:
            description: Description of the rollback point
            pipeline_state: Current pipeline state
            performance_metrics: Optional performance metrics
            optimization_results: Optional optimization results
            notes: Optional notes

        Returns:
            bool: True if rollback point created successfully, False otherwise

        """
        try:
            # Get current configuration from the main config
            current_config = self.config.copy()

            # Create rollback point
            rollback_point = RollbackPoint(
                timestamp=datetime.now(),
                description=description,
                config_snapshot=current_config,
                pipeline_state=pipeline_state,
                performance_metrics=performance_metrics,
                optimization_results=optimization_results,
                notes=notes,
            )

            # Generate unique ID
            point_id = f"rollback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Store rollback point
            self.rollback_points[point_id] = rollback_point

            # Save to file
            self._save_rollback_point(point_id, rollback_point)

            # Cleanup old points if needed
            self._cleanup_old_rollback_points()

            self.logger.info(f"✅ Rollback point created: {point_id} - {description}")
            return True

        except Exception as e:
            self.logger.error(error(f"❌ Error creating rollback point: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="rollback point saving",
    )
    def _save_rollback_point(
        self,
        point_id: str,
        rollback_point: RollbackPoint,
    ) -> None:
        """Save rollback point to file.

        Args:
            point_id: Unique identifier for the rollback point
            rollback_point: Rollback point to save

        """
        try:
            rollback_dir = Path(self.storage_config["rollback_directory"])
            point_file = rollback_dir / f"{point_id}.json"

            # Convert to dictionary
            point_data = asdict(rollback_point)
            point_data["point_id"] = point_id

            # Save to file
            with open(point_file, "w") as f:
                json.dump(point_data, f, indent=2, default=str)

            self.logger.info(f"💾 Rollback point saved to: {point_file}")

        except Exception as e:
            self.logger.error(error(f"Error saving rollback point: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="rollback point loading",
    )
    def load_rollback_points(self) -> None:
        """Load rollback points from storage."""
        try:
            rollback_dir = Path(self.storage_config["rollback_directory"])

            if not rollback_dir.exists():
                self.logger.info("No rollback directory found, starting fresh")
                return

            # Load all rollback point files
            for point_file in rollback_dir.glob("*.json"):
                try:
                    with open(point_file, "r") as f:
                        point_data = json.load(f)

                    # Extract point ID from filename
                    point_id = point_file.stem

                    # Convert back to RollbackPoint
                    rollback_point = RollbackPoint(
                        timestamp=datetime.fromisoformat(point_data["timestamp"]),
                        description=point_data["description"],
                        config_snapshot=point_data["config_snapshot"],
                        pipeline_state=point_data["pipeline_state"],
                        performance_metrics=point_data.get("performance_metrics"),
                        optimization_results=point_data.get("optimization_results"),
                        notes=point_data.get("notes"),
                    )

                    self.rollback_points[point_id] = rollback_point

                except Exception as e:
                    self.logger.warning(
                        f"Error loading rollback point {point_file}: {e}",
                    )

            self.logger.info(f"📂 Loaded {len(self.rollback_points)} rollback points")

        except Exception as e:
            self.logger.error(error(f"Error loading rollback points: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="old rollback points cleanup",
    )
    def _cleanup_old_rollback_points(self) -> None:
        """Cleanup old rollback points based on configuration."""
        try:
            max_points = self.storage_config["max_rollback_points"]
            auto_cleanup_days = self.storage_config["auto_cleanup_days"]

            if len(self.rollback_points) <= max_points:
                return

            # Sort points by timestamp
            sorted_points = sorted(
                self.rollback_points.items(),
                key=lambda x: x[1].timestamp,
                reverse=True,
            )

            # Keep only the most recent points
            points_to_remove = sorted_points[max_points:]

            # Remove old points
            for point_id, _ in points_to_remove:
                self._remove_rollback_point(point_id)

            self.logger.info(
                f"🧹 Cleaned up {len(points_to_remove)} old rollback points",
            )

        except Exception as e:
            self.logger.error(error(f"Error cleaning up old rollback points: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="rollback point removal",
    )
    def _remove_rollback_point(self, point_id: str) -> None:
        """Remove a rollback point.

        Args:
            point_id: ID of the rollback point to remove

        """
        try:
            # Remove from memory
            if point_id in self.rollback_points:
                del self.rollback_points[point_id]

            # Remove from file system
            rollback_dir = Path(self.storage_config["rollback_directory"])
            point_file = rollback_dir / f"{point_id}.json"

            if point_file.exists():
                point_file.unlink()

            self.logger.info(f"🗑️ Removed rollback point: {point_id}")

        except Exception as e:
            self.logger.error(error(f"Error removing rollback point: {e}"))

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="rollback point validation",
    )
    def validate_rollback_point(self, target_point_id: str) -> bool:
        """Validate if a rollback point can be used.

        Args:
            target_point_id: ID of the rollback point to validate

        Returns:
            bool: True if rollback point is valid, False otherwise

        """
        try:
            # Check if rollback point exists
            if target_point_id not in self.rollback_points:
                self.logger.warning(f"Rollback point not found: {target_point_id}")
                return False

            # Check if configuration is compatible
            rollback_point = self.rollback_points[target_point_id]

            # Basic validation - check if required fields exist
            if not rollback_point.config_snapshot:
                self.logger.warning("Invalid rollback point: missing config snapshot")
                return False

            if not rollback_point.pipeline_state:
                self.logger.warning("Invalid rollback point: missing pipeline state")
                return False

            self.logger.info(f"✅ Rollback point validated: {target_point_id}")
            return True

        except Exception as e:
            self.logger.error(error(f"Error validating rollback point: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="rollback execution",
    )
    def execute_rollback(
        self,
        target_point_id: str,
        current_pipeline_state: Dict[str, Any],
    ) -> bool:
        """Execute rollback to a specific point.

        Args:
            target_point_id: ID of the rollback point to restore
            current_pipeline_state: Current pipeline state for comparison

        Returns:
            bool: True if rollback successful, False otherwise

        """
        try:
            # Validate rollback point
            if not self.validate_rollback_point(target_point_id):
                return False

            rollback_point = self.rollback_points[target_point_id]

            # This is a simplified implementation
            # In production, you would need to carefully apply the configuration
            # and ensure system consistency

            # Update current configuration with rollback snapshot
            self.config.update(rollback_point.config_snapshot)

            # Record rollback operation
            rollback_operation = RollbackOperation(
                timestamp=datetime.now(),
                from_point="current",
                to_point=target_point_id,
                parameters_changed=list(rollback_point.config_snapshot.keys()),
                success=True,
            )

            self.rollback_history.append(rollback_operation)

            self.logger.info(f"🔄 Rollback executed to: {target_point_id}")
            return True

        except Exception as e:
            self.logger.error(error(f"Error executing rollback: {e}"))
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="rollback points listing",
    )
    def list_rollback_points(self) -> Optional[Dict[str, Any]]:
        """List all available rollback points.

        Returns:
            Dict containing rollback points information

        """
        try:
            points = {}
            for point_id, rollback_point in self.rollback_points.items():
                points[point_id] = {
                    "timestamp": rollback_point.timestamp.isoformat(),
                    "description": rollback_point.description,
                    "has_performance_metrics": rollback_point.performance_metrics
                    is not None,
                    "has_optimization_results": rollback_point.optimization_results
                    is not None,
                    "notes": rollback_point.notes,
                }

            return {
                "total_points": len(points),
                "points": points,
            }

        except Exception as e:
            self.logger.error(error(f"Error listing rollback points: {e}"))
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="rollback point retrieval",
    )
    def get_rollback_point(self, point_id: str) -> Optional[RollbackPoint]:
        """Get a specific rollback point.

        Args:
            point_id: ID of the rollback point to retrieve

        Returns:
            RollbackPoint if found, None otherwise

        """
        try:
            if point_id not in self.rollback_points:
                return None

            return self.rollback_points[point_id]

        except Exception as e:
            self.logger.error(error(f"Error retrieving rollback point: {e}"))
            return None

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="rollback statistics",
    )
    def get_rollback_statistics(self) -> Optional[Dict[str, Any]]:
        """Get rollback system statistics.

        Returns:
            Dict containing rollback statistics

        """
        try:
            return {
                "total_rollback_points": len(self.rollback_points),
                "total_rollback_operations": len(self.rollback_history),
                "storage_directory": self.storage_config["rollback_directory"],
                "max_rollback_points": self.storage_config["max_rollback_points"],
                "auto_cleanup_days": self.storage_config["auto_cleanup_days"],
            }

        except Exception as e:
            self.logger.error(error(f"Error getting rollback statistics: {e}"))
            return None


def create_rollback_manager(config: Optional[Dict[str, Any]] = None) -> RollbackManager:
    """Create a rollback manager instance.

    Args:
        config: Optional configuration dictionary

    Returns:
        RollbackManager instance

    """
    if config is None:
        config = {}

    return RollbackManager(config)
