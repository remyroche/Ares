# src/optimization/rollback_manager.py

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config_optuna import get_optuna_config, update_parameter_value
from src.utils.error_handler import (
    handle_errors,
    handle_specific_errors,
)
from src.utils.logger import system_logger


@dataclass
class RollbackPoint:
    """Rollback point for parameter configuration."""

    timestamp: datetime
    description: str
    config_snapshot: dict[str, Any]
    pipeline_state: dict[str, Any]
    performance_metrics: dict[str, Any] | None = None
    optimization_results: dict[str, Any] | None = None
    notes: str | None = None


@dataclass
class RollbackOperation:
    """Rollback operation details."""

    timestamp: datetime
    from_point: str
    to_point: str
    parameters_changed: list[str]
    success: bool
    error_message: str | None = None


class RollbackManager:
    """
    Manages rollback points and allows manual reversion to previous parameter configurations.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        """
        Initialize rollback manager.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = system_logger.getChild("RollbackManager")

        # Rollback storage
        self.rollback_points: dict[str, RollbackPoint] = {}
        self.rollback_history: list[RollbackOperation] = []

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
            self.logger.error(f"Error initializing rollback storage: {e}")

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
        pipeline_state: dict[str, Any],
        performance_metrics: dict[str, Any] | None = None,
        optimization_results: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> bool:
        """
        Create a rollback point with current configuration.

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
            # Get current configuration
            current_config = get_optuna_config()

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
            self.logger.error(f"❌ Error creating rollback point: {e}")
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
        """
        Save rollback point to file.

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
            self.logger.error(f"Error saving rollback point: {e}")

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
                    with open(point_file) as f:
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
            self.logger.error(f"Error loading rollback points: {e}")

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
            points_to_keep = sorted_points[:max_points]
            points_to_remove = sorted_points[max_points:]

            # Remove old points
            for point_id, _ in points_to_remove:
                self._remove_rollback_point(point_id)

            self.logger.info(
                f"🧹 Cleaned up {len(points_to_remove)} old rollback points",
            )

        except Exception as e:
            self.logger.error(f"Error cleaning up old rollback points: {e}")

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=None,
        context="rollback point removal",
    )
    def _remove_rollback_point(self, point_id: str) -> None:
        """
        Remove a rollback point.

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

        except Exception as e:
            self.logger.error(f"Error removing rollback point {point_id}: {e}")

    @handle_specific_errors(
        error_handlers={
            ValueError: (False, "Invalid rollback operation"),
            AttributeError: (False, "Missing rollback parameters"),
            KeyError: (False, "Rollback point not found"),
        },
        default_return=False,
        context="rollback execution",
    )
    def execute_rollback(self, target_point_id: str) -> bool:
        """
        Execute rollback to a specific point.

        Args:
            target_point_id: ID of the rollback point to revert to

        Returns:
            bool: True if rollback successful, False otherwise
        """
        try:
            # Check if rollback point exists
            if target_point_id not in self.rollback_points:
                self.logger.error(f"Rollback point {target_point_id} not found")
                return False

            # Get current point ID for rollback operation
            current_point_id = self._get_current_point_id()

            # Get target rollback point
            target_point = self.rollback_points[target_point_id]

            self.logger.info(f"🔄 Executing rollback to: {target_point_id}")
            self.logger.info(f"   Description: {target_point.description}")
            self.logger.info(f"   Timestamp: {target_point.timestamp}")

            # Apply rollback configuration
            success = self._apply_rollback_configuration(target_point.config_snapshot)

            # Record rollback operation
            rollback_operation = RollbackOperation(
                timestamp=datetime.now(),
                from_point=current_point_id,
                to_point=target_point_id,
                parameters_changed=self._get_changed_parameters(
                    target_point.config_snapshot,
                ),
                success=success,
                error_message=None
                if success
                else "Failed to apply rollback configuration",
            )

            self.rollback_history.append(rollback_operation)

            if success:
                self.logger.info(
                    f"✅ Rollback to {target_point_id} completed successfully",
                )
            else:
                self.logger.error(f"❌ Rollback to {target_point_id} failed")

            return success

        except Exception as e:
            self.logger.error(f"❌ Error executing rollback: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="rollback configuration application",
    )
    def _apply_rollback_configuration(self, config_snapshot: dict[str, Any]) -> bool:
        """
        Apply rollback configuration to current system.

        Args:
            config_snapshot: Configuration snapshot to apply

        Returns:
            bool: True if configuration applied successfully, False otherwise
        """
        try:
            # This is a simplified implementation
            # In production, you would need to carefully apply the configuration
            # and ensure system consistency

            applied_params = []
            failed_params = []

            # Apply configuration parameters
            for section_name, section_config in config_snapshot.items():
                if hasattr(section_config, "__dataclass_fields__"):
                    for field_name, field_value in section_config.__dict__.items():
                        param_path = f"{section_name}.{field_name}"
                        if update_parameter_value(param_path, field_value):
                            applied_params.append(param_path)
                        else:
                            failed_params.append(param_path)

            # Log results
            if applied_params:
                self.logger.info(f"✅ Applied {len(applied_params)} parameters")
            if failed_params:
                self.logger.warning(
                    f"⚠️ Failed to apply {len(failed_params)} parameters",
                )

            return len(failed_params) == 0

        except Exception as e:
            self.logger.error(f"Error applying rollback configuration: {e}")
            return False

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=[],
        context="changed parameters detection",
    )
    def _get_changed_parameters(self, target_config: dict[str, Any]) -> list[str]:
        """
        Get list of parameters that would be changed by rollback.

        Args:
            target_config: Target configuration

        Returns:
            List[str]: List of parameter paths that would change
        """
        try:
            current_config = get_optuna_config()
            changed_params = []

            # Compare configurations
            for section_name, section_config in target_config.items():
                if section_name in current_config:
                    current_section = current_config[section_name]
                    if hasattr(current_section, "__dataclass_fields__"):
                        for field_name, field_value in section_config.__dict__.items():
                            param_path = f"{section_name}.{field_name}"
                            current_value = getattr(current_section, field_name, None)
                            if current_value != field_value:
                                changed_params.append(param_path)

            return changed_params

        except Exception as e:
            self.logger.error(f"Error detecting changed parameters: {e}")
            return []

    def _get_current_point_id(self) -> str:
        """
        Get current point ID based on current configuration.

        Returns:
            str: Current point ID
        """
        return f"current_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def get_rollback_points(self) -> dict[str, dict[str, Any]]:
        """
        Get all rollback points.

        Returns:
            Dict[str, dict[str, Any]]: Rollback points with metadata
        """
        try:
            points = {}
            for point_id, rollback_point in self.rollback_points.items():
                points[point_id] = {
                    "timestamp": rollback_point.timestamp.isoformat(),
                    "description": rollback_point.description,
                    "notes": rollback_point.notes,
                    "has_performance_metrics": rollback_point.performance_metrics
                    is not None,
                    "has_optimization_results": rollback_point.optimization_results
                    is not None,
                }

            return points

        except Exception as e:
            self.logger.error(f"Error getting rollback points: {e}")
            return {}

    def get_rollback_point_details(self, point_id: str) -> dict[str, Any] | None:
        """
        Get detailed information about a specific rollback point.

        Args:
            point_id: ID of the rollback point

        Returns:
            Optional[dict[str, Any]]: Detailed rollback point information
        """
        try:
            if point_id not in self.rollback_points:
                return None

            rollback_point = self.rollback_points[point_id]

            return {
                "point_id": point_id,
                "timestamp": rollback_point.timestamp.isoformat(),
                "description": rollback_point.description,
                "notes": rollback_point.notes,
                "config_snapshot": rollback_point.config_snapshot,
                "pipeline_state": rollback_point.pipeline_state,
                "performance_metrics": rollback_point.performance_metrics,
                "optimization_results": rollback_point.optimization_results,
            }

        except Exception as e:
            self.logger.error(f"Error getting rollback point details: {e}")
            return None

    def get_rollback_history(self) -> list[dict[str, Any]]:
        """
        Get rollback operation history.

        Returns:
            List[dict[str, Any]]: Rollback operation history
        """
        try:
            return [asdict(operation) for operation in self.rollback_history]

        except Exception as e:
            self.logger.error(f"Error getting rollback history: {e}")
            return []

    def delete_rollback_point(self, point_id: str) -> bool:
        """
        Delete a rollback point.

        Args:
            point_id: ID of the rollback point to delete

        Returns:
            bool: True if deletion successful, False otherwise
        """
        try:
            if point_id not in self.rollback_points:
                self.logger.error(f"Rollback point {point_id} not found")
                return False

            self._remove_rollback_point(point_id)
            self.logger.info(f"✅ Rollback point {point_id} deleted successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting rollback point {point_id}: {e}")
            return False

    def get_rollback_summary(self) -> dict[str, Any]:
        """
        Get rollback manager summary.

        Returns:
            dict[str, Any]: Rollback manager summary
        """
        try:
            return {
                "total_rollback_points": len(self.rollback_points),
                "total_rollback_operations": len(self.rollback_history),
                "storage_directory": self.storage_config["rollback_directory"],
                "max_rollback_points": self.storage_config["max_rollback_points"],
                "auto_cleanup_days": self.storage_config["auto_cleanup_days"],
                "recent_operations": len(
                    self.rollback_history[-10:],
                ),  # Last 10 operations
            }

        except Exception as e:
            self.logger.error(f"Error getting rollback summary: {e}")
            return {"error": str(e)}


@handle_errors(
    exceptions=(Exception,),
    default_return=None,
    context="rollback manager setup",
)
def setup_rollback_manager(
    config: dict[str, Any] | None = None,
) -> RollbackManager | None:
    """
    Setup rollback manager.

    Args:
        config: Configuration dictionary

    Returns:
        RollbackManager | None: Rollback manager instance or None
    """
    try:
        if config is None:
            config = {}

        manager = RollbackManager(config)

        # Load existing rollback points
        manager.load_rollback_points()

        return manager

    except Exception as e:
        system_logger.error(f"Error setting up rollback manager: {e}")
        return None
