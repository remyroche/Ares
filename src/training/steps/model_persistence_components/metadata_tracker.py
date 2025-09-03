"""Metadata tracker component for model persistence."""

import asyncio
import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.decorators import handles_errors, log_execution_time
from src.utils.logger import system_logger


class MetadataTracker:
    """Handles comprehensive metadata tracking for models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the metadata tracker.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config.get("metadata_tracking", {})
        self.logger = system_logger.getChild("metadata_tracker")
        
        # Tracking configuration
        self.track_lineage = self.config.get("track_lineage", True)
        self.track_data_stats = self.config.get("track_data_stats", True)
        self.track_performance = self.config.get("track_performance", True)
        self.track_environment = self.config.get("track_environment", True)
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="metadata creation"
    )
    async def create_metadata(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any],
        saved_artifacts: Dict[str, Any],
        version_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive metadata for a training run.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Complete pipeline state
            saved_artifacts: Saved artifact paths
            version_info: Version information
            
        Returns:
            Comprehensive metadata dictionary
        """
        self.logger.info("Creating comprehensive metadata...")
        
        metadata = {
            "version": version_info,
            "training_input": self._sanitize_training_input(training_input),
            "artifacts": saved_artifacts,
            "created_at": datetime.now().isoformat(),
            "metadata_version": "1.0"
        }
        
        # Add lineage information
        if self.track_lineage:
            metadata["lineage"] = await self._track_lineage(pipeline_state)
        
        # Add data statistics
        if self.track_data_stats:
            metadata["data_statistics"] = await self._track_data_statistics(pipeline_state)
        
        # Add performance metrics
        if self.track_performance:
            metadata["performance_metrics"] = await self._track_performance_metrics(pipeline_state)
        
        # Add environment information
        if self.track_environment:
            metadata["environment"] = await self._track_environment()
        
        # Add pipeline execution summary
        metadata["pipeline_execution"] = await self._track_pipeline_execution(pipeline_state)
        
        # Calculate metadata checksum
        metadata["checksum"] = self._calculate_checksum(metadata)
        
        return metadata
    
    def _sanitize_training_input(self, training_input: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize training input for storage."""
        sanitized = {}
        
        # Only include serializable values
        for key, value in training_input.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                sanitized[key] = value
            else:
                sanitized[key] = str(value)
        
        return sanitized
    
    async def _track_lineage(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Track model lineage through the pipeline."""
        lineage = {
            "pipeline_steps": [],
            "data_sources": [],
            "transformations": [],
            "model_dependencies": {}
        }
        
        # Track completed steps
        for key in sorted(pipeline_state.keys()):
            if key.endswith("_completed") and pipeline_state[key]:
                step_name = key.replace("_completed", "")
                lineage["pipeline_steps"].append({
                    "step": step_name,
                    "completed": True,
                    "timestamp": pipeline_state.get(f"{step_name}_timestamp", "unknown")
                })
        
        # Track data sources
        if "data_sources" in pipeline_state:
            lineage["data_sources"] = pipeline_state["data_sources"]
        elif "market_data" in pipeline_state:
            data = pipeline_state["market_data"]
            if hasattr(data, 'index'):
                lineage["data_sources"].append({
                    "type": "market_data",
                    "start_date": str(data.index.min()),
                    "end_date": str(data.index.max()),
                    "rows": len(data)
                })
        
        # Track transformations
        transformation_keys = [
            "feature_engineering", "labeling", "regime_discovery",
            "enhancement", "calibration"
        ]
        
        for key in transformation_keys:
            for state_key in pipeline_state:
                if key in state_key and "results" in state_key:
                    lineage["transformations"].append({
                        "type": key,
                        "step": state_key
                    })
        
        # Track model dependencies
        if "analyst_ensembles" in pipeline_state:
            lineage["model_dependencies"]["analyst_models"] = True
        if "tactician_specialist_models" in pipeline_state:
            lineage["model_dependencies"]["tactician_models"] = True
        
        return lineage
    
    async def _track_data_statistics(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Track data statistics from the pipeline."""
        stats = {
            "dataset_info": {},
            "feature_statistics": {},
            "label_distribution": {},
            "regime_statistics": {}
        }
        
        # Dataset information
        if "tactician_labeled_data" in pipeline_state:
            data = pipeline_state["tactician_labeled_data"]
            stats["dataset_info"] = {
                "total_samples": len(data),
                "features": len(data.columns),
                "memory_usage": str(data.memory_usage(deep=True).sum() / 1024 / 1024) + " MB"
            }
            
            # Label distribution
            if "label" in data.columns:
                label_counts = data["label"].value_counts()
                stats["label_distribution"] = {
                    str(label): count 
                    for label, count in label_counts.items()
                }
        
        # Feature statistics
        if "features" in pipeline_state:
            features = pipeline_state["features"]
            if hasattr(features, 'describe'):
                stats["feature_statistics"]["summary"] = features.describe().to_dict()
        
        # Regime statistics
        if "regime_data" in pipeline_state:
            regime_data = pipeline_state["regime_data"]
            stats["regime_statistics"] = {
                "n_regimes": len(regime_data),
                "regime_names": list(regime_data.keys())
            }
            
            for regime_id, regime_info in regime_data.items():
                if "mask" in regime_info:
                    mask = regime_info["mask"]
                    if hasattr(mask, '__len__'):
                        stats["regime_statistics"][f"{regime_id}_samples"] = sum(mask)
        
        return stats
    
    async def _track_performance_metrics(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Track performance metrics from the pipeline."""
        metrics = {
            "training_metrics": {},
            "validation_metrics": {},
            "test_metrics": {},
            "best_models": {}
        }
        
        # Extract specialist evaluation results
        if "specialist_evaluation_results" in pipeline_state:
            eval_results = pipeline_state["specialist_evaluation_results"]
            
            if "overall_metrics" in eval_results:
                metrics["training_metrics"] = eval_results["overall_metrics"]
            
            if "regime_evaluations" in eval_results:
                # Find best model per regime
                for regime_id, regime_eval in eval_results["regime_evaluations"].items():
                    if "best_tactic" in regime_eval:
                        metrics["best_models"][regime_id] = regime_eval["best_tactic"]
        
        # Extract validation results
        validation_keys = [
            "step16_confidence_calibration_results",
            "step18_walk_forward_validation_results",
            "step19_monte_carlo_validation_results"
        ]
        
        for key in validation_keys:
            if key in pipeline_state:
                validation_type = key.split("_")[1]
                if "overall_metrics" in pipeline_state[key]:
                    metrics["validation_metrics"][validation_type] = pipeline_state[key]["overall_metrics"]
        
        return metrics
    
    async def _track_environment(self) -> Dict[str, Any]:
        """Track environment information."""
        import platform
        import sys
        
        environment = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }
        
        # Track package versions
        try:
            import pkg_resources
            
            key_packages = [
                "numpy", "pandas", "scikit-learn",
                "lightgbm", "xgboost", "torch"
            ]
            
            environment["packages"] = {}
            for package in key_packages:
                try:
                    version = pkg_resources.get_distribution(package).version
                    environment["packages"][package] = version
                except:
                    pass
        except:
            pass
        
        return environment
    
    async def _track_pipeline_execution(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Track pipeline execution details."""
        execution = {
            "completed_steps": [],
            "failed_steps": [],
            "execution_times": {},
            "total_duration": None
        }
        
        # Track completed and failed steps
        for key in pipeline_state:
            if key.endswith("_completed"):
                step_name = key.replace("_completed", "")
                if pipeline_state[key]:
                    execution["completed_steps"].append(step_name)
                else:
                    execution["failed_steps"].append(step_name)
            
            elif key.endswith("_duration"):
                step_name = key.replace("_duration", "")
                execution["execution_times"][step_name] = pipeline_state[key]
        
        # Calculate total duration
        if execution["execution_times"]:
            execution["total_duration"] = sum(execution["execution_times"].values())
        
        return execution
    
    def _calculate_checksum(self, metadata: Dict[str, Any]) -> str:
        """Calculate checksum for metadata."""
        # Convert to JSON string for consistent hashing
        metadata_str = json.dumps(metadata, sort_keys=True, default=str)
        
        # Calculate SHA256 hash
        return hashlib.sha256(metadata_str.encode()).hexdigest()
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={},
        context="metadata validation"
    )
    async def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate metadata integrity.
        
        Args:
            metadata: Metadata to validate
            
        Returns:
            Validation results
        """
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required fields
        required_fields = ["version", "created_at", "artifacts"]
        for field in required_fields:
            if field not in metadata:
                validation["errors"].append(f"Missing required field: {field}")
                validation["is_valid"] = False
        
        # Validate checksum if present
        if "checksum" in metadata:
            # Recalculate checksum
            metadata_copy = metadata.copy()
            stored_checksum = metadata_copy.pop("checksum")
            calculated_checksum = self._calculate_checksum(metadata_copy)
            
            if stored_checksum != calculated_checksum:
                validation["errors"].append("Checksum mismatch - metadata may be corrupted")
                validation["is_valid"] = False
        
        # Check for empty sections
        for section in ["lineage", "data_statistics", "performance_metrics"]:
            if section in metadata and not metadata[section]:
                validation["warnings"].append(f"Empty section: {section}")
        
        return validation