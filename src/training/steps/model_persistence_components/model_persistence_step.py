"""Step 21: Model Persistence - Migrated to use BaseStep pattern.

This step handles comprehensive model saving/loading with versioning.
"""

import asyncio
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.decorators import handles_errors, log_execution_time, validates
from src.training.base_step import BaseStep
from src.utils.logger import system_logger

# Import component modules
from .model_serializer import ModelSerializer
from .version_manager import VersionManager
from .metadata_tracker import MetadataTracker
from .model_registry import ModelRegistry


class ModelPersistenceStep(BaseStep):
    """Step 21: Model Persistence with comprehensive saving and versioning."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Model Persistence step.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config, "21", "model_persistence")
        
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        # Initialize components
        self.model_serializer = ModelSerializer(self.config)
        self.version_manager = VersionManager(self.config)
        self.metadata_tracker = MetadataTracker(self.config)
        self.model_registry = ModelRegistry(self.config)
        
        # Initialize persistence configuration
        self.persistence_config = self._initialize_persistence_config()
        
        # Storage for saved artifacts
        self.saved_artifacts: Dict[str, str] = {}
        self.artifact_metadata: Dict[str, Any] = {}
        
    def _initialize_persistence_config(self) -> Dict[str, Any]:
        """Initialize persistence configuration."""
        return {
            # Storage configuration
            "base_dir": self.config.get("model_storage_dir", "models"),
            "enable_versioning": self.config.get("enable_versioning", True),
            "compression": self.config.get("model_compression", True),
            "save_formats": self.config.get("save_formats", ["pickle", "joblib", "onnx"]),
            
            # Metadata configuration
            "track_lineage": self.config.get("track_model_lineage", True),
            "save_training_data_stats": self.config.get("save_training_data_stats", True),
            "save_feature_importance": self.config.get("save_feature_importance", True),
            
            # Registry configuration
            "use_model_registry": self.config.get("use_model_registry", True),
            "registry_backend": self.config.get("registry_backend", "local"),
            
            # MLflow integration
            "enable_mlflow": self.config.get("enable_mlflow", False),
            "mlflow_tracking_uri": self.config.get("mlflow_tracking_uri", None),
            
            # Backup configuration
            "create_backups": self.config.get("create_backups", True),
            "backup_location": self.config.get("backup_location", "model_backups"),
            "max_backups": self.config.get("max_backups", 5)
        }
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate step inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for models to save
        model_keys = [
            "tactician_specialist_models",
            "analyst_ensembles",
            "calibrated_models",
            "enhanced_analyst_models"
        ]
        
        has_models = any(key in pipeline_state for key in model_keys)
        
        if not has_models:
            errors.append("No models found in pipeline state to persist")
        
        # Check for required metadata
        if "symbol" not in training_input:
            errors.append("Missing 'symbol' in training input")
        
        if "exchange" not in training_input:
            errors.append("Missing 'exchange' in training input")
        
        return len(errors) == 0, errors
    
    @handles_errors(
        exceptions=(Exception,),
        default_return={"success": False},
        context="model persistence logic"
    )
    async def execute_logic(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the model persistence logic.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with persistence results
        """
        self.logger.info("💾 Starting model persistence...")
        
        # Extract metadata
        symbol = training_input["symbol"]
        exchange = training_input["exchange"]
        timestamp = datetime.now()
        
        # Create version for this training run
        version_info = await self.version_manager.create_version(
            symbol, exchange, timestamp
        )
        
        self.logger.info(f"Created version: {version_info['version']}")
        
        # Collect all models to save
        models_to_save = self._collect_models(pipeline_state)
        
        # Save each model category
        for category, models in models_to_save.items():
            self.logger.info(f"Saving {category}...")
            
            saved_paths = await self._save_model_category(
                category,
                models,
                version_info,
                pipeline_state
            )
            
            self.saved_artifacts[category] = saved_paths
        
        # Create comprehensive metadata
        metadata = await self.metadata_tracker.create_metadata(
            training_input,
            pipeline_state,
            self.saved_artifacts,
            version_info
        )
        
        # Save metadata
        metadata_path = await self._save_metadata(metadata, version_info)
        self.saved_artifacts["metadata"] = metadata_path
        
        # Register models if configured
        if self.persistence_config["use_model_registry"]:
            registry_entries = await self.model_registry.register_models(
                self.saved_artifacts,
                metadata,
                version_info
            )
            self.logger.info(f"Registered {len(registry_entries)} models in registry")
        
        # Create training summary
        training_summary = await self._create_training_summary(
            pipeline_state,
            self.saved_artifacts,
            metadata
        )
        
        # Save training report
        report_path = await self._save_training_report(
            training_summary,
            version_info
        )
        self.saved_artifacts["training_report"] = report_path
        
        # Update pipeline state
        result = pipeline_state.copy()
        result["model_persistence_results"] = {
            "version": version_info,
            "saved_artifacts": self.saved_artifacts,
            "metadata": metadata,
            "summary": self._create_persistence_summary()
        }
        
        return result
    
    def _collect_models(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Collect all models from pipeline state.
        
        Args:
            pipeline_state: Pipeline state
            
        Returns:
            Dictionary of models by category
        """
        models = {}
        
        # Tactician specialist models
        if "tactician_specialist_models" in pipeline_state:
            models["tactician_specialists"] = pipeline_state["tactician_specialist_models"]
        
        # Analyst models
        if "analyst_ensembles" in pipeline_state:
            models["analyst_ensembles"] = pipeline_state["analyst_ensembles"]
        
        if "enhanced_analyst_models" in pipeline_state:
            models["enhanced_analysts"] = pipeline_state["enhanced_analyst_models"]
        
        # Calibrated models
        if "calibrated_models" in pipeline_state:
            models["calibrated_models"] = pipeline_state["calibrated_models"]
        
        return models
    
    async def _save_model_category(
        self,
        category: str,
        models: Any,
        version_info: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, str]:
        """Save a category of models.
        
        Args:
            category: Model category name
            models: Models to save
            version_info: Version information
            pipeline_state: Pipeline state for additional context
            
        Returns:
            Dictionary of saved file paths
        """
        saved_paths = {}
        
        # Flatten nested model structures
        flattened_models = self._flatten_models(models)
        
        for model_name, model in flattened_models.items():
            try:
                # Create unique model identifier
                model_id = f"{category}_{model_name}"
                
                # Extract model-specific metadata
                model_metadata = await self._extract_model_metadata(
                    model, model_name, category, pipeline_state
                )
                
                # Save model in multiple formats
                for format_name in self.persistence_config["save_formats"]:
                    try:
                        path = await self.model_serializer.save_model(
                            model,
                            model_id,
                            format_name,
                            version_info,
                            model_metadata
                        )
                        
                        if path:
                            saved_paths[f"{model_id}_{format_name}"] = path
                            self.logger.info(f"  Saved {model_id} as {format_name}")
                            
                    except Exception as e:
                        self.logger.warning(f"  Failed to save {model_id} as {format_name}: {str(e)}")
                
                # Save feature importance if available
                if self.persistence_config["save_feature_importance"]:
                    importance_path = await self._save_feature_importance(
                        model, model_id, version_info
                    )
                    if importance_path:
                        saved_paths[f"{model_id}_importance"] = importance_path
                        
            except Exception as e:
                self.logger.error(f"Failed to save model {model_name}: {str(e)}")
        
        return saved_paths
    
    def _flatten_models(self, models: Any, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested model structures.
        
        Args:
            models: Models to flatten
            prefix: Prefix for model names
            
        Returns:
            Flattened dictionary of models
        """
        flattened = {}
        
        if isinstance(models, dict):
            for key, value in models.items():
                new_prefix = f"{prefix}_{key}" if prefix else key
                
                if isinstance(value, dict) and not hasattr(value, 'predict'):
                    # Recursive flattening
                    flattened.update(self._flatten_models(value, new_prefix))
                else:
                    # Model object
                    flattened[new_prefix] = value
        else:
            # Single model
            flattened[prefix or "model"] = models
        
        return flattened
    
    async def _extract_model_metadata(
        self,
        model: Any,
        model_name: str,
        category: str,
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract metadata for a model.
        
        Args:
            model: Model object
            model_name: Model name
            category: Model category
            pipeline_state: Pipeline state
            
        Returns:
            Model metadata dictionary
        """
        metadata = {
            "model_name": model_name,
            "category": category,
            "model_type": model.__class__.__name__,
            "created_at": datetime.now().isoformat()
        }
        
        # Add model-specific metadata
        if hasattr(model, 'get_params'):
            metadata["parameters"] = model.get_params()
        
        # Add performance metrics if available
        if category == "tactician_specialists":
            eval_results = pipeline_state.get("specialist_evaluation_results", {})
            # Extract relevant metrics
            metadata["performance_metrics"] = self._extract_performance_metrics(
                model_name, eval_results
            )
        
        # Add training data statistics if configured
        if self.persistence_config["save_training_data_stats"]:
            metadata["training_data_stats"] = self._extract_training_stats(pipeline_state)
        
        return metadata
    
    def _extract_performance_metrics(
        self,
        model_name: str,
        evaluation_results: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extract performance metrics for a model."""
        metrics = {}
        
        # Search for model metrics in evaluation results
        for regime_id, regime_results in evaluation_results.items():
            if isinstance(regime_results, dict):
                for tactic, tactic_results in regime_results.items():
                    if model_name.endswith(f"{regime_id}_{tactic}"):
                        if "average_metrics" in tactic_results:
                            metrics = tactic_results["average_metrics"]
                            break
        
        return metrics
    
    def _extract_training_stats(self, pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Extract training data statistics."""
        stats = {
            "total_samples": 0,
            "feature_count": 0,
            "regime_count": 0,
            "date_range": {}
        }
        
        # Extract from labeled data
        if "tactician_labeled_data" in pipeline_state:
            data = pipeline_state["tactician_labeled_data"]
            stats["total_samples"] = len(data)
            stats["feature_count"] = len(data.columns)
            
            if hasattr(data.index, 'min') and hasattr(data.index, 'max'):
                stats["date_range"] = {
                    "start": str(data.index.min()),
                    "end": str(data.index.max())
                }
        
        # Extract regime count
        if "regime_data" in pipeline_state:
            stats["regime_count"] = len(pipeline_state["regime_data"])
        
        return stats
    
    async def _save_feature_importance(
        self,
        model: Any,
        model_id: str,
        version_info: Dict[str, Any]
    ) -> Optional[str]:
        """Save feature importance for a model.
        
        Args:
            model: Model object
            model_id: Model identifier
            version_info: Version information
            
        Returns:
            Path to saved importance file
        """
        if not hasattr(model, 'feature_importances_'):
            return None
        
        try:
            importance_data = {
                "model_id": model_id,
                "feature_importances": model.feature_importances_.tolist(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Save as JSON
            base_dir = Path(self.persistence_config["base_dir"])
            version_dir = base_dir / version_info["version"] / "feature_importance"
            version_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = version_dir / f"{model_id}_importance.json"
            
            with open(file_path, 'w') as f:
                json.dump(importance_data, f, indent=2)
            
            return str(file_path)
            
        except Exception as e:
            self.logger.warning(f"Failed to save feature importance: {str(e)}")
            return None
    
    async def _save_metadata(
        self,
        metadata: Dict[str, Any],
        version_info: Dict[str, Any]
    ) -> str:
        """Save comprehensive metadata.
        
        Args:
            metadata: Metadata dictionary
            version_info: Version information
            
        Returns:
            Path to saved metadata file
        """
        base_dir = Path(self.persistence_config["base_dir"])
        version_dir = base_dir / version_info["version"]
        version_dir.mkdir(parents=True, exist_ok=True)
        
        metadata_path = version_dir / "metadata.json"
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        return str(metadata_path)
    
    async def _create_training_summary(
        self,
        pipeline_state: Dict[str, Any],
        saved_artifacts: Dict[str, str],
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create comprehensive training summary.
        
        Args:
            pipeline_state: Pipeline state
            saved_artifacts: Saved artifact paths
            metadata: Training metadata
            
        Returns:
            Training summary dictionary
        """
        summary = {
            "training_metadata": metadata,
            "saved_artifacts": saved_artifacts,
            "pipeline_summary": {},
            "model_inventory": {},
            "performance_overview": {}
        }
        
        # Extract pipeline summary
        for step_num in range(1, 22):
            step_key = f"step{step_num:02d}_*_completed"
            completed_steps = [k for k in pipeline_state.keys() if k.startswith(f"step{step_num:02d}_") and k.endswith("_completed")]
            if completed_steps:
                summary["pipeline_summary"][f"step_{step_num}"] = "completed"
        
        # Create model inventory
        for category, artifacts in saved_artifacts.items():
            if category != "metadata":
                summary["model_inventory"][category] = len([p for p in artifacts if not p.endswith("_importance")])
        
        # Extract performance overview
        if "specialist_evaluation_results" in pipeline_state:
            eval_results = pipeline_state["specialist_evaluation_results"]
            if "overall_metrics" in eval_results:
                summary["performance_overview"] = eval_results["overall_metrics"]
        
        return summary
    
    async def _save_training_report(
        self,
        training_summary: Dict[str, Any],
        version_info: Dict[str, Any]
    ) -> str:
        """Save training report.
        
        Args:
            training_summary: Training summary
            version_info: Version information
            
        Returns:
            Path to saved report
        """
        base_dir = Path(self.persistence_config["base_dir"])
        version_dir = base_dir / version_info["version"]
        
        report_path = version_dir / "training_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)
        
        # Also create a human-readable report
        readable_report_path = version_dir / "training_report.txt"
        
        with open(readable_report_path, 'w') as f:
            f.write("TRAINING REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Version: {version_info['version']}\n")
            f.write(f"Created: {version_info['timestamp']}\n\n")
            
            f.write("MODEL INVENTORY\n")
            f.write("-" * 30 + "\n")
            for category, count in training_summary["model_inventory"].items():
                f.write(f"{category}: {count} models\n")
            
            f.write("\nPIPELINE SUMMARY\n")
            f.write("-" * 30 + "\n")
            for step, status in training_summary["pipeline_summary"].items():
                f.write(f"{step}: {status}\n")
            
            if training_summary.get("performance_overview"):
                f.write("\nPERFORMANCE OVERVIEW\n")
                f.write("-" * 30 + "\n")
                for metric, value in training_summary["performance_overview"].items():
                    if isinstance(value, float):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
        
        return str(report_path)
    
    def _create_persistence_summary(self) -> Dict[str, Any]:
        """Create persistence operation summary."""
        summary = {
            "total_artifacts_saved": len(self.saved_artifacts),
            "categories_saved": list(self.saved_artifacts.keys()),
            "formats_used": self.persistence_config["save_formats"],
            "versioning_enabled": self.persistence_config["enable_versioning"],
            "registry_enabled": self.persistence_config["use_model_registry"]
        }
        
        # Count models by category
        model_counts = {}
        for category, paths in self.saved_artifacts.items():
            if category not in ["metadata", "training_report"]:
                if isinstance(paths, dict):
                    model_counts[category] = len(paths)
                else:
                    model_counts[category] = 1
        
        summary["models_by_category"] = model_counts
        summary["total_models_saved"] = sum(model_counts.values())
        
        return summary
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate step outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for persistence results
        if "model_persistence_results" not in pipeline_state:
            errors.append("No model persistence results found")
        else:
            results = pipeline_state["model_persistence_results"]
            
            if "saved_artifacts" not in results or len(results["saved_artifacts"]) == 0:
                errors.append("No artifacts were saved")
            
            if "version" not in results:
                errors.append("No version information created")
        
        return len(errors) == 0, errors
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required inputs for this step."""
        return [
            "symbol",
            "exchange"
        ]
    
    def get_produced_outputs(self) -> List[str]:
        """Get list of outputs produced by this step."""
        return [
            "model_persistence_results"
        ]
    
    def get_dependencies(self) -> List[str]:
        """Get list of step dependencies."""
        # This step can run after any model training step
        return []