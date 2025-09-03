"""Base validation step using BaseStep pattern."""

from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.base_step import BaseStep
from src.utils.logger import system_logger


class BaseValidationStep(BaseStep):
    """Base class for all validation steps."""
    
    def __init__(self, config: Dict[str, Any], step_number: str, validation_type: str):
        """Initialize the validation step.
        
        Args:
            config: Configuration dictionary
            step_number: Step number (e.g., "16", "17")
            validation_type: Type of validation (e.g., "confidence_calibration")
        """
        super().__init__(config, step_number, validation_type)
        
        # Common validation configuration
        self.validation_config = {
            "min_samples": config.get("min_validation_samples", 100),
            "validation_split": config.get("validation_split", 0.2),
            "random_state": config.get("random_state", 42),
            "parallel_processing": config.get("parallel_processing", True),
            "save_results": config.get("save_validation_results", True)
        }
    
    def validate_inputs(
        self, 
        training_input: Dict[str, Any], 
        pipeline_state: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate common inputs for validation steps.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for trained models
        if "tactician_specialist_models" not in pipeline_state:
            errors.append("No trained models found for validation")
        
        # Check for data
        if "features" not in pipeline_state and "market_data" not in pipeline_state:
            errors.append("No data found for validation")
        
        # Step-specific validation
        step_errors = self._validate_step_specific_inputs(training_input, pipeline_state)
        errors.extend(step_errors)
        
        return len(errors) == 0, errors
    
    @abstractmethod
    def _validate_step_specific_inputs(
        self,
        training_input: Dict[str, Any],
        pipeline_state: Dict[str, Any]
    ) -> List[str]:
        """Validate step-specific inputs.
        
        Args:
            training_input: Training input parameters
            pipeline_state: Current pipeline state
            
        Returns:
            List of validation errors
        """
    
    def validate_outputs(self, pipeline_state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate common outputs for validation steps.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            Tuple of (is_valid, errors)
        """
        errors = []
        
        # Check for validation results
        validation_key = f"{self.full_step_name}_results"
        if validation_key not in pipeline_state:
            errors.append(f"Missing validation results: {validation_key}")
        
        # Step-specific validation
        step_errors = self._validate_step_specific_outputs(pipeline_state)
        errors.extend(step_errors)
        
        return len(errors) == 0, errors
    
    @abstractmethod
    def _validate_step_specific_outputs(
        self,
        pipeline_state: Dict[str, Any]
    ) -> List[str]:
        """Validate step-specific outputs.
        
        Args:
            pipeline_state: Updated pipeline state
            
        Returns:
            List of validation errors
        """
    
    def _extract_validation_data(
        self,
        pipeline_state: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract data for validation.
        
        Args:
            pipeline_state: Pipeline state
            
        Returns:
            Tuple of (features, labels)
        """
        # Try different data sources
        if "tactician_labeled_data" in pipeline_state:
            data = pipeline_state["tactician_labeled_data"]
            if "label" in data.columns:
                labels = data["label"]
                features = data.drop(columns=["label"])
                return features, labels
        
        if "features" in pipeline_state and "labels" in pipeline_state:
            return pipeline_state["features"], pipeline_state["labels"]
        
        if "market_data" in pipeline_state:
            data = pipeline_state["market_data"]
            # Generate synthetic labels for validation
            if "close" in data.columns:
                returns = data["close"].pct_change()
                labels = (returns > 0).astype(int)
                return data, labels
        
        return pd.DataFrame(), pd.Series()
    
    def _get_models_for_validation(
        self,
        pipeline_state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get models to validate.
        
        Args:
            pipeline_state: Pipeline state
            
        Returns:
            Dictionary of models
        """
        models = {}
        
        # Get specialist models
        if "tactician_specialist_models" in pipeline_state:
            specialist_models = pipeline_state["tactician_specialist_models"]
            
            # Flatten the structure
            for regime_id, regime_models in specialist_models.items():
                for tactic_name, tactic_models in regime_models.items():
                    model_key = f"{regime_id}_{tactic_name}"
                    
                    if isinstance(tactic_models, dict):
                        models.update({
                            f"{model_key}_{model_name}": model
                            for model_name, model in tactic_models.items()
                        })
                    else:
                        models[model_key] = tactic_models
        
        # Get analyst models if available
        if "analyst_ensembles" in pipeline_state:
            analyst_ensembles = pipeline_state["analyst_ensembles"]
            
            for regime_id, ensemble_data in analyst_ensembles.items():
                if isinstance(ensemble_data, dict) and "ensemble" in ensemble_data:
                    for ens_type, ens_model in ensemble_data["ensemble"].items():
                        models[f"analyst_{regime_id}_{ens_type}"] = ens_model
        
        return models
    
    def _create_validation_summary(
        self,
        validation_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a summary of validation results.
        
        Args:
            validation_results: Raw validation results
            
        Returns:
            Validation summary
        """
        summary = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "validation_type": self.step_name,
            "models_validated": 0,
            "overall_metrics": {},
            "key_findings": [],
            "warnings": [],
            "recommendations": []
        }
        
        # Count models
        if "model_results" in validation_results:
            summary["models_validated"] = len(validation_results["model_results"])
        
        # Calculate overall metrics
        if "overall_metrics" in validation_results:
            summary["overall_metrics"] = validation_results["overall_metrics"]
        
        # Add step-specific summary items
        self._add_step_specific_summary(summary, validation_results)
        
        return summary
    
    @abstractmethod
    def _add_step_specific_summary(
        self,
        summary: Dict[str, Any],
        validation_results: Dict[str, Any]
    ) -> None:
        """Add step-specific items to summary.
        
        Args:
            summary: Summary dictionary to update
            validation_results: Validation results
        """