"""
Model Saving Utilities

This module provides standardized functions for saving models with probability outputs
and loading them back. It ensures consistent model data structure across all training steps.
"""

import os
import pickle
import joblib
import json
from typing import Dict, Any, Optional, Union
from datetime import datetime
import logging
import numpy as np
import pandas as pd

from .model_probability_generator import ModelProbabilityGenerator

logger = logging.getLogger(__name__)


def save_model_with_probabilities(
    model_data: Dict[str, Any],
    model_path: str,
    price_action_probabilities: Dict[str, float],
    save_format: str = "joblib"
) -> Dict[str, Any]:
    """
    Save model with standardized structure including probability outputs.
    
    Args:
        model_data: Dictionary containing model and metadata
        model_path: Path where to save the model
        price_action_probabilities: Dictionary of probability outputs
        save_format: Format to save model ('joblib' or 'pickle')
        
    Returns:
        Dict containing the standardized model data structure
    """
    try:
        # Create standardized model data structure
        standardized_model_data = {
            "model": model_data.get("model"),
            "model_type": model_data.get("model_type", "unknown"),
            "training_date": model_data.get("training_date", datetime.now().isoformat()),
            "hyperparameters": model_data.get("hyperparameters", {}),
            "metrics": model_data.get("metrics", {}),
            "feature_importance": model_data.get("feature_importance", {}),
            "symbol": model_data.get("symbol", ""),
            "exchange": model_data.get("exchange", ""),
            "step_name": model_data.get("step_name", ""),
            "version": model_data.get("version", "1.0"),
            # NEW: Required probability outputs
            "price_action_probabilities": price_action_probabilities,
            # Additional metadata
            "save_timestamp": datetime.now().isoformat(),
            "save_format": save_format
        }
        
        # Ensure directory exists
        model_dir = os.path.dirname(model_path)
        if model_dir:  # Only create directory if path is not empty
            os.makedirs(model_dir, exist_ok=True)
        
        # Save model based on format
        if save_format.lower() == "joblib":
            joblib.dump(standardized_model_data, model_path)
        elif save_format.lower() == "pickle":
            with open(model_path, 'wb') as f:
                pickle.dump(standardized_model_data, f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        logger.info(f"Model saved successfully to {model_path}")
        logger.info(f"Model type: {standardized_model_data['model_type']}")
        logger.info(f"Probabilities: {price_action_probabilities}")
        
        return standardized_model_data
        
    except Exception as e:
        logger.error(f"Error saving model with probabilities: {e}")
        raise


def save_multi_output_model_with_probabilities(
    model_data: Dict[str, Any],
    model_path: str,
    save_format: str = "joblib"
) -> Dict[str, Any]:
    """
    Save multi-output model with probability outputs.
    
    Args:
        model_data: Dictionary containing multi-output model and metadata
        model_path: Path where to save the model
        save_format: Format to save model ('joblib' or 'pickle')
        
    Returns:
        Dict containing the standardized multi-output model data structure
    """
    try:
        # Extract multi-output components
        multi_output_trainer = model_data.get("multi_output_trainer")
        multi_output_models = model_data.get("multi_output_models")
        
        # Generate probability outputs if trainer is available
        if multi_output_trainer and multi_output_models:
            # Use test data to generate probabilities
            X_test = model_data.get("X_test", np.random.randn(100, 10))
            market_data = model_data.get("market_data", pd.DataFrame({
                'close': np.random.randn(100),
                'volume': np.random.randn(100)
            }))
            
            price_action_probabilities = multi_output_trainer.predict_probabilities(
                X_test, market_data
            )
        else:
            price_action_probabilities = model_data.get("price_action_probabilities", {})
        
        # Create standardized model data structure
        standardized_model_data = {
            "model_type": "multi_output",
            "multi_output_trainer": multi_output_trainer,
            "multi_output_models": multi_output_models,
            "ensemble_weights": multi_output_trainer.ensemble_weights if multi_output_trainer else None,
            "calibrators": multi_output_trainer.calibrators if multi_output_trainer else None,
            "price_action_probabilities": price_action_probabilities,
            "training_date": model_data.get("training_date", datetime.now().isoformat()),
            "hyperparameters": model_data.get("hyperparameters", {}),
            "metrics": model_data.get("metrics", {}),
            "symbol": model_data.get("symbol", ""),
            "exchange": model_data.get("exchange", ""),
            "step_name": model_data.get("step_name", ""),
            "version": model_data.get("version", "1.0"),
            "save_timestamp": datetime.now().isoformat(),
            "save_format": save_format
        }
        
        # Ensure directory exists
        model_dir = os.path.dirname(model_path)
        if model_dir:  # Only create directory if path is not empty
            os.makedirs(model_dir, exist_ok=True)
        
        # Save model based on format
        if save_format.lower() == "joblib":
            joblib.dump(standardized_model_data, model_path)
        elif save_format.lower() == "pickle":
            with open(model_path, 'wb') as f:
                pickle.dump(standardized_model_data, f)
        else:
            raise ValueError(f"Unsupported save format: {save_format}")
        
        logger.info(f"Multi-output model saved successfully to {model_path}")
        logger.info(f"Model type: {standardized_model_data['model_type']}")
        logger.info(f"Probabilities: {price_action_probabilities}")
        
        return standardized_model_data
        
    except Exception as e:
        logger.error(f"Error saving multi-output model: {e}")
        raise


def load_model_with_probabilities(model_path: str) -> Dict[str, Any]:
    """
    Load model with probability outputs from file.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Dict containing the loaded model data and probabilities
    """
    try:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Try to load with joblib first
        try:
            model_data = joblib.load(model_path)
        except:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
        
        # Validate loaded data
        if not isinstance(model_data, dict):
            raise ValueError("Loaded model data is not a dictionary")
        
        # Check for required fields based on model type
        model_type = model_data.get('model_type', 'unknown')
        
        if model_type == "multi_output":
            # Multi-output model validation
            required_fields = ["model_type", "price_action_probabilities"]
            for field in required_fields:
                if field not in model_data:
                    logger.warning(f"Missing required field in multi-output model: {field}")
            
            # Check for multi-output specific components
            multi_output_trainer = model_data.get("multi_output_trainer")
            multi_output_models = model_data.get("multi_output_models")
            
            if multi_output_trainer and multi_output_models:
                logger.info("✅ Loaded multi-output model successfully")
            else:
                logger.warning("⚠️ Multi-output model missing components")
        else:
            # Standard model validation
            required_fields = ["model", "model_type", "price_action_probabilities"]
            for field in required_fields:
                if field not in model_data:
                    logger.warning(f"Missing required field in loaded model: {field}")
        
        logger.info(f"Model loaded successfully from {model_path}")
        logger.info(f"Model type: {model_type}")
        
        return model_data
        
    except Exception as e:
        logger.error(f"Error loading model with probabilities: {e}")
        raise


def load_multi_output_model_with_probabilities(model_path: str) -> Dict[str, Any]:
    """
    Load multi-output model with probability outputs from file.
    
    Args:
        model_path: Path to the saved multi-output model file
        
    Returns:
        Dict containing the loaded multi-output model data and probabilities
    """
    try:
        model_data = load_model_with_probabilities(model_path)
        
        # Check if it's a multi-output model
        if model_data.get("model_type") == "multi_output":
            multi_output_trainer = model_data.get("multi_output_trainer")
            multi_output_models = model_data.get("multi_output_models")
            
            if multi_output_trainer and multi_output_models:
                logger.info("✅ Loaded multi-output model successfully")
                return model_data
            else:
                logger.warning("⚠️ Multi-output model missing components")
                return model_data
        else:
            logger.info("ℹ️ Standard model loaded (not multi-output)")
            return model_data
        
    except Exception as e:
        logger.error(f"Error loading multi-output model: {e}")
        raise


def validate_model_probabilities(model_data: Dict[str, Any]) -> bool:
    """
    Validate that a model has valid probability outputs.
    
    Args:
        model_data: Dictionary containing model data
        
    Returns:
        bool: True if probabilities are valid
    """
    try:
        # Check if probabilities exist
        if "price_action_probabilities" not in model_data:
            logger.error("Model missing price_action_probabilities")
            return False
        
        probabilities = model_data["price_action_probabilities"]
        model_type = model_data.get("model_type", "unknown")
        
        # Check required probability keys
        required_keys = [
            "triple_barrier_probability",
            "direction_probability",
            "magnitude_probability", 
            "barrier_avoidance_probability"
        ]
        
        for key in required_keys:
            if key not in probabilities:
                logger.error(f"Missing required probability key: {key}")
                return False
            
            prob = probabilities[key]
            if not isinstance(prob, (int, float)) or not 0.0 <= prob <= 1.0:
                logger.error(f"Invalid probability value for {key}: {prob}")
                return False
        
        # Additional validation for multi-output models
        if model_type == "multi_output":
            multi_output_trainer = model_data.get("multi_output_trainer")
            multi_output_models = model_data.get("multi_output_models")
            
            if not multi_output_trainer:
                logger.error("Multi-output model missing trainer")
                return False
            
            if not multi_output_models:
                logger.error("Multi-output model missing models")
                return False
            
            # Check if trainer is trained
            if not hasattr(multi_output_trainer, 'is_trained') or not multi_output_trainer.is_trained:
                logger.warning("Multi-output trainer not trained")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating model probabilities: {e}")
        return False


def update_model_probabilities(
    model_path: str,
    new_probabilities: Dict[str, float],
    save_format: str = "joblib"
) -> bool:
    """
    Update probability outputs for an existing saved model.
    
    Args:
        model_path: Path to the existing model file
        new_probabilities: New probability outputs
        save_format: Format to save the updated model
        
    Returns:
        bool: True if update was successful
    """
    try:
        # Load existing model
        model_data = load_model_with_probabilities(model_path)
        
        # Update probabilities
        model_data["price_action_probabilities"] = new_probabilities
        model_data["update_timestamp"] = datetime.now().isoformat()
        
        # Save updated model
        save_model_with_probabilities(
            model_data, model_path, new_probabilities, save_format
        )
        
        logger.info(f"Model probabilities updated successfully: {model_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error updating model probabilities: {e}")
        return False


def generate_and_save_model_probabilities(
    model_data: Dict[str, Any],
    model_path: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    market_data: pd.DataFrame,
    save_format: str = "joblib"
) -> Dict[str, Any]:
    """
    Generate probability outputs and save model with them.
    
    Args:
        model_data: Dictionary containing model and metadata
        model_path: Path where to save the model
        X_test: Test features
        y_test: Test targets
        market_data: Market data
        save_format: Format to save model
        
    Returns:
        Dict containing the saved model data
    """
    try:
        # Generate probability outputs
        probability_generator = ModelProbabilityGenerator()
        model_type = model_data.get("model_type", "classification")
        
        price_action_probabilities = probability_generator.generate_price_action_probabilities(
            model_data["model"], X_test, y_test, market_data, model_type
        )
        
        # Validate probabilities
        if not probability_generator.validate_probabilities(price_action_probabilities):
            logger.warning("Generated probabilities failed validation, using defaults")
            price_action_probabilities = probability_generator._get_default_probabilities(model_type)
        
        # Save model with probabilities
        return save_model_with_probabilities(
            model_data, model_path, price_action_probabilities, save_format
        )
        
    except Exception as e:
        logger.error(f"Error generating and saving model probabilities: {e}")
        raise


def list_models_with_probabilities(directory: str) -> list:
    """
    List all models in a directory that have probability outputs.
    
    Args:
        directory: Directory to search for models
        
    Returns:
        list: List of model file paths with valid probabilities
    """
    try:
        valid_models = []
        
        for filename in os.listdir(directory):
            if filename.endswith(('.pkl', '.joblib')):
                model_path = os.path.join(directory, filename)
                try:
                    model_data = load_model_with_probabilities(model_path)
                    if validate_model_probabilities(model_data):
                        valid_models.append(model_path)
                except Exception as e:
                    logger.warning(f"Could not load model {filename}: {e}")
        
        return valid_models
        
    except Exception as e:
        logger.error(f"Error listing models with probabilities: {e}")
        return []


def get_model_probability_summary(model_path: str) -> Dict[str, Any]:
    """
    Get a summary of model probability outputs.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Dict containing probability summary
    """
    try:
        model_data = load_model_with_probabilities(model_path)
        
        if "price_action_probabilities" not in model_data:
            return {"error": "No probability outputs found"}
        
        probabilities = model_data["price_action_probabilities"]
        
        summary = {
            "model_type": model_data.get("model_type", "unknown"),
            "training_date": model_data.get("training_date", "unknown"),
            "symbol": model_data.get("symbol", "unknown"),
            "exchange": model_data.get("exchange", "unknown"),
            "probabilities": probabilities,
            "average_probability": np.mean(list(probabilities.values())),
            "min_probability": np.min(list(probabilities.values())),
            "max_probability": np.max(list(probabilities.values()))
        }
        
        return summary
        
    except Exception as e:
        logger.error(f"Error getting model probability summary: {e}")
        return {"error": str(e)}


def batch_validate_models(directory: str) -> Dict[str, Any]:
    """
    Validate all models in a directory for probability outputs.
    
    Args:
        directory: Directory containing model files
        
    Returns:
        Dict containing validation results
    """
    try:
        results = {
            "total_models": 0,
            "valid_models": 0,
            "invalid_models": 0,
            "errors": [],
            "valid_model_paths": [],
            "invalid_model_paths": []
        }
        
        for filename in os.listdir(directory):
            if filename.endswith(('.pkl', '.joblib')):
                model_path = os.path.join(directory, filename)
                results["total_models"] += 1
                
                try:
                    model_data = load_model_with_probabilities(model_path)
                    if validate_model_probabilities(model_data):
                        results["valid_models"] += 1
                        results["valid_model_paths"].append(model_path)
                    else:
                        results["invalid_models"] += 1
                        results["invalid_model_paths"].append(model_path)
                except Exception as e:
                    results["invalid_models"] += 1
                    results["errors"].append(f"{filename}: {str(e)}")
                    results["invalid_model_paths"].append(model_path)
        
        return results
        
    except Exception as e:
        logger.error(f"Error in batch validation: {e}")
        return {"error": str(e)}