"""
Model Saving Utilities

This module provides standardized functions for saving models with probability outputs
and loading them back. It ensures consistent model data structure across all training steps.
"""

import os
import pickle
import joblib
from datetime import datetime
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from .model_probability_generator import ModelProbabilityGenerator

logger = logging.getLogger(__name__)


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
        else:
                raise ValueError(f"Unsupported save format: {save_format}")

        logger.info(f"Model saved successfully to {model_path}")
        logger.info(f"Model type: {standardized_model_data['model_type']}")
        logger.info(f"Probabilities: {price_action_probabilities}")

        return standardized_model_data

    except Exception as e:
        # Extract multi-output components
        multi_output_trainer = model_data.get("multi_output_trainer")
        multi_output_models = model_data.get("multi_output_models")

        # Generate probability outputs if trainer is available
        if multi_output_trainer and multi_output_models:
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
        else:
                raise ValueError(f"Unsupported save format: {save_format}")

        logger.info(f"Multi-output model saved successfully to {model_path}")
        logger.info(f"Model type: {standardized_model_data['model_type']}")
        logger.info(f"Probabilities: {price_action_probabilities}")

        return standardized_model_data

    except Exception as e:

        # Try to load with joblib first
        try:
            model_data = joblib.load(model_path)
        except:

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
        # Generate probability outputs
        probability_generator = ModelProbabilityGenerator()
        model_type = model_data.get("model_type", "classification")

        price_action_probabilities = probability_generator.generate_price_action_probabilities(
            model_data["model"], X_test, y_test, market_data, model_type
        )

        # Validate probabilities
        if not probability_generator.validate_probabilities(price_action_probabilities):
            price_action_probabilities = probability_generator._get_default_probabilities(model_type)

        # Save model with probabilities
        return save_model_with_probabilities(
            model_data, model_path, price_action_probabilities, save_format
        )

    except Exception as e:
                    if validate_model_probabilities(model_data):
valid_models.append(model_path)
                except Exception as e:

        return valid_models

    except Exception as e:
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
        results = {
            "total_models": 0,
            "valid_models": 0,
            "invalid_models": 0,
            "errors": [],
            "valid_model_paths": [],
            "invalid_model_paths": []
        }

        for filename in os.listdir(directory):
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
                    results["errors"].append(f"{filename}: {str(e)}")
                    results["invalid_model_paths"].append(model_path)

        return results

    except Exception as e:
        return {"error": str(e)}