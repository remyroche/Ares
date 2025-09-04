#!/usr/bin/env python3
"""
Enhanced Confidence Calibration Step

Enhanced version of confidence calibration with comprehensive protection:
- Data validation and integrity checks
- Error handling and recovery
- Performance monitoring
- State management
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import pickle

from src.utils.logger import system_logger
from src.training.steps.optimisation.optimisation_decorators import (
    protect_optimisation_operation,
    protect_data_operation,
    data_protection,
    error_handling,
    performance_monitoring,
    operation_logging
)
from src.training.steps.optimisation.optimisation_utilities import (
    get_data_formatting_utils,
    get_analysis_operations_utils,
    get_data_access_control,
    get_pipeline_state_manager,
    get_performance_optimizer
)
from src.utils.pipeline_protection_framework import (
    ValidationLevel,
    OperationType,
    DataIntegrityCheck
)
from src.utils.common_operations import (
    ensure_directory,
    safe_file_exists,
    safe_json_dump,
    safe_json_load,
    format_datetime,
    get_current_datetime
)


class EnhancedConfidenceCalibrationStep:
    """Enhanced confidence calibration step with comprehensive protection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("EnhancedConfidenceCalibrationStep")
        
        # Initialize utilities
        self.data_formatter = get_data_formatting_utils()
        self.analysis_ops = get_analysis_operations_utils()
        self.data_access = get_data_access_control()
        self.state_manager = get_pipeline_state_manager()
        self.performance_optimizer = get_performance_optimizer()
        
        # Configuration
        self.calibration_methods = config.get("calibration_methods", ["isotonic", "sigmoid"])
        self.cv_folds = config.get("cv_folds", 5)
        self.random_state = config.get("random_state", 42)
        self.min_samples = config.get("min_samples", 100)
        
    @protect_optimisation_operation(ValidationLevel.CRITICAL)
    @data_protection(ValidationLevel.CRITICAL, backup_enabled=True)
    @error_handling(retry_count=3, critical_errors=["data_corruption", "insufficient_data"])
    @performance_monitoring(alert_threshold=300.0)  # 5 minutes
    @operation_logging(log_level="INFO", audit_trail=True)
    async def calibrate_confidence(self, 
                                 symbol: str,
                                 exchange: str,
                                 timeframe: str,
                                 data_dir: str) -> bool:
        """Calibrate confidence with comprehensive protection."""
        try:
            self.logger.info(f"🎯 Starting enhanced confidence calibration for {symbol} on {exchange}")
            
            # Step 1: Load and validate data
            training_data = await self._load_and_validate_training_data(symbol, exchange, data_dir)
            if training_data is None:
                return False
            
            # Step 2: Load model predictions
            model_predictions = await self._load_model_predictions(symbol, exchange, data_dir)
            if model_predictions is None:
                return False
            
            # Step 3: Prepare calibration data
            calibration_data = await self._prepare_calibration_data(training_data, model_predictions)
            if calibration_data is None:
                return False
            
            # Step 4: Perform calibration
            calibration_results = await self._perform_calibration(calibration_data)
            if calibration_results is None:
                return False
            
            # Step 5: Validate calibration results
            validation_result = await self._validate_calibration_results(calibration_results)
            if not validation_result:
                return False
            
            # Step 6: Save results
            success = await self._save_calibration_results(calibration_results, symbol, exchange, data_dir)
            
            if success:
                self.logger.info("✅ Enhanced confidence calibration completed successfully")
                return True
            else:
                self.logger.error("❌ Failed to save calibration results")
                return False
                
        except Exception as e:
            self.logger.exception(f"❌ Enhanced confidence calibration failed: {e}")
            return False
    
    @protect_data_operation(ValidationLevel.STANDARD)
    async def _load_and_validate_training_data(self, 
                                             symbol: str,
                                             exchange: str,
                                             data_dir: str) -> Optional[pd.DataFrame]:
        """Load and validate training data."""
        try:
            self.logger.info("📁 Loading and validating training data...")
            
            # Construct data file path
            data_file = f"{data_dir}/aggtrades_{exchange}_{symbol}_consolidated.parquet"
            
            # Load data with access control
            training_data = self.data_access.secure_data_loading(
                data_file, 
                user_id="confidence_calibration",
                validate_integrity=True
            )
            
            if training_data is None:
                self.logger.error(f"❌ Failed to load training data from {data_file}")
                return None
            
            # Validate data quality
            validation = self.data_formatter.data_validator.validate_dataframe(
                training_data,
                min_rows=self.min_samples,
                max_null_ratio=0.1
            )
            
            if not validation.passed:
                self.logger.error(f"❌ Training data validation failed: {validation}")
                return None
            
            # Optimize memory usage
            training_data = self.performance_optimizer.optimize_memory_usage(training_data)
            
            self.logger.info(f"✅ Training data loaded and validated: {len(training_data)} rows")
            return training_data
            
        except Exception as e:
            self.logger.exception(f"❌ Training data loading failed: {e}")
            return None
    
    @protect_data_operation(ValidationLevel.STANDARD)
    async def _load_model_predictions(self, 
                                    symbol: str,
                                    exchange: str,
                                    data_dir: str) -> Optional[Dict[str, Any]]:
        """Load model predictions."""
        try:
            self.logger.info("🤖 Loading model predictions...")
            
            # Construct predictions file path
            predictions_file = f"{data_dir}/{exchange}_{symbol}_model_predictions.pkl"
            
            # Load predictions with access control
            predictions = self.data_access.secure_data_loading(
                predictions_file,
                user_id="confidence_calibration",
                validate_integrity=False  # Custom validation for predictions
            )
            
            if predictions is None:
                self.logger.error(f"❌ Failed to load model predictions from {predictions_file}")
                return None
            
            # Validate predictions structure
            if not isinstance(predictions, dict):
                self.logger.error("❌ Model predictions must be a dictionary")
                return None
            
            required_keys = ["y_true", "y_pred", "y_prob"]
            missing_keys = [key for key in required_keys if key not in predictions]
            if missing_keys:
                self.logger.error(f"❌ Missing required prediction keys: {missing_keys}")
                return None
            
            # Validate prediction arrays
            for key in required_keys:
                if not isinstance(predictions[key], (np.ndarray, list)):
                    self.logger.error(f"❌ Predictions '{key}' must be an array")
                    return None
                
                if len(predictions[key]) == 0:
                    self.logger.error(f"❌ Predictions '{key}' is empty")
                    return None
            
            self.logger.info(f"✅ Model predictions loaded: {len(predictions['y_true'])} samples")
            return predictions
            
        except Exception as e:
            self.logger.exception(f"❌ Model predictions loading failed: {e}")
            return None
    
    @protect_data_operation(ValidationLevel.STANDARD)
    async def _prepare_calibration_data(self, 
                                      training_data: pd.DataFrame,
                                      model_predictions: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Prepare data for calibration."""
        try:
            self.logger.info("🔧 Preparing calibration data...")
            
            # Extract prediction arrays
            y_true = np.array(model_predictions["y_true"])
            y_pred = np.array(model_predictions["y_pred"])
            y_prob = np.array(model_predictions["y_prob"])
            
            # Validate array lengths
            if not (len(y_true) == len(y_pred) == len(y_prob)):
                self.logger.error("❌ Prediction arrays have different lengths")
                return None
            
            # Check for sufficient samples
            if len(y_true) < self.min_samples:
                self.logger.error(f"❌ Insufficient samples for calibration: {len(y_true)} < {self.min_samples}")
                return None
            
            # Handle missing values
            valid_indices = ~(np.isnan(y_true) | np.isnan(y_pred) | np.isnan(y_prob).any(axis=1))
            if not np.any(valid_indices):
                self.logger.error("❌ No valid samples after handling missing values")
                return None
            
            # Filter valid samples
            y_true_clean = y_true[valid_indices]
            y_pred_clean = y_pred[valid_indices]
            y_prob_clean = y_prob[valid_indices]
            
            # Create calibration data structure
            calibration_data = {
                "y_true": y_true_clean,
                "y_pred": y_pred_clean,
                "y_prob": y_prob_clean,
                "n_samples": len(y_true_clean),
                "n_classes": len(np.unique(y_true_clean)),
                "class_distribution": np.bincount(y_true_clean.astype(int)),
                "valid_indices": valid_indices,
                "preparation_timestamp": get_current_datetime().isoformat()
            }
            
            self.logger.info(f"✅ Calibration data prepared: {calibration_data['n_samples']} samples, {calibration_data['n_classes']} classes")
            return calibration_data
            
        except Exception as e:
            self.logger.exception(f"❌ Calibration data preparation failed: {e}")
            return None
    
    @protect_optimisation_operation(ValidationLevel.CRITICAL)
    async def _perform_calibration(self, calibration_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Perform confidence calibration."""
        try:
            self.logger.info("🎯 Performing confidence calibration...")
            
            y_true = calibration_data["y_true"]
            y_prob = calibration_data["y_prob"]
            
            calibration_results = {
                "calibration_methods": {},
                "best_method": None,
                "best_score": -np.inf,
                "calibration_metadata": {
                    "n_samples": len(y_true),
                    "n_classes": len(np.unique(y_true)),
                    "calibration_timestamp": get_current_datetime().isoformat(),
                    "methods_tested": self.calibration_methods
                }
            }
            
            # Test each calibration method
            for method in self.calibration_methods:
                try:
                    self.logger.info(f"🔧 Testing calibration method: {method}")
                    
                    method_result = await self._calibrate_with_method(y_true, y_prob, method)
                    if method_result is not None:
                        calibration_results["calibration_methods"][method] = method_result
                        
                        # Update best method
                        if method_result["score"] > calibration_results["best_score"]:
                            calibration_results["best_score"] = method_result["score"]
                            calibration_results["best_method"] = method
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Calibration method {method} failed: {e}")
                    continue
            
            # Validate that at least one method succeeded
            if not calibration_results["calibration_methods"]:
                self.logger.error("❌ All calibration methods failed")
                return None
            
            # Calculate overall calibration metrics
            calibration_results["overall_metrics"] = await self._calculate_overall_metrics(
                y_true, y_prob, calibration_results
            )
            
            self.logger.info(f"✅ Confidence calibration completed: best method = {calibration_results['best_method']}")
            return calibration_results
            
        except Exception as e:
            self.logger.exception(f"❌ Confidence calibration failed: {e}")
            return None
    
    async def _calibrate_with_method(self, 
                                   y_true: np.ndarray,
                                   y_prob: np.ndarray,
                                   method: str) -> Optional[Dict[str, Any]]:
        """Calibrate using a specific method."""
        try:
            from sklearn.calibration import CalibratedClassifierCV
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import cross_val_score
            
            if method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds='clip')
            elif method == "sigmoid":
                calibrator = LogisticRegression()
            else:
                self.logger.error(f"❌ Unknown calibration method: {method}")
                return None
            
            # For binary classification, use the positive class probabilities
            if y_prob.ndim > 1 and y_prob.shape[1] == 2:
                y_prob_cal = y_prob[:, 1]
            else:
                y_prob_cal = y_prob.flatten()
            
            # Fit calibrator
            calibrator.fit(y_prob_cal, y_true)
            
            # Generate calibrated probabilities
            y_prob_calibrated = calibrator.predict_proba(y_prob_cal.reshape(-1, 1)) if hasattr(calibrator, 'predict_proba') else calibrator.predict(y_prob_cal)
            
            # Calculate calibration score (Brier score)
            from sklearn.metrics import brier_score_loss
            brier_score = brier_score_loss(y_true, y_prob_calibrated)
            calibration_score = 1 - brier_score  # Higher is better
            
            # Calculate reliability diagram metrics
            reliability_metrics = await self._calculate_reliability_metrics(y_true, y_prob_calibrated)
            
            result = {
                "method": method,
                "calibrator": calibrator,
                "score": calibration_score,
                "brier_score": brier_score,
                "reliability_metrics": reliability_metrics,
                "y_prob_calibrated": y_prob_calibrated
            }
            
            self.logger.info(f"✅ Method {method} completed: score = {calibration_score:.4f}")
            return result
            
        except Exception as e:
            self.logger.exception(f"❌ Calibration method {method} failed: {e}")
            return None
    
    async def _calculate_reliability_metrics(self, 
                                           y_true: np.ndarray,
                                           y_prob: np.ndarray) -> Dict[str, Any]:
        """Calculate reliability diagram metrics."""
        try:
            # Create bins for reliability diagram
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0  # Expected Calibration Error
            mce = 0  # Maximum Calibration Error
            bin_accuracies = []
            bin_confidences = []
            bin_counts = []
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    
                    bin_accuracies.append(accuracy_in_bin)
                    bin_confidences.append(avg_confidence_in_bin)
                    bin_counts.append(in_bin.sum())
                    
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
            
            reliability_metrics = {
                "expected_calibration_error": ece,
                "maximum_calibration_error": mce,
                "bin_accuracies": bin_accuracies,
                "bin_confidences": bin_confidences,
                "bin_counts": bin_counts,
                "n_bins": n_bins
            }
            
            return reliability_metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Reliability metrics calculation failed: {e}")
            return {}
    
    async def _calculate_overall_metrics(self, 
                                       y_true: np.ndarray,
                                       y_prob: np.ndarray,
                                       calibration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall calibration metrics."""
        try:
            # Get best method results
            best_method = calibration_results["best_method"]
            best_result = calibration_results["calibration_methods"][best_method]
            
            # Calculate additional metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            y_pred = (best_result["y_prob_calibrated"] > 0.5).astype(int)
            
            overall_metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "calibration_score": best_result["score"],
                "brier_score": best_result["brier_score"],
                "expected_calibration_error": best_result["reliability_metrics"].get("expected_calibration_error", 0),
                "maximum_calibration_error": best_result["reliability_metrics"].get("maximum_calibration_error", 0)
            }
            
            return overall_metrics
            
        except Exception as e:
            self.logger.exception(f"❌ Overall metrics calculation failed: {e}")
            return {}
    
    @protect_data_operation(ValidationLevel.STANDARD)
    async def _validate_calibration_results(self, calibration_results: Dict[str, Any]) -> bool:
        """Validate calibration results."""
        try:
            self.logger.info("🔍 Validating calibration results...")
            
            # Check required fields
            required_fields = ["calibration_methods", "best_method", "best_score", "overall_metrics"]
            missing_fields = [field for field in required_fields if field not in calibration_results]
            if missing_fields:
                self.logger.error(f"❌ Missing required fields: {missing_fields}")
                return False
            
            # Validate best method
            if calibration_results["best_method"] not in calibration_results["calibration_methods"]:
                self.logger.error("❌ Best method not found in calibration methods")
                return False
            
            # Validate overall metrics
            overall_metrics = calibration_results["overall_metrics"]
            required_metrics = ["accuracy", "precision", "recall", "f1_score", "calibration_score"]
            missing_metrics = [metric for metric in required_metrics if metric not in overall_metrics]
            if missing_metrics:
                self.logger.error(f"❌ Missing required metrics: {missing_metrics}")
                return False
            
            # Validate metric values
            if overall_metrics["accuracy"] < 0.5:
                self.logger.warning("⚠️ Low accuracy detected in calibration results")
            
            if overall_metrics["calibration_score"] < 0.5:
                self.logger.warning("⚠️ Low calibration score detected")
            
            self.logger.info("✅ Calibration results validation passed")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Calibration results validation failed: {e}")
            return False
    
    @protect_data_operation(ValidationLevel.STANDARD, backup_enabled=True)
    async def _save_calibration_results(self, 
                                      calibration_results: Dict[str, Any],
                                      symbol: str,
                                      exchange: str,
                                      data_dir: str) -> bool:
        """Save calibration results."""
        try:
            self.logger.info("💾 Saving calibration results...")
            
            # Prepare save data
            save_data = {
                "calibration_results": calibration_results,
                "metadata": {
                    "symbol": symbol,
                    "exchange": exchange,
                    "timestamp": get_current_datetime().isoformat(),
                    "version": "enhanced_v1.0"
                }
            }
            
            # Save main results file
            results_file = f"{data_dir}/{exchange}_{symbol}_calibrated_models.pkl"
            success = self.data_access.secure_data_saving(
                save_data,
                results_file,
                user_id="confidence_calibration",
                backup_existing=True
            )
            
            if not success:
                self.logger.error(f"❌ Failed to save calibration results to {results_file}")
                return False
            
            # Save metadata file
            metadata_file = f"{data_dir}/{exchange}_{symbol}_calibration_metadata.json"
            metadata = {
                "symbol": symbol,
                "exchange": exchange,
                "timestamp": get_current_datetime().isoformat(),
                "best_method": calibration_results["best_method"],
                "best_score": calibration_results["best_score"],
                "overall_metrics": calibration_results["overall_metrics"],
                "calibration_successful": True
            }
            
            success = self.data_access.secure_data_saving(
                metadata,
                metadata_file,
                user_id="confidence_calibration",
                backup_existing=True
            )
            
            if not success:
                self.logger.error(f"❌ Failed to save calibration metadata to {metadata_file}")
                return False
            
            # Save detailed results file
            results_json_file = f"{data_dir}/{exchange}_{symbol}_calibration_results.json"
            success = self.data_access.secure_data_saving(
                calibration_results,
                results_json_file,
                user_id="confidence_calibration",
                backup_existing=True
            )
            
            if not success:
                self.logger.error(f"❌ Failed to save detailed calibration results to {results_json_file}")
                return False
            
            self.logger.info("✅ Calibration results saved successfully")
            return True
            
        except Exception as e:
            self.logger.exception(f"❌ Calibration results saving failed: {e}")
            return False