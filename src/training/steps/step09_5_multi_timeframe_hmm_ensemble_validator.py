# src/training/steps/step09_5_multi_timeframe_hmm_ensemble_validator.py

"""Validator for Step 9.5: Multi-Timeframe HMM Ensemble Training.

This validator ensures that the multi-timeframe HMM ensemble training step
produces valid outputs and meets quality standards.
"""

import json
from pathlib import Path
from typing import Any, Dict


from src.utils.logger import system_logger
from src.utils.error_handler import handle_errors
from src.utils.validator_base import BaseValidator


class Step9_5MultiTimeframeHMMEnsembleValidator(BaseValidator):
    """Validator for Step 9.5: Multi-Timeframe HMM Ensemble Training."""

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        """Initialize the validator.

        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        self.logger = system_logger.getChild("Step9_5MultiTimeframeHMMEnsembleValidator")
        self.step_name = "step09_5_multi_timeframe_hmm_ensemble"

    @handle_errors(
        exceptions=(Exception,),
        default_return={"validation_passed": False, "error": "Unknown error"},
        context="multi-timeframe HMM ensemble validation",
    )
    async def validate_step_outputs(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate Step 9.5 outputs.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path
            **kwargs: Additional arguments

        Returns:
            Dict containing validation results
        """
        try:
            self.logger.info(f"🔍 Validating Step 9.5 outputs for {symbol} on {exchange}")

            validation_results = {
                "validation_passed": True,
                "errors": [],
                "warnings": [],
                "checks_passed": 0,
                "checks_failed": 0,
            }

            # Check 1: Validate ensemble model files exist
            models_dir = Path("models") / "multi_timeframe_hmm_ensemble" / f"{exchange}_{symbol}"
            
            required_files = [
                "ensemble_metadata.json",
                "meta_learner.joblib",
            ]
            
            for file in required_files:
                file_path = models_dir / file
                if file_path.exists():
                    validation_results["checks_passed"] += 1
                    self.logger.info(f"✅ Found required file: {file}")
                else:
                    validation_results["checks_failed"] += 1
                    validation_results["errors"].append(f"Missing required file: {file}")
                    self.logger.error(f"❌ Missing required file: {file}")

            # Check 2: Validate ensemble metadata
            metadata_path = models_dir / "ensemble_metadata.json"
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Validate metadata structure
                    required_keys = ["trained", "ensemble_weights", "symbol", "exchange"]
                    missing_keys = [key for key in required_keys if key not in metadata]
                    
                    if not missing_keys:
                        validation_results["checks_passed"] += 1
                        self.logger.info("✅ Ensemble metadata structure is valid")
                    else:
                        validation_results["checks_failed"] += 1
                        validation_results["errors"].append(f"Missing metadata keys: {missing_keys}")
                        self.logger.error(f"❌ Missing metadata keys: {missing_keys}")
                    
                    # Validate training status
                    if metadata.get("trained", False):
                        validation_results["checks_passed"] += 1
                        self.logger.info("✅ Ensemble is marked as trained")
                    else:
                        validation_results["checks_failed"] += 1
                        validation_results["errors"].append("Ensemble not marked as trained")
                        self.logger.error("❌ Ensemble not marked as trained")
                    
                    # Validate ensemble weights
                    ensemble_weights = metadata.get("ensemble_weights", {})
                    if ensemble_weights:
                        total_weight = sum(ensemble_weights.values())
                        if abs(total_weight - 1.0) < 0.01:
                            validation_results["checks_passed"] += 1
                            self.logger.info("✅ Ensemble weights sum to 1.0")
                        else:
                            validation_results["checks_failed"] += 1
                            validation_results["warnings"].append(f"Ensemble weights don't sum to 1.0: {total_weight}")
                            self.logger.warning(f"⚠️ Ensemble weights don't sum to 1.0: {total_weight}")
                    else:
                        validation_results["checks_failed"] += 1
                        validation_results["errors"].append("No ensemble weights found")
                        self.logger.error("❌ No ensemble weights found")
                        
                except Exception as e:
                    validation_results["checks_failed"] += 1
                    validation_results["errors"].append(f"Failed to parse metadata: {str(e)}")
                    self.logger.error(f"❌ Failed to parse metadata: {e}")

            # Check 3: Validate meta-learner model
            meta_learner_path = models_dir / "meta_learner.joblib"
            if meta_learner_path.exists():
                try:
                    import joblib
                    meta_learner = joblib.load(meta_learner_path)
                    
                    # Check if it has required methods
                    if hasattr(meta_learner, 'predict') and hasattr(meta_learner, 'predict_proba'):
                        validation_results["checks_passed"] += 1
                        self.logger.info("✅ Meta-learner has required methods")
                    else:
                        validation_results["checks_failed"] += 1
                        validation_results["errors"].append("Meta-learner missing required methods")
                        self.logger.error("❌ Meta-learner missing required methods")
                        
                except Exception as e:
                    validation_results["checks_failed"] += 1
                    validation_results["errors"].append(f"Failed to load meta-learner: {str(e)}")
                    self.logger.error(f"❌ Failed to load meta-learner: {e}")

            # Check 4: Validate configuration consistency
            try:
                from src.config.multi_timeframe_hmm_ensemble_config import (
                    get_multi_timeframe_hmm_ensemble_config,
                )
                
                config = get_multi_timeframe_hmm_ensemble_config()
                ensemble_config = config.get("MULTI_TIMEFRAME_HMM_ENSEMBLE", {})
                
                if ensemble_config.get("enabled", False):
                    validation_results["checks_passed"] += 1
                    self.logger.info("✅ Multi-timeframe HMM ensemble is enabled in config")
                else:
                    validation_results["checks_failed"] += 1
                    validation_results["warnings"].append("Multi-timeframe HMM ensemble is disabled in config")
                    self.logger.warning("⚠️ Multi-timeframe HMM ensemble is disabled in config")
                    
            except Exception as e:
                validation_results["checks_failed"] += 1
                validation_results["errors"].append(f"Failed to load configuration: {str(e)}")
                self.logger.error(f"❌ Failed to load configuration: {e}")

            # Determine overall validation result
            if validation_results["checks_failed"] == 0:
                validation_results["validation_passed"] = True
                self.logger.info(f"✅ Step 9.5 validation passed: {validation_results['checks_passed']} checks passed")
            else:
                validation_results["validation_passed"] = False
                self.logger.error(f"❌ Step 9.5 validation failed: {validation_results['checks_failed']} checks failed")

            return validation_results

        except Exception as e:
            self.logger.exception(f"❌ Step 9.5 validation failed: {e}")
            return {
                "validation_passed": False,
                "error": str(e),
                "checks_passed": 0,
                "checks_failed": 1,
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return={"validation_passed": False, "error": "Unknown error"},
        context="multi-timeframe HMM ensemble data validation",
    )
    async def validate_input_data(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate input data for Step 9.5.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path
            **kwargs: Additional arguments

        Returns:
            Dict containing validation results
        """
        try:
            self.logger.info(f"🔍 Validating Step 9.5 input data for {symbol} on {exchange}")

            validation_results = {
                "validation_passed": True,
                "errors": [],
                "warnings": [],
                "checks_passed": 0,
                "checks_failed": 0,
            }

            # Check 1: Validate regime forecasting directory exists
            rf_dir = Path(data_dir) / "regime_forecasting"
            if rf_dir.exists():
                validation_results["checks_passed"] += 1
                self.logger.info("✅ Regime forecasting directory exists")
            else:
                validation_results["checks_failed"] += 1
                validation_results["errors"].append("Regime forecasting directory not found")
                self.logger.error("❌ Regime forecasting directory not found")

            # Check 2: Validate regime forecasting files exist for expected timeframes
            expected_timeframes = ["5m", "15m", "30m", "1h"]
            found_timeframes = []
            
            for tf in expected_timeframes:
                rf_file = rf_dir / f"{exchange}_{symbol}_{tf}_regime_forecasting.json"
                if rf_file.exists():
                    found_timeframes.append(tf)
                    validation_results["checks_passed"] += 1
                    self.logger.info(f"✅ Found regime forecasting file for {tf}")
                else:
                    validation_results["checks_failed"] += 1
                    validation_results["warnings"].append(f"Missing regime forecasting file for {tf}")
                    self.logger.warning(f"⚠️ Missing regime forecasting file for {tf}")

            # Check 3: Validate regime forecasting file structure
            for tf in found_timeframes:
                rf_file = rf_dir / f"{exchange}_{symbol}_{tf}_regime_forecasting.json"
                try:
                    with open(rf_file, 'r') as f:
                        rf_data = json.load(f)
                    
                    # Check required keys
                    required_keys = ["timeframe", "current_regime", "next_regime_probabilities"]
                    missing_keys = [key for key in required_keys if key not in rf_data]
                    
                    if not missing_keys:
                        validation_results["checks_passed"] += 1
                        self.logger.info(f"✅ Regime forecasting file structure valid for {tf}")
                    else:
                        validation_results["checks_failed"] += 1
                        validation_results["errors"].append(f"Missing keys in {tf} regime forecasting: {missing_keys}")
                        self.logger.error(f"❌ Missing keys in {tf} regime forecasting: {missing_keys}")
                        
                except Exception as e:
                    validation_results["checks_failed"] += 1
                    validation_results["errors"].append(f"Failed to parse {tf} regime forecasting: {str(e)}")
                    self.logger.error(f"❌ Failed to parse {tf} regime forecasting: {e}")

            # Check 4: Validate minimum data requirements
            if len(found_timeframes) >= 2:
                validation_results["checks_passed"] += 1
                self.logger.info(f"✅ Sufficient timeframes found: {found_timeframes}")
            else:
                validation_results["checks_failed"] += 1
                validation_results["errors"].append(f"Insufficient timeframes: {found_timeframes}")
                self.logger.error(f"❌ Insufficient timeframes: {found_timeframes}")

            # Determine overall validation result
            if validation_results["checks_failed"] == 0:
                validation_results["validation_passed"] = True
                self.logger.info(f"✅ Step 9.5 input validation passed: {validation_results['checks_passed']} checks passed")
            else:
                validation_results["validation_passed"] = False
                self.logger.error(f"❌ Step 9.5 input validation failed: {validation_results['checks_failed']} checks failed")

            return validation_results

        except Exception as e:
            self.logger.exception(f"❌ Step 9.5 input validation failed: {e}")
            return {
                "validation_passed": False,
                "error": str(e),
                "checks_passed": 0,
                "checks_failed": 1,
            }

    @handle_errors(
        exceptions=(Exception,),
        default_return={"validation_passed": False, "error": "Unknown error"},
        context="multi-timeframe HMM ensemble performance validation",
    )
    async def validate_performance(
        self,
        symbol: str,
        exchange: str,
        data_dir: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Validate performance metrics for Step 9.5.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            data_dir: Data directory path
            **kwargs: Additional arguments

        Returns:
            Dict containing validation results
        """
        try:
            self.logger.info(f"🔍 Validating Step 9.5 performance for {symbol} on {exchange}")

            validation_results = {
                "validation_passed": True,
                "errors": [],
                "warnings": [],
                "checks_passed": 0,
                "checks_failed": 0,
                "performance_metrics": {},
            }

            # Check 1: Validate ensemble model performance
            models_dir = Path("models") / "multi_timeframe_hmm_ensemble" / f"{exchange}_{symbol}"
            metadata_path = models_dir / "ensemble_metadata.json"
            
            if metadata_path.exists():
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Check training time
                    training_time = metadata.get("training_time", 0)
                    if training_time > 0 and training_time < 3600:  # Less than 1 hour
                        validation_results["checks_passed"] += 1
                        validation_results["performance_metrics"]["training_time"] = training_time
                        self.logger.info(f"✅ Training time is reasonable: {training_time:.2f}s")
                    else:
                        validation_results["checks_failed"] += 1
                        validation_results["warnings"].append(f"Training time may be too long: {training_time:.2f}s")
                        self.logger.warning(f"⚠️ Training time may be too long: {training_time:.2f}s")
                    
                    # Check ensemble weights distribution
                    ensemble_weights = metadata.get("ensemble_weights", {})
                    if ensemble_weights:
                        weight_values = list(ensemble_weights.values())
                        min_weight = min(weight_values)
                        max_weight = max(weight_values)
                        
                        if max_weight - min_weight < 0.5:  # Reasonable weight distribution
                            validation_results["checks_passed"] += 1
                            validation_results["performance_metrics"]["weight_distribution"] = {
                                "min": min_weight,
                                "max": max_weight,
                                "range": max_weight - min_weight,
                            }
                            self.logger.info(f"✅ Weight distribution is reasonable: {min_weight:.3f} - {max_weight:.3f}")
                        else:
                            validation_results["checks_failed"] += 1
                            validation_results["warnings"].append(f"Weight distribution may be imbalanced: {min_weight:.3f} - {max_weight:.3f}")
                            self.logger.warning(f"⚠️ Weight distribution may be imbalanced: {min_weight:.3f} - {max_weight:.3f}")
                            
                except Exception as e:
                    validation_results["checks_failed"] += 1
                    validation_results["errors"].append(f"Failed to validate performance: {str(e)}")
                    self.logger.error(f"❌ Failed to validate performance: {e}")

            # Determine overall validation result
            if validation_results["checks_failed"] == 0:
                validation_results["validation_passed"] = True
                self.logger.info(f"✅ Step 9.5 performance validation passed: {validation_results['checks_passed']} checks passed")
            else:
                validation_results["validation_passed"] = False
                self.logger.error(f"❌ Step 9.5 performance validation failed: {validation_results['checks_failed']} checks failed")

            return validation_results

        except Exception as e:
            self.logger.exception(f"❌ Step 9.5 performance validation failed: {e}")
            return {
                "validation_passed": False,
                "error": str(e),
                "checks_passed": 0,
                "checks_failed": 1,
            }