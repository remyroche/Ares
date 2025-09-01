# src/tactician/ml_target_validator.py

"""
ML Target Validator for validating machine learning targets and predictions.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.error_handler import handle_errors
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
    failed,
    invalid,
    validation_error,
)
from src.utils.centralized_decorators import validate_data_quality


class MLTargetValidator:
    passpass"""
    Enhanced ML Target Validator component with DI, type hints, and robust error handling.
    """

    def __init__(...) -> ...:
    pass"""..."""
    passself.config = config
        self.logger = system_logger.getChild("MLTargetValidator")

        # State tracking
        self.is_running: bool = False
        self.status: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

        # Configuration
        self.validator_config: Dict[str, Any] = self.config.get("ml_target_validator", {})
        self.validation_interval: int = self.validator_config.get("validation_interval", 60)
        self.max_history: int = self.validator_config.get("max_history", 100)

        # Validation thresholds
        self.min_confidence_threshold = self.validator_config.get("min_confidence_threshold", 0.1)
        self.max_confidence_threshold = self.validator_config.get("max_confidence_threshold", 0.9)
        self.min_target_value = self.validator_config.get("min_target_value", -1.0)
        self.max_target_value = self.validator_config.get("max_target_value", 1.0)

    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="ML target validator initialization"
    )
    async def initialize(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            self.logger.info("Initializing ML Target Validator...")

            # Validate configuration
            if not self._validate_configuration():
    passself.logger.error("Invalid ML target validator configuration")
                return False

            # Clear history
            self.history.clear()
            self.status.clear()

            self.logger.info("✅ ML Target Validator initialized successfully")
            return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ ML Target Validator initialization failed: {e}")
            return False

    def _validate_configuration(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            if self.validation_interval <= 0:
    passself.logger.error("Validation interval must be positive")
                return False

            if self.max_history <= 0:
    passself.logger.error("Max history must be positive")
                return False

            if not (0 <= self.min_confidence_threshold <= self.max_confidence_threshold <= 1):
    passself.logger.error("Confidence thresholds must be between 0 and 1")
                return False

            if self.min_target_value >= self.max_target_value:
    passself.logger.error("Min target value must be less than max target value")
                return False

            return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Configuration validation failed: {e}")
            return False

    @validate_data_quality(
        required_columns=None,  # This method validates dict input, not DataFrame
        min_rows=1,
        max_null_ratio=0.0,
        check_duplicates=False,
        check_timestamps=False,
        context="ML target validation"
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="target validation"
    )
    async def validate_target(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            self.logger.info("Validating ML target...")

            # Extract target components
            target_value = target_data.get("target_value")
            confidence = target_data.get("confidence", 0.0)
            timestamp = target_data.get("timestamp")
            symbol = target_data.get("symbol", "unknown")

            # Validate target value
            if target_value is None:
    passself.logger.error(validation_error("Target value is missing"))
                return False

            if not isinstance(target_value, (int, float)):
    passself.logger.error(validation_error("Target value must be numeric"))
                return False

            if not self.min_target_value <= target_value <= self.max_target_value:
    passself.logger.error(
                    validation_error(
                        f"Target value {target_value} outside valid range [{self.min_target_value}, {self.max_target_value}]"
                    )
                )
                return False

            # Validate confidence
            if not isinstance(confidence, (int, float)):
    passself.logger.error(validation_error("Confidence must be numeric"))
                return False

            if not 0 <= confidence <= 1:
    passself.logger.error(validation_error("Confidence must be between 0 and 1"))
                return False

            if not self.min_confidence_threshold <= confidence <= self.max_confidence_threshold:
    passself.logger.warning(
                    f"Confidence {confidence:.3f} outside preferred range [{self.min_confidence_threshold}, {self.max_confidence_threshold}]"
                )

            # Validate timestamp
            if timestamp:
    passtry:
    passdatetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
    passpassself.logger.error(validation_error("Invalid timestamp format"))
                    return False

            # Record validation
            validation_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "target_value": target_value,
                "confidence": confidence,
                "is_valid": True,
                "validation_notes": "Target validation passed"
            }

            self._add_to_history(validation_record)

            self.logger.info(f"✅ Target validation passed for {symbol}: {target_value:.4f} (confidence: {confidence:.3f})")
            return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Target validation failed: {e}")

            # Record failed validation
            validation_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": target_data.get("symbol", "unknown"),
                "target_value": target_data.get("target_value"),
                "confidence": target_data.get("confidence"),
                "is_valid": False,
                "validation_notes": f"Validation failed: {e}"
            }

            self._add_to_history(validation_record)
            return False

    @validate_data_quality(
        required_columns=None,  # This method validates dict input, not DataFrame
        min_rows=1,
        max_null_ratio=0.0,
        check_duplicates=False,
        check_timestamps=False,
        context="ML prediction validation"
    )
    @handle_errors(
        exceptions=(ValueError, AttributeError),
        default_return=False,
        context="prediction validation"
    )
    async def validate_prediction(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            self.logger.info("Validating ML prediction...")

            # Extract prediction components
            prediction_value = prediction_data.get("prediction_value")
            confidence = prediction_data.get("confidence", 0.0)
            model_name = prediction_data.get("model_name", "unknown")
            symbol = prediction_data.get("symbol", "unknown")

            # Validate prediction value
            if prediction_value is None:
    passself.logger.error(validation_error("Prediction value is missing"))
                return False

            if not isinstance(prediction_value, (int, float)):
    passself.logger.error(validation_error("Prediction value must be numeric"))
                return False

            if not self.min_target_value <= prediction_value <= self.max_target_value:
    passself.logger.error(
                    validation_error(
                        f"Prediction value {prediction_value} outside valid range [{self.min_target_value}, {self.max_target_value}]"
                    )
                )
                return False

            # Validate confidence
            if not isinstance(confidence, (int, float)):
    passself.logger.error(validation_error("Confidence must be numeric"))
                return False

            if not 0 <= confidence <= 1:
    passself.logger.error(validation_error("Confidence must be between 0 and 1"))
                return False

            # Validate model name
            if not model_name or not isinstance(model_name, str):
    passself.logger.error(validation_error("Model name is required"))
                return False

            # Record validation
            validation_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "model_name": model_name,
                "prediction_value": prediction_value,
                "confidence": confidence,
                "is_valid": True,
                "validation_notes": "Prediction validation passed"
            }

            self._add_to_history(validation_record)

            self.logger.info(
                f"✅ Prediction validation passed for {symbol} ({model_name}): {prediction_value:.4f} (confidence: {confidence:.3f})"
            )
            return True

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Prediction validation failed: {e}")

            # Record failed validation
            validation_record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": prediction_data.get("symbol", "unknown"),
                "model_name": prediction_data.get("model_name", "unknown"),
                "prediction_value": prediction_data.get("prediction_value"),
                "confidence": prediction_data.get("confidence"),
                "is_valid": False,
                "validation_notes": f"Validation failed: {e}"
            }

            self._add_to_history(validation_record)
            return False

    def _add_to_history(...) -> ...:
    """..."""
    passtry:
    passself.history.append(record)

            # Maintain history size limit
            if len(self.history) > self.max_history:
    passself.history = self.history[-self.max_history:]

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Error adding to history: {e}")

    def get_validation_history(...) -> ...:
    """..."""
    passtry:
    passif limit:
    passreturn self.history[-limit:]
            return self.history.copy()

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Error getting validation history: {e}")
            return []

    def get_validation_statistics(...) -> ...:
    """..."""
    passtry:
    passself.logger.error(f"Error in {file_path}: {{e}}")
except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"Error in {file_path}: {{e}}")
            if not self.history:
    passreturn {
                    "total_validations": 0,
                    "valid_count": 0,
                    "invalid_count": 0,
                    "success_rate": 0.0,
                    "average_confidence": 0.0
                }

            total_validations = len(self.history)
            valid_count = sum(1 for record in self.history if record.get("is_valid", False))
            invalid_count = total_validations - valid_count
            success_rate = valid_count / total_validations if total_validations > 0 else 0.0

            # Calculate average confidence
            confidences = [record.get("confidence", 0.0) for record in self.history if record.get("confidence") is not None]
            average_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            return {
                "total_validations": total_validations,
                "valid_count": valid_count,
                "invalid_count": invalid_count,
                "success_rate": success_rate,
                "average_confidence": average_confidence
            }

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ Error calculating validation statistics: {e}")
            return {}

    def get_status(...) -> ...:
    """..."""
    passreturn {
            "is_running": self.is_running,
            "validation_interval": self.validation_interval,
            "history_size": len(self.history),
            "max_history": self.max_history,
            "statistics": self.get_validation_statistics()
        }

    async def cleanup(...) -> ...:
    """..."""
    passtry:
    passself.logger.info("Cleaning up ML Target Validator...")

            # Clear history
            self.history.clear()
            self.status.clear()

            self.logger.info("✅ ML Target Validator cleanup completed")

        except Exception as e:
    passpasspasspasspasspasspassself.logger.error(f"❌ ML Target Validator cleanup failed: {e}")
