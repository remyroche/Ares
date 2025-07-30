import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import warnings

from src.analyst.ml_dynamic_target_predictor import MLDynamicTargetPredictor
from src.utils.logger import system_logger


class MLTargetValidator:
    """
    Validates ML target predictions and ensures robust fallback mechanisms.
    
    This class performs:
    - Sanity checks on ML predictions
    - Performance monitoring of ML models
    - Automatic fallback when predictions are unreliable
    - Model health monitoring and alerts
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("ml_target_validator", {})
        self.logger = system_logger.getChild('MLTargetValidator')
        
        # Validation thresholds
        self.max_reasonable_tp_multiplier = self.config.get("max_reasonable_tp_multiplier", 10.0)
        self.min_reasonable_tp_multiplier = self.config.get("min_reasonable_tp_multiplier", 0.1)
        self.max_reasonable_sl_multiplier = self.config.get("max_reasonable_sl_multiplier", 5.0)
        self.min_reasonable_sl_multiplier = self.config.get("min_reasonable_sl_multiplier", 0.05)
        
        # Model performance tracking
        self.min_confidence_threshold = self.config.get("min_confidence_threshold", 0.3)
        self.performance_window_size = self.config.get("performance_window_size", 100)
        self.max_consecutive_failures = self.config.get("max_consecutive_failures", 5)
        
        # Fallback configuration
        self.enable_automatic_fallback = self.config.get("enable_automatic_fallback", True)
        self.fallback_tp_multiplier = self.config.get("fallback_tp_multiplier", 2.0)
        self.fallback_sl_multiplier = self.config.get("fallback_sl_multiplier", 0.5)
        
        # Performance tracking state
        self.prediction_history = []
        self.consecutive_failures = 0
        self.model_health_status = "healthy"
        self.last_health_check = None
        
    def validate_prediction(self, 
                          prediction: Dict[str, Any], 
                          signal_type: str,
                          current_atr: float,
                          market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an ML target prediction and return corrected version if needed.
        
        Args:
            prediction: ML prediction result
            signal_type: Signal type (SR_FADE_LONG, etc.)
            current_atr: Current ATR value
            market_data: Current market data
            
        Returns:
            Validated and potentially corrected prediction
        """
        try:
            validation_result = {
                "is_valid": True,
                "corrected_prediction": prediction.copy(),
                "validation_issues": [],
                "confidence_score": prediction.get("prediction_confidence", 0.0),
                "used_fallback": False
            }
            
            # Basic sanity checks
            issues = self._perform_sanity_checks(prediction, current_atr, market_data)
            validation_result["validation_issues"].extend(issues)
            
            # Check confidence level
            confidence = prediction.get("prediction_confidence", 0.0)
            if confidence < self.min_confidence_threshold:
                issues.append(f"Low confidence: {confidence:.2f}")
                validation_result["is_valid"] = False
            
            # Validate multipliers
            tp_mult = prediction.get("tp_multiplier", 0)
            sl_mult = prediction.get("sl_multiplier", 0)
            
            if not self._is_multiplier_reasonable(tp_mult, "tp"):
                issues.append(f"Unreasonable TP multiplier: {tp_mult:.2f}")
                validation_result["is_valid"] = False
                
            if not self._is_multiplier_reasonable(sl_mult, "sl"):
                issues.append(f"Unreasonable SL multiplier: {sl_mult:.2f}")
                validation_result["is_valid"] = False
            
            # Check for prediction consistency
            consistency_issues = self._check_prediction_consistency(prediction, signal_type)
            validation_result["validation_issues"].extend(consistency_issues)
            if consistency_issues:
                validation_result["is_valid"] = False
            
            # Apply corrections if needed
            if not validation_result["is_valid"] and self.enable_automatic_fallback:
                corrected = self._apply_fallback_correction(
                    prediction, signal_type, current_atr, market_data
                )
                validation_result["corrected_prediction"] = corrected
                validation_result["used_fallback"] = True
                validation_result["is_valid"] = True  # Fallback is always valid
                
                self.logger.warning(
                    f"Applied fallback correction for {signal_type}. "
                    f"Issues: {', '.join(issues[:3])}..."  # Show first 3 issues
                )
            
            # Track prediction for performance monitoring
            self._track_prediction_performance(validation_result)
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating prediction: {e}", exc_info=True)
            return self._get_emergency_fallback(signal_type, current_atr, market_data)
    
    def _perform_sanity_checks(self, 
                              prediction: Dict[str, Any],
                              current_atr: float,
                              market_data: Dict[str, Any]) -> List[str]:
        """Perform basic sanity checks on prediction."""
        issues = []
        
        # Check for required fields
        required_fields = ["take_profit", "stop_loss", "tp_multiplier", "sl_multiplier"]
        for field in required_fields:
            if field not in prediction or prediction[field] is None:
                issues.append(f"Missing required field: {field}")
        
        # Check for NaN or infinite values
        for field in required_fields:
            value = prediction.get(field, 0)
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    issues.append(f"Invalid numeric value in {field}: {value}")
        
        # Check price relationships
        entry_price = prediction.get("entry_price", 0)
        take_profit = prediction.get("take_profit", 0)
        stop_loss = prediction.get("stop_loss", 0)
        current_price = market_data.get("current_price", 0)
        
        if entry_price > 0 and take_profit > 0 and stop_loss > 0:
            # For long positions: TP > entry > SL
            # For short positions: SL > entry > TP
            if take_profit > entry_price and stop_loss < entry_price:
                # Looks like a long position - this is correct
                pass
            elif take_profit < entry_price and stop_loss > entry_price:
                # Looks like a short position - this is correct
                pass
            else:
                issues.append("Inconsistent price relationships (TP/SL/Entry)")
        
        # Check ATR consistency
        if current_atr > 0:
            tp_mult = prediction.get("tp_multiplier", 0)
            sl_mult = prediction.get("sl_multiplier", 0)
            
            if tp_mult > 0 and abs(take_profit - entry_price) > 0:
                implied_atr_tp = abs(take_profit - entry_price) / tp_mult
                if abs(implied_atr_tp - current_atr) / current_atr > 0.5:  # 50% tolerance
                    issues.append(f"ATR inconsistency in TP calculation")
                    
            if sl_mult > 0 and abs(stop_loss - entry_price) > 0:
                implied_atr_sl = abs(stop_loss - entry_price) / sl_mult
                if abs(implied_atr_sl - current_atr) / current_atr > 0.5:  # 50% tolerance
                    issues.append(f"ATR inconsistency in SL calculation")
        
        return issues
    
    def _is_multiplier_reasonable(self, multiplier: float, mult_type: str) -> bool:
        """Check if a multiplier value is within reasonable bounds."""
        if mult_type == "tp":
            return (self.min_reasonable_tp_multiplier <= multiplier <= 
                   self.max_reasonable_tp_multiplier)
        elif mult_type == "sl":
            return (self.min_reasonable_sl_multiplier <= multiplier <= 
                   self.max_reasonable_sl_multiplier)
        return False
    
    def _check_prediction_consistency(self, 
                                    prediction: Dict[str, Any], 
                                    signal_type: str) -> List[str]:
        """Check prediction consistency with signal type."""
        issues = []
        
        entry_price = prediction.get("entry_price", 0)
        take_profit = prediction.get("take_profit", 0)
        stop_loss = prediction.get("stop_loss", 0)
        
        if entry_price <= 0 or take_profit <= 0 or stop_loss <= 0:
            return issues  # Skip consistency checks if prices are invalid
        
        # Check direction consistency
        if "LONG" in signal_type:
            if take_profit <= entry_price:
                issues.append("Long signal but TP <= entry price")
            if stop_loss >= entry_price:
                issues.append("Long signal but SL >= entry price")
                
        elif "SHORT" in signal_type:
            if take_profit >= entry_price:
                issues.append("Short signal but TP >= entry price")
            if stop_loss <= entry_price:
                issues.append("Short signal but SL <= entry price")
        
        # Check risk-reward ratio
        if "LONG" in signal_type:
            potential_profit = take_profit - entry_price
            potential_loss = entry_price - stop_loss
        else:
            potential_profit = entry_price - take_profit
            potential_loss = stop_loss - entry_price
        
        if potential_loss > 0:
            risk_reward_ratio = potential_profit / potential_loss
            if risk_reward_ratio < 0.5:  # Less than 0.5:1 RR
                issues.append(f"Poor risk-reward ratio: {risk_reward_ratio:.2f}")
            elif risk_reward_ratio > 20:  # More than 20:1 RR (unrealistic)
                issues.append(f"Unrealistic risk-reward ratio: {risk_reward_ratio:.2f}")
        
        return issues
    
    def _apply_fallback_correction(self, 
                                  prediction: Dict[str, Any],
                                  signal_type: str,
                                  current_atr: float,
                                  market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply fallback correction to invalid prediction."""
        corrected = prediction.copy()
        
        entry_price = prediction.get("entry_price", market_data.get("current_price", 0))
        
        # Use fallback multipliers
        if "LONG" in signal_type:
            corrected["take_profit"] = entry_price + (current_atr * self.fallback_tp_multiplier)
            corrected["stop_loss"] = entry_price - (current_atr * self.fallback_sl_multiplier)
        else:  # SHORT
            corrected["take_profit"] = entry_price - (current_atr * self.fallback_tp_multiplier)
            corrected["stop_loss"] = entry_price + (current_atr * self.fallback_sl_multiplier)
        
        corrected["tp_multiplier"] = self.fallback_tp_multiplier
        corrected["sl_multiplier"] = self.fallback_sl_multiplier
        corrected["prediction_confidence"] = 0.1  # Low confidence for fallback
        
        return corrected
    
    def _get_emergency_fallback(self, 
                               signal_type: str,
                               current_atr: float,
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get emergency fallback when validation completely fails."""
        current_price = market_data.get("current_price", 0)
        
        if "LONG" in signal_type:
            take_profit = current_price + (current_atr * self.fallback_tp_multiplier)
            stop_loss = current_price - (current_atr * self.fallback_sl_multiplier)
        else:
            take_profit = current_price - (current_atr * self.fallback_tp_multiplier)
            stop_loss = current_price + (current_atr * self.fallback_sl_multiplier)
        
        return {
            "is_valid": True,
            "corrected_prediction": {
                "take_profit": take_profit,
                "stop_loss": stop_loss,
                "tp_multiplier": self.fallback_tp_multiplier,
                "sl_multiplier": self.fallback_sl_multiplier,
                "entry_price": current_price,
                "prediction_confidence": 0.0
            },
            "validation_issues": ["Emergency fallback - validation failed"],
            "confidence_score": 0.0,
            "used_fallback": True
        }
    
    def _track_prediction_performance(self, validation_result: Dict[str, Any]):
        """Track prediction performance for model health monitoring."""
        performance_record = {
            "timestamp": datetime.now(),
            "is_valid": validation_result["is_valid"],
            "confidence": validation_result["confidence_score"],
            "used_fallback": validation_result["used_fallback"],
            "issue_count": len(validation_result["validation_issues"])
        }
        
        self.prediction_history.append(performance_record)
        
        # Keep only recent history
        if len(self.prediction_history) > self.performance_window_size:
            self.prediction_history = self.prediction_history[-self.performance_window_size:]
        
        # Update consecutive failures counter
        if not validation_result["is_valid"]:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # Update model health status
        self._update_model_health_status()
    
    def _update_model_health_status(self):
        """Update overall model health status."""
        previous_status = self.model_health_status
        
        # Check consecutive failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            self.model_health_status = "critical"
        elif len(self.prediction_history) >= 10:
            # Check recent performance
            recent_records = self.prediction_history[-10:]
            valid_predictions = sum(1 for r in recent_records if r["is_valid"])
            fallback_usage = sum(1 for r in recent_records if r["used_fallback"])
            avg_confidence = np.mean([r["confidence"] for r in recent_records])
            
            if valid_predictions < 5:  # Less than 50% valid
                self.model_health_status = "poor"
            elif fallback_usage > 7:  # More than 70% fallback usage
                self.model_health_status = "degraded"
            elif avg_confidence < 0.4:
                self.model_health_status = "low_confidence"
            else:
                self.model_health_status = "healthy"
        
        # Log status changes
        if previous_status != self.model_health_status:
            self.logger.warning(
                f"ML model health status changed: {previous_status} -> {self.model_health_status}"
            )
        
        self.last_health_check = datetime.now()
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report of ML target prediction system."""
        if not self.prediction_history:
            return {
                "status": "unknown",
                "message": "No prediction history available"
            }
        
        recent_records = self.prediction_history[-min(50, len(self.prediction_history)):]
        
        total_predictions = len(recent_records)
        valid_predictions = sum(1 for r in recent_records if r["is_valid"])
        fallback_usage = sum(1 for r in recent_records if r["used_fallback"])
        avg_confidence = np.mean([r["confidence"] for r in recent_records])
        avg_issues_per_prediction = np.mean([r["issue_count"] for r in recent_records])
        
        return {
            "overall_status": self.model_health_status,
            "last_health_check": self.last_health_check,
            "consecutive_failures": self.consecutive_failures,
            "statistics": {
                "total_predictions": total_predictions,
                "valid_predictions": valid_predictions,
                "validation_rate": valid_predictions / total_predictions if total_predictions > 0 else 0,
                "fallback_usage": fallback_usage,
                "fallback_rate": fallback_usage / total_predictions if total_predictions > 0 else 0,
                "average_confidence": avg_confidence,
                "average_issues_per_prediction": avg_issues_per_prediction
            },
            "recommendations": self._generate_health_recommendations()
        }
    
    def _generate_health_recommendations(self) -> List[str]:
        """Generate recommendations based on current health status."""
        recommendations = []
        
        if self.model_health_status == "critical":
            recommendations.extend([
                "ML model experiencing critical failures - consider emergency model retrain",
                "Verify training data quality and feature engineering",
                "Check for data drift or market regime changes"
            ])
        elif self.model_health_status == "poor":
            recommendations.extend([
                "Model validation rate is low - schedule model retraining",
                "Review recent market conditions for data drift",
                "Consider adjusting validation thresholds"
            ])
        elif self.model_health_status == "degraded":
            recommendations.extend([
                "High fallback usage detected - model may need retraining",
                "Check if market conditions have changed significantly"
            ])
        elif self.model_health_status == "low_confidence":
            recommendations.extend([
                "Model confidence is consistently low",
                "Consider expanding training dataset or improving features"
            ])
        
        if self.consecutive_failures > 2:
            recommendations.append(
                f"Consider temporary increase in fallback thresholds "
                f"({self.consecutive_failures} consecutive failures)"
            )
        
        return recommendations
    
    def should_trigger_retraining(self) -> bool:
        """Determine if model retraining should be triggered based on performance."""
        if len(self.prediction_history) < 20:
            return False  # Not enough data
        
        recent_records = self.prediction_history[-20:]
        validation_rate = sum(1 for r in recent_records if r["is_valid"]) / 20
        avg_confidence = np.mean([r["confidence"] for r in recent_records])
        
        return (
            self.model_health_status in ["critical", "poor"] or
            validation_rate < 0.3 or
            avg_confidence < 0.2 or
            self.consecutive_failures >= self.max_consecutive_failures
        )