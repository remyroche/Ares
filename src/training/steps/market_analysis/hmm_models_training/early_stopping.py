"""
Early Stopping and Aggressive Overfitting Detection

Enhanced overfitting detection with early stopping mechanisms for HMM training.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

logger = logging.getLogger(__name__)

@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping and overfitting detection."""
    
    # Early stopping parameters
    patience: int = 5
    min_delta: float = 0.001
    monitor_metric: str = 'validation_loss'
    mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    
    # Overfitting detection thresholds (more aggressive)
    accuracy_gap_threshold: float = 0.05  # 5% gap triggers warning
    severe_accuracy_gap_threshold: float = 0.15  # 15% gap triggers early stopping
    f1_gap_threshold: float = 0.03  # 3% F1 gap triggers warning
    severe_f1_gap_threshold: float = 0.10  # 10% F1 gap triggers early stopping
    
    # Confidence-based overfitting detection
    confidence_gap_threshold: float = 0.1  # 10% confidence gap
    overconfident_ratio_threshold: float = 0.3  # 30% overconfident predictions
    
    # Feature-based overfitting detection
    feature_concentration_threshold: float = 0.8  # 80% of importance in top features
    correlation_threshold: float = 0.95  # High correlation indicates overfitting
    
    # Cross-validation overfitting detection
    cv_variance_threshold: float = 0.05  # 5% CV variance threshold
    cv_test_gap_threshold: float = 0.08  # 8% gap between CV and test
    
    # Early stopping enabled
    enable_early_stopping: bool = True
    enable_aggressive_detection: bool = True

class EarlyStoppingMonitor:
    """Monitor for early stopping and overfitting detection."""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.best_score = float('inf') if config.mode == 'min' else float('-inf')
        self.patience_counter = 0
        self.overfitting_warnings = []
        self.early_stop_triggered = False
        
    def should_stop(self, current_score: float, train_metrics: Dict, val_metrics: Dict) -> Tuple[bool, str]:
        """
        Check if training should stop early.
        
        Args:
            current_score: Current validation score
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            Tuple[bool, str]: (should_stop, reason)
        """
        # Check for overfitting first
        overfitting_detected, overfitting_reason = self._detect_overfitting(train_metrics, val_metrics)
        if overfitting_detected and self.config.enable_aggressive_detection:
            self.early_stop_triggered = True
            return True, f"Early stopping due to overfitting: {overfitting_reason}"
        
        # Standard early stopping logic
        if not self.config.enable_early_stopping:
            return False, ""
            
        if self.config.mode == 'min':
            improved = current_score < self.best_score - self.config.min_delta
        else:
            improved = current_score > self.best_score + self.config.min_delta
            
        if improved:
            self.best_score = current_score
            self.patience_counter = 0
            return False, ""
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.config.patience:
                return True, f"Early stopping: no improvement for {self.config.patience} epochs"
                
        return False, ""
    
    def _detect_overfitting(self, train_metrics: Dict, val_metrics: Dict) -> Tuple[bool, str]:
        """
        Detect overfitting using multiple criteria.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics
            
        Returns:
            Tuple[bool, str]: (is_overfitting, reason)
        """
        reasons = []
        
        # 1. Accuracy gap detection
        train_acc = train_metrics.get('accuracy', 0)
        val_acc = val_metrics.get('accuracy', 0)
        accuracy_gap = train_acc - val_acc
        
        if accuracy_gap > self.config.severe_accuracy_gap_threshold:
            reasons.append(f"Severe accuracy gap: {accuracy_gap:.3f}")
        elif accuracy_gap > self.config.accuracy_gap_threshold:
            reasons.append(f"Accuracy gap: {accuracy_gap:.3f}")
        
        # 2. F1 score gap detection
        train_f1 = train_metrics.get('f1', 0)
        val_f1 = val_metrics.get('f1', 0)
        f1_gap = train_f1 - val_f1
        
        if f1_gap > self.config.severe_f1_gap_threshold:
            reasons.append(f"Severe F1 gap: {f1_gap:.3f}")
        elif f1_gap > self.config.f1_gap_threshold:
            reasons.append(f"F1 gap: {f1_gap:.3f}")
        
        # 3. Confidence-based overfitting detection
        train_conf = train_metrics.get('confidence', 0)
        val_conf = val_metrics.get('confidence', 0)
        if train_conf > 0 and val_conf > 0:
            conf_gap = train_conf - val_conf
            if conf_gap > self.config.confidence_gap_threshold:
                reasons.append(f"Confidence gap: {conf_gap:.3f}")
        
        # 4. Overconfident predictions
        val_overconfident = val_metrics.get('overconfident_ratio', 0)
        if val_overconfident > self.config.overconfident_ratio_threshold:
            reasons.append(f"Overconfident predictions: {val_overconfident:.3f}")
        
        # 5. Feature concentration
        feature_concentration = train_metrics.get('feature_concentration', 0)
        if feature_concentration > self.config.feature_concentration_threshold:
            reasons.append(f"Feature concentration: {feature_concentration:.3f}")
        
        # 6. Cross-validation issues
        cv_variance = val_metrics.get('cv_variance', 0)
        if cv_variance > self.config.cv_variance_threshold:
            reasons.append(f"High CV variance: {cv_variance:.3f}")
        
        cv_test_gap = val_metrics.get('cv_test_gap', 0)
        if cv_test_gap > self.config.cv_test_gap_threshold:
            reasons.append(f"CV-test gap: {cv_test_gap:.3f}")
        
        is_overfitting = len(reasons) > 0
        reason = "; ".join(reasons) if reasons else ""
        
        if is_overfitting:
            self.overfitting_warnings.append({
                'epoch': len(self.overfitting_warnings) + 1,
                'reasons': reasons,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'accuracy_gap': accuracy_gap
            })
        
        return is_overfitting, reason

class AggressiveOverfittingDetector:
    """Aggressive overfitting detection with multiple validation strategies."""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.detection_history = []
        
    def comprehensive_overfitting_analysis(self, 
                                         train_predictions: np.ndarray,
                                         val_predictions: np.ndarray,
                                         train_labels: np.ndarray,
                                         val_labels: np.ndarray,
                                         train_probabilities: Optional[np.ndarray] = None,
                                         val_probabilities: Optional[np.ndarray] = None,
                                         feature_importance: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Comprehensive overfitting analysis with aggressive thresholds.
        
        Args:
            train_predictions: Training predictions
            val_predictions: Validation predictions
            train_labels: Training labels
            val_labels: Validation labels
            train_probabilities: Training probabilities (optional)
            val_probabilities: Validation probabilities (optional)
            feature_importance: Feature importance scores (optional)
            
        Returns:
            Dict: Comprehensive overfitting analysis
        """
        analysis = {
            'is_overfitting': False,
            'severity': 'none',
            'warnings': [],
            'recommendations': [],
            'metrics': {}
        }
        
        try:
            # Calculate basic metrics
            train_acc = accuracy_score(train_labels, train_predictions)
            val_acc = accuracy_score(val_labels, val_predictions)
            accuracy_gap = train_acc - val_acc
            
            train_f1 = f1_score(train_labels, train_predictions, average='weighted')
            val_f1 = f1_score(val_labels, val_predictions, average='weighted')
            f1_gap = train_f1 - val_f1
            
            # Store metrics
            analysis['metrics'] = {
                'train_accuracy': float(train_acc),
                'val_accuracy': float(val_acc),
                'accuracy_gap': float(accuracy_gap),
                'train_f1': float(train_f1),
                'val_f1': float(val_f1),
                'f1_gap': float(f1_gap)
            }
            
            # Aggressive overfitting detection
            overfitting_indicators = []
            
            # 1. Accuracy gap analysis (more aggressive thresholds)
            if accuracy_gap > self.config.severe_accuracy_gap_threshold:
                overfitting_indicators.append('severe_accuracy_gap')
                analysis['warnings'].append(f"🚨 SEVERE overfitting: {accuracy_gap:.3f} accuracy gap")
            elif accuracy_gap > self.config.accuracy_gap_threshold:
                overfitting_indicators.append('accuracy_gap')
                analysis['warnings'].append(f"⚠️ Overfitting: {accuracy_gap:.3f} accuracy gap")
            
            # 2. F1 gap analysis
            if f1_gap > self.config.severe_f1_gap_threshold:
                overfitting_indicators.append('severe_f1_gap')
                analysis['warnings'].append(f"🚨 SEVERE F1 overfitting: {f1_gap:.3f} F1 gap")
            elif f1_gap > self.config.f1_gap_threshold:
                overfitting_indicators.append('f1_gap')
                analysis['warnings'].append(f"⚠️ F1 overfitting: {f1_gap:.3f} F1 gap")
            
            # 3. Confidence analysis (if probabilities available)
            if train_probabilities is not None and val_probabilities is not None:
                train_conf = np.mean(np.max(train_probabilities, axis=1))
                val_conf = np.mean(np.max(val_probabilities, axis=1))
                conf_gap = train_conf - val_conf
                
                analysis['metrics']['train_confidence'] = float(train_conf)
                analysis['metrics']['val_confidence'] = float(val_conf)
                analysis['metrics']['confidence_gap'] = float(conf_gap)
                
                if conf_gap > self.config.confidence_gap_threshold:
                    overfitting_indicators.append('confidence_gap')
                    analysis['warnings'].append(f"⚠️ Confidence gap: {conf_gap:.3f}")
                
                # Overconfident predictions
                overconfident_threshold = 0.9
                val_overconfident = np.mean(np.max(val_probabilities, axis=1) > overconfident_threshold)
                if val_overconfident > self.config.overconfident_ratio_threshold:
                    overfitting_indicators.append('overconfident')
                    analysis['warnings'].append(f"⚠️ Overconfident predictions: {val_overconfident:.3f}")
            
            # 4. Feature importance analysis
            if feature_importance is not None:
                # Check feature concentration
                sorted_importance = np.sort(feature_importance)[::-1]
                top_features_ratio = 0.1  # Top 10% of features
                n_top = max(1, int(len(sorted_importance) * top_features_ratio))
                concentration = np.sum(sorted_importance[:n_top]) / np.sum(sorted_importance)
                
                analysis['metrics']['feature_concentration'] = float(concentration)
                
                if concentration > self.config.feature_concentration_threshold:
                    overfitting_indicators.append('feature_concentration')
                    analysis['warnings'].append(f"⚠️ Feature concentration: {concentration:.3f}")
            
            # 5. Determine severity and recommendations
            if len(overfitting_indicators) > 0:
                analysis['is_overfitting'] = True
                
                if 'severe_accuracy_gap' in overfitting_indicators or 'severe_f1_gap' in overfitting_indicators:
                    analysis['severity'] = 'severe'
                    analysis['recommendations'].extend([
                        "🚨 IMMEDIATE ACTION: Stop training and increase regularization",
                        "🚨 Consider reducing model complexity significantly",
                        "🚨 Implement stronger cross-validation strategies"
                    ])
                elif len(overfitting_indicators) >= 3:
                    analysis['severity'] = 'high'
                    analysis['recommendations'].extend([
                        "⚠️ HIGH RISK: Increase regularization parameters",
                        "⚠️ Consider early stopping",
                        "⚠️ Reduce model complexity"
                    ])
                else:
                    analysis['severity'] = 'moderate'
                    analysis['recommendations'].extend([
                        "📊 Monitor closely: Add regularization",
                        "📊 Consider ensemble methods",
                        "📊 Implement early stopping"
                    ])
            
            # Store detection history
            self.detection_history.append({
                'timestamp': len(self.detection_history),
                'is_overfitting': analysis['is_overfitting'],
                'severity': analysis['severity'],
                'indicators': overfitting_indicators,
                'accuracy_gap': accuracy_gap
            })
            
        except Exception as e:
            logger.error(f"Overfitting analysis failed: {e}")
            analysis['warnings'].append(f"❌ Analysis failed: {str(e)}")
            analysis['is_overfitting'] = False
            
        return analysis

# Global instances for easy access
DEFAULT_EARLY_STOPPING_CONFIG = EarlyStoppingConfig()
DEFAULT_OVERFITTING_DETECTOR = AggressiveOverfittingDetector(DEFAULT_EARLY_STOPPING_CONFIG)

def get_early_stopping_config() -> EarlyStoppingConfig:
    """Get the default early stopping configuration."""
    return DEFAULT_EARLY_STOPPING_CONFIG

def get_overfitting_detector() -> AggressiveOverfittingDetector:
    """Get the default overfitting detector."""
    return DEFAULT_OVERFITTING_DETECTOR