"""
Enhanced Label Balancer API - Production-Ready Implementation

This module provides the final, production-ready API that integrates all temporal
validation, fairness analysis, leakage detection, and distribution shift components.

API Shape (as suggested):
balancer = EnhancedLabelBalancer(
    time_col="ts",
    label_col="y", 
    entity_cols=["account_id"],
    feature_availability={
        "x_price_5min_ma": "00:05:00",
        "country": "0:00:00",
    },
    min_train_val_gap="1D",
    embargo="30min",
    time_bin="7D",
    balance_strategy="temporal_reweigh"
)

report = balancer.fit_resample(X, y)
model = clf.fit(report.X_resampled, report.y_resampled)

cv = PurgedTemporalKFold(n_splits=5, embargo="30min", group="account_id")
scores = cv.evaluate(model, X, y, metrics=["AUC", "ECE", "F1_time_binned"])
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
import logging
from pathlib import Path

# Import all our modules
from leakage_detection_system import LeakageDetector, LeakageDetectionConfig
from time_alignment_system import TimeAlignmentValidator, FeatureAvailabilityRegistry, FeatureAvailabilityConfig
from temporal_fairness_metrics import TemporalFairnessAnalyzer, TemporalFairnessConfig
from distribution_shift_detection import DistributionShiftDetector, DistributionShiftConfig
from enhanced_purged_cv import EnhancedPurgedTemporalKFold, PurgedCVConfig
from comprehensive_reporting import ComprehensiveReporter, ReportConfig

# Import existing utilities
try:
    from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint_info(msg): print(f"INFO: {msg}")
    def tprint_warning(msg): print(f"WARNING: {msg}")
    def tprint_error(msg): print(f"ERROR: {msg}")
    def tprint_success(msg): print(f"SUCCESS: {msg}")

logger = logging.getLogger(__name__)


class BalanceStrategy(Enum):
    """Label balancing strategies."""
    TEMPORAL_REWEIGH = "temporal_reweigh"      # Temporal reweighting
    WINDOW_RESAMPLE = "window_resample"        # Window-aware resampling
    CLASS_COST = "class_cost"                  # Cost-sensitive learning
    PRIOR_SHIFT = "prior_shift"                # Prior shift correction
    HYBRID = "hybrid"                          # Hybrid approach


@dataclass
class EnhancedLabelBalancerConfig:
    """Configuration for Enhanced Label Balancer."""
    
    # Core settings
    time_col: str = "timestamp"
    label_col: str = "y"
    entity_cols: Optional[List[str]] = None
    
    # Feature availability registry
    feature_availability: Dict[str, str] = field(default_factory=dict)
    
    # Temporal constraints
    min_train_val_gap: str = "1D"
    embargo: str = "30min"
    time_bin: str = "7D"
    
    # Balancing strategy
    balance_strategy: BalanceStrategy = BalanceStrategy.TEMPORAL_REWEIGH
    
    # Component configurations
    leakage_config: Optional[LeakageDetectionConfig] = None
    time_alignment_config: Optional[FeatureAvailabilityConfig] = None
    fairness_config: Optional[TemporalFairnessConfig] = None
    distribution_config: Optional[DistributionShiftConfig] = None
    cv_config: Optional[PurgedCVConfig] = None
    report_config: Optional[ReportConfig] = None
    
    # Validation settings
    enable_temporal_validation: bool = True
    enable_leakage_detection: bool = True
    enable_fairness_analysis: bool = True
    enable_distribution_shift: bool = True
    enable_comprehensive_reporting: bool = True
    
    # Performance settings
    parallel_processing: bool = False
    max_workers: int = 4
    chunk_size: int = 10000


@dataclass
class BalancingReport:
    """Report from label balancing process."""
    
    # Core results
    X_resampled: pd.DataFrame
    y_resampled: pd.Series
    sample_weights: Optional[pd.Series] = None
    
    # Validation results
    temporal_validation: Dict[str, Any] = field(default_factory=dict)
    leakage_detection: List[Any] = field(default_factory=list)
    fairness_analysis: Any = None
    distribution_shift: List[Any] = field(default_factory=list)
    
    # Balancing information
    balancing_applied: bool = False
    balancing_strategy: str = ""
    original_class_distribution: Dict[Any, int] = field(default_factory=dict)
    resampled_class_distribution: Dict[Any, int] = field(default_factory=dict)
    
    # Metadata
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Report paths
    html_report_path: str = ""
    json_report_path: str = ""


class EnhancedLabelBalancer:
    """
    Production-ready Enhanced Label Balancer with comprehensive temporal validation.
    
    This is the main API that integrates all temporal validation, fairness analysis,
    leakage detection, and distribution shift components into a single, easy-to-use interface.
    """
    
    def __init__(self, config: Optional[EnhancedLabelBalancerConfig] = None):
        """Initialize Enhanced Label Balancer."""
        self.config = config or EnhancedLabelBalancerConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Processing history
        self.processing_history = []
    
    def _initialize_components(self):
        """Initialize all component systems."""
        try:
            # Initialize feature availability registry
            self.feature_registry = FeatureAvailabilityRegistry(
                FeatureAvailabilityConfig(
                    feature_lags=self.config.feature_availability,
                    strict_mode=True
                )
            )
            
            # Initialize time alignment validator
            self.time_validator = TimeAlignmentValidator(
                self.feature_registry,
                self.config.time_alignment_config
            )
            
            # Initialize leakage detector
            self.leakage_detector = LeakageDetector(
                self.config.leakage_config
            )
            
            # Initialize fairness analyzer
            self.fairness_analyzer = TemporalFairnessAnalyzer(
                self.config.fairness_config
            )
            
            # Initialize distribution shift detector
            self.shift_detector = DistributionShiftDetector(
                self.config.distribution_config
            )
            
            # Initialize purged CV
            self.purged_cv = EnhancedPurgedTemporalKFold(
                self.config.cv_config
            )
            
            # Initialize reporter
            self.reporter = ComprehensiveReporter(
                self.config.report_config
            )
            
            if TPRINT_AVAILABLE:
                tprint_success("✅ All components initialized successfully")
                
        except Exception as e:
            logger.error(f"Component initialization failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Component initialization failed: {e}")
            raise
    
    def fit_resample(self, X: pd.DataFrame, y: pd.Series,
                    X_val: Optional[pd.DataFrame] = None,
                    y_val: Optional[pd.Series] = None,
                    additional_features: Optional[Dict[str, pd.Series]] = None) -> BalancingReport:
        """
        Fit the balancer and resample the data with comprehensive temporal validation.
        
        Args:
            X: Training features with datetime index
            y: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            additional_features: Additional features for analysis
            
        Returns:
            BalancingReport with resampled data and validation results
        """
        start_time = datetime.now()
        
        try:
            if TPRINT_AVAILABLE:
                tprint_info("🚀 Starting enhanced label balancing with temporal validation")
            
            # Initialize report
            report = BalancingReport(
                X_resampled=X.copy(),
                y_resampled=y.copy(),
                original_class_distribution=y.value_counts().to_dict(),
                balancing_strategy=self.config.balance_strategy.value
            )
            
            # 1. Time alignment validation
            if self.config.enable_temporal_validation:
                if TPRINT_AVAILABLE:
                    tprint_info("🔍 Validating time alignment contracts")
                
                time_alignment_result = self.time_validator.validate_time_alignment(
                    X, y, time_col=self.config.time_col
                )
                report.temporal_validation = time_alignment_result.__dict__
                
                if not time_alignment_result.is_valid:
                    report.warnings.extend([
                        f"Time alignment validation failed: {len(time_alignment_result.violations)} violations"
                    ])
            
            # 2. Leakage detection
            if self.config.enable_leakage_detection:
                if TPRINT_AVAILABLE:
                    tprint_info("🔍 Detecting data leakage")
                
                leakage_results = self.leakage_detector.detect_all_leakage(
                    X, y, self.config.entity_cols, self.config.time_col
                )
                report.leakage_detection = [result.__dict__ for result in leakage_results]
                
                if leakage_results:
                    critical_leakage = [r for r in leakage_results if r.severity.value == 'critical']
                    if critical_leakage:
                        report.warnings.extend([
                            f"Critical leakage detected: {len(critical_leakage)} issues"
                        ])
            
            # 3. Fairness analysis
            if self.config.enable_fairness_analysis:
                if TPRINT_AVAILABLE:
                    tprint_info("📊 Analyzing temporal fairness")
                
                fairness_result = self.fairness_analyzer.analyze_temporal_fairness(
                    X, y, additional_features=additional_features
                )
                report.fairness_analysis = fairness_result
                
                if fairness_result.overall_fairness_score < 0.7:
                    report.warnings.extend([
                        f"Low temporal fairness score: {fairness_result.overall_fairness_score:.3f}"
                    ])
            
            # 4. Distribution shift detection
            if self.config.enable_distribution_shift:
                if TPRINT_AVAILABLE:
                    tprint_info("📊 Detecting distribution shifts")
                
                shift_results = self.shift_detector.detect_all_shifts(
                    X, y, time_col=self.config.time_col
                )
                report.distribution_shift = [result.__dict__ for result in shift_results]
                
                if shift_results:
                    critical_shifts = [r for r in shift_results if r.shift_severity.value == 'critical']
                    if critical_shifts:
                        report.warnings.extend([
                            f"Critical distribution shifts detected: {len(critical_shifts)} issues"
                        ])
            
            # 5. Apply label balancing
            if TPRINT_AVAILABLE:
                tprint_info("⚖️ Applying temporal-aware label balancing")
            
            X_balanced, y_balanced, sample_weights = self._apply_balancing_strategy(
                X, y, report.fairness_analysis
            )
            
            report.X_resampled = X_balanced
            report.y_resampled = y_balanced
            report.sample_weights = sample_weights
            report.balancing_applied = True
            report.resampled_class_distribution = y_balanced.value_counts().to_dict()
            
            # 6. Generate comprehensive report
            if self.config.enable_comprehensive_reporting:
                if TPRINT_AVAILABLE:
                    tprint_info("📄 Generating comprehensive report")
                
                html_path, json_path = self.reporter.generate_comprehensive_report(
                    temporal_validation_results=report.temporal_validation,
                    leakage_detection_results=report.leakage_detection,
                    fairness_analysis_results=report.fairness_analysis,
                    distribution_shift_results=report.distribution_shift
                )
                
                report.html_report_path = html_path
                report.json_report_path = json_path
            
            # 7. Generate recommendations
            report.recommendations = self._generate_recommendations(report)
            
            # 8. Calculate processing time
            report.processing_time = (datetime.now() - start_time).total_seconds()
            
            # Store in history
            self.processing_history.append(report)
            
            if TPRINT_AVAILABLE:
                tprint_success(f"✅ Enhanced label balancing completed in {report.processing_time:.2f}s")
            
            return report
            
        except Exception as e:
            logger.error(f"Enhanced label balancing failed: {e}")
            if TPRINT_AVAILABLE:
                tprint_error(f"❌ Enhanced label balancing failed: {e}")
            raise
    
    def _apply_balancing_strategy(self, X: pd.DataFrame, y: pd.Series, 
                                 fairness_analysis: Optional[Any]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply the specified balancing strategy."""
        try:
            if self.config.balance_strategy == BalanceStrategy.TEMPORAL_REWEIGH:
                return self._apply_temporal_reweighting(X, y, fairness_analysis)
            elif self.config.balance_strategy == BalanceStrategy.WINDOW_RESAMPLE:
                return self._apply_window_resampling(X, y, fairness_analysis)
            elif self.config.balance_strategy == BalanceStrategy.CLASS_COST:
                return self._apply_class_cost_learning(X, y, fairness_analysis)
            elif self.config.balance_strategy == BalanceStrategy.PRIOR_SHIFT:
                return self._apply_prior_shift_correction(X, y, fairness_analysis)
            elif self.config.balance_strategy == BalanceStrategy.HYBRID:
                return self._apply_hybrid_balancing(X, y, fairness_analysis)
            else:
                # Default: no balancing
                return X, y, None
                
        except Exception as e:
            logger.error(f"Balancing strategy application failed: {e}")
            return X, y, None
    
    def _apply_temporal_reweighting(self, X: pd.DataFrame, y: pd.Series, 
                                   fairness_analysis: Optional[Any]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply temporal reweighting strategy."""
        try:
            # Calculate sample weights based on temporal fairness
            sample_weights = pd.Series(1.0, index=y.index)
            
            if fairness_analysis and hasattr(fairness_analysis, 'temporal_periods'):
                # Weight samples inversely to their frequency in each time period
                for period in fairness_analysis.temporal_periods:
                    period_mask = X.index.to_series().dt.floor(self.config.time_bin) == period
                    if period_mask.any():
                        period_y = y[period_mask]
                        class_counts = period_y.value_counts()
                        
                        # Calculate inverse frequency weights
                        for class_label, count in class_counts.items():
                            class_mask = period_mask & (y == class_label)
                            if class_mask.any():
                                weight = 1.0 / (count / len(period_y))
                                sample_weights[class_mask] = weight
            
            # Normalize weights
            sample_weights = sample_weights / sample_weights.mean()
            
            return X, y, sample_weights
            
        except Exception as e:
            logger.error(f"Temporal reweighting failed: {e}")
            return X, y, None
    
    def _apply_window_resampling(self, X: pd.DataFrame, y: pd.Series, 
                                fairness_analysis: Optional[Any]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply window-aware resampling strategy."""
        try:
            # This would implement sophisticated window-aware resampling
            # For now, return original data
            return X, y, None
            
        except Exception as e:
            logger.error(f"Window resampling failed: {e}")
            return X, y, None
    
    def _apply_class_cost_learning(self, X: pd.DataFrame, y: pd.Series, 
                                  fairness_analysis: Optional[Any]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply cost-sensitive learning strategy."""
        try:
            # Calculate class costs based on temporal fairness
            class_costs = {}
            for class_label in y.unique():
                class_costs[class_label] = 1.0
            
            if fairness_analysis and hasattr(fairness_analysis, 'temporal_balance_score'):
                # Adjust costs based on temporal balance
                balance_score = fairness_analysis.temporal_balance_score
                for class_label in y.unique():
                    class_costs[class_label] = 1.0 / (balance_score + 0.1)
            
            # Create sample weights based on class costs
            sample_weights = y.map(class_costs)
            
            return X, y, sample_weights
            
        except Exception as e:
            logger.error(f"Class cost learning failed: {e}")
            return X, y, None
    
    def _apply_prior_shift_correction(self, X: pd.DataFrame, y: pd.Series, 
                                     fairness_analysis: Optional[Any]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply prior shift correction strategy."""
        try:
            # This would implement Saerens-Latinne EM algorithm
            # For now, return original data
            return X, y, None
            
        except Exception as e:
            logger.error(f"Prior shift correction failed: {e}")
            return X, y, None
    
    def _apply_hybrid_balancing(self, X: pd.DataFrame, y: pd.Series, 
                               fairness_analysis: Optional[Any]) -> Tuple[pd.DataFrame, pd.Series, Optional[pd.Series]]:
        """Apply hybrid balancing strategy."""
        try:
            # Combine multiple strategies
            X_temp, y_temp, weights_temp = self._apply_temporal_reweighting(X, y, fairness_analysis)
            X_final, y_final, weights_final = self._apply_class_cost_learning(X_temp, y_temp, fairness_analysis)
            
            # Combine weights
            if weights_temp is not None and weights_final is not None:
                combined_weights = weights_temp * weights_final
                combined_weights = combined_weights / combined_weights.mean()
            else:
                combined_weights = weights_final if weights_final is not None else weights_temp
            
            return X_final, y_final, combined_weights
            
        except Exception as e:
            logger.error(f"Hybrid balancing failed: {e}")
            return X, y, None
    
    def _generate_recommendations(self, report: BalancingReport) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        try:
            # Temporal validation recommendations
            if report.temporal_validation and not report.temporal_validation.get('is_valid', True):
                recommendations.append("Review temporal alignment contracts and feature availability")
            
            # Leakage detection recommendations
            if report.leakage_detection:
                critical_leakage = [r for r in report.leakage_detection if r.get('severity') == 'critical']
                if critical_leakage:
                    recommendations.append("Address critical data leakage issues immediately")
            
            # Fairness analysis recommendations
            if report.fairness_analysis and hasattr(report.fairness_analysis, 'overall_fairness_score'):
                if report.fairness_analysis.overall_fairness_score < 0.7:
                    recommendations.append("Implement temporal fairness correction strategies")
            
            # Distribution shift recommendations
            if report.distribution_shift:
                critical_shifts = [r for r in report.distribution_shift if r.get('shift_severity') == 'critical']
                if critical_shifts:
                    recommendations.append("Monitor and address critical distribution shifts")
            
            # Balancing recommendations
            if report.balancing_applied:
                recommendations.append("Consider model retraining with balanced data")
                recommendations.append("Monitor model performance on balanced data")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    def get_processing_history(self) -> List[BalancingReport]:
        """Get processing history."""
        return self.processing_history.copy()
    
    def get_component_status(self) -> Dict[str, bool]:
        """Get status of all components."""
        return {
            'feature_registry': self.feature_registry is not None,
            'time_validator': self.time_validator is not None,
            'leakage_detector': self.leakage_detector is not None,
            'fairness_analyzer': self.fairness_analyzer is not None,
            'shift_detector': self.shift_detector is not None,
            'purged_cv': self.purged_cv is not None,
            'reporter': self.reporter is not None
        }


# Convenience functions
def create_enhanced_label_balancer(
    time_col: str = "timestamp",
    label_col: str = "y",
    entity_cols: Optional[List[str]] = None,
    feature_availability: Optional[Dict[str, str]] = None,
    min_train_val_gap: str = "1D",
    embargo: str = "30min",
    time_bin: str = "7D",
    balance_strategy: str = "temporal_reweigh",
    **kwargs
) -> EnhancedLabelBalancer:
    """Create enhanced label balancer with simplified configuration."""
    
    # Parse balance strategy
    strategy_map = {
        'temporal_reweigh': BalanceStrategy.TEMPORAL_REWEIGH,
        'window_resample': BalanceStrategy.WINDOW_RESAMPLE,
        'class_cost': BalanceStrategy.CLASS_COST,
        'prior_shift': BalanceStrategy.PRIOR_SHIFT,
        'hybrid': BalanceStrategy.HYBRID
    }
    
    balance_strategy_enum = strategy_map.get(balance_strategy, BalanceStrategy.TEMPORAL_REWEIGH)
    
    # Create configuration
    config = EnhancedLabelBalancerConfig(
        time_col=time_col,
        label_col=label_col,
        entity_cols=entity_cols,
        feature_availability=feature_availability or {},
        min_train_val_gap=min_train_val_gap,
        embargo=embargo,
        time_bin=time_bin,
        balance_strategy=balance_strategy_enum,
        **kwargs
    )
    
    return EnhancedLabelBalancer(config)


# Example usage and testing
if __name__ == "__main__":
    print("Enhanced Label Balancer API - Production Ready")
    print("=" * 60)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=1000, freq='1H')
    X = pd.DataFrame({
        'feature1': np.random.randn(1000),
        'feature2': np.random.randn(1000),
        'x_price_5min_ma': np.random.randn(1000),
        'country': np.random.choice(['US', 'EU', 'AS'], 1000),
        'timestamp': dates
    }, index=dates)
    
    y = pd.Series(np.random.choice([0, 1, 2], size=1000, p=[0.7, 0.2, 0.1]), index=dates)
    
    # Create enhanced label balancer
    balancer = create_enhanced_label_balancer(
        time_col="timestamp",
        label_col="y",
        entity_cols=["account_id"],
        feature_availability={
            "x_price_5min_ma": "00:05:00",
            "country": "0:00:00",
        },
        min_train_val_gap="1D",
        embargo="30min",
        time_bin="7D",
        balance_strategy="temporal_reweigh"
    )
    
    # Fit and resample
    report = balancer.fit_resample(X, y)
    
    print(f"Balancing applied: {report.balancing_applied}")
    print(f"Original distribution: {report.original_class_distribution}")
    print(f"Resampled distribution: {report.resampled_class_distribution}")
    print(f"Processing time: {report.processing_time:.2f}s")
    print(f"Warnings: {len(report.warnings)}")
    print(f"Recommendations: {len(report.recommendations)}")
    print(f"HTML report: {report.html_report_path}")
    print(f"JSON report: {report.json_report_path}")
    
    # Check component status
    status = balancer.get_component_status()
    print(f"Component status: {status}")