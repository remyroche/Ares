"""
Pipeline Enhancements Integration.

This module integrates all the new enhancements into the existing pre-training pipeline:
1. TimeSplitManager for proper data segmentation
2. EnhancedLabeler for improved label design
3. RedundancyController for feature redundancy management
4. DriftMonitor for distribution shift detection
5. EnhancedLookbackOptimizer for robust lookback optimization
6. EnhancedFeatureSelector for stable feature selection
7. QuantitativeValidator for soundness checks
8. ReproducibilityTracker for complete reproducibility

This module provides high-level integration functions that can be used
by the existing pipeline components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import system_logger

# Import all enhancement modules
from .time_split_manager import (
    TimeSplitManager,
    SplitConfig,
    SplitStrategy,
    SplitResult,
    create_time_split_manager
)
from .enhanced_label_design import (
    EnhancedLabeler,
    LabelingMethod,
    TransactionCostModel,
    VolatilityConfig,
    TripleBarrierConfig,
    RegimeLabelingConfig,
    create_enhanced_labels
)
from .feature_redundancy_control import (
    RedundancyController,
    DriftMonitor,
    RedundancyConfig,
    DriftConfig,
    RedundancyReport,
    DriftReport
)
from .enhanced_lookback_optimizer import (
    EnhancedLookbackOptimizer,
    OptimizationObjective,
    OptimizationConstraints,
    LookbackResult,
    optimize_lookback_period
)
from .enhanced_feature_selector import (
    EnhancedFeatureSelector,
    FeatureSelectionConfig,
    EconomicTheme,
    SelectionResult,
    select_features_robust
)
from .quantitative_validation import (
    QuantitativeValidator,
    ValidationReport,
    validate_pre_training_outputs
)
from .reproducibility_tracker import (
    ReproducibilityTracker,
    ReproducibilityManifest,
    create_reproducibility_tracker
)


@dataclass
class EnhancedPipelineConfig:
    """Configuration for enhanced pipeline features."""
    
    # Data splitting
    enable_time_splitting: bool = True
    split_strategy: SplitStrategy = SplitStrategy.SIMPLE_CHRONOLOGICAL
    train_ratio: float = 0.70
    validation_ratio: float = 0.20
    test_ratio: float = 0.10
    enable_purging: bool = True
    purge_window_hours: int = 24
    embargo_window_hours: int = 12
    
    # Label design
    enable_enhanced_labeling: bool = True
    labeling_method: LabelingMethod = LabelingMethod.TRIPLE_BARRIER
    adjust_for_transaction_costs: bool = True
    enable_regime_dependent_labeling: bool = True
    
    # Feature redundancy
    enable_redundancy_control: bool = True
    correlation_threshold: float = 0.85
    enable_vif: bool = True
    
    # Drift monitoring
    enable_drift_monitoring: bool = True
    kl_threshold: float = 0.15
    
    # Lookback optimization
    enable_enhanced_lookback: bool = True
    lookback_objective: OptimizationObjective = OptimizationObjective.MAX_IC
    lookback_min: int = 5
    lookback_max: int = 300
    enable_lookback_regularization: bool = True
    
    # Feature selection
    enable_enhanced_selection: bool = True
    n_bootstrap_folds: int = 5
    min_selection_frequency: float = 0.60
    preserve_economic_themes: bool = True
    
    # Validation
    enable_quantitative_validation: bool = True
    strict_validation: bool = False
    
    # Reproducibility
    enable_reproducibility_tracking: bool = True
    track_git_info: bool = True
    track_environment: bool = True


class EnhancedPipelineOrchestrator:
    """
    Orchestrates all enhancements in the pre-training pipeline.
    """
    
    def __init__(
        self,
        config: Optional[EnhancedPipelineConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the orchestrator.
        
        Args:
            config: Pipeline enhancement configuration
            logger: Optional logger instance
        """
        self.config = config or EnhancedPipelineConfig()
        self.logger = logger or system_logger.getChild('EnhancedPipelineOrchestrator')
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all enhancement components."""
        # Time split manager
        if self.config.enable_time_splitting:
            split_config = SplitConfig(
                train_ratio=self.config.train_ratio,
                validation_ratio=self.config.validation_ratio,
                test_ratio=self.config.test_ratio,
                enable_purging=self.config.enable_purging,
                purge_window_hours=self.config.purge_window_hours,
                embargo_window_hours=self.config.embargo_window_hours
            )
            self.time_splitter = TimeSplitManager(config=split_config, logger=self.logger)
        
        # Enhanced labeler
        if self.config.enable_enhanced_labeling:
            volatility_config = VolatilityConfig()
            
            barrier_config = TripleBarrierConfig(
                adjust_for_costs=self.config.adjust_for_transaction_costs
            )
            
            regime_config = RegimeLabelingConfig(
                enable_regime_adaptation=self.config.enable_regime_dependent_labeling
            )
            
            self.enhanced_labeler = EnhancedLabeler(
                volatility_config=volatility_config,
                barrier_config=barrier_config,
                regime_config=regime_config,
                logger=self.logger
            )
        
        # Redundancy controller
        if self.config.enable_redundancy_control:
            redundancy_config = RedundancyConfig(
                correlation_threshold=self.config.correlation_threshold,
                enable_vif=self.config.enable_vif
            )
            self.redundancy_controller = RedundancyController(
                config=redundancy_config,
                logger=self.logger
            )
        
        # Drift monitor
        if self.config.enable_drift_monitoring:
            drift_config = DriftConfig(
                kl_threshold=self.config.kl_threshold
            )
            self.drift_monitor = DriftMonitor(
                config=drift_config,
                logger=self.logger
            )
        
        # Lookback optimizer
        if self.config.enable_enhanced_lookback:
            lookback_constraints = OptimizationConstraints(
                min_lookback=self.config.lookback_min,
                max_lookback=self.config.lookback_max,
                enable_regularization=self.config.enable_lookback_regularization
            )
            self.lookback_optimizer = EnhancedLookbackOptimizer(
                objective=self.config.lookback_objective,
                constraints=lookback_constraints,
                logger=self.logger
            )
        
        # Feature selector
        if self.config.enable_enhanced_selection:
            selection_config = FeatureSelectionConfig(
                n_bootstrap_folds=self.config.n_bootstrap_folds,
                min_selection_frequency=self.config.min_selection_frequency,
                preserve_economic_themes=self.config.preserve_economic_themes
            )
            self.feature_selector = EnhancedFeatureSelector(
                config=selection_config,
                logger=self.logger
            )
        
        # Quantitative validator
        if self.config.enable_quantitative_validation:
            self.validator = QuantitativeValidator(
                logger=self.logger,
                strict_mode=self.config.strict_validation
            )
        
        # Reproducibility tracker
        if self.config.enable_reproducibility_tracking:
            self.repro_tracker = create_reproducibility_tracker(logger=self.logger)
    
    def process_data_splitting(
        self,
        data: pd.DataFrame,
        regime_labels: Optional[pd.Series] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitResult]:
        """
        Process data splitting with enhancements.
        
        Args:
            data: Input data with DatetimeIndex
            regime_labels: Optional regime classifications
        
        Returns:
            Tuple of (train_data, val_data, test_data, split_result)
        """
        if not self.config.enable_time_splitting:
            # Simple split without enhancements
            n = len(data)
            train_end = int(n * self.config.train_ratio)
            val_end = train_end + int(n * self.config.validation_ratio)
            
            return (
                data.iloc[:train_end],
                data.iloc[train_end:val_end],
                data.iloc[val_end:],
                None
            )
        
        # Use enhanced time splitting
        strategy = self.config.split_strategy
        
        if regime_labels is not None and strategy == SplitStrategy.REGIME_AWARE:
            # Add regime column temporarily
            data_with_regime = data.copy()
            data_with_regime['__regime__'] = regime_labels
            
            split_config = self.time_splitter.config
            split_config.regime_column = '__regime__'
            
            split_result = self.time_splitter.split(data_with_regime, strategy=strategy)
            
            # Remove regime column
            del data_with_regime['__regime__']
        else:
            split_result = self.time_splitter.split(data, strategy=strategy)
        
        train_data = data.iloc[split_result.train_idx]
        val_data = data.iloc[split_result.validation_idx]
        test_data = data.iloc[split_result.test_idx]
        
        self.logger.info(
            f"Data split complete: train={len(train_data)}, "
            f"val={len(val_data)}, test={len(test_data)}"
        )
        
        # Track in reproducibility
        if self.config.enable_reproducibility_tracking:
            self.repro_tracker.register_dataset('train_data', train_data)
            self.repro_tracker.register_dataset('val_data', val_data)
            self.repro_tracker.register_dataset('test_data', test_data)
        
        return train_data, val_data, test_data, split_result
    
    def process_labeling(
        self,
        prices: pd.Series,
        horizon_bars: int = 48,
        ohlc: Optional[pd.DataFrame] = None,
        regime_labels: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Process labeling with enhancements.
        
        Args:
            prices: Price series
            horizon_bars: Horizon length
            ohlc: Optional OHLC data
            regime_labels: Optional regime classifications
        
        Returns:
            DataFrame with enhanced labels
        """
        if not self.config.enable_enhanced_labeling:
            # Simple return-based labels
            returns = prices.pct_change(horizon_bars)
            labels = pd.DataFrame({
                'label': np.where(returns > 0, 1, -1),
                'raw_return': returns
            }, index=prices.index)
            return labels
        
        # Use enhanced labeling
        labels = create_enhanced_labels(
            prices=prices,
            horizon_bars=horizon_bars,
            method=self.config.labeling_method,
            ohlc=ohlc,
            regime_labels=regime_labels,
            logger=self.logger
        )
        
        self.logger.info(f"Enhanced labeling complete: {len(labels)} labels created")
        
        return labels
    
    def process_feature_redundancy(
        self,
        features: pd.DataFrame,
        feature_importance: Optional[Dict[str, float]] = None
    ) -> Tuple[pd.DataFrame, RedundancyReport]:
        """
        Process feature redundancy control.
        
        Args:
            features: Feature DataFrame
            feature_importance: Optional importance scores
        
        Returns:
            Tuple of (reduced_features, redundancy_report)
        """
        if not self.config.enable_redundancy_control:
            return features, None
        
        # Analyze and reduce redundancy
        report = self.redundancy_controller.analyze_and_reduce(
            features=features,
            feature_importance=feature_importance
        )
        
        reduced_features = features[report.retained_features]
        
        self.logger.info(
            f"Redundancy control complete: {len(features.columns)} -> "
            f"{len(reduced_features.columns)} features ({report.reduction_rate:.1%} reduction)"
        )
        
        return reduced_features, report
    
    def process_drift_monitoring(
        self,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame
    ) -> DriftReport:
        """
        Monitor feature drift between train and validation.
        
        Args:
            train_features: Training features
            val_features: Validation features
        
        Returns:
            DriftReport with drift analysis
        """
        if not self.config.enable_drift_monitoring:
            return None
        
        drift_report = self.drift_monitor.detect_drift(
            reference_features=train_features,
            current_features=val_features
        )
        
        if drift_report.drifted_features:
            self.logger.warning(
                f"Drift detected in {len(drift_report.drifted_features)} features: "
                f"{drift_report.drifted_features[:5]}"
            )
        
        return drift_report
    
    def process_lookback_optimization(
        self,
        prices: pd.Series,
        labels: Optional[pd.Series] = None,
        feature_fn: Optional[Any] = None
    ) -> LookbackResult:
        """
        Process lookback optimization with enhancements.
        
        Args:
            prices: Price series
            labels: Optional labels
            feature_fn: Optional feature function
        
        Returns:
            LookbackResult with optimization results
        """
        if not self.config.enable_enhanced_lookback:
            # Simple default lookback
            return LookbackResult(
                optimal_lookback=50,
                objective_value=0.0,
                objective_name="default",
                stability_score=0.0,
                resampled_lookbacks=[50],
                lookback_std=0.0,
                all_lookbacks_tested=[50],
                all_objective_values=[0.0],
                regularization_penalty=0.0,
                raw_objective_value=0.0
            )
        
        # Use enhanced optimization
        result = self.lookback_optimizer.optimize(
            prices=prices,
            labels=labels,
            feature_fn=feature_fn,
            use_bootstrap=True
        )
        
        self.logger.info(
            f"Lookback optimization complete: optimal={result.optimal_lookback}, "
            f"stability={result.stability_score:.3f}, stable={result.is_stable}"
        )
        
        return result
    
    def process_feature_selection(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        target_n_features: Optional[int] = None,
        feature_themes: Optional[Dict[str, EconomicTheme]] = None
    ) -> Tuple[pd.DataFrame, SelectionResult]:
        """
        Process feature selection with enhancements.
        
        Args:
            features: Feature DataFrame
            labels: Target labels
            target_n_features: Target number of features
            feature_themes: Optional feature themes
        
        Returns:
            Tuple of (selected_features, selection_result)
        """
        if not self.config.enable_enhanced_selection:
            # Simple top-k selection by correlation
            correlations = features.corrwith(labels).abs().sort_values(ascending=False)
            top_k = target_n_features or min(80, len(features.columns))
            selected_cols = correlations.head(top_k).index.tolist()
            
            return features[selected_cols], None
        
        # Use enhanced selection
        result = self.feature_selector.select_features(
            features=features,
            labels=labels,
            feature_themes=feature_themes,
            target_n_features=target_n_features
        )
        
        selected_features = features[result.selected_features]
        
        self.logger.info(
            f"Feature selection complete: {len(result.selected_features)} features selected, "
            f"validation_passed={result.validation_passed}"
        )
        
        return selected_features, result
    
    def validate_outputs(
        self,
        labels: Optional[pd.DataFrame] = None,
        features: Optional[pd.DataFrame] = None,
        lookback_results: Optional[Dict[str, Any]] = None,
        regime_labels: Optional[pd.Series] = None
    ) -> ValidationReport:
        """
        Validate pipeline outputs.
        
        Args:
            labels: Labeled data
            features: Engineered features
            lookback_results: Lookback optimization results
            regime_labels: Regime classifications
        
        Returns:
            ValidationReport with validation results
        """
        if not self.config.enable_quantitative_validation:
            return None
        
        report = self.validator.validate_all(
            labels=labels,
            features=features,
            lookback_results=lookback_results,
            regime_labels=regime_labels
        )
        
        if not report.passed:
            self.logger.warning(
                f"Validation failed: {report.failures_count} failures, "
                f"{report.warnings_count} warnings"
            )
        else:
            self.logger.info(f"Validation passed: all {len(report.results)} checks passed")
        
        return report
    
    def save_reproducibility_manifest(
        self,
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ReproducibilityManifest:
        """
        Save reproducibility manifest.
        
        Args:
            output_path: Path to save manifest
            metadata: Optional additional metadata
        
        Returns:
            ReproducibilityManifest
        """
        if not self.config.enable_reproducibility_tracking:
            return None
        
        manifest = self.repro_tracker.create_manifest(metadata=metadata)
        manifest.save(output_path)
        
        self.logger.info(f"Reproducibility manifest saved to {output_path}")
        
        return manifest


def create_enhanced_pipeline_orchestrator(
    config: Optional[EnhancedPipelineConfig] = None,
    logger: Optional[logging.Logger] = None
) -> EnhancedPipelineOrchestrator:
    """
    Factory function to create an enhanced pipeline orchestrator.
    
    Args:
        config: Pipeline enhancement configuration
        logger: Optional logger instance
    
    Returns:
        EnhancedPipelineOrchestrator instance
    """
    return EnhancedPipelineOrchestrator(config=config, logger=logger)


__all__ = [
    'EnhancedPipelineOrchestrator',
    'EnhancedPipelineConfig',
    'create_enhanced_pipeline_orchestrator',
    # Re-export key classes for convenience
    'TimeSplitManager',
    'EnhancedLabeler',
    'RedundancyController',
    'DriftMonitor',
    'EnhancedLookbackOptimizer',
    'EnhancedFeatureSelector',
    'QuantitativeValidator',
    'ReproducibilityTracker',
]