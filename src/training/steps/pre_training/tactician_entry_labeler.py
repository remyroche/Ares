"""
Tactician Entry Labeler - Differentiated Entry Timing Labels for Tactician Models

This module provides entry timing label generation for Tactician models,
using enhanced entry quality scoring with regime adaptation.

Key Features:
- 15m timeframe optimization for entry timing
- Local maxima/minima detection with peak filtering  
- Enhanced entry quality scoring (adaptive multi-factor)
- Regime-aware labeling with adaptive thresholds
- Trains on ALL market data (not just Analyst green lights)
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy.signal import find_peaks

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.logger import system_logger
from src.training.steps.pre_training.components.base_component import BasePreTrainingComponent, ComponentConfig, ComponentResult
from src.training.steps.pre_training.components.contracts import PipelineState
from src.training.steps.pre_training.components.component_factory import register_component


@dataclass
class TacticianLabelingConfig:
    """Configuration for Tactician-specific differentiated labeling."""

    # Entry timing optimization
    min_entry_window_minutes: int = 3
    max_entry_window_minutes: int = 60
    entry_quality_threshold: float = 0.25

    # Price movement expectations (percentage values)
    max_adverse_movement_pct: float = 0.5
    min_favorable_movement_pct: float = 0.2

    # Enhanced entry quality scoring
    entry_quality_scoring_method: str = "adaptive_multi_factor"  # linear_weighted, adaptive_multi_factor, information_ratio, expected_utility
    enable_interaction_terms: bool = True
    enable_penalty_system: bool = True
    risk_aversion: float = 2.0  # For expected_utility method

    # Regime-aware settings
    enable_regime_adaptive_labeling: bool = True
    regime_specific_thresholds: Dict[str, Dict[str, float]] = field(default_factory=dict)


class TacticianDifferentiatedLabeler:
    """Create differentiated entry timing labels for the Tactician pipeline."""

    def __init__(self, config: TacticianLabelingConfig):
        self.config = config
        self.logger = system_logger.getChild('TacticianDifferentiatedLabeler')
        
        # Initialize enhanced quality scorer
        self._initialize_quality_scorer()
    
    def _initialize_quality_scorer(self):
        """Initialize the enhanced entry quality scorer based on configuration."""
        try:
            from src.training.steps.models_training.enhanced_entry_quality_scorer import (
                create_enhanced_scorer,
                ScoringMethod,
                EnhancedScoringConfig
            )
            
            # Map config string to ScoringMethod enum
            scoring_method_map = {
                'linear_weighted': ScoringMethod.LINEAR_WEIGHTED,
                'adaptive_multi_factor': ScoringMethod.ADAPTIVE_MULTI_FACTOR,
                'information_ratio': ScoringMethod.INFORMATION_RATIO,
                'expected_utility': ScoringMethod.EXPECTED_UTILITY,
            }
            
            method = scoring_method_map.get(
                self.config.entry_quality_scoring_method,
                ScoringMethod.ADAPTIVE_MULTI_FACTOR
            )
            
            # Create scorer configuration (converting percent to decimal)
            scorer_config = EnhancedScoringConfig(
                scoring_method=method,
                max_adverse_movement_decimal=self.config.max_adverse_movement_pct / 100.0,  # Convert % to decimal
                min_favorable_movement_decimal=self.config.min_favorable_movement_pct / 100.0,  # Convert % to decimal
                min_quality_threshold=self.config.entry_quality_threshold,
                use_regime_adaptation=self.config.enable_regime_adaptive_labeling,
                enable_interaction_terms=self.config.enable_interaction_terms,
                enable_penalty_system=self.config.enable_penalty_system,
                risk_aversion=self.config.risk_aversion,
            )
            
            self.quality_scorer = create_enhanced_scorer(
                method=method,
                **{k: v for k, v in scorer_config.__dict__.items() if k != 'scoring_method'}
            )
            
            tprint_success(f"✅ Enhanced quality scorer initialized: {method.value}")
            
        except ImportError as e:
            tprint_warning(f"⚠️ Enhanced quality scorer not available, using fallback: {e}")
            self.quality_scorer = None

    def create_entry_timing_labels(
        self,
        data: pd.DataFrame,
        analyst_signals: Optional[pd.Series] = None,
        regime_assignments: Optional[pd.Series] = None
    ) -> Tuple[pd.Series, Dict[str, float]]:
        """
        Generate entry timing labels for all data (not constrained to Analyst signals).
        
        CHANGE: Now trains on ALL data, not just Analyst green light periods.
        """
        tprint_info("🎯 Creating tactician entry timing labels for ALL market data")

        if regime_assignments is not None:
            regime_assignments = regime_assignments.reindex(data.index)

        labels = pd.Series(0.0, index=data.index, dtype=float)
        
        # CHANGE: Process ALL data, not just Analyst green light periods
        # Create sliding windows across entire dataset
        tprint_info(f"📊 Processing {len(data)} candles for entry opportunities")

        entry_points: List[pd.Timestamp] = []
        
        # Scan entire dataset with sliding window
        window_size = self.config.max_entry_window_minutes
        
        for i in range(len(data) - window_size):
            # Current potential entry point
            entry_idx = i
            entry_index = data.index[entry_idx]
            
            # Future window for quality assessment
            future_window = data.iloc[entry_idx + 1:entry_idx + 1 + window_size]
            
            if future_window.empty:
                continue
            
            # Calculate entry quality score
            score = self._calculate_entry_quality_score(
                data.iloc[entry_idx],
                future_window,
                entry_index,
                regime_assignments
            )
            
            # Store score if above threshold
            if score > self.config.entry_quality_threshold:
                labels.loc[entry_index] = score
                entry_points.append(entry_index)
        
        # Apply peak detection to identify local maxima
        if len(entry_points) > 0:
            labels = self._apply_peak_filtering(labels)
            entry_points = labels.index[labels > 0].tolist()

        quality_metrics = self._calculate_labeling_quality_metrics_all_data(
            data,
            labels,
            entry_points
        )

        tprint_success(
            "✅ Entry labeling completed on ALL data ("
            f"{int((labels > 0).sum())} optimal entries, quality={quality_metrics.get('overall_quality', 0):.3f})"
        )

        return labels, quality_metrics

    def _apply_peak_filtering(self, labels: pd.Series) -> pd.Series:
        """
        Apply peak detection to filter entry labels to local maxima.
        This prevents too many entries by selecting only the best quality peaks.
        """
        # Get non-zero labels
        non_zero_mask = labels > 0
        if non_zero_mask.sum() == 0:
            return labels
        
        # Extract scores
        scores = labels[non_zero_mask].values
        indices = labels[non_zero_mask].index
        
        # Apply peak detection
        peaks, properties = find_peaks(
            scores,
            height=self.config.entry_quality_threshold,
            distance=max(1, self.config.min_entry_window_minutes)
        )
        
        # Create filtered labels
        filtered_labels = pd.Series(0.0, index=labels.index, dtype=float)
        
        if len(peaks) > 0:
            peak_indices = [indices[p] for p in peaks if p < len(indices)]
            peak_scores = [scores[p] for p in peaks if p < len(scores)]
            
            for idx, score in zip(peak_indices, peak_scores):
                filtered_labels.loc[idx] = score
        
        # If no peaks found but we have high-quality entries, keep the best
        if filtered_labels.sum() == 0 and len(scores) > 0:
            best_idx = np.argmax(scores)
            if best_idx < len(indices):
                filtered_labels.loc[indices[best_idx]] = scores[best_idx]
        
        return filtered_labels

    def _calculate_labeling_quality_metrics_all_data(
        self,
        data: pd.DataFrame,
        labels: pd.Series,
        entry_points: List[Any]
    ) -> Dict[str, float]:
        """
        Calculate quality metrics for labeling across all data.
        """
        total_samples = len(data)
        labeled_samples = int((labels > 0).sum())
        
        metrics: Dict[str, float] = {
            'labeling_coverage': labeled_samples / total_samples if total_samples else 0.0,
            'entry_density': labeled_samples / total_samples if total_samples else 0.0,
        }
        
        positive_scores = labels[labels > 0]
        if not positive_scores.empty:
            metrics['avg_entry_quality'] = float(positive_scores.mean())
            metrics['min_entry_quality'] = float(positive_scores.min())
            metrics['max_entry_quality'] = float(positive_scores.max())
            std_value = float(positive_scores.std())
            if np.isnan(std_value):
                std_value = 0.0
            metrics['entry_quality_std'] = std_value
        else:
            metrics['avg_entry_quality'] = 0.0
            metrics['entry_quality_std'] = 0.0
        
        # Overall quality score
        metrics['overall_quality'] = (
            metrics.get('entry_density', 0.0) * 0.3 +
            metrics.get('avg_entry_quality', 0.0) * 0.7
        )
        
        return metrics

    def _calculate_entry_quality_score(
        self,
        entry_point: pd.Series,
        future_data: pd.DataFrame,
        index_label: Any,
        regime_assignments: Optional[pd.Series]
    ) -> float:
        """
        Calculate entry quality score using enhanced scoring system.
        
        CHANGE: Now uses EnhancedEntryQualityScorer with adaptive multi-factor scoring.
        """
        if future_data.empty:
            return 0.0
        
        # Use enhanced scorer if available
        if self.quality_scorer is not None:
            # Determine regime
            regime = None
            if regime_assignments is not None and self.config.enable_regime_adaptive_labeling:
                if index_label in regime_assignments.index:
                    regime_value = regime_assignments.loc[index_label]
                    regime = f"regime_{regime_value}"
            
            # Build market context (can be expanded with more features)
            market_context = {}
            
            # Calculate quality using enhanced scorer
            quality_score = self.quality_scorer.calculate_entry_quality(
                entry_point=entry_point,
                future_data=future_data,
                regime=regime,
                market_context=market_context
            )
            
            return quality_score
        
        # Fallback to old method if enhanced scorer not available
        regime_params = self._get_regime_parameters(index_label, regime_assignments)

        entry_price = entry_point['close']
        min_future_low = future_data['low'].min()
        max_future_high = future_data['high'].max()

        adverse_move = max(entry_price - min_future_low, 0.0) / max(entry_price, 1e-8) * 100
        favorable_move = max(max_future_high - entry_price, 0.0) / max(entry_price, 1e-8) * 100

        if adverse_move > regime_params['max_adverse_movement_pct']:
            return 0.0

        if favorable_move < regime_params['min_favorable_movement_pct']:
            return 0.0

        risk_reward_ratio = favorable_move / (adverse_move + 1e-8)
        timing_score = 1.0 / (1.0 + len(future_data) / self.config.max_entry_window_minutes)
        volatility = future_data['close'].pct_change().std() or 0.0
        volatility_score = 1.0 / (1.0 + (volatility * 100) / 10.0)

        quality_score = (
            risk_reward_ratio * 0.4 +
            timing_score * 0.3 +
            volatility_score * 0.3
        )

        return float(min(max(quality_score, 0.0), 1.0))

    def _get_regime_parameters(
        self,
        index_label: Any,
        regime_assignments: Optional[pd.Series]
    ) -> Dict[str, float]:
        """Retrieve regime-specific thresholds when available."""
        if regime_assignments is not None and self.config.enable_regime_adaptive_labeling:
            regime_value = regime_assignments.loc[index_label] if index_label in regime_assignments.index else None
            if regime_value is not None:
                regime_key = f"regime_{regime_value}"
                if regime_key in self.config.regime_specific_thresholds:
                    return self.config.regime_specific_thresholds[regime_key]

        return {
            'max_adverse_movement_pct': self.config.max_adverse_movement_pct,
            'min_favorable_movement_pct': self.config.min_favorable_movement_pct
        }


@register_component('tactician_entry_labeler')
class TacticianEntryLabelerComponent(BasePreTrainingComponent):
    """
    Component wrapper for Tactician Entry Labeler.
    
    This component integrates the TacticianDifferentiatedLabeler with the pre-training pipeline
    and handles proper error handling, reporting, and pipeline state management.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the Tactician entry labeler component."""
        super().__init__(config)
        self.logger = system_logger.getChild('TacticianEntryLabelerComponent')
        
        # Create Tactician-specific configuration
        tactician_config = TacticianLabelingConfig()
        
        # Override with custom parameters if provided
        if self.config and self.config.custom_params:
            custom_params = self.config.custom_params
            
            # Update parameters
            for key in ['min_entry_window_minutes', 'max_entry_window_minutes', 
                       'entry_quality_threshold', 'max_adverse_movement_pct', 
                       'min_favorable_movement_pct', 'entry_quality_scoring_method',
                       'enable_regime_adaptive_labeling']:
                if key in custom_params:
                    setattr(tactician_config, key, custom_params[key])
        
        # Create the labeler
        try:
            self.labeler = TacticianDifferentiatedLabeler(tactician_config)
            tprint_success("✅ TacticianEntryLabelerComponent initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize TacticianEntryLabelerComponent: {e}")
            raise
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result', 'labeling_report']
    
    async def execute(self, data: Any, pipeline_state: PipelineState) -> ComponentResult:
        """
        Execute Tactician entry labeling as a component.
        
        Args:
            data: Input data (typically market data DataFrame)
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with labeling results and artifacts
        """
        try:
            tprint_info("🚀 Starting Tactician Entry Labeling execution...")
            
            # Extract data from pipeline state if not provided
            if data is None:
                data = pipeline_state.get('prepared_data')
                if data is None:
                    raise ValueError("No input data provided and no prepared_data in pipeline state")
            
            # Extract analyst signals and regime assignments if available
            analyst_predictions = pipeline_state.get('analyst_predictions')
            analyst_signals = None
            if analyst_predictions is not None:
                if isinstance(analyst_predictions, pd.DataFrame):
                    # Try to extract signals from various possible column names
                    for col in ['analyst_signal', 'green_light', 'signal', 'confidence']:
                        if col in analyst_predictions.columns:
                            analyst_signals = analyst_predictions[col]
                            break
            
            regime_assignments = pipeline_state.get('regime_assignments')
            if regime_assignments is not None:
                if isinstance(regime_assignments, pd.DataFrame):
                    regime_assignments = regime_assignments.iloc[:, 0]  # Take first column
                tprint_info(f"📊 Using regime assignments for adaptive labeling")
            
            # Generate labels
            labels, quality_metrics = self.labeler.create_entry_timing_labels(
                data=data,
                analyst_signals=analyst_signals,
                regime_assignments=regime_assignments
            )
            
            # Create labels DataFrame
            label_column = 'tactician_entry_target'
            label_df = pd.DataFrame({label_column: labels}, index=data.index)
            confidence_df = pd.DataFrame(
                {f'{label_column}_confidence': labels.clip(lower=0.0, upper=1.0)},
                index=data.index
            )
            eligibility_df = pd.DataFrame(
                {f'{label_column}_eligibility': (labels > 0).astype(int)},
                index=data.index
            )
            
            # Create quality scores in expected format
            quality_scores = {
                label_column: {
                    'overall_quality': quality_metrics.get('overall_quality', 0.0),
                    'predictability': quality_metrics.get('avg_entry_quality', 0.0),
                    'stability': max(0.0, 1.0 - quality_metrics.get('entry_quality_std', 0.0)),
                    'balance': quality_metrics.get('labeling_coverage', 0.0),
                    'auc_mean': quality_metrics.get('avg_entry_quality', 0.0),
                    'class_balance': quality_metrics.get('entry_density', 0.0)
                }
            }
            
            # Create artifacts
            artifacts = {
                'multi_horizon_labeling_result': {
                    'labeled_data': label_df,
                    'labels': label_df,
                    'confidence_scores': confidence_df,
                    'eligibility_masks': eligibility_df,
                    'quality_scores': quality_scores,
                    'quality_summary': quality_metrics,
                    'method': 'tactician_entry_labeling',
                    'metadata': {
                        'symbol': self.config.symbol if self.config else 'UNKNOWN',
                        'exchange': self.config.exchange if self.config else 'UNKNOWN',
                        'timeframe': self.config.timeframe if self.config else '15m',
                        'label_focus': 'entry_timing',
                        'regime_aware': bool(regime_assignments is not None),
                        'processing_time': 0.0,
                        'n_samples': len(label_df),
                        'n_targets': 1,
                        'n_horizons': 1,
                        'source': 'all_market_data'
                    }
                },
                'labeling_report': {
                    'status': 'completed',
                    'timestamp': datetime.now().isoformat(),
                    'method': 'tactician_entry_labeling',
                    'summary': quality_metrics,
                    'entry_points': int((labels > 0).sum()),
                    'regime_aware': bool(regime_assignments is not None)
                }
            }
            
            # Create result
            result = ComponentResult(
                success=True,
                data=label_df,
                artifacts=artifacts,
                metadata={
                    'component': 'tactician_entry_labeler',
                    'timeframe': self.config.timeframe if self.config else '15m',
                    'n_entry_points': int((labels > 0).sum()),
                    'quality_metrics': quality_metrics,
                }
            )
            
            tprint_success("✅ Tactician Entry Labeling completed successfully")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Tactician Entry Labeling failed: {e}")
            
            result = ComponentResult(
                success=False,
                error_message=str(e),
                metadata={'component': 'tactician_entry_labeler'}
            )
            return result


# Convenience function for external usage
async def execute_tactician_entry_labeling(
    data: pd.DataFrame,
    analyst_signals: Optional[pd.Series] = None,
    regime_assignments: Optional[pd.Series] = None,
    config: Optional[TacticianLabelingConfig] = None,
    **kwargs
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Execute Tactician entry labeling.
    
    Args:
        data: Input market data (OHLCV format)
        analyst_signals: Optional Analyst signals (legacy support)
        regime_assignments: Optional regime assignments
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        Tuple of (labels, quality_metrics)
    """
    labeler = TacticianDifferentiatedLabeler(config or TacticianLabelingConfig())
    return labeler.create_entry_timing_labels(data, analyst_signals, regime_assignments)