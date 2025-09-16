"""
Enhanced Triple Barrier Labeling with Profit Potential Categories

This module provides an enhanced triple barrier labeling system that generates
meaningful profit potential labels instead of simple +1/-1 binary classifications.
The system creates rich, ML-friendly labels that capture the magnitude and
confidence of profit opportunities.

Key Features:
- Profit potential categories (High, Medium, Low, Break-even, Small/Medium/Large Loss)
- Dynamic profit magnitude scoring (0-10 scale)
- Confidence scoring based on barrier hit speed and market conditions
- Regime-specific profit potential adjustments
- Volatility-normalized profit expectations
- ML-friendly feature engineering from profit labels
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
from enum import Enum

import pandas as pd
import numpy as np

from src.utils.tprint import tprint
from src.utils.logger import get_logger
from src.training.steps.market_analysis.triple_barrier_labeling.unified_labeler import (
    UnifiedTripleBarrierLabeler, TripleBarrierConfig, TripleBarrierResult
)

class ProfitPotentialCategory(Enum):
    """Profit potential categories with meaningful labels."""
    EXTREME_LOSS = "extreme_loss"      # < -5%
    LARGE_LOSS = "large_loss"          # -5% to -2%
    MEDIUM_LOSS = "medium_loss"        # -2% to -0.5%
    SMALL_LOSS = "small_loss"          # -0.5% to -0.1%
    BREAK_EVEN = "break_even"          # -0.1% to 0.1%
    LOW_PROFIT = "low_profit"          # 0.1% to 0.5%
    MEDIUM_PROFIT = "medium_profit"    # 0.5% to 2%
    HIGH_PROFIT = "high_profit"        # 2% to 5%
    EXTREME_PROFIT = "extreme_profit"  # > 5%

class ConfidenceLevel(Enum):
    """Confidence levels for profit potential predictions."""
    VERY_LOW = "very_low"      # 0.0 - 0.2
    LOW = "low"                # 0.2 - 0.4
    MEDIUM = "medium"          # 0.4 - 0.6
    HIGH = "high"              # 0.6 - 0.8
    VERY_HIGH = "very_high"    # 0.8 - 1.0

@dataclass
class EnhancedTripleBarrierConfig:
    """Configuration for enhanced triple barrier labeling."""
    
    # Base triple barrier parameters
    profit_take_multiplier: float = 0.002
    stop_loss_multiplier: float = 0.001
    time_barrier_minutes: int = 30
    max_lookahead: int = 100
    transaction_cost: float = 0.0008
    
    # Enhanced labeling parameters
    enable_profit_categories: bool = True
    enable_magnitude_scoring: bool = True
    enable_confidence_scoring: bool = True
    enable_regime_adjustments: bool = True
    enable_volatility_normalization: bool = True
    
    # Profit category thresholds (as percentages)
    profit_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'extreme_profit': 0.05,    # 5%
        'high_profit': 0.02,       # 2%
        'medium_profit': 0.005,    # 0.5%
        'low_profit': 0.001,       # 0.1%
        'break_even': 0.001,       # 0.1%
        'small_loss': -0.001,      # -0.1%
        'medium_loss': -0.005,     # -0.5%
        'large_loss': -0.02,       # -2%
        'extreme_loss': -0.05      # -5%
    })
    
    # Confidence scoring parameters
    confidence_weights: Dict[str, float] = field(default_factory=lambda: {
        'barrier_hit_speed': 0.4,      # How quickly barriers were hit
        'profit_magnitude': 0.3,       # Size of profit/loss
        'market_conditions': 0.2,      # Volatility and regime
        'consistency': 0.1             # Consistency with similar patterns
    })
    
    # Regime-specific adjustments
    regime_adjustments: Dict[int, float] = field(default_factory=lambda: {
        0: 1.0,    # Bull market - no adjustment
        1: 0.8,    # Bear market - reduce profit expectations
        2: 1.2,    # High volatility - increase profit potential
        3: 0.9     # Low volatility - slightly reduce expectations
    })
    
    # Volatility normalization parameters
    volatility_window: int = 20
    volatility_threshold: float = 0.02  # 2% daily volatility threshold
    
    # ML feature engineering
    enable_ml_features: bool = True
    create_one_hot_encoding: bool = True
    create_ordinal_encoding: bool = True
    create_continuous_features: bool = True

@dataclass
class EnhancedTripleBarrierResult:
    """Result of enhanced triple barrier labeling."""
    
    # Core results
    success: bool
    labeled_data: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    
    # Enhanced labeling results
    profit_categories: Dict[str, int] = field(default_factory=dict)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    magnitude_score_stats: Dict[str, float] = field(default_factory=dict)
    
    # Execution metadata
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(default_factory=datetime.now)
    execution_duration: float = 0.0
    
    # Quality metrics
    label_quality_score: float = 0.0
    profit_distribution_balance: float = 0.0
    confidence_calibration: float = 0.0

class EnhancedTripleBarrierLabeler:
    """Enhanced triple barrier labeler with profit potential categories."""
    
    def __init__(self, config: Optional[EnhancedTripleBarrierConfig] = None):
        """Initialize the enhanced triple barrier labeler."""
        self.config = config or EnhancedTripleBarrierConfig()
        self.logger = get_logger('EnhancedTripleBarrierLabeler')
        
        # Initialize base triple barrier labeler
        base_config = TripleBarrierConfig(
            profit_take_multiplier=self.config.profit_take_multiplier,
            stop_loss_multiplier=self.config.stop_loss_multiplier,
            time_barrier_minutes=self.config.time_barrier_minutes,
            max_lookahead=self.config.max_lookahead,
            transaction_cost=self.config.transaction_cost,
            binary_classification=False  # We want all samples for enhanced processing
        )
        self.base_labeler = UnifiedTripleBarrierLabeler(base_config)
        
        # Initialize profit category mapping
        self._initialize_profit_categories()
        
        self.logger.info("🚀 Enhanced Triple Barrier Labeler initialized")
        tprint("🚀 Enhanced Triple Barrier Labeler initialized")
    
    def _initialize_profit_categories(self):
        """Initialize profit category mappings."""
        self.profit_category_map = {
            ProfitPotentialCategory.EXTREME_LOSS: (-np.inf, self.config.profit_thresholds['extreme_loss']),
            ProfitPotentialCategory.LARGE_LOSS: (self.config.profit_thresholds['extreme_loss'], self.config.profit_thresholds['large_loss']),
            ProfitPotentialCategory.MEDIUM_LOSS: (self.config.profit_thresholds['large_loss'], self.config.profit_thresholds['medium_loss']),
            ProfitPotentialCategory.SMALL_LOSS: (self.config.profit_thresholds['medium_loss'], self.config.profit_thresholds['small_loss']),
            ProfitPotentialCategory.BREAK_EVEN: (self.config.profit_thresholds['small_loss'], self.config.profit_thresholds['break_even']),
            ProfitPotentialCategory.LOW_PROFIT: (self.config.profit_thresholds['break_even'], self.config.profit_thresholds['low_profit']),
            ProfitPotentialCategory.MEDIUM_PROFIT: (self.config.profit_thresholds['low_profit'], self.config.profit_thresholds['medium_profit']),
            ProfitPotentialCategory.HIGH_PROFIT: (self.config.profit_thresholds['medium_profit'], self.config.profit_thresholds['high_profit']),
            ProfitPotentialCategory.EXTREME_PROFIT: (self.config.profit_thresholds['high_profit'], np.inf)
        }
    
    def apply_enhanced_labeling(self, data: pd.DataFrame) -> EnhancedTripleBarrierResult:
        """Apply enhanced triple barrier labeling with profit potential categories."""
        start_time = datetime.now()
        result = EnhancedTripleBarrierResult(
            success=False,
            start_time=start_time
        )
        
        try:
            tprint("🏷️ Starting Enhanced Triple Barrier Labeling")
            self.logger.info("🏷️ Starting Enhanced Triple Barrier Labeling")
            
            # Step 1: Apply base triple barrier labeling
            tprint("📊 Step 1: Applying base triple barrier labeling...")
            base_result = self.base_labeler.apply_labeling(data)
            
            if not base_result.success:
                raise Exception(f"Base labeling failed: {base_result.error_message}")
            
            labeled_data = base_result.labeled_data.copy()
            tprint(f"✅ Base labeling completed: {len(labeled_data)} samples")
            
            # Step 2: Calculate profit potential categories
            if self.config.enable_profit_categories:
                tprint("📊 Step 2: Calculating profit potential categories...")
                labeled_data = self._add_profit_categories(labeled_data)
                tprint("✅ Profit categories calculated")
            
            # Step 3: Calculate profit magnitude scores
            if self.config.enable_magnitude_scoring:
                tprint("📊 Step 3: Calculating profit magnitude scores...")
                labeled_data = self._add_magnitude_scores(labeled_data)
                tprint("✅ Magnitude scores calculated")
            
            # Step 4: Calculate confidence scores
            if self.config.enable_confidence_scoring:
                tprint("📊 Step 4: Calculating confidence scores...")
                labeled_data = self._add_confidence_scores(labeled_data)
                tprint("✅ Confidence scores calculated")
            
            # Step 5: Apply regime adjustments
            if self.config.enable_regime_adjustments and 'hmm_regime' in labeled_data.columns:
                tprint("📊 Step 5: Applying regime adjustments...")
                labeled_data = self._apply_regime_adjustments(labeled_data)
                tprint("✅ Regime adjustments applied")
            
            # Step 6: Apply volatility normalization
            if self.config.enable_volatility_normalization:
                tprint("📊 Step 6: Applying volatility normalization...")
                labeled_data = self._apply_volatility_normalization(labeled_data)
                tprint("✅ Volatility normalization applied")
            
            # Step 7: Create ML-friendly features
            if self.config.enable_ml_features:
                tprint("📊 Step 7: Creating ML-friendly features...")
                labeled_data = self._create_ml_features(labeled_data)
                tprint("✅ ML features created")
            
            # Step 8: Calculate result metrics
            tprint("📊 Step 8: Calculating result metrics...")
            self._populate_result_metrics(result, labeled_data)
            tprint("✅ Result metrics calculated")
            
            # Mark as successful
            result.success = True
            result.labeled_data = labeled_data
            result.end_time = datetime.now()
            result.execution_duration = (result.end_time - result.start_time).total_seconds()
            
            tprint("✅ Enhanced triple barrier labeling completed successfully")
            tprint(f"   Duration: {result.execution_duration:.2f}s")
            tprint(f"   Samples: {len(labeled_data)}")
            tprint(f"   Categories: {len(result.profit_categories)}")
            tprint(f"   Quality Score: {result.label_quality_score:.2%}")
            
            return result
            
        except Exception as e:
            result.success = False
            result.error_message = str(e)
            result.end_time = datetime.now()
            result.execution_duration = (result.end_time - result.start_time).total_seconds()
            
            tprint(f"❌ Enhanced triple barrier labeling failed: {e}")
            self.logger.error(f"❌ Enhanced triple barrier labeling failed: {e}")
            
            return result
    
    def _add_profit_categories(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add profit potential categories to the data."""
        profit_pcts = data['potential_profit_pct'].values
        
        # Create category labels
        categories = []
        category_codes = []
        
        for profit_pct in profit_pcts:
            category = self._get_profit_category(profit_pct)
            categories.append(category.value)
            category_codes.append(category.name)
        
        data['profit_category'] = categories
        data['profit_category_code'] = category_codes
        
        # Create ordinal encoding (0-8 scale)
        ordinal_mapping = {
            'EXTREME_LOSS': 0,
            'LARGE_LOSS': 1,
            'MEDIUM_LOSS': 2,
            'SMALL_LOSS': 3,
            'BREAK_EVEN': 4,
            'LOW_PROFIT': 5,
            'MEDIUM_PROFIT': 6,
            'HIGH_PROFIT': 7,
            'EXTREME_PROFIT': 8
        }
        data['profit_category_ordinal'] = data['profit_category_code'].map(ordinal_mapping)
        
        return data
    
    def _get_profit_category(self, profit_pct: float) -> ProfitPotentialCategory:
        """Get profit category for a given profit percentage."""
        for category, (min_val, max_val) in self.profit_category_map.items():
            if min_val <= profit_pct < max_val:
                return category
        return ProfitPotentialCategory.BREAK_EVEN  # Default fallback
    
    def _add_magnitude_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add profit magnitude scores (0-10 scale)."""
        profit_pcts = data['potential_profit_pct'].values
        
        # Calculate magnitude scores
        magnitude_scores = []
        for profit_pct in profit_pcts:
            if profit_pct >= 0:
                # Positive profits: scale 5-10
                if profit_pct >= 0.05:  # 5%+
                    score = 10.0
                elif profit_pct >= 0.02:  # 2-5%
                    score = 8.0 + (profit_pct - 0.02) / 0.03 * 2.0
                elif profit_pct >= 0.005:  # 0.5-2%
                    score = 6.0 + (profit_pct - 0.005) / 0.015 * 2.0
                elif profit_pct >= 0.001:  # 0.1-0.5%
                    score = 5.0 + (profit_pct - 0.001) / 0.004 * 1.0
                else:  # 0-0.1%
                    score = 5.0
            else:
                # Negative profits: scale 0-5
                if profit_pct <= -0.05:  # -5% or worse
                    score = 0.0
                elif profit_pct <= -0.02:  # -5% to -2%
                    score = 1.0 + (profit_pct + 0.05) / 0.03 * 1.0
                elif profit_pct <= -0.005:  # -2% to -0.5%
                    score = 2.0 + (profit_pct + 0.02) / 0.015 * 1.0
                elif profit_pct <= -0.001:  # -0.5% to -0.1%
                    score = 3.0 + (profit_pct + 0.005) / 0.004 * 1.0
                else:  # -0.1% to 0%
                    score = 4.0 + (profit_pct + 0.001) / 0.001 * 1.0
            
            magnitude_scores.append(score)
        
        data['profit_magnitude_score'] = magnitude_scores
        
        return data
    
    def _add_confidence_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add confidence scores based on multiple factors."""
        confidence_scores = []
        
        for idx, row in data.iterrows():
            confidence = self._calculate_confidence_score(row)
            confidence_scores.append(confidence)
        
        data['confidence_score'] = confidence_scores
        
        # Add confidence categories
        confidence_categories = []
        for score in confidence_scores:
            if score >= 0.8:
                category = ConfidenceLevel.VERY_HIGH
            elif score >= 0.6:
                category = ConfidenceLevel.HIGH
            elif score >= 0.4:
                category = ConfidenceLevel.MEDIUM
            elif score >= 0.2:
                category = ConfidenceLevel.LOW
            else:
                category = ConfidenceLevel.VERY_LOW
            confidence_categories.append(category.value)
        
        data['confidence_category'] = confidence_categories
        
        return data
    
    def _calculate_confidence_score(self, row: pd.Series) -> float:
        """Calculate confidence score for a single sample."""
        weights = self.config.confidence_weights
        
        # Factor 1: Barrier hit speed (how quickly barriers were hit)
        barrier_speed_score = self._calculate_barrier_hit_speed_score(row)
        
        # Factor 2: Profit magnitude (larger profits/losses are more confident)
        profit_magnitude_score = self._calculate_profit_magnitude_confidence(row)
        
        # Factor 3: Market conditions (volatility and regime)
        market_conditions_score = self._calculate_market_conditions_score(row)
        
        # Factor 4: Consistency (consistency with similar patterns)
        consistency_score = self._calculate_consistency_score(row)
        
        # Weighted combination
        confidence = (
            weights['barrier_hit_speed'] * barrier_speed_score +
            weights['profit_magnitude'] * profit_magnitude_score +
            weights['market_conditions'] * market_conditions_score +
            weights['consistency'] * consistency_score
        )
        
        return np.clip(confidence, 0.0, 1.0)
    
    def _calculate_barrier_hit_speed_score(self, row: pd.Series) -> float:
        """Calculate score based on how quickly barriers were hit."""
        # This is a simplified version - in practice, you'd track actual hit times
        profit_pct = abs(row['potential_profit_pct'])
        
        # Higher profit percentages (achieved quickly) get higher scores
        if profit_pct >= 0.02:  # 2%+
            return 1.0
        elif profit_pct >= 0.005:  # 0.5-2%
            return 0.8
        elif profit_pct >= 0.001:  # 0.1-0.5%
            return 0.6
        else:  # < 0.1%
            return 0.4
    
    def _calculate_profit_magnitude_confidence(self, row: pd.Series) -> float:
        """Calculate confidence based on profit magnitude."""
        profit_pct = abs(row['potential_profit_pct'])
        
        # Larger profits/losses are more confident signals
        if profit_pct >= 0.01:  # 1%+
            return 1.0
        elif profit_pct >= 0.005:  # 0.5-1%
            return 0.8
        elif profit_pct >= 0.002:  # 0.2-0.5%
            return 0.6
        elif profit_pct >= 0.001:  # 0.1-0.2%
            return 0.4
        else:  # < 0.1%
            return 0.2
    
    def _calculate_market_conditions_score(self, row: pd.Series) -> float:
        """Calculate confidence based on market conditions."""
        # Simplified version - in practice, you'd use actual volatility and regime data
        base_score = 0.5
        
        # Adjust based on regime if available
        if 'hmm_regime' in row and not pd.isna(row['hmm_regime']):
            regime = int(row['hmm_regime'])
            regime_adjustment = self.config.regime_adjustments.get(regime, 1.0)
            base_score *= regime_adjustment
        
        return np.clip(base_score, 0.0, 1.0)
    
    def _calculate_consistency_score(self, row: pd.Series) -> float:
        """Calculate consistency score (placeholder for now)."""
        # This would be calculated based on historical consistency
        # For now, return a base score
        return 0.5
    
    def _apply_regime_adjustments(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply regime-specific adjustments to profit expectations."""
        adjusted_scores = []
        
        for idx, row in data.iterrows():
            if 'hmm_regime' in row and not pd.isna(row['hmm_regime']):
                regime = int(row['hmm_regime'])
                adjustment = self.config.regime_adjustments.get(regime, 1.0)
                
                # Adjust magnitude score based on regime
                original_score = row['profit_magnitude_score']
                adjusted_score = original_score * adjustment
                adjusted_scores.append(adjusted_score)
            else:
                adjusted_scores.append(row['profit_magnitude_score'])
        
        data['profit_magnitude_score_adjusted'] = adjusted_scores
        
        return data
    
    def _apply_volatility_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply volatility normalization to profit expectations."""
        if 'close' not in data.columns:
            return data
        
        # Calculate rolling volatility
        returns = data['close'].pct_change()
        rolling_vol = returns.rolling(window=self.config.volatility_window).std()
        
        # Normalize profit expectations based on volatility
        normalized_scores = []
        for idx, row in data.iterrows():
            if not pd.isna(rolling_vol.iloc[idx]):
                vol = rolling_vol.iloc[idx]
                vol_adjustment = min(2.0, max(0.5, vol / self.config.volatility_threshold))
                
                original_score = row.get('profit_magnitude_score_adjusted', row['profit_magnitude_score'])
                normalized_score = original_score * vol_adjustment
                normalized_scores.append(normalized_score)
            else:
                normalized_scores.append(row.get('profit_magnitude_score_adjusted', row['profit_magnitude_score']))
        
        data['profit_magnitude_score_normalized'] = normalized_scores
        
        return data
    
    def _create_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create ML-friendly features from profit potential labels."""
        
        # One-hot encoding for profit categories
        if self.config.create_one_hot_encoding:
            category_dummies = pd.get_dummies(data['profit_category'], prefix='profit_cat')
            data = pd.concat([data, category_dummies], axis=1)
        
        # One-hot encoding for confidence categories
        if self.config.create_one_hot_encoding:
            confidence_dummies = pd.get_dummies(data['confidence_category'], prefix='conf_cat')
            data = pd.concat([data, confidence_dummies], axis=1)
        
        # Continuous features
        if self.config.create_continuous_features:
            # Profit magnitude features
            data['profit_magnitude_log'] = np.log1p(data['profit_magnitude_score'])
            data['profit_magnitude_sqrt'] = np.sqrt(data['profit_magnitude_score'])
            data['profit_magnitude_squared'] = data['profit_magnitude_score'] ** 2
            
            # Confidence features
            data['confidence_log'] = np.log1p(data['confidence_score'])
            data['confidence_squared'] = data['confidence_score'] ** 2
            
            # Interaction features
            data['profit_confidence_interaction'] = data['profit_magnitude_score'] * data['confidence_score']
            data['profit_confidence_ratio'] = np.where(
                data['confidence_score'] > 0,
                data['profit_magnitude_score'] / data['confidence_score'],
                0
            )
        
        return data
    
    def _populate_result_metrics(self, result: EnhancedTripleBarrierResult, data: pd.DataFrame):
        """Populate result metrics."""
        # Profit category distribution
        if 'profit_category' in data.columns:
            result.profit_categories = data['profit_category'].value_counts().to_dict()
        
        # Confidence distribution
        if 'confidence_category' in data.columns:
            result.confidence_distribution = data['confidence_category'].value_counts().to_dict()
        
        # Magnitude score statistics
        if 'profit_magnitude_score' in data.columns:
            scores = data['profit_magnitude_score']
            result.magnitude_score_stats = {
                'mean': float(scores.mean()),
                'std': float(scores.std()),
                'min': float(scores.min()),
                'max': float(scores.max()),
                'median': float(scores.median())
            }
        
        # Label quality score
        result.label_quality_score = self._calculate_label_quality_score(data)
        
        # Profit distribution balance
        result.profit_distribution_balance = self._calculate_profit_distribution_balance(data)
        
        # Confidence calibration
        result.confidence_calibration = self._calculate_confidence_calibration(data)
    
    def _calculate_label_quality_score(self, data: pd.DataFrame) -> float:
        """Calculate overall label quality score."""
        score = 1.0
        
        # Check for balanced profit categories
        if 'profit_category' in data.columns:
            category_counts = data['profit_category'].value_counts()
            if len(category_counts) > 1:
                balance_ratio = category_counts.min() / category_counts.max()
                score *= (0.5 + 0.5 * balance_ratio)
        
        # Check for reasonable confidence distribution
        if 'confidence_score' in data.columns:
            conf_scores = data['confidence_score']
            if conf_scores.std() > 0:
                # Good confidence distribution should have reasonable spread
                conf_quality = min(1.0, conf_scores.std() * 2)
                score *= (0.7 + 0.3 * conf_quality)
        
        return score
    
    def _calculate_profit_distribution_balance(self, data: pd.DataFrame) -> float:
        """Calculate profit distribution balance score."""
        if 'profit_category' in data.columns:
            category_counts = data['profit_category'].value_counts()
            if len(category_counts) > 1:
                return category_counts.min() / category_counts.max()
        return 0.0
    
    def _calculate_confidence_calibration(self, data: pd.DataFrame) -> float:
        """Calculate confidence calibration score."""
        if 'confidence_score' in data.columns and 'profit_magnitude_score' in data.columns:
            # Good calibration: higher confidence should correlate with higher magnitude
            correlation = data['confidence_score'].corr(data['profit_magnitude_score'])
            return max(0.0, correlation)
        return 0.0

# Convenience functions
def create_enhanced_triple_barrier_labeler(
    profit_take_multiplier: float = 0.002,
    stop_loss_multiplier: float = 0.001,
    time_barrier_minutes: int = 30,
    max_lookahead: int = 100,
    transaction_cost: float = 0.0008,
    enable_profit_categories: bool = True,
    enable_magnitude_scoring: bool = True,
    enable_confidence_scoring: bool = True,
    enable_regime_adjustments: bool = True,
    enable_volatility_normalization: bool = True,
    enable_ml_features: bool = True
) -> EnhancedTripleBarrierLabeler:
    """Create an enhanced triple barrier labeler with specified parameters."""
    config = EnhancedTripleBarrierConfig(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        max_lookahead=max_lookahead,
        transaction_cost=transaction_cost,
        enable_profit_categories=enable_profit_categories,
        enable_magnitude_scoring=enable_magnitude_scoring,
        enable_confidence_scoring=enable_confidence_scoring,
        enable_regime_adjustments=enable_regime_adjustments,
        enable_volatility_normalization=enable_volatility_normalization,
        enable_ml_features=enable_ml_features
    )
    
    return EnhancedTripleBarrierLabeler(config)

def apply_enhanced_triple_barrier_labeling(
    data: pd.DataFrame,
    profit_take_multiplier: float = 0.002,
    stop_loss_multiplier: float = 0.001,
    time_barrier_minutes: int = 30,
    max_lookahead: int = 100,
    transaction_cost: float = 0.0008,
    enable_profit_categories: bool = True,
    enable_magnitude_scoring: bool = True,
    enable_confidence_scoring: bool = True,
    enable_regime_adjustments: bool = True,
    enable_volatility_normalization: bool = True,
    enable_ml_features: bool = True
) -> EnhancedTripleBarrierResult:
    """Apply enhanced triple barrier labeling to data."""
    labeler = create_enhanced_triple_barrier_labeler(
        profit_take_multiplier=profit_take_multiplier,
        stop_loss_multiplier=stop_loss_multiplier,
        time_barrier_minutes=time_barrier_minutes,
        max_lookahead=max_lookahead,
        transaction_cost=transaction_cost,
        enable_profit_categories=enable_profit_categories,
        enable_magnitude_scoring=enable_magnitude_scoring,
        enable_confidence_scoring=enable_confidence_scoring,
        enable_regime_adjustments=enable_regime_adjustments,
        enable_volatility_normalization=enable_volatility_normalization,
        enable_ml_features=enable_ml_features
    )
    
    return labeler.apply_enhanced_labeling(data)

if __name__ == '__main__':
    # Test the enhanced implementation
    tprint('🧪 Testing Enhanced Triple Barrier Labeling')
    
    # Create test data
    dates = pd.date_range('2024-01-01', periods=1000, freq='1min')
    data = pd.DataFrame({
        'open': np.random.uniform(100, 110, 1000),
        'high': np.random.uniform(105, 115, 1000),
        'low': np.random.uniform(95, 105, 1000),
        'close': np.random.uniform(100, 110, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'hmm_regime': np.random.choice([0, 1, 2, 3], 1000)  # Add regime data
    }, index=dates)
    
    # Test enhanced labeling
    tprint('\n📊 Testing enhanced triple barrier labeling...')
    result = apply_enhanced_triple_barrier_labeling(data)
    
    if result.success:
        tprint(f'✅ Enhanced labeling completed successfully')
        tprint(f'   Duration: {result.execution_duration:.2f}s')
        tprint(f'   Samples: {len(result.labeled_data)}')
        tprint(f'   Profit categories: {result.profit_categories}')
        tprint(f'   Confidence distribution: {result.confidence_distribution}')
        tprint(f'   Magnitude stats: {result.magnitude_score_stats}')
        tprint(f'   Quality score: {result.label_quality_score:.2%}')
        
        # Show sample of enhanced labels
        sample_data = result.labeled_data.head(10)
        tprint(f'\n📋 Sample enhanced labels:')
        for idx, row in sample_data.iterrows():
            tprint(f'   {idx}: {row["profit_category"]} (score: {row["profit_magnitude_score"]:.1f}, conf: {row["confidence_score"]:.2f})')
    else:
        tprint(f'❌ Enhanced labeling failed: {result.error_message}')
    
    tprint('✅ Enhanced Triple Barrier Labeling test completed!')