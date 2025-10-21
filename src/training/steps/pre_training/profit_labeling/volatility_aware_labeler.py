"""
Volatility Aware Multi-Horizon Labeler

This module provides a simplified volatility-aware labeling system for the pre-training pipeline.
It's designed to work with the feature_generation_labeling_integration_step.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from datetime import datetime
import warnings

# Import the enhanced labeler from research
try:
    from src.research.profit_labeling.enhanced_multi_horizon_labeler import (
        EnhancedMultiHorizonLabeler,
        EnhancedLabelingConfig,
        EnhancementLevel,
        create_enhanced_labeler
    )
    ENHANCED_LABELER_AVAILABLE = True
except ImportError:
    ENHANCED_LABELER_AVAILABLE = False

@dataclass
class VolatilityAwareConfig:
    """Configuration for volatility-aware labeling."""
    # Basic parameters
    min_horizon: int = 1
    max_horizon: int = 20
    volatility_threshold: float = 0.02
    profit_threshold: float = 0.001
    
    # Advanced parameters
    use_enhanced_labeler: bool = True
    enhancement_level: str = "basic"  # basic, ml_enhanced, adaptive, ensemble, fully_optimized

class LabelDefinitionType(Enum):
    """Types of label definitions."""
    SIMPLE_RETURNS = "simple_returns"
    VOLATILITY_AWARE = "volatility_aware"
    ENHANCED = "enhanced"

@dataclass
class LabelingResult:
    """Result from labeling process."""
    labels: pd.DataFrame
    metadata: Dict[str, Any]
    success: bool
    error_message: Optional[str] = None

class VolatilityAwareMultiHorizonLabeler:
    """Volatility-aware multi-horizon labeler."""
    
    def __init__(self, config: Optional[VolatilityAwareConfig] = None):
        self.config = config or VolatilityAwareConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize enhanced labeler if available
        if self.config.use_enhanced_labeler and ENHANCED_LABELER_AVAILABLE:
            try:
                enhancement_level = EnhancementLevel(self.config.enhancement_level)
                self.enhanced_labeler = create_enhanced_labeler(enhancement_level)
            except Exception as e:
                self.logger.warning(f"Failed to initialize enhanced labeler: {e}")
                self.enhanced_labeler = None
        else:
            self.enhanced_labeler = None
    
    def generate_labels(self, data: pd.DataFrame) -> LabelingResult:
        """Generate labels for the given data."""
        try:
            # Validate input data
            if data.empty or 'close' not in data.columns:
                return LabelingResult(
                    labels=pd.DataFrame(),
                    metadata={},
                    success=False,
                    error_message="Invalid data: empty or missing 'close' column"
                )
            
            # Use enhanced labeler if available
            if self.enhanced_labeler is not None:
                return self._generate_enhanced_labels(data)
            else:
                return self._generate_simple_labels(data)
                
        except Exception as e:
            self.logger.error(f"Error generating labels: {e}")
            return LabelingResult(
                labels=pd.DataFrame(),
                metadata={},
                success=False,
                error_message=str(e)
            )
    
    def _generate_enhanced_labels(self, data: pd.DataFrame) -> LabelingResult:
        """Generate labels using the enhanced labeler."""
        try:
            # Create configuration for enhanced labeler
            config = EnhancedLabelingConfig(
                enhancement_level=EnhancementLevel(self.config.enhancement_level)
            )
            
            # Generate labels
            result = self.enhanced_labeler.generate_labels(data, config)
            
            # Convert to our format
            if hasattr(result, 'labels') and result.labels is not None:
                labels_df = result.labels
                if isinstance(labels_df, pd.Series):
                    labels_df = labels_df.to_frame('target')
                
                return LabelingResult(
                    labels=labels_df,
                    metadata=getattr(result, 'metadata', {}),
                    success=True
                )
            else:
                return self._generate_simple_labels(data)
                
        except Exception as e:
            self.logger.warning(f"Enhanced labeler failed, falling back to simple: {e}")
            return self._generate_simple_labels(data)
    
    def _generate_simple_labels(self, data: pd.DataFrame) -> LabelingResult:
        """Generate simple volatility-aware labels."""
        try:
            # Calculate basic returns
            close_prices = data['close'].copy()
            returns = close_prices.pct_change().fillna(0)
            
            # Calculate rolling volatility
            volatility = returns.rolling(window=20).std().fillna(returns.std())
            
            # Create volatility-aware labels
            labels = []
            metadata = {
                'method': 'simple_volatility_aware',
                'volatility_threshold': self.config.volatility_threshold,
                'profit_threshold': self.config.profit_threshold
            }
            
            for i in range(len(close_prices)):
                if i < self.config.min_horizon:
                    labels.append(0.0)
                    continue
                
                # Look ahead for profit opportunities
                current_price = close_prices.iloc[i]
                current_vol = volatility.iloc[i]
                
                # Find best return in the horizon
                max_return = 0.0
                for h in range(self.config.min_horizon, min(self.config.max_horizon + 1, len(close_prices) - i)):
                    future_price = close_prices.iloc[i + h]
                    future_return = (future_price - current_price) / current_price
                    
                    # Adjust for volatility
                    if current_vol > self.config.volatility_threshold:
                        # In high volatility, require higher returns
                        adjusted_return = future_return / (1 + current_vol)
                    else:
                        # In low volatility, use returns as-is
                        adjusted_return = future_return
                    
                    max_return = max(max_return, adjusted_return)
                
                # Apply profit threshold
                if max_return > self.config.profit_threshold:
                    labels.append(max_return)
                else:
                    labels.append(0.0)
            
            # Create labels DataFrame
            labels_series = pd.Series(labels, index=data.index, name='target')
            labels_df = labels_series.to_frame()
            
            # Add metadata
            metadata.update({
                'total_samples': len(labels),
                'positive_labels': int((labels_series > 0).sum()),
                'positive_rate': float((labels_series > 0).mean()),
                'mean_return': float(labels_series.mean()),
                'std_return': float(labels_series.std())
            })
            
            return LabelingResult(
                labels=labels_df,
                metadata=metadata,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in simple labeling: {e}")
            return LabelingResult(
                labels=pd.DataFrame(),
                metadata={},
                success=False,
                error_message=str(e)
            )

def create_enhanced_analyst_labeler() -> VolatilityAwareMultiHorizonLabeler:
    """Create an enhanced analyst labeler with optimal configuration."""
    config = VolatilityAwareConfig(
        min_horizon=1,
        max_horizon=15,
        volatility_threshold=0.015,
        profit_threshold=0.0008,
        use_enhanced_labeler=True,
        enhancement_level="basic"  # Start with basic, can be upgraded
    )
    
    return VolatilityAwareMultiHorizonLabeler(config)

# Export the main classes and functions
__all__ = [
    "VolatilityAwareMultiHorizonLabeler",
    "VolatilityAwareConfig",
    "LabelingResult",
    "LabelDefinitionType",
    "create_enhanced_analyst_labeler"
]