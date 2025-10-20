"""
Feature Generation Feature Selection Step

This step performs feature selection as part of the feature generation pipeline
using the BaseStep architecture.
"""

from __future__ import annotations

import logging
import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

from src.training.steps.base_step import BaseStep


@dataclass
class FeatureSelectionResult:
    """Result of feature selection step."""
    
    success: bool
    selected_features: List[str]
    selection_metadata: Dict[str, Any]
    selection_metrics: Dict[str, Any]
    artifacts: Dict[str, Any]
    error_message: Optional[str] = None


class FeatureGenerationFeatureSelectionStep(BaseStep):
    """Feature selection step using BaseStep architecture."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the feature selection step."""
        super().__init__("feature_generation_feature_selection_step", config)
    
    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature selection step using BaseStep architecture."""
        
        self.logger.info("🎯 Starting feature selection step")
        
        try:
            # Extract parameters from config
            data = config.get('data')
            symbol = config.get('symbol', 'ETHUSDT')
            timeframe = config.get('timeframe', '15m')
            direction = config.get('direction', 'longs')
            intensity = config.get('intensity', 'blank')
            lookback_days = config.get('lookback_days')
            start_date = config.get('start_date')
            end_date = config.get('end_date')
            exchange = config.get('exchange', 'binance')
            custom_overrides = config.get('custom_overrides')
            
            # If no data provided, create sample data for feature selection
            if data is None:
                self.logger.warning("No data provided, creating sample data for feature selection")
                data = pd.DataFrame({
                    'open': np.random.randn(1000).cumsum() + 100,
                    'high': np.random.randn(1000).cumsum() + 105,
                    'low': np.random.randn(1000).cumsum() + 95,
                    'close': np.random.randn(1000).cumsum() + 100,
                    'volume': np.random.randint(1000, 10000, 1000)
                })
                # Add some additional features
                data['sma_20'] = data['close'].rolling(window=20).mean()
                data['rsi'] = 50 + np.random.randn(1000) * 10
                data['macd'] = np.random.randn(1000)
            
            # Perform feature selection
            selection_result = await self._perform_feature_selection(
                data=data,
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                intensity=intensity,
                lookback_days=lookback_days,
                start_date=start_date,
                end_date=end_date,
                exchange=exchange,
                custom_overrides=custom_overrides
            )
            
            # Save selection results as artifacts
            self._save_metadata(selection_result['selected_features'], 'selected_features')
            self._save_metadata(selection_result['selection_metadata'], 'selection_metadata')
            
            if selection_result['success']:
                self.logger.info(f"✅ Feature selection completed successfully with {len(selection_result['selected_features'])} selected features")
            else:
                self.logger.error(f"❌ Feature selection failed: {selection_result.get('error_message', 'Unknown error')}")
            
            return {
                'success': selection_result['success'],
                'artifacts': ['selected_features', 'selection_metadata'],
                'metrics': {
                    'selection_metrics': selection_result['selection_metrics'],
                    'selection_metadata': selection_result['selection_metadata']
                },
                'error': selection_result.get('error_message')
            }
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection step failed with exception: {e}")
            return {
                'success': False,
                'artifacts': [],
                'metrics': {},
                'error': str(e)
            }
    
    async def _perform_feature_selection(self,
                                       data: pd.DataFrame,
                                       symbol: str,
                                       timeframe: str,
                                       direction: str,
                                       intensity: str,
                                       lookback_days: Optional[int],
                                       start_date: Optional[str],
                                       end_date: Optional[str],
                                       exchange: str,
                                       custom_overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform the actual feature selection logic."""
        
        try:
            # Get numeric columns only
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target columns if they exist
            target_columns = ['profit_label', 'profit_continuous', 'volatility_label']
            feature_columns = [col for col in numeric_columns if col not in target_columns]
            
            if len(feature_columns) == 0:
                return {
                    'success': False,
                    'selected_features': [],
                    'selection_metadata': {},
                    'selection_metrics': {},
                    'error_message': 'No numeric features found for selection'
                }
            
            # Apply feature selection methods
            selected_features = []
            selection_scores = {}
            
            # 1. Variance threshold
            variance_features = self._variance_selection(data[feature_columns])
            selected_features.extend(variance_features)
            selection_scores['variance'] = len(variance_features)
            
            # 2. Correlation filtering
            correlation_features = self._correlation_selection(data[feature_columns])
            selected_features.extend(correlation_features)
            selection_scores['correlation'] = len(correlation_features)
            
            # 3. Statistical significance
            statistical_features = self._statistical_selection(data[feature_columns])
            selected_features.extend(statistical_features)
            selection_scores['statistical'] = len(statistical_features)
            
            # Remove duplicates and limit to top features
            selected_features = list(set(selected_features))
            max_features = min(len(selected_features), 20)  # Limit to top 20 features
            selected_features = selected_features[:max_features]
            
            # Generate selection metadata
            selection_metadata = {
                'selection_methods': ['variance', 'correlation', 'statistical'],
                'total_features_available': len(feature_columns),
                'features_selected': len(selected_features),
                'selection_scores': selection_scores,
                'symbol': symbol,
                'timeframe': timeframe,
                'direction': direction,
                'intensity': intensity,
                'lookback_days': lookback_days,
                'start_date': start_date,
                'end_date': end_date,
                'exchange': exchange,
                'selection_timestamp': datetime.now().isoformat()
            }
            
            # Calculate selection metrics
            selection_metrics = {
                'selection_ratio': len(selected_features) / len(feature_columns) if feature_columns else 0,
                'feature_importance_scores': self._calculate_feature_importance(data[selected_features]),
                'redundancy_score': self._calculate_redundancy_score(data[selected_features]),
                'diversity_score': self._calculate_diversity_score(data[selected_features])
            }
            
            # Apply any custom overrides
            if custom_overrides:
                selection_metadata.update(custom_overrides)
            
            return {
                'success': True,
                'selected_features': selected_features,
                'selection_metadata': selection_metadata,
                'selection_metrics': selection_metrics,
                'error_message': None
            }
            
        except Exception as e:
            return {
                'success': False,
                'selected_features': [],
                'selection_metadata': {},
                'selection_metrics': {},
                'error_message': str(e)
            }
    
    def _variance_selection(self, data: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Select features based on variance threshold."""
        try:
            variances = data.var()
            return variances[variances > threshold].index.tolist()
        except Exception:
            return []
    
    def _correlation_selection(self, data: pd.DataFrame, threshold: float = 0.95) -> List[str]:
        """Select features based on correlation filtering."""
        try:
            corr_matrix = data.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            
            # Find features to drop
            to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
            
            # Return features that are not highly correlated
            return [col for col in data.columns if col not in to_drop]
        except Exception:
            return data.columns.tolist()
    
    def _statistical_selection(self, data: pd.DataFrame) -> List[str]:
        """Select features based on statistical significance."""
        try:
            # Simple statistical test - select features with good distribution properties
            selected = []
            for col in data.columns:
                if data[col].notna().sum() > len(data) * 0.8:  # At least 80% non-null
                    if data[col].std() > 0:  # Non-zero standard deviation
                        selected.append(col)
            return selected
        except Exception:
            return data.columns.tolist()
    
    def _calculate_feature_importance(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate feature importance scores."""
        try:
            importance_scores = {}
            for col in data.columns:
                # Simple importance based on variance and non-null ratio
                variance_score = data[col].var() if data[col].var() > 0 else 0
                completeness_score = data[col].notna().mean()
                importance_scores[col] = variance_score * completeness_score
            return importance_scores
        except Exception:
            return {}
    
    def _calculate_redundancy_score(self, data: pd.DataFrame) -> float:
        """Calculate redundancy score (lower is better)."""
        try:
            if len(data.columns) < 2:
                return 0.0
            
            corr_matrix = data.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            return upper_tri.mean().mean()
        except Exception:
            return 0.0
    
    def _calculate_diversity_score(self, data: pd.DataFrame) -> float:
        """Calculate diversity score (higher is better)."""
        try:
            if len(data.columns) < 2:
                return 1.0
            
            # Calculate diversity based on different statistical properties
            diversity_scores = []
            for col in data.columns:
                col_data = data[col].dropna()
                if len(col_data) > 0:
                    # Skewness and kurtosis diversity
                    skewness = abs(col_data.skew())
                    kurtosis = abs(col_data.kurtosis())
                    diversity_scores.append(skewness + kurtosis)
            
            return np.mean(diversity_scores) if diversity_scores else 0.0
        except Exception:
            return 0.0


# Command handler for ares_launcher integration
async def handle_feature_generation_feature_selection_step(
    symbol: str = "ETHUSDT",
    timeframe: str = "15m",
    direction: str = "longs",
    intensity: str = "blank",
    lookback_days: Optional[int] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    exchange: str = "binance",
    custom_overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Handle feature generation feature selection step command.
    
    Args:
        symbol: Trading symbol (default: "ETHUSDT")
        timeframe: Timeframe (default: "15m")
        direction: Direction (default: "longs")
        intensity: Pipeline intensity (default: "blank")
        lookback_days: Lookback days (optional)
        start_date: Start date (optional)
        end_date: End date (optional)
        exchange: Exchange (default: "binance")
        custom_overrides: Custom configuration overrides (optional)
        **kwargs: Additional arguments
        
    Returns:
        Dict with feature selection results
    """
    # Create step instance and execute
    step = FeatureGenerationFeatureSelectionStep()
    
    config = {
        'symbol': symbol,
        'timeframe': timeframe,
        'direction': direction,
        'intensity': intensity,
        'lookback_days': lookback_days,
        'start_date': start_date,
        'end_date': end_date,
        'exchange': exchange,
        'custom_overrides': custom_overrides
    }
    
    return await step.run(config)