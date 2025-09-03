#!/usr/bin/env python3
"""Implementation of per-regime processing for Steps 5-7."""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
import asyncio
from datetime import datetime

from src.utils.logger import system_logger
from src.utils.pipeline_standards import pipeline_standards
from src.utils.common_operations import ensure_directory, safe_json_dump

logger = system_logger.getChild("RegimeStepsImplementation")


# ============================================================================
# STEP 5: Per-Regime Labeling Implementation
# ============================================================================

class RegimeAwareLabelingStep:
    """Enhanced Step 5 with per-regime labeling."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("RegimeAwareLabelingStep")
        self.standards = pipeline_standards
        
        # Regime-specific triple barrier parameters
        self.regime_params = {
            'bull': {
                'profit_target': 0.020,  # 2.0% profit target in bull markets
                'stop_loss': 0.010,      # 1.0% stop loss (tighter in trends)
                'time_barrier': 60,      # 60 bars max holding period
                'min_samples': 1000
            },
            'bear': {
                'profit_target': 0.015,  # 1.5% profit target (more conservative)
                'stop_loss': 0.015,      # 1.5% stop loss (symmetric)
                'time_barrier': 30,      # 30 bars (faster exits)
                'min_samples': 1000
            },
            'sideways': {
                'profit_target': 0.010,  # 1.0% profit target (smaller moves)
                'stop_loss': 0.010,      # 1.0% stop loss
                'time_barrier': 20,      # 20 bars (quick trades)
                'min_samples': 1000
            }
        }
        
    async def execute(self, data: pd.DataFrame, regime_labels: np.ndarray, 
                     symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Execute per-regime labeling."""
        
        self.logger.info(f"Starting per-regime labeling for {symbol}")
        
        # Validate inputs
        if len(data) != len(regime_labels):
            raise ValueError("Data and regime labels must have same length")
            
        results = {
            'labeled_data_by_regime': {},
            'statistics': {},
            'metadata': {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        # Process each regime separately
        for regime in ['bull', 'bear', 'sideways']:
            self.logger.info(f"Processing {regime} regime...")
            
            # Get regime-specific data
            regime_mask = regime_labels == regime
            regime_data = data[regime_mask].copy()
            
            if len(regime_data) < self.regime_params[regime]['min_samples']:
                self.logger.warning(
                    f"Insufficient data for {regime} regime: "
                    f"{len(regime_data)} samples (min: {self.regime_params[regime]['min_samples']})"
                )
                continue
                
            # Apply regime-specific labeling
            labeled_data = await self._label_regime_data(regime_data, regime)
            
            # Calculate regime statistics
            stats = self._calculate_regime_statistics(labeled_data, regime)
            
            # Store results
            results['labeled_data_by_regime'][regime] = labeled_data
            results['statistics'][regime] = stats
            
            # Save regime-specific labels
            output_path = self._get_output_path(symbol, exchange, timeframe, regime)
            labeled_data.to_parquet(output_path)
            self.logger.info(f"Saved {regime} labels to {output_path}")
            
        # Generate summary report
        results['summary'] = self._generate_summary_report(results)
        
        return results
    
    async def _label_regime_data(self, data: pd.DataFrame, regime: str) -> pd.DataFrame:
        """Apply triple barrier labeling with regime-specific parameters."""
        
        params = self.regime_params[regime]
        
        # Import triple barrier labeling
        from src.training.steps.step4_analyst_labeling_feature_engineering_components.optimized_triple_barrier_labeling import OptimizedTripleBarrierLabeling
        
        labeler = OptimizedTripleBarrierLabeling()
        
        # Configure for regime
        labeler.profit_target = params['profit_target']
        labeler.stop_loss = params['stop_loss']
        labeler.time_barrier = params['time_barrier']
        
        # Apply labeling
        labels = await labeler.label(data)
        
        # Add regime information
        labels['regime'] = regime
        labels['regime_confidence'] = self._calculate_regime_confidence(data, regime)
        
        return labels
    
    def _calculate_regime_confidence(self, data: pd.DataFrame, regime: str) -> pd.Series:
        """Calculate confidence score for regime assignment."""
        
        # Placeholder - would use HMM probabilities in real implementation
        # For now, use volatility-based confidence
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std()
        
        # Different confidence patterns per regime
        if regime == 'bull':
            # Higher confidence when volatility is moderate and returns positive
            rolling_returns = returns.rolling(20).mean()
            confidence = (1 - volatility) * (rolling_returns > 0).astype(float)
        elif regime == 'bear':
            # Higher confidence when volatility is high and returns negative
            rolling_returns = returns.rolling(20).mean()
            confidence = volatility * (rolling_returns < 0).astype(float)
        else:  # sideways
            # Higher confidence when volatility is low
            confidence = 1 - volatility
            
        # Normalize to [0, 1]
        confidence = (confidence - confidence.min()) / (confidence.max() - confidence.min())
        confidence = confidence.fillna(0.5)
        
        return confidence
    
    def _calculate_regime_statistics(self, labeled_data: pd.DataFrame, regime: str) -> Dict[str, Any]:
        """Calculate statistics for regime-labeled data."""
        
        stats = {
            'regime': regime,
            'sample_count': len(labeled_data),
            'label_distribution': labeled_data['label'].value_counts().to_dict(),
            'avg_profit_potential': labeled_data['potential_profit_pct'].mean(),
            'avg_holding_period': labeled_data['holding_period'].mean(),
            'regime_confidence': {
                'mean': labeled_data['regime_confidence'].mean(),
                'std': labeled_data['regime_confidence'].std(),
                'min': labeled_data['regime_confidence'].min(),
                'max': labeled_data['regime_confidence'].max()
            }
        }
        
        return stats
    
    def _get_output_path(self, symbol: str, exchange: str, timeframe: str, regime: str) -> Path:
        """Get output path for regime-specific labels."""
        
        output_dir = Path(self.standards.build_path("labels", exchange, symbol))
        ensure_directory(output_dir)
        
        filename = f"{exchange}_{symbol}_{timeframe}_{regime}_labels.parquet"
        return output_dir / filename
    
    def _generate_summary_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary report across all regimes."""
        
        summary = {
            'total_samples': sum(
                stats['sample_count'] 
                for stats in results['statistics'].values()
            ),
            'regime_distribution': {
                regime: stats['sample_count'] 
                for regime, stats in results['statistics'].items()
            },
            'overall_metrics': {}
        }
        
        # Calculate weighted averages
        total_samples = summary['total_samples']
        if total_samples > 0:
            for metric in ['avg_profit_potential', 'avg_holding_period']:
                weighted_sum = sum(
                    stats[metric] * stats['sample_count']
                    for stats in results['statistics'].values()
                )
                summary['overall_metrics'][metric] = weighted_sum / total_samples
                
        return summary


# ============================================================================
# STEP 6: Per-Regime Feature Engineering Implementation
# ============================================================================

class RegimeAwareFeatureEngineering:
    """Enhanced Step 6 with regime-specific feature engineering."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("RegimeAwareFeatureEngineering")
        self.standards = pipeline_standards
        
        # Define regime-specific feature sets
        self.regime_features = {
            'bull': {
                'primary': [
                    'momentum_rsi', 'momentum_macd', 'momentum_stoch',
                    'trend_strength_adx', 'trend_direction',
                    'breakout_resistance', 'volume_momentum'
                ],
                'secondary': [
                    'bollinger_position', 'atr_normalized',
                    'volume_ratio', 'price_acceleration'
                ]
            },
            'bear': {
                'primary': [
                    'support_distance', 'resistance_strength',
                    'volatility_realized', 'volatility_garch',
                    'volume_divergence', 'put_call_ratio'
                ],
                'secondary': [
                    'oversold_indicators', 'fear_greed_index',
                    'safe_haven_correlation', 'drawdown_metrics'
                ]
            },
            'sideways': {
                'primary': [
                    'mean_reversion_zscore', 'bollinger_bands_position',
                    'rsi_divergence', 'range_position',
                    'oscillator_composite', 'volume_profile'
                ],
                'secondary': [
                    'range_breakout_probability', 'mean_distance',
                    'cycle_indicators', 'market_efficiency'
                ]
            }
        }
        
    async def execute(self, labeled_data_by_regime: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Execute per-regime feature engineering."""
        
        self.logger.info("Starting per-regime feature engineering")
        
        results = {
            'features_by_regime': {},
            'feature_importance': {},
            'feature_statistics': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'feature_version': '2.0'
            }
        }
        
        # Process each regime
        for regime, labeled_data in labeled_data_by_regime.items():
            self.logger.info(f"Engineering features for {regime} regime...")
            
            # Engineer regime-specific features
            features = await self._engineer_regime_features(labeled_data, regime)
            
            # Add cross-regime features
            features = self._add_cross_regime_features(features, regime)
            
            # Feature selection
            selected_features = await self._select_features_for_regime(features, regime)
            
            # Calculate feature importance
            importance = self._calculate_feature_importance(selected_features, regime)
            
            # Calculate statistics
            stats = self._calculate_feature_statistics(selected_features)
            
            # Store results
            results['features_by_regime'][regime] = selected_features
            results['feature_importance'][regime] = importance
            results['feature_statistics'][regime] = stats
            
            # Save features
            output_path = self._get_feature_output_path(regime)
            selected_features.to_parquet(output_path)
            self.logger.info(f"Saved {regime} features to {output_path}")
            
        return results
    
    async def _engineer_regime_features(self, data: pd.DataFrame, regime: str) -> pd.DataFrame:
        """Engineer features specific to regime."""
        
        features = data.copy()
        feature_list = self.regime_features[regime]
        
        # Primary features
        for feature_name in feature_list['primary']:
            feature_values = await self._calculate_feature(data, feature_name, regime)
            features[f'{regime}_{feature_name}'] = feature_values
            
        # Secondary features
        for feature_name in feature_list['secondary']:
            feature_values = await self._calculate_feature(data, feature_name, regime)
            features[f'{regime}_{feature_name}_secondary'] = feature_values
            
        # Regime-specific combinations
        features = self._create_regime_combinations(features, regime)
        
        return features
    
    async def _calculate_feature(self, data: pd.DataFrame, feature_name: str, regime: str) -> pd.Series:
        """Calculate individual feature with regime-specific parameters."""
        
        # Feature calculation with regime-specific parameters
        if 'momentum' in feature_name:
            window = 10 if regime == 'bull' else 20 if regime == 'bear' else 15
            return self._calculate_momentum(data, window)
            
        elif 'volatility' in feature_name:
            window = 20 if regime == 'bear' else 15
            return self._calculate_volatility(data, window)
            
        elif 'mean_reversion' in feature_name:
            window = 20 if regime == 'sideways' else 30
            return self._calculate_mean_reversion(data, window)
            
        # Add more feature calculations...
        
        return pd.Series(np.zeros(len(data)), index=data.index)
    
    def _calculate_momentum(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate momentum features."""
        close = data['close']
        return close.pct_change(window)
    
    def _calculate_volatility(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate volatility features."""
        returns = data['close'].pct_change()
        return returns.rolling(window).std()
    
    def _calculate_mean_reversion(self, data: pd.DataFrame, window: int) -> pd.Series:
        """Calculate mean reversion features."""
        close = data['close']
        sma = close.rolling(window).mean()
        return (close - sma) / sma


# ============================================================================
# STEP 7: Per-Regime Matrix Operations Implementation
# ============================================================================

class RegimeAwareMatrixOperations:
    """Enhanced Step 7 with regime-specific matrix operations."""
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("RegimeAwareMatrixOperations")
        self.standards = pipeline_standards
        
        # Regime-specific matrix configurations
        self.regime_configs = {
            'bull': {
                'correlation_threshold': 0.75,  # Allow more correlation in trends
                'pca_variance_threshold': 0.95, # Preserve more information
                'regularization_alpha': 0.01,   # Light regularization
                'feature_scaling': 'robust'     # Robust to outliers
            },
            'bear': {
                'correlation_threshold': 0.65,  # Moderate correlation limit
                'pca_variance_threshold': 0.90, # More dimension reduction
                'regularization_alpha': 0.1,    # Strong regularization
                'feature_scaling': 'standard'   # Standard scaling
            },
            'sideways': {
                'correlation_threshold': 0.60,  # Strict correlation limit
                'pca_variance_threshold': 0.85, # Aggressive reduction
                'regularization_alpha': 0.05,   # Moderate regularization
                'feature_scaling': 'minmax'     # MinMax scaling
            }
        }
        
    async def execute(self, features_by_regime: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Execute per-regime matrix operations."""
        
        self.logger.info("Starting per-regime matrix operations")
        
        results = {
            'optimized_features': {},
            'transformation_info': {},
            'computational_stats': {}
        }
        
        for regime, features in features_by_regime.items():
            self.logger.info(f"Optimizing matrices for {regime} regime...")
            
            # Get regime configuration
            config = self.regime_configs[regime]
            
            # Remove correlated features
            decorrelated = await self._remove_correlations(features, config)
            
            # Apply PCA if needed
            reduced = await self._apply_pca(decorrelated, config)
            
            # Scale features
            scaled = await self._scale_features(reduced, config)
            
            # Apply regularization
            final_features = await self._apply_regularization(scaled, config)
            
            # Store results
            results['optimized_features'][regime] = final_features
            results['transformation_info'][regime] = {
                'original_features': features.shape[1],
                'after_decorrelation': decorrelated.shape[1],
                'after_pca': reduced.shape[1],
                'final_features': final_features.shape[1]
            }
            
            # Save optimized features
            output_path = self._get_matrix_output_path(regime)
            final_features.to_parquet(output_path)
            
        return results
    
    async def _remove_correlations(self, features: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Remove highly correlated features based on regime-specific threshold."""
        
        # Calculate correlation matrix
        corr_matrix = features.corr().abs()
        
        # Find features to remove
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        to_drop = [
            column for column in upper_tri.columns 
            if any(upper_tri[column] > config['correlation_threshold'])
        ]
        
        # Drop features
        return features.drop(columns=to_drop)
    
    async def _apply_pca(self, features: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply PCA with regime-specific variance threshold."""
        
        from sklearn.decomposition import PCA
        
        # Fit PCA
        pca = PCA(n_components=config['pca_variance_threshold'])
        transformed = pca.fit_transform(features)
        
        # Create DataFrame with PCA components
        pca_features = pd.DataFrame(
            transformed,
            index=features.index,
            columns=[f'pca_{i}' for i in range(transformed.shape[1])]
        )
        
        return pca_features


# ============================================================================
# Main Integration Function
# ============================================================================

async def run_regime_aware_pipeline(
    data: pd.DataFrame,
    regime_labels: np.ndarray,
    symbol: str,
    exchange: str,
    timeframe: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run the complete regime-aware pipeline for steps 5-7."""
    
    logger.info("Starting regime-aware pipeline execution")
    
    results = {
        'step5_labeling': None,
        'step6_features': None,
        'step7_matrices': None,
        'execution_time': {},
        'success': True
    }
    
    try:
        # Step 5: Per-regime labeling
        start_time = datetime.now()
        labeling_step = RegimeAwareLabelingStep(config)
        labeling_results = await labeling_step.execute(
            data, regime_labels, symbol, exchange, timeframe
        )
        results['step5_labeling'] = labeling_results
        results['execution_time']['step5'] = (datetime.now() - start_time).total_seconds()
        
        # Step 6: Per-regime feature engineering
        start_time = datetime.now()
        feature_step = RegimeAwareFeatureEngineering(config)
        feature_results = await feature_step.execute(
            labeling_results['labeled_data_by_regime']
        )
        results['step6_features'] = feature_results
        results['execution_time']['step6'] = (datetime.now() - start_time).total_seconds()
        
        # Step 7: Per-regime matrix operations
        start_time = datetime.now()
        matrix_step = RegimeAwareMatrixOperations(config)
        matrix_results = await matrix_step.execute(
            feature_results['features_by_regime']
        )
        results['step7_matrices'] = matrix_results
        results['execution_time']['step7'] = (datetime.now() - start_time).total_seconds()
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        results['success'] = False
        results['error'] = str(e)
        
    return results


if __name__ == "__main__":
    # Example usage
    async def main():
        # Load sample data
        data = pd.read_parquet("data/sample_data.parquet")
        regime_labels = np.load("data/regime_labels.npy")
        
        config = {
            'symbol': 'BTCUSDT',
            'exchange': 'binance',
            'timeframe': '1h'
        }
        
        results = await run_regime_aware_pipeline(
            data, regime_labels, 
            config['symbol'], config['exchange'], config['timeframe'],
            config
        )
        
        print(f"Pipeline completed: {results['success']}")
        
    asyncio.run(main())