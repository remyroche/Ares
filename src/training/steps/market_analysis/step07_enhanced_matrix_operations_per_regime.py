"""Step 7: Enhanced Matrix Operations - Per-Regime Implementation.

This module provides per-HMM regime matrix operations functionality, ensuring that
matrix operations are performed specifically for each regime's characteristics.
"""

import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.training.steps.step07_enhanced_matrix_operations import Step7EnhancedMatrixOperations
from src.training.steps.regime_handler import regime_handler
from src.training.steps.regime_processing_decorator import (
    per_regime_processing,
    aggregate_regime_results,
    RegimeProcessingContext
)
from src.training.steps.regime_continuity_decorator import per_regime_step
from src.utils.logger import getChild as get_logger
from src.utils.pipeline_standards import pipeline_standards
from src.core.decorators import traced, validates, handles_errors


logger = get_logger('Step7EnhancedMatrixOperationsPerRegime')


class PerRegimeEnhancedMatrixOperationsStep(Step7EnhancedMatrixOperations):
    """Enhanced matrix operations step that processes each regime separately."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.per_regime_enabled = config.get('per_regime_matrix_operations', True)
        self.regime_specific_configs = config.get('regime_specific_matrix_configs', {})
        self.adaptive_matrix_operations = config.get('adaptive_matrix_operations_per_regime', True)
        
    @traced(span_name='execute_per_regime_matrix_operations')
    @per_regime_step('step07_enhanced_matrix_operations')
    async def execute_per_regime_matrix_operations(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        force_rerun: bool = False,
        regime_id: Optional[int] = None,
        regime_context: Optional[Any] = None,
        per_regime: bool = True
    ) -> bool:
        """Execute matrix operations on a per-regime basis.
        
        Each regime may have different data characteristics, so matrix operations
        should be optimized specifically for each regime's data patterns.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            force_rerun: Force rerun flag
            regime_id: Regime ID (provided by decorator)
            regime_context: Regime context (provided by decorator)
            per_regime: Per-regime flag (provided by decorator)
            
        Returns:
            Success status
        """
        try:
            self.logger.info(f"🚀 Starting per-regime matrix operations for regime {regime_id}")
            
            # Load feature engineered data from previous step
            feature_data = await self._load_feature_data(symbol, exchange, timeframe, data_dir, regime_id)
            if feature_data is None:
                self.logger.error(f"❌ Failed to load feature data for regime {regime_id}")
                return False
            
            # Get regime-specific configuration
            regime_config = self._get_regime_matrix_config(regime_id)
            
            # Apply regime-specific matrix operations
            matrix_results = await self._apply_regime_matrix_operations(
                feature_data, regime_config, regime_id
            )
            
            if matrix_results is None:
                self.logger.error(f"❌ Failed matrix operations for regime {regime_id}")
                return False
            
            # Save regime-specific results
            success = await self._save_regime_matrix_results(
                matrix_results, symbol, exchange, timeframe, data_dir, regime_id
            )
            
            if success:
                self.logger.info(f"✅ Successfully completed matrix operations for regime {regime_id}")
            else:
                self.logger.error(f"❌ Failed to save matrix results for regime {regime_id}")
            
            return success
            
        except Exception as e:
            self.logger.exception(f"❌ Error in per-regime matrix operations for regime {regime_id}: {e}")
            return False
    
    async def _load_feature_data(
        self,
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> Optional[pd.DataFrame]:
        """Load feature engineered data for a specific regime.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            Feature DataFrame or None
        """
        try:
            # Try per-regime feature data first
            feature_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_features_per_regime_regime_{regime_id}.parquet'
            
            if not feature_path.exists():
                # Fall back to aggregated feature data and filter by regime
                feature_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_features_per_regime.parquet'
                
                if feature_path.exists():
                    data = pd.read_parquet(feature_path)
                    # Filter by regime
                    if 'feature_regime_id' in data.columns:
                        data = data[data['feature_regime_id'] == regime_id]
                        self.logger.info(f"✅ Loaded and filtered feature data for regime {regime_id}: {len(data)} rows")
                        return data
                
                # Final fallback to standard feature data
                feature_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_features.parquet'
            
            if feature_path.exists():
                data = pd.read_parquet(feature_path)
                self.logger.info(f"✅ Loaded feature data for regime {regime_id}: {len(data)} rows")
                return data
            else:
                self.logger.error(f"❌ Feature data not found: {feature_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error loading feature data for regime {regime_id}: {e}")
            return None
    
    def _get_regime_matrix_config(self, regime_id: int) -> Dict[str, Any]:
        """Get matrix operations configuration for a specific regime.
        
        Different regimes may benefit from different matrix operation parameters.
        
        Args:
            regime_id: Regime ID
            
        Returns:
            Dictionary of regime-specific matrix configuration
        """
        # Check if custom config exists for this regime
        if f'regime_{regime_id}' in self.regime_specific_configs:
            return self.regime_specific_configs[f'regime_{regime_id}']
        
        # Create adaptive configuration based on regime characteristics
        base_config = {
            'enable_correlation_analysis': True,
            'enable_covariance_analysis': True,
            'enable_principal_component_analysis': True,
            'enable_factor_analysis': True,
            'enable_clustering_analysis': True
        }
        
        # Adapt based on regime ID patterns
        if regime_id <= 2:
            # Low regime IDs - often trending markets
            # Emphasize trend analysis and momentum matrices
            return {
                **base_config,
                'matrix_operations': {
                    'correlation_threshold': 0.7,
                    'pca_components': 0.95,
                    'clustering_methods': ['kmeans', 'hierarchical'],
                    'emphasis': 'trend_analysis'
                },
                'additional_operations': [
                    'trend_correlation_matrix',
                    'momentum_covariance_matrix',
                    'trend_pca_analysis'
                ]
            }
        elif regime_id >= 5:
            # High regime IDs - often volatile/ranging markets
            # Emphasize volatility analysis and mean reversion matrices
            return {
                **base_config,
                'matrix_operations': {
                    'correlation_threshold': 0.5,
                    'pca_components': 0.90,
                    'clustering_methods': ['dbscan', 'spectral'],
                    'emphasis': 'volatility_analysis'
                },
                'additional_operations': [
                    'volatility_correlation_matrix',
                    'mean_reversion_covariance_matrix',
                    'volatility_pca_analysis'
                ]
            }
        else:
            # Medium regime IDs - balanced approach
            return {
                **base_config,
                'matrix_operations': {
                    'correlation_threshold': 0.6,
                    'pca_components': 0.92,
                    'clustering_methods': ['kmeans', 'gaussian_mixture'],
                    'emphasis': 'balanced_analysis'
                },
                'additional_operations': [
                    'balanced_correlation_matrix',
                    'mixed_covariance_matrix',
                    'balanced_pca_analysis'
                ]
            }
    
    async def _apply_regime_matrix_operations(
        self,
        feature_data: pd.DataFrame,
        regime_config: Dict[str, Any],
        regime_id: int
    ) -> Optional[Dict[str, Any]]:
        """Apply matrix operations to regime data.
        
        Args:
            feature_data: Feature DataFrame
            regime_config: Regime configuration
            regime_id: Regime ID
            
        Returns:
            Matrix operation results or None
        """
        try:
            self.logger.info(f"🔧 Applying matrix operations for regime {regime_id}")
            
            # Get feature columns (exclude metadata columns)
            feature_columns = [col for col in feature_data.columns 
                             if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                          'composite_cluster_id', 'feature_regime_id', 'label',
                                          'regime_emphasis', 'optimization_priority']]
            
            if not feature_columns:
                self.logger.warning(f"⚠️ No feature columns found for regime {regime_id}")
                return None
            
            # Extract feature matrix
            feature_matrix = feature_data[feature_columns].values
            
            # Remove any rows with NaN values
            valid_mask = ~np.isnan(feature_matrix).any(axis=1)
            feature_matrix = feature_matrix[valid_mask]
            
            if len(feature_matrix) < 10:
                self.logger.warning(f"⚠️ Insufficient valid data for regime {regime_id}: {len(feature_matrix)} rows")
                return None
            
            results = {
                'regime_id': regime_id,
                'matrix_shape': feature_matrix.shape,
                'feature_columns': feature_columns,
                'valid_rows': len(feature_matrix),
                'operations': {}
            }
            
            # Apply correlation analysis
            if regime_config.get('enable_correlation_analysis', True):
                correlation_matrix = np.corrcoef(feature_matrix.T)
                results['operations']['correlation_matrix'] = {
                    'matrix': correlation_matrix,
                    'high_correlations': self._find_high_correlations(
                        correlation_matrix, feature_columns, 
                        regime_config.get('matrix_operations', {}).get('correlation_threshold', 0.6)
                    )
                }
            
            # Apply PCA analysis
            if regime_config.get('enable_principal_component_analysis', True):
                pca_results = self._apply_pca_analysis(
                    feature_matrix, regime_config.get('matrix_operations', {}).get('pca_components', 0.95)
                )
                results['operations']['pca_analysis'] = pca_results
            
            # Apply clustering analysis
            if regime_config.get('enable_clustering_analysis', True):
                clustering_results = self._apply_clustering_analysis(
                    feature_matrix, regime_config.get('matrix_operations', {}).get('clustering_methods', ['kmeans'])
                )
                results['operations']['clustering_analysis'] = clustering_results
            
            # Apply regime-specific additional operations
            additional_ops = regime_config.get('additional_operations', [])
            for op_name in additional_ops:
                op_result = self._apply_additional_operation(op_name, feature_matrix, feature_columns)
                if op_result:
                    results['operations'][op_name] = op_result
            
            self.logger.info(f"✅ Completed matrix operations for regime {regime_id}: {len(results['operations'])} operations")
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Error applying matrix operations for regime {regime_id}: {e}")
            return None
    
    def _find_high_correlations(
        self,
        correlation_matrix: np.ndarray,
        feature_names: List[str],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Find high correlations in the correlation matrix.
        
        Args:
            correlation_matrix: Correlation matrix
            feature_names: List of feature names
            threshold: Correlation threshold
            
        Returns:
            List of high correlation pairs
        """
        high_correlations = []
        
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                corr_value = correlation_matrix[i, j]
                if abs(corr_value) >= threshold:
                    high_correlations.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': float(corr_value),
                        'abs_correlation': float(abs(corr_value))
                    })
        
        # Sort by absolute correlation
        high_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)
        return high_correlations
    
    def _apply_pca_analysis(
        self,
        feature_matrix: np.ndarray,
        variance_threshold: float
    ) -> Dict[str, Any]:
        """Apply PCA analysis to the feature matrix.
        
        Args:
            feature_matrix: Feature matrix
            variance_threshold: Variance threshold for component selection
            
        Returns:
            PCA analysis results
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_matrix = scaler.fit_transform(feature_matrix)
            
            # Apply PCA
            pca = PCA()
            pca_result = pca.fit_transform(scaled_matrix)
            
            # Find number of components for variance threshold
            cumsum_variance = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum_variance >= variance_threshold) + 1
            
            return {
                'explained_variance_ratio': pca.explained_variance_ratio_[:n_components].tolist(),
                'cumulative_variance': cumsum_variance[:n_components].tolist(),
                'n_components': int(n_components),
                'total_variance_explained': float(cumsum_variance[n_components - 1]),
                'components': pca_result[:, :n_components].tolist()
            }
            
        except ImportError:
            self.logger.warning("⚠️ sklearn not available for PCA analysis")
            return {'error': 'sklearn not available'}
        except Exception as e:
            self.logger.error(f"❌ Error in PCA analysis: {e}")
            return {'error': str(e)}
    
    def _apply_clustering_analysis(
        self,
        feature_matrix: np.ndarray,
        methods: List[str]
    ) -> Dict[str, Any]:
        """Apply clustering analysis to the feature matrix.
        
        Args:
            feature_matrix: Feature matrix
            methods: List of clustering methods to apply
            
        Returns:
            Clustering analysis results
        """
        try:
            from sklearn.cluster import KMeans, DBSCAN
            from sklearn.mixture import GaussianMixture
            from sklearn.preprocessing import StandardScaler
from src.core.decorators.errors import handles_errors
            
            results = {}
            
            # Standardize the data
            scaler = StandardScaler()
            scaled_matrix = scaler.fit_transform(feature_matrix)
            
            for method in methods:
                try:
                    if method == 'kmeans':
                        # K-means clustering
                        kmeans = KMeans(n_clusters=min(5, len(feature_matrix) // 10), random_state=42)
                        clusters = kmeans.fit_predict(scaled_matrix)
                        results['kmeans'] = {
                            'clusters': clusters.tolist(),
                            'n_clusters': len(np.unique(clusters)),
                            'inertia': float(kmeans.inertia_)
                        }
                    
                    elif method == 'dbscan':
                        # DBSCAN clustering
                        dbscan = DBSCAN(eps=0.5, min_samples=5)
                        clusters = dbscan.fit_predict(scaled_matrix)
                        results['dbscan'] = {
                            'clusters': clusters.tolist(),
                            'n_clusters': len(np.unique(clusters)),
                            'n_noise': int(np.sum(clusters == -1))
                        }
                    
                    elif method == 'gaussian_mixture':
                        # Gaussian Mixture clustering
                        gmm = GaussianMixture(n_components=min(3, len(feature_matrix) // 20), random_state=42)
                        clusters = gmm.fit_predict(scaled_matrix)
                        results['gaussian_mixture'] = {
                            'clusters': clusters.tolist(),
                            'n_components': gmm.n_components,
                            'converged': gmm.converged_
                        }
                
                except Exception as e:
                    self.logger.warning(f"⚠️ Error applying {method} clustering: {e}")
                    results[method] = {'error': str(e)}
            
            return results
            
        except ImportError:
            self.logger.warning("⚠️ sklearn not available for clustering analysis")
            return {'error': 'sklearn not available'}
        except Exception as e:
            self.logger.error(f"❌ Error in clustering analysis: {e}")
            return {'error': str(e)}
    
    def _apply_additional_operation(
        self,
        operation_name: str,
        feature_matrix: np.ndarray,
        feature_names: List[str]
    ) -> Optional[Dict[str, Any]]:
        """Apply additional regime-specific matrix operations.
        
        Args:
            operation_name: Name of the operation
            feature_matrix: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Operation results or None
        """
        try:
            if 'trend' in operation_name.lower():
                # Trend-specific analysis
                return self._apply_trend_analysis(feature_matrix, feature_names)
            elif 'volatility' in operation_name.lower():
                # Volatility-specific analysis
                return self._apply_volatility_analysis(feature_matrix, feature_names)
            elif 'balanced' in operation_name.lower():
                # Balanced analysis
                return self._apply_balanced_analysis(feature_matrix, feature_names)
            else:
                self.logger.warning(f"⚠️ Unknown additional operation: {operation_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Error in additional operation {operation_name}: {e}")
            return None
    
    def _apply_trend_analysis(self, feature_matrix: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Apply trend-specific matrix analysis."""
        # Calculate trend strength matrix
        trend_strength = np.abs(np.diff(feature_matrix, axis=0)).mean(axis=0)
        
        return {
            'trend_strength': trend_strength.tolist(),
            'trend_features': [name for name, strength in zip(feature_names, trend_strength) if strength > np.median(trend_strength)]
        }
    
    def _apply_volatility_analysis(self, feature_matrix: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Apply volatility-specific matrix analysis."""
        # Calculate volatility matrix
        volatility = np.std(feature_matrix, axis=0)
        
        return {
            'volatility': volatility.tolist(),
            'high_volatility_features': [name for name, vol in zip(feature_names, volatility) if vol > np.median(volatility)]
        }
    
    def _apply_balanced_analysis(self, feature_matrix: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
        """Apply balanced matrix analysis."""
        # Calculate feature importance based on variance
        feature_importance = np.var(feature_matrix, axis=0)
        
        return {
            'feature_importance': feature_importance.tolist(),
            'important_features': [name for name, importance in zip(feature_names, feature_importance) if importance > np.median(feature_importance)]
        }
    
    async def _save_regime_matrix_results(
        self,
        matrix_results: Dict[str, Any],
        symbol: str,
        exchange: str,
        timeframe: str,
        data_dir: str,
        regime_id: int
    ) -> bool:
        """Save matrix operation results for a specific regime.
        
        Args:
            matrix_results: Matrix operation results
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Timeframe
            data_dir: Data directory
            regime_id: Regime ID
            
        Returns:
            True if successful
        """
        try:
            # Save regime-specific results
            regime_results_path = Path(data_dir) / 'training' / f'{exchange}_{symbol}_{timeframe}_matrix_operations_regime_{regime_id}.json'
            
            with open(regime_results_path, 'w') as f:
                json.dump(matrix_results, f, indent=2, default=str)
            
            self.logger.info(f"✅ Saved matrix results for regime {regime_id}: {regime_results_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Error saving matrix results for regime {regime_id}: {e}")
            return False


@traced(span_name='run_per_regime_matrix_operations_step')
@validates()
@handles_errors
async def run_per_regime_step(
    symbol: str,
    exchange: str,
    timeframe: str,
    data_dir: str = None,
    force_rerun: bool = False,
    config: Optional[Dict[str, Any]] = None
) -> bool:
    """Run the enhanced per-regime matrix operations step.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe for data
        data_dir: Data directory
        force_rerun: Force rerun the step
        config: Configuration dictionary
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("🚀 Starting Step 7: Per-Regime Enhanced Matrix Operations")
    
    if config is None:
        config = {}
        
    if data_dir is None:
        data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
    
    # Enable per-regime processing
    config['per_regime_matrix_operations'] = True
    
    # Initialize and run the per-regime matrix operations step
    step = PerRegimeEnhancedMatrixOperationsStep(config)
    
    success = await step.execute_per_regime_matrix_operations(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        data_dir=data_dir,
        force_rerun=force_rerun
    )
    
    if success:
        logger.info("✅ Step 7: Per-Regime Enhanced Matrix Operations completed successfully")
    else:
        logger.error("❌ Step 7: Per-Regime Enhanced Matrix Operations failed")
        
    return success


if __name__ == '__main__':
    async def test():
        """Test the per-regime matrix operations step."""
        success = await run_per_regime_step(
            symbol='ETHUSDT',
            exchange='BINANCE',
            timeframe='1m',
            data_dir='data_cache'
        )
        print(f'Per-regime matrix operations result: {success}')
        
    asyncio.run(test())