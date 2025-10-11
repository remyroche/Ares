"""
Optimized Cross Timeframe Analysis Advanced Methods

This module contains the advanced feature selection and metrics calculation methods.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

from src.utils.logger import system_logger
from src.utils.math_validation import (

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

# Optional GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None
    validate_finite, validate_positive, validate_range,
    safe_divide, safe_log, safe_sqrt, safe_power,
    MathValidationError
)

logger = system_logger.getChild('OptimizedCrossTimeframeAdvanced')

class OptimizedCrossTimeframeAdvanced:
    """Advanced methods for cross timeframe analysis."""
    
    def __init__(self, parent_analyzer):
        """Initialize with reference to parent analyzer."""
        self.analyzer = parent_analyzer
        self.config = parent_analyzer.config
        self.logger = logger.getChild('OptimizedAdvanced')
        
        # Get optimizers from parent
        self.memory_optimizer = parent_analyzer.memory_optimizer
        self.cpu_optimizer = parent_analyzer.cpu_optimizer
        self.gpu_manager = parent_analyzer.gpu_manager
        self.feature_selector = parent_analyzer.feature_selector
        self.data_validator = parent_analyzer.data_validator
        self.data_cleaner = parent_analyzer.data_cleaner
        self.data_transformer = parent_analyzer.data_transformer
    
    async def _perform_advanced_feature_selection(
        self,
        features: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """Perform advanced feature selection using step08 utilities."""
        self.logger.info("🎯 Performing advanced feature selection")
        
        try:
            if not self.feature_selector:
                self.logger.warning("⚠️ Advanced feature selector not available, using basic selection")
                return self._perform_basic_feature_selection(features)
            
            # Prepare data for feature selection
            # Add regime information if available (placeholder for now)
            features_with_regime = features.copy()
            if 'composite_cluster_id' not in features_with_regime.columns:
                # Create synthetic regime for demonstration
                features_with_regime['composite_cluster_id'] = 0
            
            # Create training input structure
            training_input = {
                'data': features_with_regime,
                'target_column': 'base_returns',  # Use base returns as target
                'regime_column': 'composite_cluster_id'
            }
            
            # Create pipeline state
            pipeline_state = {
                'step_name': 'cross_timeframe_feature_selection',
                'config': self.config.__dict__,
                'memory_optimizer': self.memory_optimizer,
                'gpu_manager': self.gpu_manager
            }
            
            # Perform advanced feature selection
            try:
                selection_result = await self.feature_selector.execute(training_input, pipeline_state)
                
                if hasattr(selection_result, 'selected_features'):
                    selected_features = selection_result.selected_features
                else:
                    # Fallback to basic selection
                    selected_features = self._perform_basic_feature_selection(features)
                
                self.logger.info(f"✅ Advanced feature selection completed: {len(selected_features.get('final', []))} features selected")
                return selected_features
                
            except Exception as e:
                self.logger.warning(f"⚠️ Advanced feature selection failed: {e}, using basic selection")
                return self._perform_basic_feature_selection(features)
            
        except Exception as e:
            self.logger.error(f"❌ Feature selection failed: {e}")
            return {'final': list(features.columns)[:50]}  # Fallback to first 50 features
    
    def _perform_basic_feature_selection(self, features: pd.DataFrame) -> Dict[str, List[str]]:
        """Perform basic feature selection as fallback."""
        try:
            # Calculate feature importance using correlation with base returns
            if 'base_returns' in features.columns:
                base_returns = features['base_returns'].dropna()
                
                feature_importance = {}
                for col in features.columns:
                    if col != 'base_returns':
                        try:
                            # Align data
                            aligned_data = features[[col, 'base_returns']].dropna()
                            if len(aligned_data) > 10:
                                corr = aligned_data[col].corr(aligned_data['base_returns'])
                                feature_importance[col] = abs(corr) if not np.isnan(corr) else 0.0
                        except:
                            feature_importance[col] = 0.0
                
                # Sort by importance and select top features
                sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
                top_features = [feat[0] for feat in sorted_features[:50]]  # Top 50 features
                
                return {
                    'initial': list(features.columns),
                    'correlation_filtered': top_features,
                    'final': top_features
                }
            else:
                # No target column, return all features
                return {
                    'initial': list(features.columns),
                    'final': list(features.columns)
                }
                
        except Exception as e:
            self.logger.error(f"❌ Basic feature selection failed: {e}")
            return {'final': list(features.columns)[:50]}
    
    async def _calculate_interaction_metrics_optimized(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate interaction metrics with optimizations."""
        self.logger.info("📊 Calculating interaction metrics with optimizations")
        
        try:
            metrics = {}
            timeframes = list(aligned_data.keys())
            
            # Use parallel processing for correlation calculations
            if self.cpu_optimizer:
                executor = self.cpu_optimizer.create_optimized_thread_pool(max_workers=self.config.max_workers)
            else:
                executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            try:
                # Create tasks for pairwise correlations
                correlation_tasks = []
                for i, tf1 in enumerate(timeframes):
                    for j, tf2 in enumerate(timeframes[i+1:], i+1):
                        task = self._calculate_pairwise_correlation(aligned_data[tf1], aligned_data[tf2], tf1, tf2)
                        correlation_tasks.append(task)
                
                # Execute correlation tasks in parallel
                correlation_results = await asyncio.gather(*correlation_tasks, return_exceptions=True)
                
                # Process correlation results
                correlations = {}
                for result in correlation_results:
                    if isinstance(result, Exception):
                        self.logger.warning(f"⚠️ Correlation calculation failed: {result}")
                        continue
                    
                    if isinstance(result, dict):
                        correlations.update(result)
                
                # Calculate interaction strength
                strong_interactions = []
                for pair, corrs in correlations.items():
                    if abs(corrs.get('avg_correlation', 0)) > self.config.correlation_threshold:
                        strong_interactions.append(pair)
                
                metrics = {
                    'pairwise_correlations': correlations,
                    'strong_interactions': strong_interactions,
                    'interaction_strength': len(strong_interactions) / len(correlations) if correlations else 0,
                    'total_interactions': len(correlations)
                }
                
            finally:
                executor.shutdown(wait=False)
            
            self.logger.info("✅ Interaction metrics calculated with optimizations")
            return metrics
            
        except Exception as e:
            self.logger.error(f"❌ Interaction metrics calculation failed: {e}")
            return {}
    
    async def _calculate_pairwise_correlation(
        self,
        data1: pd.DataFrame,
        data2: pd.DataFrame,
        tf1: str,
        tf2: str
    ) -> Dict[str, Any]:
        """Calculate pairwise correlation between two timeframes."""
        try:
            # Price correlation
            price_corr = data1['close'].corr(data2['close'])
            
            # Volume correlation
            volume_corr = data1['volume'].corr(data2['volume'])
            
            # Returns correlation
            returns1 = data1['close'].pct_change()
            returns2 = data2['close'].pct_change()
            returns_corr = returns1.corr(returns2)
            
            # Calculate average correlation
            correlations = [price_corr, volume_corr, returns_corr]
            valid_correlations = [c for c in correlations if not np.isnan(c)]
            avg_correlation = np.mean(valid_correlations) if valid_correlations else 0.0
            
            return {
                f'{tf1}_{tf2}': {
                    'price_correlation': price_corr if not np.isnan(price_corr) else 0.0,
                    'volume_correlation': volume_corr if not np.isnan(volume_corr) else 0.0,
                    'returns_correlation': returns_corr if not np.isnan(returns_corr) else 0.0,
                    'avg_correlation': avg_correlation
                }
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pairwise correlation calculation failed for {tf1}_{tf2}: {e}")
            return {}
    
    async def _calculate_timeframe_correlations_optimized(
        self,
        aligned_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Calculate detailed timeframe correlations with optimizations."""
        self.logger.info("📊 Calculating timeframe correlations with optimizations")
        
        try:
            correlations = {}
            timeframes = list(aligned_data.keys())
            
            # Create correlation matrix for each metric
            metrics = ['close', 'volume', 'returns', 'volatility']
            
            # Use parallel processing for correlation matrix calculations
            if self.cpu_optimizer:
                executor = self.cpu_optimizer.create_optimized_thread_pool(max_workers=self.config.max_workers)
            else:
                executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            try:
                # Create tasks for each metric
                correlation_tasks = []
                for metric in metrics:
                    task = self._calculate_metric_correlation_matrix(aligned_data, timeframes, metric)
                    correlation_tasks.append(task)
                
                # Execute correlation tasks in parallel
                correlation_results = await asyncio.gather(*correlation_tasks, return_exceptions=True)
                
                # Process correlation results
                for i, result in enumerate(correlation_results):
                    if isinstance(result, Exception):
                        self.logger.warning(f"⚠️ Correlation matrix calculation failed for {metrics[i]}: {result}")
                        continue
                    
                    if isinstance(result, dict):
                        correlations.update(result)
                
                # Calculate average correlation
                if correlations:
                    avg_corr = np.mean([corr.values for corr in correlations.values() if isinstance(corr, pd.DataFrame)], axis=0)
                    correlations['average'] = pd.DataFrame(avg_corr, index=timeframes, columns=timeframes)
                
            finally:
                executor.shutdown(wait=False)
            
            self.logger.info("✅ Timeframe correlations calculated with optimizations")
            return correlations
            
        except Exception as e:
            self.logger.error(f"❌ Timeframe correlations calculation failed: {e}")
            return {}
    
    async def _calculate_metric_correlation_matrix(
        self,
        aligned_data: Dict[str, pd.DataFrame],
        timeframes: List[str],
        metric: str
    ) -> Dict[str, pd.DataFrame]:
        """Calculate correlation matrix for a specific metric."""
        try:
            corr_matrix = pd.DataFrame(index=timeframes, columns=timeframes)
            
            for tf1 in timeframes:
                for tf2 in timeframes:
                    data1 = aligned_data[tf1]
                    data2 = aligned_data[tf2]
                    
                    if metric == 'close':
                        corr_value = data1['close'].corr(data2['close'])
                    elif metric == 'volume':
                        corr_value = data1['volume'].corr(data2['volume'])
                    elif metric == 'returns':
                        returns1 = data1['close'].pct_change()
                        returns2 = data2['close'].pct_change()
                        corr_value = returns1.corr(returns2)
                    elif metric == 'volatility':
                        vol1 = data1['close'].pct_change().rolling(20).std()
                        vol2 = data2['close'].pct_change().rolling(20).std()
                        corr_value = vol1.corr(vol2)
                    
                    corr_matrix.loc[tf1, tf2] = corr_value if not np.isnan(corr_value) else 1.0 if tf1 == tf2 else 0.0
            
            return {metric: corr_matrix}
            
        except Exception as e:
            self.logger.error(f"❌ Metric correlation matrix calculation failed for {metric}: {e}")
            return {}
    
    async def _calculate_feature_importance_optimized(
        self,
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate feature importance with optimizations."""
        self.logger.info("📊 Calculating feature importance with optimizations")
        
        try:
            # Calculate correlation with base returns
            if 'base_returns' in features.columns:
                base_returns = features['base_returns'].dropna()
                
                # Use parallel processing for feature importance calculation
                if self.cpu_optimizer:
                    executor = self.cpu_optimizer.create_optimized_thread_pool(max_workers=self.config.max_workers)
                else:
                    executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
                
                try:
                    # Create tasks for feature importance calculation
                    importance_tasks = []
                    feature_columns = [col for col in features.columns if col != 'base_returns']
                    
                    # Process features in chunks for better performance
                    chunk_size = max(1, len(feature_columns) // self.config.max_workers)
                    for i in range(0, len(feature_columns), chunk_size):
                        chunk = feature_columns[i:i+chunk_size]
                        task = self._calculate_feature_importance_chunk(features, base_returns, chunk)
                        importance_tasks.append(task)
                    
                    # Execute importance tasks in parallel
                    importance_results = await asyncio.gather(*importance_tasks, return_exceptions=True)
                    
                    # Combine results
                    feature_importance = {}
                    for result in importance_results:
                        if isinstance(result, Exception):
                            self.logger.warning(f"⚠️ Feature importance calculation failed: {result}")
                            continue
                        
                        if isinstance(result, dict):
                            feature_importance.update(result)
                    
                    # Sort by importance
                    sorted_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
                    
                    # Get top features
                    top_features = dict(list(sorted_importance.items())[:20])
                    
                    importance_metrics = {
                        'feature_importance': sorted_importance,
                        'top_features': top_features,
                        'avg_importance': np.mean(list(sorted_importance.values())) if sorted_importance else 0.0,
                        'max_importance': max(sorted_importance.values()) if sorted_importance else 0.0
                    }
                    
                finally:
                    executor.shutdown(wait=False)
                
                self.logger.info("✅ Feature importance calculated with optimizations")
                return importance_metrics
            
            else:
                self.logger.warning("⚠️ Base returns not found, skipping feature importance calculation")
                return {}
            
        except Exception as e:
            self.logger.error(f"❌ Feature importance calculation failed: {e}")
            return {}
    
    async def _calculate_feature_importance_chunk(
        self,
        features: pd.DataFrame,
        base_returns: pd.Series,
        feature_chunk: List[str]
    ) -> Dict[str, float]:
        """Calculate feature importance for a chunk of features."""
        try:
            feature_importance = {}
            
            for col in feature_chunk:
                try:
                    # Align data
                    aligned_data = features[[col, 'base_returns']].dropna()
                    if len(aligned_data) > 10:
                        corr = aligned_data[col].corr(aligned_data['base_returns'])
                        feature_importance[col] = abs(corr) if not np.isnan(corr) else 0.0
                    else:
                        feature_importance[col] = 0.0
                except:
                    feature_importance[col] = 0.0
            
            return feature_importance
            
        except Exception as e:
            self.logger.error(f"❌ Feature importance chunk calculation failed: {e}")
            return {}
    
    async def _calculate_financial_risk_metrics(
        self,
        features: pd.DataFrame
    ) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Calculate financial and risk metrics."""
        self.logger.info("📊 Calculating financial and risk metrics")
        
        try:
            financial_metrics = None
            risk_metrics = None
            
            if 'base_returns' in features.columns:
                returns = features['base_returns'].dropna()
                
                if len(returns) > 0:
                    # Calculate financial metrics
                    financial_metrics = {
                        'returns': {
                            'daily': returns.mean(),
                            'annualized': returns.mean() * 252,
                            'volatility': returns.std(),
                            'volatility_annualized': returns.std() * np.sqrt(252)
                        },
                        'sharpe_ratio': {
                            'daily': returns.mean() / returns.std() if returns.std() > 0 else 0,
                            'annualized': (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
                        },
                        'max_drawdown': self._calculate_max_drawdown(returns),
                        'var_95': np.percentile(returns, 5),
                        'var_99': np.percentile(returns, 1)
                    }
                    
                    # Calculate risk metrics
                    risk_metrics = {
                        'portfolio_var': np.percentile(returns, 5),
                        'portfolio_es': returns[returns <= np.percentile(returns, 5)].mean(),
                        'concentration_risk': self._calculate_concentration_risk(features),
                        'model_risk': self._calculate_model_risk(features),
                        'overfitting_risk': self._calculate_overfitting_risk(features),
                        'overall_risk_score': 0.0  # Placeholder
                    }
                    
                    # Calculate overall risk score
                    risk_metrics['overall_risk_score'] = self._calculate_overall_risk_score(risk_metrics)
            
            self.logger.info("✅ Financial and risk metrics calculated")
            return financial_metrics, risk_metrics
            
        except Exception as e:
            self.logger.error(f"❌ Financial and risk metrics calculation failed: {e}")
            return None, None
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown."""
        try:
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()
        except:
            return 0.0
    
    def _calculate_concentration_risk(self, features: pd.DataFrame) -> float:
        """Calculate concentration risk."""
        try:
            # Calculate feature variance
            variances = features.var()
            total_variance = variances.sum()
            
            # Calculate concentration (Herfindahl index)
            concentration = (variances / total_variance).pow(2).sum()
            return concentration
        except:
            return 0.0
    
    def _calculate_model_risk(self, features: pd.DataFrame) -> float:
        """Calculate model risk."""
        try:
            # Calculate feature correlation
            corr_matrix = features.corr().abs()
            avg_correlation = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            
            # Higher correlation indicates higher model risk
            return avg_correlation
        except:
            return 0.0
    
    def _calculate_overfitting_risk(self, features: pd.DataFrame) -> float:
        """Calculate overfitting risk."""
        try:
            # Calculate feature count vs sample count ratio
            feature_count = len(features.columns)
            sample_count = len(features)
            
            # Higher ratio indicates higher overfitting risk
            ratio = feature_count / sample_count
            return min(ratio, 1.0)  # Cap at 1.0
        except:
            return 0.0
    
    def _calculate_overall_risk_score(self, risk_metrics: Dict[str, Any]) -> float:
        """Calculate overall risk score."""
        try:
            # Weighted average of risk metrics
            weights = {
                'portfolio_var': 0.3,
                'concentration_risk': 0.2,
                'model_risk': 0.2,
                'overfitting_risk': 0.3
            }
            
            overall_score = 0.0
            for metric, weight in weights.items():
                if metric in risk_metrics:
                    overall_score += risk_metrics[metric] * weight
            
            return overall_score
        except:
            return 0.0
    
    async def _generate_quality_report(
        self,
        timeframe_data: Dict[str, pd.DataFrame],
        features: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        self.logger.info("📋 Generating quality report")
        
        try:
            quality_report = {
                'data_quality': {},
                'feature_quality': {},
                'overall_quality_score': 0.0
            }
            
            # Data quality assessment
            for timeframe, data in timeframe_data.items():
                quality_report['data_quality'][timeframe] = {
                    'row_count': len(data),
                    'column_count': len(data.columns),
                    'missing_values': data.isnull().sum().sum(),
                    'duplicate_rows': data.duplicated().sum(),
                    'data_types': data.dtypes.to_dict()
                }
            
            # Feature quality assessment
            quality_report['feature_quality'] = {
                'feature_count': len(features.columns),
                'row_count': len(features),
                'missing_values': features.isnull().sum().sum(),
                'infinite_values': np.isinf(features.select_dtypes(include=[np.number])).sum().sum(),
                'constant_features': (features.nunique() == 1).sum(),
                'high_correlation_pairs': self._count_high_correlation_pairs(features)
            }
            
            # Calculate overall quality score
            quality_report['overall_quality_score'] = self._calculate_quality_score(quality_report)
            
            self.logger.info("✅ Quality report generated")
            return quality_report
            
        except Exception as e:
            self.logger.error(f"❌ Quality report generation failed: {e}")
            return {'overall_quality_score': 0.0}
    
    def _count_high_correlation_pairs(self, features: pd.DataFrame) -> int:
        """Count high correlation feature pairs."""
        try:
            numeric_features = features.select_dtypes(include=[np.number])
            if len(numeric_features.columns) < 2:
                return 0
            
            corr_matrix = numeric_features.corr().abs()
            upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_pairs = (upper_triangle > 0.95).sum().sum()
            return high_corr_pairs
        except:
            return 0
    
    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """Calculate overall quality score."""
        try:
            score = 1.0
            
            # Penalize missing values
            total_missing = sum(
                data_quality.get('missing_values', 0)
                for data_quality in quality_report.get('data_quality', {}).values()
            )
            if total_missing > 0:
                score -= min(0.3, total_missing / 10000)  # Penalty for missing values
            
            # Penalize duplicate rows
            total_duplicates = sum(
                data_quality.get('duplicate_rows', 0)
                for data_quality in quality_report.get('data_quality', {}).values()
            )
            if total_duplicates > 0:
                score -= min(0.2, total_duplicates / 1000)  # Penalty for duplicates
            
            # Penalize constant features
            constant_features = quality_report.get('feature_quality', {}).get('constant_features', 0)
            if constant_features > 0:
                score -= min(0.2, constant_features / 10)  # Penalty for constant features
            
            # Penalize high correlation pairs
            high_corr_pairs = quality_report.get('feature_quality', {}).get('high_correlation_pairs', 0)
            if high_corr_pairs > 0:
                score -= min(0.3, high_corr_pairs / 100)  # Penalty for high correlations
            
            return max(0.0, score)  # Ensure non-negative score
        except:
            return 0.5  # Default score
    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and self.use_vectorbt and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
