"""
Feature Generation Feature Selection Step.

This step performs optimized feature selection from generated features using
VectorBTRollingOptimizer and UnifiedVectorizationManager for maximum performance.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import time
from pathlib import Path

from src.training.steps.base_step import BaseStep
from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_performance, tprint_timer, tprint_success
from src.feature_generation.utils.vectorbt_rolling_optimizer import VectorBTRollingOptimizer
from src.feature_generation.utils.unified_vectorization_manager import (
    UnifiedVectorizationManager, 
    OperationType, 
    OptimizationStrategy,
    VectorizationConfig
)

logger = logging.getLogger(__name__)


class FeatureGenerationFeatureSelectionStep(BaseStep):
    """
    Optimized Feature Generation Feature Selection Step.

    Performs high-performance feature selection using VectorBTRollingOptimizer
    and UnifiedVectorizationManager for maximum efficiency and scalability.
    """

    def __init__(self, step_name: str = "feature_generation_feature_selection_step"):
        """Initialize the optimized feature selection step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('OptimizedFeatureSelection')
        
        # Initialize optimization components
        self.vectorbt_optimizer = VectorBTRollingOptimizer(
            enable_parallel=True,
            memory_efficient=True,
            chunk_size=1000,
            fast_fail=True,
            enable_logging=True
        )
        
        # Create VectorizationConfig with appropriate settings
        vectorization_config = VectorizationConfig(
            enable_gpu=False,  # GPU support can be enabled if available
            enable_parallel=True,
            memory_efficient=True,
            enable_monitoring=True
        )
        
        self.vectorization_manager = UnifiedVectorizationManager(
            config=vectorization_config,
            fast_fail=False,
            enable_logging=True
        )
        
        # Performance tracking
        self.performance_stats = {
            'total_execution_time': 0.0,
            'feature_loading_time': 0.0,
            'selection_time': 0.0,
            'optimization_time': 0.0,
            'vectorbt_operations': 0,
            'memory_optimizations': 0,
            'parallel_operations': 0
        }

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute optimized feature selection with performance monitoring.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '15m')
                - execution_mode: 'full', 'light', or 'blank'
                - feature_selection_config: Optional configuration for selection methods

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': dict of created artifacts
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        start_time = time.time()
        tprint(f"🚀 Starting optimized feature selection for {config.get('symbol', 'UNKNOWN')}", "INFO")
        
        try:
            # Load and validate features
            features_df = await self._load_features(config)
            if features_df is None or features_df.empty:
                raise ValueError("No features available for selection")
            
            # Perform optimized feature selection
            selected_features, selection_metrics = await self._perform_optimized_selection(
                features_df, config
            )
            
            # Calculate performance metrics
            total_time = time.time() - start_time
            self.performance_stats['total_execution_time'] = total_time
            
            # Create artifacts
            artifacts = {
                'selected_features': {
                    'selected_features': selected_features,
                    'selection_method': selection_metrics['method'],
                    'n_features_selected': len(selected_features),
                    'n_features_original': len(features_df.columns),
                    'selection_ratio': len(selected_features) / len(features_df.columns),
                    'performance_stats': self.performance_stats,
                    'metadata': {
                        'symbol': config['symbol'],
                        'exchange': config['exchange'],
                        'timeframe': config['timeframe'],
                        'execution_mode': config.get('execution_mode', 'light'),
                        'created_at': datetime.now().isoformat(),
                        'optimization_used': True
                    }
                }
            }

            # Enhanced metrics with performance data
            metrics = {
                'n_features_selected': len(selected_features),
                'n_features_original': len(features_df.columns),
                'selection_ratio': len(selected_features) / len(features_df.columns),
                'selection_method': selection_metrics['method'],
                'execution_mode': config.get('execution_mode', 'light'),
                'total_execution_time': total_time,
                'performance_stats': self.performance_stats,
                'optimization_used': True,
                'success': True
            }

            tprint_performance(f"✅ Optimized feature selection completed: {metrics['n_features_selected']}/{metrics['n_features_original']} features", total_time)
            
            # Generate outcome report
            report_path = self._generate_outcome_report(metrics, artifacts, config)
            if report_path:
                tprint(f"📊 Report generated: {report_path}", "INFO")
            
            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics
            }

        except Exception as e:
            error_msg = f"Optimized feature selection failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg)

            return {
                'success': False,
                'artifacts': {},
                'metrics': {'performance_stats': self.performance_stats},
                'error': error_msg
            }

    async def _load_features(self, config: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """
        Load features from the feature generation step with optimization.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            DataFrame containing features or None if loading fails
        """
        start_time = time.time()
        tprint("📊 Loading features with optimization...", "INFO")
        
        try:
            # Try to load from feature generation step artifacts
            symbol = config.get('symbol', 'UNKNOWN')
            exchange = config.get('exchange', 'binance')
            timeframe = config.get('timeframe', '15m')
            
            # Look for feature files in artifacts directory (from previous steps)
            artifacts_path = Path("artifacts")
            feature_paths = [
                # Look for generated features from feature_generation_step
                artifacts_path / f"feature_generation_feature_generation_step_generated_features_long_Analyst_*.parquet",
                # Look for optimized features from lookback optimization step
                artifacts_path / f"feature_generation_period_lookback_optimization_step_lookback_optimization_long_Analyst_*.parquet",
                # Look for labeled data that might contain features
                artifacts_path / f"feature_generation_labeling_integration_step_labeled_data_{symbol}_{timeframe}_long_Analyst_*.parquet",
                # Fallback to data_cache
                Path("data_cache") / "unified_cache" / f"{exchange}_{symbol}_{timeframe}_features.parquet",
                Path("data_cache") / "feature_states" / f"{exchange}_{symbol}_{timeframe}_features.parquet",
                Path("data_cache") / f"{exchange}" / f"{symbol}_{timeframe}_features.parquet"
            ]
            
            features_df = None
            for pattern in feature_paths:
                if '*' in str(pattern):
                    # Handle glob patterns
                    import glob
                    matching_files = glob.glob(str(pattern))
                    if matching_files:
                        # Get the most recent file
                        latest_file = max(matching_files, key=lambda x: Path(x).stat().st_mtime)
                        tprint(f"📁 Loading features from: {latest_file}", "INFO")
                        features_df = pd.read_parquet(latest_file)
                        break
                else:
                    # Handle direct paths
                    if pattern.exists():
                        tprint(f"📁 Loading features from: {pattern}", "INFO")
                        features_df = pd.read_parquet(pattern)
                        break
            
            # If no features found, raise an error instead of creating samples
            if features_df is None or features_df.empty:
                raise FileNotFoundError("No feature artifacts found from previous steps. Please run feature_generation_step first.")
            
            load_time = time.time() - start_time
            self.performance_stats['feature_loading_time'] = load_time
            
            tprint_success(f"✅ Features loaded: {features_df.shape[1]} features, {features_df.shape[0]} rows in {load_time:.2f}s")
            return features_df
            
        except Exception as e:
            tprint(f"❌ Feature loading failed: {e}", "ERROR")
            return None

    def _generate_outcome_report(self, metrics: Dict[str, Any], artifacts: Dict[str, Any], config: Dict[str, Any]) -> Optional[str]:
        """Generate outcome report in markdown format."""
        try:
            from pathlib import Path
            
            # Create outcomes directory if it doesn't exist
            outcomes_dir = Path('outcomes')
            outcomes_dir.mkdir(exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"{self.step_name}_report_{timestamp}.md"
            report_path = outcomes_dir / report_filename
            
            # Generate markdown report
            with open(report_path, 'w') as f:
                f.write(f"# Feature Selection Report\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Step:** {self.step_name}\n\n")
                
                f.write("## Configuration\n\n")
                f.write(f"- **Symbol:** {config.get('symbol', 'N/A')}\n")
                f.write(f"- **Exchange:** {config.get('exchange', 'N/A')}\n")
                f.write(f"- **Timeframe:** {config.get('timeframe', 'N/A')}\n")
                f.write(f"- **Execution Mode:** {config.get('execution_mode', 'N/A')}\n\n")
                
                f.write("## Feature Selection Results\n\n")
                f.write(f"- **Original Features:** {metrics.get('n_features_original', 'N/A')}\n")
                f.write(f"- **Selected Features:** {metrics.get('n_features_selected', 'N/A')}\n")
                f.write(f"- **Selection Ratio:** {metrics.get('selection_ratio', 0):.2%}\n")
                f.write(f"- **Selection Method:** {metrics.get('selection_method', 'N/A')}\n")
                f.write(f"- **Optimization Used:** {metrics.get('optimization_used', False)}\n\n")
                
                f.write("## Performance Metrics\n\n")
                performance_stats = metrics.get('performance_stats', {})
                f.write(f"- **Total Execution Time:** {metrics.get('total_execution_time', 0):.2f}s\n")
                f.write(f"- **Feature Loading Time:** {performance_stats.get('feature_loading_time', 0):.2f}s\n")
                f.write(f"- **Selection Time:** {performance_stats.get('selection_time', 0):.2f}s\n\n")
                
                # Add selected features list if available
                selected_features = artifacts.get('selected_features', {}).get('selected_features', [])
                if selected_features:
                    f.write("## Selected Features\n\n")
                    f.write("The following features were selected for the model:\n\n")
                    for i, feature in enumerate(selected_features[:50], 1):  # Show first 50 features
                        f.write(f"{i}. {feature}\n")
                    if len(selected_features) > 50:
                        f.write(f"\n... and {len(selected_features) - 50} more features\n")
                    f.write("\n")
                
                f.write("## Summary\n\n")
                f.write(f"Feature selection completed successfully using {metrics.get('selection_method', 'optimized')} method. ")
                f.write(f"Selected {metrics.get('n_features_selected', 0)} out of {metrics.get('n_features_original', 0)} features ")
                f.write(f"({metrics.get('selection_ratio', 0):.1%} selection ratio). ")
                f.write(f"Total execution time: {metrics.get('total_execution_time', 0):.2f} seconds.\n")
            
            return str(report_path)
            
        except Exception as e:
            tprint(f"❌ Report generation failed: {e}", "ERROR")
            return None


    async def _perform_optimized_selection(self, features_df: pd.DataFrame, config: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
        """
        Perform optimized feature selection using VectorBT and unified vectorization.
        
        Args:
            features_df: DataFrame containing features
            config: Configuration dictionary
            
        Returns:
            Tuple of (selected_features, selection_metrics)
        """
        start_time = time.time()
        tprint("🔍 Performing optimized feature selection...", "INFO")
        
        try:
            # Get selection configuration
            selection_config = config.get('feature_selection_config', {})
            method = selection_config.get('method', 'univariate_selection')
            # Use percentage-based selection instead of fixed number
            removal_percentage = selection_config.get('removal_percentage', 0.1)  # Remove 10% of low-variance features by default
            n_features = int(len(features_df.columns) * (1 - removal_percentage))
            
            tprint(f"🔧 Selection configuration: method={method}, removal_percentage={removal_percentage:.1%}, n_features={n_features}", "DEBUG")
            tprint(f"📊 Input features shape: {features_df.shape}", "DEBUG")
            tprint(f"📊 Input features dtypes: {features_df.dtypes.value_counts().to_dict()}", "DEBUG")
            
            # Use unified vectorization for feature selection
            if method == 'univariate_selection':
                tprint("🎯 Using univariate selection method", "INFO")
                selected_features, metrics = await self._univariate_selection_optimized(
                    features_df, n_features, removal_percentage
                )
            elif method == 'correlation_based':
                tprint("🎯 Using correlation-based selection method", "INFO")
                selected_features, metrics = await self._correlation_based_selection_optimized(
                    features_df, n_features, removal_percentage
                )
            elif method == 'variance_based':
                tprint("🎯 Using variance-based selection method", "INFO")
                selected_features, metrics = await self._variance_based_selection_optimized(
                    features_df, n_features, removal_percentage
                )
            elif method == 'stability_based':
                tprint("🎯 Using stability-based selection method", "INFO")
                selected_features, metrics = await self._stability_based_selection_optimized(
                    features_df, n_features, removal_percentage
                )
            else:
                # Default to univariate selection
                tprint(f"🎯 Unknown method '{method}', defaulting to univariate selection", "INFO")
                selected_features, metrics = await self._univariate_selection_optimized(
                    features_df, n_features, removal_percentage
                )
            
            selection_time = time.time() - start_time
            self.performance_stats['selection_time'] = selection_time
            
            tprint_success(f"✅ Feature selection completed: {len(selected_features)} features selected in {selection_time:.2f}s")
            return selected_features, metrics
            
        except Exception as e:
            tprint(f"❌ Feature selection failed: {e}", "ERROR")
            raise

    async def _univariate_selection_optimized(self, features_df: pd.DataFrame, n_features: int, removal_percentage: float = 0.1) -> tuple[List[str], Dict[str, Any]]:
        """Optimized univariate feature selection using VectorBT operations."""
        tprint("📈 Performing optimized univariate selection...", "INFO")
        tprint(f"🔍 Input data shape: {features_df.shape}", "DEBUG")
        tprint(f"🎯 Target features to select: {n_features}", "DEBUG")
        
        try:
            # Use VectorBT for efficient statistical calculations
            feature_scores = {}
            numeric_columns = 0
            processed_columns = 0
            valid_features = 0
            
            tprint(f"🔍 Processing {len(features_df.columns)} columns...", "DEBUG")
            
            for i, column in enumerate(features_df.columns):
                if i % 50 == 0:  # Log every 50 columns
                    tprint(f"🔍 Processing column {i+1}/{len(features_df.columns)}: {column}", "DEBUG")
                
                if pd.api.types.is_numeric_dtype(features_df[column]):
                    numeric_columns += 1
                    tprint(f"📊 Processing numeric column: {column} (dtype: {features_df[column].dtype})", "DEBUG")
                    
                    # Use VectorBT rolling operations for efficient calculations
                    data = features_df[column].dropna()
                    tprint(f"📊 Column {column}: {len(data)} non-null values out of {len(features_df[column])}", "DEBUG")
                    
                    if len(data) > 0:
                        try:
                            # Calculate variance using optimized rolling operations
                            window_size = min(20, len(data))
                            tprint(f"📊 Calculating variance for {column} with window size {window_size}", "DEBUG")
                            
                            variance_result = self.vectorbt_optimizer.rolling_var(data, window=window_size)
                            tprint(f"📊 Variance calculation result type: {type(variance_result)}", "DEBUG")
                            
                            if hasattr(variance_result, 'iloc'):
                                variance = variance_result.iloc[-1]
                                tprint(f"📊 Variance for {column}: {variance}", "DEBUG")
                            else:
                                variance = variance_result
                                tprint(f"📊 Direct variance for {column}: {variance}", "DEBUG")
                            
                            if not pd.isna(variance) and variance > 0:
                                feature_scores[column] = float(variance)
                                valid_features += 1
                                self.performance_stats['vectorbt_operations'] += 1
                                tprint(f"✅ Feature {column} added with variance: {variance}", "DEBUG")
                            else:
                                tprint(f"❌ Feature {column} rejected: variance={variance} (NaN or <= 0)", "DEBUG")
                        except Exception as e:
                            tprint(f"❌ VectorBT operation failed for {column}: {e}", "DEBUG")
                            # Try fallback to standard pandas variance
                            try:
                                fallback_variance = data.var()
                                tprint(f"📊 Fallback variance for {column}: {fallback_variance}", "DEBUG")
                                if not pd.isna(fallback_variance) and fallback_variance > 0:
                                    feature_scores[column] = float(fallback_variance)
                                    valid_features += 1
                                    tprint(f"✅ Feature {column} added with fallback variance: {fallback_variance}", "DEBUG")
                                else:
                                    tprint(f"❌ Feature {column} rejected in fallback: variance={fallback_variance}", "DEBUG")
                            except Exception as fallback_e:
                                tprint(f"❌ Fallback also failed for {column}: {fallback_e}", "DEBUG")
                        
                        processed_columns += 1
                    else:
                        tprint(f"❌ Column {column} has no valid data after dropna", "DEBUG")
                else:
                    tprint(f"⏭️ Skipping non-numeric column: {column} (dtype: {features_df[column].dtype})", "DEBUG")
            
            tprint(f"📊 Summary: {numeric_columns} numeric columns, {processed_columns} processed, {valid_features} valid features", "INFO")
            tprint(f"📊 Feature scores collected: {len(feature_scores)}", "INFO")
            
            if len(feature_scores) > 0:
                tprint(f"📊 Sample feature scores: {dict(list(feature_scores.items())[:5])}", "DEBUG")
            
            # Select top features by variance
            sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in sorted_features[:n_features]]
            
            tprint(f"📊 Final selection: {len(selected_features)} features selected", "INFO")
            if len(selected_features) > 0:
                tprint(f"📊 Selected features: {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}", "INFO")
            
            metrics = {
                'method': 'univariate_selection_optimized',
                'n_features_available': len(feature_scores),
                'n_features_selected': len(selected_features),
                'removal_percentage': removal_percentage,
                'features_removed': len(feature_scores) - len(selected_features),
                'selection_criteria': 'variance',
                'vectorbt_operations_used': self.performance_stats['vectorbt_operations'],
                'numeric_columns_found': numeric_columns,
                'processed_columns': processed_columns,
                'valid_features_found': valid_features
            }
            
            return selected_features, metrics
            
        except Exception as e:
            tprint(f"❌ Univariate selection failed: {e}", "ERROR")
            tprint(f"🔍 Falling back to simple selection of first {n_features} features", "INFO")
            # Fallback to simple selection
            fallback_features = features_df.columns[:n_features].tolist()
            tprint(f"📊 Fallback features: {fallback_features}", "INFO")
            return fallback_features, {'method': 'fallback_selection'}

    async def _correlation_based_selection_optimized(self, features_df: pd.DataFrame, n_features: int) -> tuple[List[str], Dict[str, Any]]:
        """Optimized correlation-based feature selection."""
        tprint("🔗 Performing optimized correlation-based selection...", "INFO")
        
        try:
            # Use unified vectorization for correlation calculations
            numeric_features = features_df.select_dtypes(include=[np.number])
            correlation_matrix = numeric_features.corr().abs()
            
            # Find features with low correlation to each other
            selected_features = []
            remaining_features = list(numeric_features.columns)
            
            while len(selected_features) < n_features and remaining_features:
                # Select feature with highest variance
                variances = numeric_features[remaining_features].var()
                best_feature = variances.idxmax()
                selected_features.append(best_feature)
                remaining_features.remove(best_feature)
                
                # Remove highly correlated features
                if best_feature in correlation_matrix.columns:
                    high_corr = correlation_matrix[best_feature] > 0.8
                    remaining_features = [f for f in remaining_features if not high_corr.get(f, False)]
            
            metrics = {
                'method': 'correlation_based_optimized',
                'n_features_available': len(numeric_features.columns),
                'n_features_selected': len(selected_features),
                'selection_criteria': 'low_correlation_high_variance'
            }
            
            return selected_features, metrics
            
        except Exception as e:
            tprint(f"❌ Correlation-based selection failed: {e}", "ERROR")
            return features_df.columns[:n_features].tolist(), {'method': 'fallback_selection'}

    async def _variance_based_selection_optimized(self, features_df: pd.DataFrame, n_features: int) -> tuple[List[str], Dict[str, Any]]:
        """Optimized variance-based feature selection using VectorBT."""
        tprint("📊 Performing optimized variance-based selection...", "INFO")
        
        try:
            # Use VectorBT for efficient variance calculations
            feature_variances = {}
            
            for column in features_df.columns:
                if pd.api.types.is_numeric_dtype(features_df[column]):
                    data = features_df[column].dropna()
                    if len(data) > 0:
                        # Use VectorBT rolling variance for efficiency
                        variance = self.vectorbt_optimizer.rolling_var(data, window=min(10, len(data))).iloc[-1]
                        if not pd.isna(variance) and variance > 0:
                            feature_variances[column] = float(variance)
                            self.performance_stats['vectorbt_operations'] += 1
            
            # Select features with highest variance
            sorted_features = sorted(feature_variances.items(), key=lambda x: x[1], reverse=True)
            selected_features = [feat[0] for feat in sorted_features[:n_features]]
            
            metrics = {
                'method': 'variance_based_optimized',
                'n_features_available': len(feature_variances),
                'n_features_selected': len(selected_features),
                'selection_criteria': 'variance',
                'vectorbt_operations_used': self.performance_stats['vectorbt_operations']
            }
            
            return selected_features, metrics
            
        except Exception as e:
            tprint(f"❌ Variance-based selection failed: {e}", "ERROR")
            return features_df.columns[:n_features].tolist(), {'method': 'fallback_selection'}

    async def _stability_based_selection_optimized(self, features_df: pd.DataFrame, n_features: int, removal_percentage: float = 0.1) -> tuple[List[str], Dict[str, Any]]:
        """Optimized stability-based feature selection using CV stability analysis."""
        tprint("🔒 Performing optimized stability-based selection...", "INFO")
        tprint(f"🔍 Input data shape: {features_df.shape}", "DEBUG")
        tprint(f"🎯 Target features to select: {n_features}", "DEBUG")
        
        try:
            from sklearn.model_selection import KFold
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            import numpy as np
            
            # Configuration for stability analysis
            stability_config = {
                'cv_folds': 5,
                'stability_threshold': 0.7,
                'max_removal_percentage': 0.10,  # Max 10% removal
                'min_samples': 100
            }
            
            tprint(f"🔧 Stability configuration: {stability_config}", "DEBUG")
            
            # Get numeric features only
            numeric_features = features_df.select_dtypes(include=[np.number])
            tprint(f"📊 Numeric features found: {len(numeric_features.columns)}", "DEBUG")
            
            if len(numeric_features.columns) == 0:
                tprint("❌ No numeric features found for stability analysis", "ERROR")
                return [], {}
            
            # Check if we have enough data for CV
            if len(numeric_features) < stability_config['min_samples']:
                tprint(f"⚠️ Insufficient data for stability analysis: {len(numeric_features)} < {stability_config['min_samples']}", "WARNING")
                # Fallback to variance-based selection
                return await self._variance_based_selection_optimized(features_df, n_features, removal_percentage)
            
            # Create a dummy target for stability analysis (using first feature as proxy)
            # In real implementation, this would use the actual target variable
            target_proxy = numeric_features.iloc[:, 0]  # Use first feature as proxy target
            
            stability_scores = {}
            cv_importance_scores = {}
            
            tprint("🔄 Computing stability scores for each feature...", "INFO")
            
            for i, column in enumerate(numeric_features.columns):
                try:
                    feature_data = numeric_features[column].dropna()
                    
                    if len(feature_data) < stability_config['min_samples']:
                        tprint(f"⏭️ Skipping {column}: insufficient data ({len(feature_data)} samples)", "DEBUG")
                        continue
                    
                    # Calculate CV stability
                    cv_scores = []
                    kf = KFold(n_splits=stability_config['cv_folds'], shuffle=True, random_state=42)
                    
                    for train_idx, val_idx in kf.split(feature_data):
                        try:
                            # Create feature matrix and target for this fold
                            X_fold = feature_data.iloc[train_idx].values.reshape(-1, 1)
                            y_fold = target_proxy.iloc[train_idx].values
                            
                            if len(X_fold) < 10:  # Skip if too few samples
                                continue
                                
                            # Train a simple model to get feature importance
                            rf = RandomForestRegressor(n_estimators=10, random_state=42, max_depth=3)
                            rf.fit(X_fold, y_fold)
                            
                            # Get feature importance (stability proxy)
                            importance = rf.feature_importances_[0] if len(rf.feature_importances_) > 0 else 0
                            cv_scores.append(importance)
                            
                        except Exception as e:
                            tprint(f"⚠️ Error in CV fold for {column}: {e}", "DEBUG")
                            continue
                    
                    if len(cv_scores) >= 2:  # Need at least 2 valid folds
                        # Calculate stability as consistency of importance across folds
                        cv_std = np.std(cv_scores)
                        cv_mean = np.mean(cv_scores)
                        
                        # Stability score: higher mean, lower std = more stable
                        stability_score = cv_mean / (1 + cv_std) if cv_std > 0 else cv_mean
                        stability_scores[column] = stability_score
                        cv_importance_scores[column] = cv_scores
                        
                        tprint(f"📊 {column}: stability={stability_score:.4f}, cv_std={cv_std:.4f}", "DEBUG")
                    else:
                        tprint(f"⏭️ Skipping {column}: insufficient valid CV folds", "DEBUG")
                        
                except Exception as e:
                    tprint(f"⚠️ Error processing {column}: {e}", "DEBUG")
                    continue
            
            if not stability_scores:
                tprint("❌ No features passed stability analysis", "ERROR")
                return [], {}
            
            # Sort features by stability score
            sorted_features = sorted(stability_scores.items(), key=lambda x: x[1], reverse=True)
            tprint(f"📈 Stability analysis completed: {len(sorted_features)} features analyzed", "INFO")
            
            # Apply stability threshold
            stable_features = [f for f, score in sorted_features if score >= stability_config['stability_threshold']]
            tprint(f"🔒 Features passing stability threshold: {len(stable_features)}/{len(sorted_features)}", "INFO")
            
            # If we have too many stable features, select top n_features
            if len(stable_features) > n_features:
                selected_features = stable_features[:n_features]
                tprint(f"🎯 Selected top {len(selected_features)} stable features", "INFO")
            else:
                selected_features = stable_features
                tprint(f"🎯 All {len(selected_features)} stable features selected", "INFO")
            
            # Calculate removal statistics
            features_removed = len(stability_scores) - len(selected_features)
            actual_removal_percentage = features_removed / len(stability_scores) if len(stability_scores) > 0 else 0
            
            tprint(f"📊 Stability selection results: {len(selected_features)} selected, {features_removed} removed ({actual_removal_percentage:.1%})", "INFO")
            
            # Update performance stats
            self.performance_stats['stability_operations'] = len(stability_scores)
            self.performance_stats['cv_folds_processed'] = sum(len(scores) for scores in cv_importance_scores.values())
            
            metrics = {
                'method': 'stability_based_optimized',
                'n_features_available': len(stability_scores),
                'n_features_selected': len(selected_features),
                'removal_percentage': actual_removal_percentage,
                'features_removed': features_removed,
                'selection_criteria': 'cv_stability',
                'stability_threshold': stability_config['stability_threshold'],
                'cv_folds': stability_config['cv_folds'],
                'stability_operations_used': self.performance_stats['stability_operations'],
                'cv_folds_processed': self.performance_stats['cv_folds_processed'],
                'numeric_columns_found': len(numeric_features.columns),
                'stable_features_found': len(stable_features)
            }
            
            return selected_features, metrics
            
        except Exception as e:
            tprint(f"❌ Stability-based selection failed: {e}", "ERROR")
            raise

    async def run(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run method required by BaseStep interface."""
        return await self.execute(config)


# Register the step
def register_feature_generation_feature_selection_step():
    """Register the feature selection step."""
    from src.training.steps.base_step import step_registry

    step_registry.register("feature_generation_feature_selection_step", FeatureGenerationFeatureSelectionStep)
    tprint("✅ Feature generation feature selection step registered", "SUCCESS")


# Auto-register when module is imported
register_feature_generation_feature_selection_step()
