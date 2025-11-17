"""
Statsmodel Clustering Pipeline Step

This module provides the main entry point for statsmodel Markov-switching clustering.
It connects the statsmodel_clustering components with the core pipeline infrastructure,
integrating with BaseStep, ares_launcher, and the artifact management system.

The pipeline:
1. Loads market data using BaseStep methods
2. Generates features using feature bank (based on cluster_features.config)
3. Applies transformations (scaling, rolling windows, PCA)
4. Performs Markov-switching clustering
5. Assesses cluster quality
6. Generates cluster artifacts for downstream steps

Supports execution modes with centrally configured lookback windows:
- blank: Uses centralized blank-mode lookback days from ares_launcher
- light: Uses centralized light-mode lookback days from ares_launcher
- full: Complete data as defined by ares_launcher

Author: Claude Code
Version: 1.0
"""

import asyncio
import configparser
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Scikit-learn imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler

# Base step
from src.training.steps.base_step import BaseStep

# Feature generation
from src.feature_generation.core.feature_bank import FeatureBank, get_global_feature_bank
from src.feature_generation.categories.regime_feature_categorization import (
    RegimeFeatureCategorizer,
    FeatureUseCase,
    get_regime_clustering_features
)
from src.feature_generation.categories.regime_feature_integration import (
    RegimeFeatureIntegration,
    RegimeFeatureConfig,
    generate_regime_features
)

# Statsmodel clustering components
from src.training.steps.market_analysis.statsmodel_clustering.core.markov_regression_adapter import (
    MarkovRegressionAdapter
)
from src.training.steps.market_analysis.statsmodel_clustering.feature_engineering.enhanced_features import (
    EnhancedFeatureEngineer
)

# Clustering quality assessment
from src.training.steps.market_analysis.clusters.cluster_quality_assessor import (
    create_cluster_quality_assessor,
    ClusterQualityMetrics
)
from src.training.steps.market_analysis.clusters.clustering_optimization_goals import (
    DEFAULT_CLUSTERING_GOALS,
    DEFAULT_OPTIMIZATION_TARGETS,
    format_metrics_report
)

# Utilities
from src.utils.logger import system_logger
from src.utils.tprint import tprint
from src.training.steps.market_analysis.statsmodel_clustering.feature_engineering.rank_normalization import (
    RankNormalizer,
)
from src.training.steps.market_analysis.shared_utils.execution_mode_lookback_config import (
    get_execution_mode_config,
)
from src.features_common.transforms import (
    zscore_normalize,
    robust_normalize
)

logger = logging.getLogger(__name__)


class StatsmodelClusteringPipelineStep(BaseStep):
    """
    Statsmodel Clustering Pipeline Step.

    Main entry point for statsmodel Markov-switching clustering that integrates
    with the Ares pipeline infrastructure.
    """

    def __init__(self, step_name: str = "statsmodel_clustering_pipeline"):
        """Initialize the statsmodel clustering pipeline step."""
        super().__init__(step_name)
        self.logger = system_logger.getChild('StatsmodelClusteringPipeline')

        # Load configuration
        self.config_path = Path(__file__).parent / "statsmodel_clustering" / "cluster_features.config"
        self.feature_config = self._load_feature_config()

        # Initialize components (lazy loading)
        self._feature_bank = None
        self._markov_adapter = None
        self._feature_engineer = None
        self._regime_categorizer = None

        tprint(f"✅ Initialized Statsmodel Clustering Pipeline Step", "SUCCESS")

    @property
    def feature_bank(self):
        """Lazy initialization of feature bank."""
        if self._feature_bank is None:
            self._feature_bank = get_global_feature_bank()
        return self._feature_bank

    @property
    def markov_adapter(self):
        """Lazy initialization of Markov regression adapter."""
        if self._markov_adapter is None:
            self._markov_adapter = MarkovRegressionAdapter()
        return self._markov_adapter

    @property
    def feature_engineer(self):
        """Lazy initialization of feature engineer."""
        if self._feature_engineer is None:
            self._feature_engineer = EnhancedFeatureEngineer()
        return self._feature_engineer

    @property
    def regime_categorizer(self):
        """Lazy initialization of regime categorizer."""
        if self._regime_categorizer is None:
            self._regime_categorizer = RegimeFeatureCategorizer()
        return self._regime_categorizer

    def _load_feature_config(self) -> configparser.ConfigParser:
        """
        Load feature configuration from cluster_features.config.

        Returns:
            ConfigParser object with feature configuration
        """
        config = configparser.ConfigParser()

        if not self.config_path.exists():
            self.logger.warning(f"⚠️ Feature config not found at {self.config_path}, using defaults")
            return config

        try:
            config.read(self.config_path)
            tprint(f"✅ Loaded feature config from {self.config_path}", "SUCCESS")
            return config
        except Exception as e:
            self.logger.error(f"❌ Failed to load feature config: {e}")
            return configparser.ConfigParser()

    async def execute(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the statsmodel clustering pipeline.

        Args:
            config: Configuration dictionary containing:
                - symbol: Trading symbol (e.g., 'ETHUSDT')
                - exchange: Exchange name (e.g., 'binance')
                - timeframe: Timeframe (e.g., '1h')
                - execution_mode: 'full', 'light', or 'blank'
                - direction: Trading direction (e.g., 'long')

        Returns:
            Dict containing:
            - 'success': bool indicating if step completed successfully
            - 'artifacts': list of artifact paths created
            - 'metrics': dict of performance metrics
            - 'error': error message if step failed (optional)
        """
        start_time = datetime.now()

        symbol = config.get('symbol', 'ETHUSDT')
        exchange = config.get('exchange', 'binance')
        timeframe = config.get('timeframe', '1h')
        execution_mode = config.get('execution_mode', 'light')
        direction = config.get('direction', 'long')

        tprint(f"🚀 Starting Statsmodel Clustering Pipeline for {symbol} ({execution_mode} mode)", "INFO")

        # Set context for artifact management
        self.set_context(
            step_name=self.step_name,
            symbol=symbol,
            exchange=exchange,
            datetime=datetime.now(),
            direction=direction,
            model="Analyst"
        )

        try:
            # Step 1: Load market data
            tprint("📥 Step 1: Loading market data...", "INFO")
            market_data = await self._load_market_data(config)

            if market_data is None or len(market_data) < 100:
                raise ValueError(f"Insufficient market data: {len(market_data) if market_data is not None else 0} samples")

            tprint(f"✅ Loaded {len(market_data)} samples of market data", "SUCCESS")

            # Step 2: Generate features
            tprint("🔧 Step 2: Generating features...", "INFO")
            features = await self._generate_features(market_data, config)

            if features is None or len(features) == 0:
                raise ValueError("Feature generation failed or returned empty features")

            tprint(f"✅ Generated {len(features.columns)} features from {len(features)} samples", "SUCCESS")

            # Step 3: Transform features (scaling, rolling, PCA)
            tprint("🔄 Step 3: Transforming features...", "INFO")
            transformed_features, transformers = await self._transform_features(features, config)

            tprint(f"✅ Transformed features to {len(transformed_features.columns)} dimensions", "SUCCESS")

            # Step 4: Perform Markov-switching clustering
            tprint("🎯 Step 4: Performing Markov-switching clustering...", "INFO")
            clustering_result = await self._perform_clustering(transformed_features, config)

            tprint(f"✅ Clustering completed: {clustering_result['n_regimes']} regimes identified", "SUCCESS")

            # Step 5: Assess cluster quality
            tprint("📊 Step 5: Assessing cluster quality...", "INFO")

            # Add regime probabilities to config for quality assessment
            quality_config = config.copy()
            quality_config['regime_probabilities'] = clustering_result.get('regime_probabilities')

            quality_metrics = await self._assess_cluster_quality(
                features=transformed_features,
                labels=clustering_result['regime_labels'],
                market_data=market_data,
                config=quality_config
            )

            tprint(f"✅ Quality assessment completed", "SUCCESS")

            # Generate CSV quality report
            tprint("📄 Generating quality report CSV...", "INFO")
            assessor = create_cluster_quality_assessor(artifact_manager=self.artifact_manager)
            csv_quality_path, csv_trials_path = assessor.generate_comprehensive_csv_report(
                metrics=quality_metrics,
                all_trials=None,  # Could pass if we have multiple trials
                symbol=symbol,
                output_dir='outcomes',
                method_specific_config={
                    'k_regimes': clustering_result['n_regimes'],
                    'aic': clustering_result.get('aic'),
                    'bic': clustering_result.get('bic'),
                    'log_likelihood': clustering_result.get('log_likelihood')
                }
            )

            # Step 6: Generate artifacts
            tprint("💾 Step 6: Generating artifacts...", "INFO")
            artifacts = await self._generate_artifacts(
                market_data=market_data,
                features=features,
                transformed_features=transformed_features,
                clustering_result=clustering_result,
                quality_metrics=quality_metrics,
                transformers=transformers,
                config=config
            )

            # Add CSV report to artifacts if generated
            if csv_quality_path:
                artifacts.append(csv_quality_path)
                tprint(f"✅ Quality report saved to: {csv_quality_path}", "SUCCESS")
            else:
                tprint(f"⚠️ Failed to generate quality report CSV", "WARNING")

            tprint(f"✅ Generated {len(artifacts)} artifacts", "SUCCESS")

            # Step 7: Create metrics summary
            execution_time = (datetime.now() - start_time).total_seconds()
            metrics = self._create_metrics_summary(
                clustering_result=clustering_result,
                quality_metrics=quality_metrics,
                execution_time=execution_time,
                n_features=len(transformed_features.columns),
                n_samples=len(market_data)
            )

            tprint(f"✅ Statsmodel Clustering Pipeline completed in {execution_time:.2f}s", "SUCCESS")

            return {
                'success': True,
                'artifacts': artifacts,
                'metrics': metrics,
                'execution_time': execution_time
            }

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Statsmodel clustering pipeline failed: {str(e)}"
            tprint(f"❌ {error_msg}", "ERROR")
            self.logger.error(error_msg, exc_info=True)

            return {
                'success': False,
                'error': error_msg,
                'artifacts': [],
                'metrics': {},
                'execution_time': execution_time
            }

    async def _load_market_data(self, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Load market data using BaseStep methods and artifact manager.

        Args:
            config: Configuration dictionary

        Returns:
            DataFrame with market data (OHLCV)
        """
        symbol = config.get('symbol', 'ETHUSDT')
        exchange = config.get('exchange', 'binance')
        timeframe = config.get('timeframe', '1h')
        execution_mode = config.get('execution_mode', 'light')

        # Try to load from artifacts first (from data collection step)
        try:
            market_data = self._get_artifact(
                artifact_name=f"market_data_{symbol}_{exchange}_{timeframe}",
                artifact_type='data'
            )

            if market_data is not None and len(market_data) > 0:
                # Apply execution mode filtering
                market_data = self._apply_execution_mode_filter(market_data, execution_mode, timeframe)
                return market_data

        except Exception as e:
            self.logger.debug(f"Could not load from artifacts: {e}")

        # If not in artifacts, we need to raise an error
        # The data collection step should have run before this
        raise ValueError(
            f"Market data not found in artifacts. "
            f"Please run data collection step first for {symbol}/{exchange}/{timeframe}"
        )

    def _apply_execution_mode_filter(
        self,
        data: pd.DataFrame,
        execution_mode: str,
        timeframe: str
    ) -> pd.DataFrame:
        """Apply execution mode filtering to data using centralized lookback days.

        Args:
            data: Market data DataFrame
            execution_mode: 'blank', 'light', or 'full'
            timeframe: Timeframe string

        Returns:
            Filtered DataFrame
        """
        if execution_mode == 'full':
            return data

        # Resolve days limit from centralized execution mode configuration
        mode = (execution_mode or 'light').lower()
        exec_config = get_execution_mode_config()
        days_limit = exec_config.get_data_loading_days(mode)

        # Fallback: if config returns None (e.g., full mode), do not filter
        if days_limit is None:
            return data

        # Calculate samples based on timeframe
        samples_per_day_map = {
            '1m': 1440, '3m': 480, '5m': 288, '15m': 96,
            '30m': 48, '1h': 24, '4h': 6, '1d': 1
        }

        samples_per_day = samples_per_day_map.get(timeframe, 24)  # Default to 1h
        limit = days_limit * samples_per_day

        if len(data) > limit:
            filtered = data.tail(limit).copy()
            tprint(
                f"⏱️ {execution_mode.upper()} mode: filtered data from {len(data):,} to {len(filtered):,} samples "
                f"({days_limit} days of {timeframe} data)",
                "INFO"
            )
            return filtered

        return data

    async def _generate_features(
        self,
        market_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Generate features using feature bank based on cluster_features.config.

        Args:
            market_data: Market data DataFrame
            config: Configuration dictionary

        Returns:
            DataFrame with generated features
        """
        # Get enabled feature categories from config
        enabled_categories = self.feature_config.get(
            'feature_categories',
            'enabled_categories',
            fallback='volatility,volume,trend,momentum'
        ).split(',')
        enabled_categories = [cat.strip() for cat in enabled_categories]

        # Check if we should use regime categorization
        use_regime_categorization = self.feature_config.getboolean(
            'advanced',
            'use_regime_categorization',
            fallback=True
        )

        all_features = []

        # Generate regime-specific features if enabled
        if use_regime_categorization:
            regime_use_case_str = self.feature_config.get(
                'advanced',
                'regime_use_case',
                fallback='REGIME_CLUSTERING'
            )

            try:
                regime_use_case = FeatureUseCase[regime_use_case_str]
                regime_features = self.regime_categorizer.get_features_for_use_case(regime_use_case)

                # Generate regime integration features
                regime_config = RegimeFeatureConfig(
                    lookback_period=int(self.feature_config.get('general', 'lookback_period', fallback='20')),
                    enable_regime_detection=True,
                    enable_adaptive_features=True,
                    enable_regime_transitions=True
                )

                regime_feature_gen = RegimeFeatureIntegration(config=regime_config)
                regime_feature_dict = regime_feature_gen._generate_regime_features(market_data)

                if regime_feature_dict:
                    regime_df = pd.DataFrame([regime_feature_dict], index=[market_data.index[-1]])
                    # Broadcast to all indices
                    regime_df = regime_df.reindex(market_data.index, method='ffill')
                    all_features.append(regime_df)

                tprint(f"✅ Generated {len(regime_feature_dict)} regime-specific features", "SUCCESS")

            except Exception as e:
                self.logger.warning(f"⚠️ Failed to generate regime features: {e}")

        # Generate features from each enabled category using feature bank
        for category in enabled_categories:
            try:
                category_features = await self._generate_category_features(
                    market_data, category, config
                )
                if category_features is not None and not category_features.empty:
                    all_features.append(category_features)
                    tprint(f"✅ Generated {len(category_features.columns)} {category} features", "SUCCESS")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to generate {category} features: {e}")

        if not all_features:
            raise ValueError("No features were successfully generated")

        # Combine all features
        features = pd.concat(all_features, axis=1)

        # Handle NaN values
        drop_nan = self.feature_config.getboolean('advanced', 'drop_nan', fallback=True)
        if drop_nan:
            features = features.dropna()
            tprint(f"ℹ️ Dropped NaN values, {len(features)} samples remaining", "INFO")
        else:
            features = features.fillna(method='ffill').fillna(method='bfill').fillna(0)

        # Drop constant features
        drop_constant = self.feature_config.getboolean('advanced', 'drop_constant', fallback=True)
        if drop_constant:
            constant_cols = features.columns[features.std() == 0].tolist()
            if constant_cols:
                features = features.drop(columns=constant_cols)
                tprint(f"ℹ️ Dropped {len(constant_cols)} constant features", "INFO")

        return features

    async def _generate_category_features(
        self,
        market_data: pd.DataFrame,
        category: str,
        config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Generate features for a specific category.

        Args:
            market_data: Market data DataFrame
            category: Feature category name
            config: Configuration dictionary

        Returns:
            DataFrame with features for this category
        """
        category_section = f'features.{category}'

        if not self.feature_config.has_section(category_section):
            self.logger.warning(f"⚠️ No configuration for category: {category}")
            return None

        # Get feature list from config
        feature_names = self.feature_config.get(
            category_section,
            'features',
            fallback=''
        ).split(',')
        feature_names = [f.strip() for f in feature_names if f.strip()]

        if not feature_names:
            return None

        # Get lookback periods if specified
        lookback_periods = None
        if self.feature_config.has_option(category_section, 'lookback_periods'):
            lookback_str = self.feature_config.get(category_section, 'lookback_periods')
            lookback_periods = [int(p.strip()) for p in lookback_str.split(',')]

        # Use feature bank to generate features
        # This is a simplified version - actual implementation would use the feature bank's API
        category_features = pd.DataFrame(index=market_data.index)

        # For now, use the enhanced feature engineer
        # In a full implementation, you would call the feature bank directly
        try:
            engineered_features = self.feature_engineer.generate_features(
                market_data,
                feature_names=feature_names,
                lookback_periods=lookback_periods
            )

            if engineered_features is not None:
                category_features = pd.concat([category_features, engineered_features], axis=1)
        except Exception as e:
            self.logger.warning(f"⚠️ Feature engineer failed for {category}: {e}")

        return category_features if not category_features.empty else None

    async def _transform_features(
        self,
        features: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Apply transformations to features (scaling, rolling windows, PCA).

        Args:
            features: Input features DataFrame
            config: Configuration dictionary

        Returns:
            Tuple of (transformed features DataFrame, transformers dict)
        """
        transformers = {}
        transformed = features.copy()

        # Step 1: Normalization/Scaling
        tprint("🔄 Applying normalization...", "INFO")

        # Get scaler type from config
        default_scaler = self.feature_config.get('normalization', 'default_scaler', fallback='RobustScaler')
        volatile_scaler = self.feature_config.get('normalization', 'volatile_features_scaler', fallback='RankPercentile')

        # Apply scaling
        if default_scaler == 'RobustScaler':
            scaler = RobustScaler()
        elif default_scaler == 'StandardScaler':
            scaler = StandardScaler()
        elif default_scaler == 'MinMaxScaler':
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()  # Default fallback

        # Identify volatile features
        volatile_features_str = self.feature_config.get('normalization', 'volatile_features', fallback='')
        volatile_keywords = [kw.strip() for kw in volatile_features_str.split(',') if kw.strip()]

        volatile_cols = []
        normal_cols = []

        for col in transformed.columns:
            is_volatile = any(keyword in col.lower() for keyword in volatile_keywords)
            if is_volatile:
                volatile_cols.append(col)
            else:
                normal_cols.append(col)

        # Scale normal features with default scaler
        if normal_cols:
            transformed[normal_cols] = scaler.fit_transform(transformed[normal_cols])
            transformers['scaler_normal'] = scaler
            tprint(f"✅ Scaled {len(normal_cols)} features with {default_scaler}", "SUCCESS")

        # Scale volatile features with rank percentile
        if volatile_cols and volatile_scaler == 'RankPercentile':
            rank_normalizer = RankNormalizer()
            for col in volatile_cols:
                transformed[col] = rank_normalizer.fit_transform(transformed[[col]]).flatten()
            transformers['rank_normalizer'] = rank_normalizer
            tprint(f"✅ Applied rank percentile normalization to {len(volatile_cols)} volatile features", "SUCCESS")

        # Step 2: Rolling windows
        include_rolling = self.feature_config.getboolean('rolling_windows', 'include_window_stats', fallback=True)

        if include_rolling:
            windows_str = self.feature_config.get('rolling_windows', 'windows', fallback='24,72')
            windows = [int(w.strip()) for w in windows_str.split(',')]

            stats_str = self.feature_config.get('rolling_windows', 'window_stats', fallback='mean,std')
            stats = [s.strip() for s in stats_str.split(',')]

            rolling_features = []
            for window in windows:
                for stat in stats:
                    if stat == 'mean':
                        rolling_feat = transformed.rolling(window=window).mean()
                        rolling_feat.columns = [f"{col}_roll{window}_mean" for col in rolling_feat.columns]
                        rolling_features.append(rolling_feat)
                    elif stat == 'std':
                        rolling_feat = transformed.rolling(window=window).std()
                        rolling_feat.columns = [f"{col}_roll{window}_std" for col in rolling_feat.columns]
                        rolling_features.append(rolling_feat)
                    elif stat == 'min':
                        rolling_feat = transformed.rolling(window=window).min()
                        rolling_feat.columns = [f"{col}_roll{window}_min" for col in rolling_feat.columns]
                        rolling_features.append(rolling_feat)
                    elif stat == 'max':
                        rolling_feat = transformed.rolling(window=window).max()
                        rolling_feat.columns = [f"{col}_roll{window}_max" for col in rolling_feat.columns]
                        rolling_features.append(rolling_feat)

            if rolling_features:
                transformed = pd.concat([transformed] + rolling_features, axis=1)
                # Drop NaN from rolling windows
                transformed = transformed.dropna()
                tprint(f"✅ Added rolling window features: {len(windows)} windows × {len(stats)} stats", "SUCCESS")

        # Step 3: Multi-timeframe features
        multi_tf_enabled = self.feature_config.getboolean('multi_timeframe', 'enabled', fallback=False)

        if multi_tf_enabled:
            # This would require loading data from multiple timeframes
            # For now, we'll skip this as it requires additional data loading
            tprint("ℹ️ Multi-timeframe features enabled but not implemented in this version", "INFO")

        # Step 4: PCA dimensionality reduction
        pca_enabled = self.feature_config.getboolean('pca', 'enabled', fallback=True)

        if pca_enabled:
            n_components = self.feature_config.getint('pca', 'n_components', fallback=12)
            scale_before_pca = self.feature_config.getboolean('pca', 'scale_before_pca', fallback=True)

            # Scale before PCA if requested
            if scale_before_pca:
                pca_scaler_type = self.feature_config.get('pca', 'pca_scaler', fallback='StandardScaler')
                if pca_scaler_type == 'StandardScaler':
                    pca_scaler = StandardScaler()
                else:
                    pca_scaler = RobustScaler()

                transformed_for_pca = pca_scaler.fit_transform(transformed)
                transformers['pca_scaler'] = pca_scaler
            else:
                transformed_for_pca = transformed.values

            # Apply PCA
            n_components = min(n_components, transformed.shape[1], transformed.shape[0])
            pca = PCA(n_components=n_components)
            pca_features = pca.fit_transform(transformed_for_pca)

            # Create DataFrame with PCA features
            pca_df = pd.DataFrame(
                pca_features,
                index=transformed.index,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )

            transformers['pca'] = pca
            transformed = pca_df

            explained_variance = pca.explained_variance_ratio_.sum()
            tprint(
                f"✅ Applied PCA: {n_components} components explaining {explained_variance:.2%} variance",
                "SUCCESS"
            )

        return transformed, transformers

    async def _perform_clustering(
        self,
        features: pd.DataFrame,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Perform Markov-switching clustering using MarkovRegressionAdapter.

        Args:
            features: Transformed features DataFrame
            config: Configuration dictionary

        Returns:
            Dictionary with clustering results
        """
        # Prepare data for Markov regression
        X = features.values

        # Fit Markov-switching model
        # The adapter will handle the complexity of Markov regression
        try:
            result = self.markov_adapter.fit(X)

            # Extract cluster labels from MarkovRegressionResult dataclass
            cluster_labels = result.cluster_labels

            if cluster_labels is None or len(cluster_labels) == 0:
                raise ValueError("Markov adapter did not return cluster labels")

            # If we got probabilities, convert to hard labels
            if len(cluster_labels.shape) > 1:
                cluster_labels = cluster_labels.argmax(axis=1)

            n_regimes = len(np.unique(cluster_labels))

            return {
                'regime_labels': cluster_labels,  # Rename for consistency downstream
                'n_regimes': n_regimes,
                'model': result.fitted_model,
                'regime_probabilities': result.cluster_probabilities,
                'transition_matrix': result.transition_matrix,
                'regime_means': result.regime_params,
                'regime_covariances': result.regime_params,
                'aic': result.aic,
                'bic': result.bic,
                'log_likelihood': result.log_likelihood
            }

        except Exception as e:
            self.logger.error(f"❌ Markov clustering failed: {e}")
            raise

    async def _assess_cluster_quality(
        self,
        features: pd.DataFrame,
        labels: np.ndarray,
        market_data: pd.DataFrame,
        config: Dict[str, Any]
    ) -> ClusterQualityMetrics:
        """
        Assess cluster quality using unified cluster quality assessor.

        Args:
            features: Transformed features used for clustering
            labels: Cluster labels
            market_data: Original market data
            config: Configuration dictionary

        Returns:
            ClusterQualityMetrics object
        """
        # Create quality assessor
        assessor = create_cluster_quality_assessor(
            artifact_manager=self.artifact_manager
        )

        # Assess quality with regime probabilities
        quality_metrics = assessor.assess_clustering_quality(
            features=features.values,
            cluster_labels=labels,
            market_data=market_data,
            regime_probabilities=config.get('regime_probabilities'),  # Pass probabilities from clustering result
            config=config
        )

        return quality_metrics

    async def _generate_artifacts(
        self,
        market_data: pd.DataFrame,
        features: pd.DataFrame,
        transformed_features: pd.DataFrame,
        clustering_result: Dict[str, Any],
        quality_metrics: ClusterQualityMetrics,
        transformers: Dict[str, Any],
        config: Dict[str, Any]
    ) -> List[str]:
        """
        Generate and save clustering artifacts for downstream steps.

        Args:
            market_data: Original market data
            features: Generated features
            transformed_features: Transformed features
            clustering_result: Clustering results
            quality_metrics: Quality assessment metrics
            transformers: Feature transformers (scalers, PCA, etc.)
            config: Configuration dictionary

        Returns:
            List of artifact paths
        """
        artifacts = []

        # 1. Save cluster labels with market data
        clustered_data = market_data.copy()
        clustered_data['regime'] = clustering_result['regime_labels']

        artifact_path = self._save_artifact(
            data=clustered_data,
            artifact_name='statsmodel_clustered_data',
            artifact_type='data',
            metadata={
                'n_regimes': clustering_result['n_regimes'],
                'n_samples': len(clustered_data),
                'n_features_used': len(transformed_features.columns),
                'aic': clustering_result.get('aic'),
                'bic': clustering_result.get('bic')
            }
        )
        artifacts.append(artifact_path)

        # 2. Save regime probabilities
        if clustering_result.get('regime_probabilities') is not None:
            regime_probs_df = pd.DataFrame(
                clustering_result['regime_probabilities'],
                index=market_data.index,
                columns=[f'regime_{i}_prob' for i in range(clustering_result['n_regimes'])]
            )

            artifact_path = self._save_artifact(
                data=regime_probs_df,
                artifact_name='statsmodel_regime_probabilities',
                artifact_type='data',
                metadata={'n_regimes': clustering_result['n_regimes']}
            )
            artifacts.append(artifact_path)

        # 3. Save clustering metadata
        clustering_metadata = {
            'n_regimes': clustering_result['n_regimes'],
            'n_samples': len(market_data),
            'n_features': len(transformed_features.columns),
            'aic': clustering_result.get('aic'),
            'bic': clustering_result.get('bic'),
            'log_likelihood': clustering_result.get('log_likelihood'),
            'transition_matrix': clustering_result.get('transition_matrix', []).tolist() if isinstance(clustering_result.get('transition_matrix'), np.ndarray) else None,
            'quality_score': quality_metrics.overall_quality if hasattr(quality_metrics, 'overall_quality') else None,
            'silhouette_score': quality_metrics.silhouette_score if hasattr(quality_metrics, 'silhouette_score') else None,
            'timestamp': datetime.now().isoformat()
        }

        artifact_path = self._save_artifact(
            data=clustering_metadata,
            artifact_name='statsmodel_clustering_metadata',
            artifact_type='metadata',
            metadata={'artifact_type': 'clustering_metadata'}
        )
        artifacts.append(artifact_path)

        # 4. Save transformers (for reproducing transformations)
        # We'll pickle the transformers for later use
        import pickle
        transformers_artifact = {
            'scalers': transformers,
            'feature_columns': features.columns.tolist(),
            'transformed_columns': transformed_features.columns.tolist()
        }

        artifact_path = self._save_artifact(
            data=transformers_artifact,
            artifact_name='statsmodel_transformers',
            artifact_type='model',
            metadata={'transformer_count': len(transformers)}
        )
        artifacts.append(artifact_path)

        tprint(f"✅ Saved {len(artifacts)} artifacts", "SUCCESS")

        return artifacts

    def _create_metrics_summary(
        self,
        clustering_result: Dict[str, Any],
        quality_metrics: ClusterQualityMetrics,
        execution_time: float,
        n_features: int,
        n_samples: int
    ) -> Dict[str, Any]:
        """
        Create a summary of metrics for the pipeline execution.

        Args:
            clustering_result: Clustering results
            quality_metrics: Quality metrics
            execution_time: Execution time in seconds
            n_features: Number of features used
            n_samples: Number of samples processed

        Returns:
            Dictionary with metrics summary
        """
        return {
            'execution_time': execution_time,
            'n_regimes': clustering_result['n_regimes'],
            'n_features': n_features,
            'n_samples': n_samples,
            'aic': clustering_result.get('aic'),
            'bic': clustering_result.get('bic'),
            'log_likelihood': clustering_result.get('log_likelihood'),
            'silhouette_score': quality_metrics.silhouette_score if hasattr(quality_metrics, 'silhouette_score') else None,
            'davies_bouldin_score': quality_metrics.davies_bouldin_score if hasattr(quality_metrics, 'davies_bouldin_score') else None,
            'calinski_harabasz_score': quality_metrics.calinski_harabasz_score if hasattr(quality_metrics, 'calinski_harabasz_score') else None,
            'overall_quality': quality_metrics.overall_quality if hasattr(quality_metrics, 'overall_quality') else None,
            'timestamp': datetime.now().isoformat()
        }
