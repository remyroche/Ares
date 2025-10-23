"""
Component Factory for Market Analysis Pipeline Components.

This factory manages the creation and registration of all pipeline components.
"""

import numpy as np
import pandas as pd
import glob
import pickle
import warnings
from typing import Dict, Type, Any, Optional, List
from src.utils.tprint import (tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer)
from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult

# Import ModularComponent for enhanced component creation - REMOVED
MODULAR_COMPONENT_AVAILABLE = False
from .sr_parameter_optimization import SRParameterOptimizationComponent
from .sr_detection import SRDetectionComponent
from .sr_clustering import SRClusteringComponent
# from .hmm_regime_discovery import HMMRegimeDiscoveryComponent  # DEPRECATED
# NAS/TAS components removed - no longer needed for market_analysis
# NAS/TAS clustering components removed
# from ..hmm_clustering.components.clustering_component import OptimalRegimeClusteringComponent  # DEPRECATED
# HMM training components moved to hmm_models_training module
# from .hmm_models_training import HMMModelsTrainingComponent
# from .hmm_ensemble_training_component import HMMEnsembleTrainingComponent  # DEPRECATED
# RegimeDataSplittingComponent imported lazily to avoid circular imports
# TripleBarrierLabelingComponent moved to triple_barrier_labeling package
from .cross_timeframe_analysis import CrossTimeframeAnalysisComponent  # Now uses PID-based feature generation
# Removed unused NAS-TAS components - system uses regime_models_training and regime_ensemble_training instead
# NAS ensemble training component removed
from .regime_models_training import RegimeModelsTrainingComponent
from .regime_ensemble_training import RegimeEnsembleTrainingComponent
# PID-based feature generation moved to pre_training stage

# NAS/TAS clustering components removed

class MultiHorizonComponentWrapper(BaseMarketAnalysisComponent):
    """Wrapper for Multi-Horizon Profit Labeler to work as a component."""

    def __init__(self, adapter_class, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.adapter_class = adapter_class
        self.adapter_instance = None

    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['multi_horizon_labeling_result']

    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute multi-horizon labeling as a component."""
        try:
            # Create adapter instance if not exists
            if self.adapter_instance is None:
                self.adapter_instance = self.adapter_class()

            # Extract configuration from component config
            labeling_config = {}
            if self.config and hasattr(self.config, 'custom_params'):
                labeling_config = self.config.custom_params.get('multi_horizon_labeling', {})

            # Execute multi-horizon labeling with proper execution mode detection
            execution_mode = 'full'  # Default

            # Try multiple sources for execution mode
            if pipeline_state.get('execution_mode'):
                execution_mode = pipeline_state.get('execution_mode')
            elif self.config and hasattr(self.config, 'mode'):
                execution_mode = self.config.mode.value if hasattr(self.config.mode, 'value') else str(self.config.mode)
            elif pipeline_state.get('mode'):
                execution_mode = pipeline_state.get('mode')

            # Force data filtering before calling the adapter
            original_data_size = len(data)
            if execution_mode.lower() == 'light' and original_data_size > 20000:
                data = data.tail(14400).copy()  # 10 days for 1m data
                print(f"🔥 COMPONENT FACTORY LIGHT FILTERING: {original_data_size:,} → {len(data):,} rows")
            elif execution_mode.lower() == 'blank' and original_data_size > 300000:
                data = data.tail(259200).copy()  # 180 days for 1m data
                print(f"🔥 COMPONENT FACTORY BLANK FILTERING: {original_data_size:,} → {len(data):,} rows")

            # Extract regime labels from pipeline state artifacts
            artifacts = pipeline_state.get('artifacts', {})
            regime_clustering_result = artifacts.get('regime_clustering_result', {})
            regime_labels = regime_clustering_result.get('regime_assignments')

            result = self.adapter_instance.execute_multi_horizon_labeling_step(
                data=data,
                regime_labels=regime_labels,
                config=labeling_config,
                symbol=pipeline_state.get('symbol', 'UNKNOWN'),
                exchange=pipeline_state.get('exchange', 'UNKNOWN'),
                timeframe=pipeline_state.get('timeframe', 'UNKNOWN'),
                mode=execution_mode,
                features=pipeline_state.get('pid_based_features')  # Pass optimized features for enhanced labeling
            )

            # Convert to ComponentResult

            # Handle case where result is None
            if result is None:
                return ComponentResult(
                    success=False,
                    artifacts={},
                    metadata={},
                    error_message="Multi-horizon labeling returned None result"
                )

            if result.get('status') == 'completed':
                # Save artifacts persistently using the artifact manager
                try:
                    save_report = await self.save_artifacts(result.get('artifacts', {}), result.get('metadata', {}))
                    print(
                        f"💾 [MULTI_HORIZON] Artifacts saved persistently (correlation_id={save_report.correlation_id}): {list(save_report.paths.keys())}"
                    )
                except Exception as e:
                    print(f"⚠️ [MULTI_HORIZON] Failed to save artifacts persistently: {e}")

                return ComponentResult(
                    success=True,
                    artifacts=result.get('artifacts', {}),
                    metadata={
                        **result.get('metadata', {}),
                        'artifacts_saved_persistently': True
                    },
                    error_message=None
                )
            else:
                return ComponentResult(
                    success=False,
                    artifacts=result.get('artifacts', {}),
                    metadata=result.get('metadata', {}),
                    error_message=result.get('error', 'Unknown error in multi-horizon labeling')
                )

        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                metadata={},
                error_message=f"Multi-horizon labeling component failed: {str(e)}"
            )

class HMMModelsTrainingComponentWrapper(BaseMarketAnalysisComponent):
    """Wrapper for HMM Models Training Enhanced to work as a component."""

    def __init__(self, training_class, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.training_class = training_class
        self.training_instance = None

    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_models_training_result']

    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute HMM models training as a component."""
        try:
            # Create training instance if not exists
            if self.training_instance is None:
                self.training_instance = self.training_class()
                try:
                    # Enforce 15m timeframe for HMM models at runtime
                    if hasattr(self.training_instance, 'config'):
                        setattr(self.training_instance.config, 'timeframe', '15m')
                        if getattr(self.training_instance.config, 'timeframe', None) != '15m':
                            print("⚠️ HMM Models: Non-15m timeframe supplied; overriding to 15m for consistency")
                except Exception:
                    pass

            # Extract required data from pipeline state
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            cluster_assignments = pipeline_state.get('cluster_assignments')
            feature_names = pipeline_state.get('feature_names')
            market_data = pipeline_state.get('market_data') or data  # Use data as fallback if market_data not in pipeline state

            # Validate X and y alignment
            if X is not None and y is not None and len(X) != len(y):
                print(f"❌ X and y length mismatch: X={len(X)}, y={len(y)}")
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=f"X and y length mismatch: X={len(X)}, y={len(y)}"
                )

            # If cluster_assignments is missing, try to get from hmm_clusters
            if cluster_assignments is None:
                hmm_clusters = pipeline_state.get('hmm_clusters', {})
                cluster_assignments = hmm_clusters.get('cluster_assignments')
                if cluster_assignments is not None:
                    print(f"✅ Found cluster_assignments in hmm_clusters: {len(cluster_assignments)} samples")

            # Load cluster assignments directly from HMM training input file
            if cluster_assignments is None:
                try:
                    import glob
                    import pickle

                    # Find the latest HMM training input file
                    hmm_input_pattern = "optimal_clusters/binance/ETHUSDT/15m/market_analysis_hmm_training_input_ETHUSDT_BINANCE_15m_*.pkl"
                    hmm_input_files = glob.glob(hmm_input_pattern)

                    if hmm_input_files:
                        # Get the most recent file
                        latest_file = max(hmm_input_files, key=lambda x: x.split('_')[-1].replace('.pkl', ''))
                        print(f"🔍 Loading cluster assignments from latest HMM training input file: {latest_file}")

                        with open(latest_file, 'rb') as f:
                            hmm_input_data = pickle.load(f)

                        if 'cluster_assignments' in hmm_input_data:
                            cluster_assignments = hmm_input_data['cluster_assignments']
                            print(f"✅ Loaded {len(cluster_assignments)} cluster assignments from HMM training input file")
                            print(f"📊 Cluster assignments shape: {cluster_assignments.shape}, Unique clusters: {len(set(cluster_assignments))}")
                        else:
                            print(f"❌ No cluster_assignments found in HMM training input file")
                            raise ValueError("No cluster_assignments found in HMM training input file")
                    else:
                        print(f"❌ No HMM training input files found matching pattern: {hmm_input_pattern}")
                        raise ValueError("No HMM training input files found")

                except Exception as e:
                    print(f"❌ Error loading cluster assignments from HMM training input file: {e}")
                    raise ValueError(f"Failed to load cluster assignments: {e}")

            # If we don't have features/targets, try to extract from dataframe
            if X is None or y is None:
                dataframe = pipeline_state.get('dataframe')
                if dataframe is not None:
                    import pandas as pd
                    import numpy as np

                    # Create basic features and targets from OHLCV data
                    if 'close' in dataframe.columns:
                        # Create lagged features to avoid data leakage
                        # Shift returns by 1 period to ensure features are from past, target is from future
                        raw_returns = dataframe['close'].pct_change().fillna(0)

                        # Features: lagged returns (past information only)
                        returns_lag1 = raw_returns.shift(1).fillna(0)  # 1-period lagged returns
                        returns_lag2 = raw_returns.shift(2).fillna(0)  # 2-period lagged returns
                        returns_lag5 = raw_returns.shift(5).fillna(0)  # 5-period lagged returns

                        # Volatility features (also lagged)
                        volatility = raw_returns.rolling(20).std().fillna(0).shift(1).fillna(0)

                        # Volume features
                        min_periods_30d = min(len(dataframe), 96)  # At least 1 day of data
                        volume_30d_avg = dataframe['volume'].rolling(window=2880, min_periods=min_periods_30d).mean()
                        volume_ratio_30d = (dataframe['volume'] / volume_30d_avg.replace(0, dataframe['volume'].mean())).fillna(1) if 'volume' in dataframe.columns else pd.Series([1] * len(dataframe), index=dataframe.index)

                        # Additional technical features - more diverse indicators
                        sma_20 = dataframe['close'].rolling(20).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        sma_50 = dataframe['close'].rolling(50).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        price_position = (dataframe['close'] - sma_20) / sma_20.shift(1).fillna(1)

                        # More technical indicators
                        ema_12 = dataframe['close'].ewm(span=12).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        ema_26 = dataframe['close'].ewm(span=26).mean().shift(1).fillna(dataframe['close'].iloc[0])

                        # RSI-like indicator
                        price_changes = raw_returns
                        gains = np.where(price_changes > 0, price_changes, 0)
                        losses = np.where(price_changes < 0, -price_changes, 0)
                        avg_gain = pd.Series(gains).rolling(14).mean().fillna(0).shift(1).fillna(0)
                        avg_loss = pd.Series(losses).rolling(14).mean().fillna(0).shift(1).fillna(0)
                        rs = avg_gain / avg_loss.replace(0, 1e-8)
                        rsi = 100 - (100 / (1 + rs))

                        # Bollinger Bands position
                        bb_middle = sma_20
                        bb_std = raw_returns.rolling(20).std().shift(1).fillna(0)
                        bb_upper = bb_middle + (bb_std * 2)
                        bb_lower = bb_middle - (bb_std * 2)
                        bb_position = (dataframe['close'] - bb_middle) / (bb_upper - bb_lower).replace(0, 1)

                        # Volume-weighted average price (VWAP) components
                        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
                        vwap = (typical_price * dataframe['volume']).rolling(20).sum() / dataframe['volume'].rolling(20).sum()
                        vwap_position = (dataframe['close'] - vwap.shift(1).fillna(dataframe['close'].iloc[0])) / dataframe['close'].shift(1).fillna(dataframe['close'].iloc[0])

                        # Price momentum indicators
                        momentum_5 = dataframe['close'] / dataframe['close'].shift(5).fillna(1) - 1
                        momentum_10 = dataframe['close'] / dataframe['close'].shift(10).fillna(1) - 1

                        # Volatility ratios
                        vol_short = raw_returns.rolling(5).std().shift(1).fillna(0)
                        vol_long = raw_returns.rolling(20).std().shift(1).fillna(0)
                        vol_ratio = vol_short / vol_long.replace(0, 1)

                        X = np.column_stack([
                            returns_lag1.values,    # Lagged returns (past info)
                            returns_lag2.values,    # Lagged returns (past info)
                            returns_lag5.values,    # Lagged returns (past info)
                            volatility.values,      # Historical volatility
                            volume_ratio_30d.values, # Volume ratio
                            sma_20.values,          # Moving average
                            sma_50.values,          # Moving average
                            price_position.values,  # Price position
                            ema_12.values,          # Exponential moving average
                            ema_26.values,          # Exponential moving average
                            rsi.values,             # RSI indicator
                            bb_position.values,     # Bollinger Bands position
                            vwap_position.values,   # VWAP position
                            momentum_5.values,      # Short-term momentum
                            momentum_10.values,     # Medium-term momentum
                            vol_ratio.values        # Volatility ratio
                        ])
                        feature_names = [
                            'returns_lag1', 'returns_lag2', 'returns_lag5',
                            'volatility', 'volume_ratio_30d', 'sma_20', 'sma_50', 'price_position',
                            'ema_12', 'ema_26', 'rsi', 'bb_position', 'vwap_position',
                            'momentum_5', 'momentum_10', 'vol_ratio'
                        ]

                        # Create targets from future returns (not current returns) to avoid data leakage
                        future_returns = raw_returns.shift(-1).fillna(0)  # Next period returns

                        # Convert continuous future returns to discrete classes for predictive modeling
                        # Class 0: Strong Down (< -2%), Class 1: Down (-2% to -0.5%),
                        # Class 2: Sideways (-0.5% to 0.5%), Class 3: Up (0.5% to 2%), Class 4: Strong Up (> 2%)
                        y_continuous = future_returns.values
                        y = np.zeros_like(y_continuous, dtype=int)
                        y[y_continuous < -0.02] = 0  # Strong Down
                        y[(y_continuous >= -0.02) & (y_continuous < -0.005)] = 1  # Down
                        y[(y_continuous >= -0.005) & (y_continuous <= 0.005)] = 2  # Sideways
                        y[(y_continuous > 0.005) & (y_continuous <= 0.02)] = 3  # Up
                        y[y_continuous > 0.02] = 4  # Strong Up

                        # Remove first row where returns is NaN (due to pct_change)
                        X = X[1:]
                        y = y[1:]

                        # Adjust cluster_assignments length if necessary
                        if cluster_assignments is not None and len(cluster_assignments) > len(X):
                            cluster_assignments = cluster_assignments[:len(X)]

            if X is None or y is None or cluster_assignments is None:
              # Detailed error reporting for missing data
              missing_data = []
              if X is None:
                  missing_data.append("features")
              if y is None:
                  missing_data.append("targets")
              if cluster_assignments is None:
                  missing_data.append("cluster_assignments")

              if missing_data:
                  available_keys = list(pipeline_state.keys())
                  error_msg = (
                      f"Missing required data: {', '.join(missing_data)}. "
                      f"Available pipeline state keys: {available_keys}"
                  )
                  raise ValueError(error_msg)

            # Use HMM state recognition as the training objective
            y = cluster_assignments

            # Execute training with comprehensive features
            results = self.training_instance.execute(X, y, cluster_assignments, feature_names, market_data=market_data)

            # Create comprehensive artifact
            artifact = {
                'hmm_models_training_result': {
                    'hmm_models': results.get('model_results', {}),
                    'hmm_training_metrics': results.get('comprehensive_report', {}),
                    'metadata': results.get('metadata', {}),
                    'training_time': results.get('training_time', 0),
                    'success': 'error' not in results
                }
            }

            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={'component_type': 'hmm_models_training', 'execution_time': results.get('training_time', 0)}
            )

        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'hmm_models_training'}
            )

class HMMEnsembleTrainingComponentWrapper(BaseMarketAnalysisComponent):
    """Wrapper for HMM Ensemble Training Component to work as a component."""

    def __init__(self, training_class, config: Optional[ComponentConfig] = None):
        super().__init__(config)
        self.training_class = training_class
        self.training_instance = None

    def _convert_to_numpy_array(self, data):
        """Convert list data to numpy array if needed."""
        if data is not None:
            if isinstance(data, list):
                return np.array(data)
        return data

    def get_required_artifacts(self) -> list[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_ensemble_training_result']

    async def execute(self, data, pipeline_state: Dict[str, Any]) -> 'ComponentResult':
        """Execute HMM ensemble training as a component."""
        try:
            # Create training instance if not exists
            if self.training_instance is None:
                self.training_instance = self.training_class()
                try:
                    # Enforce 15m timeframe for HMM ensemble at runtime
                    if hasattr(self.training_instance, 'config'):
                        setattr(self.training_instance.config, 'timeframe', '15m')
                        if getattr(self.training_instance.config, 'timeframe', None) != '15m':
                            print("⚠️ HMM Ensemble: Non-15m timeframe supplied; overriding to 15m for consistency")
                except Exception:
                    pass

            # Extract required data from pipeline state
            X = pipeline_state.get('features')
            y = pipeline_state.get('targets')
            cluster_assignments = pipeline_state.get('cluster_assignments')
            feature_names = pipeline_state.get('feature_names')
            hmm_states = pipeline_state.get('hmm_states')
            base_hmm_models = pipeline_state.get('hmm_models', {}).get('hmm_models', {})
            hmm_training_metrics = pipeline_state.get('hmm_models', {}).get('hmm_training_metrics', {})

            # If cluster_assignments is missing, try to get from hmm_clusters
            if cluster_assignments is None:
                hmm_clusters = pipeline_state.get('hmm_clusters', {})
                cluster_assignments = hmm_clusters.get('cluster_assignments')
                if cluster_assignments is not None:
                    print(f"✅ Found cluster_assignments in hmm_clusters: {len(cluster_assignments)} samples")

            # Load cluster assignments directly from HMM training input file
            if cluster_assignments is None:
                try:

                    # Find the latest HMM training input file
                    hmm_input_pattern = "optimal_clusters/binance/ETHUSDT/15m/market_analysis_hmm_training_input_ETHUSDT_BINANCE_15m_*.pkl"
                    hmm_input_files = glob.glob(hmm_input_pattern)

                    if hmm_input_files:
                        # Get the most recent file
                        latest_file = max(hmm_input_files, key=lambda x: x.split('_')[-1].replace('.pkl', ''))
                        print(f"🔍 Loading cluster assignments from latest HMM training input file: {latest_file}")

                        with open(latest_file, 'rb') as f:
                            hmm_input_data = pickle.load(f)

                        if 'cluster_assignments' in hmm_input_data:
                            cluster_assignments = hmm_input_data['cluster_assignments']
                            print(f"✅ Loaded {len(cluster_assignments)} cluster assignments from HMM training input file")
                            print(f"📊 Cluster assignments shape: {cluster_assignments.shape}, Unique clusters: {len(set(cluster_assignments))}")
                        else:
                            print(f"❌ No cluster_assignments found in HMM training input file")
                            raise ValueError("No cluster_assignments found in HMM training input file")
                    else:
                        print(f"❌ No HMM training input files found matching pattern: {hmm_input_pattern}")
                        raise ValueError("No HMM training input files found")

                except Exception as e:
                    print(f"❌ Error loading cluster assignments from HMM training input file: {e}")
                    raise ValueError(f"Failed to load cluster assignments: {e}")

            # If we don't have features/targets, try to extract from dataframe
            if X is None or y is None:
                dataframe = pipeline_state.get('dataframe')
                if dataframe is not None:

                    # Create basic features and targets from OHLCV data
                    if 'close' in dataframe.columns:
                        # Create lagged features to avoid data leakage
                        # Shift returns by 1 period to ensure features are from past, target is from future
                        raw_returns = dataframe['close'].pct_change().fillna(0)

                        # Features: lagged returns (past information only)
                        returns_lag1 = raw_returns.shift(1).fillna(0)  # 1-period lagged returns
                        returns_lag2 = raw_returns.shift(2).fillna(0)  # 2-period lagged returns
                        returns_lag5 = raw_returns.shift(5).fillna(0)  # 5-period lagged returns

                        # Volatility features (also lagged)
                        volatility = raw_returns.rolling(20).std().fillna(0).shift(1).fillna(0)

                        # Volume features
                        min_periods_30d = min(len(dataframe), 96)  # At least 1 day of data
                        volume_30d_avg = dataframe['volume'].rolling(window=2880, min_periods=min_periods_30d).mean()
                        volume_ratio_30d = (dataframe['volume'] / volume_30d_avg.replace(0, dataframe['volume'].mean())).fillna(1) if 'volume' in dataframe.columns else pd.Series([1] * len(dataframe), index=dataframe.index)

                        # Additional technical features - more diverse indicators
                        sma_20 = dataframe['close'].rolling(20).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        sma_50 = dataframe['close'].rolling(50).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        price_position = (dataframe['close'] - sma_20) / sma_20.shift(1).fillna(1)

                        # More technical indicators
                        ema_12 = dataframe['close'].ewm(span=12).mean().shift(1).fillna(dataframe['close'].iloc[0])
                        ema_26 = dataframe['close'].ewm(span=26).mean().shift(1).fillna(dataframe['close'].iloc[0])

                        # RSI-like indicator
                        price_changes = raw_returns
                        gains = np.where(price_changes > 0, price_changes, 0)
                        losses = np.where(price_changes < 0, -price_changes, 0)
                        avg_gain = pd.Series(gains).rolling(14).mean().fillna(0).shift(1).fillna(0)
                        avg_loss = pd.Series(losses).rolling(14).mean().fillna(0).shift(1).fillna(0)
                        rs = avg_gain / avg_loss.replace(0, 1e-8)
                        rsi = 100 - (100 / (1 + rs))

                        # Bollinger Bands position
                        bb_middle = sma_20
                        bb_std = raw_returns.rolling(20).std().shift(1).fillna(0)
                        bb_upper = bb_middle + (bb_std * 2)
                        bb_lower = bb_middle - (bb_std * 2)
                        bb_position = (dataframe['close'] - bb_middle) / (bb_upper - bb_lower).replace(0, 1)

                        # Volume-weighted average price (VWAP) components
                        typical_price = (dataframe['high'] + dataframe['low'] + dataframe['close']) / 3
                        vwap = (typical_price * dataframe['volume']).rolling(20).sum() / dataframe['volume'].rolling(20).sum()
                        vwap_position = (dataframe['close'] - vwap.shift(1).fillna(dataframe['close'].iloc[0])) / dataframe['close'].shift(1).fillna(dataframe['close'].iloc[0])

                        # Price momentum indicators
                        momentum_5 = dataframe['close'] / dataframe['close'].shift(5).fillna(1) - 1
                        momentum_10 = dataframe['close'] / dataframe['close'].shift(10).fillna(1) - 1

                        # Volatility ratios
                        vol_short = raw_returns.rolling(5).std().shift(1).fillna(0)
                        vol_long = raw_returns.rolling(20).std().shift(1).fillna(0)
                        vol_ratio = vol_short / vol_long.replace(0, 1)

                        X = np.column_stack([
                            returns_lag1.values,    # Lagged returns (past info)
                            returns_lag2.values,    # Lagged returns (past info)
                            returns_lag5.values,    # Lagged returns (past info)
                            volatility.values,      # Historical volatility
                            volume_ratio_30d.values, # Volume ratio
                            sma_20.values,          # Moving average
                            sma_50.values,          # Moving average
                            price_position.values,  # Price position
                            ema_12.values,          # Exponential moving average
                            ema_26.values,          # Exponential moving average
                            rsi.values,             # RSI indicator
                            bb_position.values,     # Bollinger Bands position
                            vwap_position.values,   # VWAP position
                            momentum_5.values,      # Short-term momentum
                            momentum_10.values,     # Medium-term momentum
                            vol_ratio.values        # Volatility ratio
                        ])
                        feature_names = [
                            'returns_lag1', 'returns_lag2', 'returns_lag5',
                            'volatility', 'volume_ratio_30d', 'sma_20', 'sma_50', 'price_position',
                            'ema_12', 'ema_26', 'rsi', 'bb_position', 'vwap_position',
                            'momentum_5', 'momentum_10', 'vol_ratio'
                        ]

                        # Create targets from future returns (not current returns) to avoid data leakage
                        future_returns = raw_returns.shift(-1).fillna(0)  # Next period returns

                        # Convert continuous future returns to discrete classes for predictive modeling
                        # Class 0: Strong Down (< -2%), Class 1: Down (-2% to -0.5%),
                        # Class 2: Sideways (-0.5% to 0.5%), Class 3: Up (0.5% to 2%), Class 4: Strong Up (> 2%)
                        y_continuous = future_returns.values
                        y = np.zeros_like(y_continuous, dtype=int)
                        y[y_continuous < -0.02] = 0  # Strong Down
                        y[(y_continuous >= -0.02) & (y_continuous < -0.005)] = 1  # Down
                        y[(y_continuous >= -0.005) & (y_continuous <= 0.005)] = 2  # Sideways
                        y[(y_continuous > 0.005) & (y_continuous <= 0.02)] = 3  # Up
                        y[y_continuous > 0.02] = 4  # Strong Up

                        # Remove first row where returns is NaN (due to pct_change)
                        X = X[1:]
                        y = y[1:]

                        # Adjust cluster_assignments length if necessary
                        if cluster_assignments is not None and len(cluster_assignments) > len(X):
                            cluster_assignments = cluster_assignments[:len(X)]

            if X is None or y is None or cluster_assignments is None:
                missing_items = []
                if X is None: missing_items.append("features")
                if y is None: missing_items.append("targets")
                if cluster_assignments is None: missing_items.append("cluster_assignments")
                raise ValueError(f"Missing required data: {', '.join(missing_items)}")

            # Ensure all data is in proper numpy format before training
            cluster_assignments = self._convert_to_numpy_array(cluster_assignments)

            # Use HMM state recognition as the training objective
            y = cluster_assignments

            # Execute training
            results = self.training_instance.execute(
                X, y, cluster_assignments, feature_names, hmm_states,
                base_hmm_models, hmm_training_metrics
            )

            # Create comprehensive artifact
            artifact = {
                'hmm_ensemble_training_result': {
                    'hmm_ensemble': results.get('models', {}),
                    'hmm_ensemble_metrics': results.get('comprehensive_report', {}),
                    'ensemble_metrics': results.get('ensemble_metrics', {}),
                    'performance_summary': results.get('performance_summary', {}),
                    'metadata': results.get('metadata', {}),
                    'training_time': results.get('training_time', 0),
                    'success': 'error' not in results
                }
            }

            return ComponentResult(
                success=True,
                artifacts=artifact,
                metadata={'component_type': 'hmm_ensemble_training', 'execution_time': results.get('training_time', 0)}
            )

        except Exception as e:
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={'component_type': 'hmm_ensemble_training'}
            )

class ComponentFactory:
    """
    Factory for creating market analysis pipeline components.

    Provides centralized component creation and management.
    """

    _components: Dict[str, Type[BaseMarketAnalysisComponent]] = {
        'sr_parameter_optimization': SRParameterOptimizationComponent,
        'sr_detection': SRDetectionComponent,
        'sr_clustering': SRClusteringComponent,
        # NAS/TAS components removed - no longer needed for market_analysis
        'regime_models_training': RegimeModelsTrainingComponent,  # Regime detection models training
        'regime_ensemble_training': RegimeEnsembleTrainingComponent,  # Regime detection ensemble training
        # 'hmm_models_training': HMMModelsTrainingComponent,  # Moved to hmm_models_training module
        # 'hmm_ensemble_training': HMMEnsembleTrainingComponent,  # Removed
        # 'regime_data_splitting': RegimeDataSplittingComponent,  # Imported lazily to avoid circular imports
        # 'triple_barrier_labeling': TripleBarrierLabelingComponent,  # Moved to triple_barrier_labeling package
        'cross_timeframe_analysis': CrossTimeframeAnalysisComponent,  # Now uses PID-based feature generation
        # Feature engineering components moved to pre_training stage:
        # 'feature_lookback_optimization', 'pid_based_feature_generation', 'final_feature_selection'
    }

    @classmethod
    def create_component(
        self,
        component_name: str,
        config: Optional[ComponentConfig] = None
    ) -> BaseMarketAnalysisComponent:
        """
        Create a component instance.

        Args:
            component_name: Name of the component to create
            config: Component configuration

        Returns:
            Component instance

        Raises:
            ValueError: If component name is not registered
        """
        tprint(f"🏭 [COMPONENT_FACTORY] Creating component: {component_name}", color="cyan")
        # Handle lazy imports for components that might cause circular imports
        if component_name == 'regime_data_splitting':
            try:
                tprint("🔧 [COMPONENT_FACTORY] Loading RegimeDataSplittingComponent", color="yellow")
                from .regime_data_splitting import RegimeDataSplittingComponent
                component = RegimeDataSplittingComponent(config)
                tprint(f"✅ [COMPONENT_FACTORY] Created RegimeDataSplittingComponent", color="green")
                return component
            except ImportError as e:
                raise ValueError(f"Failed to import RegimeDataSplittingComponent: {e}")

        # Multi-horizon profit labeler moved to pre_training stage
        # Handle HMM training components (DEPRECATED - removed)
        if component_name == 'hmm_models_training':
            tprint("⚠️ [COMPONENT_FACTORY] HMM models training is deprecated", color="yellow")
            raise ValueError("HMM models training is deprecated and no longer available")

        if component_name == 'hmm_ensemble_training':
            tprint("⚠️ [COMPONENT_FACTORY] HMM ensemble training is deprecated", color="yellow")
            raise ValueError("HMM ensemble training is deprecated and no longer available")

        if component_name not in self._components:
            available_components = list(self._components.keys()) + ['regime_data_splitting']
            tprint(f"❌ [COMPONENT_FACTORY] Unknown component: {component_name}", color="red")
            tprint(f"📊 [COMPONENT_FACTORY] Available components: {available_components}", color="cyan")
            raise ValueError(
                f"Unknown component: {component_name}. "
                f"Available components: {available_components}"
            )

        tprint(f"🔧 [COMPONENT_FACTORY] Creating {component_name} from registered components", color="yellow")
        component_class = self._components[component_name]

        # Handle None component classes
        if component_class is None:
            tprint(f"❌ [COMPONENT_FACTORY] Component {component_name} is not available", color="red")
            raise ValueError(f"Component {component_name} is not available. Required dependencies may be missing.")

        # Create component instance
        try:
            component = component_class(config)
            tprint(f"✅ [COMPONENT_FACTORY] Created {component_name}", color="green")
            return component
        except Exception as e:
            tprint(f"❌ [COMPONENT_FACTORY] Failed to create {component_name}: {e}", color="red")
            raise

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

except ImportError:

    cp = None

    @classmethod
    def register_component(
        self,
        name: str,
        component_class: Type[BaseMarketAnalysisComponent]
    ) -> None:
        """
        Register a new component.

        Args:
            name: Component name
            component_class: Component class
        """
        if not issubclass(component_class, BaseMarketAnalysisComponent):
            raise ValueError(
                f"Component class must inherit from BaseMarketAnalysisComponent"
            )

        self._components[name] = component_class

    def create_modular_component(
        self,
        component_name: str,
        config: Optional[ComponentConfig] = None
    ) -> BaseMarketAnalysisComponent:
        """
        Create a ModularComponent instance with enhanced features.

        Args:
            component_name: Name of the component to create
            config: Component configuration

        Returns:
            ModularComponent instance

        Raises:
            ValueError: If component name is not registered
        """
        if not MODULAR_COMPONENT_AVAILABLE:
            # Fallback to regular component creation
            return self.create_component(component_name, config)
        
        tprint(f"🏭 [COMPONENT_FACTORY] Creating ModularComponent: {component_name}", color="cyan")
        
        # Get the component class
        if component_name not in self._components:
            available_components = list(self._components.keys()) + ['regime_data_splitting']
            tprint(f"❌ [COMPONENT_FACTORY] Unknown component: {component_name}", color="red")
            tprint(f"📊 [COMPONENT_FACTORY] Available components: {available_components}", color="cyan")
            raise ValueError(f"Unknown component: {component_name}")
        
        component_class = self._components[component_name]
        
        # Check if component is already a ModularComponent
        if issubclass(component_class, ModularComponent):
            return component_class(
                name=component_name,
                config=config.to_dict() if hasattr(config, 'to_dict') else config.__dict__,
                logger=config.logger if hasattr(config, 'logger') else None
            )
        else:
            # For non-ModularComponent classes, create regular instance
            return component_class(config)

    @classmethod
    def get_available_components(self) -> list[str]:
        """
        Get list of available component names.

        Returns:
            List of component names
        """
        # Include both registered components and lazy-loaded components
        lazy_components = ['regime_data_splitting']
        return list(self._components.keys()) + lazy_components

    @classmethod
    def is_component_available(self, component_name: str) -> bool:
        """
        Check if a component is available.

        Args:
            component_name: Name of the component

        Returns:
            True if component is available
        """
        # Check both registered components and lazy-loaded components
        lazy_components = ['regime_data_splitting']
        return component_name in self._components or component_name in lazy_components

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and
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

    def _vectorbt_apply_operation(self, data: pd.Series, func,
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)

        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
