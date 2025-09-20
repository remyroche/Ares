"""
HMM Regime Discovery Component.

This component discovers market regimes using Hidden Markov Models.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import time

from .base_component import BaseMarketAnalysisComponent, ComponentConfig, ComponentResult
from src.utils.logger import system_logger


class HMMRegimeDiscoveryComponent(BaseMarketAnalysisComponent):
    """
    HMM Regime Discovery Component.
    
    Discovers market regimes using Hidden Markov Models.
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the HMM regime discovery component."""
        super().__init__(config)
        self.logger = system_logger.getChild('HMMRegimeDiscovery')
        self._resources_to_cleanup = []
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with resource cleanup."""
        self._cleanup_resources()
        
    def _cleanup_resources(self):
        """Clean up any allocated resources."""
        try:
            for resource in self._resources_to_cleanup:
                if hasattr(resource, 'cleanup'):
                    resource.cleanup()
                elif hasattr(resource, 'close'):
                    resource.close()
            self._resources_to_cleanup.clear()
        except Exception as e:
            self.logger.warning(f"Error during resource cleanup: {e}")
    
    def __del__(self):
        """Destructor with resource cleanup."""
        self._cleanup_resources()
    
    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['hmm_regime_discovery_result']
    
    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute HMM regime discovery.
        
        Args:
            data: Market data for regime discovery
            pipeline_state: Current pipeline state
            
        Returns:
            ComponentResult with regime discovery results
        """
        self.logger.info('🔍 Starting HMM Regime Discovery')
        
        try:
            # Import HMM configuration only (no longer need EnhancedHMMRegimeDetector)
            from src.utils.ml_common.hmm_regime_detection import HMMRegimeConfig, RegimeDetectionMethod
            
            # Resolve symbol from config or pipeline state
            symbol = getattr(self.config, 'symbol', None)
            if symbol is None and 'symbol' in pipeline_state:
                symbol = pipeline_state['symbol']
            if symbol is None:
                raise ValueError("Symbol must be provided in config or pipeline state")
                
            # Resolve timeframe from config or pipeline state
            timeframe = getattr(self.config, 'timeframe', None)
            if timeframe is None and 'timeframe' in pipeline_state:
                timeframe = pipeline_state['timeframe']
            if timeframe is None:
                timeframe = '1h'  # Default timeframe for regime discovery

            # Get market data
            market_data = await self._load_market_data(data, symbol)
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for regime discovery for symbol: {symbol}")
            
            # Configure HMM regime detection with data-driven component count
            data_size = len(market_data) if market_data is not None else 1000
            
            # Calculate optimal component count based on data characteristics
            optimal_components = self._calculate_optimal_component_count(data_size, pipeline_state)
            
            self.logger.info(f"📊 Calculated optimal components: {optimal_components} (based on {data_size} data points)")
            
            hmm_config = HMMRegimeConfig(
                n_components=optimal_components,  # Data-driven component count
                method=RegimeDetectionMethod.ENHANCED_HMM,
                min_regime_samples=max(20, data_size // (optimal_components * 3)),  # More conservative minimum
                max_regime_imbalance=0.9,  # Allow more imbalance for natural regimes
                economic_significance_threshold=0.01,  # Lower threshold for regime significance
                
                # Add reasonable safety bounds to prevent runaway regime creation
                light_mode_max_regimes=min(50, optimal_components),  # Reasonable upper bound
                blank_mode_max_regimes=min(100, optimal_components),  # Reasonable upper bound
                full_mode_max_regimes=min(200, optimal_components)    # Reasonable upper bound
            )
            
            # Direct regime discovery without external detector (simplified)
            self.logger.info('🔧 Using direct HMM regime discovery with fixed parameters')
            
            # Perform regime discovery directly
            discovery_start_time = time.time()
            regime_dataframe, optimization_results = await self._perform_direct_regime_discovery(
                market_data, hmm_config
            )
            discovery_time = time.time() - discovery_start_time
            
            # Validate regime discovery results
            if regime_dataframe is None:
                raise ValueError("HMM regime discovery failed: returned None")
            if regime_dataframe.empty:
                raise ValueError("HMM regime discovery completed but no regimes were discovered (empty DataFrame)")
            if 'regime' not in regime_dataframe.columns:
                raise ValueError("HMM regime discovery result missing 'regime' column")

            # Extract and validate regime assignments
            regime_assignments = regime_dataframe['regime'].tolist()
            
            # Validate assignments before conversion
            validated_assignments = []
            for i, assignment in enumerate(regime_assignments):
                try:
                    # Check if assignment can be converted to int and is non-negative
                    int_assignment = int(assignment)
                    if int_assignment < 0:
                        raise ValueError(f"Negative regime assignment at index {i}: {int_assignment}")
                    validated_assignments.append(int_assignment)
                except (ValueError, TypeError) as e:
                    raise ValueError(f"Invalid regime assignment at index {i}: '{assignment}' - {e}")
            
            regime_assignments = validated_assignments

            # Calculate regime metrics with validation and 3D decomposition
            unique_regimes = set(regime_assignments)
            if not unique_regimes:
                raise ValueError("No unique regimes found in assignments")
            
            # Create descriptive 3D regime model names
            regime_models = []
            momentum_states = optimization_results.get('best_params', {}).get('momentum_states', 6)
            volatility_states = optimization_results.get('best_params', {}).get('volatility_states', 5)
            volume_states = optimization_results.get('best_params', {}).get('volume_states', 5)
            
            for regime_id in sorted(unique_regimes):
                # Decompose composite regime ID back to dimensional states
                momentum_state = regime_id % momentum_states
                volatility_state = (regime_id // momentum_states) % volatility_states
                volume_state = regime_id // (momentum_states * volatility_states)
                
                # Create descriptive name: "regime_M2_V1_Vol3" (Momentum=2, Volatility=1, Volume=3)
                regime_name = f"regime_M{momentum_state}_V{volatility_state}_Vol{volume_state}"
                regime_models.append(regime_name)
            regime_metrics = {
                'total_regimes': len(unique_regimes),
                'total_samples': len(regime_dataframe),
                'regime_distribution': regime_dataframe['regime'].value_counts().to_dict()
            }
            
            # Validate minimum samples per regime
            min_samples = min(regime_metrics['regime_distribution'].values())
            if min_samples < hmm_config.min_regime_samples:
                self.logger.warning(f"Some regimes have fewer than {hmm_config.min_regime_samples} samples (min: {min_samples})")
            
            # Create single consolidated artifact
            regime_distribution = self._calculate_regime_distribution(regime_assignments)
            
            # Create regime characteristics for clustering
            regime_characteristics = self._create_regime_characteristics_for_clustering(
                regime_dataframe, regime_assignments, market_data
            )
            
            # Get timeframe from config
            timeframe = getattr(self.config, 'timeframe', "1h")

            artifacts = {
                'hmm_regime_discovery_result': {
                    # Core regime data (backward compatible)
                    'regime_models': regime_models,
                    'regime_assignments': regime_assignments,
                    'regime_count': len(regime_models),
                    'total_samples': len(regime_assignments),
                    'regime_distribution': regime_distribution,
                    'regime_characteristics': regime_characteristics,
                    
                    # Enhanced 3D regime information
                    '3d_regime_info': {
                        'dimensional_structure': {
                            'momentum_states': optimization_results.get('best_params', {}).get('momentum_states', 6),
                            'volatility_states': optimization_results.get('best_params', {}).get('volatility_states', 5),
                            'volume_states': optimization_results.get('best_params', {}).get('volume_states', 5),
                            'max_lookback_hours': 6
                        },
                        'dimensional_assignments': {
                            'momentum_assignments': optimization_results.get('dimensional_assignments', {}).get('momentum_assignments', []),
                            'volatility_assignments': optimization_results.get('dimensional_assignments', {}).get('volatility_assignments', []),  
                            'volume_assignments': optimization_results.get('dimensional_assignments', {}).get('volume_assignments', [])
                        },
                        'regime_decomposition': {
                            regime_name: {
                                'composite_id': regime_id,
                                'momentum_state': regime_id % optimization_results.get('best_params', {}).get('momentum_states', 6),
                                'volatility_state': (regime_id // optimization_results.get('best_params', {}).get('momentum_states', 6)) % optimization_results.get('best_params', {}).get('volatility_states', 5),
                                'volume_state': regime_id // (optimization_results.get('best_params', {}).get('momentum_states', 6) * optimization_results.get('best_params', {}).get('volatility_states', 5))
                            } for regime_id, regime_name in zip(sorted(unique_regimes), regime_models)
                        },
                        'composite_mapping': 'regime_id = momentum + volatility*M + volume*M*V',
                        'decomposition_formula': {
                            'momentum_state': 'regime_id % momentum_states',
                            'volatility_state': '(regime_id // momentum_states) % volatility_states',
                            'volume_state': 'regime_id // (momentum_states * volatility_states)'
                        },
                        'interpretation': {
                            'regime_type': '3d_dimensional_composite',
                            'can_decompose': True,
                            'dimensional_models_available': True,
                            'regime_name_format': 'regime_M{momentum}_V{volatility}_Vol{volume}'
                        }
                    },
                    
                    'regime_metrics': regime_metrics,
                    'validation_metrics': {
                        'min_samples_per_regime': min_samples,
                        'sufficient_samples': min_samples >= hmm_config.min_regime_samples
                    },
                    'configuration': {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'optimization_mode': '3d_dimensional_hmm',
                        'min_regime_samples': hmm_config.min_regime_samples,
                        'economic_significance_threshold': hmm_config.economic_significance_threshold
                    },
                    'optimization_results': optimization_results if optimization_results else {},
                    'execution_info': {
                        'timestamp': datetime.now().isoformat(),
                        'data_points_processed': len(market_data),
                        'success': True,
                        'discovery_time': discovery_time
                    }
                }
            }
            
            self.logger.info(f'✅ HMM Regime Discovery completed: {len(regime_models)} regimes discovered (3D regime space: up to 10×10×10 states)')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'data_points_processed': len(market_data),
                    'regime_count': len(regime_models),
                    'optimization_mode': 'fast_fixed_parameters',
                    'min_samples_per_regime': min_samples,
                    'execution_successful': True
                }
            )
            
        except (ValueError, TypeError) as e:
            self.logger.error(f'❌ HMM Regime Discovery failed with data/parameter error: {e}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Data or parameter error: {str(e)}"
            )
        except ImportError as e:
            self.logger.error(f'❌ HMM Regime Discovery failed with missing dependency: {e}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Missing required dependency: {str(e)}"
            )
        except RuntimeError as e:
            self.logger.error(f'❌ HMM Regime Discovery failed with runtime error: {e}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Runtime error: {str(e)}"
            )
        except Exception as e:
            self.logger.error(f'❌ HMM Regime Discovery failed with unexpected error: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=f"Unexpected error: {str(e)}"
            )
    
    async def _load_market_data(self, data: Any, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime discovery using klines_parquet manager."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.logger.warning("⚠️ No market data provided, attempting to load from klines_parquet")

                # Validate symbol parameter
                if symbol is None:
                    raise ValueError("Symbol parameter is required for market data loading")

                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager
                
                manager = get_klines_manager()
                
                # Use provided symbol and configured timeframe for HMM regime discovery
                timeframe = getattr(self.config, 'timeframe', "1h")  # Default to 1h if not configured
                
                self.logger.info(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")
                
                # Get date filtering from config if available
                start_date = None
                end_date = None
                if hasattr(self.config, 'start_date') and self.config.start_date:
                    from datetime import datetime
                    start_date = datetime.strptime(self.config.start_date, '%Y-%m-%d')
                    self.logger.info(f"📅 Using start_date filter: {start_date}")
                if hasattr(self.config, 'end_date') and self.config.end_date:
                    from datetime import datetime
                    end_date = datetime.strptime(self.config.end_date, '%Y-%m-%d')
                    self.logger.info(f"📅 Using end_date filter: {end_date}")
                
                # Try processed data first (better for HMM analysis)
                market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="processed")
                
                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    self.logger.info(f"📊 No processed data found, trying raw {symbol} {timeframe} data")
                    market_data = manager.read_data(symbol, timeframe, start_date=start_date, end_date=end_date, data_type="raw")
                
                if market_data is None or market_data.empty:
                    self.logger.error(f"❌ No data available for {symbol} {timeframe}")
                    return None
                
                self.logger.info(f"✅ Loaded {len(market_data)} rows of {symbol} {timeframe} data")
                return market_data
            
            # If data is already a DataFrame, use it
            if isinstance(data, pd.DataFrame):
                self.logger.info(f"📊 Using provided DataFrame with {len(data)} rows")
                return data.copy()
            
            # Handle other data types if needed
            return None
            
        except Exception as e:
            self.logger.exception(f"❌ Error loading market data: {e}")
            return None
    
    async def _perform_direct_regime_discovery(
        self,
        market_data: pd.DataFrame,
        hmm_config: Any
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
        """Perform direct HMM regime discovery without external detector."""
        try:
            # Prepare data for regime detection
            prepared_data = self._prepare_data_for_regime_detection(market_data)
            
            # Perform regime discovery directly (no external detector needed)
            loop = asyncio.get_event_loop()
            regime_dataframe, optimization_results = await loop.run_in_executor(
                None,
                lambda: self._train_hmm_directly(prepared_data, hmm_config)
            )
            
            return regime_dataframe, optimization_results
            
        except (ValueError, TypeError) as e:
            # Data or parameter related errors
            raise ValueError(f"Direct regime discovery failed due to data/parameter issue: {e}") from e
        except ImportError as e:
            # Missing dependencies
            raise ImportError(f"Direct regime discovery failed due to missing dependency: {e}") from e
        except Exception as e:
            # Other unexpected errors
            raise RuntimeError(f"Direct regime discovery failed with unexpected error: {e}") from e
    
    def _prepare_data_for_regime_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare market data for regime detection with comprehensive validation."""
        # Validate input data
        if data is None:
            raise ValueError("Data cannot be None for regime detection preparation")
        if data.empty:
            raise ValueError("Data cannot be empty for regime detection preparation")

        # Work on a copy to avoid mutating the caller's DataFrame
        data_copy = data.copy()

        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data_copy.columns]

        if missing_columns:
            self.logger.warning(f"Missing columns for regime detection: {missing_columns}")
            for col in missing_columns:
                if col == 'volume':
                    if 'close' in data_copy.columns:
                        price_std = data_copy['close'].std()
                        estimated_volume = max(1000, int(price_std * 100))
                        data_copy[col] = estimated_volume
                        self.logger.info(f"Estimated volume using price volatility: {estimated_volume}")
                    else:
                        data_copy[col] = 1000
                        self.logger.warning("Using minimal volume fallback: 1000")
                elif col == 'open':
                    if 'close' in data_copy.columns:
                        data_copy[col] = data_copy['close']
                        self.logger.info("Using close price as open price fallback")
                    else:
                        raise ValueError("Cannot create open price fallback without close price")
                elif col == 'high':
                    if 'close' in data_copy.columns and 'open' in data_copy.columns:
                        data_copy[col] = data_copy[['open', 'close']].max(axis=1) * 1.001
                        self.logger.info("Created high price using open/close maximum")
                    elif 'close' in data_copy.columns:
                        data_copy[col] = data_copy['close'] * 1.001
                        self.logger.info("Created high price using close price")
                    else:
                        raise ValueError("Cannot create high price fallback without price data")
                elif col == 'low':
                    if 'close' in data_copy.columns and 'open' in data_copy.columns:
                        data_copy[col] = data_copy[['open', 'close']].min(axis=1) * 0.999
                        self.logger.info("Created low price using open/close minimum")
                    elif 'close' in data_copy.columns:
                        data_copy[col] = data_copy['close'] * 0.999
                        self.logger.info("Created low price using close price")
                    else:
                        raise ValueError("Cannot create low price fallback without price data")

        # Validate OHLC relationships and data quality
        self._validate_ohlc_relationships(data_copy)
        self._validate_data_quality(data_copy)

        return data_copy
    
    def _validate_ohlc_relationships(self, data: pd.DataFrame) -> None:
        """Validate OHLC price relationships."""
        try:
            invalid_high_low = (data['high'] < data['low']).sum()
            if invalid_high_low > 0:
                self.logger.warning(f"Found {invalid_high_low} rows where high < low")
            invalid_high_open = (data['high'] < data['open']).sum()
            invalid_high_close = (data['high'] < data['close']).sum()
            if invalid_high_open > 0:
                self.logger.warning(f"Found {invalid_high_open} rows where high < open")
            if invalid_high_close > 0:
                self.logger.warning(f"Found {invalid_high_close} rows where high < close")
            invalid_low_open = (data['low'] > data['open']).sum()
            invalid_low_close = (data['low'] > data['close']).sum()
            if invalid_low_open > 0:
                self.logger.warning(f"Found {invalid_low_open} rows where low > open")
            if invalid_low_close > 0:
                self.logger.warning(f"Found {invalid_low_close} rows where low > close")
        except Exception as e:
            self.logger.warning(f"Error validating OHLC relationships: {e}")

    def _validate_data_quality(self, data: pd.DataFrame) -> None:
        """Validate data quality for regime detection."""
        try:
            nan_counts = data.isnull().sum()
            if nan_counts.sum() > 0:
                self.logger.warning(f"Found NaN values: {nan_counts.to_dict()}")
            inf_counts = np.isinf(data.select_dtypes(include=[np.number])).sum()
            if inf_counts.sum() > 0:
                self.logger.warning(f"Found infinite values: {inf_counts.to_dict()}")
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    negative_count = (data[col] <= 0).sum()
                    if negative_count > 0:
                        self.logger.warning(f"Found {negative_count} non-positive values in {col}")
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    self.logger.warning(f"Found {negative_volume} negative volume values")
            if len(data) < 100:
                self.logger.warning(f"Data length ({len(data)}) may be insufficient for reliable regime detection")
        except Exception as e:
            self.logger.warning(f"Error validating data quality: {e}")

    def _calculate_regime_distribution(self, regime_assignments: List[int]) -> Dict[str, float]:
        """Calculate the distribution of regime assignments."""
        if not regime_assignments:
            return {}
        
        total_assignments = len(regime_assignments)
        regime_counts = {}
        
        for assignment in regime_assignments:
            regime_counts[assignment] = regime_counts.get(assignment, 0) + 1
        
        # Convert to percentages
        regime_distribution = {}
        for regime, count in regime_counts.items():
            key = f'regime_{regime}'
            regime_distribution[key] = (count / total_assignments) * 100
        
        return regime_distribution

    def _calculate_dimensional_states(self, data_size: int) -> Tuple[int, int, int]:
        """
        Calculate optimal states per dimension based on data size with parameter constraints.
        
        Args:
            data_size: Number of data points
            
        Returns:
            tuple: (momentum_states, volatility_states, volume_states)
        """
        try:
            # Calculate theoretical maximum for each dimension using parameter estimation theory
            momentum_states, volatility_states, volume_states = self._calculate_constrained_dimensional_states(data_size)
            
            self.logger.info(f"📊 Dimensional states calculated: M={momentum_states}, V={volatility_states}, Vol={volume_states}")
            self.logger.info(f"📊 Total combinations: {momentum_states * volatility_states * volume_states}")
            
            return momentum_states, volatility_states, volume_states
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating dimensional states: {e}")
            # Fallback to moderate states
            return 5, 4, 5

    def _calculate_constrained_dimensional_states(self, data_size: int) -> Tuple[int, int, int]:
        """
        Calculate dimensional states with parameter constraints from _calculate_optimal_component_count.
        
        Args:
            data_size: Number of data points
            
        Returns:
            tuple: (momentum_states, volatility_states, volume_states) with applied constraints
        """
        try:
            # Feature counts for parameter estimation (updated for behavioral features)
            momentum_features_count = 3  # momentum_strength, trend_persistence, reversal_probability
            volatility_features_count = 3  # realized_volatility, volatility_regime, price_efficiency
            volume_features_count = 3  # volume_percentile_rank, volume_ma_ratio, volume_acceleration
            
            def calculate_hmm_parameters(k_components: int, n_features: int) -> int:
                """Calculate total parameters for diagonal covariance HMM."""
                # Correct parameter count for diagonal covariance HMM (O(n) scaling):
                # - Transition probabilities: k*(k-1) (each row sums to 1)
                # - Start probabilities: k-1 (sum to 1)  
                # - Means: k * n_features
                # - Diagonal covariances: k * n_features (much better than full covariance O(n²))
                transition_params = k_components * (k_components - 1)
                start_params = k_components - 1
                mean_params = k_components * n_features
                covar_params = k_components * n_features  # O(n) instead of O(n²) for full
                return transition_params + start_params + mean_params + covar_params
            
            def max_dimensional_states_for_data(data_size: int, samples_per_param: int = 15) -> Tuple[int, int, int]:
                """Calculate maximum dimensional states based on parameter estimation theory."""
                # Use adaptive samples per parameter based on data size
                if data_size < 200:
                    samples_per_param = max(5, data_size // 20)  # More lenient for small datasets
                elif data_size < 1000:
                    samples_per_param = max(8, data_size // 30)
                else:
                    samples_per_param = 15  # Standard for large datasets
                
                # Calculate max states for each dimension separately
                max_momentum = 1
                max_volatility = 1  
                max_volume = 1
                
                # Find max momentum states
                for states in range(1, 10):
                    params = calculate_hmm_parameters(states, momentum_features_count)
                    if params * samples_per_param <= data_size:
                        max_momentum = states
                    else:
                        break
                
                # Find max volatility states  
                for states in range(1, 8):
                    params = calculate_hmm_parameters(states, volatility_features_count)
                    if params * samples_per_param <= data_size:
                        max_volatility = states
                    else:
                        break
                
                # Find max volume states
                for states in range(1, 8):
                    params = calculate_hmm_parameters(states, volume_features_count)
                    if params * samples_per_param <= data_size:
                        max_volume = states
                    else:
                        break
                
                return max_momentum, max_volatility, max_volume
            
            # Calculate theoretical maximum for each dimension
            max_mom, max_vol, max_volume = max_dimensional_states_for_data(data_size)
            
            # Apply the same constraint logic as in _calculate_optimal_component_count
            # With diagonal covariance (O(n) scaling), we can support more components for 20-regime goal
            if data_size < 500:
                desired_momentum, desired_volatility, desired_volume = 7, 7, 7  # 343 combinations
            elif data_size < 2000:
                desired_momentum, desired_volatility, desired_volume = 8, 8, 8  # 512 combinations
            elif data_size < 5000:
                desired_momentum, desired_volatility, desired_volume = 9, 8, 9  # 648 combinations
            elif data_size < 15000:
                desired_momentum, desired_volatility, desired_volume = 9, 9, 9  # 729 combinations
            elif data_size < 30000:
                desired_momentum, desired_volatility, desired_volume = 10, 9, 10  # 900 combinations
            else:
                desired_momentum, desired_volatility, desired_volume = 10, 10, 10  # 1000 combinations
            
            # Apply parameter constraints
            momentum_states = min(desired_momentum, max_mom)
            volatility_states = min(desired_volatility, max_vol)
            volume_states = min(desired_volume, max_volume)
            
            # Ensure minimum viable states
            momentum_states = max(3, momentum_states)
            volatility_states = max(3, volatility_states)
            volume_states = max(3, volume_states)
            
            self.logger.info(f"📊 Constrained dimensional states:")
            self.logger.info(f"   • Desired: M={desired_momentum}, V={desired_volatility}, Vol={desired_volume}")
            self.logger.info(f"   • Max allowed: M={max_mom}, V={max_vol}, Vol={max_volume}")
            self.logger.info(f"   • Final: M={momentum_states}, V={volatility_states}, Vol={volume_states}")
            
            return momentum_states, volatility_states, volume_states
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating constrained dimensional states: {e}")
            # Fallback to safe defaults
            return 5, 4, 5

    def _calculate_optimal_component_count(self, data_size: int, pipeline_state: Dict[str, Any]) -> int:
        """
        Calculate optimal number of HMM components based on data characteristics.
        
        Args:
            data_size: Number of data points
            pipeline_state: Current pipeline state (may contain user preferences)
            
        Returns:
            int: Optimal number of components
        """
        try:
            # Check for user-specified component count in config or pipeline state
            user_components = getattr(self.config, 'n_components', None)
            if user_components is None:
                user_components = pipeline_state.get('n_components', None)
            
            if user_components is not None:
                self.logger.info(f"🎯 Using user-specified component count: {user_components}")
                return max(3, min(user_components, 20))  # Clamp to reasonable range for 3D regime space
            
            # Data-driven calculation
            # Rule of thumb: sqrt(data_size) with bounds based on statistical reliability
            base_components = int(np.sqrt(data_size))
            
            # Use the shared helper that applies parameter constraints
            momentum_states, volatility_states, volume_states = self._calculate_constrained_dimensional_states(data_size)
            
            # Total regime combinations
            optimal_components = momentum_states * volatility_states * volume_states
            
            self.logger.info(f"📊 3D Regime Space Configuration:")
            self.logger.info(f"   • Momentum states: {momentum_states}")
            self.logger.info(f"   • Volatility states: {volatility_states}")  
            self.logger.info(f"   • Volume states: {volume_states}")
            self.logger.info(f"   • Total combinations: {optimal_components} ({momentum_states}×{volatility_states}×{volume_states})")
            
            # Constraint logic is now handled by _calculate_constrained_dimensional_states
            # Additional validation: ensure minimum samples per regime
            if data_size < 200:
                regime_based_max = max(9, data_size // 10)  # More lenient for small datasets
            elif data_size < 1000:
                regime_based_max = max(15, data_size // 15)
            else:
                regime_based_max = data_size // 20  # Standard for large datasets
            
            if optimal_components > regime_based_max:
                optimal_components = max(5, regime_based_max)
                self.logger.warning(f"⚠️ Reduced components to {optimal_components} due to regime-based constraints")
                self.logger.warning(f"   Regime-based max: {regime_based_max}")
            
            # Final bounds check with support for 20-regime goal (15-25 components)
            optimal_components = max(15, min(optimal_components, 1000))  # At least 15 for 20-regime goal, max 1000 (10x10x10)
            
            self.logger.info(f"✅ Calculated optimal components: {optimal_components} (base: {base_components}, data_size: {data_size})")
            return optimal_components
            
        except Exception as e:
            self.logger.error(f"❌ Error calculating optimal components: {e}")
            # Fallback to reasonable default
            fallback_components = 25
            self.logger.info(f"🔄 Using fallback component count: {fallback_components}")
            return fallback_components

    def _train_hmm_directly(self, market_data: pd.DataFrame, hmm_config: Any) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Perform direct HMM regime discovery with fixed parameters.
        
        Args:
            market_data: Market data DataFrame
            hmm_config: HMM configuration
            
        Returns:
            tuple: (regime_dataframe, optimization_results)
        """
        try:
            self.logger.info(f"🔍 Starting direct HMM regime discovery with {len(market_data)} data points...")
            
            # Prepare grouped features for 3D regime detection
            momentum_features, volatility_features, volume_features = self._prepare_grouped_features_for_regime_discovery(market_data)
            
            if momentum_features is None or volatility_features is None or volume_features is None:
                raise ValueError("Failed to prepare grouped features for regime discovery")
            
            # Get dimensional state counts with parameter constraints
            data_size = len(market_data)
            momentum_states, volatility_states, volume_states = self._calculate_constrained_dimensional_states(data_size)
            n_components = momentum_states * volatility_states * volume_states
            
            self.logger.info(f"🎯 Using {n_components} components for fast regime discovery (intermediate features)")
            self.logger.info("⚡ Skipping heavy Bayesian optimization - regimes are intermediate features for clustering")
            self.logger.info("📊 Final optimization happens in clustering stage using clustering quality metrics (silhouette_score, calinski_harabasz)")
            self.logger.info("🎯 Targeting 15-25 components for effective 20-regime behavioral clustering with diagonal covariance O(n) scaling")
            
            # Train separate HMM models for each dimension
            self.logger.info(f"📊 Training separate HMM models:")
            self.logger.info(f"   • Momentum: {momentum_features.shape[1]} features → {momentum_states} states")
            self.logger.info(f"   • Volatility: {volatility_features.shape[1]} features → {volatility_states} states")
            self.logger.info(f"   • Volume: {volume_features.shape[1]} features → {volume_states} states")
            
            momentum_model, momentum_assignments = self._train_dimensional_hmm(
                momentum_features, momentum_states, "momentum"
            )
            volatility_model, volatility_assignments = self._train_dimensional_hmm(
                volatility_features, volatility_states, "volatility"  
            )
            volume_model, volume_assignments = self._train_dimensional_hmm(
                volume_features, volume_states, "volume"
            )
            
            # Combine dimensional assignments into composite regime states
            regime_assignments = self._combine_dimensional_assignments(
                momentum_assignments, volatility_assignments, volume_assignments,
                momentum_states, volatility_states, volume_states
            )
            
            # Enhanced optimization results with 3D information
            optimization_results = {
                'best_params': {
                    'n_components': n_components,
                    'momentum_states': momentum_states,
                    'volatility_states': volatility_states,
                    'volume_states': volume_states,
                    'covariance_type': 'diag',
                    'n_iter': 100,
                    'tol': 1e-3
                },
                'dimensional_assignments': {
                    'momentum_assignments': momentum_assignments,
                    'volatility_assignments': volatility_assignments,
                    'volume_assignments': volume_assignments
                },
                'best_score': 0.0,
                'method': '3d_dimensional_hmm_regime_discovery',
                'optimization_time': 0.0,
                'note': 'True 3D regime space with separate dimensional HMMs'
            }
            
            # Calculate actual feature offset based on max rolling window used
            # Get the maximum lookback period from all feature calculations
            momentum_lookback = 2  # reversal_probability uses 2-period lookback
            volatility_lookback = 24  # volatility_regime uses 24-period lookback
            volume_lookback = 48  # volume_percentile_rank uses 48-period lookback
            max_lookback = max(momentum_lookback, volatility_lookback, volume_lookback)
            
            # Preserve original index and align properly
            if len(market_data) <= max_lookback:
                raise ValueError(f"Insufficient data: need at least {max_lookback + 1} rows, got {len(market_data)}")
            
            # Align data preserving original timestamps
            original_index = market_data.index
            market_data_aligned = market_data.iloc[max_lookback:].copy()
            
            # Validate feature lengths match expected alignment
            expected_length = len(market_data) - max_lookback
            if len(momentum_features) != expected_length:
                self.logger.warning(f"Feature length mismatch: expected {expected_length}, got {len(momentum_features)}")
            
            min_length = min(len(momentum_features), len(volatility_features), len(volume_features), len(market_data_aligned))
            
            # Ensure all arrays are the same length and preserve index alignment
            market_data_aligned = market_data_aligned.iloc[:min_length]
            regime_assignments_aligned = regime_assignments[:min_length]
            momentum_assignments_aligned = momentum_assignments[:min_length]
            volatility_assignments_aligned = volatility_assignments[:min_length]
            volume_assignments_aligned = volume_assignments[:min_length]
            
            # Create regime dataframe with preserved index
            regime_dataframe = market_data_aligned.copy()
            regime_dataframe['regime'] = regime_assignments_aligned
            
            # Add dimensional state information
            regime_dataframe['momentum_state'] = momentum_assignments_aligned
            regime_dataframe['volatility_state'] = volatility_assignments_aligned
            regime_dataframe['volume_state'] = volume_assignments_aligned
            
            unique_regimes = len(set(regime_assignments))
            total_possible = momentum_states * volatility_states * volume_states
            self.logger.info(f"✅ 3D Regime discovery completed: {unique_regimes} unique regime combinations found")
            self.logger.info(f"   Total possible combinations: {total_possible} ({momentum_states}×{volatility_states}×{volume_states})")
            
            return regime_dataframe, optimization_results
            
        except Exception as e:
            error_msg = f"Direct HMM regime discovery failed: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

    def _prepare_grouped_features_for_regime_discovery(self, market_data) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Prepare features grouped by dimension for 3D regime discovery.
        
        This method implements the simplified 3D dimensional approach:
        - Momentum dimension: 1h, 2h price changes (pure reactivity)
        - Volatility dimension: intrabar volatility, 6h rolling std, 6h ATR
        - Volume dimension: 3h volume momentum, 24h volume ratio baseline
        
        Features are optimized for regime detection without unnecessary complexity.
        Features are designed to be compatible with HMM clustering expectations.
        
        Args:
            market_data: Market data DataFrame with OHLCV columns
            
        Returns:
            tuple: (momentum_features, volatility_features, volume_features)
                  Each DataFrame contains clean, finite features ready for HMM training
        """
        try:
            import pandas as pd
            import numpy as np
            
            # Momentum features (behavioral instead of temporal)
            momentum_features = pd.DataFrame(index=market_data.index)
            # Behavioral momentum indicators
            momentum_features['momentum_strength'] = market_data['close'].pct_change(1).abs()  # Strength of movement
            momentum_features['trend_persistence'] = (market_data['close'].pct_change(1) * market_data['close'].pct_change(2)).sign()  # Trend continuation
            momentum_features['reversal_probability'] = -market_data['close'].pct_change(1) * market_data['close'].pct_change(2)  # Reversal signal  
            
            # Volatility features (behavioral instead of temporal)
            volatility_features = pd.DataFrame(index=market_data.index)
            # Behavioral volatility indicators
            volatility_features['realized_volatility'] = market_data['close'].pct_change(1).rolling(6).std()  # Realized volatility
            volatility_features['volatility_regime'] = (market_data['close'].pct_change(1).rolling(6).std() > market_data['close'].pct_change(1).rolling(24).std()).astype(int)  # High vs low vol regime
            volatility_features['price_efficiency'] = (market_data['close'].pct_change(1).abs() / ((market_data['high'] - market_data['low']) / market_data['close'] + 1e-8))  # Price efficiency
            
            # Volume features (behavioral instead of temporal)
            epsilon = 1e-8
            volume_features = pd.DataFrame(index=market_data.index)
            # Behavioral volume indicators
            volume_features['volume_percentile_rank'] = market_data['volume'].rolling(48).rank(pct=True)  # Volume percentile rank
            volume_features['volume_ma_ratio'] = market_data['volume'] / (market_data['volume'].rolling(48).mean() + epsilon)  # Volume vs moving average
            volume_features['volume_acceleration'] = market_data['volume'].pct_change(1)  # Volume momentum/acceleration
            
            # Clean features and align indices
            # Use the same max_lookback calculation as in _train_hmm_directly
            momentum_lookback = 2  # reversal_probability uses 2-period lookback
            volatility_lookback = 24  # volatility_regime uses 24-period lookback
            volume_lookback = 48  # volume_percentile_rank uses 48-period lookback
            max_lookback = max(momentum_lookback, volatility_lookback, volume_lookback)
            cleaned_features = []
            
            for i, (features, feature_type) in enumerate(zip([momentum_features, volatility_features, volume_features], 
                                                          ['momentum', 'volatility', 'volume'])):
                # Skip rows based on max lookback
                features = features.iloc[max_lookback:]
                
                # Apply consistent NaN handling based on feature type
                features = self._handle_feature_nans(features, feature_type)
                
                # Apply feature standardization for improved HMM sensitivity
                features = self._standardize_features(features, feature_type)
                
                # Ensure finite values and proper dtype
                features = features.astype(np.float64)
                cleaned_features.append(features)
            
            momentum_clean, volatility_clean, volume_clean = cleaned_features
            
            self.logger.info(f"📊 Prepared grouped features:")
            self.logger.info(f"   • Momentum: {momentum_clean.shape[1]} features")
            self.logger.info(f"   • Volatility: {volatility_clean.shape[1]} features")
            self.logger.info(f"   • Volume: {volume_clean.shape[1]} features")
            
            return momentum_clean, volatility_clean, volume_clean
            
        except Exception as e:
            self.logger.error(f"Failed to prepare grouped features: {e}")
            return None, None, None

    def _handle_feature_nans(self, features: pd.DataFrame, feature_type: str) -> pd.DataFrame:
        """
        Apply consistent NaN handling based on feature type.
        
        Args:
            features: Feature DataFrame
            feature_type: Type of features ('momentum', 'volatility', 'volume')
            
        Returns:
            DataFrame with NaN values handled consistently
        """
        try:
            # Replace infinite values with NaN first
            features = features.replace([np.inf, -np.inf], np.nan)
            
            for col in features.columns:
                if features[col].isnull().any():
                    if feature_type == 'momentum':
                        # For momentum features, use 0 (no change)
                        features[col] = features[col].fillna(0.0)
                    elif feature_type == 'volatility':
                        # For volatility features, use forward fill then median, fallback to small positive value
                        features[col] = features[col].fillna(method='ffill')
                        median_val = features[col].median()
                        if pd.isna(median_val) or median_val <= 0:
                            median_val = 0.01  # Small positive fallback for volatility
                        features[col] = features[col].fillna(median_val)
                    elif feature_type == 'volume':
                        if 'ratio' in col.lower():
                            # Volume ratios default to 1.0 (neutral)
                            features[col] = features[col].fillna(1.0)
                        elif 'weighted' in col.lower():
                            # Volume-weighted features use forward fill then median
                            features[col] = features[col].fillna(method='ffill')
                            median_val = features[col].median()
                            features[col] = features[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                        else:
                            # Other volume features (momentum) use 0
                            features[col] = features[col].fillna(0.0)
                    else:
                        # Default: use median or 0
                        median_val = features[col].median()
                        features[col] = features[col].fillna(median_val if not pd.isna(median_val) else 0.0)
                
                # Clip extreme values to prevent numerical issues
                if len(features[col].dropna()) > 0:
                    q99 = features[col].quantile(0.99)
                    q01 = features[col].quantile(0.01)
                    if not pd.isna(q99) and not pd.isna(q01):
                        features[col] = features[col].clip(lower=q01, upper=q99)
                
                # Final safety check - ensure no infinite or NaN values remain
                features[col] = features[col].replace([np.inf, -np.inf], 0.0)
                features[col] = features[col].fillna(0.0)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error in NaN handling for {feature_type} features: {e}")
            # Fallback: fill all NaN with 0
            return features.fillna(0.0).replace([np.inf, -np.inf], 0.0)

    def _standardize_features(self, features: pd.DataFrame, feature_type: str) -> pd.DataFrame:
        """
        Standardize features to improve HMM sensitivity by ensuring comparable scales.
        
        Args:
            features: Features DataFrame
            feature_type: Type of features (momentum, volatility, volume)
            
        Returns:
            Standardized features DataFrame
        """
        try:
            from sklearn.preprocessing import RobustScaler
            import numpy as np
            
            # Use RobustScaler to handle outliers better than StandardScaler
            scaler = RobustScaler()
            
            # Only standardize if we have enough valid data
            if len(features.dropna()) < 10:
                self.logger.warning(f"⚠️ Insufficient data for {feature_type} standardization, skipping")
                return features
            
            # Fit and transform features
            features_scaled = features.copy()
            features_scaled.iloc[:, :] = scaler.fit_transform(features.values)
            
            self.logger.info(f"✅ {feature_type.capitalize()} features standardized using RobustScaler")
            return features_scaled
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to standardize {feature_type} features: {e}, using original features")
            return features


    def _train_dimensional_hmm(self, features: pd.DataFrame, n_states: int, dimension_name: str) -> Tuple[Any, List[int]]:
        """
        Train HMM for a specific dimension.
        
        Args:
            features: Features for this dimension
            n_states: Number of states for this dimension
            dimension_name: Name of dimension for logging
            
        Returns:
            tuple: (trained_model, state_assignments)
        """
        try:
            from hmmlearn import hmm
            import numpy as np
            import pandas as pd
            
            self.logger.info(f"🏃 Training {dimension_name} HMM: {features.shape[1]} features → {n_states} states")
            
            # Clean features thoroughly before HMM training
            features_clean = features.copy()
            
            # Check for NaN values and report
            nan_counts = features_clean.isnull().sum()
            if nan_counts.sum() > 0:
                self.logger.warning(f"⚠️ {dimension_name} features contain NaN values: {nan_counts.to_dict()}")
            
            # Replace infinite values with NaN first
            features_clean = features_clean.replace([np.inf, -np.inf], np.nan)
            
            # Fill NaN values with appropriate defaults based on feature type
            for col in features_clean.columns:
                if features_clean[col].isnull().any():
                    if 'momentum' in col.lower():
                        # For momentum features, use 0 (no change)
                        features_clean[col] = features_clean[col].fillna(0.0)
                    elif 'volatility' in col.lower() or 'std' in col.lower():
                        # For volatility features, use median but preserve zero values when meaningful
                        median_val = features_clean[col].median()
                        if pd.isna(median_val):
                            # Only use 0.01 fallback if no valid data at all
                            median_val = 0.01
                        elif median_val == 0:
                            # Check if zeros are meaningful (e.g., no price movement)
                            non_zero_count = (features_clean[col] != 0).sum()
                            if non_zero_count > 0:
                                # Use median of non-zero values
                                median_val = features_clean[col][features_clean[col] != 0].median()
                                if pd.isna(median_val):
                                    median_val = 0.01
                        features_clean[col] = features_clean[col].fillna(median_val)
                    elif 'volume' in col.lower():
                        # For volume features, use median or 0
                        median_val = features_clean[col].median()
                        fill_val = median_val if not pd.isna(median_val) else 0.0
                        features_clean[col] = features_clean[col].fillna(fill_val)
                    else:
                        # Default: use 0
                        features_clean[col] = features_clean[col].fillna(0.0)
            
            # Ensure all values are finite and numeric
            features_clean = features_clean.astype(np.float64)
            
            # Final check for any remaining NaN or infinite values
            if not np.all(np.isfinite(features_clean.values)):
                self.logger.warning(f"⚠️ {dimension_name} features still contain non-finite values, applying final cleanup")
                features_clean = features_clean.replace([np.inf, -np.inf, np.nan], 0.0)
            
            # Ensure we have enough data
            if len(features_clean) < max(10, n_states * 2):
                raise ValueError(f"Insufficient data for {dimension_name} HMM: {len(features_clean)} samples for {n_states} states")
            
            # Convert to numpy array for HMM training
            X = features_clean.values.astype(np.float64)
            
            # Create HMM model with proper initialization - FIXED VERSION
            model = hmm.GaussianHMM(
                n_components=n_states,
                covariance_type='diag',
                n_iter=150,
                tol=1e-4,
                random_state=42,
                init_params='stmc'  # Let hmmlearn handle initialization properly
            )
            
            # Train the model with proper error handling
            try:
                model.fit(X)
                
                # Validate model after training
                self._validate_and_fix_transition_matrix(model, n_states)
                
            except (ValueError, np.linalg.LinAlgError) as fit_error:
                self.logger.warning(f"⚠️ {dimension_name} HMM fit failed with numerical error: {fit_error}")
                # Retry with more conservative settings
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type='diag',
                    n_iter=50,  # Reduced iterations
                    tol=1e-3,   # Relaxed tolerance
                    random_state=42,
                    init_params='stmc'
                )
                model.fit(X)
                self._validate_and_fix_transition_matrix(model, n_states)
            except Exception as fit_error:
                self.logger.error(f"❌ {dimension_name} HMM fit failed with unexpected error: {fit_error}")
                raise ValueError(f"HMM training failed for {dimension_name}: {fit_error}") from fit_error
            
            # Get state assignments
            assignments = model.predict(X)
            
            unique_states = len(set(assignments))
            self.logger.info(f"✅ {dimension_name} HMM trained: {unique_states} unique states found")
            
            return model, assignments.tolist()
            
        except Exception as e:
            self.logger.error(f"❌ Failed to train {dimension_name} HMM: {e}")
            self.logger.error(f"   Features shape: {features.shape}")
            self.logger.error(f"   Features dtypes: {features.dtypes.to_dict()}")
            if hasattr(features, 'isnull'):
                self.logger.error(f"   NaN count: {features.isnull().sum().sum()}")
            
            # Return fallback assignments
            fallback_assignments = [0] * len(features)
            return None, fallback_assignments

    def _combine_dimensional_assignments(self, momentum_assignments: List[int], volatility_assignments: List[int], 
                                       volume_assignments: List[int], momentum_states: int, 
                                       volatility_states: int, volume_states: int) -> List[int]:
        """
        Combine dimensional state assignments into composite regime IDs.
        
        Args:
            momentum_assignments: Momentum state assignments
            volatility_assignments: Volatility state assignments  
            volume_assignments: Volume state assignments
            momentum_states: Number of momentum states
            volatility_states: Number of volatility states
            volume_states: Number of volume states
            
        Returns:
            List of composite regime IDs
        """
        try:
            composite_assignments = []
            
            for i in range(len(momentum_assignments)):
                # Create composite regime ID: momentum + volatility*M + volume*M*V
                regime_id = (momentum_assignments[i] + 
                           volatility_assignments[i] * momentum_states +
                           volume_assignments[i] * momentum_states * volatility_states)
                composite_assignments.append(regime_id)
            
            unique_regimes = len(set(composite_assignments))
            total_possible = momentum_states * volatility_states * volume_states
            
            self.logger.info(f"📊 Combined dimensional assignments:")
            self.logger.info(f"   • Unique regimes found: {unique_regimes}")
            self.logger.info(f"   • Total possible: {total_possible}")
            self.logger.info(f"   • Coverage: {unique_regimes/total_possible*100:.1f}%")
            
            return composite_assignments
            
        except Exception as e:
            self.logger.error(f"❌ Failed to combine dimensional assignments: {e}")
            # Return fallback
            return [0] * len(momentum_assignments)

    # Removed legacy feature preparation method _prepare_features_for_regime_discovery (unused)
    
    def _validate_and_fix_transition_matrix(self, model, n_components: int) -> None:
        """
        Validate and fix transition matrix after training to prevent zero-sum rows.
        
        Args:
            model: Trained HMM model
            n_components: Number of components in the model
        """
        try:
            # Check if transition matrix has any zero-sum rows
            row_sums = model.transmat_.sum(axis=1)
            zero_sum_rows = np.where(np.abs(row_sums) < 1e-10)[0]
            
            if len(zero_sum_rows) > 0:
                self.logger.warning(f"⚠️ Found {len(zero_sum_rows)} zero-sum rows in transition matrix: {zero_sum_rows}")
                
                # Fix zero-sum rows by setting them to uniform distribution
                epsilon = 1e-6
                for row_idx in zero_sum_rows:
                    # Set uniform transition probabilities with slight bias towards self-transition
                    uniform_prob = (1.0 - 0.7) / (n_components - 1) if n_components > 1 else 1.0
                    model.transmat_[row_idx, :] = uniform_prob
                    if n_components > 1:
                        model.transmat_[row_idx, row_idx] = 0.7  # Higher self-transition probability
                
                # Renormalize all rows to ensure they sum to 1
                model.transmat_ = model.transmat_ / model.transmat_.sum(axis=1, keepdims=True)
                
                self.logger.info(f"✅ Fixed {len(zero_sum_rows)} zero-sum rows in transition matrix")
            
            # Additional validation: ensure no NaN or infinite values
            if np.any(np.isnan(model.transmat_)) or np.any(np.isinf(model.transmat_)):
                self.logger.warning("⚠️ Found NaN or infinite values in transition matrix, applying regularization")
                
                # Replace NaN/inf with regularized uniform distribution
                epsilon = 1e-6
                regularized_transmat = np.full((n_components, n_components), epsilon)
                np.fill_diagonal(regularized_transmat, 0.7)
                
                # Distribute remaining probability
                for i in range(n_components):
                    remaining_prob = 1.0 - regularized_transmat[i, i]
                    other_states_prob = remaining_prob / (n_components - 1) if n_components > 1 else 0.0
                    for j in range(n_components):
                        if i != j:
                            regularized_transmat[i, j] = other_states_prob
                
                # Normalize and assign
                regularized_transmat = regularized_transmat / regularized_transmat.sum(axis=1, keepdims=True)
                model.transmat_ = regularized_transmat.astype(np.float64)
                
                self.logger.info("✅ Applied regularization to fix NaN/infinite values in transition matrix")
            
            # Final validation
            final_row_sums = model.transmat_.sum(axis=1)
            if not np.allclose(final_row_sums, 1.0, atol=1e-6):
                self.logger.warning(f"⚠️ Transition matrix rows do not sum to 1: {final_row_sums}")
                # Force normalization
                model.transmat_ = model.transmat_ / model.transmat_.sum(axis=1, keepdims=True)
                self.logger.info("✅ Forced normalization of transition matrix")
            
            self.logger.debug(f"📊 Transition matrix validation completed - all rows sum to 1")
            
        except (ValueError, np.linalg.LinAlgError) as e:
            self.logger.error(f"❌ Error validating transition matrix: {e}")
            # Fallback: create a safe uniform transition matrix
            epsilon = 1e-6
            safe_transmat = np.full((n_components, n_components), 1.0 / n_components)
            model.transmat_ = safe_transmat.astype(np.float64)
            self.logger.info("✅ Applied fallback uniform transition matrix")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error validating transition matrix: {e}")
            # Emergency fallback: create a safe uniform transition matrix
            epsilon = 1e-6
            safe_transmat = np.full((n_components, n_components), 1.0 / n_components)
            model.transmat_ = safe_transmat.astype(np.float64)
            self.logger.info("✅ Applied emergency fallback uniform transition matrix")
    
    def _create_regime_characteristics_for_clustering(self, regime_dataframe: pd.DataFrame, regime_assignments: List[int], 
                                                    market_data: pd.DataFrame) -> Dict[str, Any]:
        """Create regime characteristics in the format expected by HMM clustering."""
        try:
            import pandas as pd
            import numpy as np
            
            regime_characteristics = {}
            unique_regimes = set(regime_assignments)
            
            # Recreate features for characteristics calculation
            momentum_features, volatility_features, volume_features = self._prepare_grouped_features_for_regime_discovery(market_data)
            
            if momentum_features is None or volatility_features is None or volume_features is None:
                self.logger.warning("⚠️ Failed to recreate features for regime characteristics")
                return {}
            
            # Combine all features for comprehensive characteristics
            all_features = pd.concat([momentum_features, volatility_features, volume_features], axis=1)
            
            for regime_id in unique_regimes:
                # Get indices for this regime
                regime_indices = [i for i, r in enumerate(regime_assignments) if r == regime_id]
                
                if len(regime_indices) > 0:
                    # Get regime features
                    regime_features = all_features.iloc[regime_indices]
                    
                    # Calculate means and stds
                    feature_means = regime_features.mean().fillna(0.0).to_dict()
                    # Use median for std fillna instead of constant 0.01
                    feature_stds_series = regime_features.std()
                    median_std = feature_stds_series.median()
                    if pd.isna(median_std) or median_std == 0:
                        median_std = 0.01
                    feature_stds = feature_stds_series.fillna(median_std).to_dict()
                    
                    # Debug: Log feature creation for first regime
                    if regime_id == list(unique_regimes)[0]:
                        self.logger.info(f"🔍 DEBUG Discovery: regime_features shape: {regime_features.shape}")
                        self.logger.info(f"🔍 DEBUG Discovery: regime_features columns: {list(regime_features.columns)}")
                        self.logger.info(f"🔍 DEBUG Discovery: feature_means keys: {list(feature_means.keys())}")
                        self.logger.info(f"🔍 DEBUG Discovery: sample feature_means values: {dict(list(feature_means.items())[:3])}")
                    
                    # Create characteristics in the format expected by clustering
                    regime_characteristics[f'regime_{regime_id}'] = {
                        'features': feature_means,  # Use 'features' key for clustering compatibility
                        'feature_means': feature_means,
                        'feature_stds': feature_stds,
                        'volatility': feature_stds.get(list(feature_stds.keys())[0], 0.01) if feature_stds else 0.01,
                        'sample_count': len(regime_indices)
                    }
            
            self.logger.info(f"✅ Created regime characteristics for {len(regime_characteristics)} regimes")
            return regime_characteristics
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create regime characteristics for clustering: {e}")
            return {}