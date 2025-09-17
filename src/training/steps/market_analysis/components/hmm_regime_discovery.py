""""""
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
            # Import HMM regime detection utilities
            from src.utils.ml_common.hmm_regime_detection import EnhancedHMMRegimeDetector, HMMRegimeConfig, RegimeDetectionMethod
            
            # Resolve symbol from config or pipeline state
            symbol = getattr(self.config, 'symbol', None)
            if symbol is None and 'symbol' in pipeline_state:
                symbol = pipeline_state['symbol']
            if symbol is None:
                raise ValueError("Symbol must be provided in config or pipeline state")

            # Get market data
            market_data = await self._load_market_data(data, symbol)
            if market_data is None or market_data.empty:
                raise ValueError(f"No market data available for regime discovery for symbol: {symbol}")
            
            # Configure HMM regime detection
            hmm_config = HMMRegimeConfig(
                n_components=20,  # Will be limited by mode
                method=RegimeDetectionMethod.ENHANCED_HMM,
                min_regime_samples=100,  # Minimum samples per regime
                max_regime_imbalance=0.8,
                economic_significance_threshold=0.05,
                
                # Mode-based regime limits
                light_mode_max_regimes=2,
                blank_mode_max_regimes=5,
                full_mode_max_regimes=150
            )
            
            # Create HMM regime detector
            regime_detector = EnhancedHMMRegimeDetector()
            
            # Determine optimization mode from config with validation
            optimization_mode = getattr(self.config, 'optimization_mode', 'blank')
            valid_modes = ['light', 'blank', 'full']
            if optimization_mode not in valid_modes:
                self.logger.warning(f"Invalid optimization mode '{optimization_mode}', defaulting to 'blank'")
                optimization_mode = 'blank'

            # Determine max regimes allowed for mode
            try:
                max_regimes = hmm_config.get_max_regimes_for_mode(optimization_mode)
            except Exception:
                max_regimes = {
                    'light': getattr(hmm_config, 'light_mode_max_regimes', 2),
                    'blank': getattr(hmm_config, 'blank_mode_max_regimes', 5),
                    'full': getattr(hmm_config, 'full_mode_max_regimes', 150),
                }.get(optimization_mode, 5)
            self.logger.info(f'🔧 HMM Regime Discovery mode: {optimization_mode} (max regimes: {max_regimes})')
            
            # Perform regime discovery
            discovery_start_time = time.time()
            regime_dataframe = await self._perform_regime_discovery(
                regime_detector, market_data, hmm_config, optimization_mode
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
            try:
                regime_assignments = [int(assignment) for assignment in regime_assignments]
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid regime assignments: {e}")
            if any(assignment < 0 for assignment in regime_assignments):
                raise ValueError("Regime assignments contain negative values")

            # Calculate regime metrics with validation
            unique_regimes = set(regime_assignments)
            if not unique_regimes:
                raise ValueError("No unique regimes found in assignments")
            regime_models = [f"regime_{i}" for i in sorted(unique_regimes)]
            regime_metrics = {
                'total_regimes': len(unique_regimes),
                'total_samples': len(regime_dataframe),
                'regime_distribution': regime_dataframe['regime'].value_counts().to_dict()
            }

            # Validate regime count against mode limits
            if len(unique_regimes) > max_regimes:
                self.logger.warning(f"Discovered {len(unique_regimes)} regimes, exceeding mode limit of {max_regimes}")
            
            # Validate minimum samples per regime
            min_samples = min(regime_metrics['regime_distribution'].values())
            if min_samples < hmm_config.min_regime_samples:
                self.logger.warning(f"Some regimes have fewer than {hmm_config.min_regime_samples} samples (min: {min_samples})")
            
            # Create single consolidated artifact
            regime_distribution = self._calculate_regime_distribution(regime_assignments)

            artifacts = {
                'hmm_regime_discovery_result': {
                    # Core regime data
                    'regime_models': regime_models,
                    'regime_assignments': regime_assignments,
                    'regime_count': len(regime_models),
                    'total_samples': len(regime_assignments),
                    'regime_distribution': regime_distribution,
                    'validation_metrics': {
                        'min_samples_per_regime': min_samples,
                        'max_regimes_allowed': max_regimes,
                        'regime_count_vs_limit': len(regime_models) <= max_regimes,
                        'sufficient_samples': min_samples >= hmm_config.min_regime_samples
                    },
                    'configuration': {
                        'symbol': symbol,
                        'timeframe': '1h',
                        'optimization_mode': optimization_mode,
                        'max_regimes_for_mode': max_regimes,
                        'min_regime_samples': hmm_config.min_regime_samples,
                        'economic_significance_threshold': hmm_config.economic_significance_threshold
                    },
                    'execution_info': {
                        'timestamp': datetime.now().isoformat(),
                        'data_points_processed': len(market_data),
                        'success': True,
                        'discovery_time': discovery_time
                    }
                }
            }
            
            self.logger.info(f'✅ HMM Regime Discovery completed: {len(regime_models)} regimes discovered (range: 2-150)')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': symbol,
                    'timeframe': '1h',
                    'data_points_processed': len(market_data),
                    'regime_count': len(regime_models),
                    'optimization_mode': optimization_mode,
                    'max_regimes_for_mode': max_regimes,
                    'min_samples_per_regime': min_samples,
                    'execution_successful': True
                }
            )
            
        except Exception as e:
            self.logger.error(f'❌ HMM Regime Discovery failed: {e}')
            import traceback
            self.logger.error(f'❌ Error details: {traceback.format_exc()}')
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e)
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
                
                # Use provided symbol and hardcoded timeframe for HMM regime discovery
                timeframe = "1h"    # Hardcoded timeframe for HMM regime discovery
                
                self.logger.info(f"📊 Loading {symbol} {timeframe} data using klines_parquet manager")
                
                # Try processed data first (better for HMM analysis)
                market_data = manager.read_data(symbol, timeframe, data_type="processed")
                
                if market_data is None or market_data.empty:
                    # Fallback to raw data
                    self.logger.info(f"📊 No processed data found, trying raw {symbol} {timeframe} data")
                    market_data = manager.read_data(symbol, timeframe, data_type="raw")
                
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
    
    async def _perform_regime_discovery(
        self, 
        regime_detector: Any, 
        market_data: pd.DataFrame, 
        config: Any,
        mode: str = 'blank'
    ) -> Optional[pd.DataFrame]:
        """Perform the actual regime discovery process."""
        try:
            # Prepare data for regime detection
            prepared_data = self._prepare_data_for_regime_detection(market_data)
            
            # Perform regime detection with mode using a background thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            regime_dataframe = await loop.run_in_executor(
                None,
                lambda: regime_detector.detect_regimes(prepared_data, config=config, mode=mode)
            )
            return regime_dataframe
            
        except Exception as e:
            # Propagate the error with context for higher-level handling
            raise RuntimeError(f"Regime discovery process failed: {e}")
    
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