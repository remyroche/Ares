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
            
            # Get market data
            market_data = await self._load_market_data(data)
            if market_data is None or market_data.empty:
                raise ValueError("No market data available for regime discovery")
            
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
            
            # Determine optimization mode from config or default to 'blank'
            optimization_mode = getattr(self.config, 'optimization_mode', 'blank')
            self.logger.info(f'🔧 HMM Regime Discovery mode: {optimization_mode} (range: 2-150 regimes)')
            
            # Perform regime discovery
            discovery_start_time = time.time()
            regime_dataframe = await self._perform_regime_discovery(
                regime_detector, market_data, hmm_config, optimization_mode
            )
            discovery_time = time.time() - discovery_start_time
            
            # Extract results from DataFrame
            if regime_dataframe is None or regime_dataframe.empty:
                raise ValueError("HMM regime discovery completed but no regimes were discovered")
            
            # Extract regime information from the DataFrame
            regime_assignments = regime_dataframe['regime'].tolist() if 'regime' in regime_dataframe.columns else []
            unique_labels = sorted(set(regime_assignments)) if regime_assignments else []
            label_name_mapping = {label: f"regime_{i}" for i, label in enumerate(unique_labels)}
            regime_models = [label_name_mapping[label] for label in unique_labels]
            raw_distribution = regime_dataframe['regime'].value_counts().to_dict() if 'regime' in regime_dataframe.columns else {}
            named_distribution = {label_name_mapping.get(lbl, str(lbl)): count for lbl, count in raw_distribution.items()}
            regime_metrics = {
                'total_regimes': len(unique_labels),
                'total_samples': len(regime_dataframe),
                'regime_distribution': named_distribution
            }
            
            # Validate that we have regime models and assignments
            if not regime_models or not regime_assignments:
                raise ValueError("HMM regime discovery completed but no regimes were discovered")
            
            # Create single consolidated artifact
            artifacts = {
                'hmm_regime_discovery_result': {
                    'regime_models': regime_models,
                    'regime_assignments': regime_assignments,
                    'regime_metrics': regime_metrics,
                    'regime_discovery_summary': {
                        'total_regimes': len(regime_models),
                        'total_assignments': len(regime_assignments),
                        'regime_distribution': self._calculate_regime_distribution(regime_assignments, label_name_mapping),
                        'discovery_time': discovery_time
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'data_points': len(market_data) if market_data is not None else 0,
                        'execution_timestamp': datetime.now().isoformat(),
                        'label_name_mapping': label_name_mapping,
                        'regime_limits': {
                            'light_mode_max': 2,
                            'blank_mode_max': 5,
                            'full_mode_max': 150,
                            'current_mode': optimization_mode
                        }
                    }
                }
            }
            
            self.logger.info(f'✅ HMM Regime Discovery completed: {len(regime_models)} regimes discovered (range: 2-150)')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'data_points': len(market_data),
                    'regime_count': len(regime_models),
                    'regime_range': '2-150',
                    'optimization_mode': optimization_mode
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
    
    async def _load_market_data(self, data: Any) -> Optional[pd.DataFrame]:
        """Load and prepare market data for regime discovery using klines_parquet manager."""
        try:
            if data is None or (isinstance(data, pd.DataFrame) and data.empty):
                self.logger.warning("⚠️ No market data provided, attempting to load from klines_parquet")
                
                # Try to load data using klines_parquet manager
                from src.utils.data.klines_parquet import get_klines_manager
                
                manager = get_klines_manager()
                
                # Try to get symbol and timeframe from pipeline state or use defaults
                symbol = "ETHUSDT"  # Default symbol
                timeframe = "1h"    # Default timeframe for HMM regime discovery
                
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
            
            # Perform regime detection with mode
            regime_dataframe = regime_detector.detect_regimes(prepared_data, config=config, mode=mode)
            
            return regime_dataframe
            
        except Exception as e:
            # Propagate the error with context for higher-level handling
            raise RuntimeError(f"Regime discovery process failed: {e}")
    
    def _prepare_data_for_regime_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare market data for regime detection."""
        # Work on a copy to avoid mutating the caller's DataFrame
        data_copy = data.copy()
        
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data_copy.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for regime detection: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data_copy[col] = 1000  # Default volume
                else:
                    if 'close' in data_copy.columns:
                        data_copy[col] = data_copy['close']
                    else:
                        data_copy[col] = 100.0  # Conservative fallback
        
        return data_copy
    
    def _calculate_regime_distribution(self, regime_assignments: List[int], label_name_mapping: Optional[Dict[int, str]] = None) -> Dict[str, float]:
        """Calculate the distribution of regime assignments.
        Optionally remap numeric labels to stable regime names.
        """
        if not regime_assignments:
            return {}
        
        total_assignments = len(regime_assignments)
        regime_counts = {}
        
        for assignment in regime_assignments:
            regime_counts[assignment] = regime_counts.get(assignment, 0) + 1
        
        # Convert to percentages
        regime_distribution = {}
        for regime, count in regime_counts.items():
            if label_name_mapping and regime in label_name_mapping:
                key = label_name_mapping[regime]
            else:
                key = f'regime_{regime}'
            regime_distribution[key] = (count / total_assignments) * 100
        
        return regime_distribution
