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
                light_mode_max_regimes=3,
                blank_mode_max_regimes=5,
                full_mode_max_regimes=50
            )
            
            # Create HMM regime detector
            regime_detector = EnhancedHMMRegimeDetector()
            
            # Determine optimization mode from config or default to 'blank'
            optimization_mode = getattr(self.config, 'optimization_mode', 'blank')
            
            # Perform regime discovery
            regime_result = await self._perform_regime_discovery(
                regime_detector, market_data, hmm_config, optimization_mode
            )
            
            # Extract results
            regime_models = regime_result.get('regime_models', [])
            regime_assignments = regime_result.get('regime_assignments', [])
            regime_metrics = regime_result.get('regime_metrics', {})
            
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
                        'regime_distribution': self._calculate_regime_distribution(regime_assignments),
                        'discovery_time': regime_result.get('discovery_time', 0.0)
                    },
                    'metadata': {
                        'symbol': self.config.symbol,
                        'exchange': self.config.exchange,
                        'timeframe': self.config.timeframe,
                        'data_points': len(market_data) if market_data is not None else 0,
                        'execution_timestamp': datetime.now().isoformat()
                    }
                }
            }
            
            self.logger.info(f'✅ HMM Regime Discovery completed: {len(regime_models)} regimes discovered')
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'symbol': self.config.symbol,
                    'exchange': self.config.exchange,
                    'timeframe': self.config.timeframe,
                    'data_points': len(market_data),
                    'regime_count': len(regime_models)
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
        """Load and prepare market data for regime discovery."""
        if data is None:
            return None
        
        if isinstance(data, pd.DataFrame):
            return data.copy()
        
        # Handle other data types if needed
        return None
    
    async def _perform_regime_discovery(
        self, 
        regime_detector: Any, 
        market_data: pd.DataFrame, 
        config: Any,
        mode: str = 'blank'
    ) -> Dict[str, Any]:
        """Perform the actual regime discovery process."""
        try:
            # Prepare data for regime detection
            prepared_data = self._prepare_data_for_regime_detection(market_data)
            
            # Perform regime detection with mode
            regime_result = await regime_detector.detect_regimes(prepared_data, config=config, mode=mode)
            
            return regime_result
            
        except Exception as e:
            self.logger.error(f"Regime discovery process failed: {e}")
            # Return fallback regime result
            return {
                'regime_models': [],
                'regime_assignments': [],
                'regime_metrics': {
                    'detection_method': 'fallback',
                    'error': str(e)
                },
                'discovery_time': 0.0
            }
    
    def _prepare_data_for_regime_detection(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare market data for regime detection."""
        # Ensure we have required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.warning(f"Missing columns for regime detection: {missing_columns}")
            # Use available columns or create fallback data
            for col in missing_columns:
                if col == 'volume':
                    data[col] = 1000  # Default volume
                else:
                    data[col] = data.get('close', 100.0)  # Use close price as fallback
        
        return data
    
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
            regime_distribution[f'regime_{regime}'] = (count / total_assignments) * 100
        
        return regime_distribution
