"""
TAS Regime Detector

This module provides a TAS-specific interface to the unified regime detector,
maintaining backward compatibility while using the unified implementation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Import the unified regime detector
from src.utils.nas_tas.regime_detector import create_tas_regime_detector
from src.utils.nas_tas.unified_result import UnifiedRegimeResult

# Import tprint for logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class TASRegimeDetector:
    """
    TAS Regime Detector
    
    Provides a TAS-specific interface to the unified regime detector,
    maintaining backward compatibility while using the unified implementation.
    """
    
    def __init__(self, config=None):
        """
        Initialize TAS regime detector.
        
        Args:
            config: TAS configuration (optional, uses defaults if None)
        """
        tprint_info("🌲 Initializing TAS Regime Detector")
        
        # Extract configuration parameters if provided
        n_regimes = 5
        primary_timeframe = "1m"
        
        if config is not None:
            n_regimes = getattr(config, 'n_regimes', 5)
            primary_timeframe = getattr(config, 'primary_timeframe', "1m")
        
        # Create TAS-specific regime detector
        self.detector = create_tas_regime_detector(
            n_regimes=n_regimes,
            primary_timeframe=primary_timeframe
        )
        
        tprint_success("✅ TAS Regime Detector initialized")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("✅ TAS Regime Detector initialized")
    
    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_performance: bool = True,
                      enable_clvsa_enhancement: bool = True) -> UnifiedRegimeResult:
        """
        Detect market regimes using TAS system.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_performance: Whether to use hardware optimization
            enable_clvsa_enhancement: Whether to enable CLVSA enhancement (for compatibility)
            
        Returns:
            UnifiedRegimeResult with regime detection results
        """
        tprint_info("🌲 Starting TAS regime detection...")
        
        try:
            result = self.detector.detect_regimes(
                market_data=market_data,
                timestamps=timestamps,
                optimize_performance=optimize_performance
            )
            
            tprint_success("✅ TAS regime detection completed")
            return result
            
        except Exception as e:
            tprint_error(f"❌ TAS regime detection failed: {e}")
            self.logger.error(f"TAS regime detection failed: {e}")
            raise
    
    def save_results(self, result: UnifiedRegimeResult, filepath: str) -> bool:
        """Save TAS regime detection results to file."""
        return self.detector.save_results(result, filepath)
    
    def load_results(self, filepath: str) -> Optional[UnifiedRegimeResult]:
        """Load TAS regime detection results from file."""
        return self.detector.load_results(filepath)