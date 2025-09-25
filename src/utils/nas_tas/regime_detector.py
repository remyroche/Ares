"""
Unified Regime Detector Interface

This module provides a simple interface to the unified regime detector
for both TAS and NAS systems, eliminating code duplication.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

# Import the unified regime detector
from .unified_regime_detector import UnifiedRegimeDetector
from .unified_regime_config import UnifiedRegimeConfig, RegimeSystemType, ArchitectureType
from .unified_result import UnifiedRegimeResult

# Import tprint for logging
from src.utils.tprint import (
    tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, 
    tprint_success, tprint_progress, tprint_performance, tprint_timer
)

logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Unified Regime Detector Interface
    
    Provides a simple interface for regime detection that can be used
    by both TAS and NAS systems, eliminating code duplication.
    """
    
    def __init__(self, 
                 system_type: str = "unified",
                 n_regimes: int = 5,
                 primary_timeframe: str = "1m",
                 enable_hybrid_mode: bool = True):
        """
        Initialize the unified regime detector.
        
        Args:
            system_type: Type of system ("tas", "nas", "hybrid", "unified")
            n_regimes: Number of regimes to detect
            primary_timeframe: Primary timeframe for analysis
            enable_hybrid_mode: Whether to enable hybrid TAS-NAS mode
        """
        tprint_info(f"🚀 Initializing Unified Regime Detector (System: {system_type})")
        
        # Map system type to enum
        system_type_map = {
            "tas": RegimeSystemType.TAS,
            "nas": RegimeSystemType.NAS,
            "hybrid": RegimeSystemType.HYBRID,
            "unified": RegimeSystemType.UNIFIED
        }
        
        system_enum = system_type_map.get(system_type.lower(), RegimeSystemType.UNIFIED)
        
        # Create unified configuration
        self.config = UnifiedRegimeConfig(
            system_type=system_enum,
            n_regimes=n_regimes,
            primary_timeframe=primary_timeframe,
            enable_hybrid_mode=enable_hybrid_mode,
            primary_architecture=ArchitectureType.ADAPTIVE
        )
        
        # Initialize the unified detector
        self.detector = UnifiedRegimeDetector(self.config)
        
        tprint_success("✅ Unified Regime Detector initialized")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"✅ Unified Regime Detector initialized (System: {system_type})")
    
    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      timestamps: Optional[np.ndarray] = None,
                      optimize_performance: bool = True) -> UnifiedRegimeResult:
        """
        Detect market regimes using the unified system.
        
        Args:
            market_data: Market data (OHLCV)
            timestamps: Optional timestamps
            optimize_performance: Whether to use hardware optimization
            
        Returns:
            UnifiedRegimeResult with regime detection results
        """
        tprint_info("🔍 Starting regime detection...")
        
        try:
            result = self.detector.detect_regimes(
                market_data=market_data,
                timestamps=timestamps,
                optimize_performance=optimize_performance,
                enable_hybrid_mode=self.config.enable_hybrid_mode
            )
            
            tprint_success("✅ Regime detection completed")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Regime detection failed: {e}")
            self.logger.error(f"Regime detection failed: {e}")
            raise
    
    def save_results(self, result: UnifiedRegimeResult, filepath: str) -> bool:
        """Save regime detection results to file."""
        return self.detector.save_results(result, filepath)
    
    def load_results(self, filepath: str) -> Optional[UnifiedRegimeResult]:
        """Load regime detection results from file."""
        return self.detector.load_results(filepath)


# Convenience functions for easy usage
def create_tas_regime_detector(n_regimes: int = 5, 
                              primary_timeframe: str = "1m") -> RegimeDetector:
    """Create a TAS-specific regime detector."""
    return RegimeDetector(
        system_type="tas",
        n_regimes=n_regimes,
        primary_timeframe=primary_timeframe,
        enable_hybrid_mode=False
    )


def create_nas_regime_detector(n_regimes: int = 5, 
                              primary_timeframe: str = "1m") -> RegimeDetector:
    """Create a NAS-specific regime detector."""
    return RegimeDetector(
        system_type="nas",
        n_regimes=n_regimes,
        primary_timeframe=primary_timeframe,
        enable_hybrid_mode=False
    )


def create_hybrid_regime_detector(n_regimes: int = 5, 
                                 primary_timeframe: str = "1m") -> RegimeDetector:
    """Create a hybrid TAS-NAS regime detector."""
    return RegimeDetector(
        system_type="hybrid",
        n_regimes=n_regimes,
        primary_timeframe=primary_timeframe,
        enable_hybrid_mode=True
    )


def create_unified_regime_detector(n_regimes: int = 5, 
                                  primary_timeframe: str = "1m") -> RegimeDetector:
    """Create a unified regime detector (recommended)."""
    return RegimeDetector(
        system_type="unified",
        n_regimes=n_regimes,
        primary_timeframe=primary_timeframe,
        enable_hybrid_mode=True
    )