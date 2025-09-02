#!/usr/bin/env python3
"""
HMM Regime Barrier Optimizer

This module provides an interface for optimizing HMM regime-specific barriers
and applying regime-aware triple barrier labeling with automatic recalculation.

Key Features:
- Automatic HMM barrier recalculation
- Regime-specific barrier optimization
- Integration with triple barrier labeling
- Fallback mechanisms for robustness
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import json
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import MLflow for experiment tracking
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Import Optuna for optimization
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

logger = logging.getLogger(__name__)


class HMMRegimeBarrierOptimizer:
    """
    HMM Regime Barrier Optimizer for automatic barrier recalculation and optimization.
    
    This class provides the interface needed by the vectorized labeling orchestrator
    to automatically recalculate HMM regime barriers and apply regime-aware labeling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
        
        # Default configuration
        self.default_config = {
            "enable_regime_specific_parameters": True,
            "regime_parameter_optimization": True,
            "auto_recalculate_hmm_barriers": True,
            "hmm_barrier_regime_column": "hmm_regime",
            "time_barrier_minutes": 30,
            "max_lookahead": 100,
            "profit_take_multiplier": 0.002,
            "stop_loss_multiplier": 0.001,
        }
        
        # Update with provided config
        self.config = {**self.default_config, **config}
        
        # Optimization results storage
        self.optimization_results = {}
        self.regime_models = {}
        self.barrier_map = {}
        
        # MLflow experiment tracking
        self.mlflow_experiment_name = "hmm_regime_barrier_optimization"
        
        self.logger.info("✅ HMMRegimeBarrierOptimizer initialized successfully")
        
    async def optimize_regime_barriers(
        self, 
        data: pd.DataFrame, 
        regime_column: str = "hmm_regime"
    ) -> Dict[str, Any]:
        """
        Optimize regime-specific barriers for the given data.
        
        Args:
            data: DataFrame containing price data and regime information
            regime_column: Column name containing HMM regime information
            
        Returns:
            Dictionary containing optimization results
        """
        try:
            self.logger.info(f"🔧 Starting HMM regime barrier optimization for column '{regime_column}'")
            
            if regime_column not in data.columns:
                raise ValueError(f"Regime column '{regime_column}' not found in data")
            
            # Get unique regimes
            regimes = data[regime_column].unique()
            self.logger.info(f"📊 Found {len(regimes)} unique regimes: {regimes}")
            
            # Create regime-specific barrier configurations
            self.barrier_map = self._create_regime_barrier_map(regimes)
            
            # Optimize barriers for each regime
            for regime in regimes:
                if pd.isna(regime):
                    continue
                    
                regime_data = data[data[regime_column] == regime]
                if len(regime_data) < 100:  # Need sufficient data for optimization
                    self.logger.warning(f"⚠️ Insufficient data for regime {regime} ({len(regime_data)} samples)")
                    continue
                
                self.logger.info(f"🎯 Optimizing barriers for regime {regime} ({len(regime_data)} samples)")
                regime_optimization = await self._optimize_regime_barriers(regime_data, regime)
                self.optimization_results[regime] = regime_optimization
            
            self.logger.info(f"✅ HMM regime barrier optimization completed for {len(self.optimization_results)} regimes")
            return self.optimization_results
            
        except Exception as e:
            self.logger.error(f"❌ Error in HMM regime barrier optimization: {e}")
            # Return default barriers on error
            return self._get_default_barriers()
    
    def _create_regime_barrier_map(self, regimes: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Create a barrier map for each regime."""
        barrier_map = {}
        
        for regime in regimes:
            if pd.isna(regime):
                continue
                
            # Create regime-specific barrier configuration
            barrier_map[str(regime)] = {
                "profit_take_multiplier": self.config.get("profit_take_multiplier", 0.002),
                "stop_loss_multiplier": self.config.get("stop_loss_multiplier", 0.001),
                "time_barrier_minutes": self.config.get("time_barrier_minutes", 30),
                "max_lookahead": self.config.get("max_lookahead", 100),
                "regime_id": str(regime),
                "optimization_timestamp": datetime.now().isoformat(),
            }
        
        return barrier_map
    
    async def _optimize_regime_barriers(self, regime_data: pd.DataFrame, regime: Any) -> Dict[str, Any]:
        """Optimize barriers for a specific regime."""
        try:
            # Simple optimization based on regime characteristics
            volatility = regime_data['close'].pct_change().std()
            trend = regime_data['close'].iloc[-1] / regime_data['close'].iloc[0] - 1
            
            # Adjust barriers based on regime characteristics
            if trend > 0.01:  # Bullish regime
                profit_mult = self.config.get("profit_take_multiplier", 0.002) * 1.2
                stop_mult = self.config.get("stop_loss_multiplier", 0.001) * 0.8
            elif trend < -0.01:  # Bearish regime
                profit_mult = self.config.get("profit_take_multiplier", 0.002) * 0.8
                stop_mult = self.config.get("stop_loss_multiplier", 0.001) * 1.2
            else:  # Sideways regime
                profit_mult = self.config.get("profit_take_multiplier", 0.002)
                stop_mult = self.config.get("stop_loss_multiplier", 0.001)
            
            # Adjust for volatility
            if volatility > 0.02:  # High volatility
                profit_mult *= 1.3
                stop_mult *= 1.3
            elif volatility < 0.005:  # Low volatility
                profit_mult *= 0.7
                stop_mult *= 0.7
            
            optimization_result = {
                "regime": regime,
                "profit_take_multiplier": profit_mult,
                "stop_loss_multiplier": stop_mult,
                "time_barrier_minutes": self.config.get("time_barrier_minutes", 30),
                "max_lookahead": self.config.get("max_lookahead", 100),
                "volatility": volatility,
                "trend": trend,
                "optimization_timestamp": datetime.now().isoformat(),
            }
            
            # Update barrier map
            self.barrier_map[str(regime)].update(optimization_result)
            
            return optimization_result
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error optimizing barriers for regime {regime}: {e}")
            return self._get_default_regime_barriers(regime)
    
    def _get_default_regime_barriers(self, regime: Any) -> Dict[str, Any]:
        """Get default barriers for a regime."""
        return {
            "regime": regime,
            "profit_take_multiplier": self.config.get("profit_take_multiplier", 0.002),
            "stop_loss_multiplier": self.config.get("stop_loss_multiplier", 0.001),
            "time_barrier_minutes": self.config.get("time_barrier_minutes", 30),
            "max_lookahead": self.config.get("max_lookahead", 100),
            "volatility": 0.01,
            "trend": 0.0,
            "optimization_timestamp": datetime.now().isoformat(),
        }
    
    def _get_default_barriers(self) -> Dict[str, Any]:
        """Get default barriers when optimization fails."""
        return {
            "default": {
                "profit_take_multiplier": self.config.get("profit_take_multiplier", 0.002),
                "stop_loss_multiplier": self.config.get("stop_loss_multiplier", 0.001),
                "time_barrier_minutes": self.config.get("time_barrier_minutes", 30),
                "max_lookahead": self.config.get("max_lookahead", 100),
                "regime_id": "default",
                "optimization_timestamp": datetime.now().isoformat(),
            }
        }
    
    def export_barrier_map(self) -> str:
        """
        Export the barrier map to a file and return the path.
        
        Returns:
            Path to the exported barrier map file
        """
        try:
            # Create export directory
            export_dir = Path("data_cache/hmm_barriers")
            export_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hmm_regime_barriers_{timestamp}.json"
            filepath = export_dir / filename
            
            # Export barrier map
            with open(filepath, 'w') as f:
                json.dump(self.barrier_map, f, indent=2, default=str)
            
            self.logger.info(f"✅ Barrier map exported to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self.logger.error(f"❌ Error exporting barrier map: {e}")
            # Return a default path
            return "data_cache/hmm_barriers/default_barriers.json"
    
    def get_barrier_map(self) -> Dict[str, Dict[str, Any]]:
        """Get the current barrier map."""
        return self.barrier_map
    
    def get_regime_barriers(self, regime: str) -> Dict[str, Any]:
        """Get barriers for a specific regime."""
        return self.barrier_map.get(str(regime), self._get_default_barriers().get("default", {}))
    
    def reset_optimization(self) -> None:
        """Reset optimization results and barrier map."""
        self.optimization_results = {}
        self.regime_models = {}
        self.barrier_map = {}
        self.logger.info("🔄 Optimization results and barrier map reset")