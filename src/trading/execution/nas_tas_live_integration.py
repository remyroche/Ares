"""
NAS-TAS Live Trading Integration

This module integrates NAS and TAS components into the live trading pipeline,
providing real-time regime detection and trading signal generation.

Key Features:
- Real-time NAS regime detection for Analyst (5m timeframe)
- Real-time TAS entry point optimization for Tactician (1m timeframe)
- Integration with existing live trading pipeline
- Dynamic architecture adaptation
- Performance monitoring and alerting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass
from pathlib import Path
import pickle
import asyncio

# Import NAS-TAS live trading components
from src.trading.execution.nas_enhanced_analyst_live import (
    NASEnhancedAnalystLive, NASEnhancedAnalystLiveConfig
)
from src.trading.execution.tas_enhanced_tactician_live import (
    TASEnhancedTacticianLive, TASEnhancedTacticianLiveConfig
)

# Import existing live trading components
from src.trading.execution.live_trader import LiveTrader
from src.trading.execution.trading_orchestrator import TradingOrchestrator
from src.trading.monitoring.trade_monitor import TradeMonitor
from src.trading.monitoring.performance_tracker import PerformanceTracker

# Import NAS and TAS components
from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import (
    PerfectNASConfig, NeuralArchitectureType
)
from src.training.steps.market_analysis.tas_regime.core.enhanced_tas_engine import (
    TASConfig, TreeSearchStrategy
)

from src.utils.logger import system_logger
from src.utils.tprint import tprint, tprint_info, tprint_success, tprint_error

logger = logging.getLogger(__name__)

@dataclass
class NASTASLiveIntegrationConfig:
    """Configuration for NAS-TAS Live Trading Integration."""
    # NAS Configuration
    nas_config: PerfectNASConfig
    enable_nas_analyst_live: bool = True
    
    # TAS Configuration
    tas_config: TASConfig
    enable_tas_tactician_live: bool = True
    
    # Live Trading Configuration
    analyst_timeframe: str = "5m"
    tactician_timeframe: str = "1m"
    signal_threshold: float = 0.6
    confidence_threshold: float = 0.7
    
    # Model Configuration
    remove_catboost: bool = True
    remove_xgboost: bool = True
    
    # Integration Configuration
    enable_signal_coordination: bool = True
    enable_performance_monitoring: bool = True
    enable_architecture_adaptation: bool = True

class NASTASLiveIntegration:
    """
    NAS-TAS Live Trading Integration.
    
    This class orchestrates the integration of NAS and TAS components into the
    live trading pipeline, providing real-time regime detection and trading signals.
    """
    
    def __init__(self, config: NASTASLiveIntegrationConfig):
        """Initialize NAS-TAS Live Trading Integration."""
        self.config = config
        self.logger = system_logger.getChild("NASTASLiveIntegration")
        
        # Initialize NAS-TAS live trading components
        self.nas_analyst_live = None
        self.tas_tactician_live = None
        
        # Initialize base live trading components
        self.live_trader = LiveTrader()
        self.trading_orchestrator = TradingOrchestrator()
        self.trade_monitor = TradeMonitor()
        self.performance_tracker = PerformanceTracker()
        
        # Integration state
        self.integration_results = {}
        self.performance_metrics = {}
        self.signal_history = []
        self.adaptation_history = []
        
        self.logger.info("✅ NAS-TAS Live Trading Integration initialized")
        self.logger.info(f"   NAS Analyst Live enabled: {config.enable_nas_analyst_live}")
        self.logger.info(f"   TAS Tactician Live enabled: {config.enable_tas_tactician_live}")
        self.logger.info(f"   CatBoost removed: {config.remove_catboost}")
        self.logger.info(f"   XGBoost removed: {config.remove_xgboost}")
        self.logger.info(f"   Signal coordination: {config.enable_signal_coordination}")
    
    async def initialize_live_trading(self) -> bool:
        """Initialize live trading components."""
        self.logger.info("🚀 Initializing NAS-TAS live trading components...")
        
        try:
            # Initialize NAS-Enhanced Analyst Live
            if self.config.enable_nas_analyst_live:
                nas_analyst_config = NASEnhancedAnalystLiveConfig(
                    nas_config=self.config.nas_config,
                    tas_config=self.config.tas_config,
                    enable_nas_live_detection=True,
                    enable_tas_5m=True,
                    remove_catboost=self.config.remove_catboost,
                    signal_threshold=self.config.signal_threshold,
                    confidence_threshold=self.config.confidence_threshold
                )
                self.nas_analyst_live = NASEnhancedAnalystLive(nas_analyst_config)
                self.logger.info("✅ NAS-Enhanced Analyst Live initialized")
            
            # Initialize TAS-Enhanced Tactician Live
            if self.config.enable_tas_tactician_live:
                tas_tactician_config = TASEnhancedTacticianLiveConfig(
                    tas_config=self.config.tas_config,
                    enable_tas_live_optimization=True,
                    remove_xgboost=self.config.remove_xgboost,
                    signal_threshold=self.config.signal_threshold,
                    confidence_threshold=self.config.confidence_threshold,
                    enable_tree_ensemble=True,
                    enable_boosting=True,
                    enable_bagging=True
                )
                self.tas_tactician_live = TASEnhancedTacticianLive(tas_tactician_config)
                self.logger.info("✅ TAS-Enhanced Tactician Live initialized")
            
            # Initialize base live trading components
            await self.live_trader.initialize()
            await self.trading_orchestrator.initialize()
            await self.trade_monitor.initialize()
            await self.performance_tracker.initialize()
            
            self.logger.info("✅ NAS-TAS live trading components initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize NAS-TAS live trading components: {e}")
            return False
    
    async def process_market_data(self, 
                                 market_data: Dict[str, Any], 
                                 current_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Process market data and generate trading signals.
        
        Args:
            market_data: Current market data
            current_time: Current timestamp (optional)
            
        Returns:
            Trading signals and analysis results
        """
        start_time = time.time()
        self.logger.info("🔍 Processing market data with NAS-TAS live integration...")
        
        try:
            if current_time is None:
                current_time = time.time()
            
            # Step 1: Process with NAS-Enhanced Analyst (5m timeframe)
            analyst_results = await self._process_with_nas_analyst(
                market_data, current_time
            )
            
            # Step 2: Process with TAS-Enhanced Tactician (1m timeframe)
            tactician_results = await self._process_with_tas_tactician(
                market_data, analyst_results, current_time
            )
            
            # Step 3: Coordinate signals if enabled
            coordinated_signals = await self._coordinate_signals(
                analyst_results, tactician_results, current_time
            )
            
            # Step 4: Update performance metrics
            await self._update_performance_metrics(
                analyst_results, tactician_results, coordinated_signals, current_time
            )
            
            execution_time = time.time() - start_time
            
            # Compile results
            results = {
                'success': True,
                'execution_time': execution_time,
                'analyst_results': analyst_results,
                'tactician_results': tactician_results,
                'coordinated_signals': coordinated_signals,
                'metadata': {
                    'timeframe_analyst': self.config.analyst_timeframe,
                    'timeframe_tactician': self.config.tactician_timeframe,
                    'nas_analyst_enabled': self.config.enable_nas_analyst_live,
                    'tas_tactician_enabled': self.config.enable_tas_tactician_live,
                    'signal_coordination_enabled': self.config.enable_signal_coordination,
                    'catboost_removed': self.config.remove_catboost,
                    'xgboost_removed': self.config.remove_xgboost
                }
            }
            
            self.logger.info(f"✅ Market data processing completed in {execution_time:.2f}s")
            self._log_processing_summary(results)
            
            return results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Market data processing failed: {e}")
            
            return {
                'success': False,
                'execution_time': execution_time,
                'error': str(e),
                'metadata': {'error': str(e)}
            }
    
    async def _process_with_nas_analyst(self, 
                                      market_data: Dict[str, Any], 
                                      current_time: float) -> Dict[str, Any]:
        """Process market data with NAS-Enhanced Analyst."""
        if not self.nas_analyst_live:
            return {
                'success': False,
                'error': 'NAS-Enhanced Analyst Live not enabled',
                'signal_generated': False
            }
        
        self.logger.info("🔍 Processing with NAS-Enhanced Analyst...")
        
        try:
            # Process with NAS-Enhanced Analyst
            analyst_results = await self.nas_analyst_live.process_market_data(
                market_data, current_time
            )
            
            if analyst_results.get('success', False):
                self.logger.info("✅ NAS-Enhanced Analyst processing completed")
            else:
                self.logger.warning("⚠️ NAS-Enhanced Analyst processing failed")
            
            return analyst_results
            
        except Exception as e:
            self.logger.error(f"❌ NAS-Enhanced Analyst processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'signal_generated': False
            }
    
    async def _process_with_tas_tactician(self, 
                                        market_data: Dict[str, Any], 
                                        analyst_results: Dict[str, Any],
                                        current_time: float) -> Dict[str, Any]:
        """Process market data with TAS-Enhanced Tactician."""
        if not self.tas_tactician_live:
            return {
                'success': False,
                'error': 'TAS-Enhanced Tactician Live not enabled',
                'signal_generated': False
            }
        
        self.logger.info("🔍 Processing with TAS-Enhanced Tactician...")
        
        try:
            # Extract analyst signals for Tactician
            analyst_signals = None
            if analyst_results.get('success', False) and analyst_results.get('signal_generated', False):
                analyst_signals = {
                    'signal_direction': analyst_results.get('signal_direction', 0),
                    'signal_strength': analyst_results.get('signal_strength', 0.0),
                    'confidence': analyst_results.get('confidence', 0.0),
                    'current_regime': analyst_results.get('current_regime'),
                    'regime_stability': analyst_results.get('regime_stability', 0.0),
                    'economic_significance': analyst_results.get('economic_significance', 0.0),
                    'trading_viability': analyst_results.get('trading_viability', 0.0)
                }
            
            # Process with TAS-Enhanced Tactician
            tactician_results = await self.tas_tactician_live.process_market_data(
                market_data, analyst_signals, current_time
            )
            
            if tactician_results.get('success', False):
                self.logger.info("✅ TAS-Enhanced Tactician processing completed")
            else:
                self.logger.warning("⚠️ TAS-Enhanced Tactician processing failed")
            
            return tactician_results
            
        except Exception as e:
            self.logger.error(f"❌ TAS-Enhanced Tactician processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'signal_generated': False
            }
    
    async def _coordinate_signals(self, 
                                analyst_results: Dict[str, Any], 
                                tactician_results: Dict[str, Any],
                                current_time: float) -> Dict[str, Any]:
        """Coordinate signals from Analyst and Tactician."""
        if not self.config.enable_signal_coordination:
            return {
                'signal_coordination_enabled': False,
                'final_signal': None,
                'coordination_reason': 'Signal coordination disabled'
            }
        
        self.logger.info("🔄 Coordinating signals from Analyst and Tactician...")
        
        try:
            # Check if both components generated signals
            analyst_signal = analyst_results.get('signal_generated', False)
            tactician_signal = tactician_results.get('signal_generated', False)
            
            if not analyst_signal:
                return {
                    'signal_coordination_enabled': True,
                    'final_signal': None,
                    'coordination_reason': 'No analyst signal',
                    'analyst_signal': False,
                    'tactician_signal': tactician_signal
                }
            
            if not tactician_signal:
                return {
                    'signal_coordination_enabled': True,
                    'final_signal': None,
                    'coordination_reason': 'No tactician signal',
                    'analyst_signal': analyst_signal,
                    'tactician_signal': False
                }
            
            # Both components generated signals - coordinate them
            analyst_direction = analyst_results.get('signal_direction', 0)
            tactician_direction = tactician_results.get('signal_direction', 0)
            
            # Check if directions match
            if analyst_direction == tactician_direction:
                # Directions match - use combined signal
                analyst_strength = analyst_results.get('signal_strength', 0.0)
                tactician_strength = tactician_results.get('signal_strength', 0.0)
                analyst_confidence = analyst_results.get('confidence', 0.0)
                tactician_confidence = tactician_results.get('confidence', 0.0)
                
                # Combine signals
                combined_strength = (analyst_strength + tactician_strength) / 2
                combined_confidence = (analyst_confidence + tactician_confidence) / 2
                
                final_signal = {
                    'signal_generated': True,
                    'signal_direction': analyst_direction,
                    'signal_strength': combined_strength,
                    'confidence': combined_confidence,
                    'analyst_contribution': {
                        'strength': analyst_strength,
                        'confidence': analyst_confidence
                    },
                    'tactician_contribution': {
                        'strength': tactician_strength,
                        'confidence': tactician_confidence
                    }
                }
                
                self.logger.info(f"✅ Signals coordinated successfully")
                self.logger.info(f"   Direction: {analyst_direction}")
                self.logger.info(f"   Combined strength: {combined_strength:.3f}")
                self.logger.info(f"   Combined confidence: {combined_confidence:.3f}")
                
                return {
                    'signal_coordination_enabled': True,
                    'final_signal': final_signal,
                    'coordination_reason': 'Signals coordinated successfully',
                    'analyst_signal': analyst_signal,
                    'tactician_signal': tactician_signal
                }
            else:
                # Directions don't match - no signal
                return {
                    'signal_coordination_enabled': True,
                    'final_signal': None,
                    'coordination_reason': 'Signal directions do not match',
                    'analyst_signal': analyst_signal,
                    'tactician_signal': tactician_signal,
                    'analyst_direction': analyst_direction,
                    'tactician_direction': tactician_direction
                }
                
        except Exception as e:
            self.logger.error(f"❌ Signal coordination failed: {e}")
            return {
                'signal_coordination_enabled': True,
                'final_signal': None,
                'coordination_reason': f'Signal coordination error: {e}',
                'analyst_signal': analyst_results.get('signal_generated', False),
                'tactician_signal': tactician_results.get('signal_generated', False)
            }
    
    async def _update_performance_metrics(self, 
                                        analyst_results: Dict[str, Any], 
                                        tactician_results: Dict[str, Any],
                                        coordinated_signals: Dict[str, Any],
                                        current_time: float) -> None:
        """Update performance metrics."""
        try:
            # Update signal history
            signal_entry = {
                'timestamp': current_time,
                'analyst_success': analyst_results.get('success', False),
                'tactician_success': tactician_results.get('success', False),
                'analyst_signal_generated': analyst_results.get('signal_generated', False),
                'tactician_signal_generated': tactician_results.get('signal_generated', False),
                'coordinated_signal': coordinated_signals.get('final_signal'),
                'coordination_enabled': coordinated_signals.get('signal_coordination_enabled', False)
            }
            
            self.signal_history.append(signal_entry)
            
            # Keep only last 1000 entries
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            # Update performance metrics
            self.performance_metrics.update({
                'total_signals_processed': len(self.signal_history),
                'analyst_success_rate': self._calculate_success_rate('analyst_success'),
                'tactician_success_rate': self._calculate_success_rate('tactician_success'),
                'analyst_signal_rate': self._calculate_signal_rate('analyst_signal_generated'),
                'tactician_signal_rate': self._calculate_signal_rate('tactician_signal_generated'),
                'coordination_rate': self._calculate_coordination_rate()
            })
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to update performance metrics: {e}")
    
    def _calculate_success_rate(self, field: str) -> float:
        """Calculate success rate for a field."""
        try:
            if not self.signal_history:
                return 0.0
            
            success_count = sum(1 for entry in self.signal_history if entry.get(field, False))
            total_count = len(self.signal_history)
            
            return success_count / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate success rate for {field}: {e}")
            return 0.0
    
    def _calculate_signal_rate(self, field: str) -> float:
        """Calculate signal rate for a field."""
        try:
            if not self.signal_history:
                return 0.0
            
            signal_count = sum(1 for entry in self.signal_history if entry.get(field, False))
            total_count = len(self.signal_history)
            
            return signal_count / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate signal rate for {field}: {e}")
            return 0.0
    
    def _calculate_coordination_rate(self) -> float:
        """Calculate signal coordination rate."""
        try:
            if not self.signal_history:
                return 0.0
            
            coordination_count = sum(1 for entry in self.signal_history 
                                   if entry.get('coordination_enabled', False))
            total_count = len(self.signal_history)
            
            return coordination_count / total_count if total_count > 0 else 0.0
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to calculate coordination rate: {e}")
            return 0.0
    
    def _log_processing_summary(self, results: Dict[str, Any]):
        """Log processing summary."""
        try:
            metadata = results.get('metadata', {})
            self.logger.info("📊 NAS-TAS Live Integration Processing Summary:")
            self.logger.info(f"   Success: {results.get('success', False)}")
            self.logger.info(f"   Execution time: {results.get('execution_time', 0):.2f}s")
            self.logger.info(f"   Analyst timeframe: {metadata.get('timeframe_analyst', 'unknown')}")
            self.logger.info(f"   Tactician timeframe: {metadata.get('timeframe_tactician', 'unknown')}")
            self.logger.info(f"   NAS Analyst enabled: {metadata.get('nas_analyst_enabled', False)}")
            self.logger.info(f"   TAS Tactician enabled: {metadata.get('tas_tactician_enabled', False)}")
            self.logger.info(f"   Signal coordination: {metadata.get('signal_coordination_enabled', False)}")
            self.logger.info(f"   CatBoost removed: {metadata.get('catboost_removed', False)}")
            self.logger.info(f"   XGBoost removed: {metadata.get('xgboost_removed', False)}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to log processing summary: {e}")
    
    def save_integration_state(self, filepath: str) -> bool:
        """Save integration state."""
        try:
            integration_data = {
                'integration_results': self.integration_results,
                'performance_metrics': self.performance_metrics,
                'signal_history': self.signal_history,
                'adaptation_history': self.adaptation_history,
                'config': self.config
            }
            
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(integration_data, f)
            
            self.logger.info(f"✅ Integration state saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save integration state: {e}")
            return False
    
    def load_integration_state(self, filepath: str) -> bool:
        """Load integration state."""
        try:
            with open(filepath, 'rb') as f:
                integration_data = pickle.load(f)
            
            self.integration_results = integration_data.get('integration_results', {})
            self.performance_metrics = integration_data.get('performance_metrics', {})
            self.signal_history = integration_data.get('signal_history', [])
            self.adaptation_history = integration_data.get('adaptation_history', [])
            
            self.logger.info(f"✅ Integration state loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load integration state: {e}")
            return False


# Factory function for creating NAS-TAS Live Trading Integration
def create_nas_tas_live_integration(config: Optional[NASTASLiveIntegrationConfig] = None) -> NASTASLiveIntegration:
    """Create NAS-TAS Live Trading Integration instance."""
    if config is None:
        # Default configuration
        nas_config = PerfectNASConfig(
            primary_architecture=NeuralArchitectureType.HYBRID,
            n_regimes=8,
            primary_timeframe="5m",
            enable_neural_odes=True,
            enable_vision_transformers=True,
            enable_state_space_models=True,
            enable_micro_regime_detection=True,
            population_size=30,
            generations=50
        )
        
        tas_config = TASConfig(
            search_strategy=TreeSearchStrategy.ENHANCED_BAYESIAN,
            population_size=25,
            max_generations=40,
            max_evaluations=150,
            enable_multi_objective=True,
            objective_weights={
                'performance': 1.0,
                'complexity': 0.3,
                'efficiency': 0.4,
                'interpretability': 0.5
            },
            max_trees=30,
            max_tree_depth=12,
            allow_boosting=True,
            allow_bagging=True,
            allow_ensemble_methods=True
        )
        
        config = NASTASLiveIntegrationConfig(
            nas_config=nas_config,
            tas_config=tas_config,
            enable_nas_analyst_live=True,
            enable_tas_tactician_live=True,
            remove_catboost=True,
            remove_xgboost=True,
            enable_signal_coordination=True,
            enable_performance_monitoring=True,
            enable_architecture_adaptation=True
        )
    
    return NASTASLiveIntegration(config)