"""
Multi-Tier Trading System Orchestrator

This module integrates the HMM, Analyst, and Tactician systems into a cohesive
trading pipeline with proper scheduling and data flow management.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import asyncio
import threading
import time
from dataclasses import dataclass
from enum import Enum
import json
import os

from src.utils.tprint import tprint
from src.utils.logger import system_logger
from src.core.decorators import handles_errors

# Import the three systems
from src.hmm_system.hmm_regime_detector import HMMRegimeDetector, HMMConfig, create_hmm_regime_detector
from src.analyst_system.analyst_regime_predictor import AnalystRegimePredictor, AnalystConfig, create_analyst_regime_predictor
from src.tactician_system.tactician_timing_predictor import TacticianTimingPredictor, TacticianConfig, create_tactician_timing_predictor


class SystemStatus(Enum):
    """Status of the trading system."""
    INITIALIZING = "initializing"
    TRAINING = "training"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class TradingDecision:
    """Container for final trading decision."""
    timestamp: datetime
    should_trade: bool
    entry_confidence: float
    expected_return: float
    position_size: float
    leverage: float
    regime_id: int
    hmm_confidence: float
    analyst_confidence: float
    tactician_confidence: float
    risk_score: float
    market_conditions: Dict[str, Any]
    decision_reasoning: str


@dataclass
class SystemMetrics:
    """Container for system performance metrics."""
    timestamp: datetime
    hmm_runs: int
    analyst_runs: int
    tactician_runs: int
    green_lights: int
    trade_signals: int
    system_uptime: float
    avg_processing_time: float
    error_count: int


class MultiTierTradingOrchestrator:
    """
    Orchestrates the multi-tier trading system.
    
    This class manages:
    - HMM system (runs every 15 minutes on 1h data)
    - Analyst system (runs every 2 minutes on 5m data)
    - Tactician system (runs every 30 seconds on 1m data)
    - Data flow between systems
    - Scheduling and execution
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-tier trading orchestrator."""
        self.config = config or {}
        self.logger = system_logger.getChild('MultiTierTradingOrchestrator')
        
        # System status
        self.status = SystemStatus.INITIALIZING
        self.is_running = False
        self.start_time = None
        
        # Initialize the three systems
        self.hmm_system = create_hmm_regime_detector(
            HMMConfig(**self.config.get('hmm', {}))
        )
        self.analyst_system = create_analyst_regime_predictor(
            AnalystConfig(**self.config.get('analyst', {}))
        )
        self.tactician_system = create_tactician_timing_predictor(
            TacticianConfig(**self.config.get('tactician', {}))
        )
        
        # Data storage
        self.current_data_1h: Optional[pd.DataFrame] = None
        self.current_data_5m: Optional[pd.DataFrame] = None
        self.current_data_1m: Optional[pd.DataFrame] = None
        
        # System outputs
        self.latest_hmm_output: Optional[Dict[str, Any]] = None
        self.latest_analyst_output: Optional[Dict[str, Any]] = None
        self.latest_tactician_output: Optional[Dict[str, Any]] = None
        
        # Performance tracking
        self.metrics = SystemMetrics(
            timestamp=datetime.now(),
            hmm_runs=0,
            analyst_runs=0,
            tactician_runs=0,
            green_lights=0,
            trade_signals=0,
            system_uptime=0.0,
            avg_processing_time=0.0,
            error_count=0
        )
        
        # Threading
        self.execution_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Decision history
        self.decision_history: List[TradingDecision] = []
        self.max_history = self.config.get('max_history', 1000)
        
        tprint("Multi-tier trading orchestrator initialized")
    
    @handles_errors
    def load_data(self, data_1h: pd.DataFrame, data_5m: pd.DataFrame, data_1m: pd.DataFrame) -> None:
        """Load market data for all timeframes."""
        tprint("Loading market data for all timeframes...")
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for timeframe, data in [('1h', data_1h), ('5m', data_5m), ('1m', data_1m)]:
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns in {timeframe} data")
        
        self.current_data_1h = data_1h.copy()
        self.current_data_5m = data_5m.copy()
        self.current_data_1m = data_1m.copy()
        
        tprint(f"Data loaded: 1h={len(data_1h)} bars, 5m={len(data_5m)} bars, 1m={len(data_1m)} bars")
    
    @handles_errors
    def train_systems(self) -> Dict[str, Any]:
        """Train all three systems."""
        tprint("Training all trading systems...")
        self.status = SystemStatus.TRAINING
        
        training_results = {}
        
        try:
            # Train HMM system
            tprint("Training HMM regime detection system...")
            hmm_results = self.hmm_system.train_models(self.current_data_1h)
            training_results['hmm'] = hmm_results
            
            # Get HMM regime labels for Analyst training
            hmm_features = self.hmm_system.extract_features(self.current_data_1h)
            hmm_features_scaled = self.hmm_system.scaler.transform(hmm_features)
            regime_labels = self.hmm_system.models[self.hmm_system.config.n_regimes].predict(hmm_features_scaled)
            
            # Resample regime labels to 5m timeframe
            regime_labels_5m = self._resample_regime_labels(regime_labels, self.current_data_1h.index, self.current_data_5m.index)
            
            # Train Analyst system
            tprint("Training Analyst regime prediction system...")
            analyst_results = self.analyst_system.train_regime_models(
                self.current_data_5m, 
                regime_labels_5m,
                self.latest_hmm_output
            )
            training_results['analyst'] = analyst_results
            
            # Get Analyst green lights for Tactician training
            analyst_green_lights = self._get_analyst_green_lights_for_training()
            
            # Train Tactician system
            tprint("Training Tactician timing prediction system...")
            tactician_results = self.tactician_system.train_models(
                self.current_data_1m,
                analyst_green_lights,
                self.latest_hmm_output,
                self.latest_analyst_output
            )
            training_results['tactician'] = tactician_results
            
            self.status = SystemStatus.RUNNING
            tprint("All systems trained successfully")
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Training failed: {e}")
            raise
        
        return training_results
    
    def _resample_regime_labels(self, regime_labels: np.ndarray, 
                               source_index: pd.Index, target_index: pd.Index) -> np.ndarray:
        """Resample regime labels from 1h to 5m timeframe."""
        # Create a series with regime labels
        regime_series = pd.Series(regime_labels, index=source_index)
        
        # Forward fill to 5m timeframe
        regime_series_5m = regime_series.reindex(target_index, method='ffill')
        
        return regime_series_5m.fillna(0).values
    
    def _get_analyst_green_lights_for_training(self) -> np.ndarray:
        """Get Analyst green lights for Tactician training (simplified)."""
        # This is a simplified version - in practice, you'd run the Analyst
        # on historical data to get actual green light signals
        n_samples = len(self.current_data_1m)
        # Simulate green lights (20% of the time)
        green_lights = np.random.random(n_samples) < 0.2
        return green_lights
    
    @handles_errors
    def run_hmm_analysis(self) -> Optional[Dict[str, Any]]:
        """Run HMM analysis if it's time."""
        if not self.hmm_system.should_run():
            return None
        
        try:
            result = self.hmm_system.run_analysis(self.current_data_1h)
            if result is not None:
                self.latest_hmm_output = {
                    'regime_probs': result.regime_probs.tolist(),
                    'dominant_regime': result.dominant_regime,
                    'confidence': result.confidence,
                    'regime_characteristics': result.regime_characteristics,
                    'feature_importance': result.feature_importance
                }
                self.metrics.hmm_runs += 1
                tprint(f"HMM analysis completed: Regime {result.dominant_regime}")
            
            return self.latest_hmm_output
            
        except Exception as e:
            self.logger.error(f"HMM analysis failed: {e}")
            self.metrics.error_count += 1
            return None
    
    @handles_errors
    def run_analyst_analysis(self) -> Optional[Dict[str, Any]]:
        """Run Analyst analysis if it's time."""
        if not self.analyst_system.should_run():
            return None
        
        if self.latest_hmm_output is None:
            tprint("Analyst waiting for HMM output...")
            return None
        
        try:
            # Get current regime ID
            regime_id = self.latest_hmm_output.get('dominant_regime', 0)
            
            result = self.analyst_system.run_analysis(
                self.current_data_5m, 
                regime_id,
                self.latest_hmm_output
            )
            
            if result is not None:
                self.latest_analyst_output = {
                    'should_trade': result.should_trade,
                    'confidence': result.confidence,
                    'base_model_predictions': result.base_model_predictions,
                    'meta_learner_prediction': result.meta_learner_prediction,
                    'regime_id': result.regime_id,
                    'feature_importance': result.feature_importance,
                    'market_conditions': result.market_conditions
                }
                self.metrics.analyst_runs += 1
                
                if result.should_trade:
                    self.metrics.green_lights += 1
                    tprint(f"Analyst GREEN LIGHT: Regime {regime_id}")
                else:
                    tprint(f"Analyst RED LIGHT: Regime {regime_id}")
            
            return self.latest_analyst_output
            
        except Exception as e:
            self.logger.error(f"Analyst analysis failed: {e}")
            self.metrics.error_count += 1
            return None
    
    @handles_errors
    def run_tactician_analysis(self) -> Optional[Dict[str, Any]]:
        """Run Tactician analysis if it's time."""
        if not self.tactician_system.should_run():
            return None
        
        if self.latest_analyst_output is None:
            tprint("Tactician waiting for Analyst output...")
            return None
        
        # Only run if Analyst gave green light
        if not self.latest_analyst_output.get('should_trade', False):
            tprint("Tactician waiting for Analyst green light...")
            return None
        
        try:
            result = self.tactician_system.run_analysis(
                self.current_data_1m,
                self.latest_hmm_output,
                self.latest_analyst_output
            )
            
            if result is not None:
                self.latest_tactician_output = {
                    'should_enter': result.should_enter,
                    'entry_confidence': result.entry_confidence,
                    'expected_return': result.expected_return,
                    'risk_score': result.risk_score,
                    'position_size': result.position_size,
                    'leverage': result.leverage,
                    'base_model_predictions': result.base_model_predictions,
                    'meta_learner_prediction': result.meta_learner_prediction,
                    'feature_importance': result.feature_importance,
                    'market_timing': result.market_timing
                }
                self.metrics.tactician_runs += 1
                
                if result.should_enter:
                    self.metrics.trade_signals += 1
                    tprint(f"Tactician ENTER SIGNAL: {result.expected_return:.3f}% expected return")
                else:
                    tprint(f"Tactician WAIT: {result.expected_return:.3f}% expected return")
            
            return self.latest_tactician_output
            
        except Exception as e:
            self.logger.error(f"Tactician analysis failed: {e}")
            self.metrics.error_count += 1
            return None
    
    @handles_errors
    def make_trading_decision(self) -> Optional[TradingDecision]:
        """Make final trading decision based on all system outputs."""
        if not all([self.latest_hmm_output, self.latest_analyst_output, self.latest_tactician_output]):
            return None
        
        try:
            # Extract key information
            hmm_confidence = self.latest_hmm_output.get('confidence', 0.0)
            analyst_confidence = self.latest_analyst_output.get('confidence', 0.0)
            tactician_confidence = self.latest_tactician_output.get('entry_confidence', 0.0)
            
            regime_id = self.latest_hmm_output.get('dominant_regime', 0)
            should_trade = self.latest_analyst_output.get('should_trade', False)
            should_enter = self.latest_tactician_output.get('should_enter', False)
            
            # Final decision logic
            should_trade_final = should_trade and should_enter
            
            # Calculate overall confidence
            overall_confidence = (hmm_confidence + analyst_confidence + tactician_confidence) / 3
            
            # Get position sizing
            position_size = self.latest_tactician_output.get('position_size', 0.0)
            leverage = self.latest_tactician_output.get('leverage', 1.0)
            expected_return = self.latest_tactician_output.get('expected_return', 0.0)
            risk_score = self.latest_tactician_output.get('risk_score', 0.5)
            
            # Market conditions
            market_conditions = {
                'hmm_regime': regime_id,
                'hmm_confidence': hmm_confidence,
                'analyst_confidence': analyst_confidence,
                'tactician_confidence': tactician_confidence,
                'risk_score': risk_score,
                'volatility': self.latest_tactician_output.get('market_timing', {}).get('volatility', 0.0)
            }
            
            # Decision reasoning
            if should_trade_final:
                reasoning = f"All systems aligned: HMM regime {regime_id} (conf: {hmm_confidence:.3f}), " \
                          f"Analyst green light (conf: {analyst_confidence:.3f}), " \
                          f"Tactician enter signal (conf: {tactician_confidence:.3f})"
            elif should_trade and not should_enter:
                reasoning = f"HMM and Analyst aligned but Tactician waiting: " \
                          f"Regime {regime_id}, Analyst conf: {analyst_confidence:.3f}, " \
                          f"Tactician conf: {tactician_confidence:.3f}"
            elif not should_trade:
                reasoning = f"Analyst red light: Regime {regime_id} (conf: {hmm_confidence:.3f}), " \
                          f"Analyst conf: {analyst_confidence:.3f}"
            else:
                reasoning = "Insufficient data or system misalignment"
            
            decision = TradingDecision(
                timestamp=datetime.now(),
                should_trade=should_trade_final,
                entry_confidence=overall_confidence,
                expected_return=expected_return,
                position_size=position_size,
                leverage=leverage,
                regime_id=regime_id,
                hmm_confidence=hmm_confidence,
                analyst_confidence=analyst_confidence,
                tactician_confidence=tactician_confidence,
                risk_score=risk_score,
                market_conditions=market_conditions,
                decision_reasoning=reasoning
            )
            
            # Store decision
            self.decision_history.append(decision)
            if len(self.decision_history) > self.max_history:
                self.decision_history = self.decision_history[-self.max_history:]
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Trading decision failed: {e}")
            self.metrics.error_count += 1
            return None
    
    @handles_errors
    def run_single_cycle(self) -> Optional[TradingDecision]:
        """Run a single analysis cycle."""
        start_time = time.time()
        
        # Run all systems
        self.run_hmm_analysis()
        self.run_analyst_analysis()
        self.run_tactician_analysis()
        
        # Make trading decision
        decision = self.make_trading_decision()
        
        # Update metrics
        processing_time = time.time() - start_time
        self.metrics.avg_processing_time = (
            (self.metrics.avg_processing_time * (self.metrics.hmm_runs + self.metrics.analyst_runs + self.metrics.tactician_runs - 1) + processing_time) /
            (self.metrics.hmm_runs + self.metrics.analyst_runs + self.metrics.tactician_runs)
        )
        
        if self.start_time:
            self.metrics.system_uptime = (datetime.now() - self.start_time).total_seconds()
        
        return decision
    
    @handles_errors
    def start_system(self) -> None:
        """Start the trading system."""
        if self.is_running:
            tprint("System already running")
            return
        
        if self.status != SystemStatus.RUNNING:
            tprint("System not trained, training first...")
            self.train_systems()
        
        self.is_running = True
        self.start_time = datetime.now()
        self.stop_event.clear()
        
        # Start execution thread
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        
        tprint("Multi-tier trading system started")
    
    def _execution_loop(self) -> None:
        """Main execution loop."""
        while self.is_running and not self.stop_event.is_set():
            try:
                decision = self.run_single_cycle()
                
                if decision and decision.should_trade:
                    tprint(f"TRADING DECISION: {decision.decision_reasoning}")
                
                # Sleep for 1 second before next cycle
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Execution loop error: {e}")
                self.metrics.error_count += 1
                time.sleep(5)  # Wait before retrying
    
    @handles_errors
    def stop_system(self) -> None:
        """Stop the trading system."""
        if not self.is_running:
            tprint("System not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)
        
        self.status = SystemStatus.STOPPED
        tprint("Multi-tier trading system stopped")
    
    @handles_errors
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'status': self.status.value,
            'is_running': self.is_running,
            'uptime': self.metrics.system_uptime,
            'metrics': {
                'hmm_runs': self.metrics.hmm_runs,
                'analyst_runs': self.metrics.analyst_runs,
                'tactician_runs': self.metrics.tactician_runs,
                'green_lights': self.metrics.green_lights,
                'trade_signals': self.metrics.trade_signals,
                'error_count': self.metrics.error_count,
                'avg_processing_time': self.metrics.avg_processing_time
            },
            'latest_outputs': {
                'hmm': self.latest_hmm_output is not None,
                'analyst': self.latest_analyst_output is not None,
                'tactician': self.latest_tactician_output is not None
            },
            'decision_history_count': len(self.decision_history)
        }
    
    @handles_errors
    def get_latest_decision(self) -> Optional[TradingDecision]:
        """Get the latest trading decision."""
        return self.decision_history[-1] if self.decision_history else None
    
    @handles_errors
    def get_decision_history(self, limit: int = 100) -> List[TradingDecision]:
        """Get recent trading decisions."""
        return self.decision_history[-limit:] if self.decision_history else []
    
    @handles_errors
    def save_models(self, base_path: str) -> None:
        """Save all trained models."""
        os.makedirs(base_path, exist_ok=True)
        
        self.hmm_system.save_models(os.path.join(base_path, 'hmm_models.pkl'))
        self.analyst_system.save_models(os.path.join(base_path, 'analyst_models.pkl'))
        self.tactician_system.save_models(os.path.join(base_path, 'tactician_models.pkl'))
        
        tprint(f"All models saved to {base_path}")
    
    @handles_errors
    def load_models(self, base_path: str) -> None:
        """Load all trained models."""
        self.hmm_system.load_models(os.path.join(base_path, 'hmm_models.pkl'))
        self.analyst_system.load_models(os.path.join(base_path, 'analyst_models.pkl'))
        self.tactician_system.load_models(os.path.join(base_path, 'tactician_models.pkl'))
        
        self.status = SystemStatus.RUNNING
        tprint(f"All models loaded from {base_path}")
    
    @handles_errors
    def export_metrics(self, filepath: str) -> None:
        """Export system metrics to JSON file."""
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'system_status': self.get_system_status(),
            'decision_history': [
                {
                    'timestamp': d.timestamp.isoformat(),
                    'should_trade': d.should_trade,
                    'entry_confidence': d.entry_confidence,
                    'expected_return': d.expected_return,
                    'position_size': d.position_size,
                    'leverage': d.leverage,
                    'regime_id': d.regime_id,
                    'decision_reasoning': d.decision_reasoning
                }
                for d in self.decision_history
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        tprint(f"Metrics exported to {filepath}")


# Factory function for easy instantiation
def create_multi_tier_trading_orchestrator(config: Optional[Dict[str, Any]] = None) -> MultiTierTradingOrchestrator:
    """Create and return a new multi-tier trading orchestrator instance."""
    return MultiTierTradingOrchestrator(config)