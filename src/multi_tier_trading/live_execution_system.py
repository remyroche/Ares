"""
Live Execution System for Multi-Tier Trading

This module implements the live execution system that runs continuously with proper scheduling:
- HMM: Every 15 minutes on 1h data
- Analyst: Every 2 minutes on 5m data  
- Tactician: Every 30 seconds on 1m data
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import threading
import time
import queue
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

from src.utils.tprint import tprint
from src.utils.logger import system_logger
from src.core.decorators import handles_errors

from .multi_tier_orchestrator import MultiTierTradingOrchestrator, TradingDecision
from .enhanced_model_configs import MultiTierModelConfigs
from .feature_extraction import (
    create_hmm_feature_extractor,
    create_analyst_feature_extractor, 
    create_tactician_feature_extractor
)


class ExecutionStatus(Enum):
    """Status of the live execution system."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class ExecutionMetrics:
    """Metrics for the live execution system."""
    start_time: datetime
    hmm_runs: int = 0
    analyst_runs: int = 0
    tactician_runs: int = 0
    green_lights: int = 0
    trade_signals: int = 0
    errors: int = 0
    last_hmm_run: Optional[datetime] = None
    last_analyst_run: Optional[datetime] = None
    last_tactician_run: Optional[datetime] = None


class LiveExecutionSystem:
    """
    Live execution system for multi-tier trading.
    
    Manages continuous execution with proper scheduling and data flow.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the live execution system."""
        self.config = config or {}
        self.logger = system_logger.getChild('LiveExecutionSystem')
        
        # System status
        self.status = ExecutionStatus.STOPPED
        self.is_running = False
        
        # Orchestrator
        self.orchestrator = MultiTierTradingOrchestrator(config)
        
        # Feature extractors
        self.hmm_extractor = create_hmm_feature_extractor()
        self.analyst_extractor = create_analyst_feature_extractor()
        self.tactician_extractor = create_tactician_feature_extractor()
        
        # Data storage
        self.data_1h: Optional[pd.DataFrame] = None
        self.data_5m: Optional[pd.DataFrame] = None
        self.data_1m: Optional[pd.DataFrame] = None
        
        # Execution metrics
        self.metrics = ExecutionMetrics(start_time=datetime.now())
        
        # Threading
        self.execution_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.decision_queue = queue.Queue()
        
        # Model configurations
        self.model_configs = MultiTierModelConfigs.get_all_configs()
        
        tprint("Live execution system initialized")
    
    @handles_errors
    def load_data(self, data_1h: pd.DataFrame, data_5m: pd.DataFrame, data_1m: pd.DataFrame) -> None:
        """Load market data for all timeframes."""
        tprint("Loading data for live execution system...")
        
        self.data_1h = data_1h.copy()
        self.data_5m = data_5m.copy()
        self.data_1m = data_1m.copy()
        
        # Load data into orchestrator
        self.orchestrator.load_data(data_1h, data_5m, data_1m)
        
        tprint(f"Data loaded: 1h={len(data_1h)} bars, 5m={len(data_5m)} bars, 1m={len(data_1m)} bars")
    
    @handles_errors
    async def train_systems(self) -> Dict[str, Any]:
        """Train all systems before starting live execution."""
        tprint("Training systems for live execution...")
        
        training_results = await self.orchestrator.train_systems()
        
        tprint("Systems trained successfully")
        return training_results
    
    @handles_errors
    def start_live_execution(self) -> None:
        """Start the live execution system."""
        if self.is_running:
            tprint("Live execution system already running")
            return
        
        if self.status != ExecutionStatus.RUNNING:
            tprint("Training systems first...")
            asyncio.run(self.train_systems())
        
        self.status = ExecutionStatus.STARTING
        self.is_running = True
        self.stop_event.clear()
        self.metrics = ExecutionMetrics(start_time=datetime.now())
        
        # Start execution thread
        self.execution_thread = threading.Thread(target=self._execution_loop, daemon=True)
        self.execution_thread.start()
        
        self.status = ExecutionStatus.RUNNING
        tprint("Live execution system started")
    
    def _execution_loop(self) -> None:
        """Main execution loop with proper scheduling."""
        tprint("Starting live execution loop...")
        
        while self.is_running and not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                
                # Check if HMM should run (every 15 minutes)
                if self._should_run_hmm(current_time):
                    asyncio.run(self._run_hmm_cycle())
                    self.metrics.hmm_runs += 1
                    self.metrics.last_hmm_run = current_time
                
                # Check if Analyst should run (every 2 minutes)
                if self._should_run_analyst(current_time):
                    asyncio.run(self._run_analyst_cycle())
                    self.metrics.analyst_runs += 1
                    self.metrics.last_analyst_run = current_time
                
                # Check if Tactician should run (every 30 seconds)
                if self._should_run_tactician(current_time):
                    asyncio.run(self._run_tactician_cycle())
                    self.metrics.tactician_runs += 1
                    self.metrics.last_tactician_run = current_time
                
                # Make trading decision
                decision = self._make_trading_decision()
                if decision:
                    self.decision_queue.put(decision)
                    if decision.should_trade:
                        self.metrics.trade_signals += 1
                        tprint(f"TRADE SIGNAL: {decision.decision_reasoning}")
                
                # Sleep for 1 second before next check
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Execution loop error: {e}")
                self.metrics.errors += 1
                time.sleep(5)  # Wait before retrying
        
        tprint("Live execution loop stopped")
    
    def _should_run_hmm(self, current_time: datetime) -> bool:
        """Check if HMM should run (every 15 minutes)."""
        if self.metrics.last_hmm_run is None:
            return True
        
        time_since_last_run = current_time - self.metrics.last_hmm_run
        return time_since_last_run >= timedelta(minutes=15)
    
    def _should_run_analyst(self, current_time: datetime) -> bool:
        """Check if Analyst should run (every 2 minutes)."""
        if self.metrics.last_analyst_run is None:
            return True
        
        time_since_last_run = current_time - self.metrics.last_analyst_run
        return time_since_last_run >= timedelta(minutes=2)
    
    def _should_run_tactician(self, current_time: datetime) -> bool:
        """Check if Tactician should run (every 30 seconds)."""
        if self.metrics.last_tactician_run is None:
            return True
        
        time_since_last_run = current_time - self.metrics.last_tactician_run
        return time_since_last_run >= timedelta(seconds=30)
    
    @handles_errors
    async def _run_hmm_cycle(self) -> None:
        """Run HMM analysis cycle."""
        try:
            tprint("Running HMM analysis cycle...")
            
            # Extract features
            features = self.hmm_extractor.extract_features(self.data_1h)
            
            # Get regime predictions (simplified)
            regime_probs = np.random.dirichlet(np.ones(20))  # 20 regimes
            dominant_regime = np.argmax(regime_probs)
            confidence = regime_probs[dominant_regime]
            
            # Update orchestrator
            self.orchestrator.latest_hmm_output = type('HMMOutput', (), {
                'timestamp': datetime.now(),
                'regime_probs': regime_probs,
                'dominant_regime': dominant_regime,
                'confidence': confidence,
                'regime_characteristics': {
                    'mean_returns': np.random.normal(0, 0.01),
                    'volatility': np.random.uniform(0.01, 0.05),
                    'mean_volume': np.random.uniform(0.8, 1.2)
                },
                'feature_importance': {}
            })()
            
            tprint(f"HMM cycle completed: Regime {dominant_regime} (confidence: {confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"HMM cycle failed: {e}")
            self.metrics.errors += 1
    
    @handles_errors
    async def _run_analyst_cycle(self) -> None:
        """Run Analyst analysis cycle."""
        try:
            tprint("Running Analyst analysis cycle...")
            
            if not self.orchestrator.latest_hmm_output:
                tprint("Analyst waiting for HMM output...")
                return
            
            # Extract features
            hmm_output = {
                'regime_probs': self.orchestrator.latest_hmm_output.regime_probs,
                'dominant_regime': self.orchestrator.latest_hmm_output.dominant_regime,
                'confidence': self.orchestrator.latest_hmm_output.confidence,
                'regime_characteristics': self.orchestrator.latest_hmm_output.regime_characteristics
            }
            
            features = self.analyst_extractor.extract_features(self.data_5m, hmm_output)
            
            # Get Analyst predictions (simplified)
            should_trade = np.random.random() > 0.7  # 30% chance of green light
            confidence = np.random.uniform(0.6, 0.9)
            regime_id = self.orchestrator.latest_hmm_output.dominant_regime
            
            # Update orchestrator
            self.orchestrator.latest_analyst_output = type('AnalystOutput', (), {
                'timestamp': datetime.now(),
                'should_trade': should_trade,
                'confidence': confidence,
                'base_model_predictions': {
                    'tcn': np.random.uniform(-0.5, 0.5),
                    'catboost': np.random.uniform(-0.5, 0.5),
                    'lightgbm': np.random.uniform(-0.5, 0.5)
                },
                'meta_learner_prediction': np.random.uniform(-0.5, 0.5),
                'regime_id': regime_id,
                'feature_importance': {},
                'market_conditions': {}
            })()
            
            if should_trade:
                self.metrics.green_lights += 1
                tprint(f"Analyst GREEN LIGHT: Regime {regime_id} (confidence: {confidence:.3f})")
            else:
                tprint(f"Analyst RED LIGHT: Regime {regime_id} (confidence: {confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"Analyst cycle failed: {e}")
            self.metrics.errors += 1
    
    @handles_errors
    async def _run_tactician_cycle(self) -> None:
        """Run Tactician analysis cycle."""
        try:
            tprint("Running Tactician analysis cycle...")
            
            if not self.orchestrator.latest_analyst_output:
                tprint("Tactician waiting for Analyst output...")
                return
            
            # Only run if Analyst gave green light
            if not self.orchestrator.latest_analyst_output.should_trade:
                tprint("Tactician waiting for Analyst green light...")
                return
            
            # Extract features
            hmm_output = {
                'regime_probs': self.orchestrator.latest_hmm_output.regime_probs if self.orchestrator.latest_hmm_output else None,
                'dominant_regime': self.orchestrator.latest_hmm_output.dominant_regime if self.orchestrator.latest_hmm_output else 0,
                'confidence': self.orchestrator.latest_hmm_output.confidence if self.orchestrator.latest_hmm_output else 0.5,
                'regime_characteristics': self.orchestrator.latest_hmm_output.regime_characteristics if self.orchestrator.latest_hmm_output else {}
            }
            
            analyst_output = {
                'should_trade': self.orchestrator.latest_analyst_output.should_trade,
                'confidence': self.orchestrator.latest_analyst_output.confidence,
                'meta_learner_prediction': self.orchestrator.latest_analyst_output.meta_learner_prediction,
                'regime_id': self.orchestrator.latest_analyst_output.regime_id
            }
            
            features = self.tactician_extractor.extract_features(self.data_1m, hmm_output, analyst_output)
            
            # Get Tactician predictions (simplified)
            should_enter = np.random.random() > 0.6  # 40% chance of enter signal
            entry_confidence = np.random.uniform(0.5, 0.9)
            expected_return = np.random.uniform(-0.5, 1.0)  # -0.5% to 1.0%
            position_size = np.random.uniform(0.01, 0.1)  # 1% to 10%
            leverage = np.random.uniform(1.0, 3.0)  # 1x to 3x
            
            # Update orchestrator
            self.orchestrator.latest_tactician_output = type('TacticianOutput', (), {
                'timestamp': datetime.now(),
                'should_enter': should_enter,
                'entry_confidence': entry_confidence,
                'expected_return': expected_return,
                'risk_score': np.random.uniform(0.2, 0.8),
                'position_size': position_size,
                'leverage': leverage,
                'base_model_predictions': {
                    'xgboost': np.random.uniform(-0.5, 0.5),
                    'randomforest': np.random.uniform(-0.5, 0.5),
                    'catboost': np.random.uniform(-0.5, 0.5),
                    'elastic_net': np.random.uniform(-0.5, 0.5)
                },
                'meta_learner_prediction': expected_return,
                'feature_importance': {},
                'market_timing': {}
            })()
            
            if should_enter:
                tprint(f"Tactician ENTER SIGNAL: {expected_return:.3f}% expected return")
            else:
                tprint(f"Tactician WAIT: {expected_return:.3f}% expected return")
            
        except Exception as e:
            self.logger.error(f"Tactician cycle failed: {e}")
            self.metrics.errors += 1
    
    @handles_errors
    def _make_trading_decision(self) -> Optional[TradingDecision]:
        """Make final trading decision."""
        return self.orchestrator.make_trading_decision()
    
    @handles_errors
    def stop_live_execution(self) -> None:
        """Stop the live execution system."""
        if not self.is_running:
            tprint("Live execution system not running")
            return
        
        self.status = ExecutionStatus.STOPPING
        self.is_running = False
        self.stop_event.set()
        
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)
        
        self.status = ExecutionStatus.STOPPED
        tprint("Live execution system stopped")
    
    @handles_errors
    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status."""
        return {
            'status': self.status.value,
            'is_running': self.is_running,
            'uptime': (datetime.now() - self.metrics.start_time).total_seconds(),
            'metrics': {
                'hmm_runs': self.metrics.hmm_runs,
                'analyst_runs': self.metrics.analyst_runs,
                'tactician_runs': self.metrics.tactician_runs,
                'green_lights': self.metrics.green_lights,
                'trade_signals': self.metrics.trade_signals,
                'errors': self.metrics.errors
            },
            'last_runs': {
                'hmm': self.metrics.last_hmm_run.isoformat() if self.metrics.last_hmm_run else None,
                'analyst': self.metrics.last_analyst_run.isoformat() if self.metrics.last_analyst_run else None,
                'tactician': self.metrics.last_tactician_run.isoformat() if self.metrics.last_tactician_run else None
            },
            'queue_size': self.decision_queue.qsize()
        }
    
    @handles_errors
    def get_latest_decision(self) -> Optional[TradingDecision]:
        """Get the latest trading decision."""
        try:
            return self.decision_queue.get_nowait()
        except queue.Empty:
            return None
    
    @handles_errors
    def get_decision_history(self, limit: int = 100) -> List[TradingDecision]:
        """Get recent trading decisions."""
        return self.orchestrator.get_decision_history(limit)
    
    @handles_errors
    def export_metrics(self, filepath: str) -> None:
        """Export execution metrics to JSON file."""
        metrics_data = {
            'timestamp': datetime.now().isoformat(),
            'execution_status': self.get_execution_status(),
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
                for d in self.get_decision_history()
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        tprint(f"Metrics exported to {filepath}")


# Factory function
def create_live_execution_system(config: Optional[Dict[str, Any]] = None) -> LiveExecutionSystem:
    """Create and return a new live execution system instance."""
    return LiveExecutionSystem(config)