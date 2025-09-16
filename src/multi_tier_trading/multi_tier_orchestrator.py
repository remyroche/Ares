"""
Multi-Tier Trading System Orchestrator

This module implements the complete multi-tier trading system as specified:
- HMM (1h base, runs every 15 minutes, 100 features, 15-25 regimes)
- Analyst (5m base, runs every 2 minutes, 300+ features, per-regime training)
- Tactician (1m base, runs every 30 seconds, green light dependent)
"""

import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

from src.utils.tprint import tprint
from src.utils.logger import system_logger
from src.core.decorators import handles_errors

# Import the existing training systems
from src.training.steps.market_analysis.hmm_models_training import HMMEnsembleTrainingComponent
from src.training.steps.model_training.analyst_ensemble_training import AnalystEnsembleTrainingStep
from src.training.steps.model_training.tactician_ensemble_training import TacticianEnsembleTrainingStep


class SystemStatus(Enum):
    """Status of the multi-tier trading system."""
    INITIALIZING = "initializing"
    TRAINING = "training"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


@dataclass
class HMMOutput:
    """HMM system output."""
    timestamp: datetime
    regime_probs: np.ndarray
    dominant_regime: int
    confidence: float
    regime_characteristics: Dict[str, Any]
    feature_importance: Dict[str, float]


@dataclass
class AnalystOutput:
    """Analyst system output."""
    timestamp: datetime
    should_trade: bool
    confidence: float
    base_model_predictions: Dict[str, float]
    meta_learner_prediction: float
    regime_id: int
    feature_importance: Dict[str, float]
    market_conditions: Dict[str, Any]


@dataclass
class TacticianOutput:
    """Tactician system output."""
    timestamp: datetime
    should_enter: bool
    entry_confidence: float
    expected_return: float
    risk_score: float
    position_size: float
    leverage: float
    base_model_predictions: Dict[str, float]
    meta_learner_prediction: float
    feature_importance: Dict[str, float]
    market_timing: Dict[str, Any]


@dataclass
class TradingDecision:
    """Final trading decision."""
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
    decision_reasoning: str


class MultiTierTradingOrchestrator:
    """
    Multi-tier trading system orchestrator.
    
    Coordinates:
    - HMM system (1h base, runs every 15 minutes)
    - Analyst system (5m base, runs every 2 minutes) 
    - Tactician system (1m base, runs every 30 seconds)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the multi-tier trading orchestrator."""
        self.config = config or {}
        self.logger = system_logger.getChild('MultiTierTradingOrchestrator')
        
        # System status
        self.status = SystemStatus.INITIALIZING
        self.is_running = False
        self.start_time = None
        
        # Data storage
        self.data_1h: Optional[pd.DataFrame] = None
        self.data_5m: Optional[pd.DataFrame] = None
        self.data_1m: Optional[pd.DataFrame] = None
        
        # System outputs
        self.latest_hmm_output: Optional[HMMOutput] = None
        self.latest_analyst_output: Optional[AnalystOutput] = None
        self.latest_tactician_output: Optional[TacticianOutput] = None
        
        # Training systems
        self.hmm_system: Optional[HMMEnsembleTrainingComponent] = None
        self.analyst_system: Optional[AnalystEnsembleTrainingStep] = None
        self.tactician_system: Optional[TacticianEnsembleTrainingStep] = None
        
        # Scheduling
        self.hmm_last_run: Optional[datetime] = None
        self.analyst_last_run: Optional[datetime] = None
        self.tactician_last_run: Optional[datetime] = None
        
        # Decision history
        self.decision_history: List[TradingDecision] = []
        self.max_history = self.config.get('max_history', 1000)
        
        # Threading
        self.execution_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        tprint("Multi-tier trading orchestrator initialized")
    
    @handles_errors
    def load_data(self, data_1h: pd.DataFrame, data_5m: pd.DataFrame, data_1m: pd.DataFrame) -> None:
        """Load market data for all timeframes."""
        tprint("Loading market data for multi-tier system...")
        
        # Validate data
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        for timeframe, data in [('1h', data_1h), ('5m', data_5m), ('1m', data_1m)]:
            if not all(col in data.columns for col in required_columns):
                raise ValueError(f"Missing required columns in {timeframe} data")
        
        self.data_1h = data_1h.copy()
        self.data_5m = data_5m.copy()
        self.data_1m = data_1m.copy()
        
        tprint(f"Data loaded: 1h={len(data_1h)} bars, 5m={len(data_5m)} bars, 1m={len(data_1m)} bars")
    
    @handles_errors
    async def train_systems(self) -> Dict[str, Any]:
        """Train all three systems."""
        tprint("Training multi-tier trading systems...")
        self.status = SystemStatus.TRAINING
        
        training_results = {}
        
        try:
            # Train HMM system
            tprint("Training HMM regime detection system...")
            self.hmm_system = HMMEnsembleTrainingComponent()
            hmm_results = await self.hmm_system.train_models(
                symbol="ETHUSDT",
                exchange="binance", 
                timeframe="1h",
                data_dir="./data"
            )
            training_results['hmm'] = hmm_results
            
            # Train Analyst system
            tprint("Training Analyst regime prediction system...")
            self.analyst_system = AnalystEnsembleTrainingStep()
            analyst_results = await self.analyst_system.train_models(
                symbol="ETHUSDT",
                exchange="binance",
                timeframe="5m", 
                data_dir="./data"
            )
            training_results['analyst'] = analyst_results
            
            # Train Tactician system
            tprint("Training Tactician timing prediction system...")
            self.tactician_system = TacticianEnsembleTrainingStep()
            tactician_results = await self.tactician_system.train_models(
                symbol="ETHUSDT",
                exchange="binance",
                timeframe="1m",
                data_dir="./data"
            )
            training_results['tactician'] = tactician_results
            
            self.status = SystemStatus.RUNNING
            tprint("All systems trained successfully")
            
        except Exception as e:
            self.status = SystemStatus.ERROR
            self.logger.error(f"Training failed: {e}")
            raise
        
        return training_results
    
    @handles_errors
    def should_run_hmm(self) -> bool:
        """Check if HMM should run (every 15 minutes)."""
        if self.hmm_last_run is None:
            return True
        
        time_since_last_run = datetime.now() - self.hmm_last_run
        return time_since_last_run >= timedelta(minutes=15)
    
    @handles_errors
    def should_run_analyst(self) -> bool:
        """Check if Analyst should run (every 2 minutes)."""
        if self.analyst_last_run is None:
            return True
        
        time_since_last_run = datetime.now() - self.analyst_last_run
        return time_since_last_run >= timedelta(minutes=2)
    
    @handles_errors
    def should_run_tactician(self) -> bool:
        """Check if Tactician should run (every 30 seconds)."""
        if self.tactician_last_run is None:
            return True
        
        time_since_last_run = datetime.now() - self.tactician_last_run
        return time_since_last_run >= timedelta(seconds=30)
    
    @handles_errors
    async def run_hmm_analysis(self) -> Optional[HMMOutput]:
        """Run HMM analysis if it's time."""
        if not self.should_run_hmm():
            return None
        
        if not self.hmm_system or not self.data_1h:
            return None
        
        try:
            # Extract features for HMM
            features = self._extract_hmm_features(self.data_1h)
            
            # Get regime predictions
            regime_probs = self.hmm_system.predict_regime_probabilities(features)
            dominant_regime = np.argmax(regime_probs)
            confidence = regime_probs[dominant_regime]
            
            # Get regime characteristics
            regime_characteristics = self._get_regime_characteristics(dominant_regime)
            
            # Create output
            output = HMMOutput(
                timestamp=datetime.now(),
                regime_probs=regime_probs,
                dominant_regime=dominant_regime,
                confidence=confidence,
                regime_characteristics=regime_characteristics,
                feature_importance={}  # Would be populated by actual HMM system
            )
            
            self.latest_hmm_output = output
            self.hmm_last_run = datetime.now()
            
            tprint(f"HMM analysis completed: Regime {dominant_regime} (confidence: {confidence:.3f})")
            return output
            
        except Exception as e:
            self.logger.error(f"HMM analysis failed: {e}")
            return None
    
    @handles_errors
    async def run_analyst_analysis(self) -> Optional[AnalystOutput]:
        """Run Analyst analysis if it's time."""
        if not self.should_run_analyst():
            return None
        
        if not self.analyst_system or not self.data_5m or not self.latest_hmm_output:
            return None
        
        try:
            # Extract features for Analyst
            features = self._extract_analyst_features(self.data_5m, self.latest_hmm_output)
            
            # Get regime ID from HMM
            regime_id = self.latest_hmm_output.dominant_regime
            
            # Get Analyst predictions
            should_trade, confidence, predictions = self.analyst_system.predict_trading_opportunity(
                features, regime_id
            )
            
            # Create output
            output = AnalystOutput(
                timestamp=datetime.now(),
                should_trade=should_trade,
                confidence=confidence,
                base_model_predictions=predictions,
                meta_learner_prediction=sum(predictions.values()) / len(predictions),
                regime_id=regime_id,
                feature_importance={},  # Would be populated by actual system
                market_conditions={}    # Would be populated by actual system
            )
            
            self.latest_analyst_output = output
            self.analyst_last_run = datetime.now()
            
            status = "GREEN LIGHT" if should_trade else "RED LIGHT"
            tprint(f"Analyst analysis completed: {status} (confidence: {confidence:.3f})")
            return output
            
        except Exception as e:
            self.logger.error(f"Analyst analysis failed: {e}")
            return None
    
    @handles_errors
    async def run_tactician_analysis(self) -> Optional[TacticianOutput]:
        """Run Tactician analysis if it's time."""
        if not self.should_run_tactician():
            return None
        
        if not self.tactician_system or not self.data_1m or not self.latest_analyst_output:
            return None
        
        # Only run if Analyst gave green light
        if not self.latest_analyst_output.should_trade:
            return None
        
        try:
            # Extract features for Tactician
            features = self._extract_tactician_features(
                self.data_1m, 
                self.latest_hmm_output, 
                self.latest_analyst_output
            )
            
            # Get Tactician predictions
            should_enter, confidence, expected_return, position_size, leverage = self.tactician_system.predict_entry_timing(
                features
            )
            
            # Create output
            output = TacticianOutput(
                timestamp=datetime.now(),
                should_enter=should_enter,
                entry_confidence=confidence,
                expected_return=expected_return,
                risk_score=0.5,  # Would be calculated by actual system
                position_size=position_size,
                leverage=leverage,
                base_model_predictions={},  # Would be populated by actual system
                meta_learner_prediction=expected_return,
                feature_importance={},  # Would be populated by actual system
                market_timing={}        # Would be populated by actual system
            )
            
            self.latest_tactician_output = output
            self.tactician_last_run = datetime.now()
            
            status = "ENTER" if should_enter else "WAIT"
            tprint(f"Tactician analysis completed: {status} (expected return: {expected_return:.3f}%)")
            return output
            
        except Exception as e:
            self.logger.error(f"Tactician analysis failed: {e}")
            return None
    
    @handles_errors
    def make_trading_decision(self) -> Optional[TradingDecision]:
        """Make final trading decision based on all system outputs."""
        if not all([self.latest_hmm_output, self.latest_analyst_output, self.latest_tactician_output]):
            return None
        
        try:
            # Extract key information
            hmm_confidence = self.latest_hmm_output.confidence
            analyst_confidence = self.latest_analyst_output.confidence
            tactician_confidence = self.latest_tactician_output.entry_confidence
            
            regime_id = self.latest_hmm_output.dominant_regime
            should_trade = self.latest_analyst_output.should_trade
            should_enter = self.latest_tactician_output.should_enter
            
            # Final decision logic
            should_trade_final = should_trade and should_enter
            
            # Calculate overall confidence
            overall_confidence = (hmm_confidence + analyst_confidence + tactician_confidence) / 3
            
            # Get position sizing
            position_size = self.latest_tactician_output.position_size
            leverage = self.latest_tactician_output.leverage
            expected_return = self.latest_tactician_output.expected_return
            risk_score = self.latest_tactician_output.risk_score
            
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
                decision_reasoning=reasoning
            )
            
            # Store decision
            self.decision_history.append(decision)
            if len(self.decision_history) > self.max_history:
                self.decision_history = self.decision_history[-self.max_history:]
            
            return decision
            
        except Exception as e:
            self.logger.error(f"Trading decision failed: {e}")
            return None
    
    @handles_errors
    async def run_single_cycle(self) -> Optional[TradingDecision]:
        """Run a single analysis cycle."""
        # Run all systems
        await self.run_hmm_analysis()
        await self.run_analyst_analysis()
        await self.run_tactician_analysis()
        
        # Make trading decision
        decision = self.make_trading_decision()
        
        return decision
    
    @handles_errors
    def start_system(self) -> None:
        """Start the multi-tier trading system."""
        if self.is_running:
            tprint("System already running")
            return
        
        if self.status != SystemStatus.RUNNING:
            tprint("System not trained, training first...")
            asyncio.run(self.train_systems())
        
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
                decision = asyncio.run(self.run_single_cycle())
                
                if decision and decision.should_trade:
                    tprint(f"TRADING DECISION: {decision.decision_reasoning}")
                
                # Sleep for 1 second before next cycle
                time.sleep(1)
                
            except Exception as e:
                self.logger.error(f"Execution loop error: {e}")
                time.sleep(5)  # Wait before retrying
    
    @handles_errors
    def stop_system(self) -> None:
        """Stop the multi-tier trading system."""
        if not self.is_running:
            tprint("System not running")
            return
        
        self.is_running = False
        self.stop_event.set()
        
        if self.execution_thread and self.execution_thread.is_alive():
            self.execution_thread.join(timeout=5)
        
        self.status = SystemStatus.STOPPED
        tprint("Multi-tier trading system stopped")
    
    def _extract_hmm_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract 100 features for HMM analysis."""
        # Simplified feature extraction - would be implemented by actual HMM system
        features = []
        
        # Price features
        features.extend([
            data['close'].pct_change().values,
            data['high'].pct_change().values,
            data['low'].pct_change().values,
            data['volume'].pct_change().values
        ])
        
        # Technical indicators (simplified)
        for period in [5, 10, 20, 50]:
            sma = data['close'].rolling(period).mean()
            features.append(sma.pct_change().values)
            features.append((data['close'] - sma).values)
        
        # Combine and pad to 100 features
        combined = np.concatenate(features)
        if len(combined) > 100:
            combined = combined[:100]
        else:
            combined = np.pad(combined, (0, 100 - len(combined)), 'constant')
        
        return combined.reshape(1, -1)
    
    def _extract_analyst_features(self, data: pd.DataFrame, hmm_output: HMMOutput) -> np.ndarray:
        """Extract 300+ features for Analyst analysis."""
        # Simplified feature extraction - would be implemented by actual Analyst system
        features = []
        
        # Price features
        features.extend([
            data['close'].pct_change().values,
            data['high'].pct_change().values,
            data['low'].pct_change().values,
            data['volume'].pct_change().values
        ])
        
        # HMM integration
        features.append(np.full(len(data), hmm_output.dominant_regime))
        features.append(np.full(len(data), hmm_output.confidence))
        
        # Technical indicators (simplified)
        for period in [5, 10, 20, 50, 100]:
            sma = data['close'].rolling(period).mean()
            features.append(sma.pct_change().values)
            features.append((data['close'] - sma).values)
        
        # Combine and pad to 300+ features
        combined = np.concatenate(features)
        if len(combined) > 300:
            combined = combined[:300]
        else:
            combined = np.pad(combined, (0, 300 - len(combined)), 'constant')
        
        return combined.reshape(1, -1)
    
    def _extract_tactician_features(self, data: pd.DataFrame, hmm_output: Optional[HMMOutput], 
                                  analyst_output: Optional[AnalystOutput]) -> np.ndarray:
        """Extract 50+ features for Tactician analysis."""
        # Simplified feature extraction - would be implemented by actual Tactician system
        features = []
        
        # Price features
        features.extend([
            data['close'].pct_change().values,
            data['high'].pct_change().values,
            data['low'].pct_change().values,
            data['volume'].pct_change().values
        ])
        
        # HMM integration
        if hmm_output:
            features.append(np.full(len(data), hmm_output.dominant_regime))
            features.append(np.full(len(data), hmm_output.confidence))
        
        # Analyst integration
        if analyst_output:
            features.append(np.full(len(data), analyst_output.should_trade))
            features.append(np.full(len(data), analyst_output.confidence))
        
        # Technical indicators (simplified)
        for period in [2, 5, 10, 20]:
            sma = data['close'].rolling(period).mean()
            features.append(sma.pct_change().values)
            features.append((data['close'] - sma).values)
        
        # Combine and pad to 50+ features
        combined = np.concatenate(features)
        if len(combined) > 50:
            combined = combined[:50]
        else:
            combined = np.pad(combined, (0, 50 - len(combined)), 'constant')
        
        return combined.reshape(1, -1)
    
    def _get_regime_characteristics(self, regime_id: int) -> Dict[str, Any]:
        """Get characteristics for a specific regime."""
        # Simplified regime characteristics - would be calculated by actual HMM system
        return {
            'mean_returns': np.random.normal(0, 0.01),
            'volatility': np.random.uniform(0.01, 0.05),
            'mean_volume': np.random.uniform(0.8, 1.2),
            'frequency': np.random.uniform(0.1, 0.3)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'status': self.status.value,
            'is_running': self.is_running,
            'uptime': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'latest_outputs': {
                'hmm': self.latest_hmm_output is not None,
                'analyst': self.latest_analyst_output is not None,
                'tactician': self.latest_tactician_output is not None
            },
            'decision_history_count': len(self.decision_history)
        }
    
    def get_latest_decision(self) -> Optional[TradingDecision]:
        """Get the latest trading decision."""
        return self.decision_history[-1] if self.decision_history else None
    
    def get_decision_history(self, limit: int = 100) -> List[TradingDecision]:
        """Get recent trading decisions."""
        return self.decision_history[-limit:] if self.decision_history else []


# Factory function
def create_multi_tier_trading_orchestrator(config: Optional[Dict[str, Any]] = None) -> MultiTierTradingOrchestrator:
    """Create and return a new multi-tier trading orchestrator instance."""
    return MultiTierTradingOrchestrator(config)