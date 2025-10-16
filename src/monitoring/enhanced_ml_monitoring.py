#!/usr/bin/env python3
"""
Enhanced ML Monitoring System

Comprehensive monitoring for ML models and ensembles with detailed explanations
using SHAP/LIME for trade decisions across backtesting, paper trading, and live trading.
"""

import json
import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

from src.utils.logger import system_logger
from src.core.decorators import handles_errors

# SHAP and LIME analyzers will be imported from shap_lime_integration
# from .shap_lime_integration import SHAPAnalyzer, LIMEAnalyzer

class TradingMode(Enum):
    """Trading execution modes."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

class ModelType(Enum):
    """ML model types."""
    HMM = "hmm"
    ANALYST = "analyst"
    TACTICIAN = "tactician"
    ENSEMBLE = "ensemble"
    TRANSITION = "transition"
    SR_PREDICTOR = "sr_predictor"

@dataclass
class HMMRegimeInfo:
    """HMM regime information for trade decisions."""
    regime_id: str
    regime_name: str
    regime_probability: float
    regime_transition_probability: float
    regime_duration: int  # Number of periods in current regime
    regime_stability_score: float
    next_regime_probabilities: Dict[str, float] = None

@dataclass
class TradeContext:
    """Context information for each trade decision."""
    exchange: str
    token: str
    timestamp: datetime
    price: float
    volume: float
    timeframe: str
    regime: Optional[str] = None
    hmm_regime_info: Optional[HMMRegimeInfo] = None
    market_conditions: Optional[Dict[str, Any]] = None

@dataclass
class TradingIndicator:
    """Trading indicators and their weights."""
    name: str
    value: float
    weight: float
    confidence: float
    risk_score: float
    description: str

@dataclass
class MLModelDecision:
    """Individual ML model decision details."""
    model_id: str
    model_type: ModelType
    prediction: float
    confidence: float
    risk_score: float
    feature_importance: Dict[str, float]
    shap_values: Optional[Dict[str, float]] = None
    lime_explanation: Optional[Dict[str, Any]] = None
    processing_time_ms: float = 0.0
    model_version: str = "unknown"

@dataclass
class EnsembleDecision:
    """Ensemble decision with per-model details."""
    ensemble_id: str
    final_prediction: float
    final_confidence: float
    final_risk_score: float
    model_weights: Dict[str, float]
    model_decisions: List[MLModelDecision]
    voting_mechanism: str
    consensus_score: float
    disagreement_level: float

@dataclass
class TradeDecision:
    """Complete trade decision with all context and explanations."""
    decision_id: str
    context: TradeContext
    trading_mode: TradingMode
    timestamp: datetime
    
    # Trading indicators
    trading_indicators: List[TradingIndicator]
    overall_confidence: float
    overall_risk_score: float
    
    # Ensemble decision
    ensemble_decision: EnsembleDecision
    
    # Final decision
    action: str  # "buy", "sell", "hold"
    position_size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Performance tracking
    execution_time_ms: float = 0.0
    success_metrics: Optional[Dict[str, float]] = None

@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for individual models."""
    model_id: str
    model_type: ModelType
    timestamp: datetime
    
    # Accuracy metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_score: Optional[float] = None
    
    # Trading performance
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Model stability
    prediction_confidence_std: float
    feature_importance_stability: float
    concept_drift_score: float
    data_drift_score: float

@dataclass
class EnsemblePerformanceMetrics:
    """Performance metrics for ensembles."""
    ensemble_id: str
    timestamp: datetime
    
    # Overall performance
    accuracy: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    
    # Ensemble-specific metrics
    model_diversity_score: float
    consensus_quality: float
    disagreement_impact: float
    weight_stability: float
    
    # Individual model contributions
    model_contributions: Dict[str, float]

class EnhancedMLMonitor:
    """
    Enhanced ML monitoring system with comprehensive tracking and explanations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the enhanced ML monitor."""
        self.config = config
        self.logger = system_logger.getChild("EnhancedMLMonitor")
        
        # Configuration
        self.monitor_config = config.get("enhanced_ml_monitoring", {})
        self.enable_shap = self.monitor_config.get("enable_shap", True)
        self.enable_lime = self.monitor_config.get("enable_lime", True)
        self.csv_export_interval = self.monitor_config.get("csv_export_interval_days", 30)
        self.max_decisions_in_memory = self.monitor_config.get("max_decisions_in_memory", 10000)
        
        # Storage
        self.trade_decisions: List[TradeDecision] = []
        self.model_performances: List[ModelPerformanceMetrics] = []
        self.ensemble_performances: List[EnsemblePerformanceMetrics] = []
        
        # Export paths
        self.export_dir = Path(self.monitor_config.get("export_directory", "monitoring_exports"))
        self.export_dir.mkdir(exist_ok = True)
        
        # Initialize explainability tools
        self._initialize_explainability_tools()
        
        # Performance tracking
        self.start_time = datetime.now()
        self.decision_count = 0
        
        self.logger.info("Enhanced ML Monitor initialized")
    
    def _initialize_explainability_tools(self):
        """Initialize SHAP and LIME analyzers."""
        try:
            # Import SHAP and LIME analyzers
            from .shap_lime_integration import SHAPAnalyzer, LIMEAnalyzer
            
            self.shap_analyzer = SHAPAnalyzer(self.config) if self.enable_shap else None
            self.lime_analyzer = LIMEAnalyzer(self.config) if self.enable_lime else None
            
            self.logger.info("Explainability tools initialized")
        except ImportError as e:
            self.logger.warning(f"Could not initialize explainability tools: {e}")
            self.shap_analyzer = None
            self.lime_analyzer = None
    
    @handles_errors(default_return = None, context="enhanced_ml_monitor.record_trade_decision")
    async def record_trade_decision(self, decision: TradeDecision) -> None:
        """Record a complete trade decision with all context and explanations."""
        try:
            # Add to memory storage
            self.trade_decisions.append(decision)
            self.decision_count += 1
            
            # Maintain memory limit
            if len(self.trade_decisions) > self.max_decisions_in_memory:
                self.trade_decisions = self.trade_decisions[-self.max_decisions_in_memory:]
            
            # Log decision summary
            self.logger.info(
                f"Recorded trade decision {decision.decision_id}: "
                f"{decision.action} {decision.context.token} at {decision.context.price} "
                f"(confidence: {decision.overall_confidence:.3f}, risk: {decision.overall_risk_score:.3f})"
            )
            
            # Check if we need to export
            await self._check_and_export_if_needed()
            
        except Exception as e:
            self.logger.error(f"Error recording trade decision: {e}")
    
    @handles_errors(default_return = None, context="enhanced_ml_monitor.record_model_performance")
    async def record_model_performance(self, performance: ModelPerformanceMetrics) -> None:
        """Record model performance metrics."""
        try:
            self.model_performances.append(performance)
            
            # Maintain memory limit
            if len(self.model_performances) > self.max_decisions_in_memory:
                self.model_performances = self.model_performances[-self.max_decisions_in_memory:]
            
            self.logger.info(
                f"Recorded performance for model {performance.model_id}: "
                f"accuracy={performance.accuracy:.3f}, win_rate={performance.win_rate:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error recording model performance: {e}")
    
    @handles_errors(default_return = None, context="enhanced_ml_monitor.record_ensemble_performance")
    async def record_ensemble_performance(self, performance: EnsemblePerformanceMetrics) -> None:
        """Record ensemble performance metrics."""
        try:
            self.ensemble_performances.append(performance)
            
            # Maintain memory limit
            if len(self.ensemble_performances) > self.max_decisions_in_memory:
                self.ensemble_performances = self.ensemble_performances[-self.max_decisions_in_memory:]
            
            self.logger.info(
                f"Recorded performance for ensemble {performance.ensemble_id}: "
                f"accuracy={performance.accuracy:.3f}, diversity={performance.model_diversity_score:.3f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error recording ensemble performance: {e}")
    
    async def _check_and_export_if_needed(self) -> None:
        """Check if it's time to export data to CSV."""
        try:
            # Check if we have enough data and it's been long enough
            if (len(self.trade_decisions) > 0 and 
                (datetime.now() - self.start_time).days >= self.csv_export_interval):
                
                await self.export_monthly_report()
                self.start_time = datetime.now()  # Reset timer
                
        except Exception as e:
            self.logger.error(f"Error checking export timing: {e}")
    
    @handles_errors(default_return = False, context="enhanced_ml_monitor.export_monthly_report")
    async def export_monthly_report(self) -> bool:
        """Export comprehensive monthly monitoring report to CSV."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export trade decisions
            if self.trade_decisions:
                decisions_df = self._create_decisions_dataframe()
                decisions_path = self.export_dir / f"trade_decisions_{timestamp}.csv"
                decisions_df.to_csv(decisions_path, index = False)
                self.logger.info(f"Exported {len(decisions_df)} trade decisions to {decisions_path}")
            
            # Export model performances
            if self.model_performances:
                models_df = self._create_model_performance_dataframe()
                models_path = self.export_dir / f"model_performances_{timestamp}.csv"
                models_df.to_csv(models_path, index = False)
                self.logger.info(f"Exported {len(models_df)} model performances to {models_path}")
            
            # Export ensemble performances
            if self.ensemble_performances:
                ensembles_df = self._create_ensemble_performance_dataframe()
                ensembles_path = self.export_dir / f"ensemble_performances_{timestamp}.csv"
                ensembles_df.to_csv(ensembles_path, index = False)
                self.logger.info(f"Exported {len(ensembles_df)} ensemble performances to {ensembles_path}")
            
            # Create summary report
            await self._create_summary_report(timestamp)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting monthly report: {e}")
            return False
    
    def _create_decisions_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from trade decisions."""
        data = []
        
        for decision in self.trade_decisions:
            # Base decision data
            row = {
                'decision_id': decision.decision_id,
                'timestamp': decision.timestamp.isoformat(),
                'trading_mode': decision.trading_mode.value,
                'exchange': decision.context.exchange,
                'token': decision.context.token,
                'price': decision.context.price,
                'volume': decision.context.volume,
                'timeframe': decision.context.timeframe,
                'regime': decision.context.regime,
                'action': decision.action,
                'position_size': decision.position_size,
                'stop_loss': decision.stop_loss,
                'take_profit': decision.take_profit,
                'overall_confidence': decision.overall_confidence,
                'overall_risk_score': decision.overall_risk_score,
                'execution_time_ms': decision.execution_time_ms,
            }
            
            # Ensemble decision data
            ensemble = decision.ensemble_decision
            row.update({
                'ensemble_id': ensemble.ensemble_id,
                'final_prediction': ensemble.final_prediction,
                'final_confidence': ensemble.final_confidence,
                'final_risk_score': ensemble.final_risk_score,
                'voting_mechanism': ensemble.voting_mechanism,
                'consensus_score': ensemble.consensus_score,
                'disagreement_level': ensemble.disagreement_level,
            })
            
            # Model weights
            for model_id, weight in ensemble.model_weights.items():
                row[f'model_weight_{model_id}'] = weight
            
            # Trading indicators
            for i, indicator in enumerate(decision.trading_indicators):
                row[f'indicator_{i}_name'] = indicator.name
                row[f'indicator_{i}_value'] = indicator.value
                row[f'indicator_{i}_weight'] = indicator.weight
                row[f'indicator_{i}_confidence'] = indicator.confidence
                row[f'indicator_{i}_risk'] = indicator.risk_score
            
            # Model decisions
            for i, model_decision in enumerate(ensemble.model_decisions):
                row[f'model_{i}_id'] = model_decision.model_id
                row[f'model_{i}_type'] = model_decision.model_type.value
                row[f'model_{i}_prediction'] = model_decision.prediction
                row[f'model_{i}_confidence'] = model_decision.confidence
                row[f'model_{i}_risk'] = model_decision.risk_score
                row[f'model_{i}_processing_time_ms'] = model_decision.processing_time_ms
                row[f'model_{i}_version'] = model_decision.model_version
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_model_performance_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from model performance metrics."""
        data = []
        
        for perf in self.model_performances:
            row = {
                'model_id': perf.model_id,
                'model_type': perf.model_type.value,
                'timestamp': perf.timestamp.isoformat(),
                'accuracy': perf.accuracy,
                'precision': perf.precision,
                'recall': perf.recall,
                'f1_score': perf.f1_score,
                'auc_score': perf.auc_score,
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'sharpe_ratio': perf.sharpe_ratio,
                'max_drawdown': perf.max_drawdown,
                'prediction_confidence_std': perf.prediction_confidence_std,
                'feature_importance_stability': perf.feature_importance_stability,
                'concept_drift_score': perf.concept_drift_score,
                'data_drift_score': perf.data_drift_score,
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_ensemble_performance_dataframe(self) -> pd.DataFrame:
        """Create DataFrame from ensemble performance metrics."""
        data = []
        
        for perf in self.ensemble_performances:
            row = {
                'ensemble_id': perf.ensemble_id,
                'timestamp': perf.timestamp.isoformat(),
                'accuracy': perf.accuracy,
                'win_rate': perf.win_rate,
                'profit_factor': perf.profit_factor,
                'sharpe_ratio': perf.sharpe_ratio,
                'model_diversity_score': perf.model_diversity_score,
                'consensus_quality': perf.consensus_quality,
                'disagreement_impact': perf.disagreement_impact,
                'weight_stability': perf.weight_stability,
            }
            
            # Model contributions
            for model_id, contribution in perf.model_contributions.items():
                row[f'contribution_{model_id}'] = contribution
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    async def _create_summary_report(self, timestamp: str) -> None:
        """Create a summary report with key metrics."""
        try:
            summary = {
                'report_timestamp': timestamp,
                'total_decisions': len(self.trade_decisions),
                'total_models_tracked': len(set(p.model_id for p in self.model_performances)),
                'total_ensembles_tracked': len(set(p.ensemble_id for p in self.ensemble_performances)),
                'monitoring_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            }
            
            # Trading mode distribution
            if self.trade_decisions:
                mode_counts = {}
                for decision in self.trade_decisions:
                    mode = decision.trading_mode.value
                    mode_counts[mode] = mode_counts.get(mode, 0) + 1
                summary['trading_mode_distribution'] = mode_counts
            
            # Average performance metrics
            if self.model_performances:
                avg_accuracy = np.mean([p.accuracy for p in self.model_performances])
                avg_win_rate = np.mean([p.win_rate for p in self.model_performances])
                summary['average_model_accuracy'] = avg_accuracy
                summary['average_model_win_rate'] = avg_win_rate
            
            # Save summary
            summary_path = self.export_dir / f"monitoring_summary_{timestamp}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent = 2, default = str)
            
            self.logger.info(f"Created monitoring summary report: {summary_path}")
            
        except Exception as e:
            self.logger.error(f"Error creating summary report: {e}")
    
    @handles_errors(default_return = None, context="enhanced_ml_monitor.get_model_explanations")
    async def get_model_explanations(self, model_id: str, features: np.ndarray, 
                                model: Any) -> Dict[str, Any]:
        """Get SHAP and LIME explanations for a model prediction."""
        explanations = {}
        
        try:
            # SHAP explanations
            if self.shap_analyzer and self.shap_analyzer.shap_available:
                shap_explanations = await self.shap_analyzer.explain_prediction(
                    model, features, model_id
                )
                explanations['shap'] = shap_explanations
            
            # LIME explanations
            if self.lime_analyzer and self.lime_analyzer.lime_available:
                lime_explanations = await self.lime_analyzer.explain_prediction(
                    model, features, model_id
                )
                explanations['lime'] = lime_explanations
            
        except Exception as e:
            self.logger.error(f"Error getting model explanations for {model_id}: {e}")
        
        return explanations
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics."""
        return {
            'total_decisions': len(self.trade_decisions),
            'total_model_performances': len(self.model_performances),
            'total_ensemble_performances': len(self.ensemble_performances),
            'monitoring_duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'decisions_per_hour': self.decision_count / max(1, (datetime.now() - self.start_time).total_seconds() / 3600),
            'memory_usage_mb': len(str(self.trade_decisions)) / 1024 / 1024,  # Rough estimate
        }
    
    @handles_errors(default_return = False, context="enhanced_ml_monitor.force_export")
    async def force_export(self) -> bool:
        """Force immediate export of all monitoring data."""
        return await self.export_monthly_report()