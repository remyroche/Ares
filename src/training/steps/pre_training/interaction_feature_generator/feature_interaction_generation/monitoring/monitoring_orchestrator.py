#!/usr/bin/env python3
from ...utils.logger import system_logger
from src.core.decorators import handles_errors
"""
Monitoring Orchestrator for Enhanced ML Monitoring

Orchestrates all monitoring components and provides a unified interface
for comprehensive ML model and ensemble monitoring across all trading modes.
"""

from dataclasses import dataclass, asdict

from .utils.common import (
    get_current_datetime, format_datetime, ensure_directory,
    safe_json_dump, safe_json_load, safe_file_exists,
    timed_operation, format_bytes, safe_log_metric, safe_log_params
)

# Import all monitoring components
from .monitoring.enhanced_ml_monitor import (
    EnhancedMLMonitor, TradeContext, TradingIndicator, MLModelDecision,
    EnsembleDecision, TradeDecision, TradingMode, ModelType,
    ModelPerformanceMetrics, EnsemblePerformanceMetrics, HMMRegimeInfo
)
from .monitoring.explainability_integration import ExplainabilityIntegrator
from .monitoring.ensemble_monitor import EnsembleMonitor, ModelContribution
from .monitoring.csv_export_manager import CSVExportManager
from .monitoring.trading_integration import TradingSystemIntegrator
from .monitoring.daily_summary_tracker import DailySummaryTracker

import datetime

import numpy as np

@dataclass
class MonitoringConfig:
    """Comprehensive monitoring configuration."""
    # Core monitoring settings
    enable_monitoring: bool = True
    enable_explanations: bool = True
    enable_ensemble_monitoring: bool = True
    enable_csv_export: bool = True
    
    # Performance settings
    max_memory_decisions: int = 10000
    export_interval_days: int = 30
    real_time_export: bool = False
    
    # SHAP/LIME settings
    enable_shap: bool = True
    enable_lime: bool = True
    max_features_explained: int = 20
    explanation_cache_size: int = 1000
    
    # Ensemble settings
    weight_update_frequency_hours: int = 24
    performance_window_days: int = 30
    min_weight_threshold: float = 0.01
    max_weight_threshold: float = 0.8
    
    # Export settings
    export_directory: str = "monitoring_exports"
    include_summary_stats: bool = True
    compression: str = "none"
    decimal_precision: int = 6
    
    # Trading integration settings
    enable_backtesting_integration: bool = True
    enable_paper_trading_integration: bool = True
    enable_live_trading_integration: bool = True

class MonitoringOrchestrator:
    """
    Orchestrates all monitoring components for comprehensive ML monitoring.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the monitoring orchestrator."""
        self.config = config
        self.logger = system_logger.getChild("MonitoringOrchestrator")
        
        # Parse configuration
        self.monitoring_config = MonitoringConfig(**config.get("enhanced_monitoring", {}))
        
        # Initialize components
        self.enhanced_monitor: Optional[EnhancedMLMonitor] = None
        self.explainability_integrator: Optional[ExplainabilityIntegrator] = None
        self.ensemble_monitor: Optional[EnsembleMonitor] = None
        self.csv_export_manager: Optional[CSVExportManager] = None
        self.trading_integrator: Optional[TradingSystemIntegrator] = None
        self.daily_summary_tracker: Optional[DailySummaryTracker] = None
        
        # Orchestrator state
        self.is_initialized = False
        self.start_time = datetime.now()
        self.total_decisions_processed = 0
        self.total_exports_performed = 0
        
        self.logger.info("Monitoring Orchestrator created")
    
    @handles_errors(default_return = False, context="monitoring_orchestrator.initialize")
    async def initialize(self) -> bool:
        """Initialize all monitoring components."""
        try:
            self.logger.info("Initializing Enhanced ML Monitoring System...")
            
            if not self.monitoring_config.enable_monitoring:
                self.logger.info("Monitoring disabled in configuration")
                return True
            
            # Initialize core monitoring
            self.enhanced_monitor = EnhancedMLMonitor(self.config)
            await self.enhanced_monitor.initialize()
            
            # Initialize explainability integration
            if self.monitoring_config.enable_explanations:
                self.explainability_integrator = ExplainabilityIntegrator(self.config)
            
            # Initialize ensemble monitoring
            if self.monitoring_config.enable_ensemble_monitoring:
                self.ensemble_monitor = EnsembleMonitor(self.config)
            
            # Initialize CSV export manager
            if self.monitoring_config.enable_csv_export:
                self.csv_export_manager = CSVExportManager(self.config)
            
            # Initialize daily summary tracker
            self.daily_summary_tracker = DailySummaryTracker(self.config)
            
            # Initialize trading system integrator
            self.trading_integrator = TradingSystemIntegrator(self.config)
            
            self.is_initialized = True
            self.logger.info("✅ Enhanced ML Monitoring System initialized successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing monitoring orchestrator: {e}")
            return False
    
    @handles_errors(default_return = None, context="monitoring_orchestrator.record_trade_decision")
    async def record_trade_decision(self, decision: TradeDecision) -> None:
        """Record a trade decision with comprehensive monitoring."""
        try:
            if not self.is_initialized or not self.enhanced_monitor:
                self.logger.warning("Monitoring not initialized, skipping decision recording")
                return
            
            # Record in enhanced monitor
            await self.enhanced_monitor.record_trade_decision(decision)
            
            # Generate explanations if enabled
            if self.explainability_integrator and self.monitoring_config.enable_explanations:
                await self._generate_decision_explanations(decision)
            
            # Update ensemble monitoring
            if self.ensemble_monitor and self.monitoring_config.enable_ensemble_monitoring:
                await self._update_ensemble_monitoring(decision)
            
            # Update daily summary tracking
            if self.daily_summary_tracker:
                await self.daily_summary_tracker.add_trade(decision)
            
            # Update statistics
            self.total_decisions_processed += 1
            
            # Check for export if needed
            if self.monitoring_config.real_time_export:
                await self._check_and_export()
            
            self.logger.debug(f"Recorded trade decision {decision.decision_id}")
            
        except Exception as e:
            self.logger.error(f"Error recording trade decision: {e}")
    
    async def _generate_decision_explanations(self, decision: TradeDecision):
        """Generate explanations for a trade decision."""
        try:
            if not self.explainability_integrator:
                return
            
            # Generate explanations for each model in the ensemble
            for model_decision in decision.ensemble_decision.model_decisions:
                # This would require access to the actual model and features
                # For now, we'll skip detailed explanation generation
                # In a real implementation, you would:
                # 1. Get the model from a model registry
                # 2. Get the features used for the prediction
                # 3. Generate SHAP/LIME explanations
                # 4. Update the model_decision with explanations
                pass
            
        except Exception as e:
            self.logger.error(f"Error generating decision explanations: {e}")
    
    async def _update_ensemble_monitoring(self, decision: TradeDecision):
        """Update ensemble monitoring with decision data."""
        try:
            if not self.ensemble_monitor:
                return
            
            # Extract model contributions from the decision
            model_contributions = []
            for model_decision in decision.ensemble_decision.model_decisions:
                contribution = ModelContribution(
                    model_id = model_decision.model_id,
                    model_type = model_decision.model_type.value,
                    contribution_score = model_decision.confidence * decision.ensemble_decision.model_weights.get(model_decision.model_id, 0.0),
                    accuracy_contribution = model_decision.confidence,
                    profit_contribution = 0.0,  # Would need actual profit data
                    risk_contribution = model_decision.risk_score,
                    prediction_agreement = 1.0 - decision.ensemble_decision.disagreement_level,
                    feature_diversity = 0.5,  # Would need feature analysis
                    timestamp = decision.timestamp
                )
                model_contributions.append(contribution)
            
            # Record ensemble performance
            performance_metrics = {
                'accuracy': decision.overall_confidence,
                'win_rate': 0.5,  # Would need actual win/loss data
                'profit_factor': 1.0,  # Would need actual profit data
                'sharpe_ratio': 0.0,  # Would need actual returns data
                'model_diversity_score': 1.0 - decision.ensemble_decision.disagreement_level,
                'consensus_quality': decision.ensemble_decision.consensus_score,
                'disagreement_impact': decision.ensemble_decision.disagreement_level,
                'weight_stability': 0.8  # Would need historical weight data
            }
            
            await self.ensemble_monitor.record_ensemble_performance(
                decision.ensemble_decision.ensemble_id,
                performance_metrics,
                model_contributions
            )
            
        except Exception as e:
            self.logger.error(f"Error updating ensemble monitoring: {e}")
    
    async def _check_and_export(self):
        """Check if it's time to export data."""
        try:
            if not self.csv_export_manager:
                return
            
            # Check if we should export based on time or decision count
            time_since_start = datetime.now() - self.start_time
            should_export_time = time_since_start.days >= self.monitoring_config.export_interval_days
            should_export_count = self.total_decisions_processed >= self.monitoring_config.max_memory_decisions
            
            if should_export_time or should_export_count:
                await self.export_monitoring_data()
                self.start_time = datetime.now()  # Reset timer
                self.total_decisions_processed = 0  # Reset counter
                
        except Exception as e:
            self.logger.error(f"Error checking export timing: {e}")
    
    @handles_errors(default_return = False, context="monitoring_orchestrator.export_monitoring_data")
    async def export_monitoring_data(self) -> bool:
        """Export all monitoring data to CSV files."""
        try:
            if not self.csv_export_manager or not self.enhanced_monitor:
                self.logger.warning("Export components not available")
                return False
            
            self.logger.info("Starting comprehensive monitoring data export...")
            
            # Export trade decisions
            if self.enhanced_monitor.trade_decisions:
                success = await self.csv_export_manager.export_trade_decisions(
                    self.enhanced_monitor.trade_decisions
                )
                if success:
                    self.logger.info(f"Exported {len(self.enhanced_monitor.trade_decisions)} trade decisions")
            
            # Export model performances
            if self.enhanced_monitor.model_performances:
                success = await self.csv_export_manager.export_model_performances(
                    self.enhanced_monitor.model_performances
                )
                if success:
                    self.logger.info(f"Exported {len(self.enhanced_monitor.model_performances)} model performances")
            
            # Export ensemble performances
            if self.enhanced_monitor.ensemble_performances:
                success = await self.csv_export_manager.export_ensemble_performances(
                    self.enhanced_monitor.ensemble_performances
                )
                if success:
                    self.logger.info(f"Exported {len(self.enhanced_monitor.ensemble_performances)} ensemble performances")
            
            # Export daily summaries
            if self.daily_summary_tracker and self.daily_summary_tracker.daily_summaries:
                summaries = list(self.daily_summary_tracker.daily_summaries.values())
                success = await self.csv_export_manager.export_daily_summaries(summaries)
                if success:
                    self.logger.info(f"Exported {len(summaries)} daily summaries")
            
            self.total_exports_performed += 1
            self.logger.info("✅ Monitoring data export completed successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting monitoring data: {e}")
            return False
    
    @handles_errors(default_return = False, context="monitoring_orchestrator.integrate_trading_system")
    async def integrate_trading_system(self, trading_system: Any, 
                                     system_type: str, 
                                     system_id: Optional[str] = None) -> bool:
        """Integrate monitoring with a trading system."""
        try:
            if not self.trading_integrator:
                self.logger.warning("Trading integrator not available")
                return False
            
            system_id = system_id or f"{system_type}_{int(time.time())}"
            
            if system_type.lower() == "backtesting":
                return await self.trading_integrator.integrate_backtesting(trading_system, system_id)
            elif system_type.lower() == "paper_trading":
                return await self.trading_integrator.integrate_paper_trading(trading_system, system_id)
            elif system_type.lower() == "live_trading":
                return await self.trading_integrator.integrate_live_trading(trading_system, system_id)
            else:
                self.logger.error(f"Unknown trading system type: {system_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error integrating trading system: {e}")
            return False
    
    @handles_errors(default_return = None, context="monitoring_orchestrator.get_ensemble_analysis")
    async def get_ensemble_analysis(self, ensemble_id: str) -> Dict[str, Any]:
        """Get comprehensive ensemble analysis."""
        try:
            if not self.ensemble_monitor:
                return {'error': 'Ensemble monitoring not available'}
            
            return await self.ensemble_monitor.get_ensemble_analysis(ensemble_id)
            
        except Exception as e:
            self.logger.error(f"Error getting ensemble analysis: {e}")
            return {'error': str(e)}
    
    @handles_errors(default_return = None, context="monitoring_orchestrator.get_model_explanations")
    async def get_model_explanations(self, model_id: str, features: np.ndarray, 
                                   model: Any) -> Dict[str, Any]:
        """Get model explanations using SHAP/LIME."""
        try:
            if not self.explainability_integrator:
                return {'error': 'Explainability integration not available'}
            
            return await self.explainability_integrator.explain_model_prediction(
                model_id, model, features, [], 0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error getting model explanations: {e}")
            return {'error': str(e)}
    
    @handles_errors(default_return = False, context="monitoring_orchestrator.update_ensemble_weights")
    async def update_ensemble_weights(self, ensemble_id: str, 
                                    model_performances: Dict[str, Dict[str, float]],
                                    current_weights: Dict[str, float]) -> Dict[str, float]:
        """Update ensemble model weights."""
        try:
            if not self.ensemble_monitor:
                return current_weights
            
            return await self.ensemble_monitor.update_ensemble_weights(
                ensemble_id, model_performances, current_weights
            )
            
        except Exception as e:
            self.logger.error(f"Error updating ensemble weights: {e}")
            return current_weights
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        try:
            stats = {
                'orchestrator': {
                    'is_initialized': self.is_initialized,
                    'start_time': self.start_time.isoformat(),
                    'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
                    'total_decisions_processed': self.total_decisions_processed,
                    'total_exports_performed': self.total_exports_performed
                },
                'configuration': asdict(self.monitoring_config)
            }
            
            # Add component stats
            if self.enhanced_monitor:
                stats['enhanced_monitor'] = self.enhanced_monitor.get_monitoring_stats()
            
            if self.ensemble_monitor:
                stats['ensemble_monitor'] = self.ensemble_monitor.get_ensemble_stats()
            
            if self.explainability_integrator:
                stats['explainability_integrator'] = self.explainability_integrator.get_explanation_stats()
            
            if self.csv_export_manager:
                stats['csv_export_manager'] = self.csv_export_manager.get_export_stats()
            
            if self.trading_integrator:
                stats['trading_integrator'] = self.trading_integrator.get_integration_stats()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive stats: {e}")
            return {'error': str(e)}
    
    @handles_errors(default_return = False, context="monitoring_orchestrator.shutdown")
    async def shutdown(self) -> bool:
        """Shutdown the monitoring orchestrator gracefully."""
        try:
            self.logger.info("Shutting down Enhanced ML Monitoring System...")
            
            # Export any remaining data
            if self.total_decisions_processed > 0:
                await self.export_monitoring_data()
            
            # Clear components
            self.enhanced_monitor = None
            self.explainability_integrator = None
            self.ensemble_monitor = None
            self.csv_export_manager = None
            self.trading_integrator = None
            
            self.is_initialized = False
            self.logger.info("✅ Enhanced ML Monitoring System shutdown completed")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            return False

# Factory function for easy initialization
@handles_errors(default_return = None, context="monitoring_orchestrator.create_monitoring_orchestrator")
async def create_monitoring_orchestrator(config: Dict[str, Any]) -> Optional[MonitoringOrchestrator]:
    """Create and initialize a monitoring orchestrator."""
    try:
        orchestrator = MonitoringOrchestrator(config)
        success = await orchestrator.initialize()
        
        if success:
            return orchestrator
        else:
            return None
            
    except Exception as e:
        system_logger.error(f"Error creating monitoring orchestrator: {e}")
        return None