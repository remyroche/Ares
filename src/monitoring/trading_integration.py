#!/usr/bin/env python3
"""
Trading Integration for Enhanced ML Monitoring

Integrates the enhanced monitoring system with backtesting, paper trading,
and live trading systems to capture comprehensive trade decision data.
"""


import time
import uuid

from .utils.common_operations import (
    get_current_datetime, format_datetime, ensure_directory,
)
from .utils.logger import system_logger

# Import monitoring components
from src.monitoring.enhanced_ml_monitoring import (
    EnhancedMLMonitor, TradeContext, TradingIndicator, MLModelDecision,
    EnsembleDecision, TradeDecision, TradingMode, ModelType
)
from .monitoring.explainability_integration import ExplainabilityIntegrator


@dataclass
class TradingSystemConfig:
    """Configuration for trading system integration."""
    enable_monitoring: bool = True
    capture_explanations: bool = True
    capture_performance_metrics: bool = True
    real_time_export: bool = False
    export_interval_minutes: int = 60
    max_memory_decisions: int = 10000


class TradingSystemIntegrator:
    """
    Integrates enhanced monitoring with trading systems.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trading system integrator."""
        self.config = config
        self.logger = system_logger.getChild("TradingSystemIntegrator")
        
        # Configuration
        self.integration_config = TradingSystemConfig(**config.get("trading_integration", {}))
        
        # Initialize monitoring components
        self.enhanced_monitor = EnhancedMLMonitor(config)
        self.explainability_integrator = ExplainabilityIntegrator(config)
        self.ensemble_monitor = EnsembleMonitor(config)
        
        # Integration state
        self.active_integrations: Dict[str, Dict[str, Any]] = {}
        self.decision_callbacks: Dict[str, List[Callable]] = {}
        
        self.logger.info("Trading System Integrator initialized")
    
    @handles_errors(default_return=False, context="trading_integration.integrate_backtesting")
    async def integrate_backtesting(self, backtesting_system: Any, 
                                system_id: str = "backtesting") -> bool:
        """Integrate monitoring with backtesting system."""
        try:
            if not self.integration_config.enable_monitoring:
                self.logger.info("Monitoring disabled, skipping backtesting integration")
                return True
            
            # Store integration info
            self.active_integrations[system_id] = {
                'type': 'backtesting',
                'system': backtesting_system,
                'start_time': datetime.now(),
                'decisions_captured': 0
            }
            
            # Hook into backtesting decision points
            await self._hook_backtesting_system(backtesting_system, system_id)
            
            self.logger.info(f"Integrated monitoring with backtesting system: {system_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error integrating backtesting system: {e}")
            return False
    
    @handles_errors(default_return=False, context="trading_integration.integrate_paper_trading")
    async def integrate_paper_trading(self, paper_trading_system: Any, 
                                    system_id: str = "paper_trading") -> bool:
        """Integrate monitoring with paper trading system."""
        try:
            if not self.integration_config.enable_monitoring:
                self.logger.info("Monitoring disabled, skipping paper trading integration")
                return True
            
            # Store integration info
            self.active_integrations[system_id] = {
                'type': 'paper_trading',
                'system': paper_trading_system,
                'start_time': datetime.now(),
                'decisions_captured': 0
            }
            
            # Hook into paper trading decision points
            await self._hook_paper_trading_system(paper_trading_system, system_id)
            
            self.logger.info(f"Integrated monitoring with paper trading system: {system_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error integrating paper trading system: {e}")
            return False
    
    @handles_errors(default_return=False, context="trading_integration.integrate_live_trading")
    async def integrate_live_trading(self, live_trading_system: Any, 
                                system_id: str = "live_trading") -> bool:
        """Integrate monitoring with live trading system."""
        try:
            if not self.integration_config.enable_monitoring:
                self.logger.info("Monitoring disabled, skipping live trading integration")
                return True
            
            # Store integration info
            self.active_integrations[system_id] = {
                'type': 'live_trading',
                'system': live_trading_system,
                'start_time': datetime.now(),
                'decisions_captured': 0
            }
            
            # Hook into live trading decision points
            await self._hook_live_trading_system(live_trading_system, system_id)
            
            self.logger.info(f"Integrated monitoring with live trading system: {system_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error integrating live trading system: {e}")
            return False
    
    async def _hook_backtesting_system(self, backtesting_system: Any, system_id: str):
        """Hook into backtesting system to capture decisions."""
        try:
            # This would depend on the specific backtesting system implementation
            # For now, we'll create a generic hook mechanism
            
            # Hook into trade execution
            if hasattr(backtesting_system, 'execute_trade'):
                original_execute_trade = backtesting_system.execute_trade
                
                async def monitored_execute_trade(*args, **kwargs):
                    # Capture decision before execution
                    decision_data = await self._capture_trading_decision(
                        system_id, TradingMode.BACKTEST, *args, **kwargs
                    )
                    
                    # Execute original trade
                    result = await original_execute_trade(*args, **kwargs)
                    
                    # Update decision with results
                    if decision_data:
                        await self._update_decision_with_results(decision_data, result)
                    
                    return result
                
                backtesting_system.execute_trade = monitored_execute_trade
            
            # Hook into model predictions
            if hasattr(backtesting_system, 'get_prediction'):
                original_get_prediction = backtesting_system.get_prediction
                
                async def monitored_get_prediction(*args, **kwargs):
                    # Capture prediction context
                    await self._capture_prediction_context(system_id, *args, **kwargs)
                    
                    # Get original prediction
                    return await original_get_prediction(*args, **kwargs)
                
                backtesting_system.get_prediction = monitored_get_prediction
            
        except Exception as e:
            self.logger.error(f"Error hooking backtesting system: {e}")
    
    async def _hook_paper_trading_system(self, paper_trading_system: Any, system_id: str):
        """Hook into paper trading system to capture decisions."""
        try:
            # Similar to backtesting but for paper trading
            if hasattr(paper_trading_system, 'execute_trade'):
                original_execute_trade = paper_trading_system.execute_trade
                
                async def monitored_execute_trade(*args, **kwargs):
                    # Capture decision before execution
                    decision_data = await self._capture_trading_decision(
                        system_id, TradingMode.PAPER, *args, **kwargs
                    )
                    
                    # Execute original trade
                    result = await original_execute_trade(*args, **kwargs)
                    
                    # Update decision with results
                    if decision_data:
                        await self._update_decision_with_results(decision_data, result)
                    
                    return result
                
                paper_trading_system.execute_trade = monitored_execute_trade
            
        except Exception as e:
            self.logger.error(f"Error hooking paper trading system: {e}")
    
    async def _hook_live_trading_system(self, live_trading_system: Any, system_id: str):
        """Hook into live trading system to capture decisions."""
        try:
            # Similar to paper trading but for live trading
            if hasattr(live_trading_system, 'execute_trade'):
                original_execute_trade = live_trading_system.execute_trade
                
                async def monitored_execute_trade(*args, **kwargs):
                    # Capture decision before execution
                    decision_data = await self._capture_trading_decision(
                        system_id, TradingMode.LIVE, *args, **kwargs
                    )
                    
                    # Execute original trade
                    result = await original_execute_trade(*args, **kwargs)
                    
                    # Update decision with results
                    if decision_data:
                        await self._update_decision_with_results(decision_data, result)
                    
                    return result
                
                live_trading_system.execute_trade = monitored_execute_trade
            
        except Exception as e:
            self.logger.error(f"Error hooking live trading system: {e}")
    
    @handles_errors(default_return=None, context="trading_integration._capture_trading_decision")
    async def _capture_trading_decision(self, system_id: str, trading_mode: TradingMode,
                                    *args, **kwargs) -> Optional[TradeDecision]:
        """Capture a trading decision with full context and explanations."""
        try:
            start_time = time.time()
            
            # Extract decision context from arguments
            context = await self._extract_decision_context(*args, **kwargs)
            if not context:
                return None
            
            # Generate decision ID
            decision_id = f"{system_id}_{uuid.uuid4().hex[:8]}_{int(time.time())}"
            
            # Extract trading indicators
            trading_indicators = await self._extract_trading_indicators(*args, **kwargs)
            
            # Extract ensemble decision
            ensemble_decision = await self._extract_ensemble_decision(*args, **kwargs)
            
            # Calculate overall metrics
            overall_confidence = self._calculate_overall_confidence(trading_indicators, ensemble_decision)
            overall_risk_score = self._calculate_overall_risk_score(trading_indicators, ensemble_decision)
            
            # Extract final decision
            action, position_size, stop_loss, take_profit = await self._extract_final_decision(*args, **kwargs)
            
            # Create trade decision
            trade_decision = TradeDecision(
                decision_id=decision_id,
                context=context,
                trading_mode=trading_mode,
                timestamp=datetime.now(),
                trading_indicators=trading_indicators,
                overall_confidence=overall_confidence,
                overall_risk_score=overall_risk_score,
                ensemble_decision=ensemble_decision,
                action=action,
                position_size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                execution_time_ms=(time.time() - start_time) * 1000
            )
            
            # Record the decision
            await self.enhanced_monitor.record_trade_decision(trade_decision)
            
            # Update integration stats
            if system_id in self.active_integrations:
                self.active_integrations[system_id]['decisions_captured'] += 1
            
            return trade_decision
            
        except Exception as e:
            self.logger.error(f"Error capturing trading decision: {e}")
            return None
    
    async def _extract_decision_context(self, *args, **kwargs) -> Optional[TradeContext]:
        """Extract decision context from trading system arguments."""
        try:
            # This would need to be customized based on the actual trading system interface
            # For now, we'll create a generic extraction method
            
            # Try to extract from common argument patterns
            exchange = kwargs.get('exchange', 'unknown')
            token = kwargs.get('token', kwargs.get('symbol', 'unknown'))
            price = kwargs.get('price', 0.0)
            volume = kwargs.get('volume', 0.0)
            timeframe = kwargs.get('timeframe', '1h')
            regime = kwargs.get('regime')
            
            # Market conditions
            market_conditions = {}
            for key in ['volatility', 'trend', 'volume_profile', 'support_resistance']:
                if key in kwargs:
                    market_conditions[key] = kwargs[key]
            
            return TradeContext(
                exchange=exchange,
                token=token,
                timestamp=datetime.now(),
                price=price,
                volume=volume,
                timeframe=timeframe,
                regime=regime,
                market_conditions=market_conditions if market_conditions else None
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting decision context: {e}")
            return None
    
    async def _extract_trading_indicators(self, *args, **kwargs) -> List[TradingIndicator]:
        """Extract trading indicators from arguments."""
        try:
            indicators = []
            
            # Common trading indicators
            indicator_mappings = {
                'rsi': 'RSI',
                'macd': 'MACD',
                'bollinger_bands': 'Bollinger Bands',
                'moving_average': 'Moving Average',
                'volume_profile': 'Volume Profile',
                'support_resistance': 'Support/Resistance',
                'momentum': 'Momentum',
                'volatility': 'Volatility'
            }
            
            for key, name in indicator_mappings.items():
                if key in kwargs:
                    value = kwargs[key]
                    if isinstance(value, (int, float)):
                        indicators.append(TradingIndicator(
                            name=name,
                            value=value,
                            weight=kwargs.get(f'{key}_weight', 0.1),
                            confidence=kwargs.get(f'{key}_confidence', 0.5),
                            risk_score=kwargs.get(f'{key}_risk', 0.5),
                            description=f"{name} indicator value"
                        ))
            
            return indicators
            
        except Exception as e:
            self.logger.error(f"Error extracting trading indicators: {e}")
            return []
    
    async def _extract_ensemble_decision(self, *args, **kwargs) -> EnsembleDecision:
        """Extract ensemble decision from arguments."""
        try:
            # Extract ensemble information
            ensemble_id = kwargs.get('ensemble_id', 'default_ensemble')
            final_prediction = kwargs.get('prediction', kwargs.get('final_prediction', 0.0))
            final_confidence = kwargs.get('confidence', kwargs.get('final_confidence', 0.5))
            final_risk_score = kwargs.get('risk_score', kwargs.get('final_risk_score', 0.5))
            voting_mechanism = kwargs.get('voting_mechanism', 'weighted_average')
            
            # Extract model weights
            model_weights = kwargs.get('model_weights', {})
            if not model_weights:
                # Try to extract from individual model predictions
                model_weights = self._extract_model_weights_from_predictions(kwargs)
            
            # Extract model decisions
            model_decisions = await self._extract_model_decisions(kwargs)
            
            # Calculate consensus metrics
            consensus_score = self._calculate_consensus_score(model_decisions)
            disagreement_level = self._calculate_disagreement_level(model_decisions)
            
            return EnsembleDecision(
                ensemble_id=ensemble_id,
                final_prediction=final_prediction,
                final_confidence=final_confidence,
                final_risk_score=final_risk_score,
                model_weights=model_weights,
                model_decisions=model_decisions,
                voting_mechanism=voting_mechanism,
                consensus_score=consensus_score,
                disagreement_level=disagreement_level
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting ensemble decision: {e}")
            # Return minimal ensemble decision
            return EnsembleDecision(
                ensemble_id='unknown',
                final_prediction=0.0,
                final_confidence=0.0,
                final_risk_score=0.5,
                model_weights={},
                model_decisions=[],
                voting_mechanism='unknown',
                consensus_score=0.0,
                disagreement_level=1.0
            )
    
    def _extract_model_weights_from_predictions(self, kwargs: Dict[str, Any]) -> Dict[str, float]:
        """Extract model weights from prediction arguments."""
        model_weights = {}
        
        # Look for model-specific predictions
        for key, value in kwargs.items():
            if key.startswith('model_') and key.endswith('_prediction'):
                model_id = key.replace('model_', '').replace('_prediction', '')
                weight = kwargs.get(f'model_{model_id}_weight', 0.1)
                model_weights[model_id] = weight
        
        # If no weights found, create equal weights
        if not model_weights:
            model_weights = {'default_model': 1.0}
        
        return model_weights
    
    async def _extract_model_decisions(self, kwargs: Dict[str, Any]) -> List[MLModelDecision]:
        """Extract individual model decisions from arguments."""
        model_decisions = []
        
        try:
            # Look for model-specific data
            model_ids = set()
            for key in kwargs.keys():
                if key.startswith('model_') and '_' in key:
                    parts = key.split('_')
                    if len(parts) >= 2:
                        model_ids.add(parts[1])
            
            for model_id in model_ids:
                prediction = kwargs.get(f'model_{model_id}_prediction', 0.0)
                confidence = kwargs.get(f'model_{model_id}_confidence', 0.5)
                risk_score = kwargs.get(f'model_{model_id}_risk', 0.5)
                model_type = kwargs.get(f'model_{model_id}_type', 'unknown')
                processing_time = kwargs.get(f'model_{model_id}_processing_time_ms', 0.0)
                model_version = kwargs.get(f'model_{model_id}_version', 'unknown')
                
                # Extract feature importance
                feature_importance = {}
                for key, value in kwargs.items():
                    if key.startswith(f'model_{model_id}_feature_'):
                        feature_name = key.replace(f'model_{model_id}_feature_', '')
                        if isinstance(value, (int, float)):
                            feature_importance[feature_name] = value
                
                model_decision = MLModelDecision(
                    model_id=model_id,
                    model_type=ModelType(model_type) if model_type in [e.value for e in ModelType] else ModelType.HMM,
                    prediction=prediction,
                    confidence=confidence,
                    risk_score=risk_score,
                    feature_importance=feature_importance,
                    processing_time_ms=processing_time,
                    model_version=model_version
                )
                
                model_decisions.append(model_decision)
            
        except Exception as e:
            self.logger.error(f"Error extracting model decisions: {e}")
        
        return model_decisions
    
    def _calculate_consensus_score(self, model_decisions: List[MLModelDecision]) -> float:
        """Calculate consensus score among model decisions."""
        if not model_decisions:
            return 0.0
        
        predictions = [md.prediction for md in model_decisions]
        if not predictions:
            return 0.0
        
        # Calculate variance in predictions
        variance = np.var(predictions)
        # Convert to consensus score (lower variance = higher consensus)
        consensus = max(0.0, 1.0 - variance)
        
        return consensus
    
    def _calculate_disagreement_level(self, model_decisions: List[MLModelDecision]) -> float:
        """Calculate disagreement level among model decisions."""
        if not model_decisions:
            return 1.0
        
        predictions = [md.prediction for md in model_decisions]
        if not predictions:
            return 1.0
        
        # Calculate standard deviation of predictions
        std_dev = np.std(predictions)
        # Normalize to 0-1 range
        disagreement = min(1.0, std_dev)
        
        return disagreement
    
    def _calculate_overall_confidence(self, trading_indicators: List[TradingIndicator],
                                    ensemble_decision: EnsembleDecision) -> float:
        """Calculate overall confidence from indicators and ensemble."""
        confidence_factors = []
        
        # Add ensemble confidence
        confidence_factors.append(ensemble_decision.final_confidence)
        
        # Add indicator confidences
        for indicator in trading_indicators:
            confidence_factors.append(indicator.confidence)
        
        # Add model confidences
        for model_decision in ensemble_decision.model_decisions:
            confidence_factors.append(model_decision.confidence)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def _calculate_overall_risk_score(self, trading_indicators: List[TradingIndicator],
                                    ensemble_decision: EnsembleDecision) -> float:
        """Calculate overall risk score from indicators and ensemble."""
        risk_factors = []
        
        # Add ensemble risk
        risk_factors.append(ensemble_decision.final_risk_score)
        
        # Add indicator risks
        for indicator in trading_indicators:
            risk_factors.append(indicator.risk_score)
        
        # Add model risks
        for model_decision in ensemble_decision.model_decisions:
            risk_factors.append(model_decision.risk_score)
        
        return np.mean(risk_factors) if risk_factors else 0.5
    
    async def _extract_final_decision(self, *args, **kwargs) -> Tuple[str, float, Optional[float], Optional[float]]:
        """Extract final trading decision from arguments."""
        action = kwargs.get('action', 'hold')
        position_size = kwargs.get('position_size', kwargs.get('size', 0.0))
        stop_loss = kwargs.get('stop_loss')
        take_profit = kwargs.get('take_profit')
        
        return action, position_size, stop_loss, take_profit
    
    async def _update_decision_with_results(self, decision: TradeDecision, result: Any):
        """Update decision with execution results."""
        try:
            # Extract success metrics from result
            success_metrics = {}
            
            if hasattr(result, 'profit_loss'):
                success_metrics['profit_loss'] = result.profit_loss
            
            if hasattr(result, 'execution_price'):
                success_metrics['execution_price'] = result.execution_price
            
            if hasattr(result, 'slippage'):
                success_metrics['slippage'] = result.slippage
            
            if hasattr(result, 'commission'):
                success_metrics['commission'] = result.commission
            
            # Update decision
            decision.success_metrics = success_metrics
            
            # Re-record the updated decision
            await self.enhanced_monitor.record_trade_decision(decision)
            
        except Exception as e:
            self.logger.error(f"Error updating decision with results: {e}")
    
    async def _capture_prediction_context(self, system_id: str, *args, **kwargs):
        """Capture prediction context for analysis."""
        try:
            # This could be used to capture model prediction contexts
            # without full trade decisions
            pass
            
        except Exception as e:
            self.logger.error(f"Error capturing prediction context: {e}")
    
    @handles_errors(default_return=False, context="trading_integration.force_export")
    async def force_export(self) -> bool:
        """Force export of all monitoring data."""
        try:
            return await self.enhanced_monitor.force_export()
        except Exception as e:
            self.logger.error(f"Error in force export: {e}")
            return False
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get statistics about trading system integrations."""
        stats = {
            'active_integrations': len(self.active_integrations),
            'integration_details': {}
        }
        
        for system_id, integration in self.active_integrations.items():
            stats['integration_details'][system_id] = {
                'type': integration['type'],
                'start_time': integration['start_time'].isoformat(),
                'decisions_captured': integration['decisions_captured'],
                'uptime_hours': (datetime.now() - integration['start_time']).total_seconds() / 3600
            }
        
        # Add monitoring stats
        stats['monitoring_stats'] = self.enhanced_monitor.get_monitoring_stats()
        stats['ensemble_stats'] = self.ensemble_monitor.get_ensemble_stats()
        stats['explainability_stats'] = self.explainability_integrator.get_explanation_stats()
        
        return stats