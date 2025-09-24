from typing import List
from typing import Any
from typing import Dict
import pandas as pd
from typing import Optional
from typing import Tuple
import numpy as np
from ..utils.logger import system_logger
'Explainability orchestrator for coordinating all model explanations.\n\nThis module provides a centralized orchestrator for managing SHAP/LIME explanations\nacross all ML models and creating comprehensive trade decision traces.\n'
from datetime import datetime
import asyncio
from pathlib import Path
import json
from .explainability.base_explainer import TradeDecisionTracer, TradeDecisionTrace
from .explainability.tactician_explainer import TacticianExplainer
from .explainability.hmm_explainer import HMMExplainer
from .explainability.sr_explainer import SRExplainer
from .explainability.analyst_explainer import AnalystExplainer
import logging
import time

class ExplainabilityOrchestrator:
    """Orchestrator for managing all model explanations and decision traces."""

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize explainability orchestrator."""
        self.config = config
        self.logger = system_logger.getChild('ExplainabilityOrchestrator')
        self.tactician_explainer = TacticianExplainer(config)
        self.hmm_explainer = HMMExplainer(config)
        self.sr_explainer = SRExplainer(config)
        self.analyst_explainer = AnalystExplainer(config)
        self.decision_tracer = TradeDecisionTracer(config)
        self.explain_config = config.get('explainability', {})
        self.enable_explanations = self.explain_config.get('enable_explanations', True)
        self.enable_decision_tracing = self.explain_config.get('enable_decision_tracing', True)
        self.explanation_timeout = self.explain_config.get('explanation_timeout', 30)
        self.registered_models = {'tactician': {}, 'hmm': {}, 'sr': {}, 'analyst': {}}
        self.active_traces: Dict[str, TradeDecisionTrace] = {}

    async def register_model(self, model_type: str, model_name: str, model: Any, training_data: Optional[pd.DataFrame]=None) -> bool:
        """Register a model for explanation."""
        try:
            if model_type not in self.registered_models:
                self.logger.error(f'❌ Unknown model type: {model_type}')
                return False
            self.logger.info(f'📝 Registering {model_type} model: {model_name}')
            self.registered_models[model_type][model_name] = {'model': model, 'training_data': training_data, 'initialized': False}
            if training_data is not None:
                await self._initialize_model_explainers(model_type, model_name)
            self.logger.info(f'✅ Model {model_name} registered successfully')
            return True
        except Exception as e:
            self.logger.error(f'❌ Failed to register model {model_name}: {e}')
            return False

    async def _initialize_model_explainers(self, model_type: str, model_name: str) -> bool:
        """Initialize explainers for a registered model."""
        try:
            model_info = self.registered_models[model_type][model_name]
            model = model_info['model']
            training_data = model_info['training_data']
            if training_data is None:
                self.logger.warning(f'⚠️ No training data available for {model_name}')
                return False
            if model_type == 'tactician':
                success = await self.tactician_explainer.initialize_explainers(model, training_data)
            elif model_type == 'hmm':
                success = await self.hmm_explainer.initialize_explainers(model, training_data)
            elif model_type == 'sr':
                success = await self.sr_explainer.initialize_explainers(model, training_data)
            elif model_type == 'analyst':
                success = await self.analyst_explainer.initialize_explainers(model, training_data)
            else:
                self.logger.error(f'❌ Unknown model type for initialization: {model_type}')
                return False
            if success:
                model_info['initialized'] = True
                self.logger.info(f'✅ Explainers initialized for {model_name}')
            else:
                self.logger.warning(f'⚠️ Failed to initialize explainers for {model_name}')
            return success
        except Exception as e:
            self.logger.error(f'❌ Failed to initialize explainers for {model_name}: {e}')
            return False

    async def explain_model_prediction(self, model_type: str, model_name: str, features: np.ndarray, feature_names: List[str], prediction: Optional[Any]=None, market_conditions: Optional[Dict[str, Any]]=None) -> Optional[Any]:
        """Explain a model prediction."""
        try:
            if not self.enable_explanations:
                self.logger.info('🔇 Explanations disabled')
                return None
            if model_type not in self.registered_models:
                self.logger.error(f'❌ Unknown model type: {model_type}')
                return None
            if model_name not in self.registered_models[model_type]:
                self.logger.error(f'❌ Model {model_name} not registered')
                return None
            model_info = self.registered_models[model_type][model_name]
            model = model_info['model']
            self.logger.info(f'🔍 Explaining {model_type} model: {model_name}')
            if model_type == 'tactician':
                explainer = self.tactician_explainer
            elif model_type == 'hmm':
                explainer = self.hmm_explainer
            elif model_type == 'sr':
                explainer = self.sr_explainer
            elif model_type == 'analyst':
                explainer = self.analyst_explainer
            else:
                self.logger.error(f'❌ No explainer available for model type: {model_type}')
                return None
            try:
                explanation = await asyncio.wait_for(explainer.explain_prediction(model, features, feature_names, prediction), timeout = self.explanation_timeout)
                if explanation:
                    if market_conditions:
                        explanation.metadata['market_conditions'] = market_conditions
                    self.logger.info(f'✅ Explanation completed for {model_name}')
                    return explanation
                else:
                    self.logger.warning(f'⚠️ No explanation generated for {model_name}')
                    return None
            except asyncio.TimeoutError:
                self.logger.error(f'⏰ Explanation timeout for {model_name}')
                return None
        except Exception as e:
            self.logger.error(f'❌ Failed to explain {model_name}: {e}')
            return None

    async def start_trade_decision_trace(self, decision_id: str, decision_type: str, market_conditions: Optional[Dict[str, Any]]=None) -> TradeDecisionTrace:
        """Start tracing a trade decision."""
        try:
            if not self.enable_decision_tracing:
                self.logger.info('🔇 Decision tracing disabled')
                return None
            trace = await self.decision_tracer.start_decision_trace(decision_id, decision_type, market_conditions)
            self.active_traces[decision_id] = trace
            self.logger.info(f'🔍 Started decision trace: {decision_id}')
            return trace
        except Exception as e:
            self.logger.error(f'❌ Failed to start decision trace {decision_id}: {e}')
            return None

    async def add_explanation_to_trace(self, decision_id: str, model_type: str, explanation: Any) -> bool:
        """Add model explanation to decision trace."""
        try:
            if decision_id not in self.active_traces:
                self.logger.error(f'❌ No active trace found for decision {decision_id}')
                return False
            success = await self.decision_tracer.add_model_explanation(decision_id, model_type, explanation)
            if success:
                self.logger.info(f'📊 Added {model_type} explanation to trace {decision_id}')
            else:
                self.logger.warning(f'⚠️ Failed to add {model_type} explanation to trace {decision_id}')
            return success
        except Exception as e:
            self.logger.error(f'❌ Failed to add explanation to trace {decision_id}: {e}')
            return False

    async def finalize_trade_decision_trace(self, decision_id: str, final_decision: Any, confidence: float) -> Optional[TradeDecisionTrace]:
        """Finalize trade decision trace."""
        try:
            if decision_id not in self.active_traces:
                self.logger.error(f'❌ No active trace found for decision {decision_id}')
                return None
            trace = await self.decision_tracer.finalize_decision_trace(decision_id, final_decision, confidence)
            if trace:
                del self.active_traces[decision_id]
                self.logger.info(f'✅ Finalized decision trace: {decision_id}')
            else:
                self.logger.warning(f'⚠️ Failed to finalize decision trace: {decision_id}')
            return trace
        except Exception as e:
            self.logger.error(f'❌ Failed to finalize decision trace {decision_id}: {e}')
            return None

    async def explain_complete_trading_decision(self, decision_id: str, decision_type: str, market_data: pd.DataFrame, tactician_features: Optional[Tuple[np.ndarray, List[str]]]=None, hmm_features: Optional[Tuple[np.ndarray, List[str]]]=None, sr_features: Optional[Tuple[np.ndarray, List[str]]]=None, analyst_features: Optional[Tuple[np.ndarray, List[str]]]=None, final_decision: Optional[Any]=None, confidence: float = 0.5) -> Optional[TradeDecisionTrace]:
        """Explain a complete trading decision across all models."""
        try:
            self.logger.info(f'🔍 Explaining complete trading decision: {decision_id}')
            market_conditions = self._extract_market_conditions(market_data)
            trace = await self.start_trade_decision_trace(decision_id, decision_type, market_conditions)
            if not trace:
                self.logger.error(f'❌ Failed to start trace for {decision_id}')
                return None
            explanations_added = 0
            if tactician_features is not None:
                features, feature_names = tactician_features
                explanation = await self.explain_model_prediction('tactician', 'main', features, feature_names, market_conditions = market_conditions)
                if explanation:
                    await self.add_explanation_to_trace(decision_id, 'tactician', explanation)
                    explanations_added += 1
            if hmm_features is not None:
                features, feature_names = hmm_features
                explanation = await self.explain_model_prediction('hmm', 'main', features, feature_names, market_conditions = market_conditions)
                if explanation:
                    await self.add_explanation_to_trace(decision_id, 'hmm', explanation)
                    explanations_added += 1
            if sr_features is not None:
                features, feature_names = sr_features
                explanation = await self.explain_model_prediction('sr', 'main', features, feature_names, market_conditions = market_conditions)
                if explanation:
                    await self.add_explanation_to_trace(decision_id, 'sr', explanation)
                    explanations_added += 1
            if analyst_features is not None:
                features, feature_names = analyst_features
                explanation = await self.explain_model_prediction('analyst', 'main', features, feature_names, market_conditions = market_conditions)
                if explanation:
                    await self.add_explanation_to_trace(decision_id, 'analyst', explanation)
                    explanations_added += 1
            self.logger.info(f'📊 Added {explanations_added} explanations to trace {decision_id}')
            final_trace = await self.finalize_trade_decision_trace(decision_id, final_decision, confidence)
            if final_trace:
                self.logger.info(f'✅ Complete trading decision explained: {decision_id}')
            else:
                self.logger.warning(f'⚠️ Failed to finalize complete trading decision: {decision_id}')
            return final_trace
        except Exception as e:
            self.logger.error(f'❌ Failed to explain complete trading decision {decision_id}: {e}')
            return None

    def _extract_market_conditions(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract market conditions from market data."""
        try:
            if market_data.empty:
                return {}
            conditions = {}
            if 'close' in market_data.columns:
                conditions['current_price'] = float(market_data['close'].iloc[-1])
                conditions['price_change'] = float(market_data['close'].pct_change().iloc[-1]) if len(market_data) > 1 else 0.0
            if 'volume' in market_data.columns:
                conditions['current_volume'] = float(market_data['volume'].iloc[-1])
                conditions['volume_change'] = float(market_data['volume'].pct_change().iloc[-1]) if len(market_data) > 1 else 0.0
            if 'volatility_20' in market_data.columns:
                conditions['volatility'] = float(market_data['volatility_20'].iloc[-1])
            for indicator in ['rsi', 'macd', 'bb_position', 'atr', 'adx']:
                if indicator in market_data.columns:
                    conditions[indicator] = float(market_data[indicator].iloc[-1])
            conditions['timestamp'] = datetime.now().isoformat()
            return conditions
        except Exception as e:
            self.logger.error(f'❌ Failed to extract market conditions: {e}')
            return {}

    async def get_decision_trace_summary(self, decision_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a decision trace."""
        try:
            if decision_id in self.active_traces:
                trace = self.active_traces[decision_id]
            else:
                trace = await self._load_decision_trace(decision_id)
            if not trace:
                self.logger.error(f'❌ No trace found for decision {decision_id}')
                return None
            summary = {'decision_id': trace.decision_id, 'timestamp': trace.timestamp.isoformat(), 'decision_type': trace.decision_type, 'final_decision': trace.final_decision, 'confidence': trace.confidence, 'market_conditions': trace.market_conditions, 'explanations_available': {'tactician': trace.tactician_explanation is not None, 'hmm': trace.hmm_explanation is not None, 'sr': trace.sr_explanation is not None, 'analyst': trace.analyst_explanation is not None}, 'top_contributing_factors': trace.top_contributing_factors[:5], 'risk_factors_count': len(trace.risk_factors), 'opportunity_factors_count': len(trace.opportunity_factors)}
            return summary
        except Exception as e:
            self.logger.error(f'❌ Failed to get decision trace summary {decision_id}: {e}')
            return None

    async def _load_decision_trace(self, decision_id: str) -> Optional[TradeDecisionTrace]:
        """Load a decision trace from storage."""
        try:
            traces_storage = Path(self.explain_config.get('traces_storage_path', 'data/decision_traces'))
            trace_files = list(traces_storage.glob(f'decision_trace_{decision_id}_*.json'))
            if not trace_files:
                return None
            latest_trace_file = max(trace_files, key=lambda x: x.stat().st_mtime)
            with open(latest_trace_file, 'r') as f:
                trace_data = json.load(f)
            trace = TradeDecisionTrace(decision_id = trace_data['decision_id'], timestamp = datetime.fromisoformat(trace_data['timestamp']), decision_type = trace_data['decision_type'], final_decision = trace_data['final_decision'], confidence = trace_data['confidence'], top_contributing_factors = trace_data.get('top_contributing_factors', []), risk_factors = trace_data.get('risk_factors', []), opportunity_factors = trace_data.get('opportunity_factors', []), market_conditions = trace_data.get('market_conditions', {}), metadata = trace_data.get('metadata', {}))
            return trace
        except Exception as e:
            self.logger.error(f'❌ Failed to load decision trace {decision_id}: {e}')
            return None

    async def get_model_explanation_summary(self, model_type: str, model_name: str, explanation: Any) -> Optional[str]:
        """Get a human-readable summary of a model explanation."""
        try:
            if model_type == 'tactician':
                return self.tactician_explainer.generate_explanation_summary(explanation)
            elif model_type == 'hmm':
                return self.hmm_explainer.generate_regime_explanation_summary(explanation)
            elif model_type == 'sr':
                return self.sr_explainer.generate_sr_explanation_summary(explanation)
            elif model_type == 'analyst':
                return self.analyst_explainer.generate_analyst_explanation_summary(explanation)
            else:
                self.logger.error(f'❌ Unknown model type for summary: {model_type}')
                return None
        except Exception as e:
            self.logger.error(f'❌ Failed to generate explanation summary: {e}')
            return None

    def get_registered_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about registered models."""
        return {model_type: {model_name: {'initialized': model_info['initialized'], 'has_training_data': model_info['training_data'] is not None} for model_name, model_info in models.items()} for model_type, models in self.registered_models.items()}

    async def cleanup_old_explanations(self, days_to_keep: int = 30) -> int:
        """Clean up old explanation files."""
        try:
            explanations_storage = Path(self.explain_config.get('storage_path', 'data/explanations'))
            traces_storage = Path(self.explain_config.get('traces_storage_path', 'data/decision_traces'))
            cutoff_time = datetime.now().timestamp() - days_to_keep * 24 * 3600
            files_removed = 0
            for storage_path in [explanations_storage, traces_storage]:
                if storage_path.exists():
                    for file_path in storage_path.glob('*.json'):
                        if file_path.stat().st_mtime < cutoff_time:
                            file_path.unlink()
                            files_removed += 1
            self.logger.info(f'🧹 Cleaned up {files_removed} old explanation files')
            return files_removed
        except Exception as e:
            self.logger.error(f'❌ Failed to cleanup old explanations: {e}')
            return 0