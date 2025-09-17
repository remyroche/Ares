"""
Trade Analyzer

Detailed analysis of individual trades with ML explanations,
feature importance analysis, and trade quality assessment.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from src.utils.logger import system_logger
from src.utils.tprint import tprint_info, tprint_warning, tprint_error, tprint_success, tprint_structured, LogLevel
from ..monitoring.comprehensive_trade_monitor import DetailedTradeMetrics
from ..utils.error_handling import TradingError, TradingErrorSeverity, trading_error_handler
from ..utils.helpers import format_trading_metrics

logger = system_logger.getChild('TradeAnalyzer')

class TradeAnalyzer:
    """
    Detailed trade analyzer for individual trade analysis.
    
    Provides:
    - Trade quality assessment
    - ML model contribution analysis
    - SHAP/LIME explanation interpretation
    - Risk-return analysis
    - Timing analysis
    - Execution quality assessment
    """
    
    def __init__(self):
        self.logger = logger.getChild('TradeAnalyzer')
    
    @trading_error_handler(
        error_types=(Exception,),
        severity=TradingErrorSeverity.LOW,
        raise_on_error=False
    )
    async def analyze_trade(
        self,
        trade: DetailedTradeMetrics,
        include_explanations: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a single trade.
        
        Args:
            trade: Detailed trade metrics
            include_explanations: Whether to include SHAP/LIME analysis
            
        Returns:
            Comprehensive trade analysis
        """
        try:
            tprint_info(f"🔍 Analyzing trade: {trade.trade_id}")
            
            analysis = {
                'trade_overview': await self._analyze_trade_overview(trade),
                'performance_analysis': await self._analyze_trade_performance(trade),
                'model_analysis': await self._analyze_model_contributions(trade),
                'risk_analysis': await self._analyze_trade_risk(trade),
                'timing_analysis': await self._analyze_trade_timing(trade),
                'execution_analysis': await self._analyze_trade_execution(trade),
                'market_context_analysis': await self._analyze_market_context(trade)
            }
            
            if include_explanations:
                analysis['explainability_analysis'] = await self._analyze_explanations(trade)
            
            # Calculate overall trade score
            analysis['trade_quality_score'] = await self._calculate_trade_quality_score(trade, analysis)
            
            tprint_success(f"✅ Completed analysis for trade: {trade.trade_id}")
            
            return analysis
            
        except Exception as e:
            tprint_error(f"❌ Failed to analyze trade {trade.trade_id}: {e}")
            return {'error': str(e)}
    
    async def _analyze_trade_overview(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Analyze basic trade overview."""
        return {
            'trade_id': trade.trade_id,
            'timestamp': trade.timestamp.isoformat(),
            'symbol': trade.symbol,
            'action': trade.action,
            'quantity': trade.quantity,
            'price': trade.price,
            'trading_mode': trade.trading_mode,
            'exchange': trade.exchange,
            'duration_minutes': trade.duration_minutes or 0.0
        }
    
    async def _analyze_trade_performance(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Analyze trade performance metrics."""
        try:
            performance = {
                'pnl_absolute': trade.pnl_absolute or 0.0,
                'pnl_percentage': trade.pnl_percentage or 0.0,
                'entry_price': trade.entry_price or trade.price,
                'exit_price': trade.exit_price,
                'max_favorable_excursion': trade.max_favorable_excursion,
                'max_adverse_excursion': trade.max_adverse_excursion
            }
            
            # Calculate additional performance metrics
            if trade.entry_price and trade.exit_price:
                price_change = (trade.exit_price - trade.entry_price) / trade.entry_price
                performance['price_change_percentage'] = price_change
                
                # Risk-adjusted return
                if trade.portfolio_risk > 0:
                    performance['risk_adjusted_return'] = price_change / trade.portfolio_risk
                
                # Return per unit of time
                if trade.duration_minutes and trade.duration_minutes > 0:
                    performance['return_per_hour'] = (trade.pnl_percentage or 0.0) / (trade.duration_minutes / 60)
            
            # Performance classification
            if trade.pnl_absolute:
                if trade.pnl_absolute > 0:
                    performance['outcome'] = 'winning'
                    performance['outcome_quality'] = 'excellent' if trade.pnl_percentage and trade.pnl_percentage > 0.02 else 'good'
                elif trade.pnl_absolute < 0:
                    performance['outcome'] = 'losing'
                    performance['outcome_quality'] = 'poor' if trade.pnl_percentage and trade.pnl_percentage < -0.02 else 'acceptable'
                else:
                    performance['outcome'] = 'break_even'
                    performance['outcome_quality'] = 'neutral'
            
            return performance
            
        except Exception as e:
            tprint_error(f"❌ Failed to analyze trade performance: {e}")
            return {}
    
    async def _analyze_model_contributions(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Analyze ML model contributions to the trade decision."""
        try:
            model_analysis = {}
            
            for model_id, model_info in trade.models_used.items():
                prediction = trade.model_predictions.get(model_id, 0.0)
                confidence = trade.model_confidences.get(model_id, 0.0)
                weight = trade.model_weights.get(model_id, 0.0)
                
                # Calculate model contribution score
                contribution_score = prediction * confidence * weight
                
                # Assess model performance for this trade
                model_performance = 'unknown'
                if trade.pnl_absolute is not None:
                    if confidence > 0.7 and trade.pnl_absolute > 0:
                        model_performance = 'excellent'
                    elif confidence > 0.5 and trade.pnl_absolute > 0:
                        model_performance = 'good'
                    elif confidence < 0.5 and trade.pnl_absolute > 0:
                        model_performance = 'lucky'
                    elif confidence > 0.7 and trade.pnl_absolute < 0:
                        model_performance = 'poor'
                    elif confidence < 0.5 and trade.pnl_absolute < 0:
                        model_performance = 'expected'
                    else:
                        model_performance = 'neutral'
                
                model_analysis[model_id] = {
                    'model_type': model_info.get('model_type', 'unknown'),
                    'prediction': prediction,
                    'confidence': confidence,
                    'weight': weight,
                    'contribution_score': contribution_score,
                    'performance_assessment': model_performance,
                    'version': trade.model_versions.get(model_id, 'unknown')
                }
            
            # Overall model consensus
            all_predictions = list(trade.model_predictions.values())
            all_confidences = list(trade.model_confidences.values())
            
            consensus_analysis = {
                'prediction_variance': np.var(all_predictions) if len(all_predictions) > 1 else 0.0,
                'confidence_variance': np.var(all_confidences) if len(all_confidences) > 1 else 0.0,
                'model_agreement': 1.0 - (np.var(all_predictions) if len(all_predictions) > 1 else 0.0),
                'weighted_prediction': sum(p * trade.model_weights.get(mid, 1.0) for mid, p in trade.model_predictions.items()) / sum(trade.model_weights.values()) if trade.model_weights else np.mean(all_predictions) if all_predictions else 0.0
            }
            
            return {
                'individual_models': model_analysis,
                'consensus_analysis': consensus_analysis,
                'ensemble_effectiveness': await self._assess_ensemble_effectiveness(trade)
            }
            
        except Exception as e:
            tprint_error(f"❌ Failed to analyze model contributions: {e}")
            return {}
    
    async def _analyze_explanations(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Analyze SHAP and LIME explanations."""
        try:
            explanation_analysis = {
                'shap_analysis': {},
                'lime_analysis': {},
                'feature_consensus': {},
                'explanation_quality': {}
            }
            
            # Analyze SHAP explanations
            if trade.shap_explanations:
                for model_id, shap_values in trade.shap_explanations.items():
                    # Top positive and negative features
                    sorted_features = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)
                    
                    explanation_analysis['shap_analysis'][model_id] = {
                        'top_positive_features': [(f, v) for f, v in sorted_features if v > 0][:5],
                        'top_negative_features': [(f, v) for f, v in sorted_features if v < 0][:5],
                        'feature_count': len(shap_values),
                        'total_importance': sum(abs(v) for v in shap_values.values()),
                        'explanation_strength': max(abs(v) for v in shap_values.values()) if shap_values else 0.0
                    }
            
            # Analyze LIME explanations
            if trade.lime_explanations:
                for model_id, lime_values in trade.lime_explanations.items():
                    sorted_features = sorted(lime_values.items(), key=lambda x: abs(x[1]), reverse=True)
                    
                    explanation_analysis['lime_analysis'][model_id] = {
                        'top_features': sorted_features[:10],
                        'feature_count': len(lime_values),
                        'total_importance': sum(abs(v) for v in lime_values.values()),
                        'explanation_strength': max(abs(v) for v in lime_values.values()) if lime_values else 0.0
                    }
            
            # Feature consensus across models
            if trade.feature_importance:
                sorted_overall = sorted(trade.feature_importance.items(), key=lambda x: x[1], reverse=True)
                explanation_analysis['feature_consensus'] = {
                    'top_features': sorted_overall[:10],
                    'most_important_feature': sorted_overall[0] if sorted_overall else None,
                    'feature_diversity': len(trade.feature_importance)
                }
            
            # Explanation quality assessment
            explanation_analysis['explanation_quality'] = {
                'shap_coverage': len(trade.shap_explanations) / len(trade.models_used) if trade.models_used else 0.0,
                'lime_coverage': len(trade.lime_explanations) / len(trade.models_used) if trade.models_used else 0.0,
                'explanation_consistency': await self._assess_explanation_consistency(trade)
            }
            
            return explanation_analysis
            
        except Exception as e:
            tprint_error(f"❌ Failed to analyze explanations: {e}")
            return {}
    
    async def _analyze_trade_risk(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Analyze trade-specific risk metrics."""
        try:
            risk_analysis = {
                'position_risk': {
                    'position_size': trade.position_size,
                    'leverage': trade.leverage,
                    'portfolio_risk': trade.portfolio_risk,
                    'risk_per_trade': trade.risk_per_trade
                },
                'market_risk': {
                    'volatility_estimate': trade.volatility_estimate,
                    'var_95': trade.var_95,
                    'expected_shortfall': trade.expected_shortfall,
                    'max_drawdown_risk': trade.max_drawdown_risk
                },
                'execution_risk': {
                    'slippage': trade.slippage,
                    'execution_quality': trade.execution_quality,
                    'timing_quality': trade.timing_quality
                }
            }
            
            # Risk assessment
            total_risk_score = (
                trade.portfolio_risk * 0.4 +
                trade.volatility_estimate * 0.3 +
                (1.0 - trade.execution_quality) * 0.3
            )
            
            risk_analysis['risk_assessment'] = {
                'total_risk_score': total_risk_score,
                'risk_level': 'high' if total_risk_score > 0.05 else 'medium' if total_risk_score > 0.02 else 'low',
                'risk_reward_ratio': abs(trade.pnl_percentage / total_risk_score) if total_risk_score > 0 and trade.pnl_percentage else 0.0
            }
            
            return risk_analysis
            
        except Exception as e:
            tprint_error(f"❌ Failed to analyze trade risk: {e}")
            return {}
    
    async def _analyze_trade_timing(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Analyze trade timing quality."""
        try:
            timing_analysis = {
                'entry_timing': {
                    'timestamp': trade.timestamp.isoformat(),
                    'hour_of_day': trade.timestamp.hour,
                    'day_of_week': trade.timestamp.strftime('%A'),
                    'timing_quality_score': trade.timing_quality
                },
                'market_timing': {
                    'regime_type': trade.regime_type,
                    'regime_confidence': trade.regime_confidence,
                    'regime_stability': trade.regime_stability
                }
            }
            
            # Assess timing quality
            timing_score = trade.timing_quality
            
            if trade.regime_confidence > 0.8:
                timing_score += 0.1  # Bonus for high regime confidence
            
            if trade.signal_confidence > 0.8:
                timing_score += 0.1  # Bonus for high signal confidence
            
            timing_analysis['timing_assessment'] = {
                'overall_timing_score': min(timing_score, 1.0),
                'timing_quality': 'excellent' if timing_score > 0.9 else 'good' if timing_score > 0.7 else 'fair' if timing_score > 0.5 else 'poor'
            }
            
            return timing_analysis
            
        except Exception as e:
            tprint_error(f"❌ Failed to analyze trade timing: {e}")
            return {}
    
    async def _analyze_trade_execution(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Analyze trade execution quality."""
        try:
            execution_analysis = {
                'execution_metrics': {
                    'execution_time_ms': trade.execution_time_ms,
                    'slippage': trade.slippage,
                    'commission': trade.commission,
                    'execution_quality': trade.execution_quality
                },
                'cost_analysis': {
                    'total_costs': (trade.slippage or 0.0) + (trade.commission or 0.0),
                    'cost_percentage': ((trade.slippage or 0.0) + (trade.commission or 0.0)) / (trade.quantity * trade.price) if trade.quantity * trade.price > 0 else 0.0
                }
            }
            
            # Execution quality assessment
            execution_score = trade.execution_quality
            
            # Adjust for costs
            if trade.slippage and trade.slippage > 0.002:  # High slippage
                execution_score -= 0.1
            
            if trade.execution_time_ms and trade.execution_time_ms > 1000:  # Slow execution
                execution_score -= 0.05
            
            execution_analysis['execution_assessment'] = {
                'adjusted_execution_score': max(0.0, execution_score),
                'execution_quality': 'excellent' if execution_score > 0.9 else 'good' if execution_score > 0.7 else 'fair' if execution_score > 0.5 else 'poor'
            }
            
            return execution_analysis
            
        except Exception as e:
            tprint_error(f"❌ Failed to analyze trade execution: {e}")
            return {}
    
    async def _analyze_market_context(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Analyze market context during the trade."""
        try:
            context_analysis = {
                'market_conditions': trade.market_conditions,
                'technical_indicators': trade.technical_indicators,
                'support_resistance': trade.support_resistance_levels
            }
            
            # Market favorability assessment
            favorability_score = 0.5  # Neutral starting point
            
            # Assess volatility favorability
            if 'volatility' in trade.market_conditions:
                volatility = trade.market_conditions['volatility']
                if trade.action in ['buy', 'sell']:
                    # Higher volatility can be favorable for directional trades
                    favorability_score += min(volatility * 10, 0.2)
                else:
                    # Lower volatility better for hold decisions
                    favorability_score += max(0.2 - volatility * 10, 0.0)
            
            # Assess trend favorability
            if 'trend_direction' in trade.market_conditions:
                trend = trade.market_conditions['trend_direction']
                if (trade.action == 'buy' and trend == 'up') or (trade.action == 'sell' and trend == 'down'):
                    favorability_score += 0.1
            
            context_analysis['market_favorability'] = {
                'favorability_score': min(favorability_score, 1.0),
                'favorability_level': 'high' if favorability_score > 0.7 else 'medium' if favorability_score > 0.4 else 'low'
            }
            
            return context_analysis
            
        except Exception as e:
            tprint_error(f"❌ Failed to analyze market context: {e}")
            return {}
    
    async def _assess_ensemble_effectiveness(self, trade: DetailedTradeMetrics) -> Dict[str, Any]:
        """Assess how effectively the ensemble performed."""
        try:
            if not trade.models_used:
                return {'effectiveness': 'no_models'}
            
            # Model diversity
            model_types = set(info.get('model_type', 'unknown') for info in trade.models_used.values())
            diversity_score = len(model_types) / len(trade.models_used)
            
            # Prediction agreement
            predictions = list(trade.model_predictions.values())
            prediction_std = np.std(predictions) if len(predictions) > 1 else 0.0
            agreement_score = 1.0 - min(prediction_std, 1.0)
            
            # Weight distribution
            weights = list(trade.model_weights.values())
            weight_entropy = -sum(w * np.log(w + 1e-8) for w in weights if w > 0) if weights else 0.0
            
            # Overall ensemble effectiveness
            effectiveness_score = (diversity_score * 0.3 + agreement_score * 0.4 + min(weight_entropy, 1.0) * 0.3)
            
            return {
                'diversity_score': diversity_score,
                'agreement_score': agreement_score,
                'weight_entropy': weight_entropy,
                'effectiveness_score': effectiveness_score,
                'effectiveness_level': 'high' if effectiveness_score > 0.7 else 'medium' if effectiveness_score > 0.4 else 'low'
            }
            
        except Exception as e:
            tprint_error(f"❌ Failed to assess ensemble effectiveness: {e}")
            return {}
    
    async def _assess_explanation_consistency(self, trade: DetailedTradeMetrics) -> float:
        """Assess consistency between SHAP and LIME explanations."""
        try:
            if not trade.shap_explanations or not trade.lime_explanations:
                return 0.0
            
            consistency_scores = []
            
            # Compare explanations for each model
            for model_id in trade.shap_explanations:
                if model_id in trade.lime_explanations:
                    shap_values = trade.shap_explanations[model_id]
                    lime_values = trade.lime_explanations[model_id]
                    
                    # Find common features
                    common_features = set(shap_values.keys()) & set(lime_values.keys())
                    
                    if common_features:
                        shap_ranks = {f: i for i, (f, _) in enumerate(sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True))}
                        lime_ranks = {f: i for i, (f, _) in enumerate(sorted(lime_values.items(), key=lambda x: abs(x[1]), reverse=True))}
                        
                        # Calculate rank correlation
                        rank_differences = [abs(shap_ranks[f] - lime_ranks[f]) for f in common_features]
                        consistency = 1.0 - (np.mean(rank_differences) / len(common_features)) if rank_differences else 0.0
                        consistency_scores.append(max(0.0, consistency))
            
            return np.mean(consistency_scores) if consistency_scores else 0.0
            
        except Exception as e:
            tprint_error(f"❌ Failed to assess explanation consistency: {e}")
            return 0.0
    
    async def _calculate_trade_quality_score(
        self,
        trade: DetailedTradeMetrics,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate overall trade quality score."""
        try:
            # Component scores
            performance_score = 0.5
            if trade.pnl_percentage:
                performance_score = 0.5 + min(trade.pnl_percentage * 5, 0.5)  # Scale PnL to score
            
            model_score = analysis.get('model_analysis', {}).get('consensus_analysis', {}).get('effectiveness_score', 0.5)
            risk_score = 1.0 - analysis.get('risk_analysis', {}).get('risk_assessment', {}).get('total_risk_score', 0.5)
            timing_score = analysis.get('timing_analysis', {}).get('timing_assessment', {}).get('overall_timing_score', 0.5)
            execution_score = analysis.get('execution_analysis', {}).get('execution_assessment', {}).get('adjusted_execution_score', 0.5)
            
            # Weighted overall score
            overall_score = (
                performance_score * 0.3 +
                model_score * 0.25 +
                risk_score * 0.2 +
                timing_score * 0.15 +
                execution_score * 0.1
            )
            
            return {
                'component_scores': {
                    'performance': performance_score,
                    'model_effectiveness': model_score,
                    'risk_management': risk_score,
                    'timing': timing_score,
                    'execution': execution_score
                },
                'overall_score': overall_score,
                'quality_grade': 'A' if overall_score > 0.8 else 'B' if overall_score > 0.6 else 'C' if overall_score > 0.4 else 'D',
                'trade_classification': await self._classify_trade_quality(overall_score, trade)
            }
            
        except Exception as e:
            tprint_error(f"❌ Failed to calculate trade quality score: {e}")
            return {}
    
    async def _classify_trade_quality(self, score: float, trade: DetailedTradeMetrics) -> str:
        """Classify trade quality based on score and outcome."""
        try:
            if score > 0.8:
                if trade.pnl_absolute and trade.pnl_absolute > 0:
                    return "excellent_winner"
                else:
                    return "excellent_process_poor_outcome"
            elif score > 0.6:
                if trade.pnl_absolute and trade.pnl_absolute > 0:
                    return "good_winner"
                else:
                    return "good_process_poor_outcome"
            elif score > 0.4:
                if trade.pnl_absolute and trade.pnl_absolute > 0:
                    return "lucky_winner"
                else:
                    return "fair_process_fair_outcome"
            else:
                if trade.pnl_absolute and trade.pnl_absolute > 0:
                    return "very_lucky_winner"
                else:
                    return "poor_process_poor_outcome"
                    
        except Exception:
            return "unknown"

# Global instance
trade_analyzer = TradeAnalyzer()

# Convenience function
async def analyze_trade_performance(
    trade: DetailedTradeMetrics,
    include_explanations: bool = True
) -> Dict[str, Any]:
    """Analyze individual trade performance."""
    return await trade_analyzer.analyze_trade(trade, include_explanations)