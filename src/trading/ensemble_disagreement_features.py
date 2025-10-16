"""
Trading Integration for Ensemble Disagreement Features

This module provides trading-specific integration for disagreement meta-features
that can be called from trading modules to make informed trading decisions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging

# Import meta-feature generator from feature engineering
from src.feature_engineering_roadmap.ensemble_meta_features import EnsembleMetaFeatureGenerator

class TradingDisagreementAnalyzer:
    """
    Trading-specific disagreement analyzer that provides trading signals
    based on ensemble disagreement and uncertainty.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the trading disagreement analyzer.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.meta_feature_generator = EnsembleMetaFeatureGenerator(logger)

        # Trading thresholds for disagreement features
        self.disagreement_thresholds = {
            'high_disagreement': 0.3,  # Above this, avoid trading
            'moderate_disagreement': 0.15,  # Above this, reduce position size
            'confidence_threshold': 0.7,  # Minimum confidence for trading
            'direction_agreement_threshold': 0.7  # Minimum agreement for trading
        }

    def analyze_trading_signal_reliability(
        self,
        ensemble_predictions: Dict[str, Any],
        current_features: pd.DataFrame,
        is_live: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze the reliability of trading signals based on ensemble disagreement.

        Args:
            ensemble_predictions: Dict containing ensemble prediction data
            current_features: Current market features
            is_live: Whether this is for live trading or backtesting

        Returns:
            Dict containing trading signal reliability analysis
        """
        try:
            # Generate disagreement features
            disagreement_features = self.meta_feature_generator.disagreement_calculator.calculate_disagreement_features_for_ensemble(
                ensemble_predictions, is_live=is_live
            )

            # Analyze signal reliability
            reliability_analysis = {
                'signal_reliable': True,
                'recommended_action': 'TRADE',
                'position_size_multiplier': 1.0,
                'confidence_level': 'HIGH',
                'disagreement_level': 'LOW',
                'risk_factors': [],
                'disagreement_features': disagreement_features
            }

            # Check prediction dispersion
            prediction_dispersion = disagreement_features.get('prediction_dispersion', 0.0)
            if prediction_dispersion > self.disagreement_thresholds['high_disagreement']:
                reliability_analysis['signal_reliable'] = False
                reliability_analysis['recommended_action'] = 'AVOID'
                reliability_analysis['confidence_level'] = 'LOW'
                reliability_analysis['disagreement_level'] = 'HIGH'
                reliability_analysis['risk_factors'].append('High prediction dispersion')
            elif prediction_dispersion > self.disagreement_thresholds['moderate_disagreement']:
                reliability_analysis['position_size_multiplier'] = 0.5
                reliability_analysis['confidence_level'] = 'MEDIUM'
                reliability_analysis['disagreement_level'] = 'MODERATE'
                reliability_analysis['risk_factors'].append('Moderate prediction dispersion')

            # Check direction conflict
            direction_conflict = disagreement_features.get('direction_conflict', 0.0)
            long_ratio = disagreement_features.get('long_ratio', 0.5)

            if direction_conflict > self.disagreement_thresholds['high_disagreement']:
                reliability_analysis['signal_reliable'] = False
                reliability_analysis['recommended_action'] = 'AVOID'
                reliability_analysis['confidence_level'] = 'LOW'
                reliability_analysis['risk_factors'].append('High direction conflict')
            elif direction_conflict > self.disagreement_thresholds['moderate_disagreement']:
                reliability_analysis['position_size_multiplier'] *= 0.7
                reliability_analysis['confidence_level'] = 'MEDIUM'
                reliability_analysis['risk_factors'].append('Moderate direction conflict')

            # Check confidence gap
            confidence_gap = disagreement_features.get('confidence_gap', 0.0)
            max_confidence = disagreement_features.get('max_confidence', 0.0)

            if max_confidence < self.disagreement_thresholds['confidence_threshold']:
                reliability_analysis['signal_reliable'] = False
                reliability_analysis['recommended_action'] = 'AVOID'
                reliability_analysis['confidence_level'] = 'LOW'
                reliability_analysis['risk_factors'].append('Low ensemble confidence')
            elif confidence_gap < 0.1:  # Small gap between top predictions
                reliability_analysis['position_size_multiplier'] *= 0.8
                reliability_analysis['confidence_level'] = 'MEDIUM'
                reliability_analysis['risk_factors'].append('Small confidence gap')

            # Check entropy/uncertainty
            uncertainty = disagreement_features.get('uncertainty', 0.0)
            if uncertainty > 0.8:  # High uncertainty
                reliability_analysis['signal_reliable'] = False
                reliability_analysis['recommended_action'] = 'AVOID'
                reliability_analysis['confidence_level'] = 'LOW'
                reliability_analysis['risk_factors'].append('High market uncertainty')
            elif uncertainty > 0.6:  # Moderate uncertainty
                reliability_analysis['position_size_multiplier'] *= 0.6
                reliability_analysis['confidence_level'] = 'MEDIUM'
                reliability_analysis['risk_factors'].append('Moderate market uncertainty')

            # Check pairwise divergence
            avg_divergence = disagreement_features.get('avg_divergence', 0.0)
            if avg_divergence > 0.5:  # High divergence between models
                reliability_analysis['position_size_multiplier'] *= 0.7
                reliability_analysis['confidence_level'] = 'MEDIUM'
                reliability_analysis['risk_factors'].append('High model divergence')

            # Final recommendation
            if reliability_analysis['position_size_multiplier'] < 0.3:
                reliability_analysis['recommended_action'] = 'AVOID'
                reliability_analysis['signal_reliable'] = False

            return reliability_analysis

        except Exception as e:
            self.logger.error(f"Error analyzing trading signal reliability: {e}")
            return {
                'signal_reliable': False,
                'recommended_action': 'AVOID',
                'position_size_multiplier': 0.0,
                'confidence_level': 'LOW',
                'disagreement_level': 'UNKNOWN',
                'risk_factors': ['Analysis failed'],
                'disagreement_features': {}
            }

    def get_trading_recommendation(
        self,
        analyst_predictions: Dict[str, Any],
        tactician_predictions: Dict[str, Any],
        current_features: pd.DataFrame,
        is_live: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive trading recommendation based on both analyst and tactician disagreement.

        Args:
            analyst_predictions: Analyst ensemble predictions
            tactician_predictions: Tactician ensemble predictions
            current_features: Current market features
            is_live: Whether this is for live trading or backtesting

        Returns:
            Dict containing comprehensive trading recommendation
        """
        try:
            # Analyze analyst signal reliability
            analyst_analysis = self.analyze_trading_signal_reliability(
                analyst_predictions, current_features, is_live
            )

            # Analyze tactician signal reliability
            tactician_analysis = self.analyze_trading_signal_reliability(
                tactician_predictions, current_features, is_live
            )

            # Combine analyses
            combined_recommendation = {
                'analyst_analysis': analyst_analysis,
                'tactician_analysis': tactician_analysis,
                'overall_recommendation': 'AVOID',
                'overall_confidence': 'LOW',
                'position_size_multiplier': 0.0,
                'risk_factors': [],
                'trading_decision': 'HOLD'
            }

            # Determine overall recommendation
            analyst_reliable = analyst_analysis['signal_reliable']
            tactician_reliable = tactician_analysis['signal_reliable']

            if analyst_reliable and tactician_reliable:
                combined_recommendation['overall_recommendation'] = 'TRADE'
                combined_recommendation['overall_confidence'] = 'HIGH'
                combined_recommendation['position_size_multiplier'] = min(
                    analyst_analysis['position_size_multiplier'],
                    tactician_analysis['position_size_multiplier']
                )
                combined_recommendation['trading_decision'] = 'EXECUTE'
            elif analyst_reliable or tactician_reliable:
                combined_recommendation['overall_recommendation'] = 'CAUTIOUS_TRADE'
                combined_recommendation['overall_confidence'] = 'MEDIUM'
                combined_recommendation['position_size_multiplier'] = 0.5
                combined_recommendation['trading_decision'] = 'REDUCED_SIZE'
            else:
                combined_recommendation['overall_recommendation'] = 'AVOID'
                combined_recommendation['overall_confidence'] = 'LOW'
                combined_recommendation['position_size_multiplier'] = 0.0
                combined_recommendation['trading_decision'] = 'HOLD'

            # Collect all risk factors
            combined_recommendation['risk_factors'] = (
                analyst_analysis['risk_factors'] + tactician_analysis['risk_factors']
            )

            return combined_recommendation

        except Exception as e:
            self.logger.error(f"Error getting trading recommendation: {e}")
            return {
                'analyst_analysis': {'signal_reliable': False, 'recommended_action': 'AVOID'},
                'tactician_analysis': {'signal_reliable': False, 'recommended_action': 'AVOID'},
                'overall_recommendation': 'AVOID',
                'overall_confidence': 'LOW',
                'position_size_multiplier': 0.0,
                'risk_factors': ['Analysis failed'],
                'trading_decision': 'HOLD'
            }

    def update_disagreement_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update disagreement thresholds for trading decisions.

        Args:
            new_thresholds: Dict containing new threshold values
        """
        try:
            self.disagreement_thresholds.update(new_thresholds)
            self.logger.info(f"Updated disagreement thresholds: {self.disagreement_thresholds}")
        except Exception as e:
            self.logger.error(f"Error updating disagreement thresholds: {e}")

    def get_disagreement_summary(self, disagreement_features: Dict[str, float]) -> str:
        """
        Get a human-readable summary of disagreement features.

        Args:
            disagreement_features: Dict containing disagreement features

        Returns:
            String summary of disagreement analysis
        """
        try:
            summary_parts = []

            # Prediction dispersion
            dispersion = disagreement_features.get('prediction_dispersion', 0.0)
            if dispersion > 0.3:
                summary_parts.append("High prediction dispersion - models disagree strongly")
            elif dispersion > 0.15:
                summary_parts.append("Moderate prediction dispersion - some model disagreement")
            else:
                summary_parts.append("Low prediction dispersion - models agree well")

            # Direction conflict
            conflict = disagreement_features.get('direction_conflict', 0.0)
            if conflict > 0.3:
                summary_parts.append("High direction conflict - mixed long/short signals")
            elif conflict > 0.15:
                summary_parts.append("Moderate direction conflict - some directional disagreement")
            else:
                summary_parts.append("Low direction conflict - consistent directional signals")

            # Confidence gap
            gap = disagreement_features.get('confidence_gap', 0.0)
            if gap > 0.3:
                summary_parts.append("High confidence gap - clear top prediction")
            elif gap > 0.1:
                summary_parts.append("Moderate confidence gap - somewhat clear top prediction")
            else:
                summary_parts.append("Low confidence gap - uncertain top prediction")

            # Uncertainty
            uncertainty = disagreement_features.get('uncertainty', 0.0)
            if uncertainty > 0.8:
                summary_parts.append("High market uncertainty - scattered beliefs")
            elif uncertainty > 0.6:
                summary_parts.append("Moderate market uncertainty - some scattered beliefs")
            else:
                summary_parts.append("Low market uncertainty - focused beliefs")

            return " | ".join(summary_parts)

        except Exception as e:
            self.logger.error(f"Error generating disagreement summary: {e}")
            return "Unable to analyze disagreement features"
