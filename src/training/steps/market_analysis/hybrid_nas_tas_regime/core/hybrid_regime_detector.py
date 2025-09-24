"""
Hybrid Regime Detector

Main component that combines TAS and NAS regime detection outputs to create
a coherent regime modeling system with economic and financial relevance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass, field

from ..config.hybrid_config import (
    HybridRegimeConfig, HybridNASConfig, HybridTASConfig,
    RegimeType, EconomicRegimeType, FinancialRegimeType
)
from ..integration.tas_integration import TASIntegration
from ..integration.nas_integration import NASIntegration
from ..clustering.hybrid_clusterer import HybridClusterer
from ..modeling.regime_modeler import RegimeModeler


@dataclass
class HybridRegimeResult:
    """Result from hybrid regime detection."""
    success: bool
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    regime_labels: List[str]
    economic_regime_predictions: Optional[np.ndarray] = None
    financial_regime_predictions: Optional[np.ndarray] = None
    micro_regime_predictions: Optional[np.ndarray] = None
    regime_stability_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    regime_transition_probabilities: np.ndarray = field(default_factory=lambda: np.array([]))
    economic_significance_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    financial_significance_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    trading_viability_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    uncertainty_estimates: np.ndarray = field(default_factory=lambda: np.array([]))
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class HybridRegimeDetector:
    """
    Hybrid Regime Detector that combines TAS and NAS regime detection.
    
    This is the main component that:
    1. Integrates TAS and NAS regime detection outputs
    2. Creates coherent regime modeling with economic and financial relevance
    3. Performs clustering based on combined TAS & NAS inputs
    4. Tags existing data with regime information
    5. Replaces hmm_clustering functionality
    """
    
    def __init__(self, 
                 config: HybridRegimeConfig,
                 nas_config: Optional[HybridNASConfig] = None,
                 tas_config: Optional[HybridTASConfig] = None):
        """
        Initialize Hybrid Regime Detector.
        
        Args:
            config: Main hybrid regime configuration
            nas_config: NAS-specific configuration
            tas_config: TAS-specific configuration
        """
        self.config = config
        self.nas_config = nas_config or HybridNASConfig()
        self.tas_config = tas_config or HybridTASConfig()
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize integration components
        self.tas_integration = TASIntegration(self.tas_config)
        self.nas_integration = NASIntegration(self.nas_config)
        
        # Initialize clustering and modeling components
        self.clusterer = HybridClusterer(config)
        self.modeler = RegimeModeler(config)
        
        # Initialize regime state
        self.current_regimes = None
        self.regime_history = []
        self.performance_history = []
        
        self.logger.info("✅ Hybrid Regime Detector initialized")
        self.logger.info(f"🔗 Integration strategy: {config.integration_strategy.value}")
        self.logger.info(f"📊 NAS weight: {config.nas_weight}")
        self.logger.info(f"🌳 TAS weight: {config.tas_weight}")
        self.logger.info(f"🏛️ Economic modeling: {config.economic_modeling_enabled}")
        self.logger.info(f"💰 Financial modeling: {config.financial_modeling_enabled}")
    
    def detect_regimes(self,
                      market_data: Union[pd.DataFrame, np.ndarray],
                      tas_inputs: Optional[Dict[str, Any]] = None,
                      nas_inputs: Optional[Dict[str, Any]] = None,
                      timestamps: Optional[np.ndarray] = None,
                      enable_economic_analysis: bool = True,
                      enable_financial_analysis: bool = True) -> HybridRegimeResult:
        """
        Detect market regimes using hybrid TAS and NAS approach.
        
        Args:
            market_data: Market data (OHLCV or features)
            tas_inputs: TAS regime detection inputs
            nas_inputs: NAS regime detection inputs
            timestamps: Optional timestamps
            enable_economic_analysis: Whether to perform economic analysis
            enable_financial_analysis: Whether to perform financial analysis
            
        Returns:
            HybridRegimeResult with regime detection results
        """
        start_time = time.time()
        self.logger.info("🚀 Starting hybrid regime detection")
        
        try:
            # Step 1: Get TAS regime detection results
            tas_results = self._get_tas_regime_results(market_data, tas_inputs)
            
            # Step 2: Get NAS regime detection results
            nas_results = self._get_nas_regime_results(market_data, nas_inputs)
            
            # Step 3: Integrate TAS and NAS results
            integrated_results = self._integrate_tas_nas_results(tas_results, nas_results)
            
            # Step 4: Perform hybrid clustering
            clustering_results = self._perform_hybrid_clustering(
                market_data, integrated_results, timestamps
            )
            
            # Step 5: Create regime model
            regime_model = self._create_regime_model(
                market_data, clustering_results, integrated_results
            )
            
            # Step 6: Economic analysis (if enabled)
            economic_results = None
            if enable_economic_analysis and self.config.economic_modeling_enabled:
                economic_results = self._perform_economic_analysis(
                    market_data, regime_model, timestamps
                )
            
            # Step 7: Financial analysis (if enabled)
            financial_results = None
            if enable_financial_analysis and self.config.financial_modeling_enabled:
                financial_results = self._perform_financial_analysis(
                    market_data, regime_model, timestamps
                )
            
            # Step 8: Generate final regime predictions
            final_results = self._generate_final_predictions(
                regime_model, economic_results, financial_results
            )
            
            execution_time = time.time() - start_time
            
            # Create result object
            result = HybridRegimeResult(
                success=True,
                regime_predictions=final_results['regime_predictions'],
                regime_probabilities=final_results['regime_probabilities'],
                regime_labels=final_results['regime_labels'],
                economic_regime_predictions=economic_results['predictions'] if economic_results else None,
                financial_regime_predictions=financial_results['predictions'] if financial_results else None,
                micro_regime_predictions=final_results.get('micro_regime_predictions'),
                regime_stability_scores=final_results['stability_scores'],
                regime_transition_probabilities=final_results['transition_probabilities'],
                economic_significance_scores=economic_results['significance_scores'] if economic_results else np.array([]),
                financial_significance_scores=financial_results['significance_scores'] if financial_results else np.array([]),
                trading_viability_scores=final_results['trading_viability_scores'],
                uncertainty_estimates=final_results['uncertainty_estimates'],
                execution_time=execution_time,
                metadata={
                    'tas_results': tas_results,
                    'nas_results': nas_results,
                    'integrated_results': integrated_results,
                    'clustering_results': clustering_results,
                    'regime_model': regime_model,
                    'economic_results': economic_results,
                    'financial_results': financial_results
                }
            )
            
            # Update internal state
            self.current_regimes = result.regime_predictions
            self.regime_history.append(result)
            
            self.logger.info(f"✅ Hybrid regime detection completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Detected {len(set(result.regime_predictions))} regimes")
            self.logger.info(f"🎯 Average stability: {np.mean(result.regime_stability_scores):.3f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Hybrid regime detection failed: {e}")
            
            return HybridRegimeResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                regime_labels=[],
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _get_tas_regime_results(self, 
                                market_data: Union[pd.DataFrame, np.ndarray],
                                tas_inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get TAS regime detection results."""
        self.logger.info("🌳 Getting TAS regime detection results")
        
        if tas_inputs is not None:
            # Use provided TAS inputs
            return tas_inputs
        else:
            # Perform TAS regime detection
            return self.tas_integration.detect_regimes(market_data)
    
    def _get_nas_regime_results(self, 
                                market_data: Union[pd.DataFrame, np.ndarray],
                                nas_inputs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Get NAS regime detection results."""
        self.logger.info("🧠 Getting NAS regime detection results")
        
        if nas_inputs is not None:
            # Use provided NAS inputs
            return nas_inputs
        else:
            # Perform NAS regime detection
            return self.nas_integration.detect_regimes(market_data)
    
    def _integrate_tas_nas_results(self, 
                                   tas_results: Dict[str, Any],
                                   nas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate TAS and NAS regime detection results."""
        self.logger.info("🔗 Integrating TAS and NAS results")
        
        # Extract regime predictions and probabilities
        tas_predictions = tas_results.get('regime_predictions', np.array([]))
        nas_predictions = nas_results.get('regime_predictions', np.array([]))
        tas_probabilities = tas_results.get('regime_probabilities', np.array([]))
        nas_probabilities = nas_results.get('regime_probabilities', np.array([]))
        
        # Get weights (adaptive if enabled)
        if self.config.adaptive_weighting:
            nas_weight, tas_weight = self._calculate_adaptive_weights(
                tas_results, nas_results
            )
        else:
            nas_weight = self.config.nas_weight
            tas_weight = self.config.tas_weight
        
        # Integrate based on strategy
        if self.config.integration_strategy.value == "weighted_average":
            integrated_predictions = self._weighted_average_integration(
                tas_predictions, nas_predictions, tas_weight, nas_weight
            )
            integrated_probabilities = self._weighted_average_integration(
                tas_probabilities, nas_probabilities, tas_weight, nas_weight
            )
        elif self.config.integration_strategy.value == "ensemble":
            integrated_predictions = self._ensemble_integration(
                tas_predictions, nas_predictions, tas_weight, nas_weight
            )
            integrated_probabilities = self._ensemble_integration(
                tas_probabilities, nas_probabilities, tas_weight, nas_weight
            )
        elif self.config.integration_strategy.value == "hierarchical":
            integrated_predictions = self._hierarchical_integration(
                tas_predictions, nas_predictions, tas_weight, nas_weight
            )
            integrated_probabilities = self._hierarchical_integration(
                tas_probabilities, nas_probabilities, tas_weight, nas_weight
            )
        else:  # adaptive
            integrated_predictions = self._adaptive_integration(
                tas_predictions, nas_predictions, tas_weight, nas_weight
            )
            integrated_probabilities = self._adaptive_integration(
                tas_probabilities, nas_probabilities, tas_weight, nas_weight
            )
        
        return {
            'integrated_predictions': integrated_predictions,
            'integrated_probabilities': integrated_probabilities,
            'tas_predictions': tas_predictions,
            'nas_predictions': nas_predictions,
            'tas_probabilities': tas_probabilities,
            'nas_probabilities': nas_probabilities,
            'nas_weight': nas_weight,
            'tas_weight': tas_weight,
            'integration_confidence': self._calculate_integration_confidence(
                tas_results, nas_results
            )
        }
    
    def _calculate_adaptive_weights(self, 
                                   tas_results: Dict[str, Any],
                                   nas_results: Dict[str, Any]) -> Tuple[float, float]:
        """Calculate adaptive weights based on performance."""
        # Get performance metrics
        tas_performance = tas_results.get('performance_score', 0.5)
        nas_performance = nas_results.get('performance_score', 0.5)
        
        # Calculate weights based on performance
        total_performance = tas_performance + nas_performance
        if total_performance > 0:
            nas_weight = nas_performance / total_performance
            tas_weight = tas_performance / total_performance
        else:
            nas_weight = self.config.nas_weight
            tas_weight = self.config.tas_weight
        
        # Apply learning rate for smooth adaptation
        nas_weight = (1 - self.config.weight_adaptation_rate) * self.config.nas_weight + \
                     self.config.weight_adaptation_rate * nas_weight
        tas_weight = (1 - self.config.weight_adaptation_rate) * self.config.tas_weight + \
                     self.config.weight_adaptation_rate * tas_weight
        
        # Normalize weights
        total_weight = nas_weight + tas_weight
        nas_weight /= total_weight
        tas_weight /= total_weight
        
        return nas_weight, tas_weight
    
    def _weighted_average_integration(self, 
                                     tas_data: np.ndarray,
                                     nas_data: np.ndarray,
                                     tas_weight: float,
                                     nas_weight: float) -> np.ndarray:
        """Weighted average integration of TAS and NAS results."""
        if len(tas_data) == 0 and len(nas_data) == 0:
            return np.array([])
        elif len(tas_data) == 0:
            return nas_data
        elif len(nas_data) == 0:
            return tas_data
        else:
            # Ensure same length
            min_len = min(len(tas_data), len(nas_data))
            tas_data = tas_data[:min_len]
            nas_data = nas_data[:min_len]
            
            return tas_weight * tas_data + nas_weight * nas_data
    
    def _ensemble_integration(self, 
                              tas_data: np.ndarray,
                              nas_data: np.ndarray,
                              tas_weight: float,
                              nas_weight: float) -> np.ndarray:
        """Ensemble integration of TAS and NAS results."""
        if len(tas_data) == 0 and len(nas_data) == 0:
            return np.array([])
        elif len(tas_data) == 0:
            return nas_data
        elif len(nas_data) == 0:
            return tas_data
        else:
            # Use voting for discrete predictions, weighted average for probabilities
            if tas_data.dtype in [np.int32, np.int64] or nas_data.dtype in [np.int32, np.int64]:
                # Voting for discrete predictions
                return self._ensemble_voting(tas_data, nas_data, tas_weight, nas_weight)
            else:
                # Weighted average for continuous values
                return self._weighted_average_integration(tas_data, nas_data, tas_weight, nas_weight)
    
    def _ensemble_voting(self, 
                         tas_data: np.ndarray,
                         nas_data: np.ndarray,
                         tas_weight: float,
                         nas_weight: float) -> np.ndarray:
        """Ensemble voting for discrete predictions."""
        min_len = min(len(tas_data), len(nas_data))
        tas_data = tas_data[:min_len]
        nas_data = nas_data[:min_len]
        
        # Weighted voting
        votes = np.zeros(min_len)
        votes += tas_weight * tas_data
        votes += nas_weight * nas_data
        
        # Round to nearest integer for discrete predictions
        return np.round(votes).astype(int)
    
    def _hierarchical_integration(self, 
                                  tas_data: np.ndarray,
                                  nas_data: np.ndarray,
                                  tas_weight: float,
                                  nas_weight: float) -> np.ndarray:
        """Hierarchical integration of TAS and NAS results."""
        if len(tas_data) == 0 and len(nas_data) == 0:
            return np.array([])
        elif len(tas_data) == 0:
            return nas_data
        elif len(nas_data) == 0:
            return tas_data
        else:
            # Use NAS as primary, TAS as secondary
            if tas_weight > nas_weight:
                primary_data = tas_data
                secondary_data = nas_data
                primary_weight = tas_weight
                secondary_weight = nas_weight
            else:
                primary_data = nas_data
                secondary_data = tas_data
                primary_weight = nas_weight
                secondary_weight = tas_weight
            
            min_len = min(len(primary_data), len(secondary_data))
            primary_data = primary_data[:min_len]
            secondary_data = secondary_data[:min_len]
            
            # Hierarchical combination
            return primary_weight * primary_data + secondary_weight * secondary_data
    
    def _adaptive_integration(self, 
                              tas_data: np.ndarray,
                              nas_data: np.ndarray,
                              tas_weight: float,
                              nas_weight: float) -> np.ndarray:
        """Adaptive integration of TAS and NAS results."""
        # Start with weighted average
        integrated = self._weighted_average_integration(tas_data, nas_data, tas_weight, nas_weight)
        
        # Apply adaptive adjustments based on data characteristics
        if len(integrated) > 0:
            # Smooth transitions
            integrated = self._smooth_transitions(integrated)
            
            # Apply confidence weighting
            integrated = self._apply_confidence_weighting(integrated, tas_data, nas_data)
        
        return integrated
    
    def _smooth_transitions(self, data: np.ndarray) -> np.ndarray:
        """Smooth regime transitions."""
        if len(data) < 3:
            return data
        
        # Apply moving average for smoothing
        window_size = min(5, len(data) // 3)
        smoothed = np.convolve(data, np.ones(window_size) / window_size, mode='same')
        
        return smoothed
    
    def _apply_confidence_weighting(self, 
                                    integrated_data: np.ndarray,
                                    tas_data: np.ndarray,
                                    nas_data: np.ndarray) -> np.ndarray:
        """Apply confidence weighting to integrated data."""
        if len(tas_data) == 0 or len(nas_data) == 0:
            return integrated_data
        
        min_len = min(len(integrated_data), len(tas_data), len(nas_data))
        integrated_data = integrated_data[:min_len]
        tas_data = tas_data[:min_len]
        nas_data = nas_data[:min_len]
        
        # Calculate confidence based on agreement
        agreement = np.abs(tas_data - nas_data)
        confidence = 1.0 - (agreement / (np.max(agreement) + 1e-8))
        
        # Apply confidence weighting
        return integrated_data * confidence
    
    def _calculate_integration_confidence(self, 
                                           tas_results: Dict[str, Any],
                                           nas_results: Dict[str, Any]) -> float:
        """Calculate confidence in the integration."""
        tas_confidence = tas_results.get('confidence', 0.5)
        nas_confidence = nas_results.get('confidence', 0.5)
        
        # Integration confidence is the minimum of individual confidences
        return min(tas_confidence, nas_confidence)
    
    def _perform_hybrid_clustering(self, 
                                    market_data: Union[pd.DataFrame, np.ndarray],
                                    integrated_results: Dict[str, Any],
                                    timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform hybrid clustering on integrated results."""
        self.logger.info("🔍 Performing hybrid clustering")
        
        # Prepare data for clustering
        clustering_data = self._prepare_clustering_data(market_data, integrated_results)
        
        # Perform clustering
        clustering_results = self.clusterer.cluster(
            clustering_data,
            n_clusters=self.config.n_regimes,
            method=self.config.clustering_method.value
        )
        
        return clustering_results
    
    def _prepare_clustering_data(self, 
                                 market_data: Union[pd.DataFrame, np.ndarray],
                                 integrated_results: Dict[str, Any]) -> np.ndarray:
        """Prepare data for clustering."""
        # Combine market data with integrated regime information
        if isinstance(market_data, pd.DataFrame):
            market_features = market_data.values
        else:
            market_features = market_data
        
        # Add integrated regime features
        integrated_predictions = integrated_results.get('integrated_predictions', np.array([]))
        integrated_probabilities = integrated_results.get('integrated_probabilities', np.array([]))
        
        if len(integrated_predictions) > 0 and len(integrated_probabilities) > 0:
            # Ensure same length
            min_len = min(len(market_features), len(integrated_predictions), len(integrated_probabilities))
            market_features = market_features[:min_len]
            integrated_predictions = integrated_predictions[:min_len]
            integrated_probabilities = integrated_probabilities[:min_len]
            
            # Combine features
            clustering_data = np.column_stack([
                market_features,
                integrated_predictions.reshape(-1, 1),
                integrated_probabilities.reshape(-1, 1)
            ])
        else:
            clustering_data = market_features
        
        return clustering_data
    
    def _create_regime_model(self, 
                             market_data: Union[pd.DataFrame, np.ndarray],
                             clustering_results: Dict[str, Any],
                             integrated_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create regime model from clustering results."""
        self.logger.info("🏗️ Creating regime model")
        
        # Create regime model
        regime_model = self.modeler.create_model(
            market_data,
            clustering_results,
            integrated_results
        )
        
        return regime_model
    
    def _perform_economic_analysis(self, 
                                    market_data: Union[pd.DataFrame, np.ndarray],
                                    regime_model: Dict[str, Any],
                                    timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform economic regime analysis."""
        self.logger.info("🏛️ Performing economic analysis")
        
        # This would integrate with economic analysis components
        # For now, return placeholder
        return {
            'predictions': np.array([]),
            'significance_scores': np.array([]),
            'economic_regime_types': [],
            'economic_indicators': {}
        }
    
    def _perform_financial_analysis(self, 
                                    market_data: Union[pd.DataFrame, np.ndarray],
                                    regime_model: Dict[str, Any],
                                    timestamps: Optional[np.ndarray]) -> Dict[str, Any]:
        """Perform financial regime analysis."""
        self.logger.info("💰 Performing financial analysis")
        
        # This would integrate with financial analysis components
        # For now, return placeholder
        return {
            'predictions': np.array([]),
            'significance_scores': np.array([]),
            'financial_regime_types': [],
            'financial_indicators': {}
        }
    
    def _generate_final_predictions(self, 
                                    regime_model: Dict[str, Any],
                                    economic_results: Optional[Dict[str, Any]],
                                    financial_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final regime predictions."""
        self.logger.info("🎯 Generating final predictions")
        
        # Extract regime predictions from model
        regime_predictions = regime_model.get('regime_predictions', np.array([]))
        regime_probabilities = regime_model.get('regime_probabilities', np.array([]))
        
        # Generate regime labels
        regime_labels = self._generate_regime_labels(regime_predictions)
        
        # Calculate stability scores
        stability_scores = self._calculate_stability_scores(regime_predictions)
        
        # Calculate transition probabilities
        transition_probabilities = self._calculate_transition_probabilities(regime_predictions)
        
        # Calculate trading viability scores
        trading_viability_scores = self._calculate_trading_viability_scores(
            regime_predictions, economic_results, financial_results
        )
        
        # Calculate uncertainty estimates
        uncertainty_estimates = self._calculate_uncertainty_estimates(
            regime_probabilities, regime_model
        )
        
        return {
            'regime_predictions': regime_predictions,
            'regime_probabilities': regime_probabilities,
            'regime_labels': regime_labels,
            'stability_scores': stability_scores,
            'transition_probabilities': transition_probabilities,
            'trading_viability_scores': trading_viability_scores,
            'uncertainty_estimates': uncertainty_estimates
        }
    
    def _generate_regime_labels(self, regime_predictions: np.ndarray) -> List[str]:
        """Generate regime labels from predictions."""
        unique_regimes = np.unique(regime_predictions)
        regime_labels = []
        
        for regime_id in unique_regimes:
            if regime_id == 0:
                regime_labels.append("normal")
            elif regime_id == 1:
                regime_labels.append("bull_market")
            elif regime_id == 2:
                regime_labels.append("bear_market")
            elif regime_id == 3:
                regime_labels.append("high_volatility")
            elif regime_id == 4:
                regime_labels.append("low_volatility")
            elif regime_id == 5:
                regime_labels.append("trending_up")
            elif regime_id == 6:
                regime_labels.append("trending_down")
            elif regime_id == 7:
                regime_labels.append("mean_reverting")
            elif regime_id == 8:
                regime_labels.append("breakout")
            elif regime_id == 9:
                regime_labels.append("consolidation")
            elif regime_id == 10:
                regime_labels.append("crisis")
            else:
                regime_labels.append("unknown")
        
        return regime_labels
    
    def _calculate_stability_scores(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime stability scores."""
        if len(regime_predictions) < 2:
            return np.array([1.0] * len(regime_predictions))
        
        stability_scores = np.zeros(len(regime_predictions))
        
        for i in range(len(regime_predictions)):
            # Look at surrounding regimes for stability
            window_size = min(10, len(regime_predictions) // 4)
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(regime_predictions), i + window_size // 2 + 1)
            
            window_regimes = regime_predictions[start_idx:end_idx]
            current_regime = regime_predictions[i]
            
            # Stability is based on consistency within window
            consistency = np.sum(window_regimes == current_regime) / len(window_regimes)
            stability_scores[i] = consistency
        
        return stability_scores
    
    def _calculate_transition_probabilities(self, regime_predictions: np.ndarray) -> np.ndarray:
        """Calculate regime transition probabilities."""
        if len(regime_predictions) < 2:
            return np.array([0.0] * len(regime_predictions))
        
        transition_probabilities = np.zeros(len(regime_predictions))
        
        for i in range(1, len(regime_predictions)):
            current_regime = regime_predictions[i]
            previous_regime = regime_predictions[i-1]
            
            # Transition probability is 1 if regime changed, 0 if same
            transition_probabilities[i] = 1.0 if current_regime != previous_regime else 0.0
        
        return transition_probabilities
    
    def _calculate_trading_viability_scores(self, 
                                            regime_predictions: np.ndarray,
                                            economic_results: Optional[Dict[str, Any]],
                                            financial_results: Optional[Dict[str, Any]]) -> np.ndarray:
        """Calculate trading viability scores."""
        # Base viability on regime stability and economic/financial significance
        stability_scores = self._calculate_stability_scores(regime_predictions)
        
        viability_scores = stability_scores.copy()
        
        # Adjust based on economic significance
        if economic_results and 'significance_scores' in economic_results:
            economic_scores = economic_results['significance_scores']
            if len(economic_scores) == len(viability_scores):
                viability_scores = 0.6 * viability_scores + 0.4 * economic_scores
        
        # Adjust based on financial significance
        if financial_results and 'significance_scores' in financial_results:
            financial_scores = financial_results['significance_scores']
            if len(financial_scores) == len(viability_scores):
                viability_scores = 0.7 * viability_scores + 0.3 * financial_scores
        
        return viability_scores
    
    def _calculate_uncertainty_estimates(self, 
                                         regime_probabilities: np.ndarray,
                                         regime_model: Dict[str, Any]) -> np.ndarray:
        """Calculate uncertainty estimates."""
        if len(regime_probabilities) == 0:
            return np.array([])
        
        # Uncertainty is based on probability distribution entropy
        uncertainty_scores = np.zeros(len(regime_probabilities))
        
        for i, probs in enumerate(regime_probabilities):
            if isinstance(probs, (list, np.ndarray)) and len(probs) > 1:
                # Calculate entropy
                probs = np.array(probs)
                probs = probs / (np.sum(probs) + 1e-8)  # Normalize
                entropy = -np.sum(probs * np.log(probs + 1e-8))
                uncertainty_scores[i] = entropy
            else:
                # Single probability value
                prob = float(probs) if not isinstance(probs, (list, np.ndarray)) else probs[0]
                uncertainty_scores[i] = -prob * np.log(prob + 1e-8) - (1-prob) * np.log(1-prob + 1e-8)
        
        return uncertainty_scores
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get summary of current regime detection."""
        if not self.current_regimes is not None:
            return {"error": "No regime detection performed yet"}
        
        unique_regimes, counts = np.unique(self.current_regimes, return_counts=True)
        
        return {
            "total_samples": len(self.current_regimes),
            "unique_regimes": len(unique_regimes),
            "regime_distribution": dict(zip(unique_regimes, counts)),
            "most_common_regime": unique_regimes[np.argmax(counts)],
            "regime_stability": float(np.mean(self._calculate_stability_scores(self.current_regimes)))
        }
    
    def adapt_to_new_data(self, new_data: Union[pd.DataFrame, np.ndarray]) -> HybridRegimeResult:
        """Adapt regime detection to new data."""
        self.logger.info("🔄 Adapting to new data")
        
        # Perform regime detection on new data
        return self.detect_regimes(new_data)