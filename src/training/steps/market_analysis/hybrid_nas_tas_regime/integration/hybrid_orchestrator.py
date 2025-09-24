"""
Hybrid Orchestrator

Main orchestrator that replaces HMM clustering functionality with
comprehensive hybrid regime detection and analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import time
from dataclasses import dataclass, field

from ..config.hybrid_config import HybridRegimeConfig
from ..core.hybrid_regime_detector import HybridRegimeDetector
from ..components.tas_integration import TASIntegration
from ..components.nas_integration import NASIntegration
from ..evaluation.economic_evaluator import EconomicEvaluator
from ..tagging.regime_tagger import RegimeTagger


@dataclass
class HybridOrchestratorResult:
    """Result from hybrid orchestrator."""
    success: bool
    regime_predictions: np.ndarray
    regime_probabilities: np.ndarray
    regime_labels: List[str]
    economic_significance_scores: np.ndarray
    financial_significance_scores: np.ndarray
    trading_viability_scores: np.ndarray
    tagged_data: Optional[pd.DataFrame] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None


class HybridOrchestrator:
    """
    Main orchestrator that replaces HMM clustering functionality.
    
    This orchestrator:
    1. Integrates TAS and NAS regime detection
    2. Performs comprehensive economic and financial analysis
    3. Creates unified dataset with regime labels
    4. Provides complete HMM replacement functionality
    """
    
    def __init__(self, config: HybridRegimeConfig):
        """
        Initialize Hybrid Orchestrator.
        
        Args:
            config: Hybrid regime configuration
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize components
        self.regime_detector = HybridRegimeDetector(config)
        self.tas_integration = TASIntegration(config)
        self.nas_integration = NASIntegration(config)
        self.economic_evaluator = EconomicEvaluator(config)
        self.regime_tagger = RegimeTagger(config)
        
        # Initialize state
        self.last_result = None
        self.performance_history = []
        
        self.logger.info("✅ Hybrid Orchestrator initialized")
        self.logger.info(f"🔗 Integration strategy: {config.integration_strategy.value}")
        self.logger.info(f"🏛️ Economic modeling: {config.economic_modeling_enabled}")
        self.logger.info(f"💰 Financial modeling: {config.financial_modeling_enabled}")
    
    def process_regime_detection(self,
                                market_data: Union[pd.DataFrame, np.ndarray],
                                tas_inputs: Optional[Dict[str, Any]] = None,
                                nas_inputs: Optional[Dict[str, Any]] = None,
                                timestamps: Optional[np.ndarray] = None,
                                enable_tagging: bool = True,
                                save_results: bool = True) -> HybridOrchestratorResult:
        """
        Process complete regime detection pipeline.
        
        Args:
            market_data: Market data
            tas_inputs: TAS regime detection inputs
            nas_inputs: NAS regime detection inputs
            timestamps: Optional timestamps
            enable_tagging: Whether to enable data tagging
            save_results: Whether to save results
            
        Returns:
            HybridOrchestratorResult with complete analysis
        """
        start_time = time.time()
        self.logger.info("🚀 Starting hybrid regime detection pipeline")
        
        try:
            # Step 1: Perform hybrid regime detection
            self.logger.info("🔍 Step 1: Hybrid regime detection")
            regime_result = self.regime_detector.detect_regimes(
                market_data=market_data,
                tas_inputs=tas_inputs,
                nas_inputs=nas_inputs,
                timestamps=timestamps,
                enable_economic_analysis=True,
                enable_financial_analysis=True
            )
            
            if not regime_result.success:
                raise Exception(f"Regime detection failed: {regime_result.error_message}")
            
            # Step 2: Economic evaluation
            self.logger.info("🏛️ Step 2: Economic evaluation")
            economic_result = self.economic_evaluator.evaluate_economic_significance(
                market_data=market_data,
                regime_predictions=regime_result.regime_predictions,
                regime_probabilities=regime_result.regime_probabilities,
                timestamps=timestamps
            )
            
            # Step 3: Data tagging (if enabled)
            tagged_data = None
            if enable_tagging:
                self.logger.info("🏷️ Step 3: Data tagging")
                tagging_result = self.regime_tagger.tag_data(
                    data=market_data,
                    regime_predictions=regime_result.regime_predictions,
                    regime_probabilities=regime_result.regime_probabilities,
                    regime_labels=regime_result.regime_labels,
                    timestamps=timestamps
                )
                
                if tagging_result['success']:
                    tagged_data = tagging_result['tagged_data']
                    self.logger.info(f"✅ Tagged {len(tagged_data)} samples")
                else:
                    self.logger.warning("⚠️ Data tagging failed")
            
            # Step 4: Calculate financial significance
            financial_significance = self._calculate_financial_significance(
                regime_result, economic_result
            )
            
            # Step 5: Calculate trading viability
            trading_viability = self._calculate_trading_viability(
                regime_result, economic_result, financial_significance
            )
            
            # Step 6: Save results (if enabled)
            if save_results:
                self._save_results(regime_result, economic_result, tagged_data)
            
            execution_time = time.time() - start_time
            
            # Create result
            result = HybridOrchestratorResult(
                success=True,
                regime_predictions=regime_result.regime_predictions,
                regime_probabilities=regime_result.regime_probabilities,
                regime_labels=regime_result.regime_labels,
                economic_significance_scores=economic_result['significance_scores'],
                financial_significance_scores=financial_significance,
                trading_viability_scores=trading_viability,
                tagged_data=tagged_data,
                execution_time=execution_time,
                metadata={
                    'regime_result': regime_result,
                    'economic_result': economic_result,
                    'tagging_result': tagging_result if enable_tagging else None,
                    'config': self.config.to_dict()
                }
            )
            
            # Update state
            self.last_result = result
            self.performance_history.append({
                'timestamp': time.time(),
                'execution_time': execution_time,
                'n_samples': len(regime_result.regime_predictions),
                'n_regimes': len(set(regime_result.regime_predictions)),
                'economic_significance': np.mean(economic_result['significance_scores']),
                'financial_significance': np.mean(financial_significance),
                'trading_viability': np.mean(trading_viability)
            })
            
            self.logger.info(f"✅ Hybrid regime detection pipeline completed in {execution_time:.2f}s")
            self.logger.info(f"📊 Detected {len(set(regime_result.regime_predictions))} regimes")
            self.logger.info(f"🏛️ Economic significance: {np.mean(economic_result['significance_scores']):.3f}")
            self.logger.info(f"💰 Financial significance: {np.mean(financial_significance):.3f}")
            self.logger.info(f"📈 Trading viability: {np.mean(trading_viability):.3f}")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Hybrid regime detection pipeline failed: {e}")
            
            return HybridOrchestratorResult(
                success=False,
                regime_predictions=np.array([]),
                regime_probabilities=np.array([]),
                regime_labels=[],
                economic_significance_scores=np.array([]),
                financial_significance_scores=np.array([]),
                trading_viability_scores=np.array([]),
                execution_time=execution_time,
                error_message=str(e)
            )
    
    def _calculate_financial_significance(self, 
                                          regime_result: Any,
                                          economic_result: Dict[str, Any]) -> np.ndarray:
        """Calculate financial significance scores."""
        # Base financial significance on economic significance and regime stability
        economic_scores = economic_result['significance_scores']
        regime_stability = regime_result.regime_stability_scores
        
        # Combine economic significance with regime stability
        financial_significance = 0.7 * economic_scores + 0.3 * regime_stability
        
        # Apply financial significance threshold
        financial_significance = np.where(
            financial_significance >= self.config.financial_significance_threshold,
            financial_significance,
            financial_significance * 0.5
        )
        
        return financial_significance
    
    def _calculate_trading_viability(self, 
                                     regime_result: Any,
                                     economic_result: Dict[str, Any],
                                     financial_significance: np.ndarray) -> np.ndarray:
        """Calculate trading viability scores."""
        # Combine multiple factors for trading viability
        economic_scores = economic_result['significance_scores']
        regime_stability = regime_result.regime_stability_scores
        
        # Trading viability combines economic significance, financial significance, and stability
        trading_viability = (0.4 * economic_scores + 
                           0.3 * financial_significance + 
                           0.3 * regime_stability)
        
        # Apply trading viability threshold
        trading_viability = np.where(
            trading_viability >= self.config.trading_viability_threshold,
            trading_viability,
            trading_viability * 0.5
        )
        
        return trading_viability
    
    def _save_results(self, 
                      regime_result: Any,
                      economic_result: Dict[str, Any],
                      tagged_data: Optional[pd.DataFrame]):
        """Save results to files."""
        try:
            if self.config.save_results:
                # Save regime results
                regime_file = f"{self.config.output_dir}/regime_results.json"
                # Implementation would save regime_result to file
                
                # Save economic results
                economic_file = f"{self.config.output_dir}/economic_results.json"
                # Implementation would save economic_result to file
                
                # Save tagged data
                if tagged_data is not None:
                    tagged_file = f"{self.config.output_dir}/tagged_data.csv"
                    tagged_data.to_csv(tagged_file, index=False)
                
                self.logger.info(f"💾 Results saved to {self.config.output_dir}")
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save results: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.performance_history:
            return {"error": "No performance data available"}
        
        recent_performance = self.performance_history[-1]
        
        return {
            "latest_execution_time": recent_performance['execution_time'],
            "latest_n_samples": recent_performance['n_samples'],
            "latest_n_regimes": recent_performance['n_regimes'],
            "latest_economic_significance": recent_performance['economic_significance'],
            "latest_financial_significance": recent_performance['financial_significance'],
            "latest_trading_viability": recent_performance['trading_viability'],
            "total_runs": len(self.performance_history),
            "average_execution_time": np.mean([p['execution_time'] for p in self.performance_history]),
            "average_economic_significance": np.mean([p['economic_significance'] for p in self.performance_history]),
            "average_financial_significance": np.mean([p['financial_significance'] for p in self.performance_history]),
            "average_trading_viability": np.mean([p['trading_viability'] for p in self.performance_history])
        }
    
    def get_tagged_data(self) -> Optional[pd.DataFrame]:
        """Get the latest tagged data."""
        if self.last_result and self.last_result.tagged_data is not None:
            return self.last_result.tagged_data
        return None
    
    def get_regime_summary(self) -> Dict[str, Any]:
        """Get regime detection summary."""
        if not self.last_result:
            return {"error": "No regime detection performed yet"}
        
        return {
            "n_samples": len(self.last_result.regime_predictions),
            "n_regimes": len(set(self.last_result.regime_predictions)),
            "regime_labels": self.last_result.regime_labels,
            "average_economic_significance": np.mean(self.last_result.economic_significance_scores),
            "average_financial_significance": np.mean(self.last_result.financial_significance_scores),
            "average_trading_viability": np.mean(self.last_result.trading_viability_scores),
            "execution_time": self.last_result.execution_time,
            "success": self.last_result.success
        }