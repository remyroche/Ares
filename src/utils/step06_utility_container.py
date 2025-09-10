"""
Step06 Utility Container - Moved to Utilities

This module contains the original step06 utility container functionality
now available as utilities. All functionality has been preserved from the original step06.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple
import logging
import warnings
from pathlib import Path

# Import step06 utilities
from .step06_enhanced_feature_engineering import EnhancedFeatureEngineering
from .step06_labeling_components import (
    OptimizedTripleBarrierLabeling,
    FractionalTripleBarrierLabeling,
    RegimeSpecificTripleBarrierOptimizer,
    ProfitBasedFeatureEngineering,
    RegimeAwareTripleBarrierLabeling
)

logger = logging.getLogger(__name__)

class Step06UtilityContainer:
    """
    Step06 Utility Container for dependency injection and utility management.
    This is the original step06 functionality now available as utilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize step06 utility container.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = logger
        
        # Initialize utilities
        self._initialize_utilities()
        
        self.logger.info("📦 Step06 Utility Container (Step06 Utilities) initialized")
        self.logger.info(f"   Available utilities: {list(self.utilities.keys())}")

    def _initialize_utilities(self) -> None:
        """Initialize all step06 utilities."""
        self.utilities = {}
        
        try:
            # Initialize enhanced feature engineering
            self.utilities['feature_engineering'] = EnhancedFeatureEngineering(self.config)
            
            # Initialize labeling utilities
            self.utilities['triple_barrier_labeling'] = OptimizedTripleBarrierLabeling(
                profit_take_multiplier=self.config.get('profit_take_multiplier', 0.004),
                stop_loss_multiplier=self.config.get('stop_loss_multiplier', 0.003),
                transaction_cost=self.config.get('transaction_cost', 0.0008),
                time_barrier_minutes=self.config.get('time_barrier_minutes', 30),
                max_lookahead=self.config.get('max_lookahead', 100)
            )
            
            self.utilities['fractional_triple_barrier_labeling'] = FractionalTripleBarrierLabeling(
                d=self.config.get('fractional_d', 0.5),
                threshold=self.config.get('stationarity_threshold', 0.01),
                profit_take_multiplier=self.config.get('profit_take_multiplier', 0.004),
                stop_loss_multiplier=self.config.get('stop_loss_multiplier', 0.003),
                transaction_cost=self.config.get('transaction_cost', 0.0008)
            )
            
            self.utilities['regime_specific_optimizer'] = RegimeSpecificTripleBarrierOptimizer(
                regime_threshold=self.config.get('regime_threshold', 0.7),
                base_profit_take=self.config.get('profit_take_multiplier', 0.004),
                base_stop_loss=self.config.get('stop_loss_multiplier', 0.003),
                base_transaction_cost=self.config.get('transaction_cost', 0.0008)
            )
            
            self.utilities['profit_based_feature_engineering'] = ProfitBasedFeatureEngineering(
                profit_threshold=self.config.get('profit_threshold', 0.002),
                risk_reward_ratio=self.config.get('risk_reward_ratio', 2.0),
                min_profit_margin=self.config.get('min_profit_margin', 0.001),
                max_profit_margin=self.config.get('max_profit_margin', 0.01)
            )
            
            self.utilities['regime_aware_labeling'] = RegimeAwareTripleBarrierLabeling(
                regime_threshold=self.config.get('regime_threshold', 0.7),
                base_profit_take=self.config.get('profit_take_multiplier', 0.004),
                base_stop_loss=self.config.get('stop_loss_multiplier', 0.003),
                base_transaction_cost=self.config.get('transaction_cost', 0.0008)
            )
            
            self.logger.info("✅ All step06 utilities initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize step06 utilities: {e}")
            raise

    def get_utility(self, utility_name: str) -> Any:
        """
        Get a specific utility by name.
        
        Args:
            utility_name: Name of the utility to retrieve
            
        Returns:
            The requested utility instance
        """
        if utility_name not in self.utilities:
            raise ValueError(f"Utility '{utility_name}' not found. Available utilities: {list(self.utilities.keys())}")
        
        return self.utilities[utility_name]

    def get_feature_engineering_utility(self) -> EnhancedFeatureEngineering:
        """Get the enhanced feature engineering utility."""
        return self.get_utility('feature_engineering')

    def get_triple_barrier_labeling_utility(self) -> OptimizedTripleBarrierLabeling:
        """Get the optimized triple barrier labeling utility."""
        return self.get_utility('triple_barrier_labeling')

    def get_fractional_triple_barrier_labeling_utility(self) -> FractionalTripleBarrierLabeling:
        """Get the fractional triple barrier labeling utility."""
        return self.get_utility('fractional_triple_barrier_labeling')

    def get_regime_specific_optimizer_utility(self) -> RegimeSpecificTripleBarrierOptimizer:
        """Get the regime-specific triple barrier optimizer utility."""
        return self.get_utility('regime_specific_optimizer')

    def get_profit_based_feature_engineering_utility(self) -> ProfitBasedFeatureEngineering:
        """Get the profit-based feature engineering utility."""
        return self.get_utility('profit_based_feature_engineering')

    def get_regime_aware_labeling_utility(self) -> RegimeAwareTripleBarrierLabeling:
        """Get the regime-aware triple barrier labeling utility."""
        return self.get_utility('regime_aware_labeling')

    def process_market_data(self, market_data: pd.DataFrame,
                          regime_labels: Optional[pd.Series] = None,
                          regime_confidence: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Process market data using all available utilities.
        
        Args:
            market_data: OHLCV market data
            regime_labels: Market regime labels (optional)
            regime_confidence: Regime confidence scores (optional)
            
        Returns:
            Dictionary with processed data from all utilities
        """
        self.logger.info("🔄 Processing market data with step06 utilities...")
        
        try:
            results = {}
            
            # 1. Enhanced feature engineering
            feature_engineering_util = self.get_feature_engineering_utility()
            
            # Extract technical indicators
            periods_config = self.config.get('periods_config', {
                'RSI': [14, 21, 28],
                'MACD': [12, 26],
                'Bollinger_Bands': [20, 30],
                'SMA': [10, 20, 50],
                'EMA': [12, 26, 50],
                'ATR': [14, 21],
                'Stochastic': [14, 21],
                'ADX': [14, 21],
                'OBV': [1],
                'MFI': [14, 21]
            })
            
            technical_indicators = feature_engineering_util.extract_indicators_batch(
                market_data, periods_config
            )
            
            # Create sophisticated interactions
            interaction_features = feature_engineering_util.create_sophisticated_interactions(
                technical_indicators
            )
            
            results['technical_indicators'] = technical_indicators
            results['interaction_features'] = interaction_features
            
            # 2. Triple barrier labeling
            triple_barrier_util = self.get_triple_barrier_labeling_utility()
            triple_barrier_labels = triple_barrier_util.apply_triple_barrier_labeling_vectorized(
                market_data
            )
            results['triple_barrier_labels'] = triple_barrier_labels
            
            # 3. Fractional triple barrier labeling
            fractional_util = self.get_fractional_triple_barrier_labeling_utility()
            fractional_labels = fractional_util.create_fractional_labels(market_data)
            results['fractional_labels'] = fractional_labels
            
            # 4. Profit-based feature engineering
            profit_util = self.get_profit_based_feature_engineering_utility()
            returns = market_data['close'].pct_change()
            labels = triple_barrier_labels['label']
            profit_features = profit_util.create_profit_based_features(
                market_data, returns, labels
            )
            results['profit_features'] = profit_features
            
            # 5. Regime-aware processing (if regime data available)
            if regime_labels is not None and regime_confidence is not None:
                regime_optimizer_util = self.get_regime_specific_optimizer_utility()
                regime_aware_util = self.get_regime_aware_labeling_utility()
                
                # Optimize regime parameters
                optimized_params = regime_optimizer_util.optimize_regime_thresholds(
                    market_data, regime_labels, regime_confidence
                )
                
                # Create regime-aware labels
                regime_aware_labels = regime_aware_util.create_regime_aware_labels(
                    market_data, regime_labels, regime_confidence, optimized_params
                )
                
                results['optimized_regime_params'] = optimized_params
                results['regime_aware_labels'] = regime_aware_labels
            
            self.logger.info("✅ Market data processing completed with step06 utilities")
            self.logger.info(f"   Results keys: {list(results.keys())}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Market data processing failed: {e}")
            raise

    def get_utility_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics from all utilities.
        
        Returns:
            Dictionary with statistics from all utilities
        """
        self.logger.info("📊 Collecting statistics from all step06 utilities...")
        
        try:
            stats = {}
            
            # Get statistics from each utility
            for utility_name, utility in self.utilities.items():
                try:
                    if hasattr(utility, 'get_processing_stats'):
                        stats[utility_name] = utility.get_processing_stats()
                    elif hasattr(utility, 'get_statistics'):
                        stats[utility_name] = utility.get_statistics()
                    else:
                        stats[utility_name] = {'status': 'no_statistics_available'}
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to get statistics from {utility_name}: {e}")
                    stats[utility_name] = {'error': str(e)}
            
            self.logger.info("✅ Statistics collection completed")
            return stats
            
        except Exception as e:
            self.logger.error(f"❌ Statistics collection failed: {e}")
            raise

    def reset_all_utilities(self) -> None:
        """Reset all utilities to their initial state."""
        self.logger.info("🔄 Resetting all step06 utilities...")
        
        try:
            for utility_name, utility in self.utilities.items():
                if hasattr(utility, 'reset_stats'):
                    utility.reset_stats()
                    self.logger.info(f"   Reset {utility_name}")
            
            self.logger.info("✅ All utilities reset successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Utility reset failed: {e}")
            raise

    def get_available_utilities(self) -> List[str]:
        """
        Get list of available utilities.
        
        Returns:
            List of available utility names
        """
        return list(self.utilities.keys())

    def validate_utilities(self) -> Dict[str, bool]:
        """
        Validate all utilities are working correctly.
        
        Returns:
            Dictionary with validation status for each utility
        """
        self.logger.info("🔍 Validating all step06 utilities...")
        
        try:
            validation_results = {}
            
            for utility_name, utility in self.utilities.items():
                try:
                    # Basic validation - check if utility has required methods
                    if hasattr(utility, '__init__'):
                        validation_results[utility_name] = True
                    else:
                        validation_results[utility_name] = False
                except Exception as e:
                    self.logger.warning(f"⚠️ Validation failed for {utility_name}: {e}")
                    validation_results[utility_name] = False
            
            self.logger.info("✅ Utility validation completed")
            return validation_results
            
        except Exception as e:
            self.logger.error(f"❌ Utility validation failed: {e}")
            raise