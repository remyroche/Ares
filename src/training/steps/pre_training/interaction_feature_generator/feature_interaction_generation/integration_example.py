"""
Integration Example: Data-Driven Lookback Optimization in Ares Pipeline

This example shows how to integrate the lookback optimization system into the
existing Ares trading pipeline, replacing hardcoded lookback ceilings with
data-driven inference.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Import the optimization system
from .orchestrator import LookbackOptimizationOrchestrator
from .config import create_production_config, FamilyType

# Import existing pipeline components (these would be actual imports in practice)
# from src.training.steps.pre_training.feature_lookback_optimization import FeatureLookbackOptimizer
# from src.training.steps.pre_training.pid_based_feature_generation.interaction_feature_generator import InteractionFeatureGenerator

logger = logging.getLogger(__name__)


class OptimizedFeaturePipeline:
    """
    Enhanced feature pipeline that uses data-driven lookback optimization
    instead of hardcoded ceilings.
    """
    
    def __init__(self, config=None):
        """Initialize the optimized feature pipeline."""
        self.config = config or create_production_config()
        self.orchestrator = LookbackOptimizationOrchestrator(self.config)
        
        # Cache for optimization results
        self.optimization_cache = {}
        self.last_optimization_time = None
        
        # Feature generation components
        self.feature_generator = None
        self.interaction_generator = None
        
        logger.info("Initialized OptimizedFeaturePipeline with data-driven lookbacks")
    
    def optimize_lookbacks(self, data, targets, feature_names=None):
        """
        Run lookback optimization for all symbols and families.
        
        This replaces the hardcoded lookback ceilings with data-driven inference.
        """
        logger.info("Starting data-driven lookback optimization...")
        
        # Run the three-stage optimization
        result = self.orchestrator.optimize_lookbacks(data, targets, feature_names)
        
        if result.success:
            # Cache the results
            self.optimization_cache = result
            self.last_optimization_time = datetime.now()
            
            logger.info(f"Lookback optimization completed successfully in {result.execution_time:.3f}s")
            
            # Generate and log summary
            self._log_optimization_summary(result)
            
            return result
        else:
            logger.error(f"Lookback optimization failed: {result.error_message}")
            return None
    
    def generate_optimized_features(self, data, symbol, use_cache=True):
        """
        Generate features using optimized lookbacks.
        
        This replaces the hardcoded lookback approach with data-driven choices.
        """
        if use_cache and self.optimization_cache and symbol in self.optimization_cache.decisions:
            # Use cached optimization results
            symbol_decisions = self.optimization_cache.decisions[symbol]
            symbol_data = data[symbol] if isinstance(data, dict) else data
            
            logger.info(f"Generating optimized features for {symbol} using cached decisions")
            
            # Generate features using optimized lookbacks
            from .feature_families import MultiFamilyFeatureGenerator
            
            feature_generator = MultiFamilyFeatureGenerator(self.config)
            feature_names = {family: f"{family.value}_feature" for family in FamilyType}
            
            feature_results = feature_generator.generate_features(
                symbol_data, symbol_decisions, feature_names
            )
            
            # Create feature matrix
            feature_matrix, feature_names = feature_generator.create_feature_matrix(feature_results)
            
            return feature_matrix, feature_names, feature_results
            
        else:
            logger.warning(f"No cached optimization results for {symbol}, using default lookbacks")
            return self._generate_default_features(data, symbol)
    
    def generate_interaction_features(self, feature_matrix, feature_names, target, symbol):
        """
        Generate interaction features using optimized parent features.
        
        This integrates with the existing interaction feature generator.
        """
        logger.info(f"Generating interaction features for {symbol}")
        
        # Get optimized lookback periods for this symbol
        optimized_lookbacks = {}
        if (self.optimization_cache and 
            symbol in self.optimization_cache.decisions):
            
            symbol_decisions = self.optimization_cache.decisions[symbol]
            for family, decision in symbol_decisions.items():
                if decision.lookback_spec.effective_lookback is not None:
                    optimized_lookbacks[f"{family.value}_lookback"] = decision.lookback_spec.effective_lookback
        
        # Use the existing interaction feature generator with optimized lookbacks
        # In practice, this would be:
        # from src.training.steps.pre_training.pid_based_feature_generation.interaction_feature_generator import InteractionFeatureGenerator
        # interaction_generator = InteractionFeatureGenerator()
        # interaction_result = await interaction_generator.generate_interaction_features(
        #     feature_matrix, feature_names, optimized_lookbacks, target
        # )
        
        # For this example, we'll simulate the interaction generation
        interaction_result = self._simulate_interaction_generation(
            feature_matrix, feature_names, optimized_lookbacks, target
        )
        
        return interaction_result
    
    def should_retrain_lookbacks(self, force=False):
        """
        Determine if lookback optimization should be retrained.
        
        This implements the retrain cadence (daily at 02:00 ET).
        """
        if force:
            return True
        
        if self.last_optimization_time is None:
            return True
        
        # Check if it's been more than 24 hours
        time_since_optimization = datetime.now() - self.last_optimization_time
        return time_since_optimization.total_seconds() > 24 * 3600
    
    def _log_optimization_summary(self, result):
        """Log a summary of optimization results."""
        logger.info("=" * 60)
        logger.info("LOOKBACK OPTIMIZATION SUMMARY")
        logger.info("=" * 60)
        
        # Count decision types
        decision_counts = {'discrete': 0, 'blend': 0, 'default': 0, 'inactive': 0}
        for symbol_decisions in result.decisions.values():
            for decision in symbol_decisions.values():
                decision_type = decision.lookback_spec.decision_type.value
                decision_counts[decision_type] += 1
        
        logger.info(f"Execution time: {result.execution_time:.3f}s")
        logger.info(f"Symbols processed: {len(result.ic_surface_results)}")
        logger.info(f"Decision types: {decision_counts}")
        
        # Log family performance
        logger.info("\nFamily performance:")
        for family in FamilyType:
            family_ics = []
            for symbol_results in result.ic_surface_results.values():
                if family in symbol_results:
                    family_ics.append(symbol_results[family].optimal_ic)
            
            if family_ics:
                avg_ic = np.mean(family_ics)
                logger.info(f"  {family.value}: {avg_ic:.4f} (avg IC)")
        
        # Log feature quality
        if result.feature_results:
            all_quality_scores = []
            for symbol_results in result.feature_results.values():
                for feature_result in symbol_results.values():
                    all_quality_scores.append(feature_result.quality_score)
            
            if all_quality_scores:
                avg_quality = np.mean(all_quality_scores)
                logger.info(f"\nAverage feature quality: {avg_quality:.3f}")
        
        logger.info("=" * 60)
    
    def _generate_default_features(self, data, symbol):
        """Generate features using default lookbacks as fallback."""
        logger.warning(f"Using default lookbacks for {symbol}")
        
        # Default lookbacks (fallback)
        default_lookbacks = {
            FamilyType.MOMENTUM: 12,
            FamilyType.VOLATILITY: 12,
            FamilyType.GK: 12,
            FamilyType.VWAP_ROLL: 12,
            FamilyType.RSI: 14,
            FamilyType.AUTOCORR: 12
        }
        
        # This would use the existing feature generation logic
        # with hardcoded lookbacks as fallback
        symbol_data = data[symbol] if isinstance(data, dict) else data
        
        # Simulate feature generation
        feature_matrix = np.random.randn(len(symbol_data), len(default_lookbacks))
        feature_names = [f"{family.value}_feature" for family in default_lookbacks.keys()]
        
        return feature_matrix, feature_names, {}
    
    def _simulate_interaction_generation(self, feature_matrix, feature_names, optimized_lookbacks, target):
        """Simulate interaction feature generation."""
        logger.info(f"Simulating interaction generation with {len(optimized_lookbacks)} optimized lookbacks")
        
        # Simulate some interaction features
        n_interactions = min(15, len(feature_names) * (len(feature_names) - 1) // 2)
        interaction_features = np.random.randn(feature_matrix.shape[0], n_interactions)
        
        interaction_names = [f"interaction_{i}" for i in range(n_interactions)]
        
        return {
            'interaction_features': interaction_features,
            'interaction_names': interaction_names,
            'optimized_lookbacks_used': optimized_lookbacks
        }


def demonstrate_pipeline_integration():
    """Demonstrate how the optimized pipeline integrates with existing systems."""
    logger.info("Demonstrating optimized feature pipeline integration...")
    
    # Initialize the optimized pipeline
    pipeline = OptimizedFeaturePipeline()
    
    # Generate sample data (in practice, this would come from your data sources)
    data, targets = generate_sample_market_data()
    
    # Step 1: Optimize lookbacks (runs at 02:00 ET daily)
    logger.info("Step 1: Running lookback optimization...")
    optimization_result = pipeline.optimize_lookbacks(data, targets)
    
    if optimization_result is None:
        logger.error("Lookback optimization failed, falling back to default approach")
        return
    
    # Step 2: Generate optimized features for each symbol
    logger.info("Step 2: Generating optimized features...")
    
    for symbol in data.keys():
        logger.info(f"Processing {symbol}...")
        
        # Generate features using optimized lookbacks
        feature_matrix, feature_names, feature_results = pipeline.generate_optimized_features(
            data, symbol, use_cache=True
        )
        
        logger.info(f"Generated {feature_matrix.shape[1]} features for {symbol}")
        
        # Generate interaction features
        target = targets[symbol]
        interaction_result = pipeline.generate_interaction_features(
            feature_matrix, feature_names, target, symbol
        )
        
        logger.info(f"Generated {len(interaction_result['interaction_names'])} interaction features for {symbol}")
        
        # In practice, you would now use these features for model training
        # model.fit(feature_matrix, target)
    
    # Step 3: Check if retraining is needed
    if pipeline.should_retrain_lookbacks():
        logger.info("Lookback optimization retraining needed")
    else:
        logger.info("Using cached lookback optimization results")
    
    logger.info("Pipeline integration demonstration completed!")


def generate_sample_market_data():
    """Generate sample market data for demonstration."""
    logger.info("Generating sample market data...")
    
    data = {}
    targets = {}
    
    for i in range(3):
        symbol = f"SYMBOL_{i+1}"
        
        # Generate price data
        np.random.seed(42 + i)
        n_days = 2000
        
        returns = np.random.normal(0.0001, 0.02, n_days)
        prices = 100 * np.exp(np.cumsum(returns))
        
        # Generate OHLCV data
        high_low_noise = np.random.uniform(0.001, 0.005, n_days)
        df = pd.DataFrame({
            'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_days)),
            'high': prices * (1 + high_low_noise),
            'low': prices * (1 - high_low_noise),
            'close': prices,
            'volume': np.random.uniform(1000000, 5000000, n_days)
        })
        
        data[symbol] = df
        
        # Generate target (future returns)
        future_returns = df['close'].pct_change(5).shift(-5)
        targets[symbol] = future_returns.fillna(0).values
    
    logger.info(f"Generated data for {len(data)} symbols")
    return data, targets


def compare_with_hardcoded_approach():
    """Compare optimized approach with hardcoded lookbacks."""
    logger.info("Comparing optimized vs hardcoded approach...")
    
    # Generate sample data
    data, targets = generate_sample_market_data()
    
    # Initialize both approaches
    optimized_pipeline = OptimizedFeaturePipeline()
    
    # Run optimized approach
    logger.info("Running optimized approach...")
    start_time = datetime.now()
    
    optimization_result = optimized_pipeline.optimize_lookbacks(data, targets)
    
    if optimization_result:
        optimized_time = (datetime.now() - start_time).total_seconds()
        
        # Generate features using optimized lookbacks
        optimized_features = {}
        for symbol in data.keys():
            feature_matrix, feature_names, _ = optimized_pipeline.generate_optimized_features(
                data, symbol, use_cache=True
            )
            optimized_features[symbol] = feature_matrix
        
        logger.info(f"Optimized approach completed in {optimized_time:.3f}s")
        logger.info(f"Generated features for {len(optimized_features)} symbols")
        
        # In practice, you would compare model performance here
        # optimized_performance = evaluate_model_performance(optimized_features, targets)
        # hardcoded_performance = evaluate_model_performance(hardcoded_features, targets)
        
        logger.info("Comparison completed!")
    else:
        logger.error("Optimized approach failed, cannot compare")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("Data-Driven Lookback Optimization - Pipeline Integration Example")
    print("=" * 70)
    
    # Demonstrate pipeline integration
    demonstrate_pipeline_integration()
    
    print("\n" + "=" * 70)
    
    # Compare with hardcoded approach
    compare_with_hardcoded_approach()
    
    print("\n" + "=" * 70)
    print("Integration example completed!")
    print("=" * 70)