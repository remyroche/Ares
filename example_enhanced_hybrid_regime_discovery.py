"""
Example script demonstrating the enhanced hybrid NAS-TAS regime discovery system.

This script shows how to use the new multi-objective optimization approach
with CV minimization and cluster distribution targets.
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

# Import the enhanced hybrid orchestrator
from src.training.steps.market_analysis.hybrid_nas_tas_regime.hybrid_orchestrator import (
    HybridOrchestrator, HybridOrchestratorConfig
)

# Import enhanced components
from src.training.steps.market_analysis.hybrid_nas_tas_regime.regime_alignment_manager import (
    RegimeAlignmentManager, AlignmentConfig
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.enhanced_economic_evaluator import (
    EnhancedEconomicEvaluator, EconomicEvaluationConfig
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.consensus_validator import (
    ConsensusValidator, ConsensusValidationConfig
)
from src.training.steps.market_analysis.hybrid_nas_tas_regime.multi_objective_optimizer import (
    MultiObjectiveOptimizer, MultiObjectiveConfig
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_market_data(symbol: str = "BTCUSDT", days: int = 30) -> pd.DataFrame:
    """Create sample market data for testing."""
    try:
        # Generate timestamps
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='15min')
        
        # Generate realistic OHLCV data
        np.random.seed(42)  # For reproducibility
        n_points = len(timestamps)
        
        # Generate price data with regime-like behavior
        base_price = 50000  # Starting price
        prices = [base_price]
        
        # Create different regimes
        regime_length = n_points // 4
        regimes = []
        for i in range(4):
            regimes.extend([i] * regime_length)
        regimes.extend([3] * (n_points - len(regimes)))  # Fill remaining
        
        for i in range(1, n_points):
            regime = regimes[i]
            
            # Different volatility and drift for each regime
            if regime == 0:  # Low volatility, slight uptrend
                drift = 0.0001
                volatility = 0.01
            elif regime == 1:  # High volatility, downtrend
                drift = -0.0002
                volatility = 0.03
            elif regime == 2:  # Medium volatility, sideways
                drift = 0.00005
                volatility = 0.02
            else:  # High volatility, strong uptrend
                drift = 0.0003
                volatility = 0.025
            
            # Generate price change
            price_change = np.random.normal(drift, volatility)
            new_price = prices[-1] * (1 + price_change)
            prices.append(new_price)
        
        # Generate OHLCV data
        data = []
        for i, (timestamp, price) in enumerate(zip(timestamps, prices)):
            # Generate realistic OHLCV
            volatility_factor = 0.005
            high = price * (1 + np.random.uniform(0, volatility_factor))
            low = price * (1 - np.random.uniform(0, volatility_factor))
            open_price = prices[i-1] if i > 0 else price
            close_price = price
            volume = np.random.uniform(1000, 10000)
            
            data.append({
                'timestamp': timestamp,
                'open': open_price,
                'high': high,
                'low': low,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"✅ Generated sample market data: {len(df)} rows for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample market data: {e}")
        return pd.DataFrame()


def create_sample_features(market_data: pd.DataFrame) -> pd.DataFrame:
    """Create sample features for regime discovery."""
    try:
        features = pd.DataFrame(index=market_data.index)
        
        # Price-based features
        features['returns'] = market_data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['momentum'] = features['returns'].rolling(5).sum()
        
        # Volume features
        features['volume_ma'] = market_data['volume'].rolling(20).mean()
        features['volume_ratio'] = market_data['volume'] / features['volume_ma']
        
        # Technical indicators
        features['rsi'] = calculate_rsi(market_data['close'])
        features['bb_position'] = calculate_bollinger_position(market_data['close'])
        
        # Entropy calculation
        features['entropy'] = calculate_entropy(features['returns'])
        
        # Fill NaN values
        features = features.fillna(method='bfill').fillna(0)
        
        logger.info(f"✅ Generated sample features: {features.shape}")
        return features
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample features: {e}")
        return pd.DataFrame()


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    except Exception:
        return pd.Series([50] * len(prices), index=prices.index)


def calculate_bollinger_position(prices: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Bollinger Bands position."""
    try:
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)
        position = (prices - lower_band) / (upper_band - lower_band)
        return position
    except Exception:
        return pd.Series([0.5] * len(prices), index=prices.index)


def calculate_entropy(returns: pd.Series, window: int = 20) -> pd.Series:
    """Calculate entropy from returns."""
    try:
        entropy_values = []
        for i in range(len(returns)):
            if i < window:
                entropy_values.append(0)
            else:
                window_returns = returns.iloc[i-window:i]
                # Discretize returns
                bins = pd.cut(window_returns, bins=10, labels=False, include_lowest=True)
                bin_counts = bins.value_counts()
                probabilities = bin_counts / len(bins)
                probabilities = probabilities[probabilities > 0]
                entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
                entropy_values.append(entropy)
        
        return pd.Series(entropy_values, index=returns.index)
    except Exception:
        return pd.Series([0] * len(returns), index=returns.index)


async def run_enhanced_hybrid_regime_discovery():
    """Run the enhanced hybrid regime discovery example."""
    try:
        logger.info("🚀 Starting Enhanced Hybrid NAS-TAS Regime Discovery Example")
        
        # Step 1: Create sample data
        logger.info("📊 Step 1: Creating sample market data")
        market_data = create_sample_market_data("BTCUSDT", days=30)
        features = create_sample_features(market_data)
        
        if market_data.empty:
            logger.error("❌ No market data available")
            return
        
        # Step 2: Configure enhanced hybrid orchestrator
        logger.info("⚙️ Step 2: Configuring enhanced hybrid orchestrator")
        config = HybridOrchestratorConfig(
            symbol="BTCUSDT",
            timeframe="15m",
            start_date="2024-01-01",
            end_date="2024-12-31",
            use_standardized_features=True,
            feature_categories=['momentum', 'volatility', 'volume', 'trend'],
            significance_threshold=0.5,
            min_regime_duration=10,
            viability_threshold=0.5,
            minimum_regime_duration=5,
            max_iterations=100,
            use_bayesian_optimization=True,
            population_size=50,  # Reduced for example
            max_generations=25,  # Reduced for example
            use_nsga2=True,
            use_spea2=True,
            use_gpu_acceleration=False,  # Disabled for example
            memory_limit_gb=4.0,
            include_detailed_metrics=True,
            save_to_file=False
        )
        
        # Step 3: Initialize enhanced hybrid orchestrator
        logger.info("🔧 Step 3: Initializing enhanced hybrid orchestrator")
        orchestrator = HybridOrchestrator(config)
        
        # Step 4: Run enhanced hybrid analysis
        logger.info("🧠 Step 4: Running enhanced hybrid regime discovery")
        results = orchestrator.orchestrate_tas_nas_detection(
            market_data, 
            timeframes=['15m']
        )
        
        # Step 5: Analyze results
        logger.info("📈 Step 5: Analyzing enhanced results")
        analyze_enhanced_results(results)
        
        # Step 6: Demonstrate individual components
        logger.info("🔬 Step 6: Demonstrating individual enhanced components")
        await demonstrate_individual_components(market_data, features)
        
        logger.info("✅ Enhanced hybrid regime discovery example completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Enhanced hybrid regime discovery example failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def analyze_enhanced_results(results: Dict[str, Any]):
    """Analyze and display enhanced results."""
    try:
        logger.info("📊 Analyzing Enhanced Hybrid Results")
        
        # Check if hybrid analysis was successful
        hybrid_analysis = results.get('hybrid_analysis', {})
        if not hybrid_analysis.get('success', False):
            logger.error(f"❌ Hybrid analysis failed: {hybrid_analysis.get('error', 'Unknown error')}")
            return
        
        # Display basic results
        hybrid_labels = hybrid_analysis.get('hybrid_labels', [])
        if len(hybrid_labels) > 0:
            unique_regimes = len(np.unique(hybrid_labels))
            logger.info(f"🎯 Found {unique_regimes} unique regimes")
            logger.info(f"📊 Total predictions: {len(hybrid_labels)}")
            
            # Display regime distribution
            regime_counts = {}
            for regime in hybrid_labels:
                regime_counts[regime] = regime_counts.get(regime, 0) + 1
            
            logger.info("📈 Regime Distribution:")
            for regime, count in regime_counts.items():
                percentage = (count / len(hybrid_labels)) * 100
                logger.info(f"   Regime {regime}: {count} samples ({percentage:.1f}%)")
        
        # Display alignment results
        alignment_result = hybrid_analysis.get('alignment_result', {})
        if alignment_result:
            quality_metrics = alignment_result.get('quality_metrics', {})
            logger.info(f"🔄 Alignment Quality: {quality_metrics.get('alignment_quality', 0.0):.3f}")
            logger.info(f"📊 Alignment Coverage: {quality_metrics.get('coverage', 0.0):.3f}")
        
        # Display optimization results
        optimization_result = hybrid_analysis.get('optimization_result', {})
        if optimization_result:
            best_solution = optimization_result.get('best_solution', {})
            objectives = best_solution.get('objectives', {})
            logger.info("🎯 Multi-Objective Optimization Results:")
            logger.info(f"   Statistical Score: {objectives.get('silhouette_score', 0.0):.3f}")
            logger.info(f"   Economic Score: {objectives.get('economic_significance', 0.0):.3f}")
            logger.info(f"   CV Optimization: {objectives.get('cv_optimization', 0.0):.3f}")
            logger.info(f"   Distribution Quality: {objectives.get('distribution_quality', 0.0):.3f}")
        
        # Display economic evaluation
        economic_evaluation = hybrid_analysis.get('economic_evaluation', {})
        if economic_evaluation:
            overall_quality = economic_evaluation.get('overall_quality', 0.0)
            logger.info(f"💰 Economic Evaluation Quality: {overall_quality:.3f}")
            
            cv_optimization = economic_evaluation.get('cv_optimization', {})
            if cv_optimization:
                avg_weighted_cv = cv_optimization.get('avg_weighted_cv', 0.0)
                logger.info(f"📊 Average Weighted CV: {avg_weighted_cv:.3f}")
        
        # Display validation results
        validation_result = hybrid_analysis.get('validation_result', {})
        if validation_result:
            overall_quality = validation_result.get('overall_quality', 0.0)
            validation_passed = validation_result.get('validation_passed', False)
            logger.info(f"🔍 Consensus Validation Quality: {overall_quality:.3f}")
            logger.info(f"✅ Validation Passed: {validation_passed}")
        
        # Display comprehensive quality
        comprehensive_quality = hybrid_analysis.get('comprehensive_quality', {})
        if comprehensive_quality:
            silhouette_score = comprehensive_quality.get('silhouette_score', 0.0)
            calinski_harabasz_score = comprehensive_quality.get('calinski_harabasz_score', 0.0)
            davies_bouldin_score = comprehensive_quality.get('davies_bouldin_score', 0.0)
            
            logger.info("🔬 Comprehensive Quality Metrics:")
            logger.info(f"   Silhouette Score: {silhouette_score:.3f}")
            logger.info(f"   Calinski-Harabasz Score: {calinski_harabasz_score:.3f}")
            logger.info(f"   Davies-Bouldin Score: {davies_bouldin_score:.3f}")
        
    except Exception as e:
        logger.error(f"❌ Results analysis failed: {e}")


async def demonstrate_individual_components(market_data: pd.DataFrame, features: pd.DataFrame):
    """Demonstrate individual enhanced components."""
    try:
        logger.info("🔬 Demonstrating Individual Enhanced Components")
        
        # Create sample predictions for demonstration
        np.random.seed(42)
        nas_predictions = np.random.randint(0, 5, len(market_data))
        tas_predictions = np.random.randint(0, 6, len(market_data))
        
        # 1. Regime Alignment Manager
        logger.info("🔄 Testing Regime Alignment Manager")
        alignment_config = AlignmentConfig(
            method='hungarian',
            min_overlap_threshold=0.1,
            alignment_confidence_threshold=0.3
        )
        aligner = RegimeAlignmentManager(alignment_config)
        alignment_result = aligner.align_regimes(nas_predictions, tas_predictions, market_data)
        
        if 'error' not in alignment_result:
            quality_metrics = alignment_result.get('quality_metrics', {})
            logger.info(f"   Alignment Quality: {quality_metrics.get('alignment_quality', 0.0):.3f}")
            logger.info(f"   Total Alignments: {len(alignment_result.get('alignment_matrix', {}))}")
        
        # 2. Enhanced Economic Evaluator
        logger.info("💰 Testing Enhanced Economic Evaluator")
        economic_config = EconomicEvaluationConfig(
            target_cluster_count_min=6,
            target_cluster_count_max=15,
            max_cluster_distribution=0.25,
            min_cluster_distribution=0.03,
            volatility_cv_weight=0.4,
            returns_cv_weight=0.3,
            volume_cv_weight=0.3,
            momentum_cv_weight=0.1,
            entropy_cv_weight=0.1
        )
        economic_evaluator = EnhancedEconomicEvaluator(economic_config)
        economic_result = economic_evaluator.evaluate_regime_clustering(
            nas_predictions, market_data, features
        )
        
        if 'error' not in economic_result:
            overall_quality = economic_result.get('overall_quality', 0.0)
            cv_optimization = economic_result.get('cv_optimization', {})
            logger.info(f"   Economic Quality: {overall_quality:.3f}")
            logger.info(f"   CV Optimization Score: {cv_optimization.get('cv_optimization_score', 0.0):.3f}")
        
        # 3. Consensus Validator
        logger.info("🔍 Testing Consensus Validator")
        consensus_config = ConsensusValidationConfig(
            min_consensus_quality=0.6,
            enable_multi_objective=True
        )
        validator = ConsensusValidator(consensus_config)
        
        # Create mock results for validation
        nas_result = {'regime_predictions': nas_predictions}
        tas_result = {'regime_predictions': tas_predictions}
        
        validation_result = validator.validate_consensus(
            nas_predictions, nas_result, tas_result, market_data, features
        )
        
        if 'error' not in validation_result:
            overall_quality = validation_result.get('overall_quality', 0.0)
            validation_passed = validation_result.get('validation_passed', False)
            logger.info(f"   Validation Quality: {overall_quality:.3f}")
            logger.info(f"   Validation Passed: {validation_passed}")
        
        # 4. Multi-Objective Optimizer
        logger.info("🎯 Testing Multi-Objective Optimizer")
        multi_objective_config = MultiObjectiveConfig(
            target_cluster_count_min=6,
            target_cluster_count_max=15,
            max_cluster_distribution=0.25,
            min_cluster_distribution=0.03,
            max_iterations=20,  # Reduced for example
            population_size=20,  # Reduced for example
            enable_pareto_frontier=True
        )
        optimizer = MultiObjectiveOptimizer(multi_objective_config)
        optimization_result = optimizer.optimize_regime_clustering(
            nas_predictions, tas_predictions, market_data, features
        )
        
        if optimization_result.get('optimization_success', False):
            best_solution = optimization_result.get('best_solution', {})
            objectives = best_solution.get('objectives', {})
            logger.info(f"   Optimization Success: {optimization_result.get('optimization_success', False)}")
            logger.info(f"   Best Solution Score: {best_solution.get('combined_score', 0.0):.3f}")
            logger.info(f"   Statistical Score: {objectives.get('silhouette_score', 0.0):.3f}")
            logger.info(f"   Economic Score: {objectives.get('economic_significance', 0.0):.3f}")
        
        logger.info("✅ Individual component demonstration completed")
        
    except Exception as e:
        logger.error(f"❌ Individual component demonstration failed: {e}")


def main():
    """Main function to run the enhanced hybrid regime discovery example."""
    try:
        logger.info("🚀 Starting Enhanced Hybrid NAS-TAS Regime Discovery System")
        logger.info("=" * 80)
        
        # Run the enhanced hybrid regime discovery
        asyncio.run(run_enhanced_hybrid_regime_discovery())
        
        logger.info("=" * 80)
        logger.info("✅ Enhanced Hybrid NAS-TAS Regime Discovery System completed")
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()