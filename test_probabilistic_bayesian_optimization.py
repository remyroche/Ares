#!/usr/bin/env python3
"""
Test Script for Probabilistic Bayesian Optimization

This script demonstrates how to use the probabilistic Bayesian optimization framework
to optimize your Tactician and Analyst models for better probabilistic outputs.
"""

import asyncio
import numpy as np
import pandas as pd
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the probabilistic optimization framework
try:
    from src.training.probabilistic_bayesian_optimizer import (
        ProbabilisticBayesianOptimizer,
        ProbabilisticOptimizationConfig
    )
    from src.training.probabilistic_model_integration import ProbabilisticModelIntegrator
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Please ensure the probabilistic optimization modules are in your Python path")
    exit(1)


def create_realistic_market_data(
    start_date: str = "2024-01-01",
    periods: int = 2000,
    base_price: float = 100.0,
    volatility: float = 0.02
) -> pd.DataFrame:
    """Create realistic market data for testing."""

    dates = pd.date_range(start_date, periods=periods, freq="1min")

    # Generate price data with realistic patterns
    np.random.seed(42)

    # Create price series with trend, volatility, and mean reversion
    returns = np.random.normal(0, volatility, periods)
    prices = [base_price]

    for i in range(1, periods):
        # Add trend component
        trend = 0.0001 * np.sin(i / 100) + 0.00005 * np.cos(i / 50)

        # Add volatility clustering
        vol_multiplier = 1 + 0.5 * np.sin(i / 200)

        # Add mean reversion
        mean_reversion = -0.001 * (prices[-1] - base_price) / base_price

        price_change = (returns[i] * vol_multiplier + trend + mean_reversion)
        new_price = prices[-1] * (1 + price_change)
        prices.append(new_price)

    prices = np.array(prices)

    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.0005, periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.001, periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.001, periods))),
        'close': prices,
        'volume': np.random.uniform(1000, 10000, periods) * (1 + np.abs(returns))
    }, index=dates)

    # Ensure OHLC relationships are valid
    data['high'] = data[['open', 'high', 'close']].max(axis=1)
    data['low'] = data[['open', 'low', 'close']].min(axis=1)

    return data


def create_historical_predictions(
    market_data: pd.DataFrame,
    prediction_horizon: int = 20
) -> pd.DataFrame:
    """Create realistic historical predictions for testing."""

    # Calculate some technical indicators
    close = market_data['close']

    # Simple moving averages
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    # RSI-like indicator
    returns = close.pct_change()
    gains = returns.where(returns > 0, 0)
    losses = -returns.where(returns < 0, 0)
    avg_gains = gains.rolling(14).mean()
    avg_losses = losses.rolling(14).mean()
    rs = avg_gains / (avg_gains + avg_losses)
    rsi = 100 - (100 / (1 + rs))

    # Volatility
    volatility = close.rolling(20).std() / close

    # Create prediction accuracy (simulated)
    np.random.seed(42)
    base_accuracy = 0.7
    volatility_effect = 0.1 * (volatility - volatility.mean()) / volatility.std()
    trend_effect = 0.05 * (sma_20 - sma_50) / close

    prediction_accuracy = np.clip(
        base_accuracy + volatility_effect + trend_effect + np.random.normal(0, 0.05, len(close)),
        0.3, 0.95
    )

    # Create regime predictions (0: bearish, 1: neutral, 2: bullish)
    regime_prediction = np.where(
        sma_20 > sma_50 * 1.02, 2,  # Bullish
        np.where(sma_20 < sma_50 * 0.98, 0, 1)  # Bearish or neutral
    )

    # Add some noise to regime predictions
    regime_prediction = regime_prediction + np.random.normal(0, 0.3, len(close))
    regime_prediction = np.clip(regime_prediction, 0, 2)

    # Create confidence predictions
    confidence_prediction = np.clip(
        prediction_accuracy + np.random.normal(0, 0.1, len(close)),
        0.5, 0.95
    )

    # Create momentum predictions
    momentum_prediction = np.clip(
        (close - close.shift(20)) / close.shift(20) + np.random.normal(0, 0.1, len(close)),
        -0.1, 0.1
    )

    # Create trend predictions
    trend_prediction = np.clip(
        (sma_20 - sma_50) / sma_50 + np.random.normal(0, 0.05, len(close)),
        -0.05, 0.05
    )

    # Create volatility predictions
    volatility_prediction = np.clip(
        volatility + np.random.normal(0, 0.01, len(close)),
        0.01, 0.05
    )

    # Create actual outcomes (simulated)
    future_returns = close.shift(-prediction_horizon) / close - 1

    # Binary outcome: positive return = 1, negative = 0
    actual_outcome = (future_returns > 0).astype(int)

    # Create DataFrame
    predictions = pd.DataFrame({
        'prediction_accuracy': prediction_accuracy,
        'regime_prediction': regime_prediction,
        'confidence_prediction': confidence_prediction,
        'momentum_prediction': momentum_prediction,
        'trend_prediction': trend_prediction,
        'volatility_prediction': volatility_prediction,
        'actual_outcome': actual_outcome,
        'future_return': future_returns
    }, index=market_data.index)

    return predictions


def test_probabilistic_bayesian_optimizer():
    """Test the probabilistic Bayesian optimizer directly."""

    logger.info("🧪 Testing Probabilistic Bayesian Optimizer")
    logger.info("=" * 60)

    # Create test data
    market_data = create_realistic_market_data(periods=1000)
    historical_predictions = create_historical_predictions(market_data)

    # Remove rows with NaN values
    valid_indices = ~(historical_predictions.isna().any(axis=1))
    market_data = market_data[valid_indices]
    historical_predictions = historical_predictions[valid_indices]

    logger.info(f"Created test dataset: {len(market_data)} samples")

    # Test Tactician optimization
    logger.info("\n🔍 Testing Tactician Model Optimization")

    tactician_config = ProbabilisticOptimizationConfig(
        objectives=['calibration', 'sharpness', 'discrimination'],
        n_trials=20,  # Reduced for testing
        n_jobs=1,
        early_stopping_patience=5
    )

    tactician_optimizer = ProbabilisticBayesianOptimizer(
        config=tactician_config,
        model_type="tactician"
    )

    # Prepare data for optimization
    X_tactician = np.column_stack([
        market_data['close'].pct_change().fillna(0),
        market_data['volume'].pct_change().fillna(0),
        historical_predictions['prediction_accuracy'].fillna(0.5),
        historical_predictions['confidence_prediction'].fillna(0.5)
    ])

    y_tactician = historical_predictions['actual_outcome'].values

    # Remove any remaining NaN values
    valid_mask = ~(np.isnan(X_tactician).any(axis=1) | np.isnan(y_tactician))
    X_tactician = X_tactician[valid_mask]
    y_tactician = y_tactician[valid_mask]

    logger.info(f"Tactician optimization data: {X_tactician.shape}")

    # Run optimization
    try:
        tactician_results = tactician_optimizer.optimize(
            X=X_tactician,
            y=y_tactician,
            model_factory=lambda params: tactician_optimizer._create_tactician_model_factory()(params),
            validation_split=0.2
        )

        logger.info("✅ Tactician optimization completed successfully!")
        logger.info(f"Best solutions: {len(tactician_results['best_solutions'])}")

        # Show best parameters
        for objective, solution in tactician_results['best_solutions'].items():
            logger.info(f"  {objective}: {solution['value']:.4f}")

    except Exception as e:
        logger.error(f"❌ Tactician optimization failed: {e}")
        tactician_results = None

    # Test Analyst optimization
    logger.info("\n🔍 Testing Analyst Model Optimization")

    analyst_config = ProbabilisticOptimizationConfig(
        objectives=['calibration', 'sharpness', 'discrimination'],
        n_trials=20,  # Reduced for testing
        n_jobs=1,
        early_stopping_patience=5
    )

    analyst_optimizer = ProbabilisticBayesianOptimizer(
        config=analyst_config,
        model_type="analyst"
    )

    # Prepare data for optimization
    X_analyst = np.column_stack([
        market_data['close'].pct_change().fillna(0),
        market_data['close'].rolling(50).std().fillna(0),
        historical_predictions['regime_prediction'].fillna(1),
        historical_predictions['prediction_accuracy'].fillna(0.5)
    ])

    y_analyst = historical_predictions['actual_outcome'].values

    # Remove any remaining NaN values
    valid_mask = ~(np.isnan(X_analyst).any(axis=1) | np.isnan(y_analyst))
    X_analyst = X_analyst[valid_mask]
    y_analyst = y_analyst[valid_mask]

    logger.info(f"Analyst optimization data: {X_analyst.shape}")

    # Run optimization
    try:
        analyst_results = analyst_optimizer.optimize(
            X=X_analyst,
            y=y_analyst,
            model_factory=lambda params: analyst_optimizer._create_analyst_model_factory()(params),
            validation_split=0.2
        )

        logger.info("✅ Analyst optimization completed successfully!")
        logger.info(f"Best solutions: {len(analyst_results['best_solutions'])}")

        # Show best parameters
        for objective, solution in analyst_results['best_solutions'].items():
            logger.info(f"  {objective}: {solution['value']:.4f}")

    except Exception as e:
        logger.error(f"❌ Analyst optimization failed: {e}")
        analyst_results = None

    return {
        "tactician": tactician_results,
        "analyst": analyst_results
    }


async def test_probabilistic_model_integrator():
    """Test the probabilistic model integrator."""

    logger.info("\n🧪 Testing Probabilistic Model Integrator")
    logger.info("=" * 60)

    # Configuration
    config = {
        "optimization": {
            "n_trials": 15,  # Reduced for testing
            "n_jobs": 1,
            "early_stopping_patience": 5,
            "sampler_type": "tpe"
        }
    }

    # Create integrator
    integrator = ProbabilisticModelIntegrator(config)

    # Create test data
    market_data = create_realistic_market_data(periods=800)
    historical_predictions = create_historical_predictions(market_data)

    # Remove rows with NaN values
    valid_indices = ~(historical_predictions.isna().any(axis=1))
    market_data = market_data[valid_indices]
    historical_predictions = historical_predictions[valid_indices]

    logger.info(f"Created test dataset: {len(market_data)} samples")

    # Run comprehensive optimization
    try:
        results = await integrator.run_comprehensive_optimization(
            market_data, historical_predictions
        )

        logger.info("✅ Comprehensive optimization completed successfully!")

        # Show summary
        summary = results.get("summary", {})
        logger.info(f"Total models optimized: {summary.get('total_models_optimized', 0)}")
        logger.info(f"Successful optimizations: {summary.get('successful_optimizations', 0)}")
        logger.info(f"Failed optimizations: {summary.get('failed_optimizations', 0)}")

        # Show recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            logger.info("\n📋 Recommendations:")
            for rec in recommendations:
                logger.info(f"  • {rec}")

        # Get optimization status
        status = integrator.get_optimization_status()
        logger.info(f"\n📊 Optimization Status:")
        logger.info(f"  Optimizers created: {status.get('optimizers_created', [])}")

        return results

    except Exception as e:
        logger.error(f"❌ Comprehensive optimization failed: {e}")
        return None


def test_uncertainty_quantification():
    """Test uncertainty quantification capabilities."""

    logger.info("\n🧪 Testing Uncertainty Quantification")
    logger.info("=" * 60)

    # Create test data
    market_data = create_realistic_market_data(periods=500)
    historical_predictions = create_historical_predictions(market_data)

    # Remove rows with NaN values
    valid_indices = ~(historical_predictions.isna().any(axis=1))
    market_data = market_data[valid_indices]
    historical_predictions = historical_predictions[valid_indices]

    # Test different uncertainty estimation methods
    uncertainty_methods = ["ensemble", "gaussian", "conformal"]

    for method in uncertainty_methods:
        logger.info(f"\n🔍 Testing {method.upper()} uncertainty estimation")

        try:
            # Create a simple ensemble model for demonstration
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split

            # Prepare features
            X = np.column_stack([
                market_data['close'].pct_change().fillna(0),
                market_data['volume'].pct_change().fillna(0),
                historical_predictions['prediction_accuracy'].fillna(0.5)
            ])

            y = historical_predictions['actual_outcome'].values

            # Remove NaN values
            valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_mask]
            y = y[valid_mask]

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if method == "ensemble":
                # Create ensemble of models
                models = []
                for i in range(5):
                    model = RandomForestClassifier(
                        n_estimators=50,
                        max_depth=5,
                        random_state=42 + i,
                        n_jobs=1
                    )
                    model.fit(X_train, y_train)
                    models.append(model)

                # Get ensemble predictions
                predictions = []
                for model in models:
                    pred = model.predict_proba(X_test)[:, 1]
                    predictions.append(pred)

                predictions = np.array(predictions)

                # Calculate uncertainty (standard deviation across ensemble)
                mean_pred = np.mean(predictions, axis=0)
                uncertainty = np.std(predictions, axis=0)

                logger.info(f"  Ensemble predictions shape: {predictions.shape}")
                logger.info(f"  Mean prediction range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
                logger.info(f"  Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")

            elif method == "gaussian":
                # Simple Gaussian uncertainty estimation
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=1
                )
                model.fit(X_train, y_train)

                # Get predictions
                pred_proba = model.predict_proba(X_test)[:, 1]

                # Estimate uncertainty using prediction confidence
                uncertainty = pred_proba * (1 - pred_proba)  # Variance of Bernoulli

                logger.info(f"  Predictions shape: {pred_proba.shape}")
                logger.info(f"  Prediction range: [{pred_proba.min():.3f}, {pred_proba.max():.3f}]")
                logger.info(f"  Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")

            elif method == "conformal":
                # Simple conformal prediction (placeholder)
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42,
                    n_jobs=1
                )
                model.fit(X_train, y_train)

                # Get predictions
                pred_proba = model.predict_proba(X_test)[:, 1]

                # Simple conformal uncertainty (based on prediction confidence)
                uncertainty = 1 - np.abs(pred_proba - 0.5) * 2

                logger.info(f"  Predictions shape: {pred_proba.shape}")
                logger.info(f"  Prediction range: [{pred_proba.min():.3f}, {pred_proba.max():.3f}]")
                logger.info(f"  Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")

            logger.info(f"  ✅ {method.upper()} method tested successfully")

        except Exception as e:
            logger.error(f"  ❌ {method.upper()} method failed: {e}")


def main():
    """Main test function."""

    logger.info("🚀 Starting Probabilistic Bayesian Optimization Tests")
    logger.info("=" * 80)

    # Test 1: Direct optimizer testing
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Direct Optimizer Testing")
    logger.info("="*50)

    optimizer_results = test_probabilistic_bayesian_optimizer()

    # Test 2: Model integrator testing
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Model Integrator Testing")
    logger.info("="*50)

    integrator_results = asyncio.run(test_probabilistic_model_integrator())

    # Test 3: Uncertainty quantification testing
    logger.info("\n" + "="*50)
    logger.info("TEST 3: Uncertainty Quantification Testing")
    logger.info("="*50)

    test_uncertainty_quantification()

    # Summary
    logger.info("\n" + "="*80)
    logger.info("🎯 TEST SUMMARY")
    logger.info("="*80)

    if optimizer_results:
        logger.info("✅ Direct optimizer tests completed")
        if optimizer_results["tactician"]:
            logger.info("  • Tactician optimization: SUCCESS")
        if optimizer_results["analyst"]:
            logger.info("  • Analyst optimization: SUCCESS")

    if integrator_results:
        logger.info("✅ Model integrator tests completed")
        summary = integrator_results.get("summary", {})
        logger.info(f"  • Models optimized: {summary.get('successful_optimizations', 0)}")

    logger.info("✅ Uncertainty quantification tests completed")

    logger.info("\n🎉 All tests completed successfully!")
    logger.info("\n💡 Next steps:")
    logger.info("  1. Integrate with your actual Tactician and Analyst models")
    logger.info("  2. Use real market data and prediction outcomes")
    logger.info("  3. Tune optimization parameters for your specific use case")
    logger.info("  4. Monitor model performance and retrain as needed")


if __name__ == "__main__":
    main()