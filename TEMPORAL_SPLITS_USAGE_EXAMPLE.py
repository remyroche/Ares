"""
Example: Using Temporal Splits Throughout the Pipeline

This demonstrates how to use the new temporal splitting system
to ensure proper train/validation/test separation and prevent data leakage.
"""

from datetime import datetime
from src.utils.versioned_artifacts import (
    create_temporal_split_config_for_pipeline,
    get_data_for_purpose
)


# ============================================================================
# STEP 1: Create Temporal Split Configuration (Done Once at Pipeline Start)
# ============================================================================

def setup_pipeline_temporal_splits(symbol="ETHUSDT", exchange="binance", timeframe="15m"):
    """
    Create or load temporal split configuration for the pipeline.

    This should be done once at the start of the pipeline.
    The config is saved to: config/temporal_splits/ETHUSDT_binance_15m.json
    """

    # This will either:
    # 1. Load existing config from config/temporal_splits/
    # 2. Create new config if data range is provided

    config = create_temporal_split_config_for_pipeline(
        symbol=symbol,
        exchange=exchange,
        timeframe=timeframe,
        # Only needed if creating new config:
        data_start=datetime(2020, 1, 1),
        data_end=datetime(2025, 1, 1)
    )

    print(f"✅ Temporal split configuration:")
    print(f"   Training:   {config.training.start} to {config.training.effective_end}")
    print(f"   Validation: {config.validation.start} to {config.validation.effective_end}")
    print(f"   Test:       {config.test.start} to {config.test.end}")

    return config


# ============================================================================
# STEP 2: Use in Model Training Steps
# ============================================================================

class ExampleAnalystTrainingStep:
    """Example of using temporal splits in model training."""

    async def execute(self, config):
        """Execute analyst training with proper temporal separation."""

        # Get temporal config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        # Get features from versioned store
        features_view = self.versioned_store.get_view("features_v1")

        # ✅ Filter to TRAINING PERIOD ONLY
        training_view = get_data_for_purpose(
            features_view,
            purpose='training',  # Only 2020-2023 data
            config=temporal_config
        )

        # Materialize training data
        training_data = training_view.materialize()

        print(f"📚 Training data: {len(training_data)} samples")
        print(f"   Period: {training_data.index[0]} to {training_data.index[-1]}")

        # Train models on training data ONLY
        models = self.train_models(training_data)

        # Generate predictions on training data
        predictions = self.generate_predictions(models, training_data)

        # Save predictions (still timestamped with training period indices)
        self.versioned_store.add_data(
            predictions,
            version_name="analyst_predictions_v1",
            metadata={'period': 'training'}
        )

        return {'success': True}


# ============================================================================
# STEP 3: Use in Parameter Optimization
# ============================================================================

class ExampleFinalParametersOptimizer:
    """Example of using temporal splits in parameter optimization."""

    async def execute(self, config):
        """Optimize parameters on validation period."""

        # Get temporal config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        # Get ML predictions from trained models
        predictions_view = self.versioned_store.get_view("analyst_predictions_v1")

        # ✅ Filter to VALIDATION PERIOD ONLY
        # This data was NOT used for training the ML models!
        validation_view = get_data_for_purpose(
            predictions_view,
            purpose='validation',  # Only 2023-2024 data
            config=temporal_config
        )

        validation_data = validation_view.materialize()

        print(f"🔧 Validation data: {len(validation_data)} samples")
        print(f"   Period: {validation_data.index[0]} to {validation_data.index[-1]}")

        # Optimize parameters using nested CV WITHIN the validation period
        best_params = self.optimize_parameters(
            validation_data,
            cv_folds=5  # CV splits are all WITHIN validation period
        )

        # Save optimized parameters
        self.save_artifact(best_params, "optimized_params")

        return {'success': True, 'params': best_params}


# ============================================================================
# STEP 4: Use in Final Backtesting
# ============================================================================

class ExampleBasicBacktestingPost:
    """Example of using temporal splits in final backtesting."""

    async def execute(self, config):
        """Run final backtest on test period."""

        # Get temporal config
        temporal_config = create_temporal_split_config_for_pipeline(
            symbol=config['symbol'],
            exchange=config['exchange'],
            timeframe=config['timeframe']
        )

        # Get price data
        price_view = self.versioned_store.get_view("price_data")

        # Get ML predictions
        predictions_view = self.versioned_store.get_view("analyst_predictions_v1")

        # ✅ Filter to TEST PERIOD ONLY
        # This data was NOT used for training OR parameter optimization!
        test_price_view = get_data_for_purpose(
            price_view,
            purpose='test',  # Only 2024-2025 data
            config=temporal_config
        )

        test_predictions_view = get_data_for_purpose(
            predictions_view,
            purpose='test',
            config=temporal_config
        )

        # Materialize test data
        test_price_data = test_price_view.materialize()
        test_predictions = test_predictions_view.materialize()

        print(f"🧪 Test data: {len(test_price_data)} samples")
        print(f"   Period: {test_price_data.index[0]} to {test_price_data.index[-1]}")

        # Load optimized parameters
        optimized_params = self.load_artifact("optimized_params")

        # Run backtest on full test period
        backtest_results = self._run_vectorbt_backtest(
            test_price_data,
            test_predictions,
            optimized_params,
            config
        )

        # Run walk-forward CV WITHIN the test period
        # This validates strategy robustness within unseen data
        cv_results = self._run_time_series_cv_backtest(
            test_price_data,
            test_predictions,
            optimized_params,
            config
        )

        return {
            'success': True,
            'backtest_results': backtest_results,
            'cv_results': cv_results
        }


# ============================================================================
# COMPLETE DATA FLOW
# ============================================================================

def demonstrate_complete_data_flow():
    """
    Demonstrates how data flows through the pipeline with proper separation.
    """

    print("=" * 80)
    print("TEMPORAL DATA FLOW DEMONSTRATION")
    print("=" * 80)

    # Setup configuration
    config = setup_pipeline_temporal_splits()

    print("\n📊 Data Flow:")
    print("   All Historical Data (2020-2025)")
    print("   │")
    print("   ├─ Training Period (2020-01-01 to 2023-01-01)")
    print("   │  ├─ Analyst Training Step ──────────┐")
    print("   │  ├─ Tactician Training Step ─────────┤")
    print("   │  └─ All ML models trained here ──────┘")
    print("   │     [Models see ONLY 2020-2023 data]")
    print("   │")
    print("   ├─ [EMBARGO: 30 days]")
    print("   │")
    print("   ├─ Validation Period (2023-02-01 to 2024-01-01)")
    print("   │  └─ Final Parameters Optimization")
    print("   │        └─ Nested CV within validation period")
    print("   │           ├─ Fold 1: Feb-May 2023 → Jun 2023")
    print("   │           ├─ Fold 2: Feb-Aug 2023 → Sep 2023")
    print("   │           └─ etc.")
    print("   │        [Parameters optimized on UNSEEN data]")
    print("   │")
    print("   ├─ [EMBARGO: 30 days]")
    print("   │")
    print("   └─ Test Period (2024-02-01 to 2025-01-01)")
    print("      └─ Basic Backtesting Post")
    print("            └─ Walk-forward CV within test period")
    print("               ├─ Fold 1: Feb-Apr 2024 → May 2024")
    print("               ├─ Fold 2: Feb-Jun 2024 → Jul 2024")
    print("               └─ etc.")
    print("            [Final validation on COMPLETELY UNSEEN data]")
    print()

    print("✅ Data Leakage Prevention:")
    print("   • Training data (2020-2023) ≠ Validation data (2023-2024) ≠ Test data (2024-2025)")
    print("   • 30-day embargo between periods prevents look-ahead bias")
    print("   • CV within each period is safe (no contamination from other periods)")
    print("   • All steps use same config → consistency guaranteed")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    demonstrate_complete_data_flow()

    print("\n" + "=" * 80)
    print("To use this in your pipeline steps:")
    print("=" * 80)
    print("""
    1. Import the functions:
       from src.utils.versioned_artifacts import (
           create_temporal_split_config_for_pipeline,
           get_data_for_purpose
       )

    2. Get temporal config:
       config = create_temporal_split_config_for_pipeline(
           symbol='ETHUSDT',
           exchange='binance',
           timeframe='15m'
       )

    3. Filter data by purpose:
       # In training steps:
       training_data = get_data_for_purpose(view, 'training', config)

       # In optimization steps:
       validation_data = get_data_for_purpose(view, 'validation', config)

       # In backtesting steps:
       test_data = get_data_for_purpose(view, 'test', config)

    See src/utils/versioned_artifacts/TEMPORAL_SPLITS_GUIDE.md for full documentation.
    """)
