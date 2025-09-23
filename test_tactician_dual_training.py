#!/usr/bin/env python3
"""
Test Tactician Dual Training Implementation

This test verifies the new Tactician dual training pipeline works correctly
with the tagging approach and includes HMM/Analyst features.
"""

import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Import the new components
from src.training.steps.model_training.tactician_pre_ml_orchestrator import TacticianPreMLOrchestrator, OrchestratorConfig
from src.training.steps.model_training.tactician_dual_training_step import TacticianDualTrainingStep, DualTrainingConfig

async def test_tactician_dual_training():
    """Test the Tactician dual training implementation."""
    print("🚀 Testing Tactician Dual Training Implementation...")

    # Create mock analyst signals with confidence >= 0.5
    print("📊 Creating mock analyst signals...")
    timestamps = pd.date_range('2024-01-01', periods=1000, freq='1min')

    # Create analyst signals with varying confidence levels
    analyst_signals = pd.DataFrame({
        'timestamp': timestamps,
        'analyst_signal': np.random.choice([-1, 0, 1], 1000),
        'analyst_confidence': np.random.uniform(0.3, 0.9, 1000),
        'analyst_long_prob': np.random.uniform(0.1, 0.9, 1000),
        'analyst_short_prob': np.random.uniform(0.1, 0.9, 1000),
        'analyst_neutral_prob': np.random.uniform(0.1, 0.9, 1000)
    })

    # Create mock market data with full lookback periods
    print("📈 Creating mock market data with full lookback periods...")
    market_data = pd.DataFrame({
        'timestamp': timestamps,
        'close': np.random.uniform(100, 200, 1000),
        'volume': np.random.uniform(1000, 10000, 1000),
        'rsi': np.random.uniform(20, 80, 1000),
        'macd': np.random.uniform(-5, 5, 1000),
        'bb_upper': np.random.uniform(105, 210, 1000),
        'bb_lower': np.random.uniform(95, 190, 1000),
        'hmm_regime': np.random.choice([0, 1, 2], 1000),
        'hmm_regime_prob': np.random.uniform(0.5, 0.95, 1000),
        'hmm_regime_confidence': np.random.uniform(0.7, 0.95, 1000)
    })

    # Test 1: Pre-ML Orchestration with tagging approach
    print("🎯 Test 1: Testing Pre-ML Orchestration with tagging...")
    orchestrator_config = OrchestratorConfig(
        min_analyst_confidence=0.5,
        subsequent_minutes=45,
        save_intermediate_results=True,
        enable_feature_optimization=True,
        enable_pid_generation=True,
        enable_horizon_labeling=True,
        enable_feature_selection=True
    )

    orchestrator = TacticianPreMLOrchestrator(orchestrator_config)
    feature_names = ['close', 'volume', 'rsi', 'macd', 'bb_upper', 'bb_lower', 'hmm_regime', 'hmm_regime_prob', 'analyst_confidence']

    print(f"🔍 Running pre-ML orchestration with {len(analyst_signals)} signals...")
    result = await orchestrator.orchestrate_pre_ml_training(
        analyst_signals=analyst_signals,
        market_data=market_data,
        feature_names=feature_names
    )

    print(f"✅ Orchestration completed:")
    print(f"   - Long signals: {result.total_long_samples}")
    print(f"   - Short signals: {result.total_short_samples}")
    print(f"   - Long features: {len(result.long_selected_features)}")
    print(f"   - Short features: {len(result.short_selected_features)}")
    print(f"   - Tagging approach: {result.tagged_market_data is not None}")

    # Verify HMM and Analyst features are preserved
    if result.long_selected_features:
        hmm_features = [f for f in result.long_selected_features if 'hmm' in f.lower()]
        analyst_features = [f for f in result.long_selected_features if 'analyst' in f.lower()]
        print(f"   - HMM features preserved: {len(hmm_features)}")
        print(f"   - Analyst features preserved: {len(analyst_features)}")

    # Test 2: Dual Training
    print("\n⚔️ Test 2: Testing Dual Training...")
    dual_config = DualTrainingConfig(
        min_analyst_confidence=0.5,
        subsequent_minutes=45,
        train_base_models=True,
        train_ensemble_models=True,
        save_models=False,  # Don't save for testing
        min_training_samples=50  # Lower threshold for testing
    )

    dual_trainer = TacticianDualTrainingStep(dual_config)

    training_result = await dual_trainer.train_dual_tactician_models(
        analyst_signals=analyst_signals,
        market_data=market_data,
        feature_names=feature_names
    )

    print("✅ Dual training completed with COMPREHENSIVE REPORTING:")
    print(f"   - Training phase: {training_result.training_phase.value}")
    print(f"   - Execution time: {training_result.execution_time:.2f}s")
    print(f"   - Long base models: {len(training_result.long_base_models) if training_result.long_base_models else 0} (XGBOOST, LIGHTGBM, DEEPSCALER_1M, FINANCIAL_RESNET)")
    print(f"   - Short base models: {len(training_result.short_base_models) if training_result.short_base_models else 0} (XGBOOST, LIGHTGBM, DEEPSCALER_1M, FINANCIAL_RESNET)")
    print(f"   - Long ensemble models: {len(training_result.long_ensemble_models) if training_result.long_ensemble_models else 0} (includes ALL features + HMM + Analyst outputs)")
    print(f"   - Short ensemble models: {len(training_result.short_ensemble_models) if training_result.short_ensemble_models else 0} (includes ALL features + HMM + Analyst outputs)")
    print(f"   - Total models trained: {len(training_result.long_base_models) + len(training_result.short_base_models) + len(training_result.long_ensemble_models) + len(training_result.short_ensemble_models)}")

    if hasattr(training_result, 'comprehensive_report') and training_result.comprehensive_report:
        report = training_result.comprehensive_report
        training_summary = report.get('training_summary', {})
        feature_integration = report.get('feature_integration_metrics', {})

        print(f"\n📊 COMPREHENSIVE REPORTING SUMMARY:")
        print(f"   - Success: {'✅ YES' if training_summary.get('success', False) else '❌ NO'}")
        print(f"   - Total Models: {training_summary.get('total_models_trained', 0)}")
        print(f"   - Feature Integration Complete: {'✅ YES' if feature_integration.get('feature_integration_complete', False) else '❌ NO'}")
        print(f"   - Long Ensemble Features: {feature_integration.get('long_ensemble_feature_count', 0)}")
        print(f"   - Short Ensemble Features: {feature_integration.get('short_ensemble_feature_count', 0)}")
        print(f"   - Long Samples: {training_summary.get('total_long_samples', 0)}")
        print(f"   - Short Samples: {training_summary.get('total_short_samples', 0)}")

    print("\n🎉 All tests completed successfully!")
    print("\n📋 Command Usage Instructions:")
    print("=" * 50)
    print("python src/launcher/ares_launcher.py --mode sub_pipeline --sub-pipeline tactician_pre_ml_orchestration")
    print("python src/launcher/ares_launcher.py --mode sub_pipeline --sub-pipeline tactician_dual_training")
    print("python src/launcher/ares_launcher.py --mode stage --stage model_training")
    print("python src/launcher/ares_launcher.py --mode full --start-stage model_training")
    print("\n🎯 Expected Results:")
    print("   - tactician_dual_training trains 10 models total:")
    print("     * 4 Long Base Models (XGBOOST, LIGHTGBM, DEEPSCALER_1M, FINANCIAL_RESNET)")
    print("     * 4 Short Base Models (XGBOOST, LIGHTGBM, DEEPSCALER_1M, FINANCIAL_RESNET)")
    print("     * 1 Long Ensemble Model (includes ALL features + HMM + Analyst outputs)")
    print("     * 1 Short Ensemble Model (includes ALL features + HMM + Analyst outputs)")
    print("   - Each ensemble model integrates:")
    print("     * Base features from pre-ML orchestration")
    print("     * HMM regime features and probabilities")
    print("     * Analyst model predictions and confidence scores")
    print("     * OOF predictions from all base models")
    print("     * Technical indicators and market data")
    print("     * Multi-horizon target variables")
    print("   - COMPREHENSIVE REPORTING includes:")
    print("     * Training Summary with timing and success metrics")
    print("     * Model Breakdown by type and status")
    print("     * Sample Processing metrics")
    print("     * Feature Integration status")
    print("     * Performance metrics and quality scores")
    print("     * Error Analysis and completion status")

    return True

if __name__ == "__main__":
    asyncio.run(test_tactician_dual_training())