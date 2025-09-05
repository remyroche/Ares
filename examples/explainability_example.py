#!/usr/bin/env python3
import src.explainability
import numpy as np

"""Example usage of the explainability system.

This example demonstrates how to use the explainability framework to:
1. Generate SHAP/LIME explanations for ML models
2. Trace trade decisions back to individual factors
3. Visualize explanations and decision traces
4. Integrate explanations into existing trading pipelines
"""

import asyncio
from datetime import datetime
import yaml
from pathlib import Path

# Import explainability components
import pandas as pd

    ExplainabilityOrchestrator,
    TacticianExplainer,
    HMMExplainer,
    SRExplainer,
    AnalystExplainer,
    ExplanationVisualizer,
    DecisionTraceVisualizer,
    FeatureExtractor,
    explainable_tactician_prediction,
    explainable_hmm_prediction,
    explainable_sr_prediction,
    explainable_analyst_prediction,
    explainable_trading_decision
)

# Mock ML models for demonstration
class MockTacticianModel:
    """Mock Tactician model for demonstration."""
    
    def __init__(self):
        self.feature_importances_ = np.array([0.1, 0.15, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
        self.trained = True
    
    def predict(self, X):
        """Predict trading decision."""
        # Simple mock prediction based on features
        if X.shape[1] >= 3:
            # Use first 3 features for decision
            decision_score = np.sum(X[:, :3], axis=1)
            return np.where(decision_score > 0, "BUY", "SELL")
        return np.array(["HOLD"] * X.shape[0])
    
    def predict_proba(self, X):
        """Predict probabilities."""
        predictions = self.predict(X)
        probas = np.zeros((X.shape[0], 3))
        for i, pred in enumerate(predictions):
            if pred == "BUY":
                probas[i] = [0.1, 0.7, 0.2]  # [SELL, BUY, HOLD]
            elif pred == "SELL":
                probas[i] = [0.7, 0.1, 0.2]
            else:
                probas[i] = [0.2, 0.2, 0.6]
        return probas

class MockHMMModel:
    """Mock HMM model for demonstration."""
    
    def __init__(self):
        self.feature_importances_ = np.array([0.2, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.regime_names = ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE']
    
    def predict(self, X):
        """Predict regime."""
        # Simple mock prediction
        regime_scores = np.sum(X, axis=1)
        regimes = np.zeros(X.shape[0], dtype=int)
        regimes[regime_scores > 0.5] = 0  # BULL
        regimes[regime_scores < -0.5] = 1  # BEAR
        regimes[(regime_scores >= -0.2) & (regime_scores <= 0.2)] = 2  # SIDEWAYS
        regimes[(regime_scores > 0.2) & (regime_scores <= 0.5)] = 3  # VOLATILE
        return regimes
    
    def predict_proba(self, X):
        """Predict regime probabilities."""
        predictions = self.predict(X)
        probas = np.zeros((X.shape[0], 4))
        for i, pred in enumerate(predictions):
            probas[i, pred] = 0.8
            # Add some noise to other regimes
            for j in range(4):
                if j != pred:
                    probas[i, j] = 0.2 / 3
        return probas

class MockSRModel:
    """Mock SR model for demonstration."""
    
    def __init__(self):
        self.feature_importances_ = np.array([0.15, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    def predict(self, X):
        """Predict SR level quality."""
        # Simple mock prediction
        quality_scores = np.mean(X, axis=1)
        return np.clip(quality_scores, 0, 1)
    
    def predict_proba(self, X):
        """Predict breakout probabilities."""
        predictions = self.predict(X)
        probas = np.zeros((X.shape[0], 2))
        probas[:, 0] = 1 - predictions  # No breakout
        probas[:, 1] = predictions      # Breakout
        return probas

class MockAnalystModel:
    """Mock Analyst model for demonstration."""
    
    def __init__(self):
        self.feature_importances_ = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        self.regime_classifier = MockHMMModel()
        self.location_classifier = MockHMMModel()
    
    def predict(self, X):
        """Predict market analysis."""
        # Simple mock prediction combining regime and location
        regime_pred = self.regime_classifier.predict(X)
        location_pred = self.location_classifier.predict(X)
        
        # Combine predictions
        combined = (regime_pred + location_pred) / 2
        return combined
    
    def predict_proba(self, X):
        """Predict analysis probabilities."""
        regime_probas = self.regime_classifier.predict_proba(X)
        location_probas = self.location_classifier.predict_proba(X)
        
        # Average the probabilities
        combined_probas = (regime_probas + location_probas) / 2
        return combined_probas

def create_sample_data():
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Create sample market data
    n_samples = 100
    market_data = pd.DataFrame({
        'close': np.random.randn(n_samples).cumsum() + 100,
        'volume': np.random.rand(n_samples) * 1000,
        'volatility_20': np.random.rand(n_samples) * 0.1,
        'rsi': np.random.rand(n_samples) * 100,
        'macd': np.random.randn(n_samples),
        'bb_position': np.random.rand(n_samples),
        'atr': np.random.rand(n_samples) * 2,
        'adx': np.random.rand(n_samples) * 50,
        'momentum': np.random.randn(n_samples),
        'trend': np.random.randn(n_samples)
    })
    
    # Create sample features for prediction
    features = np.random.randn(9)
    feature_names = [
        'close', 'volume', 'volatility_20', 'rsi', 'macd',
        'bb_position', 'atr', 'adx', 'momentum'
    ]
    
    return market_data, features, feature_names

def load_config():
    """Load explainability configuration."""
    config_path = Path("config/explainability_config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Default configuration
        config = {
            "explainability": {
                "enable_explanations": True,
                "enable_decision_tracing": True,
                "explanation_timeout": 30,
                "storage_path": "data/explanations",
                "traces_storage_path": "data/decision_traces",
                "visualization": {
                    "output_path": "data/visualizations",
                    "style": {
                        "figure_size": [12, 8],
                        "dpi": 300
                    }
                }
            }
        }
    
    return config

async def example_basic_explanations():
    """Example 1: Basic model explanations."""
    print("🔍 Example 1: Basic Model Explanations")
    print("=" * 50)
    
    # Load configuration
    config = await {func}()
    
    # Create sample data
    market_data, features, feature_names = await {func}()
    
    # Initialize explainers
    tactician_explainer = TacticianExplainer(config)
    hmm_explainer = HMMExplainer(config)
    sr_explainer = SRExplainer(config)
    analyst_explainer = AnalystExplainer(config)
    
    # Create mock models
    tactician_model = MockTacticianModel()
    hmm_model = MockHMMModel()
    sr_model = MockSRModel()
    analyst_model = MockAnalystModel()
    
    # Initialize explainers with training data
    await tactician_explainer.initialize_explainers(tactician_model, market_data)
    await hmm_explainer.initialize_explainers(hmm_model, market_data)
    await sr_explainer.initialize_explainers(sr_model, market_data)
    await analyst_explainer.initialize_explainers(analyst_model, market_data)
    
    # Generate explanations
    print("Generating Tactician explanation...")
    tactician_explanation = await tactician_explainer.explain_prediction(
        tactician_model, features, feature_names
    )
    
    print("Generating HMM explanation...")
    hmm_explanation = await hmm_explainer.explain_prediction(
        hmm_model, features, feature_names
    )
    
    print("Generating SR explanation...")
    sr_explanation = await sr_explainer.explain_prediction(
        sr_model, features, feature_names
    )
    
    print("Generating Analyst explanation...")
    analyst_explanation = await analyst_explainer.explain_prediction(
        analyst_model, features, feature_names
    )
    
    # Print results
    if tactician_explanation:
        print(f"✅ Tactician explanation generated: {tactician_explanation.model_name}")
        print(f"   Prediction: {tactician_explanation.prediction}")
        print(f"   Confidence: {tactician_explanation.confidence:.3f}")
    else:
        print("⚠️ Tactician explanation not available (SHAP/LIME may not be installed)")
    
    if hmm_explanation:
        print(f"✅ HMM explanation generated: {hmm_explanation.model_name}")
        print(f"   Prediction: {hmm_explanation.prediction}")
        print(f"   Confidence: {hmm_explanation.confidence:.3f}")
    else:
        print("⚠️ HMM explanation not available (SHAP/LIME may not be installed)")
    
    print()

async def example_decision_tracing():
    """Example 2: Trade decision tracing."""
    print("🔍 Example 2: Trade Decision Tracing")
    print("=" * 50)
    
    # Load configuration
    config = await {func}()
    
    # Create sample data
    market_data, features, feature_names = await {func}()
    
    # Initialize orchestrator
    orchestrator = ExplainabilityOrchestrator(config)
    
    # Register models
    tactician_model = MockTacticianModel()
    hmm_model = MockHMMModel()
    
    await orchestrator.register_model(
        'tactician', 'main', tactician_model, market_data
    )
    await orchestrator.register_model(
        'hmm', 'main', hmm_model, market_data
    )
    
    # Start decision trace
    decision_id = f"example_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"Starting decision trace: {decision_id}")
    
    trace = await orchestrator.start_trade_decision_trace(
        decision_id, "entry", market_data.iloc[-1].to_dict()
    )
    
    # Add model explanations to trace
    tactician_explanation = await orchestrator.explain_model_prediction(
        'tactician', 'main', features, feature_names
    )
    
    if tactician_explanation:
        await orchestrator.add_explanation_to_trace(
            decision_id, 'tactician', tactician_explanation
        )
        print("✅ Added Tactician explanation to trace")
    
    hmm_explanation = await orchestrator.explain_model_prediction(
        'hmm', 'main', features, feature_names
    )
    
    if hmm_explanation:
        await orchestrator.add_explanation_to_trace(
            decision_id, 'hmm', hmm_explanation
        )
        print("✅ Added HMM explanation to trace")
    
    # Finalize trace
    final_trace = await orchestrator.finalize_trade_decision_trace(
        decision_id, "BUY", 0.75
    )
    
    if final_trace:
        print(f"✅ Decision trace finalized")
        print(f"   Final decision: {final_trace.final_decision}")
        print(f"   Confidence: {final_trace.confidence:.3f}")
        print(f"   Top contributing factors: {len(final_trace.top_contributing_factors)}")
        print(f"   Risk factors: {len(final_trace.risk_factors)}")
        print(f"   Opportunity factors: {len(final_trace.opportunity_factors)}")
    
    print()

async def example_complete_trading_decision():
    """Example 3: Complete trading decision explanation."""
    print("🔍 Example 3: Complete Trading Decision Explanation")
    print("=" * 50)
    
    # Load configuration
    config = await {func}()
    
    # Create sample data
    market_data, features, feature_names = await {func}()
    
    # Initialize orchestrator
    orchestrator = ExplainabilityOrchestrator(config)
    
    # Register all models
    tactician_model = MockTacticianModel()
    hmm_model = MockHMMModel()
    sr_model = MockSRModel()
    analyst_model = MockAnalystModel()
    
    await orchestrator.register_model('tactician', 'main', tactician_model, market_data)
    await orchestrator.register_model('hmm', 'main', hmm_model, market_data)
    await orchestrator.register_model('sr', 'main', sr_model, market_data)
    await orchestrator.register_model('analyst', 'main', analyst_model, market_data)
    
    # Explain complete trading decision
    decision_id = f"complete_decision_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    trace = await orchestrator.explain_complete_trading_decision(
        decision_id=decision_id,
        decision_type="entry",
        market_data=market_data,
        tactician_features=(features, feature_names),
        hmm_features=(features, feature_names),
        sr_features=(features, feature_names),
        analyst_features=(features, feature_names),
        final_decision="BUY",
        confidence=0.8
    )
    
    if trace:
        print(f"✅ Complete trading decision explained: {decision_id}")
        print(f"   Decision: {trace.final_decision}")
        print(f"   Confidence: {trace.confidence:.3f}")
        print(f"   Models used: {sum([1 for exp in [trace.tactician_explanation, trace.hmm_explanation, trace.sr_explanation, trace.analyst_explanation] if exp is not None])}")
        print(f"   Contributing factors: {len(trace.top_contributing_factors)}")
        
        # Get trace summary
        summary = await orchestrator.get_decision_trace_summary(decision_id)
        if summary:
            print(f"   Risk factors: {summary['risk_factors_count']}")
            print(f"   Opportunity factors: {summary['opportunity_factors_count']}")
    else:
        print("⚠️ Complete trading decision explanation not available")
    
    print()

async def example_visualization():
    """Example 4: Visualization of explanations."""
    print("🔍 Example 4: Visualization of Explanations")
    print("=" * 50)
    
    # Load configuration
    config = await {func}()
    
    # Create sample data
    market_data, features, feature_names = await {func}()
    
    # Initialize explainer and model
    tactician_explainer = TacticianExplainer(config)
    tactician_model = MockTacticianModel()
    
    await tactician_explainer.initialize_explainers(tactician_model, market_data)
    
    # Generate explanation
    explanation = await tactician_explainer.explain_prediction(
        tactician_model, features, feature_names
    )
    
    if explanation:
        # Initialize visualizer
        visualizer = ExplanationVisualizer(config)
        
        # Create visualizations
        print("Creating SHAP visualization...")
        shap_path = visualizer.visualize_shap_values(explanation)
        if shap_path:
            print(f"✅ SHAP visualization saved to: {shap_path}")
        
        print("Creating LIME visualization...")
        lime_path = visualizer.visualize_lime_explanation(explanation)
        if lime_path:
            print(f"✅ LIME visualization saved to: {lime_path}")
        
        print("Creating feature importance visualization...")
        importance_path = visualizer.visualize_feature_importance(explanation)
        if importance_path:
            print(f"✅ Feature importance visualization saved to: {importance_path}")
        
        print("Creating explanation dashboard...")
        dashboard_path = visualizer.create_explanation_dashboard(explanation)
        if dashboard_path:
            print(f"✅ Explanation dashboard saved to: {dashboard_path}")
    else:
        print("⚠️ No explanation available for visualization")
    
    print()

async def example_integration_decorators():
    """Example 5: Using integration decorators."""
    print("🔍 Example 5: Integration Decorators")
    print("=" * 50)
    
    # Load configuration
    config = await {func}()
    
    # Create sample data
    market_data, features, feature_names = await {func}()
    
    # Example: Decorated Tactician prediction
    @explainable_tactician_prediction(
        model_name="example_tactician",
        feature_extractor=FeatureExtractor.from_dataframe(['close', 'volume', 'rsi'])
    )
    async def predict_trading_decision(market_data):
        """Example trading decision function."""
        # Simulate some trading logic
        current_price = market_data['close'].iloc[-1]
        volume = market_data['volume'].iloc[-1]
        rsi = market_data['rsi'].iloc[-1]
        
        # Simple decision logic
        if rsi < 30 and volume > market_data['volume'].mean():
            return {"decision": "BUY", "confidence": 0.8, "reason": "Oversold with high volume"}
        elif rsi > 70:
            return {"decision": "SELL", "confidence": 0.7, "reason": "Overbought"}
        else:
            return {"decision": "HOLD", "confidence": 0.5, "reason": "Neutral conditions"}
    
    # Example: Decorated HMM prediction
    @explainable_hmm_prediction(
        model_name="example_hmm",
        feature_extractor=FeatureExtractor.from_dataframe(['volatility_20', 'atr', 'adx'])
    )
    async def predict_market_regime(market_data):
        """Example market regime prediction function."""
        volatility = market_data['volatility_20'].iloc[-1]
        atr = market_data['atr'].iloc[-1]
        adx = market_data['adx'].iloc[-1]
        
        # Simple regime logic
        if volatility > 0.05 and adx > 25:
            return {"regime": "VOLATILE", "confidence": 0.8}
        elif adx < 20:
            return {"regime": "SIDEWAYS", "confidence": 0.7}
        elif atr > market_data['atr'].mean():
            return {"regime": "BULL", "confidence": 0.6}
        else:
            return {"regime": "BEAR", "confidence": 0.6}
    
    # Example: Decorated trading decision
    @explainable_trading_decision(
        decision_type="entry",
        model_types=['tactician', 'hmm'],
        feature_extractors={
            'tactician': FeatureExtractor.from_dataframe(['close', 'volume', 'rsi']),
            'hmm': FeatureExtractor.from_dataframe(['volatility_20', 'atr', 'adx'])
        }
    )
    async def make_trading_decision(market_data):
        """Example complete trading decision function."""
        # Get individual predictions
        trading_pred = await predict_trading_decision(market_data)
        regime_pred = await predict_market_regime(market_data)
        
        # Combine decisions
        if trading_pred['decision'] == 'BUY' and regime_pred['regime'] in ['BULL', 'VOLATILE']:
            return {
                'action': 'BUY',
                'confidence': (trading_pred['confidence'] + regime_pred['confidence']) / 2,
                'tactician_decision': trading_pred,
                'regime_analysis': regime_pred
            }
        elif trading_pred['decision'] == 'SELL' and regime_pred['regime'] in ['BEAR', 'VOLATILE']:
            return {
                'action': 'SELL',
                'confidence': (trading_pred['confidence'] + regime_pred['confidence']) / 2,
                'tactician_decision': trading_pred,
                'regime_analysis': regime_pred
            }
        else:
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'tactician_decision': trading_pred,
                'regime_analysis': regime_pred
            }
    
    # Test the decorated functions
    print("Testing decorated trading decision...")
    result = await make_trading_decision(market_data)
    
    print(f"✅ Trading decision result:")
    print(f"   Action: {result['action']}")
    print(f"   Confidence: {result['confidence']:.3f}")
    print(f"   Tactician decision: {result['tactician_decision']['decision']}")
    print(f"   Regime analysis: {result['regime_analysis']['regime']}")
    
    # Check if explanation was added
    if 'decision_trace' in result:
        print(f"   Decision trace available: {result['decision_trace'] is not None}")
    
    print()

async def main():
    """Run all examples."""
    print("🚀 Explainability System Examples")
    print("=" * 60)
    print()
    
    try:
        await example_basic_explanations()
        await example_decision_tracing()
        await example_complete_trading_decision()
        await example_visualization()
        await example_integration_decorators()
        
        print("✅ All examples completed successfully!")
        print()
        print("📝 Note: Some features may not work if SHAP/LIME/matplotlib are not installed.")
        print("   Install them with: pip install shap lime matplotlib plotly")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())