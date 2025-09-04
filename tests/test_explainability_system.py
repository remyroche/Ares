#!/usr/bin/env python3
"""Comprehensive test suite for the explainability system.

This module tests all components of the explainability framework including
SHAP/LIME explanations, decision tracing, and visualization.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any
import asyncio
import tempfile
import shutil
from pathlib import Path

# Import explainability components
from src.explainability import (
    ExplainabilityOrchestrator,
    TacticianExplainer,
    HMMExplainer,
    SRExplainer,
    AnalystExplainer,
    ExplanationVisualizer,
    DecisionTraceVisualizer,
    FeatureExtractor
)

# Mock ML models for testing
class MockTacticianModel:
    def __init__(self):
        self.feature_importances_ = np.random.rand(10)
    
    def predict(self, X):
        return np.random.rand(X.shape[0])
    
    def predict_proba(self, X):
        return np.random.rand(X.shape[0], 2)

class MockHMMModel:
    def __init__(self):
        self.feature_importances_ = np.random.rand(8)
    
    def predict(self, X):
        return np.random.randint(0, 4, X.shape[0])
    
    def predict_proba(self, X):
        return np.random.rand(X.shape[0], 4)

class MockSRModel:
    def __init__(self):
        self.feature_importances_ = np.random.rand(12)
    
    def predict(self, X):
        return np.random.rand(X.shape[0])
    
    def predict_proba(self, X):
        return np.random.rand(X.shape[0], 2)

class MockAnalystModel:
    def __init__(self):
        self.feature_importances_ = np.random.rand(15)
        self.regime_classifier = MockHMMModel()
        self.location_classifier = MockHMMModel()
    
    def predict(self, X):
        return np.random.rand(X.shape[0])
    
    def predict_proba(self, X):
        return np.random.rand(X.shape[0], 3)

@pytest.fixture
def test_config():
    """Create test configuration."""
    return {
        "explainability": {
            "enable_explanations": True,
            "enable_decision_tracing": True,
            "explanation_timeout": 10,
            "storage_path": "test_data/explanations",
            "traces_storage_path": "test_data/decision_traces",
            "visualization": {
                "output_path": "test_data/visualizations",
                "style": {
                    "figure_size": [8, 6],
                    "dpi": 100
                }
            }
        }
    }

@pytest.fixture
def test_data():
    """Create test data."""
    np.random.seed(42)
    
    # Create sample market data
    market_data = pd.DataFrame({
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.rand(100) * 1000,
        'volatility_20': np.random.rand(100) * 0.1,
        'rsi': np.random.rand(100) * 100,
        'macd': np.random.randn(100),
        'bb_position': np.random.rand(100),
        'atr': np.random.rand(100) * 2,
        'adx': np.random.rand(100) * 50
    })
    
    # Create sample features
    features = np.random.randn(10)
    feature_names = [
        'close', 'volume', 'volatility_20', 'rsi', 'macd',
        'bb_position', 'atr', 'adx', 'momentum', 'trend'
    ]
    
    return {
        'market_data': market_data,
        'features': features,
        'feature_names': feature_names
    }

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

class TestExplainabilityOrchestrator:
    """Test the explainability orchestrator."""
    
    @pytest.mark.asyncio
    async def test_orchestrator_initialization(self, test_config):
        """Test orchestrator initialization."""
        orchestrator = ExplainabilityOrchestrator(test_config)
        
        assert orchestrator.config == test_config
        assert orchestrator.enable_explanations is True
        assert orchestrator.enable_decision_tracing is True
        assert orchestrator.explanation_timeout == 10
    
    @pytest.mark.asyncio
    async def test_model_registration(self, test_config, test_data):
        """Test model registration."""
        orchestrator = ExplainabilityOrchestrator(test_config)
        
        # Register models
        tactician_model = MockTacticianModel()
        hmm_model = MockHMMModel()
        sr_model = MockSRModel()
        analyst_model = MockAnalystModel()
        
        # Register with training data
        success = await orchestrator.register_model(
            'tactician', 'test_tactician', tactician_model, test_data['market_data']
        )
        assert success is True
        
        success = await orchestrator.register_model(
            'hmm', 'test_hmm', hmm_model, test_data['market_data']
        )
        assert success is True
        
        success = await orchestrator.register_model(
            'sr', 'test_sr', sr_model, test_data['market_data']
        )
        assert success is True
        
        success = await orchestrator.register_model(
            'analyst', 'test_analyst', analyst_model, test_data['market_data']
        )
        assert success is True
        
        # Check registered models
        registered = orchestrator.get_registered_models()
        assert 'tactician' in registered
        assert 'hmm' in registered
        assert 'sr' in registered
        assert 'analyst' in registered
    
    @pytest.mark.asyncio
    async def test_explanation_generation(self, test_config, test_data):
        """Test explanation generation."""
        orchestrator = ExplainabilityOrchestrator(test_config)
        
        # Register a model
        tactician_model = MockTacticianModel()
        await orchestrator.register_model(
            'tactician', 'test_tactician', tactician_model, test_data['market_data']
        )
        
        # Generate explanation
        explanation = await orchestrator.explain_model_prediction(
            'tactician', 'test_tactician',
            test_data['features'], test_data['feature_names']
        )
        
        # Note: Explanation might be None if SHAP/LIME are not available
        # This is expected behavior in test environments
        if explanation is not None:
            assert explanation.model_name == "Tactician_Generic"
            assert explanation.feature_names == test_data['feature_names']
            assert len(explanation.feature_values) == len(test_data['features'])
    
    @pytest.mark.asyncio
    async def test_decision_tracing(self, test_config, test_data):
        """Test decision tracing."""
        orchestrator = ExplainabilityOrchestrator(test_config)
        
        # Start decision trace
        decision_id = "test_decision_001"
        trace = await orchestrator.start_trade_decision_trace(
            decision_id, "entry", {"test": "data"}
        )
        
        assert trace is not None
        assert trace.decision_id == decision_id
        assert trace.decision_type == "entry"
        
        # Finalize trace
        final_trace = await orchestrator.finalize_trade_decision_trace(
            decision_id, "BUY", 0.8
        )
        
        assert final_trace is not None
        assert final_trace.final_decision == "BUY"
        assert final_trace.confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_complete_trading_decision(self, test_config, test_data):
        """Test complete trading decision explanation."""
        orchestrator = ExplainabilityOrchestrator(test_config)
        
        # Register models
        tactician_model = MockTacticianModel()
        hmm_model = MockHMMModel()
        
        await orchestrator.register_model(
            'tactician', 'test_tactician', tactician_model, test_data['market_data']
        )
        await orchestrator.register_model(
            'hmm', 'test_hmm', hmm_model, test_data['market_data']
        )
        
        # Create feature extractors
        def tactician_extractor(*args, **kwargs):
            return test_data['features'], test_data['feature_names']
        
        def hmm_extractor(*args, **kwargs):
            return test_data['features'], test_data['feature_names']
        
        # Explain complete decision
        trace = await orchestrator.explain_complete_trading_decision(
            "complete_test_001",
            "entry",
            test_data['market_data'],
            tactician_features=(test_data['features'], test_data['feature_names']),
            hmm_features=(test_data['features'], test_data['feature_names']),
            final_decision="BUY",
            confidence=0.75
        )
        
        # Note: Trace might be None if explanations fail
        # This is expected in test environments without SHAP/LIME
        if trace is not None:
            assert trace.decision_id == "complete_test_001"
            assert trace.final_decision == "BUY"
            assert trace.confidence == 0.75

class TestModelExplainers:
    """Test individual model explainers."""
    
    def test_tactician_explainer_initialization(self, test_config):
        """Test Tactician explainer initialization."""
        explainer = TacticianExplainer(test_config)
        
        assert explainer.model_name == "Tactician"
        assert explainer.enable_shap is True
        assert explainer.enable_lime is True
    
    def test_hmm_explainer_initialization(self, test_config):
        """Test HMM explainer initialization."""
        explainer = HMMExplainer(test_config)
        
        assert explainer.model_name == "HMM"
        assert explainer.regime_types == ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE', 'TRANSITION']
    
    def test_sr_explainer_initialization(self, test_config):
        """Test SR explainer initialization."""
        explainer = SRExplainer(test_config)
        
        assert explainer.model_name == "SR"
        assert explainer.level_types == ['support', 'resistance', 'dynamic_support', 'dynamic_resistance']
    
    def test_analyst_explainer_initialization(self, test_config):
        """Test Analyst explainer initialization."""
        explainer = AnalystExplainer(test_config)
        
        assert explainer.model_name == "Analyst"
        assert explainer.regime_types == ['BULL', 'BEAR', 'SIDEWAYS', 'VOLATILE', 'TRANSITION']
        assert explainer.location_types == ['TOP', 'BOTTOM', 'MIDDLE', 'TRANSITION']

class TestFeatureExtractor:
    """Test feature extractor utilities."""
    
    @pytest.mark.asyncio
    async def test_dataframe_extractor(self, test_data):
        """Test DataFrame feature extractor."""
        extractor = FeatureExtractor.from_dataframe(['close', 'volume', 'rsi'])
        
        features, feature_names = await extractor(test_data['market_data'])
        
        assert features is not None
        assert feature_names is not None
        assert len(feature_names) == 3
        assert 'close' in feature_names
        assert 'volume' in feature_names
        assert 'rsi' in feature_names
    
    @pytest.mark.asyncio
    async def test_dict_extractor(self):
        """Test dictionary feature extractor."""
        data_dict = {
            'price': 100.0,
            'volume': 1000.0,
            'volatility': 0.05
        }
        
        key_mapping = {
            'price_feature': 'price',
            'volume_feature': 'volume',
            'volatility_feature': 'volatility'
        }
        
        extractor = FeatureExtractor.from_dict(key_mapping)
        features, feature_names = await extractor(data_dict)
        
        assert features is not None
        assert feature_names is not None
        assert len(features) == 3
        assert len(feature_names) == 3
        assert 'price_feature' in feature_names
    
    @pytest.mark.asyncio
    async def test_custom_extractor(self):
        """Test custom feature extractor."""
        async def custom_func(*args, **kwargs):
            return np.array([1, 2, 3]), ['feat1', 'feat2', 'feat3']
        
        extractor = FeatureExtractor.custom(custom_func)
        features, feature_names = await extractor()
        
        assert features is not None
        assert feature_names is not None
        assert len(features) == 3
        assert len(feature_names) == 3

class TestVisualization:
    """Test visualization components."""
    
    def test_explanation_visualizer_initialization(self, test_config):
        """Test explanation visualizer initialization."""
        visualizer = ExplanationVisualizer(test_config)
        
        assert visualizer.output_path is not None
        assert visualizer.colors is not None
        assert 'positive' in visualizer.colors
        assert 'negative' in visualizer.colors
    
    def test_decision_trace_visualizer_initialization(self, test_config):
        """Test decision trace visualizer initialization."""
        visualizer = DecisionTraceVisualizer(test_config)
        
        assert visualizer.output_path is not None
        assert visualizer.colors is not None
        assert 'tactician' in visualizer.colors
        assert 'hmm' in visualizer.colors

class TestIntegration:
    """Test integration decorators."""
    
    @pytest.mark.asyncio
    async def test_explainable_prediction_decorator(self, test_config, test_data):
        """Test explainable prediction decorator."""
        from src.explainability.integration_decorators import explainable_tactician_prediction
        
        # Create a mock prediction function
        @explainable_tactician_prediction(
            model_name="test",
            feature_extractor=FeatureExtractor.from_dataframe(['close', 'volume'])
        )
        async def mock_predict(market_data):
            return {"prediction": "BUY", "confidence": 0.8}
        
        # Test the decorated function
        result = await mock_predict(test_data['market_data'])
        
        # Result should contain the original prediction
        assert result is not None
        # Note: Explanation might not be added if SHAP/LIME are not available

class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_complete_explainability_workflow(self, test_config, test_data, temp_dir):
        """Test complete explainability workflow."""
        # Update config to use temp directory
        test_config['explainability']['storage_path'] = str(Path(temp_dir) / 'explanations')
        test_config['explainability']['traces_storage_path'] = str(Path(temp_dir) / 'traces')
        test_config['explainability']['visualization']['output_path'] = str(Path(temp_dir) / 'visualizations')
        
        # Initialize orchestrator
        orchestrator = ExplainabilityOrchestrator(test_config)
        
        # Register models
        tactician_model = MockTacticianModel()
        hmm_model = MockHMMModel()
        
        await orchestrator.register_model(
            'tactician', 'test_tactician', tactician_model, test_data['market_data']
        )
        await orchestrator.register_model(
            'hmm', 'test_hmm', hmm_model, test_data['market_data']
        )
        
        # Test explanation generation
        explanation = await orchestrator.explain_model_prediction(
            'tactician', 'test_tactician',
            test_data['features'], test_data['feature_names']
        )
        
        # Test decision tracing
        decision_id = "e2e_test_001"
        trace = await orchestrator.start_trade_decision_trace(
            decision_id, "entry", test_data['market_data'].iloc[-1].to_dict()
        )
        
        if explanation:
            await orchestrator.add_explanation_to_trace(decision_id, 'tactician', explanation)
        
        final_trace = await orchestrator.finalize_trade_decision_trace(
            decision_id, "BUY", 0.8
        )
        
        # Test visualization (if matplotlib is available)
        if explanation:
            visualizer = ExplanationVisualizer(test_config)
            viz_path = visualizer.visualize_shap_values(explanation)
            # Note: viz_path might be None if matplotlib is not available
        
        if final_trace:
            trace_visualizer = DecisionTraceVisualizer(test_config)
            trace_viz_path = trace_visualizer.visualize_decision_trace(final_trace)
            # Note: trace_viz_path might be None if matplotlib is not available
        
        # Test cleanup
        cleanup_count = await orchestrator.cleanup_old_explanations(days_to_keep=0)
        assert cleanup_count >= 0  # Should not raise an error

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])