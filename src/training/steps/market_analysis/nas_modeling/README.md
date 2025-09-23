# 🏛️ Advanced Regime Detection System

This directory contains a comprehensive Neural Architecture Search (NAS) implementation specifically designed for **financial market regime detection** using advanced machine learning techniques.

## 🚀 Overview

The regime detection system focuses purely on identifying and classifying market states using:
- **Neural State Space Models**: Continuous state modeling for regime evolution
- **Vision Transformers**: Self-attention for complex temporal patterns
- **Meta-Learning**: Few-shot learning for regime adaptation
- **Advanced Preprocessing**: Multi-resolution feature engineering

## 🏗️ Architecture

### Core Components

```
nas_modeling/
├── core/                          # Core regime detection functionality
│   ├── neural_odes.py            # Neural State Space Models
│   ├── neural_state_space_nas.py # Neural SSM for regime detection
│   ├── rl_nas.py                 # Vision Transformers for time series
│   ├── meta_learning.py          # Meta-learning for adaptation
│   ├── advanced_preprocessing.py # Feature engineering
│   ├── hardware_acceleration.py  # GPU optimization
│   ├── nas_search.py             # Architecture search engine
│   ├── nas_model.py              # Neural network models
│   ├── nas_trainer.py            # Model training utilities
│   └── nas_evaluator.py          # Architecture evaluation
├── search/                       # Search strategies
│   ├── search_space.py           # Architecture search space
│   ├── random_search.py          # Random search strategy
│   ├── bayesian_search.py        # Bayesian optimization
│   └── evolutionary_search.py    # Evolutionary algorithms
├── evaluation/                   # Evaluation metrics
│   └── nas_metrics.py            # Comprehensive NAS metrics
└── utils/                        # Utilities
    ├── nas_utils.py              # NAS helper functions
    └── logging_utils.py          # Logging utilities
```

## 🎯 Key Features

### Advanced Regime Detection
- **Neural State Space Models**: Continuous state modeling for regime evolution
- **Vision Transformers**: Self-attention for complex temporal patterns
- **Meta-Learning**: Few-shot learning for regime adaptation
- **Advanced Preprocessing**: Multi-resolution feature engineering
- **Hardware Acceleration**: GPU support with mixed precision training

### Advanced Regime Detection
- **Neural State Space Models**: Continuous state modeling for regime evolution
- **Vision Transformers**: Self-attention for complex temporal patterns
- **Meta-Learning**: Few-shot learning for regime adaptation
- **Advanced Preprocessing**: Multi-resolution feature engineering
- **Hardware Acceleration**: GPU support with mixed precision training

## 🔧 Quick Start

### Basic Regime Detection

```python
from nas_modeling.core.neural_state_space_nas import ContinuousTimeRegimeDetector
from nas_modeling.core.advanced_preprocessing import AdvancedPreprocessor
from nas_modeling.core.hardware_acceleration import HardwareAccelerator

# Load and preprocess market data
market_data = pd.DataFrame({
    'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...]
})

preprocessor = AdvancedPreprocessor()
processed_data = preprocessor.preprocess(market_data)

# Create regime detector
regime_model = ContinuousTimeRegimeDetector(
    input_size=5,  # OHLCV features
    state_size=64,  # Continuous state dimension
    num_regimes=5   # Number of market regimes
)

# Get regime prediction
regime_probs = regime_model(processed_data)
predicted_regime = np.argmax(regime_probs)
regime_confidence = np.max(regime_probs)

print(f"Predicted regime: {predicted_regime}")
print(f"Confidence: {regime_confidence:.3f}")
```

### Vision Transformer for Time Series

```python
from nas_modeling.core.rl_nas import VisionTransformerTimeSeries

# Create Vision Transformer for time series
vt_model = VisionTransformerTimeSeries(
    sequence_length=100,
    feature_dim=4,  # OHLC features
    patch_size=10,
    embed_dim=64,
    num_heads=8,
    num_layers=6
)

# Process time series data
regime_features = vt_model(market_data_tensor)
print(f"Regime features shape: {regime_features.shape}")
```

### Meta-Learning for Regime Adaptation

```python
from nas_modeling.core.meta_learning import FewShotRegimeLearner, MetaLearningConfig

# Configure meta-learning
meta_config = MetaLearningConfig(
    num_shots=5,  # 5-shot learning
    adaptation_steps=10,
    meta_learning_rate=1e-3
)

# Create few-shot regime learner
regime_learner = FewShotRegimeLearner(meta_config)

# Adapt to new regime with few examples
adaptation_results = regime_learner.few_shot_adaptation(
    support_set, query_set, regime_type="volatility"
)

print(f"Adaptation accuracy: {adaptation_results['accuracy']:.3f}")
```

## 🔍 Features Used for Regime Detection

### 1. **Price-Based Features** (OHLCV Data)
- **Open, High, Low, Close prices**: Raw market price movements
- **Volume**: Trading activity and liquidity
- **Returns**: Price changes over time (log returns, percentage returns)
- **Price ratios**: High/Low, Open/Close relationships

### 2. **Technical Indicators**
- **Trend indicators**: Simple Moving Averages (SMA), Exponential Moving Averages (EMA)
- **Momentum indicators**: Relative Strength Index (RSI), MACD
- **Volatility indicators**: Bollinger Bands, Average True Range (ATR)
- **Volume indicators**: On-Balance Volume (OBV), Volume-Weighted Average Price (VWAP)

### 3. **Wavelet Transform Features**
- **Approximation coefficients**: Low-frequency trends
- **Detail coefficients**: High-frequency fluctuations
- **Multi-resolution analysis**: Different time scales
- **Wavelet energy**: Energy distribution across scales

### 4. **Fourier Analysis Features**
- **Power spectrum**: Frequency domain energy distribution
- **Dominant frequencies**: Most significant periodic components
- **Spectral centroid**: Center of mass of the spectrum
- **Phase information**: Temporal relationships between frequencies

### 5. **Statistical Features**
- **Volatility measures**: Standard deviation, variance, GARCH estimates
- **Skewness and Kurtosis**: Distribution characteristics
- **Autocorrelation**: Temporal dependencies
- **Entropy measures**: Information content and predictability

### 6. **Market Microstructure Features**
- **Price impact**: How trades affect prices
- **Liquidity measures**: Bid-ask spreads, market depth
- **Order flow**: Buy/sell pressure indicators
- **Market efficiency**: Random walk characteristics

### 7. **Seasonal and Cyclical Features**
- **Trend components**: Long-term price direction
- **Seasonal patterns**: Daily, weekly, monthly cycles
- **Business cycle indicators**: Economic regime features
- **Calendar effects**: Month-end, quarter-end effects

## 📊 Regime Detection Metrics

### Classification Performance
- **Accuracy**: Overall regime classification accuracy
- **Precision/Recall/F1**: Per-regime performance metrics
- **Confusion Matrix**: Regime transition analysis
- **AUC-ROC**: Multi-class classification performance

### Confidence and Uncertainty
- **Calibration**: How well confidence matches accuracy
- **Brier Score**: Probability calibration measure
- **Prediction Entropy**: Model uncertainty quantification
- **Confidence Intervals**: Regime prediction reliability

### Temporal Consistency
- **Regime Persistence**: How long regimes typically last
- **Transition Accuracy**: Accuracy of regime change detection
- **Prediction Horizon**: How far ahead regimes can be predicted
- **Stability Score**: Consistency of regime predictions over time

### Robustness Metrics
- **Cross-validation Scores**: Robustness across different time periods
- **Stress Test Performance**: Performance under extreme market conditions
- **Noise Sensitivity**: Sensitivity to data perturbations
- **Generalization Gap**: Train vs validation performance differences

## 🎨 Neural Architectures for Regime Detection

### Neural State Space Models
- **Continuous State Evolution**: No discrete state assumptions
- **Neural ODEs**: Differential equations solved with neural networks
- **Adaptive Integration**: Efficient time-stepping with event detection
- **State Classification**: Regimes from continuous state representations

### Vision Transformers
- **Self-Attention Mechanism**: Multi-head attention for temporal patterns
- **Patch Embedding**: Time series sequences treated as images
- **Positional Encoding**: Temporal structure preservation
- **Global Context**: Long-range dependency modeling

### Meta-Learning Architectures
- **MAML**: Model-Agnostic Meta-Learning for few-shot adaptation
- **Prototypical Networks**: Metric-based few-shot classification
- **Uncertainty Estimation**: Monte Carlo dropout and ensemble methods
- **Continual Learning**: Adaptation to evolving market conditions

### Hybrid Architectures
- **Neural ODE + Attention**: Continuous states with attention mechanisms
- **Transformer + Meta-Learning**: Self-attention with adaptation capabilities
- **Multi-Modal**: Combining different neural architectures

## 🛠️ Configuration

### Neural State Space Model Configuration
```python
from nas_modeling.core.neural_odes import NeuralSSMConfig

config = NeuralSSMConfig(
    state_size=64,           # Continuous state dimension
    hidden_size=128,         # Hidden layer size
    time_points=20,          # Integration time points
    method="dopri5",         # ODE solver method
    rtol=1e-5,               # Relative tolerance
    atol=1e-6,               # Absolute tolerance
    use_adjoint=True,        # Use adjoint method for efficiency
    event_detection=True     # Detect regime changes
)
```

### Vision Transformer Configuration
```python
from nas_modeling.core.rl_nas import VisionTransformerTimeSeries

vt_model = VisionTransformerTimeSeries(
    sequence_length=100,     # Time series length
    feature_dim=4,           # OHLC features
    patch_size=10,           # Patch size for embedding
    embed_dim=64,            # Embedding dimension
    num_heads=8,             # Attention heads
    num_layers=6             # Transformer layers
)
```

### Meta-Learning Configuration
```python
from nas_modeling.core.meta_learning import MetaLearningConfig

config = MetaLearningConfig(
    meta_learning_rate=1e-3,  # Meta-optimizer learning rate
    inner_learning_rate=0.01, # Inner-loop adaptation rate
    num_inner_steps=5,        # Inner adaptation steps
    num_shots=5,              # K-shot learning
    num_ways=5,               # N-way classification
    adaptation_steps=10       # Adaptation steps
)
```

## 📈 Usage Examples

### Complete Regime Detection Pipeline
```python
# 1. Load and preprocess market data
market_data = pd.DataFrame({
    'open': [...], 'high': [...], 'low': [...], 'close': [...], 'volume': [...]
})

from nas_modeling.core.advanced_preprocessing import AdvancedPreprocessor
preprocessor = AdvancedPreprocessor()
processed_data = preprocessor.preprocess(market_data)

# 2. Create Neural State Space Model for regime detection
from nas_modeling.core.neural_state_space_nas import ContinuousTimeRegimeDetector
regime_model = ContinuousTimeRegimeDetector(
    input_size=5,    # OHLCV features
    state_size=64,   # Continuous state dimension
    num_regimes=5    # Number of market regimes
)

# 3. Get regime prediction
regime_probs = regime_model(processed_data)
predicted_regime = np.argmax(regime_probs)
regime_confidence = np.max(regime_probs)

print(f"Predicted regime: {predicted_regime}")
print(f"Confidence: {regime_confidence:.3f}")
```

### Vision Transformer for Time Series Analysis
```python
from nas_modeling.core.rl_nas import VisionTransformerTimeSeries

# Create Vision Transformer for time series
vt_model = VisionTransformerTimeSeries(
    sequence_length=100,  # Time series length
    feature_dim=4,        # OHLC features
    patch_size=10,        # Patch size
    embed_dim=64,         # Embedding dimension
    num_heads=8,          # Attention heads
    num_layers=6          # Transformer layers
)

# Process time series data
regime_features = vt_model(market_data_tensor)
print(f"Regime features shape: {regime_features.shape}")
```

### Few-Shot Regime Adaptation
```python
from nas_modeling.core.meta_learning import FewShotRegimeLearner, MetaLearningConfig

# Configure meta-learning
meta_config = MetaLearningConfig(
    num_shots=5,        # 5-shot learning
    adaptation_steps=10,
    meta_learning_rate=1e-3
)

# Create few-shot regime learner
regime_learner = FewShotRegimeLearner(meta_config)

# Adapt to new regime with few examples
adaptation_results = regime_learner.few_shot_adaptation(
    support_set, query_set, regime_type="volatility"
)

print(f"Adaptation accuracy: {adaptation_results['accuracy']:.3f}")
```

## 🎯 Performance Tips for Regime Detection

### Hardware Optimization
- **GPU Usage**: Enable `use_gpu=True` for faster training
- **Mixed Precision**: Use FP16 for memory efficiency
- **Batch Size**: Tune based on available memory (typical: 32-128)
- **Sequence Length**: Adjust based on memory constraints

### Model Efficiency
- **State Size**: Balance between expressiveness and efficiency (32-128)
- **Integration Points**: More points for accuracy, fewer for speed
- **Attention Heads**: 4-16 heads for optimal performance
- **Layer Depth**: 3-8 layers depending on complexity

### Training Optimization
- **Early Stopping**: Enable to prevent overfitting
- **Learning Rate Scheduling**: Use cosine annealing or step decay
- **Gradient Clipping**: Prevent exploding gradients
- **Layer Normalization**: Improve training stability

## 🔧 Integration with Market Analysis

### Regime Detection Pipeline
The regime detection system integrates with market analysis workflows:

1. **Data Preprocessing**: Advanced feature engineering with wavelets, Fourier, technical indicators
2. **Model Training**: Train Neural State Space Models and Vision Transformers
3. **Regime Classification**: Identify current market regime with confidence scores
4. **Adaptation**: Use meta-learning to adapt to new market conditions
5. **Monitoring**: Track regime stability and transitions over time

### Risk Management Applications
- **Regime-Based Risk Assessment**: Adjust risk parameters based on detected regime
- **Volatility Regime Detection**: Identify high/low volatility periods
- **Market Stress Detection**: Recognize crisis vs normal market conditions
- **Liquidity Regime Analysis**: Detect periods of high/low market liquidity

### Portfolio Management Applications
- **Asset Allocation**: Adjust portfolio weights based on market regime
- **Strategy Selection**: Choose appropriate strategies for current regime
- **Rebalancing Triggers**: Use regime changes as rebalancing signals
- **Risk Budgeting**: Allocate risk budget based on regime characteristics

## 📋 Requirements

### Dependencies
- **PyTorch**: >= 1.9.0 (for neural networks)
- **torchdiffeq**: For Neural ODEs (pip install torchdiffeq)
- **NumPy**: >= 1.21.0 (for numerical operations)
- **SciPy**: >= 1.7.0 (for optimization)
- **Scikit-learn**: >= 1.0.0 (for metrics)
- **PyWavelets**: For wavelet transforms (pip install PyWavelets)
- **CUDA**: Optional (for GPU acceleration)

### Hardware Requirements
- **Minimum**: CPU with 4GB RAM
- **Recommended**: GPU with 8GB+ VRAM
- **Optimal**: Multi-GPU setup for parallel training

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```python
# Reduce batch size
batch_size = 16  # Instead of 32

# Reduce sequence length
sequence_length = 50  # Instead of 100

# Use gradient checkpointing
config = NeuralSSMConfig(use_checkpointing=True)
```

**Slow Training**
```python
# Use mixed precision
from hardware_acceleration import HardwareConfig
config = HardwareConfig(use_mixed_precision=True)

# Reduce model size
config = NeuralSSMConfig(state_size=32, hidden_size=64)
```

**Poor Regime Classification**
```python
# Increase model capacity
config = NeuralSSMConfig(state_size=128, hidden_size=256)

# Add more training data
# Use data augmentation
augmentation = DataAugmenter()
augmented_data = augmentation.augment_time_series(data)
```

**Model Not Adapting to New Regimes**
```python
# Increase meta-learning rate
config = MetaLearningConfig(meta_learning_rate=1e-3)

# Use more adaptation steps
config = MetaLearningConfig(adaptation_steps=20)

# Add regime-specific preprocessing
preprocessor = AdvancedPreprocessor()
processed_data = preprocessor.preprocess(market_data)
```

## 📚 Advanced Usage

### Custom Regime Detection Pipeline
```python
# Combine multiple models for robust regime detection
from nas_modeling.core.neural_odes import ContinuousTimeRegimeDetector
from nas_modeling.core.rl_nas import VisionTransformerTimeSeries
from nas_modeling.core.meta_learning import FewShotRegimeLearner

# Ensemble of regime detectors
regime_models = {
    'neural_ode': ContinuousTimeRegimeDetector(state_size=64),
    'vision_transformer': VisionTransformerTimeSeries(sequence_length=100),
    'few_shot_learner': FewShotRegimeLearner(MetaLearningConfig())
}

# Get ensemble prediction
def ensemble_regime_prediction(models, data):
    predictions = []
    for name, model in models.items():
        pred = model(data)
        predictions.append(pred)

    # Average predictions with weights
    weights = [0.5, 0.3, 0.2]  # Neural ODE gets highest weight
    ensemble_pred = sum(p * w for p, w in zip(predictions, weights))
    return ensemble_pred
```

### Feature Importance Analysis
```python
# Analyze which features contribute most to regime detection
def analyze_feature_importance(regime_model, market_data):
    # Use integrated gradients or SHAP for feature importance
    # Identify key indicators for each regime type
    feature_importance = {}

    for regime_id in range(5):  # 5 regimes
        importance = compute_regime_feature_importance(regime_model, market_data, regime_id)
        feature_importance[f"regime_{regime_id}"] = importance

    return feature_importance
```

### Regime Transition Analysis
```python
# Analyze regime transitions and stability
def analyze_regime_transitions(regime_predictions, timestamps):
    transitions = detect_regime_changes(regime_predictions)
    transition_matrix = compute_transition_probabilities(regime_predictions)
    regime_persistence = calculate_regime_durations(regime_predictions, timestamps)

    return {
        'transitions': transitions,
        'transition_matrix': transition_matrix,
        'persistence': regime_persistence
    }
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **New Neural Architectures**: Implement additional regime detection models
2. **Feature Engineering**: Add new preprocessing techniques
3. **Evaluation Metrics**: Enhance regime detection evaluation
4. **Meta-Learning**: Improve few-shot adaptation capabilities
5. **Documentation**: Add more examples and tutorials

## 📄 License

This module is part of the Ares trading system and follows the same licensing terms.

## 📞 Support

For support and questions, please refer to the main project documentation or create an issue in the project repository.