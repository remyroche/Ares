# Unified Probability Generator Implementation Plan

## Overview

This document outlines the implementation of a unified probability generator that will replace the multiple ML models currently used by both Analyst and Tactician. The goal is to create a single, efficient system that generates all 4 required probabilities with proper calibration and confidence scoring.

## Core Requirements

### Required Probabilities
1. **triple_barrier_probability**: Probability of reaching profit target without hitting stop-loss
2. **direction_probability**: Probability of price moving in predicted direction
3. **magnitude_probability**: Probability of price moving by expected magnitude
4. **barrier_avoidance_probability**: Probability of avoiding adverse price movements

### Usage Requirements
- **Leverage**: Use triple_barrier_probability as primary factor
- **Confidence**: Use direction_probability for trade confidence
- **Position Sizing**: Use magnitude_probability and barrier_avoidance_probability
- **Opening Positions**: Require minimum direction_probability threshold
- **Closing Positions**: Monitor triple_barrier_probability changes

## Implementation Architecture

### 1. Unified Probability Generator Class

```python
class UnifiedProbabilityGenerator:
    """
    Single model for generating all 4 required probabilities.
    Replaces multiple ML models with one efficient multi-output model.
    """
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model = None
        self.is_trained = False
        self.probability_calibrator = None
        
    async def generate_probabilities(
        self, 
        market_data: pd.DataFrame,
        features: np.ndarray,
        symbol: str,
        timeframe: str
    ) -> dict[str, float]:
        """
        Generate all 4 probabilities in a single call.
        
        Returns:
            dict: {
                "triple_barrier_probability": float,
                "direction_probability": float,
                "magnitude_probability": float,
                "barrier_avoidance_probability": float,
                "confidence_score": float,
                "metadata": dict
            }
        """
```

### 2. Multi-Output Model Structure

```python
class MultiOutputProbabilityModel:
    """
    Single model that outputs all 4 probabilities.
    Uses LightGBM with multi-output configuration.
    """
    
    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.models = {}  # One model per probability type
        self.feature_importance = {}
        
    def train(
        self,
        X_train: np.ndarray,
        y_train_dict: dict[str, np.ndarray],
        X_val: np.ndarray = None,
        y_val_dict: dict[str, np.ndarray] = None
    ) -> dict[str, Any]:
        """
        Train separate models for each probability type.
        """
        
    def predict_probabilities(
        self, 
        X: np.ndarray
    ) -> dict[str, float]:
        """
        Generate all 4 probabilities from trained models.
        """
```

### 3. Probability Calibration

```python
class ProbabilityCalibrator:
    """
    Calibrate raw model outputs to ensure proper probability distributions.
    """
    
    def calibrate_probabilities(
        self,
        raw_probabilities: dict[str, float],
        market_conditions: dict[str, Any]
    ) -> dict[str, float]:
        """
        Apply calibration to ensure probabilities are well-calibrated.
        """
        
    def validate_probabilities(
        self,
        probabilities: dict[str, float]
    ) -> bool:
        """
        Validate that all probabilities are within [0, 1] range.
        """
```

## Implementation Steps

### Phase 1: Core Unified Generator

#### Step 1.1: Create Base Structure
- Create `src/analyst/unified_probability_generator.py`
- Implement basic class structure
- Add configuration management
- Add logging and error handling

#### Step 1.2: Implement Multi-Output Model
- Create `MultiOutputProbabilityModel` class
- Implement training for all 4 probability types
- Add feature importance tracking
- Add model persistence

#### Step 1.3: Add Probability Calibration
- Create `ProbabilityCalibrator` class
- Implement calibration methods
- Add validation functions
- Add confidence scoring

### Phase 2: Integration Layer

#### Step 2.1: Analyst Integration
- Modify `src/analyst/analyst.py` to use unified generator
- Remove redundant ML model calls
- Update prediction pipeline
- Add backward compatibility

#### Step 2.2: Tactician Integration
- Create `src/tactician/probability_enhancer.py`
- Implement probability enhancement functions
- Add leverage calculation
- Add position sizing logic

#### Step 2.3: Execution Optimization
- Create `src/tactician/execution_optimizer.py`
- Implement entry/exit timing
- Add risk management
- Add performance monitoring

### Phase 3: Testing and Validation

#### Step 3.1: Unit Testing
- Test probability generation accuracy
- Test calibration effectiveness
- Test integration points
- Test error handling

#### Step 3.2: Integration Testing
- Test Analyst-Tactician communication
- Test probability flow
- Test performance impact
- Test memory usage

#### Step 3.3: Trading Validation
- Test with historical data
- Validate trading decisions
- Compare with old system
- Measure performance improvement

## Detailed Implementation

### 1. Unified Probability Generator

```python
# src/analyst/unified_probability_generator.py

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

class UnifiedProbabilityGenerator:
    """
    Unified probability generator that replaces multiple ML models.
    Generates all 4 required probabilities in a single, efficient call.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("UnifiedProbabilityGenerator")
        
        # Model state
        self.model = None
        self.is_trained = False
        self.last_training_time = None
        
        # Calibration
        self.calibrators = {}
        self.calibration_data = {}
        
        # Configuration
        self.profit_target = config.get("profit_target", 0.02)
        self.stop_loss = config.get("stop_loss", 0.01)
        self.look_ahead_periods = config.get("look_ahead_periods", 20)
        self.magnitude_threshold_factor = config.get("magnitude_threshold_factor", 0.8)
        self.adverse_threshold = config.get("adverse_threshold", 0.01)
        
    async def initialize(self) -> bool:
        """Initialize the unified probability generator."""
        try:
            self.logger.info("Initializing Unified Probability Generator...")
            
            # Initialize multi-output model
            self.model = MultiOutputProbabilityModel(self.config)
            
            # Load pre-trained models if available
            await self._load_pretrained_models()
            
            self.logger.info("✅ Unified Probability Generator initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize: {e}")
            return False
    
    async def generate_probabilities(
        self,
        market_data: pd.DataFrame,
        features: np.ndarray,
        symbol: str,
        timeframe: str
    ) -> Dict[str, float]:
        """
        Generate all 4 required probabilities.
        
        Args:
            market_data: Market data with OHLCV
            features: Feature array for prediction
            symbol: Trading symbol
            timeframe: Timeframe (1m, 5m, etc.)
            
        Returns:
            Dict containing all 4 probabilities and metadata
        """
        try:
            if not self.is_trained:
                self.logger.warning("Model not trained, using fallback probabilities")
                return self._generate_fallback_probabilities()
            
            # Generate raw probabilities
            raw_probabilities = self.model.predict_probabilities(features)
            
            # Apply calibration
            calibrated_probabilities = self._calibrate_probabilities(
                raw_probabilities, market_data
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(calibrated_probabilities)
            
            # Add metadata
            result = {
                **calibrated_probabilities,
                "confidence_score": confidence_score,
                "metadata": {
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "generation_timestamp": datetime.now().isoformat(),
                    "model_type": "unified_probability_generator",
                    "profit_target": self.profit_target,
                    "stop_loss": self.stop_loss
                }
            }
            
            self.logger.debug(f"Generated probabilities for {symbol}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating probabilities: {e}")
            return self._generate_fallback_probabilities()
    
    def _calibrate_probabilities(
        self,
        raw_probabilities: Dict[str, float],
        market_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Apply calibration to raw probabilities."""
        try:
            calibrated = {}
            
            for prob_name, raw_prob in raw_probabilities.items():
                if prob_name in self.calibrators:
                    calibrated[prob_name] = self.calibrators[prob_name].predict_proba(
                        [[raw_prob]]
                    )[0][1]
                else:
                    calibrated[prob_name] = raw_prob
            
            # Validate probabilities
            for prob_name, prob_value in calibrated.items():
                calibrated[prob_name] = np.clip(prob_value, 0.0, 1.0)
            
            return calibrated
            
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")
            return raw_probabilities
    
    def _calculate_confidence_score(
        self,
        probabilities: Dict[str, float]
    ) -> float:
        """Calculate overall confidence score from all probabilities."""
        try:
            # Weighted average of all probabilities
            weights = {
                "triple_barrier_probability": 0.4,
                "direction_probability": 0.3,
                "magnitude_probability": 0.2,
                "barrier_avoidance_probability": 0.1
            }
            
            confidence = sum(
                probabilities.get(prob_name, 0.5) * weight
                for prob_name, weight in weights.items()
            )
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _generate_fallback_probabilities(self) -> Dict[str, float]:
        """Generate fallback probabilities when model is not available."""
        return {
            "triple_barrier_probability": 0.5,
            "direction_probability": 0.5,
            "magnitude_probability": 0.5,
            "barrier_avoidance_probability": 0.5,
            "confidence_score": 0.5,
            "metadata": {
                "model_type": "fallback",
                "generation_timestamp": datetime.now().isoformat()
            }
        }
```

### 2. Multi-Output Model Implementation

```python
class MultiOutputProbabilityModel:
    """
    Multi-output model for generating all 4 probabilities.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models = {}
        self.feature_importance = {}
        self.is_trained = False
        
    def train(
        self,
        X_train: np.ndarray,
        y_train_dict: Dict[str, np.ndarray],
        X_val: np.ndarray = None,
        y_val_dict: Dict[str, np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train separate models for each probability type.
        """
        try:
            results = {}
            
            for prob_name, y_train in y_train_dict.items():
                self.logger.info(f"Training model for {prob_name}")
                
                # Create LightGBM dataset
                train_data = lgb.Dataset(X_train, label=y_train)
                
                if X_val is not None and y_val_dict is not None:
                    val_data = lgb.Dataset(X_val, label=y_val_dict[prob_name])
                else:
                    val_data = None
                
                # Train model
                model = lgb.train(
                    self._get_model_params(prob_name),
                    train_data,
                    valid_sets=[val_data] if val_data else None,
                    num_boost_round=100,
                    early_stopping_rounds=10 if val_data else None
                )
                
                self.models[prob_name] = model
                
                # Store feature importance
                self.feature_importance[prob_name] = model.feature_importance()
                
                results[prob_name] = {
                    "status": "trained",
                    "feature_importance": self.feature_importance[prob_name]
                }
            
            self.is_trained = True
            return results
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            return {"error": str(e)}
    
    def predict_probabilities(self, X: np.ndarray) -> Dict[str, float]:
        """
        Generate all 4 probabilities from trained models.
        """
        try:
            probabilities = {}
            
            for prob_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    # Classification model
                    proba = model.predict_proba(X)
                    probabilities[prob_name] = proba[0][1]  # Positive class probability
                else:
                    # Regression model
                    pred = model.predict(X)
                    probabilities[prob_name] = np.clip(pred[0], 0.0, 1.0)
            
            return probabilities
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            return {
                "triple_barrier_probability": 0.5,
                "direction_probability": 0.5,
                "magnitude_probability": 0.5,
                "barrier_avoidance_probability": 0.5
            }
    
    def _get_model_params(self, prob_name: str) -> Dict[str, Any]:
        """Get model parameters for specific probability type."""
        base_params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1
        }
        
        # Customize parameters based on probability type
        if prob_name == "triple_barrier_probability":
            base_params["num_leaves"] = 25  # More conservative
        elif prob_name == "direction_probability":
            base_params["learning_rate"] = 0.1  # Faster learning
        elif prob_name == "magnitude_probability":
            base_params["objective"] = "regression"
            base_params["metric"] = "rmse"
        elif prob_name == "barrier_avoidance_probability":
            base_params["num_leaves"] = 20  # Most conservative
        
        return base_params
```

### 3. Tactician Probability Enhancer

```python
# src/tactician/probability_enhancer.py

class ProbabilityEnhancer:
    """
    Enhances Analyst probabilities for Tactician decision making.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = system_logger.getChild("ProbabilityEnhancer")
        
    def enhance_probabilities(
        self,
        analyst_probabilities: Dict[str, float],
        market_data: pd.DataFrame,
        position_context: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Enhance Analyst probabilities for tactical decisions.
        """
        try:
            enhanced = {}
            
            # Enhance each probability based on tactical context
            enhanced["triple_barrier_probability"] = self._enhance_triple_barrier(
                analyst_probabilities["triple_barrier_probability"],
                market_data,
                position_context
            )
            
            enhanced["direction_probability"] = self._enhance_direction(
                analyst_probabilities["direction_probability"],
                market_data,
                position_context
            )
            
            enhanced["magnitude_probability"] = self._enhance_magnitude(
                analyst_probabilities["magnitude_probability"],
                market_data,
                position_context
            )
            
            enhanced["barrier_avoidance_probability"] = self._enhance_barrier_avoidance(
                analyst_probabilities["barrier_avoidance_probability"],
                market_data,
                position_context
            )
            
            return enhanced
            
        except Exception as e:
            self.logger.error(f"Enhancement error: {e}")
            return analyst_probabilities
    
    def calculate_leverage(
        self,
        probabilities: Dict[str, float],
        account_balance: float,
        risk_tolerance: float
    ) -> float:
        """
        Calculate leverage based on probabilities.
        """
        try:
            # Base leverage on triple_barrier_probability
            base_leverage = probabilities["triple_barrier_probability"]
            
            # Adjust for confidence
            confidence_factor = probabilities["direction_probability"]
            
            # Adjust for risk tolerance
            risk_factor = min(1.0, risk_tolerance / 0.02)  # Normalize to 2% risk
            
            # Calculate final leverage
            leverage = base_leverage * confidence_factor * risk_factor
            
            # Apply limits
            max_leverage = self.config.get("max_leverage", 10.0)
            leverage = min(leverage, max_leverage)
            
            return leverage
            
        except Exception as e:
            self.logger.error(f"Leverage calculation error: {e}")
            return 1.0
    
    def calculate_position_size(
        self,
        probabilities: Dict[str, float],
        account_balance: float,
        risk_per_trade: float
    ) -> float:
        """
        Calculate position size based on probabilities.
        """
        try:
            # Base size on magnitude probability
            base_size = probabilities["magnitude_probability"]
            
            # Adjust for barrier avoidance
            risk_adjustment = probabilities["barrier_avoidance_probability"]
            
            # Calculate position size
            position_size = account_balance * risk_per_trade * base_size * risk_adjustment
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Position size calculation error: {e}")
            return account_balance * risk_per_trade * 0.5
```

## Testing Strategy

### 1. Unit Tests
- Test probability generation accuracy
- Test calibration effectiveness
- Test enhancement functions
- Test error handling

### 2. Integration Tests
- Test Analyst-Tactician communication
- Test probability flow
- Test performance impact
- Test memory usage

### 3. Trading Tests
- Test with historical data
- Validate trading decisions
- Compare with old system
- Measure performance improvement

## Migration Plan

### Phase 1: Parallel Implementation
1. Implement unified generator alongside existing system
2. Test with historical data
3. Validate accuracy and performance
4. Get approval for migration

### Phase 2: Gradual Migration
1. Start using unified generator for new predictions
2. Monitor performance and accuracy
3. Gradually reduce old system usage
4. Validate trading results

### Phase 3: Full Migration
1. Switch completely to unified system
2. Remove old ML models
3. Optimize performance
4. Document new system

## Success Metrics

1. **Accuracy**: Maintain or improve prediction accuracy
2. **Performance**: Reduce model inference time by 50%
3. **Complexity**: Reduce code complexity by 70%
4. **Maintenance**: Reduce maintenance overhead by 60%
5. **Trading Results**: Maintain or improve trading performance

This implementation will significantly simplify the ML model architecture while maintaining or improving the trading system's effectiveness.