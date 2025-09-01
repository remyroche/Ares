# Multi-Output Training Implementation Plan

## 🎯 **Overview**

This plan transforms the probability output implementation from **post-training calculation** to **multi-output training**, where models are specifically trained to generate the 4 required probability outputs.

## 📋 **Current State Analysis**

### **Current Approach: Post-Training Calculation**
```python
# Current: Train standard model, calculate probabilities after
model.fit(X_train, y_train)  # Standard classification
probabilities = calculate_probabilities_post_training(model, X_test, market_data)
```

### **Target Approach: Multi-Output Training**
```python
# Target: Train model specifically for probability outputs
multi_output_model.fit(X_train, y_train_multi, market_data)
probabilities = multi_output_model.predict_probabilities(X_test, market_data)
```

## 🎯 **Phase 1: Foundation Setup (Week 1)**

### **1.1 Create Multi-Output Training Framework**
**Files to Create:**
- `src/training/multi_output_probability_trainer.py`
- `src/training/probability_target_generator.py`
- `src/training/multi_output_model.py`

**Tasks:**
- [ ] Design multi-output model architecture
- [ ] Create probability target generation functions
- [ ] Implement multi-output training pipeline
- [ ] Add validation and error handling

**Deliverables:**
```python
class MultiOutputProbabilityTrainer:
    def __init__(self, config):
        self.config = config
        self.target_generator = ProbabilityTargetGenerator()
        self.multi_output_model = MultiOutputModel()

    def prepare_multi_output_targets(self, X, y, market_data):
        """Generate 4 probability targets for training."""
        return {
            "triple_barrier": self.target_generator.generate_triple_barrier_targets(X, y, market_data),
            "direction": self.target_generator.generate_direction_targets(X, y),
            "magnitude": self.target_generator.generate_magnitude_targets(X, y, market_data),
            "barrier_avoidance": self.target_generator.generate_barrier_avoidance_targets(X, y, market_data)
        }

    def train_multi_output_model(self, X_train, y_train_multi, X_val, y_val_multi):
        """Train model on all 4 probability targets."""
        return self.multi_output_model.fit(X_train, y_train_multi, X_val, y_val_multi)

    def predict_probabilities(self, X_test, market_data):
        """Generate all 4 probability outputs."""
        return self.multi_output_model.predict_probabilities(X_test, market_data)
```

### **1.2 Create Probability Target Generator**
**Tasks:**
- [ ] Implement triple barrier target generation
- [ ] Implement direction target generation
- [ ] Implement magnitude target generation
- [ ] Implement barrier avoidance target generation
- [ ] Add target validation and quality checks

**Deliverables:**
```python
class ProbabilityTargetGenerator:
    def generate_triple_barrier_targets(self, X, y, market_data):
        """Generate triple barrier probability targets."""
        # Calculate actual triple barrier outcomes
        # Convert to probability targets
        pass

    def generate_direction_targets(self, X, y):
        """Generate direction probability targets."""
        # Calculate actual direction accuracy
        # Convert to probability targets
        pass

    def generate_magnitude_targets(self, X, y, market_data):
        """Generate magnitude probability targets."""
        # Calculate actual magnitude outcomes
        # Convert to probability targets
        pass

    def generate_barrier_avoidance_targets(self, X, y, market_data):
        """Generate barrier avoidance probability targets."""
        # Calculate actual avoidance outcomes
        # Convert to probability targets
        pass
```

### **1.3 Create Multi-Output Model Architecture**
**Tasks:**
- [ ] Design model architecture for 4 outputs
- [ ] Implement custom loss functions
- [ ] Add ensemble capabilities
- [ ] Implement probability calibration

**Deliverables:**
```python
class MultiOutputModel:
    def __init__(self):
        self.triple_barrier_model = None
        self.direction_model = None
        self.magnitude_model = None
        self.avoidance_model = None
        self.ensemble_weights = None

    def fit(self, X_train, y_train_multi, X_val, y_val_multi):
        """Train all 4 probability models."""
        # Train each model on its specific target
        # Optimize ensemble weights
        # Calibrate probabilities
        pass

    def predict_probabilities(self, X_test, market_data):
        """Generate all 4 probability outputs."""
        return {
            "triple_barrier_probability": self.triple_barrier_model.predict_proba(X_test),
            "direction_probability": self.direction_model.predict_proba(X_test),
            "magnitude_probability": self.magnitude_model.predict_proba(X_test),
            "barrier_avoidance_probability": self.avoidance_model.predict_proba(X_test)
        }
```

## 📋 **Phase 2: Target Generation Implementation (Week 2)**

### **2.1 Triple Barrier Target Generation**
**Implementation:**
```python
def generate_triple_barrier_targets(self, X, y, market_data, profit_target=0.02, stop_loss=0.01):
    """Generate triple barrier probability targets."""
    targets = []

    for i in range(len(X)):
        # Calculate actual triple barrier outcome
        entry_price = market_data['close'].iloc[i]
        future_prices = market_data['close'].iloc[i+1:i+21]  # Look ahead 20 periods

        # Check if profit target or stop loss hit first
        profit_hit = any(future_prices >= entry_price * (1 + profit_target))
        stop_hit = any(future_prices <= entry_price * (1 - stop_loss))

        if profit_hit and not stop_hit:
            target = 1.0  # Success
        elif stop_hit and not profit_hit:
            target = 0.0  # Failure
        else:
            # Partial success or no clear outcome
            target = 0.5

        targets.append(target)

    return np.array(targets)
```

### **2.2 Direction Target Generation**
**Implementation:**
```python
def generate_direction_targets(self, X, y):
    """Generate direction probability targets."""
    targets = []

    for i in range(len(X)):
        # Calculate actual direction accuracy
        predicted_direction = np.sign(y[i])
        actual_direction = np.sign(y[i])  # Assuming y contains actual price changes

        if predicted_direction == actual_direction:
            target = 1.0  # Correct direction
        else:
            target = 0.0  # Wrong direction

        targets.append(target)

    return np.array(targets)
```

### **2.3 Magnitude Target Generation**
**Implementation:**
```python
def generate_magnitude_targets(self, X, y, market_data, threshold_factor=0.8):
    """Generate magnitude probability targets."""
    targets = []

    for i in range(len(X)):
        # Calculate actual magnitude outcome
        predicted_magnitude = abs(y[i])
        actual_magnitude = abs(market_data['close'].pct_change().iloc[i])

        if predicted_magnitude >= actual_magnitude * threshold_factor:
            target = 1.0  # Magnitude prediction successful
        else:
            target = 0.0  # Magnitude prediction failed

        targets.append(target)

    return np.array(targets)
```

### **2.4 Barrier Avoidance Target Generation**
**Implementation:**
```python
def generate_barrier_avoidance_targets(self, X, y, market_data, adverse_threshold=0.01):
    """Generate barrier avoidance probability targets."""
    targets = []

    for i in range(len(X)):
        # Calculate actual avoidance outcome
        future_returns = market_data['close'].pct_change().iloc[i+1:i+11]  # Look ahead 10 periods
        adverse_movements = abs(future_returns) > adverse_threshold

        if not any(adverse_movements):
            target = 1.0  # Successfully avoided adverse movements
        else:
            target = 0.0  # Hit adverse movement

        targets.append(target)

    return np.array(targets)
```

## 📋 **Phase 3: Multi-Output Model Implementation (Week 3)**

### **3.1 Model Architecture Design**
**Implementation:**
```python
class MultiOutputModel:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.ensemble_weights = None
        self.calibrators = {}

        # Initialize individual models
        self.models['triple_barrier'] = self._create_model('triple_barrier')
        self.models['direction'] = self._create_model('direction')
        self.models['magnitude'] = self._create_model('magnitude')
        self.models['avoidance'] = self._create_model('avoidance')

    def _create_model(self, output_type):
        """Create model for specific output type."""
        if self.config.get('use_lightgbm', True):
            return lgb.LGBMClassifier(
                n_estimators=self.config.get('n_estimators', 1000),
                learning_rate=self.config.get('learning_rate', 0.01),
                max_depth=self.config.get('max_depth', 8),
                random_state=42,
                verbose=-1
            )
        else:
            return RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 200),
                max_depth=self.config.get('max_depth', 10),
                random_state=42
            )
```

### **3.2 Training Pipeline Implementation**
**Implementation:**
```python
def fit(self, X_train, y_train_multi, X_val, y_val_multi):
    """Train all 4 probability models."""
    trained_models = {}

    for output_type in ['triple_barrier', 'direction', 'magnitude', 'avoidance']:
        self.logger.info(f"Training {output_type} model...")

        # Train individual model
        model = self.models[output_type]
        y_train_target = y_train_multi[output_type]
        y_val_target = y_val_multi[output_type]

        # Handle class imbalance
        if output_type in ['triple_barrier', 'avoidance']:
            # These targets are often imbalanced
            class_weights = compute_class_weight('balanced',
                                               classes=np.unique(y_train_target),
                                               y=y_train_target)
            sample_weights = class_weights[y_train_target.astype(int)]
        else:
            sample_weights = None

        # Train model
        if hasattr(model, 'fit'):
            if sample_weights is not None:
                model.fit(X_train, y_train_target, sample_weight=sample_weights)
            else:
                model.fit(X_train, y_train_target)

        # Calibrate probabilities
        calibrator = CalibratedClassifierCV(model, cv=5, method='isotonic')
        calibrator.fit(X_val, y_val_target)
        self.calibrators[output_type] = calibrator

        trained_models[output_type] = calibrator

    # Optimize ensemble weights
    self.ensemble_weights = self._optimize_ensemble_weights(
        trained_models, X_val, y_val_multi
    )

    return trained_models
```

### **3.3 Ensemble Weight Optimization**
**Implementation:**
```python
def _optimize_ensemble_weights(self, models, X_val, y_val_multi):
    """Optimize ensemble weights for better probability accuracy."""
    from scipy.optimize import minimize

    def objective(weights):
        """Objective function to minimize."""
        total_loss = 0

        for output_type in ['triple_barrier', 'direction', 'magnitude', 'avoidance']:
            model = models[output_type]
            y_true = y_val_multi[output_type]
            y_pred_proba = model.predict_proba(X_val)[:, 1]  # Probability of positive class

            # Calculate Brier score (lower is better)
            brier_score = np.mean((y_pred_proba - y_true) ** 2)
            total_loss += brier_score * weights[output_type]

        return total_loss

    # Initial weights (equal)
    initial_weights = {
        'triple_barrier': 0.25,
        'direction': 0.25,
        'magnitude': 0.25,
        'avoidance': 0.25
    }

    # Optimize weights
    result = minimize(
        objective,
        list(initial_weights.values()),
        method='L-BFGS-B',
        bounds=[(0.1, 0.4) for _ in range(4)]  # Constrain weights
    )

    return dict(zip(initial_weights.keys(), result.x))
```

## 📋 **Phase 4: Integration with Training Steps (Week 4)**

### **4.1 Update Step 6: HMM-Based Training**
**File:** `src/training/steps/step6_hmm_based_training.py`

**Tasks:**
- [ ] Import multi-output training framework
- [ ] Replace post-training calculation with multi-output training
- [ ] Update model saving to include trained multi-output models
- [ ] Test with sample data

**Implementation:**
```python
# Add imports
from ..multi_output_probability_trainer import MultiOutputProbabilityTrainer

# Update training function
async def _train_lightgbm_model(self, data, timeframe):
    """Train LightGBM model with multi-output probability training."""
    try:
        # Prepare features (same as before)
        X, y, scaler, label_encoder = self._prepare_features(data, self.specialist_features)

        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create market data for target generation
        market_data = pd.DataFrame({
            'close': data.get('close', np.random.randn(len(data))),
            'volume': data.get('volume', np.random.randn(len(data)))
        })

        # Initialize multi-output trainer
        multi_output_trainer = MultiOutputProbabilityTrainer(self.config)

        # Generate multi-output targets
        y_train_multi = multi_output_trainer.prepare_multi_output_targets(
            X_train, y_train, market_data.iloc[:split_idx]
        )
        y_test_multi = multi_output_trainer.prepare_multi_output_targets(
            X_test, y_test, market_data.iloc[split_idx:]
        )

        # Train multi-output model
        trained_models = multi_output_trainer.train_multi_output_model(
            X_train, y_train_multi, X_test, y_test_multi
        )

        # Generate probability outputs
        price_action_probabilities = multi_output_trainer.predict_probabilities(
            X_test, market_data.iloc[split_idx:]
        )

        # Save model with probabilities
        model_data = {
            "multi_output_models": trained_models,
            "multi_output_trainer": multi_output_trainer,
            "model_type": "multi_output_classification",
            "architecture": "MultiOutputLightGBM",
            # ... other metadata
        }

        return {
            "architecture": "MultiOutputLightGBM",
            "model_data": model_data,
            "price_action_probabilities": price_action_probabilities,
            # ... other return data
        }

    except Exception as e:
        self.logger.exception(f"❌ Multi-output LightGBM training failed: {e}")
        return None
```

### **4.2 Update Step 9: Tactician Specialist Training**
**File:** `src/training/steps/step9_tactician_specialist_training.py`

**Tasks:**
- [ ] Import multi-output training framework
- [ ] Update all model training functions to use multi-output training
- [ ] Replace post-training calculation with multi-output training
- [ ] Test with sample data

**Implementation:**
```python
# Update each training function
async def _train_lightgbm(self, X_train, X_test, y_train, y_test, symbol, exchange):
    """Train LightGBM model with multi-output probability training."""
    try:
        # Initialize multi-output trainer
        multi_output_trainer = MultiOutputProbabilityTrainer(self.config)

        # Generate multi-output targets
        market_data = pd.DataFrame({
            'close': np.random.randn(len(X_train) + len(X_test)),  # Placeholder
            'volume': np.random.randn(len(X_train) + len(X_test))
        })

        y_train_multi = multi_output_trainer.prepare_multi_output_targets(
            X_train, y_train, market_data.iloc[:len(X_train)]
        )
        y_test_multi = multi_output_trainer.prepare_multi_output_targets(
            X_test, y_test, market_data.iloc[len(X_train):]
        )

        # Train multi-output model
        trained_models = multi_output_trainer.train_multi_output_model(
            X_train, y_train_multi, X_test, y_test_multi
        )

        # Generate probability outputs
        price_action_probabilities = multi_output_trainer.predict_probabilities(
            X_test, market_data.iloc[len(X_train):]
        )

        return {
            "multi_output_models": trained_models,
            "multi_output_trainer": multi_output_trainer,
            "model_type": "MultiOutputLightGBM",
            "symbol": symbol,
            "exchange": exchange,
            "price_action_probabilities": price_action_probabilities,
            # ... other return data
        }

    except Exception as e:
        self.logger.exception(f"❌ Multi-output LightGBM training failed: {e}")
        return None
```

## 📋 **Phase 5: Model Saving and Loading Updates (Week 5)**

### **5.1 Update Model Saving Utilities**
**File:** `src/training/model_saving_utils.py`

**Tasks:**
- [ ] Update save function to handle multi-output models
- [ ] Add multi-output model validation
- [ ] Update load function for multi-output models
- [ ] Add backward compatibility

**Implementation:**
```python
def save_multi_output_model_with_probabilities(
    model_data: Dict[str, Any],
    model_path: str,
    save_format: str = "joblib"
) -> Dict[str, Any]:
    """Save multi-output model with probability outputs."""
    try:
        # Extract multi-output components
        multi_output_trainer = model_data.get("multi_output_trainer")
        multi_output_models = model_data.get("multi_output_models")

        # Generate probability outputs
        if multi_output_trainer and multi_output_models:
            # Use test data to generate probabilities
            X_test = model_data.get("X_test", np.random.randn(100, 10))
            market_data = model_data.get("market_data", pd.DataFrame({
                'close': np.random.randn(100),
                'volume': np.random.randn(100)
            }))

            price_action_probabilities = multi_output_trainer.predict_probabilities(
                X_test, market_data
            )
        else:
            price_action_probabilities = model_data.get("price_action_probabilities", {})

        # Create standardized model data structure
        standardized_model_data = {
            "model_type": "multi_output",
            "multi_output_trainer": multi_output_trainer,
            "multi_output_models": multi_output_models,
            "ensemble_weights": multi_output_trainer.ensemble_weights if multi_output_trainer else None,
            "calibrators": multi_output_trainer.calibrators if multi_output_trainer else None,
            "price_action_probabilities": price_action_probabilities,
            "training_date": model_data.get("training_date", datetime.now().isoformat()),
            "hyperparameters": model_data.get("hyperparameters", {}),
            "metrics": model_data.get("metrics", {}),
            "save_timestamp": datetime.now().isoformat(),
            "save_format": save_format
        }

        # Save model
        if save_format.lower() == "joblib":
            joblib.dump(standardized_model_data, model_path)
        else:
            with open(model_path, 'wb') as f:
                pickle.dump(standardized_model_data, f)

        return standardized_model_data

    except Exception as e:
        logger.error(f"Error saving multi-output model: {e}")
        raise
```

### **5.2 Update Model Loading Utilities**
**Implementation:**
```python
def load_multi_output_model_with_probabilities(model_path: str) -> Dict[str, Any]:
    """Load multi-output model with probability outputs."""
    try:
        model_data = load_model_with_probabilities(model_path)

        # Check if it's a multi-output model
        if model_data.get("model_type") == "multi_output":
            multi_output_trainer = model_data.get("multi_output_trainer")
            multi_output_models = model_data.get("multi_output_models")

            if multi_output_trainer and multi_output_models:
                logger.info("✅ Loaded multi-output model successfully")
                return model_data
            else:
                logger.warning("⚠️ Multi-output model missing components")
                return model_data
        else:
            logger.info("ℹ️ Standard model loaded (not multi-output)")
            return model_data

    except Exception as e:
        logger.error(f"Error loading multi-output model: {e}")
        raise
```

## 📋 **Phase 6: Testing and Validation (Week 6)**

### **6.1 Create Multi-Output Testing Framework**
**Tasks:**
- [ ] Create test script for multi-output training
- [ ] Test target generation accuracy
- [ ] Test model training pipeline
- [ ] Test probability prediction accuracy
- [ ] Compare with post-training approach

**Implementation:**
```python
def test_multi_output_training():
    """Test multi-output training framework."""
    # Generate synthetic data
    X = np.random.randn(1000, 10)
    y = np.random.choice([0, 1], size=1000)
    market_data = pd.DataFrame({
        'close': np.random.randn(1000),
        'volume': np.random.randn(1000)
    })

    # Initialize multi-output trainer
    config = {
        'use_lightgbm': True,
        'n_estimators': 100,
        'learning_rate': 0.1
    }
    trainer = MultiOutputProbabilityTrainer(config)

    # Generate targets
    y_multi = trainer.prepare_multi_output_targets(X, y, market_data)

    # Split data
    X_train, X_test = X[:800], X[800:]
    y_train_multi = {k: v[:800] for k, v in y_multi.items()}
    y_test_multi = {k: v[800:] for k, v in y_multi.items()}

    # Train model
    trained_models = trainer.train_multi_output_model(
        X_train, y_train_multi, X_test, y_test_multi
    )

    # Generate predictions
    probabilities = trainer.predict_probabilities(X_test, market_data.iloc[800:])

    # Validate outputs
    for prob_type, prob_value in probabilities.items():
        assert 0.0 <= prob_value <= 1.0, f"Invalid probability for {prob_type}: {prob_value}"

    return probabilities
```

### **6.2 Performance Comparison**
**Tasks:**
- [ ] Compare accuracy of multi-output vs post-training approach
- [ ] Measure training time differences
- [ ] Compare probability calibration quality
- [ ] Test ensemble weight optimization

## 📋 **Phase 7: Documentation and Deployment (Week 7)**

### **7.1 Update Documentation**
**Tasks:**
- [ ] Update implementation plan documentation
- [ ] Create multi-output training user guide
- [ ] Document target generation methods
- [ ] Create migration guide from post-training to multi-output

### **7.2 Deployment Preparation**
**Tasks:**
- [ ] Create deployment scripts for multi-output models
- [ ] Test deployment with new model format
- [ ] Create rollback procedures
- [ ] Prepare monitoring and alerting

## 🎯 **Success Criteria**

### **Technical Criteria**
- [ ] All 4 probability outputs generated by trained models
- [ ] Multi-output training pipeline functional
- [ ] Target generation accurate and validated
- [ ] Ensemble weight optimization working
- [ ] Model saving/loading compatible

### **Performance Criteria**
- [ ] Multi-output training accuracy > post-training accuracy
- [ ] Training time acceptable (< 2x current training time)
- [ ] Probability calibration improved
- [ ] Ensemble weights optimized

### **Functional Criteria**
- [ ] Enhanced Prediction Service loads multi-output models
- [ ] All training steps (6, 9) updated successfully
- [ ] Backward compatibility maintained
- [ ] Error handling comprehensive

## 🚨 **Risk Mitigation**

### **Technical Risks**
- **Risk**: Multi-output training may be slower
- **Mitigation**: Optimize training pipeline, use efficient algorithms

- **Risk**: Target generation may be inaccurate
- **Mitigation**: Validate targets against historical data, add quality checks

- **Risk**: Model complexity may increase
- **Mitigation**: Modular design, clear separation of concerns

### **Timeline Risks**
- **Risk**: Target generation may take longer than expected
- **Mitigation**: Start with simple targets, iterate on complexity

- **Risk**: Integration may be complex
- **Mitigation**: Incremental integration, comprehensive testing

## 📊 **Resource Requirements**

### **Development Resources**
- **1 Senior ML Engineer**: Lead multi-output training framework
- **2 ML Engineers**: Implement target generation and model training
- **1 QA Engineer**: Comprehensive testing and validation

### **Infrastructure Resources**
- **Development Environment**: For testing multi-output training
- **Testing Environment**: For validation and performance testing
- **Documentation Platform**: For updating documentation

## 🎯 **Timeline Summary**

- **Week 1**: Foundation setup (multi-output framework, target generation)
- **Week 2**: Target generation implementation
- **Week 3**: Multi-output model implementation
- **Week 4**: Integration with training steps (6, 9)
- **Week 5**: Model saving and loading updates
- **Week 6**: Testing and validation
- **Week 7**: Documentation and deployment

**Total Duration**: 7 weeks
**Critical Path**: Target generation, multi-output model training

## 🎉 **Expected Outcomes**

### **Immediate Benefits**
- Models specifically trained for probability outputs
- Improved probability accuracy
- Better calibration of probability estimates
- End-to-end optimization

### **Long-term Benefits**
- Foundation for advanced probability modeling
- Scalable framework for additional probability types
- Better integration with risk management systems
- Improved trading decision confidence

This plan provides a comprehensive roadmap for transitioning from post-training probability calculation to multi-output training, ensuring that models are specifically optimized for generating the 4 required probability outputs.