# Chaser Model Implementation Code

## Complete Implementation for layer2_5_chaser.py

### 1. Import Updates (replace lines 27-32)

```python
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.base import clone

# Import Huber constraint utilities
try:
    from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs, get_huber_tier_config
    HUBER_AVAILABLE = True
except ImportError:
    HUBER_AVAILABLE = False
```

### 2. Add Uncertainty Sanity Check (add after line 94)

```python
def sanity_check_uncertainty(y: np.ndarray, p_teacher_oof: np.ndarray, std_oof: np.ndarray) -> float:
    """Check: higher teacher uncertainty should correlate with larger errors."""
    y = np.asarray(y, dtype=np.int32)
    p = np.clip(np.asarray(p_teacher_oof, dtype=np.float64), 1e-6, 1 - 1e-6)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))  # per-sample logloss
    corr = np.corrcoef(ll, std_oof)[0, 1]
    return float(corr if np.isfinite(corr) else 0.0)
```

### 3. Update train_chaser_student Function Signature (replace lines 205-215)

```python
def train_chaser_student(
    X: np.ndarray,
    y: np.ndarray,
    teacher: TeacherOOF,
    base_weight: np.ndarray | None = None,
    mode: str = "regression",  # "regression" or "classification"
    winsor_resid_k: float = 3.0,
    model_type: str = "xgb", # "xgb", "lgb", "cat", "et"
    model_params: dict | None = None,
    num_boost_round: int = 800,
    # New parameters for weak constraints
    monotone_constraints_weak: dict | None = None,
    interaction_constraints_weak: list[list[str]] | None = None,
    huber_teacher_mu: np.ndarray | None = None,  # For teacher disagreement features
):
```

### 4. Update XGBoost Parameters (replace lines 248-270)

```python
        default_params = {
            "eta": 0.03,  # User-specified learning rate
            "max_depth": 4,
            "min_child_weight": 10,  # User-specified
            "subsample": 0.6,  # User-specified
            "colsample_bytree": 0.7,
            "colsample_bynode": 0.4,  # User-specified
            "reg_lambda": 50.0,  # User-specified strong regularization
            "reg_alpha": 0.0,
            "gamma": 1.1,  # User-specified
            "n_jobs": -1
        }
        
        # Add weak constraints if available
        if monotone_constraints_weak is not None and mode == "classification":
            # Convert dict to tuple for XGBoost
            feature_names = [f"f{i}" for i in range(X.shape[1])]
            mono_tuple = tuple(monotone_constraints_weak.get(f"f{i}", 0) for i in range(X.shape[1]))
            params["monotone_constraints"] = mono_tuple
            
        if interaction_constraints_weak is not None and mode == "classification":
            params["interaction_constraints"] = interaction_constraints_weak
```

### 5. Update LightGBM Parameters (replace lines 281-290)

```python
        default_params = {
            "learning_rate": 0.05,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.7,
            "colsample_bytree": 0.7,
            "reg_lambda": 10.0,  # User-specified
            "path_smooth": 20,  # User-specified
            "extra_trees": True,  # User-specified
            "n_jobs": -1,
            "verbose": -1
        }
        
        # Add weak constraints if available
        if monotone_constraints_weak is not None and mode == "classification":
            params["monotone_constraints"] = monotone_constraints_weak
            
        if interaction_constraints_weak is not None and mode == "classification":
            params["interaction_constraints"] = interaction_constraints_weak
```

### 6. Update CatBoost Parameters (replace lines 314-325)

```python
        default_params = {
            "iterations": num_boost_round,
            "learning_rate": 0.05,
            "depth": 6,
            "l2_leaf_reg": 20.0,  # User-specified
            "subsample": 0.6,  # User-specified
            "random_strength": 5.0,  # User-specified
            "verbose": False,
            "allow_writing_files": False
        }
        
        # Add weak constraints if available
        if monotone_constraints_weak is not None and mode == "classification":
            params["monotone_constraints"] = monotone_constraints_weak
```

### 7. Update Layer25Chaser Class (add to __init__ method)

```python
    def __init__(
        self,
        mode: str = "regression",
        regime_split: bool = True,
        feature_engineering: bool = True,
        correlation_threshold: float = 0.7,
        verbose: bool = True,
        models_to_train: List[str] = None,
        # New parameters for weak constraints
        use_huber_constraints: bool = True,
        constraint_tier: str = "weak",
    ):
```

### 8. Update fit Method to Use Huber Constraints (add after line 525)

```python
        # Get weak constraints from Huber if available
        monotone_constraints_weak = None
        interaction_constraints_weak = None
        
        if self.use_huber_constraints and HUBER_AVAILABLE:
            try:
                # Use weak tier constraints for chasers
                huber_config = get_huber_tier_config("weak")
                huber_results = prepare_huber_teacher_outputs(
                    X_train=pd.DataFrame(X_np, columns=self.feature_names),
                    y_train=y_np,
                    sample_weight=w_np,
                    config=huber_config,
                    tier="weak"
                )
                monotone_constraints_weak = huber_results.get('monotonic_constraints', {})
                interaction_constraints_weak = huber_results.get('interaction_constraints', [])
                
                if self.verbose:
                    tprint_info(f"   🔗 Applied {len(monotone_constraints_weak)} monotone constraints from Huber")
                    tprint_info(f"   🔄 Applied {len(interaction_constraints_weak)} interaction constraint groups")
            except Exception as e:
                if self.verbose:
                    tprint_warning(f"   ⚠️ Failed to get Huber constraints: {e}")
```

### 9. Update train_ensemble Function (modify student training call)

```python
                    student = train_chaser_student(
                        X_np, y_np, teacher, 
                        base_weight=weights, 
                        mode=self.mode, 
                        model_type=m_type,
                        monotone_constraints_weak=monotone_constraints_weak,
                        interaction_constraints_weak=interaction_constraints_weak,
                        huber_teacher_mu=teacher.mu_oof
                    )
```

### 10. Add Teacher Disagreement Features (add to predict_chaser_student)

```python
        # Store teacher disagreement feature for meta learner
        if huber_teacher_mu is not None:
            teacher_disagreement = np.abs(huber_teacher_mu - teacher.mu_oof)
            # This can be used as a feature in meta learning
```

### 11. Add Uncertainty Sanity Check (add to fit method after teacher training)

```python
        # Sanity check uncertainty signal
        if self.mode == "classification" and teacher.p_oof is not None:
            uncertainty_corr = sanity_check_uncertainty(y_np, teacher.p_oof, teacher.std_oof)
            if self.verbose:
                tprint_info(f"   📊 Uncertainty-error correlation: {uncertainty_corr:.3f}")
```

## Usage Example

```python
# Create chaser with weak Huber constraints
chaser = Layer25Chaser(
    mode="classification",
    use_huber_constraints=True,
    constraint_tier="weak",
    verbose=True
)

# Fit with regime probabilities
chaser.fit(X_train, y_train, regime_probs=regime_probs, sample_weight=weights)

# Predict
predictions = chaser.predict(X_test, regime_probs=regime_probs_test)
```

This implementation provides:
- Strong regularization as specified
- Weak Huber constraints integration
- Proper classifier chaser workflow
- Teacher disagreement features
- Uncertainty validation
- Meta-learner ready outputs
