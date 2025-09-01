# src/analyst/predictive_ensembles/multi_timeframe_ensemble.py

"""
Multi-Timeframe Ensemble Integration

This integrates multi-timeframe training into the existing ensemble system,
making each individual model (XGBoost, LSTM, etc.) a multi-timeframe ensemble.
"""

import os
import time
from datetime import datetime
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.config import CONFIG
from src.utils.logger import system_logger
from src.utils.warning_symbols import (
error,
failed,
warning,
)


class MultiTimeframeEnsemble:
    pass  # TODO: Add implementation
class MultiTimeframeEnsemble:
    pass  # TODO: Add implementation
class MultiTimeframeEnsemble:
    """
Multi-timeframe ensemble that integrates into existing ensemble system.

Each individual model (XGBoost, LSTM, etc.) becomes a multi-timeframe ensemble.
"""

def __init__(
self,
model_name: str,
regime: str,
config: dict[str, Any] | None = None,
):
        self.model_name = model_name
self.regime = regime
self.config = config or CONFIG.get("MULTI_TIMEFRAME_ENSEMBLE", {})
self.logger = system_logger.getChild(
f"MultiTimeframeEnsemble_{model_name}_{regime}",
)

# Timeframe configuration
self.timeframes = CONFIG.get("TIMEFRAMES", {})
self.timeframe_set = CONFIG.get("DEFAULT_TIMEFRAME_SET", "intraday")
self.active_timeframes = CONFIG.get("TIMEFRAME_SETS", {}).get(
self.timeframe_set,
[],
)

# Model storage
self.models_dir = CONFIG.get("MODEL_STORAGE_DIR", "models")
self.timeframe_models: dict[str, Any] = {}
self.meta_learner: Any | None = None
self.meta_scaler: StandardScaler | None = None
self.meta_label_encoder: LabelEncoder | None = None

# Training state
self.trained = False
self.training_history: list[dict[str, Any]] = []

# Log initialization
self.logger.info(
f"🚀 Initializing MultiTimeframeEnsemble for {model_name} in {regime}",
)
self.logger.info(f"📊 Active timeframes: {self.active_timeframes}")
self.logger.info(f"⚙️ Configuration: {self.config}")

def train_multi_timeframe_ensemble(
self,
prepared_data: dict[str, pd.DataFrame],
model_type: str = "xgboost",
) -> bool:
        """
Train multi-timeframe ensemble for this specific model type.

Args:
            prepared_data: Dict with timeframe -> DataFrame mapping
model_type: Type of base model (xgboost, lstm, etc.)

Returns:
            bool: Success status
"""
start_time = time.time()

try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info(
f"🎯 Starting multi-timeframe ensemble training for {self.model_name} in {self.regime}",
)
self.logger.info(f"📈 Model type: {model_type}")
self.logger.info(f"⏰ Available timeframes: {list(prepared_data.keys())}")
self.logger.info(
f"📊 Data shapes: {[(tf, df.shape) for tf, df in prepared_data.items()]}",
)

# 1. Train individual timeframe models
timeframe_predictions = {}
timeframe_confidences = {}
training_stats = {}

for i, timeframe in enumerate(self.active_timeframes, 1):
                self.logger.info(
f"🔄 [{i}/{len(self.active_timeframes)}] Training {timeframe} timeframe...",
)

if timeframe not in prepared_data:
                    self.logger.warning(
f"⚠️ No data for timeframe {timeframe}, skipping",
)
continue

# Train model for this timeframe
tf_start_time = time.time()
success = self._train_single_timeframe(
timeframe,
prepared_data[timeframe],
model_type,
)
tf_training_time = time.time() - tf_start_time

if success:
                    self.logger.info(
f"✅ {timeframe} training completed in {tf_training_time:.2f}s",
)

# Get predictions for meta-learner training
self.logger.info(f"📊 Collecting predictions for {timeframe}...")
predictions, confidences = self._get_timeframe_predictions(
timeframe,
prepared_data[timeframe],
)
timeframe_predictions[timeframe] = predictions
timeframe_confidences[timeframe] = confidences

# Log training statistics
training_stats[timeframe] = {
"training_time": tf_training_time,
"predictions_count": len(predictions),
"avg_confidence": np.mean(confidences) if confidences else 0.0,
"success": True,
}

self.logger.info(
f"📈 {timeframe} stats: {len(predictions)} predictions, "
f"avg confidence: {np.mean(confidences):.3f}",
)
else:
                    self.print(failed("❌ {timeframe} training failed"))
training_stats[timeframe] = {
"training_time": tf_training_time,
"success": False,
}

# 2. Train meta-learner to combine timeframe predictions
if len(timeframe_predictions) > 1:
                self.logger.info(
f"🧠 Training meta-learner with {len(timeframe_predictions)} timeframes...",
)
meta_start_time = time.time()

success = self._train_meta_learner(
timeframe_predictions,
timeframe_confidences,
prepared_data,
)

meta_training_time = time.time() - meta_start_time

if success:
                    self.trained = True
total_time = time.time() - start_time

self.logger.info(
"✅ Multi-timeframe ensemble training completed successfully!",
)
self.logger.info(f"⏱️ Total training time: {total_time:.2f}s")
self.logger.info("📊 Training summary:")
self.logger.info(f"   - Model: {self.model_name}")
self.logger.info(f"   - Regime: {self.regime}")
self.logger.info(
f"   - Timeframes trained: {len(timeframe_predictions)}",
)
self.logger.info(
f"   - Meta-learner training time: {meta_training_time:.2f}s",
)

# Log detailed statistics
for tf, stats in training_stats.items():
                        if stats.get("success"):
                            self.logger.info(
f"   - {tf}: {stats['training_time']:.2f}s, "
f"{stats['predictions_count']} predictions, "
f"avg confidence: {stats['avg_confidence']:.3f}",
)
else:
                            self.print(failed("   - {tf}: FAILED"))

return True
self.print(failed("❌ Meta-learner training failed"))
return False
self.logger.error(
f"❌ Insufficient timeframes ({len(timeframe_predictions)}) for meta-learner training",
)
return False

except Exception:
            self.print(error("💥 Error in multi-timeframe ensemble training: {e}"))
return False

def _train_single_timeframe(
self,
timeframe: str,
data: pd.DataFrame,
model_type: str,
) -> bool:
        """Train a single timeframe model."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info(f"🔧 Training {model_type} model for {timeframe}")
self.logger.info(f"📊 Data shape: {data.shape}")
self.logger.info(f"📈 Data columns: {list(data.columns)}")

# Prepare features and target
X, y = self._prepare_features_target(data)

if len(X) == 0:
                self.print(warning("⚠️ No valid data for {timeframe}"))
return False

self.logger.info(f"📊 Features shape: {X.shape}")
self.logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")

# Train model based on type
if model_type == "xgboost":
                model = self._train_xgboost_model(X, y)
elif model_type == "lstm":
                model = self._train_lstm_model(X, y)
elif model_type == "random_forest":
                model = self._train_random_forest_model(X, y)
else:
                self.print(error("❌ Unknown model type: {model_type}"))
return False

if model is not None:
                self.timeframe_models[timeframe] = {
"model": model,
"model_type": model_type,
"timeframe": timeframe,
"trained_at": datetime.now(),
"features_shape": X.shape,
"target_distribution": y.value_counts().to_dict(),
}

self.logger.info(
f"✅ {timeframe} {model_type} model trained successfully",
)
return True

return False

except Exception:
            self.print(error("💥 Error training {timeframe} model: {e}"))
return False

def _train_xgboost_model(self, X: pd.DataFrame, y: pd.Series) -> Any | None:
        """Train XGBoost model."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info("🌳 Training XGBoost model...")

# Use LightGBM as XGBoost alternative
model = lgb.LGBMClassifier(
n_estimators=100,
learning_rate=0.1,
max_depth=6,
random_state=42,
verbose=-1,
)

# Cross-validation
self.logger.info("🔄 Starting cross-validation...")
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                self.logger.info(
f"📊 Fold {fold}/3: {len(train_idx)} train, {len(val_idx)} validation",
)

X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

model.fit(
X_train,
y_train,
eval_set=[(X_val, y_val)],
eval_metric="logloss",
early_stopping_rounds=10,
verbose=False,
)

self.logger.info("✅ XGBoost model training completed")
return model

except Exception:
            self.print(error("💥 Error training XGBoost model: {e}"))
return None

def _train_lstm_model(self, X: pd.DataFrame, y: pd.Series) -> Any | None:
        """Train LSTM model (simplified for now)."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info("🧠 Training LSTM model (simplified)...")

# For now, use a simple neural network as LSTM placeholder
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(
hidden_layer_sizes=(100, 50),
max_iter=200,
random_state=42,
)

model.fit(X, y)
self.logger.info("✅ LSTM model training completed")
return model

except Exception:
            self.print(error("💥 Error training LSTM model: {e}"))
return None

def _train_random_forest_model(
self,
X: pd.DataFrame,
y: pd.Series,
) -> Any | None:
        """Train Random Forest model."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info("🌲 Training Random Forest model...")

model = RandomForestClassifier(
n_estimators=100,
max_depth=10,
random_state=42,
)

model.fit(X, y)
self.logger.info("✅ Random Forest model training completed")
return model

except Exception:
            self.print(error("💥 Error training Random Forest model: {e}"))
return None

def _prepare_features_target(
self,
data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target from data."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.debug("🔧 Preparing features and target...")

# First, explicitly drop any datetime columns
datetime_columns = data.select_dtypes(
include=["datetime64[ns]", "datetime64", "datetime"],
).columns.tolist()
if datetime_columns:
                self.logger.info(f"Dropping datetime columns: {datetime_columns}")
data = data.drop(columns=datetime_columns)

# Also drop any object columns that might contain datetime strings
# But preserve target column
target_columns = ["target"]
object_columns = data.select_dtypes(include=["object"]).columns.tolist()
object_columns_to_drop = [
col for col in object_columns if col not in target_columns
]
if object_columns_to_drop:
                self.logger.info(f"Dropping object columns: {object_columns_to_drop}")
data = data.drop(columns=object_columns_to_drop)

# Remove target column and other non-feature columns
excluded_columns = target_columns + ["timestamp"]
feature_columns = [
col for col in data.columns if col not in excluded_columns
]
X = data[feature_columns].copy()

# Additional safety check - ensure all columns are numeric
for col in X.columns:
                if not pd.api.types.is_numeric_dtype(X[col]):
                    self.logger.warning(
f"Non-numeric column detected: {col} with dtype {X[col].dtype}",
)
X = X.drop(columns=[col])
feature_columns.remove(col)

# Handle missing values
missing_before = X.isnull().sum().sum()
X = X.fillna(0)
X.isnull().sum().sum()

if missing_before > 0:
                self.logger.info(f"🔧 Filled {missing_before} missing values")

# Final check - ensure X is purely numeric
if X.select_dtypes(include=[np.number]).shape[1] != X.shape[1]:
                self.print(error("Non-numeric columns still present in feature matrix"))
# Force conversion to numeric, dropping any problematic columns
X = X.select_dtypes(include=[np.number])

# Get target
if "target" in data.columns:
                y = data["target"]
self.logger.info("🎯 Using existing target column")
else:
                # Create synthetic target for demonstration
y = pd.Series(["HOLD"] * len(data), index=data.index)
self.logger.warning(
"⚠️ No target column found, using synthetic HOLD targets",
)

self.logger.debug(f"📊 Features shape: {X.shape}, Target shape: {y.shape}")
return X, y

except Exception:
            self.print(error("💥 Error preparing features/target: {e}"))
return pd.DataFrame(), pd.Series()

def _get_timeframe_predictions(
self,
timeframe: str,
data: pd.DataFrame,
) -> tuple[list[str], list[float]]:
        """Get predictions and confidences for a timeframe."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
if timeframe not in self.timeframe_models:
                self.print(warning("⚠️ No trained model for {timeframe}"))
return [], []

model_info = self.timeframe_models[timeframe]
model = model_info["model"]

X, _ = self._prepare_features_target(data)

if len(X) == 0:
                self.print(warning("⚠️ No valid features for {timeframe}"))
return [], []

# Get predictions
predictions = model.predict(X).tolist()

# Get prediction probabilities for confidence
if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)
confidences = np.max(probas, axis=1).tolist()
self.logger.debug(
f"📊 {timeframe}: {len(predictions)} predictions, "
f"avg confidence: {np.mean(confidences):.3f}",
)
else:
                confidences = [0.5] * len(predictions)
self.logger.warning(
f"⚠️ {timeframe}: Model doesn't support predict_proba, using default confidence",
)

return predictions, confidences

except Exception:
            self.print(error("💥 Error getting predictions for {timeframe}: {e}"))
return [], []

def _train_meta_learner(
self,
timeframe_predictions: dict[str, list[str]],
timeframe_confidences: dict[str, list[float]],
prepared_data: dict[str, pd.DataFrame],
) -> bool:
        """Train meta-learner to combine timeframe predictions."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info("🧠 Training meta-learner for timeframe combination...")
self.logger.info(f"📊 Timeframes: {list(timeframe_predictions.keys())}")

# Prepare meta-learner data
self.logger.info("🔧 Preparing meta-learner data...")
meta_data = self._prepare_meta_learner_data(
timeframe_predictions,
timeframe_confidences,
prepared_data,
)

if len(meta_data) == 0:
                self.print(error("❌ No valid meta-learner data"))
return False

self.logger.info(f"📊 Meta-learner data shape: {meta_data.shape}")

# Prepare features and target
X_meta = meta_data.drop(["target", "timestamp"], axis=1, errors="ignore")
y_meta = meta_data["target"]

self.logger.info(f"📊 Meta features shape: {X_meta.shape}")
self.logger.info(
f"🎯 Meta target distribution: {y_meta.value_counts().to_dict()}",
)

# Encode target
self.logger.info("🔧 Encoding target labels...")
self.meta_label_encoder = LabelEncoder()
y_encoded = self.meta_label_encoder.fit_transform(y_meta)

# Scale features
self.logger.info("🔧 Scaling features...")
self.meta_scaler = StandardScaler()
X_scaled = self.meta_scaler.fit_transform(X_meta)

# Train meta-learner
self.logger.info("🌳 Training LightGBM meta-learner...")
self.meta_learner = lgb.LGBMClassifier(
n_estimators=50,
learning_rate=0.1,
max_depth=4,
random_state=42,
verbose=-1,
)

self.meta_learner.fit(X_scaled, y_encoded)

self.logger.info("✅ Meta-learner trained successfully")
self.logger.info(
f"📊 Meta-learner feature importance: {self.meta_learner.feature_importances_[:5]}...",
)

return True

except Exception:
            self.print(error("💥 Error training meta-learner: {e}"))
return False

def _prepare_meta_learner_data(
self,
timeframe_predictions: dict[str, list[str]],
timeframe_confidences: dict[str, list[float]],
prepared_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
        """Prepare data for meta-learner training."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.debug("🔧 Preparing meta-learner data...")

# Find common timestamps across all timeframes
all_timestamps = set()
for timeframe in timeframe_predictions:
                if timeframe in prepared_data:
                    all_timestamps.update(prepared_data[timeframe].index)

self.logger.info(f"📊 Found {len(all_timestamps)} common timestamps")

# Create meta-learner DataFrame
meta_data = []

for timestamp in sorted(all_timestamps):
                row_data = {"timestamp": timestamp}

# Add predictions and confidences from each timeframe
for timeframe in self.active_timeframes:
                    if timeframe in timeframe_predictions:
                        # Find prediction for this timestamp
pred_idx = 0  # Simplified - in practice, match by timestamp
if pred_idx < len(timeframe_predictions[timeframe]):
                            row_data[f"{timeframe}_prediction"] = timeframe_predictions[
timeframe
][pred_idx]
row_data[f"{timeframe}_confidence"] = timeframe_confidences[
timeframe
][pred_idx]
else:
                            row_data[f"{timeframe}_prediction"] = "HOLD"
row_data[f"{timeframe}_confidence"] = 0.0
else:
                        row_data[f"{timeframe}_prediction"] = "HOLD"
row_data[f"{timeframe}_confidence"] = 0.0

# Add target (simplified)
row_data["target"] = "HOLD"  # In practice, use actual target

meta_data.append(row_data)

result_df = pd.DataFrame(meta_data)
self.logger.info(f"📊 Meta-learner data prepared: {result_df.shape}")
return result_df

except Exception:
            self.print(error("💥 Error preparing meta-learner data: {e}"))
return pd.DataFrame()

def get_prediction(
self,
current_features: pd.DataFrame,
**kwargs,
) -> dict[str, Any]:
        """
Get prediction from multi-timeframe ensemble.

Args:
            current_features: Current market features
**kwargs: Additional arguments

Returns:
            Dict with prediction, confidence, and timeframe details
"""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
if not self.trained:
                self.print(warning("⚠️ Multi-timeframe ensemble not trained"))
return {"prediction": "HOLD", "confidence": 0.0}

self.logger.debug(
f"🔮 Getting prediction for {self.model_name} in {self.regime}",
)

# Get predictions from all timeframe models
timeframe_predictions = {}
timeframe_confidences = {}

for timeframe in self.active_timeframes:
                if timeframe in self.timeframe_models:
                    self.logger.debug(f"📊 Getting prediction for {timeframe}...")
pred, conf = self._get_single_prediction(
timeframe,
current_features,
)
timeframe_predictions[timeframe] = pred
timeframe_confidences[timeframe] = conf

self.logger.debug(
f"📊 {timeframe}: {pred} (confidence: {conf:.3f})",
)

# Use meta-learner to combine predictions
if self.meta_learner and len(timeframe_predictions) > 0:
                self.logger.debug("🧠 Combining predictions with meta-learner...")
final_prediction, final_confidence = self._combine_with_meta_learner(
timeframe_predictions,
timeframe_confidences,
current_features,
)
else:
                self.logger.warning(
"⚠️ Using simple prediction combination (no meta-learner)",
)
# Fallback to simple averaging
final_prediction, final_confidence = self._simple_combine_predictions(
timeframe_predictions,
timeframe_confidences,
)

self.logger.info(
f"🎯 Final prediction: {final_prediction} (confidence: {final_confidence:.3f})",
)

return {
"prediction": final_prediction,
"confidence": final_confidence,
"timeframe_predictions": timeframe_predictions,
"timeframe_confidences": timeframe_confidences,
"model_name": self.model_name,
"regime": self.regime,
}

except Exception:
            self.print(error("💥 Error getting prediction: {e}"))
return {"prediction": "HOLD", "confidence": 0.0}

def _get_single_prediction(
self,
timeframe: str,
features: pd.DataFrame,
) -> tuple[str, float]:
        """Get prediction from single timeframe model."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
if timeframe not in self.timeframe_models:
                self.print(warning("⚠️ No trained model for {timeframe}"))
return "HOLD", 0.0

model_info = self.timeframe_models[timeframe]
model = model_info["model"]

# Prepare features
X, _ = self._prepare_features_target(features)

if len(X) == 0:
                self.print(warning("⚠️ No valid features for {timeframe}"))
return "HOLD", 0.0

# Get prediction
prediction = model.predict(X)[0]

# Get confidence
if hasattr(model, "predict_proba"):
                probas = model.predict_proba(X)
confidence = np.max(probas[0])
else:
                confidence = 0.5

return prediction, confidence

except Exception:
            self.print(error("💥 Error getting prediction for {timeframe}: {e}"))
return "HOLD", 0.0

def _combine_with_meta_learner(
self,
timeframe_predictions: dict[str, str],
timeframe_confidences: dict[str, float],
current_features: pd.DataFrame,
) -> tuple[str, float]:
        """Combine predictions using meta-learner."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.debug("🧠 Combining predictions with meta-learner...")

# Prepare meta-features
meta_features = []
for timeframe in self.active_timeframes:
                pred = timeframe_predictions.get(timeframe, "HOLD")
conf = timeframe_confidences.get(timeframe, 0.0)

# One-hot encode prediction
pred_encoded = [
1.0 if pred == "BUY" else 0.0,
1.0 if pred == "SELL" else 0.0,
1.0 if pred == "HOLD" else 0.0,
]
meta_features.extend(pred_encoded)
meta_features.append(conf)

self.logger.debug(f"📊 Meta-features: {meta_features}")

# Scale features
meta_features_scaled = self.meta_scaler.transform([meta_features])

# Get prediction
prediction_encoded = self.meta_learner.predict(meta_features_scaled)[0]
prediction = self.meta_label_encoder.inverse_transform(
[prediction_encoded],
)[0]

# Get confidence
if hasattr(self.meta_learner, "predict_proba"):
                probas = self.meta_learner.predict_proba(meta_features_scaled)
confidence = np.max(probas[0])
else:
                confidence = 0.5

self.logger.debug(
f"🎯 Meta-learner prediction: {prediction} (confidence: {confidence:.3f})",
)
return prediction, confidence

except Exception:
            self.print(error("💥 Error combining with meta-learner: {e}"))
return "HOLD", 0.0

def _simple_combine_predictions(
self,
timeframe_predictions: dict[str, str],
timeframe_confidences: dict[str, float],
) -> tuple[str, float]:
        """Simple combination of predictions (fallback)."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
if not timeframe_predictions:
                self.print(warning("⚠️ No timeframe predictions available"))
return "HOLD", 0.0

# Count predictions
pred_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
total_confidence = 0.0

for pred, conf in zip(
timeframe_predictions.values(),
timeframe_confidences.values(),
strict=False,
):
                pred_counts[pred] += 1
total_confidence += conf

# Get most common prediction
final_prediction = max(pred_counts, key=pred_counts.get)

# Average confidence
final_confidence = (
total_confidence / len(timeframe_confidences)
if timeframe_confidences
else 0.0
)

self.logger.debug(
f"📊 Simple combination: {pred_counts}, final: {final_prediction} (confidence: {final_confidence:.3f})",
)
return final_prediction, final_confidence

except Exception:
            self.print(error("💥 Error in simple prediction combination: {e}"))
return "HOLD", 0.0

def save_model(self, path: str) -> bool:
        """Save multi-timeframe ensemble model."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info(f"💾 Saving multi-timeframe ensemble to {path}")
os.makedirs(path, exist_ok=True)

# Save timeframe models
for timeframe, model_info in self.timeframe_models.items():
                model_path = os.path.join(path, f"{timeframe}_model.joblib")
joblib.dump(model_info["model"], model_path)
self.logger.debug(f"💾 Saved {timeframe} model")

# Save meta-learner
if self.meta_learner:
                meta_path = os.path.join(path, "meta_learner.joblib")
joblib.dump(self.meta_learner, meta_path)

scaler_path = os.path.join(path, "meta_scaler.joblib")
joblib.dump(self.meta_scaler, scaler_path)

encoder_path = os.path.join(path, "meta_encoder.joblib")
joblib.dump(self.meta_label_encoder, encoder_path)

self.logger.debug("💾 Saved meta-learner components")

# Save ensemble info
info_path = os.path.join(path, "ensemble_info.joblib")
ensemble_info = {
"model_name": self.model_name,
"regime": self.regime,
"active_timeframes": self.active_timeframes,
"trained": self.trained,
"trained_at": datetime.now(),
}
joblib.dump(ensemble_info, info_path)

self.logger.info("✅ Multi-timeframe ensemble saved successfully")
return True

except Exception:
            self.print(error("💥 Error saving model: {e}"))
return False

def load_model(self, path: str) -> bool:
        """Load multi-timeframe ensemble model."""
try:
    pass  # TODO: Add proper exception handling
except Exception as e:
    pass  # TODO: Add proper exception handling
self.logger.info(f"📂 Loading multi-timeframe ensemble from {path}")

# Load ensemble info
info_path = os.path.join(path, "ensemble_info.joblib")
if os.path.exists(info_path):
                ensemble_info = joblib.load(info_path)
self.model_name = ensemble_info["model_name"]
self.regime = ensemble_info["regime"]
self.active_timeframes = ensemble_info["active_timeframes"]
self.trained = ensemble_info["trained"]
self.logger.info(
f"📊 Loaded ensemble info: {self.model_name} in {self.regime}",
)

# Load timeframe models
for timeframe in self.active_timeframes:
                model_path = os.path.join(path, f"{timeframe}_model.joblib")
if os.path.exists(model_path):
                    cached_model = joblib.load(model_path)
self.timeframe_models[timeframe] = {
"model": cached_model,
"model_type": "loaded",
"timeframe": timeframe,
"loaded_at": datetime.now(),
}
self.logger.debug(f"📂 Loaded {timeframe} model")
else:
                    self.print(warning("⚠️ No model file found for {timeframe}"))

# Load meta-learner
meta_path = os.path.join(path, "meta_learner.joblib")
if os.path.exists(meta_path):
                self.meta_learner = joblib.load(meta_path)

scaler_path = os.path.join(path, "meta_scaler.joblib")
if os.path.exists(scaler_path):
                    self.meta_scaler = joblib.load(scaler_path)

encoder_path = os.path.join(path, "meta_encoder.joblib")
if os.path.exists(encoder_path):
                    self.meta_label_encoder = joblib.load(encoder_path)

self.logger.debug("📂 Loaded meta-learner components")
else:
                self.print(warning("⚠️ No meta-learner found"))

self.logger.info("✅ Multi-timeframe ensemble loaded successfully")
return True

except Exception:
            self.print(error("💥 Error loading model: {e}"))
return False
