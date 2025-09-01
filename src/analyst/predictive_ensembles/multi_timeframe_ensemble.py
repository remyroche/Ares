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

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="multitimeframeensemble initialization",
    )
    async def initialize(self) -> bool:
        """Initialize MultiTimeframeEnsemble."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
    pass"""
Multi-timeframe ensemble that integrates into existing ensemble system.

Each individual model (XGBoost, LSTM, etc.) becomes a multi-timeframe ensemble.
"""

def __init__(...):
    passself.model_name = model_name
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

def train_multi_timeframe_ensemble(...) -> ...:
    """..."""
    passstart_time = time.time()

try:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
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
    passself.logger.info(
f"🔄 [{i}/{len(self.active_timeframes)}] Training {timeframe} timeframe...",
)

if timeframe not in prepared_data:
    passself.logger.warning(
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
    passpassself.logger.info(
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
    passself.print(failed("❌ {timeframe} training failed"))
training_stats[timeframe] = {
"training_time": tf_training_time,
"success": False,
}

# 2. Train meta-learner to combine timeframe predictions
if len(timeframe_predictions) > 1:
    passself.logger.info(
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
    passpassself.trained = True
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
    passif stats.get("success"):
    passself.logger.info(
f"   - {tf}: {stats['training_time']:.2f}s, "
f"{stats['predictions_count']} predictions, "
f"avg confidence: {stats['avg_confidence']:.3f}",
)
else:
    passself.print(failed("   - {tf}: FAILED"))

return True
self.print(failed("❌ Meta-learner training failed"))
return False
self.logger.error(
f"❌ Insufficient timeframes ({len(timeframe_predictions)}) for meta-learner training",
)
return False

except Exception:
    passpasspassself.print(error("💥 Error in multi-timeframe ensemble training: {e}"))
return False

def _train_single_timeframe(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info(f"🔧 Training {model_type} model for {timeframe}")
self.logger.info(f"📊 Data shape: {data.shape}")
self.logger.info(f"📈 Data columns: {list(data.columns)}")

# Prepare features and target
X, y = self._prepare_features_target(data)

if len(X) == 0:
    passself.print(warning("⚠️ No valid data for {timeframe}"))
return False

self.logger.info(f"📊 Features shape: {X.shape}")
self.logger.info(f"🎯 Target distribution: {y.value_counts().to_dict()}")

# Train model based on type
if model_type == "xgboost":
    passmodel = self._train_xgboost_model(X, y)
elif model_type == "lstm":
    passpassmodel = self._train_lstm_model(X, y)
elif model_type == "random_forest":
    passpassmodel = self._train_random_forest_model(X, y)
else:
    passself.print(error("❌ Unknown model type: {model_type}"))
return False

if model is not None:
    passself.timeframe_models[timeframe] = {
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
    passpassself.print(error("💥 Error training {timeframe} model: {e}"))
return False

def _train_xgboost_model(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
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
    passself.logger.info(
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
    passpassself.print(error("💥 Error training XGBoost model: {e}"))
return None

def _train_lstm_model(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
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
    passpassself.print(error("💥 Error training LSTM model: {e}"))
return None

def _train_random_forest_model(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
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
    passpassself.print(error("💥 Error training Random Forest model: {e}"))
return None

def _prepare_features_target(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.debug("🔧 Preparing features and target...")

# First, explicitly drop any datetime columns
datetime_columns = data.select_dtypes(
include=["datetime64[ns]", "datetime64", "datetime"],
).columns.tolist()
if datetime_columns:
    passself.logger.info(f"Dropping datetime columns: {datetime_columns}")
data = data.drop(columns=datetime_columns)

# Also drop any object columns that might contain datetime strings
# But preserve target column
target_columns = ["target"]
object_columns = data.select_dtypes(include=["object"]).columns.tolist()
object_columns_to_drop = [
col for col in object_columns if col not in target_columns
]
if object_columns_to_drop:
    passpassself.logger.info(f"Dropping object columns: {object_columns_to_drop}")
data = data.drop(columns=object_columns_to_drop)

# Remove target column and other non-feature columns
excluded_columns = target_columns + ["timestamp"]
feature_columns = [
col for col in data.columns if col not in excluded_columns
]
X = data[feature_columns].copy()

# Additional safety check - ensure all columns are numeric
for col in X.columns:
    passpassif not pd.api.types.is_numeric_dtype(X[col]):
    passself.logger.warning(
f"Non-numeric column detected: {col} with dtype {X[col].dtype}",
)
X = X.drop(columns=[col])
feature_columns.remove(col)

# Handle missing values
missing_before = X.isnull().sum().sum()
X = X.fillna(0)
X.isnull().sum().sum()

if missing_before > 0:
    passpassself.logger.info(f"🔧 Filled {missing_before} missing values")

# Final check - ensure X is purely numeric
if X.select_dtypes(include=[np.number]).shape[1] != X.shape[1]:
    passself.print(error("Non-numeric columns still present in feature matrix"))
# Force conversion to numeric, dropping any problematic columns
X = X.select_dtypes(include=[np.number])

# Get target
if "target" in data.columns:
    passy = data["target"]
self.logger.info("🎯 Using existing target column")
else:
    pass# Create synthetic target for demonstration
y = pd.Series(["HOLD"] * len(data), index=data.index)
self.logger.warning(
"⚠️ No target column found, using synthetic HOLD targets",
)

self.logger.debug(f"📊 Features shape: {X.shape}, Target shape: {y.shape}")
return X, y

except Exception:
    passpassself.print(error("💥 Error preparing features/target: {e}"))
return pd.DataFrame(), pd.Series()

def _get_timeframe_predictions(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if timeframe not in self.timeframe_models:
    passself.print(warning("⚠️ No trained model for {timeframe}"))
return [], []

model_info = self.timeframe_models[timeframe]
model = model_info["model"]

X, _ = self._prepare_features_target(data)

if len(X) == 0:
    passpassself.print(warning("⚠️ No valid features for {timeframe}"))
return [], []

# Get predictions
predictions = model.predict(X).tolist()

# Get prediction probabilities for confidence
if hasattr(model, "predict_proba"):
    passpassprobas = model.predict_proba(X)
confidences = np.max(probas, axis=1).tolist()
self.logger.debug(
f"📊 {timeframe}: {len(predictions)} predictions, "
f"avg confidence: {np.mean(confidences):.3f}",
)
else:
    passconfidences = [0.5] * len(predictions)
self.logger.warning(
f"⚠️ {timeframe}: Model doesn't support predict_proba, using default confidence",
)

return predictions, confidences

except Exception:
    passpassself.print(error("💥 Error getting predictions for {timeframe}: {e}"))
return [], []

def _train_meta_learner(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
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
    passself.print(error("❌ No valid meta-learner data"))
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
    passpassself.print(error("💥 Error training meta-learner: {e}"))
return False

def _prepare_meta_learner_data(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.debug("🔧 Preparing meta-learner data...")

# Find common timestamps across all timeframes
all_timestamps = set()
for timeframe in timeframe_predictions:
    passif timeframe in prepared_data:
    passall_timestamps.update(prepared_data[timeframe].index)

self.logger.info(f"📊 Found {len(all_timestamps)} common timestamps")

# Create meta-learner DataFrame
meta_data = []

for timestamp in sorted(all_timestamps):
    passrow_data = {"timestamp": timestamp}

# Add predictions and confidences from each timeframe
for timeframe in self.active_timeframes:
    passif timeframe in timeframe_predictions:
    pass# Find prediction for this timestamp
pred_idx = 0  # Simplified - in practice, match by timestamp
if pred_idx < len(timeframe_predictions[timeframe]):
    passpassrow_data[f"{timeframe}_prediction"] = timeframe_predictions[
timeframe
][pred_idx]
row_data[f"{timeframe}_confidence"] = timeframe_confidences[
timeframe
][pred_idx]
else:
    passrow_data[f"{timeframe}_prediction"] = "HOLD"
row_data[f"{timeframe}_confidence"] = 0.0
else:
    passrow_data[f"{timeframe}_prediction"] = "HOLD"
row_data[f"{timeframe}_confidence"] = 0.0

# Add target (simplified)
row_data["target"] = "HOLD"  # In practice, use actual target

meta_data.append(row_data)

result_df = pd.DataFrame(meta_data)
self.logger.info(f"📊 Meta-learner data prepared: {result_df.shape}")
return result_df

except Exception:
    passpassself.print(error("💥 Error preparing meta-learner data: {e}"))
return pd.DataFrame()

def get_prediction(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if not self.trained:
    passself.print(warning("⚠️ Multi-timeframe ensemble not trained"))
return {"prediction": "HOLD", "confidence": 0.0}

self.logger.debug(
f"🔮 Getting prediction for {self.model_name} in {self.regime}",
)

# Get predictions from all timeframe models
timeframe_predictions = {}
timeframe_confidences = {}

for timeframe in self.active_timeframes:
    passif timeframe in self.timeframe_models:
    passself.logger.debug(f"📊 Getting prediction for {timeframe}...")
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
    passself.logger.debug("🧠 Combining predictions with meta-learner...")
final_prediction, final_confidence = self._combine_with_meta_learner(
timeframe_predictions,
timeframe_confidences,
current_features,
)
else:
    passpassself.logger.warning(
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
    passpassself.print(error("💥 Error getting prediction: {e}"))
return {"prediction": "HOLD", "confidence": 0.0}

def _get_single_prediction(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if timeframe not in self.timeframe_models:
    passself.print(warning("⚠️ No trained model for {timeframe}"))
return "HOLD", 0.0

model_info = self.timeframe_models[timeframe]
model = model_info["model"]

# Prepare features
X, _ = self._prepare_features_target(features)

if len(X) == 0:
    passpassself.print(warning("⚠️ No valid features for {timeframe}"))
return "HOLD", 0.0

# Get prediction
prediction = model.predict(X)[0]

# Get confidence
if hasattr(model, "predict_proba"):
    passpassprobas = model.predict_proba(X)
confidence = np.max(probas[0])
else:
    passconfidence = 0.5

return prediction, confidence

except Exception:
    passpassself.print(error("💥 Error getting prediction for {timeframe}: {e}"))
return "HOLD", 0.0

def _combine_with_meta_learner(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.debug("🧠 Combining predictions with meta-learner...")

# Prepare meta-features
meta_features = []
for timeframe in self.active_timeframes:
    passpasspred = timeframe_predictions.get(timeframe, "HOLD")
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
    passprobas = self.meta_learner.predict_proba(meta_features_scaled)
confidence = np.max(probas[0])
else:
    passconfidence = 0.5

self.logger.debug(
f"🎯 Meta-learner prediction: {prediction} (confidence: {confidence:.3f})",
)
return prediction, confidence

except Exception:
    passpassself.print(error("💥 Error combining with meta-learner: {e}"))
return "HOLD", 0.0

def _simple_combine_predictions(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
if not timeframe_predictions:
    passself.print(warning("⚠️ No timeframe predictions available"))
return "HOLD", 0.0

# Count predictions
pred_counts = {"BUY": 0, "SELL": 0, "HOLD": 0}
total_confidence = 0.0

for pred, conf in zip(
timeframe_predictions.values(),
timeframe_confidences.values(),
strict=False,
):
    passpred_counts[pred] += 1
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
    passpassself.print(error("💥 Error in simple prediction combination: {e}"))
return "HOLD", 0.0

def save_model(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info(f"💾 Saving multi-timeframe ensemble to {path}")
os.makedirs(path, exist_ok=True)

# Save timeframe models
for timeframe, model_info in self.timeframe_models.items():
    passmodel_path = os.path.join(path, f"{timeframe}_model.joblib")
joblib.dump(model_info["model"], model_path)
self.logger.debug(f"💾 Saved {timeframe} model")

# Save meta-learner
if self.meta_learner:
    passmeta_path = os.path.join(path, "meta_learner.joblib")
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
    passpassself.print(error("💥 Error saving model: {e}"))
return False

def load_model(...) -> ...:
    """..."""
    passtry:
    pass# Exception handling placeholder - implement specific error handling as needed
except Exception as e:
    passpasspasspasspasspasspass# Exception handling placeholder - implement specific error handling as needed
self.logger.info(f"📂 Loading multi-timeframe ensemble from {path}")

# Load ensemble info
info_path = os.path.join(path, "ensemble_info.joblib")
if os.path.exists(info_path):
    passensemble_info = joblib.load(info_path)
self.model_name = ensemble_info["model_name"]
self.regime = ensemble_info["regime"]
self.active_timeframes = ensemble_info["active_timeframes"]
self.trained = ensemble_info["trained"]
self.logger.info(
f"📊 Loaded ensemble info: {self.model_name} in {self.regime}",
)

# Load timeframe models
for timeframe in self.active_timeframes:
    passmodel_path = os.path.join(path, f"{timeframe}_model.joblib")
if os.path.exists(model_path):
    passcached_model = joblib.load(model_path)
self.timeframe_models[timeframe] = {
"model": cached_model,
"model_type": "loaded",
"timeframe": timeframe,
"loaded_at": datetime.now(),
}
self.logger.debug(f"📂 Loaded {timeframe} model")
else:
    passself.print(warning("⚠️ No model file found for {timeframe}"))

# Load meta-learner
meta_path = os.path.join(path, "meta_learner.joblib")
if os.path.exists(meta_path):
    passpassself.meta_learner = joblib.load(meta_path)

scaler_path = os.path.join(path, "meta_scaler.joblib")
if os.path.exists(scaler_path):
    passself.meta_scaler = joblib.load(scaler_path)

encoder_path = os.path.join(path, "meta_encoder.joblib")
if os.path.exists(encoder_path):
    passself.meta_label_encoder = joblib.load(encoder_path)

self.logger.debug("📂 Loaded meta-learner components")
else:
    passself.print(warning("⚠️ No meta-learner found"))

self.logger.info("✅ Multi-timeframe ensemble loaded successfully")
return True

except Exception:
    passpassself.print(error("💥 Error loading model: {e}"))
return False
