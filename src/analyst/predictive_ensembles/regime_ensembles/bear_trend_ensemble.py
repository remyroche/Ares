# src/analyst/predictive_ensembles/regime_ensembles/bear_trend_ensemble.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
import pandas_ta as ta
from arch import arch_model
import warnings

# Suppress specific warnings from ARCH library
warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")

from .base_ensemble import BaseEnsemble

class BearTrendEnsemble(BaseEnsemble):
    """
    Predictive Ensemble specifically designed for BEAR_TREND market regimes.
    Combines LSTM, Transformer (conceptual), Statistical (GARCH), and Volume models.
    """
    def __init__(self, config: dict, ensemble_name: str):
        super().__init__(config, ensemble_name)
        self.models = {
            "lstm": None, # Conceptual LSTM model
            "transformer": None, # Conceptual Transformer model
            "garch": None, # GARCH model for volatility/direction
            "lgbm": None # LightGBM for pattern recognition
        }
        self.ensemble_weights = self.config.get("ensemble_weights", {
            "lstm": 0.3, "transformer": 0.3, "garch": 0.2, "lgbm": 0.2
        })
        self.min_confluence_confidence = self.config.get("min_confluence_confidence", 0.7)

    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        """
        Trains the individual models and the meta-learner for the BEAR_TREND ensemble.
        `historical_targets` should represent the desired outcome (e.g., 'CONTINUE_DOWN', 'BOUNCE').
        """
        self.logger.info(f"Training {self.ensemble_name} ensemble models...")

        if historical_features.empty or historical_targets.empty:
            self.logger.warning(f"No data to train {self.ensemble_name}. Skipping training.")
            return

        aligned_data = historical_features.join(historical_targets.rename('target')).dropna()
        if aligned_data.empty:
            self.logger.warning(f"No aligned data after dropping NaNs for {self.ensemble_name}. Skipping training.")
            return
        
        X = aligned_data.drop(columns=['target'])
        y = aligned_data['target']

        # --- Train Individual Models (Conceptual/Simplified) ---
        self.logger.info("Training LSTM model (Conceptual)...")
        self.models["lstm"] = True # Simulate trained model

        self.logger.info("Training Transformer model (Conceptual)...")
        self.models["transformer"] = True # Simulate trained model

        self.logger.info("Training GARCH model...")
        returns = aligned_data['close'].pct_change().dropna()
        if len(returns) > 100:
            try:
                garch_model = arch_model(returns, vol='Garch', p=1, q=1, mean='AR', lags=1, dist='normal', rescale=True)
                self.models["garch"] = garch_model.fit(disp='off')
                self.logger.info("GARCH model trained.")
            except Exception as e:
                self.logger.error(f"Error training GARCH model: {e}")
                self.models["garch"] = None
        else:
            self.logger.warning("Insufficient data for GARCH model training. Skipping.")
            self.models["garch"] = None

        self.logger.info("Training LightGBM model...")
        try:
            self.models["lgbm"] = LGBMClassifier(random_state=42, verbose=-1)
            self.models["lgbm"].fit(X, y)
            self.logger.info("LightGBM model trained.")
        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {e}")
            self.models["lgbm"] = None

        # --- Train Meta-Learner ---
        meta_features_train = pd.DataFrame(index=X.index)
        meta_features_train['lstm_conf'] = np.random.uniform(0.5, 0.9, len(X))
        meta_features_train['transformer_conf'] = np.random.uniform(0.5, 0.9, len(X))
        meta_features_train['garch_volatility'] = np.random.uniform(0.001, 0.01, len(X))
        meta_features_train['lgbm_proba'] = np.random.uniform(0.5, 0.9, len(X))

        self._train_meta_learner(meta_features_train, y)
        self.trained = True

    def get_prediction(self, current_features: pd.DataFrame, 
                       klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, 
                       order_book_data: dict, current_price: float) -> dict:
        """
        Gets a combined prediction and confidence score for the BEAR_TREND regime.
        """
        if not self.trained:
            self.logger.warning(f"{self.ensemble_name} not trained. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}

        self.logger.info(f"Getting prediction from {self.ensemble_name}...")

        current_features_row = current_features.tail(1)
        if current_features_row.empty:
            self.logger.warning("Current features row is empty. Cannot get prediction.")
            return {"prediction": "HOLD", "confidence": 0.0}

        individual_model_outputs = {}

        # LSTM Prediction (Conceptual)
        lstm_conf = np.random.uniform(0.5, 0.9)
        individual_model_outputs['lstm_conf'] = lstm_conf

        # Transformer Prediction (Conceptual)
        transformer_conf = np.random.uniform(0.5, 0.9)
        individual_model_outputs['transformer_conf'] = transformer_conf

        # GARCH Prediction (Volatility/Directional Signal)
        if self.models["garch"]:
            try:
                forecast = self.models["garch"].forecast(horizon=1, method='simulation')
                garch_directional_signal = np.random.choice([-1, 1])
                garch_conf = 1 - forecast.variance.iloc[-1].values[0]
                individual_model_outputs['garch_conf'] = np.clip(garch_conf, 0.1, 0.9)
            except Exception as e:
                self.logger.error(f"Error forecasting with GARCH: {e}")
                individual_model_outputs['garch_conf'] = 0.5
        else:
            individual_model_outputs['garch_conf'] = 0.5

        # LightGBM Prediction
        if self.models["lgbm"]:
            try:
                lgbm_features = current_features_row[['ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']].copy()
                lgbm_features.fillna(0, inplace=True)
                lgbm_proba = self.models["lgbm"].predict_proba(lgbm_features)[0]
                bear_idx = np.where(self.models["lgbm"].classes_ == 'BEAR_TREND')[0]
                lgbm_conf = lgbm_proba[bear_idx][0] if len(bear_idx) > 0 else 0.5
                individual_model_outputs['lgbm_conf'] = lgbm_conf
            except Exception as e:
                self.logger.error(f"Error predicting with LightGBM: {e}")
                individual_model_outputs['lgbm_conf'] = 0.5
        else:
            individual_model_outputs['lgbm_conf'] = 0.5

        # --- Incorporate Advanced Features ---
        wyckoff_feats = self._get_wyckoff_features(klines_df, current_price)
        manipulation_feats = self._get_manipulation_features(order_book_data, current_price)
        order_flow_feats = self._get_order_flow_features(agg_trades_df, order_book_data)
        multi_timeframe_feats = self._get_multi_timeframe_features(klines_df, klines_df)

        # Boost/Penalize confidences (heuristic)
        if wyckoff_feats["is_wyckoff_sow"]:
            individual_model_outputs['lstm_conf'] = min(1.0, individual_model_outputs['lstm_conf'] * 1.1)
        if manipulation_feats["is_large_bid_wall_near"]:
            individual_model_outputs['transformer_conf'] = max(0.0, individual_model_outputs['transformer_conf'] * 0.9)
        if order_flow_feats["cvd_divergence"] > 0 and individual_model_outputs['lgbm_conf'] > 0.5:
             individual_model_outputs['lgbm_conf'] = max(0.0, individual_model_outputs['lgbm_conf'] * 0.8)

        # --- Confluence Model (Meta-Learner) ---
        meta_input_data = pd.DataFrame([individual_model_outputs])
        
        final_prediction = "HOLD"
        final_confidence = 0.0

        if self.meta_learner:
            try:
                meta_features_scaled = self.scaler.transform(meta_input_data)
                meta_learner_output = self._get_meta_prediction(individual_model_outputs)
                if meta_learner_output[0] == "ACTION":
                    final_prediction = "SELL" # For bear trend, ACTION means SELL
                    final_confidence = meta_learner_output[1]
                else:
                    final_prediction = "HOLD"
                    final_confidence = meta_learner_output[1]
            except Exception as e:
                self.logger.error(f"Error during meta-learner prediction for {self.ensemble_name}: {e}")
                final_prediction = "HOLD"
                final_confidence = 0.0
        else:
            self.logger.warning(f"Meta-learner not available for {self.ensemble_name}. Averaging confidences.")
            total_weighted_confidence = sum(individual_model_outputs[m] * self.ensemble_weights.get(m.replace('_conf', ''), 1.0) 
                                            for m in individual_model_outputs if m.endswith('_conf'))
            total_weight = sum(self.ensemble_weights.get(m.replace('_conf', ''), 1.0) for m in individual_model_outputs if m.endswith('_conf'))
            
            if total_weight > 0:
                final_confidence = total_weighted_confidence / total_weight
            
            if final_confidence > self.min_confluence_confidence:
                final_prediction = "SELL"
            else:
                final_prediction = "HOLD"

        self.logger.info(f"Ensemble Result for {self.ensemble_name}: Prediction={final_prediction}, Confidence={final_confidence:.2f}")
        return {"prediction": final_prediction, "confidence": final_confidence}

