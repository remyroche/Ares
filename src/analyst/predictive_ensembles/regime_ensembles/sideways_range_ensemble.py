# src/analyst/predictive_ensembles/regime_ensembles/sideways_range_ensemble.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans # For conceptual clustering in sideways
from lightgbm import LGBMClassifier
import pandas_ta as ta

# TensorFlow/Keras imports for Deep Learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2 # For L1/L2 regularization
# For TabNet, we'll use Dense layers as a proxy
# from pytorch_tabnet.tab_model import TabNetClassifier # Requires installation

from .base_ensemble import BaseEnsemble

class SidewaysRangeEnsemble(BaseEnsemble):
    """
    Predictive Ensemble specifically designed for SIDEWAYS_RANGE market regimes.
    Focuses on mean reversion, breakout detection, and order flow.
    """
    def __init__(self, config: dict, ensemble_name: str):
        super().__init__(config, ensemble_name)
        self.models = {
            "clustering": None, # Conceptual clustering model
            "bb_squeeze": None, # Bollinger Band Squeeze model (rule-based or simple ML)
            "order_flow": None, # Order flow analysis model (conceptual)
            "tabnet_proxy": None, # TabNet-like model using Dense layers
            "lgbm": None # LightGBM for general patterns
        }
        self.ensemble_weights = self.config.get("ensemble_weights", {
            "clustering": 0.2, "bb_squeeze": 0.2, "order_flow": 0.2, "tabnet_proxy": 0.2, "lgbm": 0.2
        })
        self.min_confluence_confidence = self.config.get("min_confluence_confidence", 0.7)

        # DL model specific configurations
        self.dl_config = {
            "dense_units": 64, # Units for TabNet-like dense layers
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "l1_reg_strength": self.config.get("meta_learner_l1_reg", 0.001),
            "l2_reg_strength": self.config.get("meta_learner_l2_reg", 0.001),
        }
        self.expected_dl_features = ['close', 'volume', 'ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']
        self.dl_target_map = {} # To store target label to integer mapping for DL models

    def _prepare_flat_data(self, df: pd.DataFrame, target_series: pd.Series = None):
        """
        Prepares flat data for models.
        Ensures consistent feature columns and handles missing data.
        """
        features = df[self.expected_dl_features].copy()
        features = features.fillna(0) # Fill NaNs

        if target_series is not None:
            aligned_df = features.join(target_series.rename('target')).dropna()
            X = aligned_df.drop(columns=['target'])
            y = aligned_df['target']
            return X, y
        else:
            return features, None

    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        """
        Trains the individual models and the meta-learner for the SIDEWAYS_RANGE ensemble.
        `historical_targets` could be 'MEAN_REVERT', 'BREAKOUT_UP', 'BREAKOUT_DOWN'.
        """
        self.logger.info(f"Training {self.ensemble_name} ensemble models...")

        if historical_features.empty or historical_targets.empty:
            self.logger.warning(f"No data to train {self.ensemble_name}. Skipping training.")
            return

        X_flat, y_flat = self._prepare_flat_data(historical_features, historical_targets)

        if X_flat.empty:
            self.logger.warning(f"Insufficient data after preparation for {self.ensemble_name}. Skipping model training.")
            return

        unique_targets = np.unique(y_flat)
        self.dl_target_map = {label: i for i, label in enumerate(unique_targets)}
        y_flat_encoded = np.array([self.dl_target_map[label] for label in y_flat])

        # --- Train Individual Models ---

        # Clustering Model (Conceptual Training)
        self.logger.info("Training Clustering model (Conceptual)...")
        self.models["clustering"] = True # Simulate trained model

        # BB Squeeze Model (Rule-based or simple ML)
        self.logger.info("Training BB Squeeze model (Conceptual/Rule-based)...")
        self.models["bb_squeeze"] = True # Simulate trained model

        # Order Flow Model (Conceptual Training)
        self.logger.info("Training Order Flow model (Conceptual)...")
        self.models["order_flow"] = True # Simulate trained model

        # TabNet-like Model (using Dense layers as a proxy)
        self.logger.info("Training TabNet-like model...")
        try:
            model = Sequential([
                Input(shape=(X_flat.shape[1],)),
                Dense(self.dl_config["dense_units"], activation='relu',
                      kernel_regularizer=l1_l2(l1=self.dl_config["l1_reg_strength"], l2=self.dl_config["l2_reg_strength"])),
                Dropout(self.dl_config["dropout_rate"]),
                Dense(self.dl_config["dense_units"] // 2, activation='relu',
                      kernel_regularizer=l1_l2(l1=self.dl_config["l1_reg_strength"], l2=self.dl_config["l2_reg_strength"])),
                Dropout(self.dl_config["dropout_rate"]),
                Dense(len(unique_targets), activation='softmax',
                      kernel_regularizer=l1_l2(l1=self.dl_config["l1_reg_strength"], l2=self.dl_config["l2_reg_strength"]))
            ])
            model.compile(optimizer=Adam(learning_rate=self.dl_config["learning_rate"]), loss='sparse_categorical_crossentropy' if len(unique_targets) > 2 else 'binary_crossentropy', metrics=['accuracy'])
            model.fit(X_flat, y_flat_encoded, epochs=self.dl_config["epochs"], batch_size=self.dl_config["batch_size"], verbose=0)
            self.models["tabnet_proxy"] = model
            self.logger.info("TabNet-like model trained.")
        except Exception as e:
            self.logger.error(f"Error training TabNet-like model: {e}")
            self.models["tabnet_proxy"] = None

        # LightGBM Model
        self.logger.info("Training LightGBM model...")
        try:
            self.models["lgbm"] = LGBMClassifier(
                random_state=42, 
                verbose=-1,
                reg_alpha=self.dl_config["l1_reg_strength"],
                reg_lambda=self.dl_config["l2_reg_strength"]
            )
            self.models["lgbm"].fit(X_flat, y_flat)
            self.logger.info("LightGBM model trained.")
        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {e}")
            self.models["lgbm"] = None

        # --- Train Meta-Learner ---
        meta_features_train = pd.DataFrame(index=X_flat.index)
        
        # Populate meta_features_train with actual (or simulated) predictions/confidences
        meta_features_train['clustering_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        meta_features_train['bb_squeeze_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        meta_features_train['order_flow_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        
        if self.models["tabnet_proxy"]:
            tabnet_probas = self.models["tabnet_proxy"].predict(X_flat)
            tabnet_conf_values = [tabnet_probas[i, y_flat_encoded[i]] for i in range(len(y_flat_encoded))]
            meta_features_train['tabnet_proxy_proba'] = pd.Series(tabnet_conf_values, index=X_flat.index)
        else:
            meta_features_train['tabnet_proxy_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))

        if self.models["lgbm"]:
            lgbm_probas = self.models["lgbm"].predict_proba(X_flat)
            lgbm_conf_values = [lgbm_probas[i, self.models["lgbm"].classes_.tolist().index(y_flat.iloc[i])] for i in range(len(y_flat))]
            meta_features_train['lgbm_proba'] = pd.Series(lgbm_conf_values, index=X_flat.index)
        else:
            meta_features_train['lgbm_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))

        # Add advanced features to meta_features_train
        meta_features_train = meta_features_train.join(historical_features[['is_wyckoff_sos', 'is_wyckoff_sow', 'is_wyckoff_spring', 'is_wyckoff_upthrust',
                                                                      'is_accumulation_phase', 'is_distribution_phase',
                                                                      'is_liquidity_sweep', 'is_fake_breakout', 'is_large_bid_wall_near', 'is_large_ask_wall_near',
                                                                      'cvd_value', 'cvd_divergence_score', 'is_absorption', 'is_exhaustion',
                                                                      'aggressive_buy_volume', 'aggressive_sell_volume',
                                                                      'htf_trend_bullish', 'mtf_trend_bullish']], how='inner').fillna(0)

        meta_features_train = meta_features_train.loc[y_flat.index].dropna()
        y_meta_train = y_flat.loc[meta_features_train.index]

        if meta_features_train.empty:
            self.logger.warning("Meta-features training set is empty. Cannot train meta-learner.")
            self.trained = False
            return

        self.meta_learner_features = meta_features_train.columns.tolist()

        self._train_meta_learner(meta_features_train, y_meta_train)
        self.trained = True

    def get_prediction(self, current_features: pd.DataFrame, 
                       klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, 
                       order_book_data: dict, current_price: float) -> dict:
        """
        Gets a combined prediction and confidence score for the SIDEWAYS_RANGE regime.
        """
        if not self.trained:
            self.logger.warning(f"{self.ensemble_name} not trained. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}

        self.logger.info(f"Getting prediction from {self.ensemble_name}...")

        current_features_row = current_features.tail(1)
        if current_features_row.empty:
            self.logger.warning("Current features row is empty. Cannot get prediction.")
            return {"prediction": "HOLD", "confidence": 0.0}

        X_current_flat, _ = self._prepare_flat_data(current_features_row)
        if X_current_flat.empty:
            self.logger.warning("Insufficient data for real-time flat prediction. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}

        individual_model_outputs = {}

        # Clustering Prediction (Conceptual)
        individual_model_outputs['clustering_conf'] = np.random.uniform(0.5, 0.9)

        # BB Squeeze Prediction (Conceptual/Rule-based)
        bb_length = self.config.get("bollinger_bands", {}).get("window", 20)
        bb_std_dev = self.config.get("bollinger_bands", {}).get("num_std_dev", 2)
        bb_squeeze_threshold = self.config.get("bband_squeeze_threshold", 0.01)

        if len(klines_df) >= bb_length:
            bb = klines_df.ta.bbands(length=bb_length, std=bb_std_dev, append=False)
            bb_width = (bb[f'BBU_{bb_length}_{bb_std_dev:.1f}'].iloc[-1] - bb[f'BBL_{bb_length}_{bb_std_dev:.1f}'].iloc[-1])
            bb_width_norm = bb_width / klines_df['close'].iloc[-1] if klines_df['close'].iloc[-1] > 0 else 0
            
            if bb_width_norm < bb_squeeze_threshold:
                individual_model_outputs['bb_squeeze_conf'] = 0.8
            else:
                individual_model_outputs['bb_squeeze_conf'] = 0.5
        else:
            individual_model_outputs['bb_squeeze_conf'] = 0.5

        # Order Flow Prediction (Conceptual)
        individual_model_outputs['order_flow_conf'] = np.random.uniform(0.5, 0.9)

        # TabNet-like Prediction
        if self.models["tabnet_proxy"]:
            try:
                tabnet_proba = self.models["tabnet_proxy"].predict(X_current_flat)[0]
                individual_model_outputs['tabnet_proxy_proba'] = np.max(tabnet_proba)
            except Exception as e:
                self.logger.error(f"Error predicting with TabNet-like model: {e}")
                individual_model_outputs['tabnet_proxy_proba'] = 0.5
        else:
            individual_model_outputs['tabnet_proxy_proba'] = 0.5

        # LightGBM Prediction
        if self.models["lgbm"]:
            try:
                lgbm_features = X_current_flat.copy()
                lgbm_proba = self.models["lgbm"].predict_proba(lgbm_features)[0]
                individual_model_outputs['lgbm_proba'] = np.max(lgbm_proba)
            except Exception as e:
                self.logger.error(f"Error predicting with LightGBM: {e}")
                individual_model_outputs['lgbm_proba'] = 0.5
        else:
            individual_model_outputs['lgbm_proba'] = 0.5

        # --- Incorporate Advanced Features ---
        wyckoff_feats = self._get_wyckoff_features(klines_df, current_price)
        manipulation_feats = self._get_manipulation_features(order_book_data, current_price, agg_trades_df)
        order_flow_feats = self._get_order_flow_features(agg_trades_df, order_book_data, klines_df)
        multi_timeframe_feats = self._get_multi_timeframe_features(klines_df, klines_df)

        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in wyckoff_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in manipulation_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in order_flow_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in multi_timeframe_feats.items()})

        # --- Confluence Model (Meta-Learner) ---
        final_prediction = "HOLD"
        final_confidence = 0.0

        if self.meta_learner:
            try:
                meta_features_for_pred = {}
                for feature_name in self.meta_learner_features:
                    meta_features_for_pred[feature_name] = individual_model_outputs.get(feature_name, 0.0)

                meta_input_data = pd.DataFrame([meta_features_for_pred])
                final_prediction, final_confidence = self._get_meta_prediction(meta_input_data.iloc[0].to_dict())
            except Exception as e:
                self.logger.error(f"Error during meta-learner prediction for {self.ensemble_name}: {e}", exc_info=True)
                final_prediction = "HOLD"
                final_confidence = 0.0
        else:
            self.logger.warning(f"Meta-learner not available for {self.ensemble_name}. Averaging confidences.")
            total_weighted_confidence = sum(individual_model_outputs.get(m, 0.5) * self.ensemble_weights.get(m.replace('_conf', '').replace('_proba', ''), 1.0) 
                                            for m in ['clustering_conf', 'bb_squeeze_conf', 'order_flow_conf', 'tabnet_proxy_proba', 'lgbm_proba'])
            total_weight = sum(self.ensemble_weights.get(m.replace('_conf', '').replace('_proba', ''), 1.0) for m in ['clustering_conf', 'bb_squeeze_conf', 'order_flow_conf', 'tabnet_proxy_proba', 'lgbm_proba'])
            
            if total_weight > 0:
                final_confidence = total_weighted_confidence / total_weight
            
            if final_confidence > self.min_confluence_confidence:
                final_prediction = np.random.choice(["MEAN_REVERT", "BREAKOUT_UP", "BREAKOUT_DOWN"])
            else:
                final_prediction = "HOLD"

        self.logger.info(f"Ensemble Result for {self.ensemble_name}: Prediction={final_prediction}, Confidence={final_confidence:.2f}")
        return {"prediction": final_prediction, "confidence": final_confidence}

