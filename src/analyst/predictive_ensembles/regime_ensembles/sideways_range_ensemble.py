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

    def _get_clustering_features(self, klines_df: pd.DataFrame) -> dict:
        """
        Conceptual clustering features for sideways markets.
        In a real scenario, this would involve applying a clustering algorithm
        (e.g., KMeans, DBSCAN) to price features (e.g., normalized price, volatility).
        For now, it's a proxy for detecting distinct price behaviors within a range.
        """
        if klines_df.empty or len(klines_df) < 50: # Need enough data for clustering
            return {"cluster_affinity_score": 0.5, "is_tight_cluster": 0}

        # Simplified: Use standard deviation of recent prices as a proxy for 'tightness'
        recent_prices = klines_df['close'].iloc[-50:]
        price_std_dev = recent_prices.std()
        avg_price = recent_prices.mean()

        if avg_price == 0: # Avoid division by zero
            return {"cluster_affinity_score": 0.5, "is_tight_cluster": 0}

        # Normalize std dev relative to price
        normalized_std_dev = price_std_dev / avg_price

        # A lower normalized std dev implies a tighter cluster
        # Map this to a score where lower std dev means higher affinity/score
        cluster_affinity_score = 1 - np.clip(normalized_std_dev * 10, 0, 1) # Scale and clip
        is_tight_cluster = 1 if normalized_std_dev < 0.005 else 0 # e.g., less than 0.5% std dev

        self.logger.debug(f"Clustering Features: Affinity={cluster_affinity_score:.2f}, Tight={is_tight_cluster}")
        return {"cluster_affinity_score": cluster_affinity_score, "is_tight_cluster": is_tight_cluster}

    def _get_bb_squeeze_features(self, klines_df: pd.DataFrame) -> dict:
        """
        Calculates Bollinger Band Squeeze features.
        A squeeze indicates low volatility, often preceding a breakout.
        """
        bb_length = self.config.get("bollinger_bands", {}).get("window", 20)
        bb_std_dev = self.config.get("bollinger_bands", {}).get("num_std_dev", 2)
        bb_squeeze_threshold = self.config.get("bband_squeeze_threshold", 0.01)

        if klines_df.empty or len(klines_df) < bb_length:
            return {"bb_width_norm": 0.0, "is_bb_squeeze": 0}

        bb = klines_df.ta.bbands(length=bb_length, std=bb_std_dev, append=False)
        
        # Ensure Bollinger Band columns exist
        upper_band_col = f'BBU_{bb_length}_{bb_std_dev:.1f}'
        lower_band_col = f'BBL_{bb_length}_{bb_std_dev:.1f}'

        if upper_band_col not in bb.columns or lower_band_col not in bb.columns:
            self.logger.warning("Bollinger Bands columns not found. Skipping BB squeeze features.")
            return {"bb_width_norm": 0.0, "is_bb_squeeze": 0}

        bb_width = (bb[upper_band_col].iloc[-1] - bb[lower_band_col].iloc[-1])
        current_close = klines_df['close'].iloc[-1]

        bb_width_norm = bb_width / current_close if current_close > 0 else 0.0
        is_bb_squeeze = 1 if bb_width_norm < bb_squeeze_threshold else 0

        self.logger.debug(f"BB Squeeze Features: Width Norm={bb_width_norm:.4f}, Squeeze={is_bb_squeeze}")
        return {"bb_width_norm": bb_width_norm, "is_bb_squeeze": is_bb_squeeze}

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

        # Clustering Model (KMeans)
        self.logger.info("Training Clustering model (KMeans)...")
        clustering_features_for_training = historical_features[['close', 'volume', 'ATR']].dropna()
        if not clustering_features_for_training.empty and len(clustering_features_for_training) >= self.config.get("kmeans_n_clusters", 3):
            try:
                self.models["clustering"] = KMeans(n_clusters=self.config.get("kmeans_n_clusters", 3), random_state=42, n_init=10) # n_init for robustness
                self.models["clustering"].fit(clustering_features_for_training)
                self.logger.info("KMeans Clustering model trained.")
            except Exception as e:
                self.logger.error(f"Error training KMeans Clustering model: {e}")
                self.models["clustering"] = None
        else:
            self.logger.warning("Insufficient data or invalid cluster count for KMeans training. Skipping.")
            self.models["clustering"] = None

        # BB Squeeze Model (Rule-based or simple ML)
        self.logger.info("Training BB Squeeze model (Conceptual/Rule-based)...")
        # No explicit training needed for rule-based, just a placeholder for existence
        self.models["bb_squeeze"] = True # Simulate trained model

        # Order Flow Model (LightGBM Classifier)
        self.logger.info("Training Order Flow model (LGBMClassifier)...")
        # Ensure historical_features contains order flow features
        order_flow_features_for_training = historical_features[['cvd_value', 'cvd_divergence_score', 'is_absorption', 'is_exhaustion', 'aggressive_buy_volume', 'aggressive_sell_volume']].dropna()
        
        # Simulate a target for order flow model based on short-term price movement
        # Example: 1 if price moves up by 0.1% in next 2 periods, -1 if down by 0.1%, 0 otherwise
        target_lookahead_of = 2
        price_change_of = historical_features['close'].pct_change(target_lookahead_of).shift(-target_lookahead_of)
        # Simplified target: 1 for up, 0 for down/sideways
        simulated_of_target = (price_change_of > 0.001).astype(int) 
        
        # Align order flow features with simulated target
        aligned_of_data = order_flow_features_for_training.join(simulated_of_target.rename('of_target')).dropna()
        
        if not aligned_of_data.empty:
            X_of = aligned_of_data.drop(columns=['of_target'])
            y_of = aligned_of_data['of_target']
            try:
                self.models["order_flow"] = LGBMClassifier(
                    random_state=42, 
                    verbose=-1,
                    reg_alpha=self.dl_config["l1_reg_strength"],
                    reg_lambda=self.dl_config["l2_reg_strength"]
                )
                self.models["order_flow"].fit(X_of, y_of)
                self.logger.info("Order Flow model trained.")
            except Exception as e:
                self.logger.error(f"Error training Order Flow model: {e}")
                self.models["order_flow"] = None
        else:
            self.logger.warning("Insufficient aligned data for Order Flow model training. Skipping.")
            self.models["order_flow"] = None


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
        if self.models["clustering"]:
            # Predict cluster labels for historical data
            historical_clustering_input = historical_features[['close', 'volume', 'ATR']].dropna()
            if not historical_clustering_input.empty:
                historical_cluster_labels = self.models["clustering"].predict(historical_clustering_input)
                meta_features_train['cluster_label'] = pd.Series(historical_cluster_labels, index=historical_clustering_input.index).reindex(X_flat.index).fillna(-1) # Use -1 for unclustered
            else:
                meta_features_train['cluster_label'] = -1 # Default if no clustering data
        else:
            meta_features_train['cluster_label'] = -1 # Default if no clustering model

        historical_bb_squeeze_features_list = []
        for idx, row in historical_features.iterrows():
            historical_bb_squeeze_features_list.append(self._get_bb_squeeze_features(historical_features.loc[:idx]))
        historical_bb_squeeze_features = pd.DataFrame(historical_bb_squeeze_features_list, index=historical_features.index).loc[X_flat.index].fillna(0)
        meta_features_train = meta_features_train.join(historical_bb_squeeze_features, how='left')
        
        if self.models["order_flow"]:
            order_flow_probas = self.models["order_flow"].predict_proba(X_of) # Use X_of from above
            # Assuming a binary classification (0 or 1), take probability of class 1
            order_flow_conf_values = [order_flow_probas[i, self.models["order_flow"].classes_.tolist().index(y_of.iloc[i])] for i in range(len(y_of))]
            meta_features_train['order_flow_proba'] = pd.Series(order_flow_conf_values, index=X_of.index).reindex(X_flat.index).fillna(0.5)
        else:
            meta_features_train['order_flow_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))


        if self.models["tabnet_proxy"]:
            tabnet_probas = self.models["tabnet_proxy"].predict(X_flat)
            if len(tabnet_probas) == len(y_flat_encoded):
                tabnet_conf_values = [tabnet_probas[i, y_flat_encoded[i]] for i in range(len(y_flat_encoded))]
                meta_features_train['tabnet_proxy_proba'] = pd.Series(tabnet_conf_values, index=X_flat.index)
            else:
                self.logger.warning("TabNet proxy probas length mismatch with y_flat_encoded. Using random uniform for tabnet_proxy_proba.")
                meta_features_train['tabnet_proxy_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))
        else:
            meta_features_train['tabnet_proxy_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))

        if self.models["lgbm"]:
            lgbm_probas = self.models["lgbm"].predict_proba(X_flat)
            if len(lgbm_probas) == len(y_flat):
                lgbm_conf_values = [lgbm_probas[i, self.models["lgbm"].classes_.tolist().index(y_flat.iloc[i])] for i in range(len(y_flat))]
                meta_features_train['lgbm_proba'] = pd.Series(lgbm_conf_values, index=X_flat.index)
            else:
                self.logger.warning("LGBM probas length mismatch with y_flat. Using random uniform for lgbm_proba.")
                meta_features_train['lgbm_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))
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

        # Clustering Prediction
        if self.models["clustering"]:
            try:
                clustering_input = current_features_row[['close', 'volume', 'ATR']].dropna()
                if not clustering_input.empty:
                    current_cluster_label = self.models["clustering"].predict(clustering_input)[0]
                    individual_model_outputs['cluster_label'] = float(current_cluster_label)
                else:
                    individual_model_outputs['cluster_label'] = -1.0
            except Exception as e:
                self.logger.error(f"Error predicting with KMeans Clustering: {e}")
                individual_model_outputs['cluster_label'] = -1.0 # Default to -1 if prediction fails
        else:
            individual_model_outputs['cluster_label'] = -1.0 # Default if no clustering model


        # BB Squeeze Prediction (Conceptual/Rule-based)
        current_bb_squeeze_features = self._get_bb_squeeze_features(klines_df)
        individual_model_outputs['bb_width_norm'] = float(current_bb_squeeze_features['bb_width_norm'])
        individual_model_outputs['is_bb_squeeze'] = float(current_bb_squeeze_features['is_bb_squeeze'])

        # Order Flow Prediction
        if self.models["order_flow"]:
            try:
                order_flow_input = current_features_row[['cvd_value', 'cvd_divergence_score', 'is_absorption', 'is_exhaustion', 'aggressive_buy_volume', 'aggressive_sell_volume']].dropna()
                if not order_flow_input.empty:
                    order_flow_proba = self.models["order_flow"].predict_proba(order_flow_input)[0]
                    # Assuming binary classification for simplicity, take probability of positive class (index 1)
                    individual_model_outputs['order_flow_proba'] = float(order_flow_proba[1]) 
                else:
                    individual_model_outputs['order_flow_proba'] = 0.5
            except Exception as e:
                self.logger.error(f"Error predicting with Order Flow model: {e}")
                individual_model_outputs['order_flow_proba'] = 0.5
        else:
            individual_model_outputs['order_flow_proba'] = 0.5


        # TabNet-like Prediction
        if self.models["tabnet_proxy"]:
            try:
                tabnet_proba = self.models["tabnet_proxy"].predict(X_current_flat)[0]
                individual_model_outputs['tabnet_proxy_proba'] = float(np.max(tabnet_proba))
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
                individual_model_outputs['lgbm_proba'] = float(np.max(lgbm_proba))
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
            # Adjust weights for the new features (cluster_label, order_flow_proba)
            # Note: cluster_label is a categorical feature, not a confidence score for averaging.
            # It should be treated as a direct input to the meta-learner.
            # For fallback averaging, we can only average actual confidence-like scores.
            
            # This fallback averaging is less meaningful now with diverse features.
            # It's better to rely on the meta-learner. If meta-learner is not trained,
            # the ensemble should ideally return a neutral prediction.
            self.logger.warning("Fallback averaging is less meaningful with implemented models. Returning HOLD.")
            final_prediction = "HOLD"
            final_confidence = 0.0


        self.logger.info(f"Ensemble Result for {self.ensemble_name}: Prediction={final_prediction}, Confidence={final_confidence:.2f}")
        return {"prediction": final_prediction, "confidence": final_confidence}
