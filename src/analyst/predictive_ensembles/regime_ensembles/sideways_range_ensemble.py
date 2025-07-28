# src/analyst/predictive_ensembles/regime_ensembles/sideways_range_ensemble.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans # For clustering in sideways
from lightgbm import LGBMClassifier
import pandas_ta as ta
import joblib # For saving/loading models

# Import TabNetClassifier
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    TABNET_AVAILABLE = True
except ImportError:
    print("Warning: pytorch-tabnet not installed. Falling back to Dense layers for TabNet-like model.")
    TABNET_AVAILABLE = False
    # Fallback for TensorFlow/Keras imports if TabNet is not available
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l1_l2 # For L1/L2 regularization


from .base_ensemble import BaseEnsemble

class SidewaysRangeEnsemble(BaseEnsemble):
    """
    Predictive Ensemble specifically designed for SIDEWAYS_RANGE market regimes.
    Focuses on mean reversion, breakout detection, and order flow.
    """
    def __init__(self, config: dict, ensemble_name: str):
        super().__init__(config, ensemble_name)
        self.models = {
            "clustering": None, # KMeans clustering model
            "bb_squeeze": None, # ML-enhanced Bollinger Band Squeeze model (LGBM)
            "order_flow": None, # Order flow analysis model (conceptual)
            "tabnet": None, # TabNetClassifier model (or Dense layer proxy)
            "lgbm": None # LightGBM for general patterns
        }
        self.ensemble_weights = self.config.get("ensemble_weights", {
            "clustering": 0.2, "bb_squeeze": 0.2, "order_flow": 0.2, "tabnet": 0.2, "lgbm": 0.2
        })
        self.min_confluence_confidence = self.config.get("min_confluence_confidence", 0.7)

        # TabNet/DL model specific configurations
        self.dl_config = {
            "tabnet_n_d": self.config.get("tabnet_n_d", 64), # Dimension of the prediction layer (TabNet specific)
            "tabnet_n_a": self.config.get("tabnet_n_a", 64), # Dimension of the attention layer (TabNet specific)
            "tabnet_n_steps": self.config.get("tabnet_n_steps", 3), # Number of steps in the attention mechanism (TabNet specific)
            "tabnet_gamma": self.config.get("tabnet_gamma", 1.3), # Gamma for attention (TabNet specific)
            "tabnet_cat_emb_dim": self.config.get("tabnet_cat_emb_dim", 1), # Categorical embedding dimension (TabNet specific)
            "tabnet_n_independent": self.config.get("tabnet_n_independent", 2), # Number of independent GLU layers in each GLU block (TabNet specific)
            "tabnet_n_shared": self.config.get("tabnet_n_shared", 2), # Number of shared GLU layers in each GLU block (TabNet specific)
            "tabnet_momentum": self.config.get("tabnet_momentum", 0.02), # Momentum for batch normalization (TabNet specific)
            "tabnet_mask_type": self.config.get("tabnet_mask_type", "entmax"), # Mask type for attention (TabNet specific)
            
            "dense_units": 64, # Fallback units for Dense layers if TabNet not available
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "l1_reg_strength": self.config.get("meta_learner_l1_reg", 0.001),
            "l2_reg_strength": self.config.get("meta_learner_l2_reg", 0.001),
            "kmeans_n_clusters": self.config.get("kmeans_n_clusters", 3), # Number of clusters for KMeans
        }
        self.expected_dl_features = ['close', 'volume', 'ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']
        self.dl_target_map = {} # To store target label to integer mapping for DL models
        self.bb_squeeze_features_list = [] # To store feature names for bb_squeeze model

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

    def _get_clustering_features(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates features for clustering, indicative of market tightness and range.
        These features will be used by the KMeans model.
        """
        if klines_df.empty or len(klines_df) < 50: # Need enough data for meaningful features
            self.logger.warning("Insufficient data for clustering features. Returning empty DataFrame.")
            return pd.DataFrame()

        recent_prices = klines_df['close'].iloc[-50:]
        recent_volumes = klines_df['volume'].iloc[-50:]
        
        features = pd.DataFrame(index=klines_df.index)

        # Volatility measures
        features['rolling_std_dev_10'] = recent_prices.rolling(window=10).std()
        features['rolling_std_dev_30'] = recent_prices.rolling(window=30).std()
        
        # Price range measures
        features['price_range_10'] = (recent_prices.max() - recent_prices.min()) / recent_prices.mean() if recent_prices.mean() > 0 else 0
        features['price_range_30'] = (recent_prices.max() - recent_prices.min()) / recent_prices.mean() if recent_prices.mean() > 0 else 0

        # Volume consistency
        features['volume_std_dev_10'] = recent_volumes.rolling(window=10).std()
        features['volume_mean_10'] = recent_volumes.rolling(window=10).mean()

        features.fillna(0, inplace=True) # Fill NaNs from rolling windows
        
        # Return only the last row of features for the current candle
        return features.tail(1)


    def _get_bb_squeeze_features(self, klines_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Bollinger Band Squeeze features.
        A squeeze indicates low volatility, often preceding a breakout.
        Returns a DataFrame of features for the BB Squeeze model.
        """
        bb_length = self.config.get("bollinger_bands", {}).get("window", 20)
        bb_std_dev = self.config.get("bollinger_bands", {}).get("num_std_dev", 2)
        
        if klines_df.empty or len(klines_df) < bb_length:
            self.logger.warning("Insufficient data for BB squeeze features. Returning empty DataFrame.")
            return pd.DataFrame()

        bb = klines_df.ta.bbands(length=bb_length, std=bb_std_dev, append=False)
        
        # Ensure Bollinger Band columns exist
        upper_band_col = f'BBU_{bb_length}_{bb_std_dev:.1f}'
        lower_band_col = f'BBL_{bb_length}_{bb_std_dev:.1f}'
        mid_band_col = f'BBM_{bb_length}_{bb_std_dev:.1f}'

        if upper_band_col not in bb.columns or lower_band_col not in bb.columns or mid_band_col not in bb.columns:
            self.logger.warning("Bollinger Bands columns not found. Skipping BB squeeze features.")
            return pd.DataFrame()

        features = pd.DataFrame(index=klines_df.index)
        features['bb_width'] = (bb[upper_band_col] - bb[lower_band_col])
        features['bb_percent_b'] = bb[f'BBP_{bb_length}_{bb_std_dev:.1f}']
        features['bb_band_ratio'] = features['bb_width'] / bb[mid_band_col] if bb[mid_band_col].iloc[-1] > 0 else 0
        
        # Keltner Channels (often used in conjunction with BB for squeeze)
        # Assuming ATR is already calculated in klines_df or can be derived
        kc_length = self.config.get("keltner_channels", {}).get("window", 20)
        kc_multiplier = self.config.get("keltner_channels", {}).get("multiplier", 2)

        if 'ATR' not in klines_df.columns:
            klines_df.ta.atr(length=kc_length, append=True, col_names=('ATR'))
        
        kc = klines_df.ta.kc(length=kc_length, scalar=kc_multiplier, append=False)
        upper_kc_col = f'KCU_{kc_length}_{kc_multiplier:.1f}'
        lower_kc_col = f'KCL_{kc_length}_{kc_multiplier:.1f}'

        if upper_kc_col in kc.columns and lower_kc_col in kc.columns:
            features['kc_width'] = (kc[upper_kc_col] - kc[lower_kc_col])
            features['squeeze_indicator'] = (features['bb_width'] / features['kc_width']) if features['kc_width'].iloc[-1] > 0 else 0
            features['is_squeeze'] = (features['squeeze_indicator'] < self.config.get("bband_squeeze_threshold", 0.8)).astype(int) # Squeeze when BB inside KC
        else:
            features['kc_width'] = 0
            features['squeeze_indicator'] = 0
            features['is_squeeze'] = 0
        
        features.fillna(0, inplace=True)
        return features.tail(1) # Return only the last row of features

    def train_ensemble(self, historical_features: pd.DataFrame, historical_klines: pd.DataFrame, historical_targets: pd.Series):
        """
        Trains the individual models and the meta-learner for the SIDEWAYS_RANGE ensemble.
        `historical_targets` could be 'MEAN_REVERT', 'BREAKOUT_UP', 'BREAKOUT_DOWN'.
        """
        self.logger.info(f"Training {self.ensemble_name} ensemble models...")

        if historical_features.empty or historical_targets.empty or historical_klines.empty:
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
        # Generate clustering features for historical data
        historical_clustering_features_list = []
        for idx, row in historical_klines.iterrows(): # Use historical_klines to derive these features
            # Pass a window of klines ending at current index for rolling features
            window_klines = historical_klines.loc[:idx].tail(50) # Ensure enough data for _get_clustering_features
            if not window_klines.empty and len(window_klines) >= 50:
                historical_clustering_features_list.append(self._get_clustering_features(window_klines).iloc[0])
            else:
                historical_clustering_features_list.append(pd.Series(0.0, index=['rolling_std_dev_10', 'rolling_std_dev_30', 'price_range_10', 'price_range_30', 'volume_std_dev_10', 'volume_mean_10']))

        historical_clustering_features = pd.DataFrame(historical_clustering_features_list, index=historical_klines.index)
        historical_clustering_features = historical_clustering_features.loc[X_flat.index].fillna(0) # Align with X_flat

        if not historical_clustering_features.empty and len(historical_clustering_features) >= self.dl_config["kmeans_n_clusters"]:
            try:
                self.kmeans_model = KMeans(n_clusters=self.dl_config["kmeans_n_clusters"], random_state=42, n_init=10)
                cluster_labels = self.kmeans_model.fit_predict(historical_clustering_features)
                X_flat['cluster_label'] = pd.Series(cluster_labels, index=X_flat.index) # Add cluster label as feature to X_flat
                self.logger.info(f"KMeans Clustering model trained with {self.dl_config['kmeans_n_clusters']} clusters.")
            except Exception as e:
                self.logger.error(f"Error training KMeans Clustering model: {e}. Skipping cluster feature.", exc_info=True)
                X_flat['cluster_label'] = -1 # Default to a neutral/unknown cluster
        else:
            self.logger.warning(f"Not enough data ({len(historical_clustering_features)} samples) to train KMeans with {self.dl_config['kmeans_n_clusters']} clusters. Skipping cluster feature.")
            X_flat['cluster_label'] = -1


        # BB Squeeze Model (LGBM Classifier)
        self.logger.info("Training BB Squeeze model (LGBM)...")
        # Derive BB Squeeze features for historical data
        historical_bb_squeeze_features_list = []
        for idx, row in historical_klines.iterrows(): # Use historical_klines to derive these features
            window_klines = historical_klines.loc[:idx].tail(max(self.config.get("bollinger_bands", {}).get("window", 20), self.config.get("keltner_channels", {}).get("window", 20)))
            if not window_klines.empty and len(window_klines) >= max(self.config.get("bollinger_bands", {}).get("window", 20), self.config.get("keltner_channels", {}).get("window", 20)):
                historical_bb_squeeze_features_list.append(self._get_bb_squeeze_features(window_klines).iloc[0])
            else:
                historical_bb_squeeze_features_list.append(pd.Series(0.0, index=['bb_width', 'bb_percent_b', 'bb_band_ratio', 'kc_width', 'squeeze_indicator', 'is_squeeze']))

        historical_bb_squeeze_features = pd.DataFrame(historical_bb_squeeze_features_list, index=historical_klines.index)
        historical_bb_squeeze_features = historical_bb_squeeze_features.loc[X_flat.index].fillna(0) # Align with X_flat
        
        # Define a target for BB Squeeze model (e.g., breakout vs mean-reversion)
        # This is a pseudo-labeling for training the BB Squeeze model
        target_lookahead_bb = self.config.get("bb_squeeze_target_lookahead", 5) # Look 5 periods ahead
        price_change_bb = historical_klines['close'].pct_change(target_lookahead_bb).shift(-target_lookahead_bb)
        
        # Example target: 'BREAKOUT_UP', 'BREAKOUT_DOWN', 'MEAN_REVERT'
        bb_squeeze_target = pd.Series('MEAN_REVERT', index=price_change_bb.index)
        bb_squeeze_target[price_change_bb > self.config.get("bb_squeeze_breakout_threshold", 0.005)] = 'BREAKOUT_UP'
        bb_squeeze_target[price_change_bb < -self.config.get("bb_squeeze_breakout_threshold", 0.005)] = 'BREAKOUT_DOWN'
        
        aligned_bb_squeeze_data = historical_bb_squeeze_features.join(bb_squeeze_target.rename('bb_target')).dropna()

        if not aligned_bb_squeeze_data.empty:
            X_bb_squeeze = aligned_bb_squeeze_data.drop(columns=['bb_target'])
            y_bb_squeeze = aligned_bb_squeeze_data['bb_target']
            self.bb_squeeze_features_list = X_bb_squeeze.columns.tolist() # Store feature names

            try:
                self.models["bb_squeeze"] = LGBMClassifier(
                    random_state=42, 
                    verbose=-1,
                    reg_alpha=self.dl_config["l1_reg_strength"],
                    reg_lambda=self.dl_config["l2_reg_strength"]
                )
                self.models["bb_squeeze"].fit(X_bb_squeeze, y_bb_squeeze)
                self.logger.info("BB Squeeze model trained.")
            except Exception as e:
                self.logger.error(f"Error training BB Squeeze model: {e}")
                self.models["bb_squeeze"] = None
        else:
            self.logger.warning("Insufficient aligned data for BB Squeeze model training. Skipping.")
            self.models["bb_squeeze"] = None


        # Order Flow Model (LightGBM Classifier)
        self.logger.info("Training Order Flow model (LGBMClassifier)...")
        # Ensure historical_features contains order flow features
        order_flow_features_for_training = historical_features[['cvd_value', 'cvd_divergence_score', 'is_absorption', 'is_exhaustion', 'aggressive_buy_volume', 'aggressive_sell_volume', 'order_book_imbalance', 'aggressive_volume_ratio', 'recent_volume_spike']].dropna()
        
        # Simulate a target for order flow model based on short-term price movement
        # Example: 1 if price moves up by 0.1% in next 2 periods, -1 if down by 0.1%, 0 otherwise
        target_lookahead_of = 2
        price_change_of = historical_klines['close'].pct_change(target_lookahead_of).shift(-target_lookahead_of)
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


        # TabNet Model (or Dense layer proxy if not available)
        self.logger.info("Training TabNet model...")
        if TABNET_AVAILABLE:
            try:
                # TabNetClassifier expects numpy arrays
                self.models["tabnet"] = TabNetClassifier(
                    n_d=self.dl_config["tabnet_n_d"],
                    n_a=self.dl_config["tabnet_n_a"],
                    n_steps=self.dl_config["tabnet_n_steps"],
                    gamma=self.dl_config["tabnet_gamma"],
                    cat_emb_dim=self.dl_config["tabnet_cat_emb_dim"],
                    n_independent=self.dl_config["tabnet_n_independent"],
                    n_shared=self.dl_config["tabnet_n_shared"],
                    momentum=self.dl_config["tabnet_momentum"],
                    mask_type=self.dl_config["tabnet_mask_type"],
                    seed=42,
                    verbose=0 # Suppress verbose output
                )
                self.models["tabnet"].fit(
                    X_flat.values, y_flat_encoded,
                    eval_set=[(X_flat.values, y_flat_encoded)], # Using training set as eval for simplicity
                    max_epochs=self.dl_config["epochs"],
                    batch_size=self.dl_config["batch_size"],
                    drop_last=False # Keep all samples
                )
                self.logger.info("TabNet model trained.")
            except Exception as e:
                self.logger.error(f"Error training TabNet model: {e}")
                self.models["tabnet"] = None
        else: # Fallback to Keras Dense layers
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
                self.models["tabnet"] = model
                self.logger.info("TabNet-like (Dense layer) model trained.")
            except Exception as e:
                self.logger.error(f"Error training TabNet-like (Dense layer) model: {e}")
                self.models["tabnet"] = None

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
            # Ensure historical_clustering_features is aligned with X_flat
            aligned_historical_clustering_features = historical_clustering_features.loc[X_flat.index].fillna(0)
            if not aligned_historical_clustering_features.empty:
                historical_cluster_labels = self.kmeans_model.predict(aligned_historical_clustering_features)
                meta_features_train['cluster_label'] = pd.Series(historical_cluster_labels, index=X_flat.index)
            else:
                meta_features_train['cluster_label'] = -1 # Default if no clustering data
        else:
            meta_features_train['cluster_label'] = -1 # Default if no clustering model

        if self.models["bb_squeeze"]:
            # Predict probabilities for historical BB Squeeze features
            aligned_historical_bb_squeeze_features = historical_bb_squeeze_features.loc[X_flat.index].fillna(0)
            if not aligned_historical_bb_squeeze_features.empty:
                bb_squeeze_probas = self.models["bb_squeeze"].predict_proba(aligned_historical_bb_squeeze_features)
                # Assuming a multi-class target, take the probability of the predicted class
                bb_squeeze_conf_values = []
                for i, proba_array in enumerate(bb_squeeze_probas):
                    predicted_class_idx = np.argmax(proba_array)
                    bb_squeeze_conf_values.append(proba_array[predicted_class_idx])
                meta_features_train['bb_squeeze_conf'] = pd.Series(bb_squeeze_conf_values, index=X_flat.index)
            else:
                meta_features_train['bb_squeeze_conf'] = 0.5 # Default neutral
        else:
            meta_features_train['bb_squeeze_conf'] = 0.5 # Default neutral
        
        if self.models["order_flow"]:
            order_flow_probas = self.models["order_flow"].predict_proba(X_of) # Use X_of from above
            # Assuming a binary classification (0 or 1), take probability of class 1
            order_flow_conf_values = [order_flow_probas[i, self.models["order_flow"].classes_.tolist().index(y_of.iloc[i])] for i in range(len(y_of))]
            meta_features_train['order_flow_proba'] = pd.Series(order_flow_conf_values, index=X_of.index).reindex(X_flat.index).fillna(0.5)
        else:
            meta_features_train['order_flow_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))


        if self.models["tabnet"]:
            if TABNET_AVAILABLE:
                tabnet_probas = self.models["tabnet"].predict_proba(X_flat.values)
            else: # Keras Dense layer proxy
                tabnet_probas = self.models["tabnet"].predict(X_flat)

            if len(tabnet_probas) == len(y_flat_encoded):
                tabnet_conf_values = [tabnet_probas[i, y_flat_encoded[i]] for i in range(len(y_flat_encoded))]
                meta_features_train['tabnet_proba'] = pd.Series(tabnet_conf_values, index=X_flat.index)
            else:
                self.logger.warning("TabNet probas length mismatch with y_flat_encoded. Using random uniform for tabnet_proba.")
                meta_features_train['tabnet_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))
        else:
            meta_features_train['tabnet_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))

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

        # Clustering Prediction (KMeans)
        if self.models["clustering"]:
            try:
                # Get clustering features for the current kline
                current_clustering_features = self._get_clustering_features(klines_df)
                if not current_clustering_features.empty:
                    current_cluster_label = self.kmeans_model.predict(current_clustering_features)[0]
                    individual_model_outputs['cluster_label'] = float(current_cluster_label)
                else:
                    individual_model_outputs['cluster_label'] = -1.0 # Default if no clustering features
            except Exception as e:
                self.logger.error(f"Error predicting with KMeans Clustering: {e}")
                individual_model_outputs['cluster_label'] = -1.0 # Default to -1 if prediction fails
        else:
            individual_model_outputs['cluster_label'] = -1.0 # Default if no clustering model


        # BB Squeeze Prediction (LGBM Classifier)
        if self.models["bb_squeeze"]:
            try:
                # Derive current BB Squeeze features
                current_bb_squeeze_features = self._get_bb_squeeze_features(klines_df)
                if not current_bb_squeeze_features.empty:
                    # Ensure feature order for prediction is same as training
                    current_bb_squeeze_features_reindexed = current_bb_squeeze_features.reindex(self.bb_squeeze_features_list, fill_value=0.0)
                    bb_squeeze_proba = self.models["bb_squeeze"].predict_proba(current_bb_squeeze_features_reindexed)[0]
                    # Get confidence of the predicted class
                    individual_model_outputs['bb_squeeze_conf'] = float(np.max(bb_squeeze_proba))
                else:
                    individual_model_outputs['bb_squeeze_conf'] = 0.5 # Default neutral
            except Exception as e:
                self.logger.error(f"Error predicting with BB Squeeze model: {e}")
                individual_model_outputs['bb_squeeze_conf'] = 0.5
        else:
            individual_model_outputs['bb_squeeze_conf'] = 0.5


        # Order Flow Prediction
        if self.models["order_flow"]:
            try:
                order_flow_input = current_features_row[['cvd_value', 'cvd_divergence_score', 'is_absorption', 'is_exhaustion', 'aggressive_buy_volume', 'aggressive_sell_volume', 'order_book_imbalance', 'aggressive_volume_ratio', 'recent_volume_spike']].dropna()
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


        # TabNet Prediction
        if self.models["tabnet"]:
            try:
                if TABNET_AVAILABLE:
                    tabnet_proba = self.models["tabnet"].predict_proba(X_current_flat.values)[0]
                else: # Keras Dense layer proxy
                    tabnet_proba = self.models["tabnet"].predict(X_current_flat)[0]
                individual_model_outputs['tabnet_proba'] = float(np.max(tabnet_proba))
            except Exception as e:
                self.logger.error(f"Error predicting with TabNet model: {e}")
                individual_model_outputs['tabnet_proba'] = 0.5
        else:
            individual_model_outputs['tabnet_proba'] = 0.5

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
