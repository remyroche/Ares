# src/analyst/predictive_ensembles/regime_ensembles/high_impact_candle_ensemble.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

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

class HighImpactCandleEnsemble(BaseEnsemble):
    """
    Predictive Ensemble specifically designed for HIGH_IMPACT_CANDLE market regimes.
    Focuses on volume imbalance, follow-through, and manipulation.
    """
    def __init__(self, config: dict, ensemble_name: str):
        super().__init__(config, ensemble_name)
        self.models = {
            "volume_imbalance": None, # LGBM model for volume imbalance features
            "tabnet": None, # TabNetClassifier model (or Dense layer proxy)
            "lgbm": None # LightGBM for general patterns
        }
        self.ensemble_weights = self.config.get("ensemble_weights", {
            "volume_imbalance": 0.4, "tabnet": 0.3, "lgbm": 0.3
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
        }
        self.expected_dl_features = ['close', 'volume', 'ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']
        self.dl_target_map = {} # To store target label to integer mapping for DL models
        self.volume_imbalance_features_list = [] # To store feature names for volume_imbalance model

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

    def _calculate_volume_imbalance_features(self, klines_df: pd.DataFrame) -> pd.Series:
        """
        Calculates volume imbalance features for the most recent candle.
        Prioritizes actual taker buy/sell volumes from klines, falls back to heuristic.
        
        Expected klines_df columns: 'volume', 'taker_buy_base_asset_volume', 'taker_sell_base_asset_volume'
        (or 'open', 'close' for heuristic fallback).
        """
        if klines_df.empty or len(klines_df) < 1:
            self.logger.warning("Volume imbalance: Input klines_df is empty.")
            return pd.Series({
                'volume_imbalance_ratio': 0.0,
                'aggressive_buy_pct': 0.0,
                'aggressive_sell_pct': 0.0,
                'imbalance_strength': 0.0
            }, dtype='float64')

        current_candle = klines_df.iloc[-1]
        total_volume = current_candle['volume']

        # Prioritize explicit taker buy/sell volumes if available
        if 'taker_buy_base_asset_volume' in current_candle and 'taker_sell_base_asset_volume' in current_candle:
            taker_buy_volume = current_candle['taker_buy_base_asset_volume']
            taker_sell_volume = current_candle['taker_sell_base_asset_volume']
            self.logger.debug("Using explicit taker buy/sell volumes for imbalance calculation.")
        elif 'taker_buy_base_asset_volume' in current_candle and total_volume > 0:
            # If only taker_buy is available, approximate taker_sell
            taker_buy_volume = current_candle['taker_buy_base_asset_volume']
            taker_sell_volume = total_volume - taker_buy_volume
            self.logger.warning("Using taker_buy_base_asset_volume and approximating taker_sell_volume for imbalance.")
        else:
            # Fallback heuristic if detailed taker volume is not directly in klines_df
            # This is a very rough approximation based on candle body and total volume
            if current_candle['close'] > current_candle['open']: # Green candle
                taker_buy_volume = total_volume * 0.6 # Assume more buying
                taker_sell_volume = total_volume * 0.4
            else: # Red candle
                taker_buy_volume = total_volume * 0.4
                taker_sell_volume = total_volume * 0.6
            self.logger.warning("Using heuristic for taker buy/sell volume. For accuracy, ensure 'taker_buy_base_asset_volume' and 'taker_sell_base_asset_volume' are available in klines.")

        if total_volume == 0:
            volume_imbalance_ratio = 0.0
            aggressive_buy_pct = 0.0
            aggressive_sell_pct = 0.0
        else:
            volume_imbalance_ratio = (taker_buy_volume - taker_sell_volume) / total_volume
            aggressive_buy_pct = taker_buy_volume / total_volume
            aggressive_sell_pct = taker_sell_volume / total_volume
        
        # Imbalance strength: Higher absolute imbalance ratio means stronger signal
        imbalance_strength = abs(volume_imbalance_ratio)

        return pd.Series({
            'volume_imbalance_ratio': volume_imbalance_ratio, # -1 (strong sell) to 1 (strong buy)
            'aggressive_buy_pct': aggressive_buy_pct,
            'aggressive_sell_pct': aggressive_sell_pct,
            'imbalance_strength': imbalance_strength
        }, dtype='float64')


    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        """
        Trains the individual models and the meta-learner for the HIGH_IMPACT_CANDLE ensemble.
        `historical_targets` could be 'FOLLOW_THROUGH_UP', 'FOLLOW_THROUGH_DOWN', 'REVERSAL'.
        """
        self.logger.info(f"Training {self.ensemble_name} ensemble models...")

        if historical_features.empty or historical_targets.empty:
            self.logger.warning(f"No data to train {self.ensemble_name}. Skipping training.")
            return

        # Prepare flat data for TabNet/LGBM
        X_flat, y_flat = self._prepare_flat_data(historical_features, historical_targets)

        if X_flat.empty:
            self.logger.warning(f"Insufficient data after preparation for {self.ensemble_name}. Skipping model training.")
            return

        # Derive volume imbalance features for the historical data
        # This requires historical_features to contain 'open', 'close', 'volume', and potentially 'taker_buy_base_asset_volume'
        # Ensure that the columns required for _calculate_volume_imbalance_features are present in historical_features
        # We need to pass the relevant klines columns for accurate imbalance calculation.
        # Assuming historical_features contains these columns (e.g., from FeatureEngineering)
        
        historical_imbalance_features_list = []
        for idx, row in historical_features.iterrows():
            # Create a dummy DataFrame for the single row to pass to _calculate_volume_imbalance_features
            # Ensure all necessary columns are present for _calculate_volume_imbalance_features
            single_row_df = pd.DataFrame([row.to_dict()])
            single_row_df.index = [idx]
            historical_imbalance_features_list.append(self._calculate_volume_imbalance_features(single_row_df))
        
        historical_imbalance_features = pd.DataFrame(historical_imbalance_features_list, index=historical_features.index)
        
        # Align historical_imbalance_features with X_flat and y_flat
        historical_imbalance_features = historical_imbalance_features.loc[X_flat.index].fillna(0)
        
        # Store volume_imbalance_features_list for consistent prediction
        self.volume_imbalance_features_list = historical_imbalance_features.columns.tolist()


        unique_targets = np.unique(y_flat)
        self.dl_target_map = {label: i for i, label in enumerate(unique_targets)}
        y_flat_encoded = np.array([self.dl_target_map[label] for label in y_flat])

        # --- Train Individual Models ---

        # Volume Imbalance Model (LGBM Classifier)
        self.logger.info("Training Volume Imbalance model (LGBM)...")
        try:
            self.models["volume_imbalance"] = LGBMClassifier(
                random_state=42, 
                verbose=-1,
                reg_alpha=self.dl_config["l1_reg_strength"],
                reg_lambda=self.dl_config["l2_reg_strength"]
            )
            self.models["volume_imbalance"].fit(historical_imbalance_features, y_flat)
            self.logger.info("Volume Imbalance model trained.")
        except Exception as e:
            self.logger.error(f"Error training Volume Imbalance model: {e}")
            self.models["volume_imbalance"] = None

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
        if self.models["volume_imbalance"]:
            imbalance_probas = self.models["volume_imbalance"].predict_proba(historical_imbalance_features)
            # Ensure imbalance_probas length matches y_flat length
            if len(imbalance_probas) == len(y_flat):
                imbalance_conf_values = [imbalance_probas[i, self.models["volume_imbalance"].classes_.tolist().index(y_flat.iloc[i])] for i in range(len(y_flat))]
                meta_features_train['volume_imbalance_conf'] = pd.Series(imbalance_conf_values, index=X_flat.index)
            else:
                self.logger.warning("Volume Imbalance probas length mismatch with y_flat. Using random uniform for volume_imbalance_conf.")
                meta_features_train['volume_imbalance_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        else:
            meta_features_train['volume_imbalance_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        
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
        Gets a combined prediction and confidence score for the HIGH_IMPACT_CANDLE regime.
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

        # Volume Imbalance Prediction
        if self.models["volume_imbalance"]:
            try:
                # Derive current volume imbalance features
                # Ensure klines_df passed has the required columns for _calculate_volume_imbalance_features
                current_imbalance_features = self._calculate_volume_imbalance_features(klines_df.tail(1)) # Pass only current candle
                # Ensure feature order for prediction is same as training
                current_imbalance_features_reindexed = current_imbalance_features.reindex(self.volume_imbalance_features_list, fill_value=0.0).to_frame().T
                
                imbalance_proba = self.models["volume_imbalance"].predict_proba(current_imbalance_features_reindexed)[0]
                individual_model_outputs['volume_imbalance_conf'] = float(np.max(imbalance_proba))
            except Exception as e:
                self.logger.error(f"Error predicting with Volume Imbalance model: {e}")
                individual_model_outputs['volume_imbalance_conf'] = 0.5
        else:
            individual_model_outputs['volume_imbalance_conf'] = np.random.uniform(0.5, 0.9)

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
            total_weighted_confidence = sum(individual_model_outputs.get(m, 0.5) * self.ensemble_weights.get(m.replace('_conf', '').replace('_proba', ''), 1.0) 
                                            for m in ['volume_imbalance_conf', 'tabnet_proba', 'lgbm_proba'])
            total_weight = sum(self.ensemble_weights.get(m.replace('_conf', '').replace('_proba', ''), 1.0) for m in ['volume_imbalance_conf', 'tabnet_proba', 'lgbm_proba'])
            
            if total_weight > 0:
                final_confidence = total_weighted_confidence / total_weight
            
            if final_confidence > self.min_confluence_confidence:
                final_prediction = np.random.choice(["FOLLOW_THROUGH_UP", "FOLLOW_THROUGH_DOWN", "REVERSAL"])
            else:
                final_prediction = "HOLD"

        self.logger.info(f"Ensemble Result for {self.ensemble_name}: Prediction={final_prediction}, Confidence={final_confidence:.2f}")
        return {"prediction": final_prediction, "confidence": final_confidence}
