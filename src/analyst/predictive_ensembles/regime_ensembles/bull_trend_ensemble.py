# src/analyst/predictive_ensembles/regime_ensembles/bull_trend_ensemble.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
import pandas_ta as ta
from arch import arch_model # For GARCH model
import warnings
from sklearn.model_selection import KFold
from backtesting.ares_deep_analyzer import run_monte_carlo_simulation, plot_results


# Suppress specific warnings from ARCH library
warnings.filterwarnings("ignore", category=UserWarning, module="arch")
warnings.filterwarnings("ignore", category=FutureWarning, module="arch")

# TensorFlow/Keras imports for Deep Learning models
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, MultiHeadSelfAttention, LayerNormalization, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1_l2 # For L1/L2 regularization
from tensorflow.keras.callbacks import EarlyStopping # For conceptual early stopping

from .base_ensemble import BaseEnsemble

class BullTrendEnsemble(BaseEnsemble):
    """
    Predictive Ensemble specifically designed for BULL_TREND market regimes.
    Combines LSTM, Transformer (conceptual), Statistical (GARCH), and LightGBM models.
    """
    def __init__(self, config: dict, ensemble_name: str):
        super().__init__(config, ensemble_name)
        # Specific models for this ensemble
        self.models = {
            "lstm": None, # LSTM model
            "transformer": None, # Transformer-like model
            "garch": None, # GARCH model for volatility/direction
            "lgbm": None # LightGBM for pattern recognition
        }
        self.ensemble_weights = self.config.get("ensemble_weights", {
            "lstm": 0.3, "transformer": 0.3, "garch": 0.2, "lgbm": 0.2
        })
        self.min_confluence_confidence = self.config.get("min_confluence_confidence", 0.7)

        # DL model specific configurations (can be moved to global config if desired)
        self.dl_config = {
            "sequence_length": 20, # Number of past time steps to consider for LSTM/Transformer
            "lstm_units": 50,
            "transformer_heads": 2,
            "transformer_key_dim": 32,
            "dropout_rate": 0.2,
            "learning_rate": 0.001,
            "epochs": 50,
            "batch_size": 32,
            "l1_reg_strength": self.config.get("meta_learner_l1_reg", 0.001), # Re-using meta-learner reg for DL
            "l2_reg_strength": self.config.get("meta_learner_l2_reg", 0.001), # Re-using meta-learner reg for DL
        }
        # Store expected feature columns for consistent input to DL models
        self.expected_dl_features = ['close', 'volume', 'ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']
        self.dl_target_map = {} # To store target label to integer mapping for DL models

    def _prepare_sequence_data(self, df: pd.DataFrame, target_series: pd.Series = None, sequence_length: int = 20):
        """
        Prepares data into sequences for LSTM/Transformer models.
        Ensures consistent feature columns and handles missing data.
        """
        # Ensure all expected features are present and fill missing ones
        features_df = df[self.expected_dl_features].copy()
        features_df = features_df.fillna(0) # Fill NaNs for DL models

        X, y = [], []
        # Iterate to create sequences
        for i in range(len(features_df) - sequence_length + 1): # +1 to include the last possible sequence
            X.append(features_df.iloc[i:(i + sequence_length)].values)
            if target_series is not None:
                # Target corresponds to the label at the end of the sequence
                y.append(target_series.iloc[i + sequence_length - 1])
        
        if not X: # If no sequences could be formed
            self.logger.warning(f"No sequences formed from data of length {len(features_df)} with sequence_length {sequence_length}.")
            return np.array([]), np.array([])

        X = np.array(X)
        y = np.array(y) if target_series is not None else None
        
        # Final check for alignment if targets are provided
        if y is not None and len(y) != len(X):
            self.logger.error(f"Mismatch between X ({len(X)}) and y ({len(y)}) lengths after sequence preparation.")
            return np.array([]), np.array([]) # Return empty on critical error

        return X, y

    def train_ensemble(self, historical_features: pd.DataFrame, historical_targets: pd.Series):
        """
        Trains the individual models and the meta-learner for the BULL_TREND ensemble.
        `historical_targets` should represent the desired outcome (e.g., 'CONTINUE_UP', 'PULLBACK').
        """
        self.logger.info(f"Training {self.ensemble_name} ensemble models...")

        if historical_features.empty or historical_targets.empty:
            self.logger.warning(f"No data to train {self.ensemble_name}. Skipping training.")
            return

        # Align features and targets for all models
        # Ensure historical_features has all expected_dl_features columns before joining
        for col in self.expected_dl_features:
            if col not in historical_features.columns:
                historical_features[col] = 0.0 # Add missing columns with default value
        
        aligned_data = historical_features.join(historical_targets.rename('target')).dropna()
        if aligned_data.empty:
            self.logger.warning(f"No aligned data after dropping NaNs for {self.ensemble_name}. Skipping training.")
            return
        
        # Prepare data for DL models (sequences)
        X_seq, y_seq = self._prepare_sequence_data(aligned_data, aligned_data['target'], self.dl_config["sequence_length"])
        
        # Prepare data for non-DL models (flat features)
        # Ensure X_flat has all expected features for LGBM consistency
        X_flat_features_list = ['ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']
        X_flat = aligned_data[X_flat_features_list].copy()
        X_flat = X_flat.fillna(0) # Fill NaNs for LGBM
        y_flat = aligned_data['target']

        if X_seq.size == 0 or X_flat.empty:
            self.logger.warning(f"Insufficient data after sequence preparation for {self.ensemble_name}. Skipping DL model training.")
            return

        # Map string targets to integers for Keras and store the mapping
        unique_targets = np.unique(y_seq)
        self.dl_target_map = {label: i for i, label in enumerate(unique_targets)}
        y_seq_encoded = np.array([self.dl_target_map[label] for label in y_seq])

        # --- Train Individual Models ---

        # LSTM Model with L1-L2 Regularization
        self.logger.info("Training LSTM model with L1-L2 Regularization...")
        try:
            model = Sequential([
                LSTM(self.dl_config["lstm_units"], return_sequences=True,
                     kernel_regularizer=l1_l2(l1=self.dl_config["l1_reg_strength"], l2=self.dl_config["l2_reg_strength"]),
                     input_shape=(X_seq.shape[1], X_seq.shape[2])),
                Dropout(self.dl_config["dropout_rate"]),
                LSTM(self.dl_config["lstm_units"] // 2,
                     kernel_regularizer=l1_l2(l1=self.dl_config["l1_reg_strength"], l2=self.dl_config["l2_reg_strength"])),
                Dropout(self.dl_config["dropout_rate"]),
                Dense(len(unique_targets), activation='softmax',
                      kernel_regularizer=l1_l2(l1=self.dl_config["l1_reg_strength"], l2=self.dl_config["l2_reg_strength"]))
            ])
            model.compile(optimizer=Adam(learning_rate=self.dl_config["learning_rate"]),
                          loss='sparse_categorical_crossentropy' if len(unique_targets) > 2 else 'binary_crossentropy',
                          metrics=['accuracy'])
            model.fit(X_seq, y_seq_encoded, epochs=self.dl_config["epochs"], batch_size=self.dl_config["batch_size"], verbose=0)
            self.models["lstm"] = model
            self.logger.info("LSTM model trained.")
        except Exception as e:
            self.logger.error(f"Error training LSTM model: {e}")
            self.models["lstm"] = None

        # Transformer Model with L1-L2 Regularization
        self.logger.info("Training Transformer-like model with L1-L2 Regularization...")
        try:
            inputs = Input(shape=(X_seq.shape[1], X_seq.shape[2]))
            attn_output = MultiHeadSelfAttention(num_heads=self.dl_config["transformer_heads"],
                                                 key_dim=self.dl_config["transformer_key_dim"])(inputs, inputs)
            norm1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
            ffn_output = Dense(X_seq.shape[2], activation="relu",
                               kernel_regularizer=l1_l2(l1=self.dl_config["l1_reg_strength"], l2=self.dl_config["l2_reg_strength"])) (norm1)
            norm2 = LayerNormalization(epsilon=1e-6)(norm1 + ffn_output)

            flattened_output = tf.keras.layers.Flatten()(norm2)
            outputs = Dense(len(unique_targets), activation='softmax',
                            kernel_regularizer=l1_l2(l1=self.dl_config["l1_reg_strength"], l2=self.dl_config["l2_reg_strength"])) (flattened_output)

            model = tf.keras.Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer=Adam(learning_rate=self.dl_config["learning_rate"]),
                          loss='sparse_categorical_crossentropy' if len(unique_targets) > 2 else 'binary_crossentropy',
                          metrics=['accuracy'])
            model.fit(X_seq, y_seq_encoded, epochs=self.dl_config["epochs"], batch_size=self.dl_config["batch_size"], verbose=0)
            self.models["transformer"] = model
            self.logger.info("Transformer-like model trained.")
        except Exception as e:
            self.logger.error(f"Error training Transformer-like model: {e}")
            self.models["transformer"] = None

        # GARCH Model
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

        # LightGBM Model with Cross-Validation and L1-L2 Regularization
        self.logger.info("Training LightGBM model with Cross-Validation and L1-L2 Regularization...")
        try:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            oof_preds = np.zeros(len(X_flat))
            models = []

            for fold, (train_index, val_index) in enumerate(kf.split(X_flat, y_flat)):
                X_train, X_val = X_flat.iloc[train_index], X_flat.iloc[val_index]
                y_train, y_val = y_flat.iloc[train_index], y_flat.iloc[val_index]

                model = LGBMClassifier(
                    random_state=42,
                    verbose=-1,
                    reg_alpha=self.dl_config["l1_reg_strength"],  # L1 regularization
                    reg_lambda=self.dl_config["l2_reg_strength"]   # L2 regularization
                )
                model.fit(X_train, y_train,
                          eval_set=[(X_val, y_val)])
                # oof_preds[val_index] = model.predict_proba(X_val)[:, 1] # Assuming binary classification for simplicity
                models.append(model)
            self.models["lgbm"] = models  # Store all fold models
            self.logger.info("LightGBM model trained with 5-fold cross-validation.")
        except Exception as e:
            self.logger.error(f"Error training LightGBM model: {e}")
            self.models["lgbm"] = None


        # --- Train Meta-Learner ---
        meta_features_train = pd.DataFrame(index=X_flat.index)

        if self.models["lstm"]:
            lstm_probas = self.models["lstm"].predict(X_seq)
            if len(lstm_probas) == len(y_seq_encoded):
                lstm_conf_values = [lstm_probas[i, y_seq_encoded[i]] for i in range(len(y_seq_encoded))]
                meta_features_train['lstm_conf'] = pd.Series(lstm_conf_values, index=aligned_data.iloc[self.dl_config["sequence_length"]-1:].index)
            else:
                self.logger.warning("LSTM probas length mismatch with y_seq_encoded. Using random uniform for lstm_conf.")
                meta_features_train['lstm_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        else:
            meta_features_train['lstm_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))

        if self.models["transformer"]:
            transformer_probas = self.models["transformer"].predict(X_seq)
            if len(transformer_probas) == len(y_seq_encoded):
                transformer_conf_values = [transformer_probas[i, y_seq_encoded[i]] for i in range(len(y_seq_encoded))]
                meta_features_train['transformer_conf'] = pd.Series(transformer_conf_values, index=aligned_data.iloc[self.dl_config["sequence_length"]-1:].index)
            else:
                self.logger.warning("Transformer probas length mismatch with y_seq_encoded. Using random uniform for transformer_conf.")
                meta_features_train['transformer_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))
        else:
            meta_features_train['transformer_conf'] = np.random.uniform(0.5, 0.9, len(X_flat))

        if self.models["garch"]:
            try:
                historical_volatility = self.models["garch"].conditional_volatility
                aligned_garch_vol = historical_volatility.reindex(aligned_data.index).fillna(method='ffill').fillna(0)
                meta_features_train['garch_volatility'] = aligned_garch_vol
            except Exception as e:
                self.logger.error(f"Error getting historical GARCH volatility for meta-learner training: {e}")
                meta_features_train['garch_volatility'] = np.random.uniform(0.001, 0.01, len(X_flat))
        else:
            meta_features_train['garch_volatility'] = np.random.uniform(0.001, 0.01, len(X_flat))

        if self.models["lgbm"]:
            # Average predictions from all fold models for the meta-feature
            # Prepare flat data for non-DL models (LGBM)
            X_flat_features_list = ['ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'autoencoder_reconstruction_error', 'Is_SR_Interacting']
            X_flat = current_features_row[X_flat_features_list].copy() # Define X_flat here
            X_flat = X_flat.fillna(0) # Fill NaNs for LGBM
            lgbm_probas = np.mean(lgbm_probas_all_folds, axis=0)

            # Get the classes from the first fold model
            lgbm_classes = self.models["lgbm"][0].classes_

            lgbm_conf_values = [lgbm_probas[i, np.where(lgbm_classes == y_flat.iloc[i])[0][0]] for i in range(len(y_flat))]
            meta_features_train['lgbm_proba'] = pd.Series(lgbm_conf_values, index=X_flat.index)

        else:
            meta_features_train['lgbm_proba'] = np.random.uniform(0.5, 0.9, len(X_flat))

        meta_features_train = meta_features_train.join(aligned_data[['is_wyckoff_sos', 'is_wyckoff_sow', 'is_wyckoff_spring', 'is_wyckoff_upthrust',
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

        # --- Monte Carlo Simulation for the trained model ---
        self.logger.info("Running Monte Carlo simulation for the newly trained model...")
        from backtesting.ares_backtester import run_backtest
        from config import CONFIG
        mc_curves, base_portfolio, mc_report = run_monte_carlo_simulation(aligned_data, CONFIG['BEST_PARAMS'])
        self.logger.info(mc_report)

        # Plot the results
        plot_results(mc_curves, base_portfolio)

    def get_prediction(self, current_features: pd.DataFrame, 
                       klines_df: pd.DataFrame, agg_trades_df: pd.DataFrame, 
                       order_book_data: dict, current_price: float) -> dict:
        """
        Gets a combined prediction and confidence score for the BULL_TREND regime.
        """
        if not self.trained:
            self.logger.warning(f"{self.ensemble_name} not trained. Returning HOLD.")
            return {"prediction": "HOLD", "confidence": 0.0}

        self.logger.info(f"Getting prediction from {self.ensemble_name}...")

        # Ensure current_features has all expected features for robust prediction
        for col in self.expected_dl_features:
            if col not in current_features.columns:
                current_features[col] = 0.0 # Add missing columns with default value
        
        current_features_row = current_features.tail(1)
        if current_features_row.empty:
            self.logger.warning("Current features row is empty. Cannot get prediction.")
            return {"prediction": "HOLD", "confidence": 0.0}

        # Prepare sequence for DL models (need sufficient past data)
        # For real-time prediction, `current_features` should be a window of recent data.
        # Here, we pass the tail of `current_features` assuming it contains enough history.
        X_current_seq, _ = self._prepare_sequence_data(current_features.tail(self.dl_config["sequence_length"]), sequence_length=self.dl_config["sequence_length"])
        
        if X_current_seq.size == 0:
             self.logger.warning("Insufficient data for real-time sequence prediction. Returning HOLD.")
             return {"prediction": "HOLD", "confidence": 0.0}

        individual_model_outputs = {}

        # LSTM Prediction
        if self.models["lstm"]:
            try:
                lstm_proba = self.models["lstm"].predict(X_current_seq)[0]
                # Get confidence for the 'BULL_TREND' class
                bull_trend_idx = self.dl_target_map.get('BULL_TREND') # Assuming 'BULL_TREND' is a target class
                if bull_trend_idx is not None and bull_trend_idx < len(lstm_proba):
                    lstm_conf = lstm_proba[bull_trend_idx]
                else: # Fallback if 'BULL_TREND' not in classes or index out of bounds
                    lstm_conf = np.max(lstm_proba) # Use max prob as confidence
                individual_model_outputs['lstm_conf'] = float(lstm_conf) # Ensure float
            except Exception as e:
                self.logger.error(f"Error predicting with LSTM: {e}")
                individual_model_outputs['lstm_conf'] = 0.5 # Neutral
        else:
            individual_model_outputs['lstm_conf'] = 0.5 # Neutral if model not trained

        # Transformer Prediction
        if self.models["transformer"]:
            try:
                transformer_proba = self.models["transformer"].predict(X_current_seq)[0]
                bull_trend_idx = self.dl_target_map.get('BULL_TREND')
                if bull_trend_idx is not None and bull_trend_idx < len(transformer_proba):
                    transformer_conf = transformer_proba[bull_trend_idx]
                else:
                    transformer_conf = np.max(transformer_proba)
                individual_model_outputs['transformer_conf'] = float(transformer_conf)
            except Exception as e:
                self.logger.error(f"Error predicting with Transformer: {e}")
                individual_model_outputs['transformer_conf'] = 0.5
        else:
            individual_model_outputs['transformer_conf'] = 0.5

        # CHANGED: Pass GARCH volatility directly as a feature
        if self.models["garch"]:
            try:
                # GARCH forecast typically provides conditional variance.
                forecast = self.models["garch"].forecast(horizon=1, method='simulation')
                garch_volatility = forecast.variance.iloc[-1].values[0]
                individual_model_outputs['garch_volatility'] = float(garch_volatility)
            except Exception as e:
                self.logger.error(f"Error forecasting with GARCH: {e}")
                individual_model_outputs['garch_volatility'] = 0.005 # Default/neutral volatility
        else:
            individual_model_outputs['garch_volatility'] = 0.005 # Default/neutral volatility

        # LightGBM Prediction
        if self.models["lgbm"]:
            try:
                # Average predictions from all fold models
                lgbm_probas_all_folds = [model.predict_proba(X_flat) for model in self.models["lgbm"]]
                lgbm_proba = np.mean(lgbm_probas_all_folds, axis=0)[0]
                # Get probability for the 'BULL_TREND' class
                bull_idx = np.where(self.models["lgbm"][0].classes_ == 'BULL_TREND')[0]
                lgbm_conf = lgbm_proba[bull_idx][0] if len(bull_idx) > 0 else np.max(lgbm_proba) # Fallback to max prob
                individual_model_outputs['lgbm_proba'] = float(lgbm_conf)
            except Exception as e:
                self.logger.error(f"Error predicting with LightGBM: {e}")
                individual_model_outputs['lgbm_proba'] = 0.5 # Neutral
        else:
            individual_model_outputs['lgbm_proba'] = 0.5 # Neutral if model not trained

        # --- Incorporate Advanced Features ---
        # Ensure all necessary dataframes are passed to feature methods
        # current_features contains the full set of features from Analyst.feature_engineering
        wyckoff_feats = self._get_wyckoff_features(klines_df, current_price)
        manipulation_feats = self._get_manipulation_features(order_book_data, current_price, agg_trades_df)
        order_flow_feats = self._get_order_flow_features(agg_trades_df, order_book_data, klines_df)
        # Assuming klines_df is sufficient for both HTF/MTF for these conceptual features
        multi_timeframe_feats = self._get_multi_timeframe_features(klines_df, klines_df) 

        # Add advanced features to individual_model_outputs for meta-learner
        # Ensure values are floats/ints for meta-learner input
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in wyckoff_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in manipulation_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in order_flow_feats.items()})
        individual_model_outputs.update({k: float(v) if isinstance(v, (int, float, np.number)) else 0.0 for k, v in multi_timeframe_feats.items()})


        # --- Confluence Model (Meta-Learner) ---
        final_prediction = "HOLD"
        final_confidence = 0.0

        if self.meta_learner:
            try:
                # Prepare input for meta-learner: ensure all expected features are present and ordered
                meta_features_for_pred = {}
                for feature_name in self.meta_learner_features: # Use the stored feature names from training
                    meta_features_for_pred[feature_name] = individual_model_outputs.get(feature_name, 0.0) # Default to 0.0 if missing

                # Create a DataFrame from the dictionary, ensuring it's a single row
                meta_input_data = pd.DataFrame([meta_features_for_pred])
                
                # Get prediction from meta-learner
                final_prediction, final_confidence = self._get_meta_prediction(meta_input_data.iloc[0].to_dict())

            except Exception as e:
                self.logger.error(f"Error during meta-learner prediction for {self.ensemble_name}: {e}", exc_info=True)
                final_prediction = "HOLD"
                final_confidence = 0.0
        else:
            self.logger.warning(f"Meta-learner not available for {self.ensemble_name}. Averaging confidences.")
            # Fallback to simple weighted average if meta-learner is not available
            # Note: GARCH volatility is not a confidence, so it's excluded from this average.
            total_weighted_confidence = sum(individual_model_outputs.get(m, 0.5) * self.ensemble_weights.get(m.replace('_conf', ''), 1.0) 
                                            for m in ['lstm_conf', 'transformer_conf', 'lgbm_proba'])
            total_weight = sum(self.ensemble_weights.get(m.replace('_conf', ''), 1.0) for m in ['lstm_conf', 'transformer_conf', 'lgbm_proba'])
            
            if total_weight > 0:
                final_confidence = total_weighted_confidence / total_weight
            
            if final_confidence > self.min_confluence_confidence:
                final_prediction = "BUY" # Default action for BULL_TREND if confidence is high
            else:
                final_prediction = "HOLD"

        self.logger.info(f"Ensemble Result for {self.ensemble_name}: Prediction={final_prediction}, Confidence={final_confidence:.2f}")
        return {"prediction": final_prediction, "confidence": final_confidence}
