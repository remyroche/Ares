import logging
import warnings
from arch import arch_model
from keras.layers import LSTM, Dense, Dropout, Flatten, Input, LayerNormalization, MultiHeadAttention
from keras.models import Model
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier
from src.utils.warning_symbols import failed
from .base_ensemble import BaseEnsemble
import numpy as np
import pandas as pd
import typing
from typing import Dict, Any, List

# Add tprint imports for enhanced logging
from src.utils.tprint import tprint, tprint_debug, tprint_info, tprint_warning, tprint_error, tprint_success, tprint_progress, tprint_performance, tprint_timer

# Import meta-feature generator
from src.feature_engineering_roadmap.ensemble_meta_features import EnsembleMetaFeatureGenerator

# VectorBT imports for native optimization
try:
    import vectorbt as vbt
    from vectorbt.generic import rolling_mean, rolling_std, rolling_var, rolling_min, rolling_max, rolling_sum, rolling_apply, rolling_corr, rolling_cov
    from vectorbt.generic import scale, rank, zscore, winsorize, clip, quantile
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    vbt = None
    rolling_mean = None
    rolling_std = None
    rolling_var = None
    rolling_min = None
    rolling_max = None
    rolling_sum = None
    rolling_apply = None
    rolling_corr = None
    rolling_cov = None
    scale = None
    rank = None
    zscore = None
    winsorize = None
    clip = None
    quantile = None
    warnings.warn("VectorBT not available. Install with: pip install vectorbt for optimized performance")

except ImportError:
    
    cp = None

class VolatileRegimeEnsemble(BaseEnsemble):
    """
    This ensemble specializes in detecting and predicting during volatile market conditions.
    It combines signals from multiple models optimized for high volatility periods.
    """

    def __init__(self, config: dict, ensemble_name: str='VolatileRegimeEnsemble') -> None:
        tprint(f"🚀 [VOLATILE_REGIME] Initializing VolatileRegimeEnsemble: {ensemble_name}", color="cyan", bold=True)
        self.logger = logging.getLogger(self.__class__.__name__)
        super().__init__(config, ensemble_name)
        self.dl_config = {'sequence_length': 20, 'lstm_units': 50, 'transformer_heads': 2, 'transformer_key_dim': 32, 'dropout_rate': 0.2, 'epochs': 50, 'batch_size': 32}
        self.models = {'lstm': None, 'transformer': None, 'garch': None, 'tabnet': None, 'order_flow_lgbm': None, 'logistic_regression': None}
        
        # Initialize meta-feature generator
        self.meta_feature_generator = EnsembleMetaFeatureGenerator(self.logger)
        
        tprint("✅ [VOLATILE_REGIME] VolatileRegimeEnsemble initialized successfully", color="green")

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray) -> None:
        """Trains multiple diverse base models for volatile regime detection."""
        tprint("🏋️ [VOLATILE_REGIME] Training VolatileRegime base models", color="yellow")
        self.logger.info('Training VolatileRegime base models...')
        X_seq, y_seq_aligned_encoded = self._prepare_sequence_data(aligned_data, pd.Series(y_encoded, index = aligned_data.index))
        num_classes = len(np.unique(y_encoded))
        if X_seq.size > 0:
            self.models['lstm'] = self._train_dl_model(X_seq, y_seq_aligned_encoded, num_classes, is_transformer = False)
            self.models['transformer'] = self._train_dl_model(X_seq, y_seq_aligned_encoded, num_classes, is_transformer = True)
        X_flat = aligned_data[self.flat_features].fillna(0)
        self.models['tabnet'] = self._train_tabnet_model(X_flat, y_encoded)
        self.logger.info('Tuning and training specialized Order Flow LGBM...')
        X_of = aligned_data[self.order_flow_features].fillna(0)
        of_params = self._tune_hyperparameters(LGBMClassifier, self._get_lgbm_search_space, X_of, y_encoded)
        self.models['order_flow_lgbm'] = self._train_with_smote(LGBMClassifier(**of_params, random_state = 42, verbose=-1), X_of, y_encoded)
        self.logger.info('Training Logistic Regression model with L1-L2 regularization...')
        self.models['logistic_regression'] = self._train_with_smote(self._get_regularized_logistic_regression(), X_flat, y_encoded)
        try:
            self.logger.info('Training GARCH model for volatility modeling...')
            self.models['garch'] = self._train_garch_model(aligned_data, y_encoded)
        except Exception:
            self.logger.exception(failed('GARCH training failed: {e}'))
        self.logger.info('✅ VolatileRegime base models training completed')

    def _prepare_sequence_data(self, df: pd.DataFrame, target_series: pd.Series = None) -> None:
        """Prepare sequence data for deep learning models."""
        try:
            sequence_length = self.dl_config['sequence_length']
            feature_cols = self.flat_features + self.order_flow_features
            X = df[feature_cols].fillna(0).values
            X_seq = []
            y_seq = []
            for i in range(sequence_length, len(X)):
                X_seq.append(X[i - sequence_length:i])
                if target_series is not None:
                    y_seq.append(target_series.iloc[i])
            if len(X_seq) > 0:
                return (np.array(X_seq), np.array(y_seq))
            return (np.array([]), np.array([]))
        except Exception:
            self.logger.exception('Error preparing sequence data: {e}')
            return (np.array([]), np.array([]))

    def _train_dl_model(self, X_seq: Any, y_seq_encoded: Any, num_classes: List[Any], is_transformer: bool = False) -> None:
        """Train deep learning model (LSTM or Transformer)."""
        try:
            if len(X_seq) == 0:
                return None
            input_shape = (X_seq.shape[1], X_seq.shape[2])
            if is_transformer:
                return self._build_transformer_model(input_shape, num_classes, X_seq, y_seq_encoded)
            return self._build_lstm_model(input_shape, num_classes, X_seq, y_seq_encoded)
        except Exception:
            self.logger.exception('Error training DL model: {e}')
            return None

    def _build_lstm_model(self, input_shape: Any, num_classes: List[Any], X_seq: Any, y_seq_encoded: Any) -> None:
        """Build LSTM model."""
        try:
            inputs = Input(shape = input_shape)
            x = LSTM(self.dl_config['lstm_units'], return_sequences = True)(inputs)
            x = Dropout(self.dl_config['dropout_rate'])(x)
            x = LSTM(self.dl_config['lstm_units'] // 2)(x)
            x = Dropout(self.dl_config['dropout_rate'])(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(self.dl_config['dropout_rate'])(x)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs = inputs, outputs = outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_seq, y_seq_encoded, epochs = self.dl_config['epochs'], batch_size = self.dl_config['batch_size'], validation_split = 0.2, verbose = 0)
            return model
        except Exception:
            self.logger.exception('Error building LSTM model: {e}')
            return None

    def _build_transformer_model(self, input_shape: Any, num_classes: List[Any], X_seq: Any, y_seq_encoded: Any) -> None:
        """Build Transformer model."""
        try:
            inputs = Input(shape = input_shape)
            x = MultiHeadAttention(num_heads = self.dl_config['transformer_heads'], key_dim = self.dl_config['transformer_key_dim'])(inputs, inputs)
            x = LayerNormalization()(x)
            x = Dropout(self.dl_config['dropout_rate'])(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(self.dl_config['dropout_rate'])(x)
            x = Dense(input_shape[1])(x)
            x = LayerNormalization()(x)
            x = Flatten()(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(self.dl_config['dropout_rate'])(x)
            x = Dense(32, activation='relu')(x)
            outputs = Dense(num_classes, activation='softmax')(x)
            model = Model(inputs = inputs, outputs = outputs)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_seq, y_seq_encoded, epochs = self.dl_config['epochs'], batch_size = self.dl_config['batch_size'], validation_split = 0.2, verbose = 0)
            return model
        except Exception:
            self.logger.exception('Error building Transformer model: {e}')
            return None

    def _train_tabnet_model(self, X_flat: Any, y_flat_encoded: Any) -> None:
        """Train TabNet model."""
        try:
            tabnet = TabNetClassifier()
            tabnet.fit(X_flat.values, y_flat_encoded, max_epochs = 50, patience = 20, batch_size = 1024)
            return tabnet
        except Exception:
            self.logger.exception(failed('TabNet training failed: {e}'))
            return None

    def _train_garch_model(self, aligned_data: Any, y_encoded: Any) -> None:
        """Train GARCH model for volatility modeling."""
        try:
            returns = aligned_data['close'].pct_change().dropna()
            garch_model = arch_model(returns, vol='GARCH', p = 1, q = 1)
            return garch_model.fit(disp='off')
        except Exception:
            self.logger.exception(failed('GARCH model training failed: {e}'))
            return None

    def _generate_meta_features(self, aligned_data: pd.DataFrame) -> pd.DataFrame:
        """Generate meta-features specific to volatile regime detection."""
        meta_features = pd.DataFrame(index = aligned_data.index)
        if 'volatility_20' in aligned_data.columns:
            meta_features['volatility_percentile'] = aligned_data['volatility_20'].rolling(100).rank(pct = True)
            meta_features['volatility_acceleration'] = aligned_data['volatility_20'].diff()
            meta_features['volatility_momentum'] = aligned_data['volatility_20'] - aligned_data['volatility_20'].shift(5)
        if 'volume' in aligned_data.columns:
            meta_features['volume_volatility'] = aligned_data['volume'].rolling(20).std()
            meta_features['volume_volatility_ratio'] = meta_features['volume_volatility'] / aligned_data['volume'].rolling(20).mean()
        if 'close' in aligned_data.columns:
            meta_features['price_volatility'] = aligned_data['close'].pct_change().rolling(20).std()
            meta_features['price_volatility_percentile'] = meta_features['price_volatility'].rolling(100).rank(pct = True)
        if 'volatility_regime' in aligned_data.columns:
            meta_features['volatility_regime_numeric'] = aligned_data['volatility_regime']
        return meta_features.fillna(0)

    def predict(self, current_features: pd.DataFrame) -> tuple[float, float]:
        """Make prediction for volatile regime."""
        if not self.trained:
            self.logger.warning('VolatileRegime ensemble not trained')
            return (0.5, 0.5)
        try:
            base_predictions = self._get_base_model_predictions(current_features, is_live = True)
            if not base_predictions:
                return (0.5, 0.5)
            predictions = list(base_predictions.values())
            confidences = [0.8] * len(predictions)
            weighted_pred = np.average(predictions, weights = confidences)
            ensemble_confidence = np.mean(confidences)
            return (weighted_pred, ensemble_confidence)
        except Exception as e:
            self.logger.exception(f'Error in VolatileRegime prediction: {e}')
            return (0.5, 0.5)
    
    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs: Any) -> pd.DataFrame:
        """
        Extract comprehensive meta-features including disagreement features for the ensemble.
        
        Args:
            df: Input DataFrame with features
            is_live: Whether this is for live trading or backtesting
            **kwargs: Additional keyword arguments
            
        Returns:
            DataFrame containing meta-features including disagreement features
        """
        try:
            tprint(f"🔍 [VOLATILE_REGIME] Generating meta-features for {self.ensemble_name}", color="cyan")
            
            # Get base model predictions for disagreement analysis
            base_predictions = self._get_base_model_predictions(df, is_live=is_live)
            
            # Use the meta-feature generator from feature engineering
            meta_features = self.meta_feature_generator.generate_meta_features_for_volatile_regime_ensemble(
                df, base_predictions, is_live
            )
            
            tprint(f"✅ [VOLATILE_REGIME] Generated {len(meta_features.columns)} meta-features", color="green")
            return meta_features
            
        except Exception as e:
            self.logger.error(f"Error generating meta-features for {self.ensemble_name}: {e}")
            # Return basic meta-features as fallback
            try:
                return self._generate_meta_features(df)
            except Exception as fallback_error:
                self.logger.error(f"Fallback meta-feature generation also failed: {fallback_error}")
                return pd.DataFrame(index=df.index)
    
    def _get_base_model_predictions(self, df: pd.DataFrame, is_live: bool = False) -> Dict[str, Any]:
        """
        Get predictions from all base models for disagreement analysis.
        
        Args:
            df: Input DataFrame with features
            is_live: Whether this is for live trading or backtesting
            
        Returns:
            Dict containing model predictions and probabilities
        """
        try:
            base_predictions = {}
            
            # Get predictions from each trained model
            for model_name, model in self.models.items():
                if model is None:
                    continue
                    
                try:
                    if model_name in ['lstm', 'transformer']:
                        # Handle deep learning models
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(df[self.flat_features + self.order_flow_features].fillna(0).values)
                            prediction = np.argmax(proba, axis=1)[0] if len(proba) > 0 else 0.5
                            confidence = np.max(proba, axis=1)[0] if len(proba) > 0 else 0.5
                        else:
                            prediction = 0.5
                            confidence = 0.5
                    elif model_name == 'garch':
                        # Handle GARCH model
                        if hasattr(model, 'forecast'):
                            try:
                                forecast = model.forecast(horizon=1)
                                prediction = float(forecast.mean.iloc[-1, 0]) if hasattr(forecast, 'mean') else 0.5
                                confidence = 0.7  # GARCH models typically have moderate confidence
                            except:
                                prediction = 0.5
                                confidence = 0.5
                        else:
                            prediction = 0.5
                            confidence = 0.5
                    elif model_name == 'tabnet':
                        # Handle TabNet model
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(df[self.flat_features].fillna(0).values)
                            prediction = np.argmax(proba, axis=1)[0] if len(proba) > 0 else 0.5
                            confidence = np.max(proba, axis=1)[0] if len(proba) > 0 else 0.5
                        else:
                            prediction = 0.5
                            confidence = 0.5
                    else:
                        # Handle other models (LGBM, Logistic Regression)
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(df[self.flat_features].fillna(0).values)
                            prediction = np.argmax(proba, axis=1)[0] if len(proba) > 0 else 0.5
                            confidence = np.max(proba, axis=1)[0] if len(proba) > 0 else 0.5
                        else:
                            prediction = 0.5
                            confidence = 0.5
                    
                    base_predictions[model_name] = {
                        'prediction': float(prediction),
                        'probability': float(prediction),  # Use prediction as probability for simplicity
                        'confidence': float(confidence)
                    }
                    
                except Exception as model_error:
                    self.logger.warning(f"Error getting prediction from {model_name}: {model_error}")
                    base_predictions[model_name] = {
                        'prediction': 0.5,
                        'probability': 0.5,
                        'confidence': 0.0
                    }
            
            return base_predictions
            
        except Exception as e:
            self.logger.error(f"Error getting base model predictions: {e}")
            return {}

    def _should_use_vectorbt(self, data) -> bool:
        """Determine if VectorBT should be used based on data size and configuration."""
        return (hasattr(self, 'use_vectorbt') and getattr(self, 'use_vectorbt', True) and 
                len(data) >= getattr(self, 'vectorbt_threshold', 1000) and 
                VECTORBT_AVAILABLE)
    
    def _vectorbt_rolling_operation(self, data: pd.Series, operation: str, 
                                  window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
        
        try:
            if operation == 'mean':
                return rolling_mean(data, window=window, **kwargs)
            elif operation == 'std':
                return rolling_std(data, window=window, **kwargs)
            elif operation == 'var':
                return rolling_var(data, window=window, **kwargs)
            elif operation == 'min':
                return rolling_min(data, window=window, **kwargs)
            elif operation == 'max':
                return rolling_max(data, window=window, **kwargs)
            elif operation == 'sum':
                return rolling_sum(data, window=window, **kwargs)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
        except Exception as e:
            logger.warning(f"VectorBT operation failed: {e}, using pandas fallback")
            return self._pandas_rolling_operation(data, operation, window, **kwargs)
    
    def _pandas_rolling_operation(self, data: pd.Series, operation: str, 
                                 window: int, **kwargs) -> pd.Series:
        """Fallback rolling operation using pandas."""
        if operation == 'mean':
            return data.rolling(window=window).mean()
        elif operation == 'std':
            return data.rolling(window=window).std()
        elif operation == 'var':
            return data.rolling(window=window).var()
        elif operation == 'min':
            return data.rolling(window=window).min()
        elif operation == 'max':
            return data.rolling(window=window).max()
        elif operation == 'sum':
            return data.rolling(window=window).sum()
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    
    def _vectorbt_apply_operation(self, data: pd.Series, func, 
                                 window: int, **kwargs) -> pd.Series:
        """Perform VectorBT rolling apply operation with fallback to pandas."""
        if not self._should_use_vectorbt(data):
            return data.rolling(window=window).apply(func, **kwargs)
        
        try:
            return rolling_apply(data, func, window=window, **kwargs)
        except Exception as e:
            logger.warning(f"VectorBT rolling apply failed: {e}, using pandas fallback")
            return data.rolling(window=window).apply(func, **kwargs)
