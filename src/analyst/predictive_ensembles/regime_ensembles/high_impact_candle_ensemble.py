import logging
import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

from .base_ensemble import BaseEnsemble

class HighImpactCandleEnsemble(BaseEnsemble):
    """
    ## CHANGE: Refactored to inherit the optimized training pipeline from BaseEnsemble.
    ## This class now focuses only on the base models (including TabNet) and meta-features
    ## specific to predicting the outcome of high-impact candle events.
    """

    def __init__(self, config: dict, ensemble_name: str = "HighImpactCandleEnsemble"):
        super().__init__(config, ensemble_name)
        self.flat_feature_subset = [
            'ADX', 'MACD_HIST', 'ATR', 'volume_delta', 'rsi', 'stoch_k',
            'bb_bandwidth', 'position_in_range', 'cvd_slope'
        ]

    def _train_base_models(self, aligned_data: pd.DataFrame, y_encoded: np.ndarray):
        """Trains base models for high-impact candles."""
        self.logger.info("Training HighImpactCandle base models...")
        
        # Ensure all features are present
        for col in self.flat_feature_subset:
            if col not in aligned_data.columns:
                aligned_data[col] = 0.0
        
        X_flat = aligned_data[self.flat_feature_subset].fillna(0)

        # TabNet Model
        try:
            self.models["tabnet"] = self._train_tabnet_model(X_flat, y_encoded)
        except Exception as e:
            self.logger.error(f"TabNet training failed: {e}")
        
        # Order Flow Model (LGBM)
        try:
            of_features = ['volume', 'volume_delta', 'cvd_slope']
            for col in of_features:
                if col not in aligned_data.columns:
                    aligned_data[col] = 0
            X_of = aligned_data[of_features].fillna(0)
            self.models["order_flow_lgbm"] = LGBMClassifier(random_state=42, verbose=-1).fit(X_of, y_encoded)
        except Exception as e:
            self.logger.error(f"Order Flow LGBM training failed: {e}")

    def _get_meta_features(self, df: pd.DataFrame, is_live: bool = False, **kwargs) -> pd.DataFrame or dict:
        """Generates meta-features for the high-impact candle meta-learner."""
        
        # Ensure all features are present
        for col in self.flat_feature_subset:
            if col not in df.columns:
                df[col] = 0.0
        
        if is_live:
            meta_features = {}
            current_row = df.tail(1)
            X_flat = current_row[self.flat_feature_subset].fillna(0)

            if self.models["tabnet"]:
                meta_features['tabnet_prob'] = np.max(self.models["tabnet"].predict_proba(X_flat.values))
            if self.models["order_flow_lgbm"]:
                X_of = current_row[['volume', 'volume_delta', 'cvd_slope']].fillna(0)
                meta_features['order_flow_prob'] = self.models["order_flow_lgbm"].predict_proba(X_of)[0][1]
            
            # Rule-based feature
            meta_features['volume_imbalance_signal'] = np.divide(current_row['volume_delta'].iloc[-1], current_row['volume'].iloc[-1]) if current_row['volume'].iloc[-1] != 0 else 0
            return meta_features
        else:
            meta_df = pd.DataFrame(index=df.index)
            X_flat = df[self.flat_feature_subset].fillna(0)

            if self.models["tabnet"]:
                meta_df['tabnet_prob'] = np.max(self.models["tabnet"].predict_proba(X_flat.values), axis=1)
            if self.models["order_flow_lgbm"]:
                X_of = df[['volume', 'volume_delta', 'cvd_slope']].fillna(0)
                meta_df['order_flow_prob'] = self.models["order_flow_lgbm"].predict_proba(X_of)[:, 1]
            
            meta_df['volume_imbalance_signal'] = np.divide(df['volume_delta'], df['volume'], where=df['volume']!=0, out=np.zeros_like(df['volume_delta'], dtype=float))
            return meta_df.fillna(0)
            
    def _train_tabnet_model(self, X_flat, y_flat_encoded):
        try:
            model = TabNetClassifier()
            model.fit(X_flat.values, y_flat_encoded, max_epochs=50, patience=20, batch_size=1024, virtual_batch_size=128)
            return model
        except Exception as e:
            self.logger.error(f"TabNet training failed: {e}")
            return None
