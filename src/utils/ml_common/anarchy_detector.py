import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.ensemble import IsolationForest
import iisignature
from src.utils.tprint import tprint_info

class AnarchyDetector:
    """
    Phase 6: Anarchy & Anomaly Layer
    Uses Isolation Forest on 'Physics' features + Path Signatures to detect market chaos.
    """
    def __init__(self, contamination: float = 0.02, n_estimators: int = 200):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=42
        )
        self.is_fit = False

    def get_signature_features(self, price_series: np.ndarray, volume_series: np.ndarray, depth: int = 2) -> np.ndarray:
        if len(price_series) < 2:
            return np.zeros(iisignature.siglength(2, depth))
        path = np.column_stack([price_series, volume_series])
        mean = np.mean(path, axis=0)
        std = np.std(path, axis=0) + 1e-8
        path_scaled = (path - mean) / std
        return iisignature.sig(path_scaled, depth)

    def generate_anarchy_features(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        tprint_info(f"   [Anarchy] Generating Features (IF Inputs + Path Sig depth=2)...")
        vol_intensity = df['volume'] / (df['bar_duration'] + 1e-9)
        returns = df['close'].pct_change().fillna(0).abs()
        vp_efficiency = returns / (df['volume'] + 1e-9)

        sigs = []
        closes = df['close'].values
        vols = df['volume'].values

        # Optimize: Stride if huge data? For now, full compute.
        for i in range(len(df)):
            if i < window:
                sigs.append(np.zeros(6))
                continue
            c_slice = closes[i-window:i]
            v_slice = vols[i-window:i]
            s = self.get_signature_features(c_slice, v_slice, depth=2)
            sigs.append(s)

        sig_df = pd.DataFrame(sigs, index=df.index, columns=[f'sig_{j}' for j in range(6)])

        X = pd.DataFrame(index=df.index)
        X['vol_intensity'] = vol_intensity
        X['vp_efficiency'] = vp_efficiency
        X['roll_vol'] = df['close'].pct_change().rolling(window).std()

        X = pd.concat([X, sig_df], axis=1)
        X = X.ffill().fillna(0)
        return X

    def fit(self, X: pd.DataFrame):
        tprint_info(f"   [Anarchy] Fitting IsolationForest on {len(X)} samples...")
        self.model.fit(X)
        self.is_fit = True
        return self

    def predict_score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.is_fit: raise ValueError("AnarchyDetector not fit")
        raw_score = self.model.decision_function(X)
        is_anarchy = raw_score < -0.2
        meta_feature = (raw_score + 0.5) / 1.0
        meta_feature = np.clip(meta_feature, 0.0, 1.0)
        return raw_score, is_anarchy, meta_feature
