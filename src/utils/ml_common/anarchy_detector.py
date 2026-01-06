import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.ensemble import IsolationForest
import iisignature

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
        """
        Compute Path Signature Level 2 features.
        Returns [dx, dy, dxx, dxy, dyx, dyy] (6 features)
        """
        if len(price_series) < 2:
            return np.zeros(iisignature.siglength(2, depth))

        # 1. Path Construction (Lead-Lag or just 2D)
        # We combine Price and Volume into a 2D path
        path = np.column_stack([price_series, volume_series])

        # 2. Scale
        # prevent explosion
        mean = np.mean(path, axis=0)
        std = np.std(path, axis=0) + 1e-8
        path_scaled = (path - mean) / std

        # 3. Signature
        sig = iisignature.sig(path_scaled, depth)

        return sig

    def generate_anarchy_features(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        """
        Generate input vector for Isolation Forest.
        X_IF = [Vol-Price Eff, Temporal Intensity, Wavelet Entropy, Path Sig L2]
        """
        # Physics basics
        vol_intensity = df['volume'] / (df['bar_duration'] + 1e-9)
        returns = df['close'].pct_change().fillna(0).abs()
        vp_efficiency = returns / (df['volume'] + 1e-9)

        # Path Signatures (Rolling)
        # Slow loop again.
        # For training, we compute all.
        sigs = []
        closes = df['close'].values
        vols = df['volume'].values

        # Stride optimization: calc every bar is fine for offline training
        # For 'window' lookback
        for i in range(len(df)):
            if i < window:
                sigs.append(np.zeros(6)) # Depth 2 dim 2 -> 6 coeffs
                continue

            c_slice = closes[i-window:i]
            v_slice = vols[i-window:i]
            s = self.get_signature_features(c_slice, v_slice, depth=2)
            sigs.append(s)

        sig_df = pd.DataFrame(sigs, index=df.index, columns=[f'sig_{j}' for j in range(6)])

        X = pd.DataFrame(index=df.index)
        X['vol_intensity'] = vol_intensity
        X['vp_efficiency'] = vp_efficiency
        # Add basic rolling volatility as proxy for entropy if not available
        X['roll_vol'] = df['close'].pct_change().rolling(window).std()

        X = pd.concat([X, sig_df], axis=1)
        X = X.ffill().fillna(0)

        return X

    def fit(self, X: pd.DataFrame):
        self.model.fit(X)
        self.is_fit = True
        return self

    def predict_score(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
        - raw_score: Decision function
        - is_anarchy: Hard veto boolean (True if anarchy)
        - meta_feature: 0-1 soft feature (1=Normal, 0=Chaos)
        """
        if not self.is_fit:
            raise ValueError("AnarchyDetector not fit")

        raw_score = self.model.decision_function(X)
        # IF decision_function: lower = more abnormal. Negative = outlier.

        # Hard Veto (Threshold < -0.2 as per prompt)
        is_anarchy = raw_score < -0.2

        # Soft Input (0=Chaos, 1=Normal)
        # raw usually [-0.5, 0.5]
        meta_feature = (raw_score + 0.5) / 1.0
        meta_feature = np.clip(meta_feature, 0.0, 1.0)

        return raw_score, is_anarchy, meta_feature
