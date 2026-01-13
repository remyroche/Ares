import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.ensemble import IsolationForest
try:
    import iisignature
except ImportError:
    iisignature = None
from src.utils.tprint import tprint_info, tprint_warning

class AnarchyDetector:
    """
    Phase 6: Anarchy & Anomaly Layer
    Uses Isolation Forest on 'Physics' features + Path Signatures to detect market chaos.
    """
    def __init__(self, contamination: float = 0.02, n_estimators: int = 200):
        import multiprocessing
        n_cpus = multiprocessing.cpu_count()
        default_n_jobs = min(n_cpus, 4) if n_cpus > 4 else max(1, n_cpus - 1)
        
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_features=1.0,
            bootstrap=False,
            n_jobs=default_n_jobs,
            random_state=42
        )
        self.is_fit = False

    def get_signature_features(self, price_series: np.ndarray, volume_series: np.ndarray, depth: int = 2) -> np.ndarray:
        if iisignature is None:
            # Fallback if iisignature is not available
            # Return zeros of appropriate length (approximation for depth 2)
            # Depth 2 signature length is (d^(2+1) - 1)/(d-1) - 1 ? No, for dimension d=2
            # siglength(d, m) is the length.
            # For d=2, m=2: length is 6 (1st level: 2, 2nd level: 4)
            return np.zeros(6)

        if len(price_series) < 2:
            return np.zeros(iisignature.siglength(2, depth))
        path = np.column_stack([price_series, volume_series])
        mean = np.mean(path, axis=0)
        std = np.std(path, axis=0) + 1e-8
        path_scaled = (path - mean) / std
        return iisignature.sig(path_scaled, depth)

    def generate_anarchy_features(self, df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
        tprint_info(f"   [Anarchy] Generating Features (IF Inputs + Path Sig depth=2)...")
        
        # Calculate bar_duration if missing
        if 'bar_duration' not in df.columns:
            if isinstance(df.index, pd.DatetimeIndex):
                durations = df.index.to_series().diff().dt.total_seconds().fillna(method='bfill').fillna(60.0)
                vol_intensity = df['volume'] / (durations + 1e-9)
            else:
                vol_intensity = df['volume'] / (60.0 + 1e-9) # Fallback to 1m
        else:
            vol_intensity = df['volume'] / (df['bar_duration'] + 1e-9)
            
        returns = df['close'].pct_change().fillna(0).abs()
        vp_efficiency = returns / (df['volume'] + 1e-9)

        sigs = np.zeros((len(df), 6))
        closes = df['close'].values
        vols = df['volume'].values

        # Optimization: Only compute every 5th row and interpolate
        # Reducing calls to iisignature.sig to reduce CPU load
        # User requested step=5 (balanced with lower bar density)
        step = 5
        for i in range(window, len(df), step):
            c_slice = closes[i-window:i]
            v_slice = vols[i-window:i]
            s = self.get_signature_features(c_slice, v_slice, depth=2)
            sigs[i] = s

        sig_df = pd.DataFrame(sigs, index=df.index, columns=[f'sig_{j}' for j in range(6)])
        sig_df.loc[sig_df.index[window::step], 'is_sampled'] = True
        sig_df[[f'sig_{j}' for j in range(6)]] = sig_df[[f'sig_{j}' for j in range(6)]].where(sig_df['is_sampled'] == True, np.nan).ffill().fillna(0)
        sig_df = sig_df.drop(columns=['is_sampled'])

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
