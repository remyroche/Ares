import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from src.utils.labeling_optimized import batch_mi_score_numba

def calculate_residual_mi_optimized(feature_df: pd.DataFrame, target_series: pd.Series,
                          lag: int = 1, n_neighbors: int = 3,
                          subsample_size: int = 50000, n_bins: int = 5) -> pd.Series:
    """
    Calculates the Residual Mutual Information (RMI) proxy for a set of features.
    Optimized version using binned MI with Numba.

    Args:
        feature_df: DataFrame of features (e.g., composite candidates).
        target_series: The target (e.g., 1-bar forward returns).
        lag: Number of lags of the target to use for residualization.
        subsample_size: Max samples for efficiency.
        n_bins: Number of bins for discretization.

    Returns:
        Series of RMI scores sorted descending.
    """
    # 1. Prepare Target Residuals (The 'Innovation')
    y = target_series.values.reshape(-1, 1)

    # Create lagged target matrix
    y_lags = np.hstack([target_series.shift(i).values.reshape(-1, 1) for i in range(1, lag + 1)])

    # Valid indices (drop NaNs from shifting)
    valid_idx = ~np.isnan(y_lags).any(axis=1)
    y_clean = y[valid_idx]
    y_lags_clean = y_lags[valid_idx]

    if len(y_clean) == 0:
        return pd.Series(0.0, index=feature_df.columns)

    # Fit AR model and get residuals
    model = LinearRegression()
    model.fit(y_lags_clean, y_clean)
    y_pred = model.predict(y_lags_clean)
    residuals = (y_clean - y_pred).flatten()

    # 2. Align Features with Residuals
    # feature_df might have NaNs? Assuming filled or handled.
    X = feature_df.iloc[valid_idx].fillna(0.0).values

    # 3. Subsample if too large (efficiency)
    if len(residuals) > subsample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(residuals), size=subsample_size, replace=False)
        X = X[idx]
        residuals = residuals[idx]

    # 4. Discretize Residuals (Target)
    # Use quantile binning for target residuals
    try:
        # qcut requires Series/DataFrame? No, numpy/pandas
        res_series = pd.Series(residuals)
        target_binned = pd.qcut(res_series, n_bins, labels=False, duplicates='drop').fillna(-1).astype(int).values
    except Exception:
        # Fallback to linear binning
        res_series = pd.Series(residuals)
        target_binned = pd.cut(res_series, n_bins, labels=False).fillna(-1).astype(int).values

    # 5. Calculate Mutual Information using Numba Batch
    # X columns are continuous, batch_mi_score_numba discretizes them internally using quantiles
    mi_scores = batch_mi_score_numba(X.astype(np.float64), target_binned, n_bins=n_bins)

    return pd.Series(mi_scores, index=feature_df.columns).sort_values(ascending=False)
