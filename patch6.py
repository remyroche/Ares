import re

with open("extreme_price_movements/simple_position_sizer.py", "r") as f:
    content = f.read()

# 1. Update imports
content = content.replace("from sklearn.linear_model import Ridge", "from sklearn.linear_model import Ridge, HuberRegressor")
content = content.replace("from sklearn.preprocessing import StandardScaler", "from sklearn.preprocessing import StandardScaler, RobustScaler")

# 2. Update clean_and_standardize to use RobustScaler and robust 1D scaling
old_clean = """def clean_and_standardize(X: np.ndarray, fit_medians: Optional[np.ndarray] = None, scaler: Optional[StandardScaler] = None, mean_1d: Optional[float] = None, std_1d: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Any, Any, Any]:
    \"\"\"Standardizes features safely handling NaNs and Infs.\"\"\"
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan

    if fit_medians is None:
        fit_medians = np.nanmedian(X_clean, axis=0)
        if np.isscalar(fit_medians):
            if np.isnan(fit_medians):
                fit_medians = 0.0
        else:
            fit_medians[np.isnan(fit_medians)] = 0.0

    if X_clean.ndim == 1:
        inds = np.isnan(X_clean)
        X_clean[inds] = fit_medians

        if mean_1d is None or std_1d is None:
            mean_1d = np.mean(X_clean)
            std_1d = np.std(X_clean)

        if std_1d > 1e-9:
            X_clean = (X_clean - mean_1d) / std_1d
        else:
            X_clean = X_clean - mean_1d
    else:
        inds = np.where(np.isnan(X_clean))
        X_clean[inds] = np.take(fit_medians, inds[1])

        if scaler is None:
            scaler = StandardScaler()
            X_clean = scaler.fit_transform(X_clean)
        else:
            X_clean = scaler.transform(X_clean)

    return X_clean, fit_medians, scaler, mean_1d, std_1d"""

new_clean = """def clean_and_standardize(X: np.ndarray, fit_medians: Optional[np.ndarray] = None, scaler: Optional[RobustScaler] = None, center_1d: Optional[float] = None, scale_1d: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Any, Any, Any]:
    \"\"\"Standardizes features safely handling NaNs and Infs, using robust statistics.\"\"\"
    X_clean = X.copy()
    X_clean[np.isinf(X_clean)] = np.nan

    if fit_medians is None:
        fit_medians = np.nanmedian(X_clean, axis=0)
        if np.isscalar(fit_medians):
            if np.isnan(fit_medians):
                fit_medians = 0.0
        else:
            fit_medians[np.isnan(fit_medians)] = 0.0

    if X_clean.ndim == 1:
        inds = np.isnan(X_clean)
        X_clean[inds] = fit_medians

        if center_1d is None or scale_1d is None:
            center_1d = np.median(X_clean)
            q75, q25 = np.percentile(X_clean, [75 ,25])
            scale_1d = q75 - q25

        if scale_1d > 1e-9:
            X_clean = (X_clean - center_1d) / scale_1d
        else:
            X_clean = X_clean - center_1d
    else:
        inds = np.where(np.isnan(X_clean))
        X_clean[inds] = np.take(fit_medians, inds[1])

        if scaler is None:
            scaler = RobustScaler()
            X_clean = scaler.fit_transform(X_clean)
        else:
            X_clean = scaler.transform(X_clean)

    return X_clean, fit_medians, scaler, center_1d, scale_1d"""

content = content.replace(old_clean, new_clean)

# 3. Update SimpleHeadRidgeSizer to use HuberRegressor
old_model = """        self.alpha = alpha
        self.model = Ridge(alpha=alpha, fit_intercept=True)"""

new_model = """        self.alpha = alpha
        self.model = HuberRegressor(alpha=alpha, fit_intercept=True)"""

content = content.replace(old_model, new_model)

with open("extreme_price_movements/simple_position_sizer.py", "w") as f:
    f.write(content)
