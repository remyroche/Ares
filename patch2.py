import re

with open("extreme_price_movements/simple_position_sizer.py", "r") as f:
    content = f.read()

# Replace clean_and_standardize completely using regex
pattern = re.compile(r"def clean_and_standardize.*?return X_clean, fit_medians", re.DOTALL)

new_func = """def clean_and_standardize(X: np.ndarray, fit_medians: Optional[np.ndarray] = None, scaler: Optional[StandardScaler] = None, mean_1d: Optional[float] = None, std_1d: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, Any, Any, Any]:
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

content = pattern.sub(new_func, content)

with open("extreme_price_movements/simple_position_sizer.py", "w") as f:
    f.write(content)
