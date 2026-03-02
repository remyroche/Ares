import numpy as np


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def smooth_utility_from_mfe_mae(mfe, mae, tp: float, sl: float, alpha: float):
    mfe = np.asarray(mfe, dtype=float)
    mae = np.asarray(mae, dtype=float)
    a = float(max(alpha, 1e-8))
    tp_v = float(tp)
    sl_v = float(sl)
    return tp_v * sigmoid(a * (mfe - tp_v)) - sl_v * sigmoid(a * (mae - sl_v))


def smooth_utility_from_log_heads(log_mfe, log_mae, tp: float, sl: float, alpha: float):
    mfe = np.clip(np.expm1(np.asarray(log_mfe, dtype=float)), 0.0, None)
    mae = np.clip(np.expm1(np.asarray(log_mae, dtype=float)), 0.0, None)
    return smooth_utility_from_mfe_mae(mfe=mfe, mae=mae, tp=tp, sl=sl, alpha=alpha)


def smooth_utility_from_log_heads_standardized(
    log_mfe,
    log_mae,
    tp: float,
    sl: float,
    alpha: float,
    mfe_mean: float,
    mfe_std: float,
    mae_mean: float,
    mae_std: float,
):
    """Utility map from z-scored log heads with thresholds converted to z-space."""
    log_mfe = np.asarray(log_mfe, dtype=float)
    log_mae = np.asarray(log_mae, dtype=float)
    _mfe_std = max(float(mfe_std), 1e-9)
    _mae_std = max(float(mae_std), 1e-9)
    z_mfe = (log_mfe - float(mfe_mean)) / _mfe_std
    z_mae = (log_mae - float(mae_mean)) / _mae_std
    z_tp = (np.log1p(max(float(tp), 0.0)) - float(mfe_mean)) / _mfe_std
    z_sl = (np.log1p(max(float(sl), 0.0)) - float(mae_mean)) / _mae_std
    a = float(max(alpha, 1e-8))
    return float(tp) * sigmoid(a * (z_mfe - z_tp)) - float(sl) * sigmoid(a * (z_mae - z_sl))


def smooth_utility_loss(y_hat, y_true, loss: str = "huber", delta: float = 1.0):
    err = np.asarray(y_hat, dtype=float) - np.asarray(y_true, dtype=float)
    if loss == "mse":
        return float(np.mean(err ** 2))
    abs_err = np.abs(err)
    quad = np.minimum(abs_err, delta)
    lin = abs_err - quad
    return float(np.mean(0.5 * quad ** 2 + delta * lin))
