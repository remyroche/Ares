from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score


class SoftLabelLogisticModel:
    """Simple logistic model trained with BCE on soft labels in [0,1]."""

    def __init__(self, lr=0.1, n_iter=800, l2=1e-4):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.l2 = float(l2)
        self.coef_ = None
        self.intercept_ = 0.0

    @staticmethod
    def _sigmoid(z):
        z = np.clip(z, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y_soft):
        X = np.asarray(X, dtype=float)
        y = np.clip(np.asarray(y_soft, dtype=float), 0.0, 1.0)
        n, d = X.shape
        w = np.zeros(d, dtype=float)
        b = 0.0

        for _ in range(self.n_iter):
            z = X @ w + b
            p = self._sigmoid(z)
            err = p - y
            grad_w = (X.T @ err) / max(n, 1) + self.l2 * w
            grad_b = float(np.mean(err))
            w -= self.lr * grad_w
            b -= self.lr * grad_b

        self.coef_ = w
        self.intercept_ = b
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        p = self._sigmoid(X @ self.coef_ + self.intercept_)
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p, p])


@dataclass
class CalibratedPWinModel:
    base_model: object
    global_calibrator: IsotonicRegression
    calibration_mode: str
    regime_calibrators: dict | None = None
    rolling_calibrators: list | None = None
    rolling_edges: list | None = None
    diagnostics: dict | None = None

    def _base_scores(self, X):
        return np.asarray(self.base_model.predict_proba(X)[:, 1], dtype=float)

    def predict_proba(self, X, regime_labels=None, row_ids=None):
        p_raw = self._base_scores(X)
        p_cal = np.asarray(self.global_calibrator.predict(p_raw), dtype=float)

        if self.calibration_mode == "regime" and regime_labels is not None and self.regime_calibrators:
            regs = np.asarray(regime_labels)
            for reg, cal in self.regime_calibrators.items():
                m = regs == reg
                if np.any(m):
                    p_cal[m] = np.asarray(cal.predict(p_raw[m]), dtype=float)
        elif self.calibration_mode == "rolling" and self.rolling_calibrators:
            if row_ids is None:
                p_cal = np.asarray(self.rolling_calibrators[-1].predict(p_raw), dtype=float)
            else:
                ids = np.asarray(row_ids)
                for i, cal in enumerate(self.rolling_calibrators):
                    lo = self.rolling_edges[i]
                    hi = self.rolling_edges[i + 1]
                    m = (ids >= lo) & (ids < hi)
                    if np.any(m):
                        p_cal[m] = np.asarray(cal.predict(p_raw[m]), dtype=float)

        p_cal = np.clip(p_cal, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p_cal, p_cal])


def _fit_isotonic(p_raw, y_true):
    cal = IsotonicRegression(out_of_bounds="clip")
    cal.fit(np.asarray(p_raw, dtype=float), np.asarray(y_true, dtype=float))
    return cal


def _bce_soft(y_hat, y_soft):
    y_hat = np.clip(np.asarray(y_hat, dtype=float), 1e-9, 1.0 - 1e-9)
    y_soft = np.clip(np.asarray(y_soft, dtype=float), 0.0, 1.0)
    return float(-np.mean(y_soft * np.log(y_hat) + (1.0 - y_soft) * np.log(1.0 - y_hat)))


def train_pwin_classifier(
    X,
    pwin_target,
    calibration_mode="regime",
    regime_labels=None,
    rolling_window=2000,
    y_hard_ref=None,
    pnl_ref=None,
):
    """Train pwin with BCE soft labels + isotonic calibration."""
    X = np.asarray(X, dtype=float)
    y_soft = np.clip(np.asarray(pwin_target, dtype=float), 0.0, 1.0)

    base = SoftLabelLogisticModel(lr=0.1, n_iter=800, l2=1e-4)
    base.fit(X, y_soft)
    p_raw = np.asarray(base.predict_proba(X)[:, 1], dtype=float)
    global_cal = _fit_isotonic(p_raw, y_soft)

    regime_calibrators = None
    rolling_calibrators = None
    rolling_edges = None

    if calibration_mode == "regime" and regime_labels is not None:
        regime_calibrators = {}
        regs = np.asarray(regime_labels)
        for reg in np.unique(regs):
            m = regs == reg
            if np.sum(m) >= 50 and np.nanstd(y_soft[m]) > 1e-8:
                regime_calibrators[reg] = _fit_isotonic(p_raw[m], y_soft[m])
    elif calibration_mode == "rolling":
        n = len(y_soft)
        rolling_edges = list(range(0, n + rolling_window, rolling_window))
        if rolling_edges[-1] != n:
            rolling_edges.append(n)
        rolling_edges = sorted(set(rolling_edges))
        rolling_calibrators = []
        for i in range(len(rolling_edges) - 1):
            lo, hi = rolling_edges[i], rolling_edges[i + 1]
            if hi - lo < 50 or np.nanstd(y_soft[lo:hi]) <= 1e-8:
                rolling_calibrators.append(global_cal)
                continue
            rolling_calibrators.append(_fit_isotonic(p_raw[lo:hi], y_soft[lo:hi]))

    p_cal = np.asarray(global_cal.predict(p_raw), dtype=float)
    diag = {
        "target_mean": float(np.mean(y_soft)),
        "target_std": float(np.std(y_soft)),
        "bce": _bce_soft(p_cal, y_soft),
        "spearman_pwin_soft": float(spearmanr(p_cal, y_soft).correlation) if len(y_soft) > 5 else float("nan"),
        "brier": float(np.mean((y_soft - p_cal) ** 2)),
    }
    if y_hard_ref is not None and len(y_hard_ref) == len(y_soft) and len(np.unique(y_hard_ref)) > 1:
        diag["auc_vs_hard"] = float(roc_auc_score(np.asarray(y_hard_ref, dtype=int), p_cal))
    if pnl_ref is not None and len(pnl_ref) == len(y_soft):
        diag["spearman_vs_realized_pnl"] = float(spearmanr(p_cal, np.asarray(pnl_ref, dtype=float)).correlation)

    return CalibratedPWinModel(
        base_model=base,
        global_calibrator=global_cal,
        calibration_mode=calibration_mode,
        regime_calibrators=regime_calibrators,
        rolling_calibrators=rolling_calibrators,
        rolling_edges=rolling_edges,
        diagnostics=diag,
    )


def _fit_quantile(X, y, tau: float):
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=float(tau),
        random_state=42,
    )
    model.fit(X, y)
    return model


def train_win_quantile_regressor(X, y_win_mag):
    mask = np.isfinite(y_win_mag) & (y_win_mag > 0)
    q50 = _fit_quantile(X[mask], y_win_mag[mask], 0.50)
    q80 = _fit_quantile(X[mask], y_win_mag[mask], 0.80)
    return {"q50": q50, "q80": q80}


def train_loss_quantile_regressor(X, y_loss_mag):
    mask = np.isfinite(y_loss_mag) & (y_loss_mag > 0)
    q50 = _fit_quantile(X[mask], y_loss_mag[mask], 0.50)
    q90 = _fit_quantile(X[mask], y_loss_mag[mask], 0.90)
    return {"q50": q50, "q90": q90}


def predict_quantiles(model_pack: dict, X, high_key: str, low_key: str = "q50"):
    low = np.maximum(model_pack[low_key].predict(X), 0.0)
    high = np.maximum(model_pack[high_key].predict(X), 0.0)
    high = np.maximum(high, low)
    return low, high
