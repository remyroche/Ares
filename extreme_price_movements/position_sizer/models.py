from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

try:
    from sklearn.ensemble import HistGradientBoostingRegressor
except Exception:  # pragma: no cover
    HistGradientBoostingRegressor = None

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover
    XGBRegressor = None


def _sigmoid(z):
    z = np.asarray(z, dtype=float)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p):
    x = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if np.sum(m) < 5:
        return float("nan")
    v = spearmanr(a[m], b[m]).correlation
    return float(v) if np.isfinite(v) else float("nan")


def _safe_weight(sample_weight, n):
    if sample_weight is None:
        return None
    w = np.asarray(sample_weight, dtype=float)
    if w.ndim != 1 or len(w) != n:
        return None
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if np.sum(w) <= 0:
        return None
    return w


def _weighted_mean(x, w=None):
    x = np.asarray(x, dtype=float)
    if w is None:
        return float(np.mean(x))
    ww = _safe_weight(w, len(x))
    if ww is None:
        return float(np.mean(x))
    return float(np.sum(x * ww) / np.sum(ww))


class IdentityCalibrator:
    def fit(self, p_raw, y_true, sample_weight=None):
        return self

    def predict(self, p_raw):
        return np.clip(np.asarray(p_raw, dtype=float), 1e-6, 1.0 - 1e-6)


class SoftLabelSigmoidCalibrator:
    """Platt-like calibrator trained directly on soft labels."""

    def __init__(self, lr=0.05, n_iter=600, l2=1e-1, coef_cap=8.0):
        self.lr = float(lr)
        self.n_iter = int(n_iter)
        self.l2 = float(l2)
        self.coef_cap = float(coef_cap)
        self.a_ = 1.0
        self.b_ = 0.0

    def fit(self, p_raw, y_true, sample_weight=None):
        x = _logit(p_raw)
        y = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)
        w = _safe_weight(sample_weight, len(y))
        if w is None:
            w = np.ones(len(y), dtype=float)
        sw = float(np.sum(w))
        if sw <= 0:
            return self

        a = 1.0
        b = 0.0
        for _ in range(self.n_iter):
            z = a * x + b
            p = _sigmoid(z)
            e = p - y
            ga = float(np.sum(w * e * x) / sw + self.l2 * a)
            gb = float(np.sum(w * e) / sw + self.l2 * b)
            a -= self.lr * ga
            b -= self.lr * gb
            a = float(np.clip(a, -self.coef_cap, self.coef_cap))
            b = float(np.clip(b, -self.coef_cap, self.coef_cap))
        self.a_ = a
        self.b_ = b
        return self

    def predict(self, p_raw):
        x = _logit(p_raw)
        p = _sigmoid(self.a_ * x + self.b_)
        return np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)


def _fit_isotonic(p_raw, y_true, sample_weight=None):
    cal = IsotonicRegression(out_of_bounds="clip")
    sw = _safe_weight(sample_weight, len(y_true))
    if sw is None:
        cal.fit(np.asarray(p_raw, dtype=float), np.asarray(y_true, dtype=float))
    else:
        cal.fit(np.asarray(p_raw, dtype=float), np.asarray(y_true, dtype=float), sample_weight=sw)
    return cal


def _fit_calibrator(
    p_raw,
    y_true,
    method="auto",
    min_samples_isotonic=1200,
    sample_weight=None,
    regularization_level="strong",
):
    p = np.asarray(p_raw, dtype=float)
    y = np.asarray(y_true, dtype=float)
    n = len(y)
    if n < 60 or np.nanstd(y) <= 1e-8:
        return IdentityCalibrator(), "identity"
    if method == "auto":
        method = "isotonic" if n >= int(min_samples_isotonic) else "sigmoid"

    if method == "sigmoid":
        if regularization_level == "strong":
            cal = SoftLabelSigmoidCalibrator(lr=0.03, n_iter=800, l2=3e-1, coef_cap=6.0)
        else:
            cal = SoftLabelSigmoidCalibrator(lr=0.05, n_iter=500, l2=8e-2, coef_cap=8.0)
        cal.fit(p, y, sample_weight=sample_weight)
        return cal, "sigmoid"

    cal = _fit_isotonic(p, y, sample_weight=sample_weight)
    return cal, "isotonic"


def _bce_soft(y_hat, y_soft, sample_weight=None):
    y_hat = np.clip(np.asarray(y_hat, dtype=float), 1e-9, 1.0 - 1e-9)
    y_soft = np.clip(np.asarray(y_soft, dtype=float), 0.0, 1.0)
    w = _safe_weight(sample_weight, len(y_hat))
    l = -(y_soft * np.log(y_hat) + (1.0 - y_soft) * np.log(1.0 - y_hat))
    return _weighted_mean(l, w)


def _ece(y_true, y_prob, n_bins=20, sample_weight=None):
    y = np.clip(np.asarray(y_true, dtype=float), 0.0, 1.0)
    p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    w = _safe_weight(sample_weight, len(y))
    bins = np.linspace(0.0, 1.0, int(n_bins) + 1)
    out = 0.0
    total = float(np.sum(w)) if w is not None else float(len(y))
    total = max(total, 1e-12)
    for i in range(len(bins) - 1):
        if i < len(bins) - 2:
            m = (p >= bins[i]) & (p < bins[i + 1])
        else:
            m = (p >= bins[i]) & (p <= bins[i + 1])
        if not np.any(m):
            continue
        if w is None:
            w_bin = float(np.sum(m))
            y_bin = float(np.mean(y[m]))
            p_bin = float(np.mean(p[m]))
        else:
            wm = w[m]
            w_bin = float(np.sum(wm))
            y_bin = float(np.sum(y[m] * wm) / max(w_bin, 1e-12))
            p_bin = float(np.sum(p[m] * wm) / max(w_bin, 1e-12))
        out += (w_bin / total) * abs(y_bin - p_bin)
    return float(out)


def _regularization_presets(level="strong"):
    level = str(level).lower()
    if level == "medium":
        return {
            "et_pwin": {
                "n_estimators": 1400,
                "min_samples_leaf": 80,
                "min_samples_split": 350,
                "max_depth": 12,
                "max_features": 0.45,
                "bootstrap": True,
                "max_samples": 0.8,
                "n_jobs": 3,
                "random_state": 42,
            },
            "xgb_pwin": {
                "n_estimators": 3500,
                "learning_rate": 0.04,
                "max_depth": 4,
                "min_child_weight": 60.0,
                "gamma": 3.0,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "colsample_bynode": 0.75,
                "reg_lambda": 20.0,
                "reg_alpha": 1.0,
                "max_bin": 512,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 3,
            },
            "hgb_quantile": {
                "learning_rate": 0.05,
                "max_iter": 450,
                "max_depth": 4,
                "min_samples_leaf": 45,
                "l2_regularization": 0.8,
                "random_state": 42,
            },
            "gbr_quantile": {
                "n_estimators": 550,
                "learning_rate": 0.04,
                "max_depth": 3,
                "min_samples_leaf": 40,
                "subsample": 0.85,
                "random_state": 42,
            },
            "xgb_quantile": {
                "n_estimators": 3000,
                "learning_rate": 0.04,
                "max_depth": 4,
                "min_child_weight": 40.0,
                "gamma": 2.0,
                "subsample": 0.75,
                "colsample_bytree": 0.75,
                "colsample_bynode": 0.75,
                "reg_lambda": 15.0,
                "reg_alpha": 0.5,
                "max_bin": 512,
                "tree_method": "hist",
                "random_state": 42,
                "n_jobs": 3,
            },
        }

    return {
        "et_pwin": {
            "n_estimators": 2200,
            "min_samples_leaf": 140,
            "min_samples_split": 600,
            "max_depth": 10,
            "max_features": 0.35,
            "bootstrap": True,
            "max_samples": 0.65,
            "n_jobs": 3,
            "random_state": 42,
        },
        "xgb_pwin": {
            "n_estimators": 6000,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 120.0,
            "gamma": 6.0,
            "subsample": 0.65,
            "colsample_bytree": 0.70,
            "colsample_bynode": 0.70,
            "reg_lambda": 40.0,
            "reg_alpha": 4.0,
            "max_bin": 512,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 3,
        },
        "hgb_quantile": {
            "learning_rate": 0.04,
            "max_iter": 650,
            "max_depth": 3,
            "min_samples_leaf": 70,
            "l2_regularization": 1.5,
            "random_state": 42,
        },
        "gbr_quantile": {
            "n_estimators": 800,
            "learning_rate": 0.03,
            "max_depth": 2,
            "min_samples_leaf": 70,
            "subsample": 0.75,
            "random_state": 42,
        },
        "xgb_quantile": {
            "n_estimators": 5000,
            "learning_rate": 0.03,
            "max_depth": 3,
            "min_child_weight": 100.0,
            "gamma": 5.0,
            "subsample": 0.65,
            "colsample_bytree": 0.70,
            "colsample_bynode": 0.70,
            "reg_lambda": 30.0,
            "reg_alpha": 2.0,
            "max_bin": 512,
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": 3,
        },
    }


def _fit_pwin_base_model(
    X_train,
    y_train,
    X_cal=None,
    y_cal=None,
    sample_weight_train=None,
    base_engine="extratrees",
    regularization_level="strong",
):
    base_engine = str(base_engine).lower()
    presets = _regularization_presets(regularization_level)
    if base_engine == "xgb":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not available, cannot use base_engine='xgb'")
        model = XGBRegressor(
            objective="reg:squarederror",
            **presets["xgb_pwin"],
        )
        kwargs = {}
        if sample_weight_train is not None:
            kwargs["sample_weight"] = sample_weight_train
        if X_cal is not None and y_cal is not None and len(y_cal) >= 50:
            kwargs["eval_set"] = [(X_cal, y_cal)]
            kwargs["verbose"] = False
            try:
                model.fit(X_train, y_train, early_stopping_rounds=120, **kwargs)
            except TypeError:
                model.fit(X_train, y_train, **kwargs)
        else:
            model.fit(X_train, y_train, **kwargs)
        return model

    model = ExtraTreesRegressor(**presets["et_pwin"])
    if sample_weight_train is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight_train)
    else:
        model.fit(X_train, y_train)
    return model


@dataclass
class CalibratedPWinModel:
    base_model: object
    global_calibrator: object
    calibration_mode: str
    regime_calibrators: dict | None = None
    rolling_calibrators: list | None = None
    rolling_edges: list | None = None
    diagnostics: dict | None = None

    def _base_scores(self, X):
        if hasattr(self.base_model, "predict_proba"):
            p = np.asarray(self.base_model.predict_proba(X)[:, 1], dtype=float)
        else:
            p = np.asarray(self.base_model.predict(X), dtype=float)
        return np.clip(p, 1e-6, 1.0 - 1e-6)

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


def _compute_metrics(prefix, y_true, y_prob, sample_weight=None, y_hard_ref=None):
    out = {
        f"bce_{prefix}": _bce_soft(y_prob, y_true, sample_weight=sample_weight),
        f"brier_{prefix}": _weighted_mean((np.asarray(y_true) - np.asarray(y_prob)) ** 2, sample_weight),
        f"ece_{prefix}": _ece(y_true, y_prob, n_bins=20, sample_weight=sample_weight),
        f"spearman_{prefix}": _safe_corr(y_prob, y_true),
    }
    if y_hard_ref is not None and len(y_hard_ref) == len(y_prob) and len(np.unique(y_hard_ref)) > 1:
        try:
            out[f"auc_{prefix}"] = float(roc_auc_score(np.asarray(y_hard_ref, dtype=int), y_prob))
        except Exception:
            out[f"auc_{prefix}"] = float("nan")
    return out


def train_pwin_classifier(
    X,
    pwin_target,
    calibration_mode="regime",
    regime_labels=None,
    rolling_window=2000,
    y_hard_ref=None,
    pnl_ref=None,
    base_engine="extratrees",
    regularization_level="strong",
    calibrator_method="auto",
    min_samples_isotonic=1200,
    calibration_frac=0.20,
    calibration_min_samples=200,
    sample_weight=None,
    diagnostics_walkforward_blocks=0,
):
    """Train pwin with soft labels and strict forward-only calibration."""
    X = np.asarray(X, dtype=float)
    y_soft = np.clip(np.asarray(pwin_target, dtype=float), 0.0, 1.0)
    n = len(y_soft)
    if n == 0:
        raise ValueError("train_pwin_classifier received empty dataset")

    w_all = _safe_weight(sample_weight, n)
    calib_n = int(max(calibration_min_samples, round(float(calibration_frac) * n)))
    calib_n = int(min(calib_n, max(50, n - 50)))
    train_end = int(max(50, n - calib_n))
    idx_tr = np.arange(0, train_end, dtype=int)
    idx_cal = np.arange(train_end, n, dtype=int)

    X_tr = X[idx_tr]
    y_tr = y_soft[idx_tr]
    X_cal = X[idx_cal]
    y_cal = y_soft[idx_cal]
    w_tr = w_all[idx_tr] if w_all is not None else None
    w_cal = w_all[idx_cal] if w_all is not None else None

    base = _fit_pwin_base_model(
        X_tr,
        y_tr,
        X_cal=X_cal,
        y_cal=y_cal,
        sample_weight_train=w_tr,
        base_engine=base_engine,
        regularization_level=regularization_level,
    )
    p_raw_all = np.clip(np.asarray(base.predict(X), dtype=float), 1e-6, 1.0 - 1e-6)
    p_raw_cal = p_raw_all[idx_cal]

    global_cal, global_method = _fit_calibrator(
        p_raw_cal,
        y_cal,
        method=calibrator_method,
        min_samples_isotonic=min_samples_isotonic,
        sample_weight=w_cal,
        regularization_level=regularization_level,
    )

    regime_calibrators = None
    rolling_calibrators = None
    rolling_edges = None
    regime_diag = {}

    if calibration_mode == "regime" and regime_labels is not None:
        regime_calibrators = {}
        regs = np.asarray(regime_labels)
        regs_cal = regs[idx_cal]
        for reg in np.unique(regs_cal):
            m = regs_cal == reg
            n_reg = int(np.sum(m))
            regime_diag[str(reg)] = {"n": n_reg}
            if n_reg < 80:
                regime_diag[str(reg)]["method"] = "fallback_global"
                continue
            w_reg = w_cal[m] if w_cal is not None else None
            cal, meth = _fit_calibrator(
                p_raw_cal[m],
                y_cal[m],
                method=calibrator_method,
                min_samples_isotonic=max(min_samples_isotonic, 1500),
                sample_weight=w_reg,
                regularization_level=regularization_level,
            )
            regime_calibrators[reg] = cal
            regime_diag[str(reg)]["method"] = meth
    elif calibration_mode == "rolling":
        rolling_edges = list(range(0, n + int(max(50, rolling_window)), int(max(50, rolling_window))))
        if rolling_edges[-1] != n:
            rolling_edges.append(n)
        rolling_edges = sorted(set(rolling_edges))
        rolling_calibrators = []
        for i in range(len(rolling_edges) - 1):
            lo, _hi = rolling_edges[i], rolling_edges[i + 1]
            if lo < 80:
                rolling_calibrators.append(IdentityCalibrator())
                continue
            hist_lo = max(0, lo - int(max(calibration_min_samples, rolling_window)))
            hist_idx = np.arange(hist_lo, lo, dtype=int)
            w_hist = w_all[hist_idx] if w_all is not None else None
            cal_i, _ = _fit_calibrator(
                p_raw_all[hist_idx],
                y_soft[hist_idx],
                method=calibrator_method,
                min_samples_isotonic=max(min_samples_isotonic, 1500),
                sample_weight=w_hist,
                regularization_level=regularization_level,
            )
            rolling_calibrators.append(cal_i)

    model = CalibratedPWinModel(
        base_model=base,
        global_calibrator=global_cal,
        calibration_mode=calibration_mode,
        regime_calibrators=regime_calibrators,
        rolling_calibrators=rolling_calibrators,
        rolling_edges=rolling_edges,
    )
    row_ids = np.arange(n)
    p_cal_all = np.asarray(model.predict_proba(X, regime_labels=regime_labels, row_ids=row_ids)[:, 1], dtype=float)
    p_cal_cal = p_cal_all[idx_cal]

    diag = {
        "target_mean": float(np.mean(y_soft)),
        "target_std": float(np.std(y_soft)),
        "calibration_mode": str(calibration_mode),
        "global_calibration_method": str(global_method),
        "base_model": f"{str(base_engine).lower()}_regressor",
        "regularization_level": str(regularization_level),
        "n_train_base": int(len(idx_tr)),
        "n_calibration": int(len(idx_cal)),
        "train_end": int(train_end),
        "regime_calibration": regime_diag,
    }
    diag.update(_compute_metrics("cal", y_cal, p_cal_cal, sample_weight=w_cal, y_hard_ref=(np.asarray(y_hard_ref)[idx_cal] if y_hard_ref is not None else None)))
    diag.update(_compute_metrics("all", y_soft, p_cal_all, sample_weight=w_all, y_hard_ref=y_hard_ref))
    # Backward-compatible keys
    diag["bce"] = float(diag.get("bce_all", np.nan))
    diag["brier"] = float(diag.get("brier_all", np.nan))
    diag["ece_20"] = float(diag.get("ece_all", np.nan))
    diag["spearman_pwin_soft"] = float(diag.get("spearman_all", np.nan))

    if pnl_ref is not None and len(pnl_ref) == n:
        diag["spearman_vs_realized_pnl_all"] = _safe_corr(p_cal_all, np.asarray(pnl_ref, dtype=float))
        diag["spearman_vs_realized_pnl_cal"] = _safe_corr(p_cal_cal, np.asarray(pnl_ref, dtype=float)[idx_cal])

    wf_blocks = int(max(0, diagnostics_walkforward_blocks))
    if wf_blocks >= 2 and len(idx_cal) >= wf_blocks * 30:
        block_edges = np.linspace(idx_cal[0], idx_cal[-1] + 1, wf_blocks + 1, dtype=int)
        wf_rows = []
        for i in range(wf_blocks):
            lo = int(block_edges[i])
            hi = int(block_edges[i + 1])
            if hi <= lo:
                continue
            m = (row_ids >= lo) & (row_ids < hi)
            if np.sum(m) < 20:
                continue
            wf_rows.append({
                "block": int(i),
                "n": int(np.sum(m)),
                "bce": _bce_soft(p_cal_all[m], y_soft[m], sample_weight=(w_all[m] if w_all is not None else None)),
                "brier": _weighted_mean((y_soft[m] - p_cal_all[m]) ** 2, (w_all[m] if w_all is not None else None)),
                "ece": _ece(y_soft[m], p_cal_all[m], n_bins=20, sample_weight=(w_all[m] if w_all is not None else None)),
                "spearman": _safe_corr(p_cal_all[m], y_soft[m]),
            })
        diag["walkforward_oos"] = wf_rows

    model.diagnostics = diag
    return model


class _ConstantRegressor:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        X = np.asarray(X)
        return np.full(len(X), self.value, dtype=float)


class _LogQuantileModel:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        yp = np.asarray(self.model.predict(X), dtype=float)
        yp = np.maximum(yp, 0.0)
        return np.maximum(np.expm1(yp), 0.0)


class _DeltaQuantileModel:
    """q_high = q_low + max(delta, 0) in log space, then expm1."""

    def __init__(self, low_model, delta_model):
        self.low_model = low_model
        self.delta_model = delta_model

    def predict(self, X):
        low = np.asarray(self.low_model.predict(X), dtype=float)
        dlt = np.asarray(self.delta_model.predict(X), dtype=float)
        out = np.maximum(low, 0.0) + np.maximum(dlt, 0.0)
        return np.maximum(np.expm1(out), 0.0)


def _fit_quantile_base(X, y_log, tau, engine, regularization_level):
    presets = _regularization_presets(regularization_level)
    engine = str(engine).lower()
    if engine == "xgb":
        if XGBRegressor is None:
            raise RuntimeError("xgboost is not available, cannot use quantile engine='xgb'")
        params = dict(presets["xgb_quantile"])
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = float(tau)
        model = XGBRegressor(**params)
        try:
            model.fit(X, y_log, verbose=False)
        except TypeError:
            model.fit(X, y_log)
        return model

    if HistGradientBoostingRegressor is not None:
        try:
            params = dict(presets["hgb_quantile"])
            model = HistGradientBoostingRegressor(
                loss="quantile",
                quantile=float(tau),
                **params,
            )
            model.fit(X, y_log)
            return model
        except Exception:
            pass

    params = dict(presets["gbr_quantile"])
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=float(tau),
        **params,
    )
    model.fit(X, y_log)
    return model


def _fit_quantile(X, y, tau, engine="sklearn", regularization_level="strong"):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    y = np.maximum(y, 0.0)
    if len(y) < 50:
        return _ConstantRegressor(float(np.nanmedian(y) if len(y) else 0.0))
    y_log = np.log1p(y)
    m = _fit_quantile_base(X, y_log, tau=tau, engine=engine, regularization_level=regularization_level)
    return _LogQuantileModel(m)


def train_win_quantile_regressor(
    X,
    y_win_mag,
    base_engine="sklearn",
    regularization_level="strong",
    delta_quantile=False,
):
    X = np.asarray(X, dtype=float)
    y_win_mag = np.asarray(y_win_mag, dtype=float)
    mask = np.isfinite(y_win_mag) & (y_win_mag > 0)
    Xm = X[mask]
    ym = y_win_mag[mask]
    q50 = _fit_quantile(Xm, ym, 0.50, engine=base_engine, regularization_level=regularization_level)
    if not bool(delta_quantile):
        q80 = _fit_quantile(Xm, ym, 0.80, engine=base_engine, regularization_level=regularization_level)
    else:
        ylog = np.log1p(ym)
        low_log = np.asarray(q50.model.predict(Xm), dtype=float) if isinstance(q50, _LogQuantileModel) else np.log1p(np.maximum(q50.predict(Xm), 0.0))
        delta = np.maximum(ylog - np.maximum(low_log, 0.0), 0.0)
        delta_model = _fit_quantile_base(Xm, delta, tau=0.80, engine=base_engine, regularization_level=regularization_level)
        low_model = q50.model if isinstance(q50, _LogQuantileModel) else _fit_quantile_base(Xm, ylog, tau=0.50, engine=base_engine, regularization_level=regularization_level)
        q80 = _DeltaQuantileModel(low_model=low_model, delta_model=delta_model)
    return {"q50": q50, "q80": q80}


def train_loss_quantile_regressor(
    X,
    y_loss_mag,
    base_engine="sklearn",
    regularization_level="strong",
    delta_quantile=False,
):
    X = np.asarray(X, dtype=float)
    y_loss_mag = np.asarray(y_loss_mag, dtype=float)
    mask = np.isfinite(y_loss_mag) & (y_loss_mag > 0)
    Xm = X[mask]
    ym = y_loss_mag[mask]
    q50 = _fit_quantile(Xm, ym, 0.50, engine=base_engine, regularization_level=regularization_level)
    if not bool(delta_quantile):
        q90 = _fit_quantile(Xm, ym, 0.90, engine=base_engine, regularization_level=regularization_level)
    else:
        ylog = np.log1p(ym)
        low_log = np.asarray(q50.model.predict(Xm), dtype=float) if isinstance(q50, _LogQuantileModel) else np.log1p(np.maximum(q50.predict(Xm), 0.0))
        delta = np.maximum(ylog - np.maximum(low_log, 0.0), 0.0)
        delta_model = _fit_quantile_base(Xm, delta, tau=0.90, engine=base_engine, regularization_level=regularization_level)
        low_model = q50.model if isinstance(q50, _LogQuantileModel) else _fit_quantile_base(Xm, ylog, tau=0.50, engine=base_engine, regularization_level=regularization_level)
        q90 = _DeltaQuantileModel(low_model=low_model, delta_model=delta_model)
    return {"q50": q50, "q90": q90}


def predict_quantiles(model_pack: dict, X, high_key: str, low_key: str = "q50"):
    low = np.maximum(model_pack[low_key].predict(X), 0.0)
    high = np.maximum(model_pack[high_key].predict(X), 0.0)
    high = np.maximum(high, low)
    return low, high
