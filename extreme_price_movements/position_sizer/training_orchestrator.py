from __future__ import annotations

import numpy as np

from .dataset import build_pwin_dataset, build_win_quantile_dataset, build_loss_quantile_dataset
from .models import train_pwin_classifier, train_win_quantile_regressor, train_loss_quantile_regressor


def _pinball_loss(y_true, y_pred, q: float) -> float:
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    e = y - yp
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e))) if len(y) else float("nan")


def _quantile_metrics(y_true, q50, qh, qh_level: float) -> dict:
    y = np.asarray(y_true, dtype=float)
    p50 = np.asarray(q50, dtype=float)
    ph = np.asarray(qh, dtype=float)
    m = np.isfinite(y) & np.isfinite(p50) & np.isfinite(ph)
    if not np.any(m):
        return {
            "n": 0,
            "pinball_q50": float("nan"),
            "pinball_qh": float("nan"),
            "coverage_q50": float("nan"),
            "coverage_qh": float("nan"),
            "mean_y": float("nan"),
            "mean_q50": float("nan"),
            "mean_qh": float("nan"),
        }
    yy = y[m]
    p50m = p50[m]
    phm = ph[m]
    return {
        "n": int(len(yy)),
        "pinball_q50": _pinball_loss(yy, p50m, 0.50),
        "pinball_qh": _pinball_loss(yy, phm, float(qh_level)),
        "coverage_q50": float(np.mean(yy <= p50m)),
        "coverage_qh": float(np.mean(yy <= phm)),
        "mean_y": float(np.mean(yy)),
        "mean_q50": float(np.mean(p50m)),
        "mean_qh": float(np.mean(phm)),
    }


def train_position_sizer_models(ps_df, feature_cols: list[str], cfg: dict):
    """Train p(win) + win/loss quantile heads for the position-sizer bundle."""
    soft_cfg = {
        "enabled": bool(cfg.get("position_sizer_pwin_soft_label_enabled", False)),
        "tp": float(cfg.get("position_sizer_pwin_soft_label_tp", 0.02)),
        "sl": float(cfg.get("position_sizer_pwin_soft_label_sl", 0.01)),
        "alpha": float(cfg.get("position_sizer_pwin_soft_label_alpha", 15.0)),
        "use_log_excursions": bool(cfg.get("position_sizer_pwin_soft_label_use_log_excursions", False)),
        "log_eps": float(cfg.get("position_sizer_pwin_soft_label_log_eps", 1e-12)),
    }
    exp_win_q = float(cfg.get("position_sizer_exp_win_quantile", 0.50))
    risk_loss_q = float(cfg.get("position_sizer_risk_loss_quantile", 0.90))
    costs_mode = str(cfg.get("position_sizer_costs_mode", "included_in_labels"))
    pwin_base_engine = str(cfg.get("position_sizer_pwin_base_engine", "extratrees")).lower()
    quant_base_engine = str(cfg.get("position_sizer_quantile_base_engine", "sklearn")).lower()
    reg_level = str(cfg.get("position_sizer_regularization_level", "strong")).lower()
    calibrator_method = str(cfg.get("position_sizer_calibrator_method", "auto")).lower()
    min_iso = int(cfg.get("position_sizer_min_samples_isotonic", 1200))
    calib_frac = float(cfg.get("position_sizer_calibration_frac", 0.20))
    calib_min = int(cfg.get("position_sizer_calibration_min_samples", 200))
    quant_delta = bool(cfg.get("position_sizer_quantile_delta", False))
    pwin_wf_blocks = int(cfg.get("position_sizer_pwin_walkforward_blocks", 0))

    ds = build_pwin_dataset(ps_df, feature_cols=feature_cols, pnl_col="pnl_label", mfe_col="mfe", mae_col="mae", pwin_soft_cfg=soft_cfg)
    reg_labels = ps_df["bucket"].values if "bucket" in ps_df.columns else None
    pwin_model = train_pwin_classifier(
        ds.X.values,
        ds.pwin_target,
        calibration_mode=str(cfg.get("position_sizer_calibration_scope", "regime")),
        regime_labels=reg_labels,
        rolling_window=int(cfg.get("position_sizer_calibration_rolling_window", 2000)),
        y_hard_ref=ds.y_win,
        pnl_ref=ps_df["pnl_label"].values if "pnl_label" in ps_df.columns else None,
        base_engine=pwin_base_engine,
        regularization_level=reg_level,
        calibrator_method=calibrator_method,
        min_samples_isotonic=min_iso,
        calibration_frac=calib_frac,
        calibration_min_samples=calib_min,
        diagnostics_walkforward_blocks=pwin_wf_blocks,
    )

    Xw, yw = build_win_quantile_dataset(ps_df, feature_cols=feature_cols, pnl_col="pnl_label")
    Xl, yl = build_loss_quantile_dataset(ps_df, feature_cols=feature_cols, pnl_col="pnl_label")
    win_model = train_win_quantile_regressor(
        Xw.values,
        yw,
        base_engine=quant_base_engine,
        regularization_level=reg_level,
        delta_quantile=quant_delta,
    )
    loss_model = train_loss_quantile_regressor(
        Xl.values,
        yl,
        base_engine=quant_base_engine,
        regularization_level=reg_level,
        delta_quantile=quant_delta,
    )

    # Diagnostics for tprint/reporting (overall + per bucket)
    pwin_diag = dict(getattr(pwin_model, "diagnostics", {}) or {})
    pwin_pred = pwin_model.predict_proba(ds.X.values, regime_labels=reg_labels)[:, 1]
    pwin_diag["pwin_mean_pred"] = float(np.mean(pwin_pred)) if len(pwin_pred) else float("nan")
    pwin_diag["pwin_mean_target"] = float(np.mean(ds.pwin_target)) if len(ds.pwin_target) else float("nan")

    wq50 = np.asarray(win_model["q50"].predict(Xw.values), dtype=float)
    wq80 = np.asarray(win_model["q80"].predict(Xw.values), dtype=float)
    lq50 = np.asarray(loss_model["q50"].predict(Xl.values), dtype=float)
    lq90 = np.asarray(loss_model["q90"].predict(Xl.values), dtype=float)

    diag = {
        "pwin": pwin_diag,
        "win_quantiles": _quantile_metrics(yw, wq50, wq80, 0.80),
        "loss_quantiles": _quantile_metrics(yl, lq50, lq90, 0.90),
        "per_bucket": {},
    }

    if "bucket" in ps_df.columns:
        bvals = ps_df["bucket"].astype(str).values
        for b in sorted(np.unique(bvals)):
            mb = bvals == b
            # pwin slice
            pb = pwin_model.predict_proba(ds.X.values[mb], regime_labels=bvals[mb])[:, 1] if np.any(mb) else np.array([])
            yb = np.asarray(ds.pwin_target, dtype=float)[mb] if np.any(mb) else np.array([])
            bb = {
                "pwin": {
                    "n": int(np.sum(mb)),
                    "mean_pred": float(np.mean(pb)) if len(pb) else float("nan"),
                    "mean_target": float(np.mean(yb)) if len(yb) else float("nan"),
                }
            }
            # win/loss quantile slices by bucket using original rows and win/loss masks.
            pnl_all = np.asarray(ps_df["pnl_label"].values, dtype=float)
            m_win = mb & np.isfinite(pnl_all) & (pnl_all > 0)
            m_loss = mb & np.isfinite(pnl_all) & (pnl_all < 0)
            if np.any(m_win):
                bb["win_quantiles"] = _quantile_metrics(
                    np.asarray(pnl_all[m_win], dtype=float),
                    np.asarray(win_model["q50"].predict(ps_df.loc[m_win, feature_cols].values), dtype=float),
                    np.asarray(win_model["q80"].predict(ps_df.loc[m_win, feature_cols].values), dtype=float),
                    0.80,
                )
            if np.any(m_loss):
                y_loss_b = np.asarray(np.maximum(-pnl_all[m_loss], 0.0), dtype=float)
                bb["loss_quantiles"] = _quantile_metrics(
                    y_loss_b,
                    np.asarray(loss_model["q50"].predict(ps_df.loc[m_loss, feature_cols].values), dtype=float),
                    np.asarray(loss_model["q90"].predict(ps_df.loc[m_loss, feature_cols].values), dtype=float),
                    0.90,
                )
            diag["per_bucket"][b] = bb

    return {
        "pwin_model": pwin_model,
        "win_model": win_model,
        "loss_model": loss_model,
        "exp_win_quantile": exp_win_q,
        "risk_loss_quantile": risk_loss_q,
        "costs_mode": costs_mode,
        "soft_label_enabled": soft_cfg["enabled"],
        "diagnostics": diag,
        "training_config": {
            "pwin_base_engine": pwin_base_engine,
            "quantile_base_engine": quant_base_engine,
            "regularization_level": reg_level,
            "calibrator_method": calibrator_method,
            "quantile_delta": quant_delta,
        },
    }
