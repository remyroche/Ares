from __future__ import annotations

import numpy as np

from .dataset import build_pwin_dataset, build_win_quantile_dataset, build_loss_quantile_dataset
from .models import train_pwin_classifier, train_win_quantile_regressor, train_loss_quantile_regressor


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

    return {
        "pwin_model": pwin_model,
        "win_model": win_model,
        "loss_model": loss_model,
        "exp_win_quantile": exp_win_q,
        "risk_loss_quantile": risk_loss_q,
        "costs_mode": costs_mode,
        "soft_label_enabled": soft_cfg["enabled"],
    }
