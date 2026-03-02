from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    from src.utils.tprint import tprint_info, tprint_warning
except ImportError:
    def tprint_info(msg): print(f"[INFO] {msg}")
    def tprint_warning(msg): print(f"[WARNING] {msg}")


def _sigmoid(x):
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def _clamp01(x):
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def _compute_soft_labels(
    mfe,
    mae,
    tp: float,
    sl: float,
    alpha: float,
    use_log_excursions: bool = False,
    log_eps: float = 1e-12,
):
    mfe = np.asarray(mfe, dtype=float)
    mae = np.asarray(mae, dtype=float)
    mfe = np.clip(mfe, 0.0, None)
    mae = np.clip(mae, 0.0, None)

    if use_log_excursions:
        mfe_eff = np.log1p(mfe + float(log_eps))
        mae_eff = np.log1p(mae + float(log_eps))
        tp_eff = np.log1p(max(float(tp), 0.0) + float(log_eps))
        sl_eff = np.log1p(max(float(sl), 0.0) + float(log_eps))
    else:
        mfe_eff, mae_eff = mfe, mae
        tp_eff, sl_eff = float(tp), float(sl)

    a = float(max(alpha, 1e-8))
    u_soft = tp_eff * _sigmoid(a * (mfe_eff - tp_eff)) - sl_eff * _sigmoid(a * (mae_eff - sl_eff))
    denom = max(tp_eff + sl_eff, 1e-12)
    pwin_soft = _clamp01((u_soft + sl_eff) / denom)
    return u_soft, pwin_soft


@dataclass
class PositionSizerDataset:
    X: pd.DataFrame
    y_win: np.ndarray
    pwin_target: np.ndarray
    y_win_mag: np.ndarray
    y_loss_mag: np.ndarray
    winners_mask: np.ndarray
    losers_mask: np.ndarray
    mfe: np.ndarray
    mae: np.ndarray
    u_soft: np.ndarray


def build_position_sizer_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    pnl_col: str = "pnl_label",
    mfe_col: str = "mfe",
    mae_col: str = "mae",
    pwin_soft_cfg: dict | None = None,
) -> PositionSizerDataset:
    tprint_info(f"Building PositionSizerDataset from DataFrame with shape {df.shape} and {len(feature_cols)} features")
    pnl = np.asarray(df[pnl_col].values, dtype=float)
    y_win = (pnl > 0.0).astype(np.int8)
    winners_mask = pnl > 0.0
    losers_mask = pnl < 0.0
    y_win_mag = np.where(winners_mask, pnl, np.nan)
    y_loss_mag = np.where(losers_mask, np.abs(pnl), np.nan)

    mfe = np.clip(np.asarray(df[mfe_col].values, dtype=float), 0.0, None) if mfe_col in df.columns else np.zeros(len(df), dtype=float)
    mae = np.clip(np.asarray(df[mae_col].values, dtype=float), 0.0, None) if mae_col in df.columns else np.zeros(len(df), dtype=float)

    soft_cfg = pwin_soft_cfg or {}
    soft_enabled = bool(soft_cfg.get("enabled", False))

    if soft_enabled:
        tprint_info(f"Computing soft labels with config: {soft_cfg}")
        u_soft, pwin_target = _compute_soft_labels(
            mfe=mfe,
            mae=mae,
            tp=float(soft_cfg.get("tp", 0.02)),
            sl=float(soft_cfg.get("sl", 0.01)),
            alpha=float(soft_cfg.get("alpha", 15.0)),
            use_log_excursions=bool(soft_cfg.get("use_log_excursions", False)),
            log_eps=float(soft_cfg.get("log_eps", 1e-12)),
        )
    else:
        tprint_info("Soft labels disabled. Using hard labels based on pnl > 0")
        u_soft = np.where(y_win > 0, 1.0, -1.0)
        pwin_target = y_win.astype(float)

    tprint_info(f"Dataset built successfully. Winners: {np.sum(winners_mask)}, Losers: {np.sum(losers_mask)}")
    return PositionSizerDataset(
        X=df[feature_cols].copy().fillna(0.0),
        y_win=y_win,
        pwin_target=np.asarray(pwin_target, dtype=float),
        y_win_mag=y_win_mag,
        y_loss_mag=y_loss_mag,
        winners_mask=winners_mask,
        losers_mask=losers_mask,
        mfe=mfe,
        mae=mae,
        u_soft=np.asarray(u_soft, dtype=float),
    )


def build_pwin_dataset(df: pd.DataFrame, feature_cols: list[str], **kwargs) -> PositionSizerDataset:
    tprint_info("Building pwin dataset")
    return build_position_sizer_dataset(df=df, feature_cols=feature_cols, **kwargs)


def build_win_quantile_dataset(df: pd.DataFrame, feature_cols: list[str], pnl_col: str = "pnl_label") -> tuple[pd.DataFrame, np.ndarray]:
    tprint_info("Building win quantile dataset")
    ds = build_position_sizer_dataset(df=df, feature_cols=feature_cols, pnl_col=pnl_col)
    mask = np.isfinite(ds.y_win_mag)
    tprint_info(f"Win quantile dataset returned {np.sum(mask)} finite samples")
    return ds.X.loc[mask].reset_index(drop=True), np.asarray(ds.y_win_mag[mask], dtype=float)


def build_loss_quantile_dataset(df: pd.DataFrame, feature_cols: list[str], pnl_col: str = "pnl_label") -> tuple[pd.DataFrame, np.ndarray]:
    tprint_info("Building loss quantile dataset")
    ds = build_position_sizer_dataset(df=df, feature_cols=feature_cols, pnl_col=pnl_col)
    mask = np.isfinite(ds.y_loss_mag)
    tprint_info(f"Loss quantile dataset returned {np.sum(mask)} finite samples")
    return ds.X.loc[mask].reset_index(drop=True), np.asarray(ds.y_loss_mag[mask], dtype=float)
