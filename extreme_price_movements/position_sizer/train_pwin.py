from .dataset import build_position_sizer_dataset
from .models import train_pwin_classifier
from ..utils import tprint


def train(
    df,
    feature_cols,
    calibration_mode="regime",
    regime_col=None,
    rolling_window=2000,
    pwin_soft_cfg=None,
    pnl_col="pnl_label",
    mfe_col="mfe",
    mae_col="mae",
):
    tprint(f"Entering function: train in train_pwin.py")
    tprint(f"Building position sizer dataset...")
    ds = build_position_sizer_dataset(
        df,
        feature_cols=feature_cols,
        pnl_col=pnl_col,
        mfe_col=mfe_col,
        mae_col=mae_col,
        pwin_soft_cfg=pwin_soft_cfg,
    )
    regime_labels = df[regime_col].values if regime_col and regime_col in df.columns else None
    tprint(f"Training pwin classifier...")
    model = train_pwin_classifier(
        ds.X,
        ds.pwin_target,
        calibration_mode=calibration_mode,
        regime_labels=regime_labels,
        rolling_window=rolling_window,
        y_hard_ref=ds.y_win,
        pnl_ref=df[pnl_col].values if pnl_col in df.columns else None,
    )
    return model, ds
