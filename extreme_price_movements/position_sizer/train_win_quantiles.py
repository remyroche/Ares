from .dataset import build_position_sizer_dataset
from .models import train_win_quantile_regressor
from ..utils import tprint


def train(df, feature_cols):
    tprint("Entering function: train in train_win_quantiles.py")
    tprint(f"Building position sizer dataset with {len(df)} rows")
    ds = build_position_sizer_dataset(df, feature_cols=feature_cols)
    tprint(f"Dataset built. X shape: {ds.X.shape}, y_win_mag shape: {ds.y_win_mag.shape}")
    tprint("Training win quantile regressor")
    result = train_win_quantile_regressor(ds.X, ds.y_win_mag)
    tprint("Finished training win quantile regressor")
    return result
