from .dataset import build_position_sizer_dataset
from .models import train_loss_quantile_regressor
from ..utils import tprint


def train(df, feature_cols):
    tprint(f"Entering function: train in train_loss_quantiles.py")
    ds = build_position_sizer_dataset(df, feature_cols=feature_cols)
    return train_loss_quantile_regressor(ds.X, ds.y_loss_mag)
