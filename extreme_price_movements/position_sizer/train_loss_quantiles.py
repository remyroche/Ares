from .dataset import build_position_sizer_dataset
from .models import train_loss_quantile_regressor


def train(df, feature_cols):
    ds = build_position_sizer_dataset(df, feature_cols=feature_cols)
    return train_loss_quantile_regressor(ds.X, ds.y_loss_mag)
