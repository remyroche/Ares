from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class PurgedKFoldTime:
    """Purged and Embargoed K-Fold for DatetimeIndex time series.

    - Splits data into sequential folds by time order.
    - For each validation fold, removes from the training set any samples whose
      timestamps fall within [val_start - purge, val_end + embargo].
    - If index is not DatetimeIndex, falls back to sample-count-based purge/embargo
      interpreted as number of rows.
    """

    n_splits: int = 5
    purge: pd.Timedelta | int = pd.Timedelta(minutes=30)
    embargo: pd.Timedelta | int = pd.Timedelta(minutes=15)
