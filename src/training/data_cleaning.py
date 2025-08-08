import pandas as pd
from typing import Optional

def handle_missing_data(df: pd.DataFrame, strategy: str = 'fill', fill_value: Optional[float] = 0) -> pd.DataFrame:
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'fill':
        return df.fillna(fill_value)
    # Placeholder for advanced imputation (KNN, MICE, etc.)
    # elif strategy == 'impute':
    #     ...
    return df