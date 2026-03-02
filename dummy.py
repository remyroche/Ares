import pandas as pd
import numpy as np

# Just test whether the columns are all present in grp_df.
# Wait, X_meta takes num = grp_df.select_dtypes(include=[np.number]).copy()
# The issue is that some features are non-numeric or they aren't computed on `grp_df`?
# Features missing usually include stuff from feats (market or custom features)
# that we didn't add to `num` before prepare_meta_features.
