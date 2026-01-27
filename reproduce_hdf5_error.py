import pandas as pd
import numpy as np
import os

try:
    # Create a wide dataframe (more than 512 columns)
    num_cols = 600
    df = pd.DataFrame(np.random.randn(10, num_cols), columns=[f'col_{i}' for i in range(num_cols)])

    filename = 'test_wide_df.h5'

    print(f"Attempting to save dataframe with {num_cols} columns to {filename} using format='table'...")

    with pd.HDFStore(filename, mode='w') as store:
        store.put('data', df, format='table', data_columns=True)

    print("Success!")

except Exception as e:
    print(f"Failed: {e}")
finally:
    if os.path.exists('test_wide_df.h5'):
        os.remove('test_wide_df.h5')
