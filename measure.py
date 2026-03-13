import time
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': np.random.rand(100000), 'b': np.random.rand(100000)})

start = time.time()
for _, row in df.iterrows():
    pass
end = time.time()
print(f"iterrows: {end - start:.4f}s")

start = time.time()
for row in df.itertuples():
    pass
end = time.time()
print(f"itertuples: {end - start:.4f}s")
