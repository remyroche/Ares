import pandas as pd
from extreme_price_movements.features import add_cross_sectional_peer_context_features
import numpy as np

# Test data
df1 = pd.DataFrame({
    'ret1h': [0.01, 0.02, 0.03, 0.05, -0.01, 0.10],
    'ret24h': [0.1, -0.1, 0.2, 0.05, 0.0, -0.05]
})

df = pd.DataFrame(np.random.rand(10, 10))

feats = {
    "ret1h": df.copy(),
}

res = add_cross_sectional_peer_context_features(feats, min_group_size=5)
print(res.keys())
