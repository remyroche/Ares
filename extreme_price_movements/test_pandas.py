import pandas as pd
import numpy as np

def run():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]}, dtype='float32')
    print(df.rolling(2).max())
run()
