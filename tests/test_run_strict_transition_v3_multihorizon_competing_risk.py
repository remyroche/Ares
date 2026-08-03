import pandas as pd
from scripts.run_strict_transition_v3_multihorizon_competing_risk import stable_features
def test_stable_features_train_only():
 d=pd.DataFrame({'source_utc':pd.date_range('2022-01-01',periods=60,tz='UTC',freq='30D'),'x':range(60),'target__x':[0]*60,'state_context__x':[0]*60})
 assert stable_features(d,d)==['x']
