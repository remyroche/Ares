import json

with open('./data/artifacts/20260214_190000/fs_reports/meta_clf_short_tf/selected_features.json', 'r') as f:
    data = json.load(f)
    print("Short TF meta selected features:", len(data['selected_features']))

try:
    with open('./data/artifacts/20260214_190000/fs_reports/meta_long_tf_reg/selected_features.json', 'r') as f:
        data = json.load(f)
        print("Long TF meta reg selected features:", len(data['selected_features']))
except Exception as e:
    print(e)
