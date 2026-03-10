import re

with open('extreme_price_movements/position_sizer_v2.py', 'r') as f:
    content = f.read()

# Add get_feature_view import
content = content.replace(
    'from extreme_price_movements.config import POSITION_SIZER_V2_FEATURE_CONFIG',
    'from extreme_price_movements.config import POSITION_SIZER_V2_FEATURE_CONFIG\nfrom extreme_price_movements.feature_views import get_feature_view'
)

# Apply linear view to model1, model2, model3 feature keys
# Ridge Position Sizer uses Ridge regression, so X_linear is the appropriate view.
# X1 = assemble_feature_matrix(feature_dict, POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"])
# -> X1 = assemble_feature_matrix(feature_dict, get_feature_view(POSITION_SIZER_V2_FEATURE_CONFIG["model1_edge_feature_keys"], "X_linear"))

content = re.sub(
    r'POSITION_SIZER_V2_FEATURE_CONFIG\["model([123])_(edge|downside|uncertainty)_feature_keys"\]',
    r'get_feature_view(POSITION_SIZER_V2_FEATURE_CONFIG["model\1_\2_feature_keys"], "X_linear")',
    content
)

with open('extreme_price_movements/position_sizer_v2.py', 'w') as f:
    f.write(content)
