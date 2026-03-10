import re

with open('extreme_price_movements/feature_views.py', 'r') as f:
    content = f.read()

# Enhance it to have a proper feature metadata registry check or just structural rules as currently written?
# The spec asks for a Feature Metadata Registry with view eligibility flags. We should probably create a quick generation of that registry or use the explicit logic.
# "Build feature metadata registry with view eligibility flags"
# We can do this implicitly by inspecting the feature string, which is highly robust.
# But let's build a dict explicitly if requested.
