#!/usr/bin/env python3
"""
Verify that interaction features are in the selected top 60 features.
"""
import sys
import glob

# Find the latest outcome report
reports = glob.glob("outcomes/final_feature_selection_outcome_report_report_*.md")
if not reports:
    print("❌ No outcome reports found")
    sys.exit(1)

latest_report = max(reports)
print(f"📄 Reading: {latest_report}\n")

with open(latest_report, 'r') as f:
    content = f.read()

# Extract the 60 features section
in_60_section = False
features_60 = []

for line in content.split('\n'):
    if '### 60 Features Set' in line:
        in_60_section = True
        continue
    if in_60_section:
        if line.startswith('### '):  # Next section
            break
        if line.strip() and line.strip()[0].isdigit() and '. ' in line:
            # Extract feature name
            feature = line.split('. ', 1)[1].strip()
            features_60.append(feature)

print(f"✅ Found {len(features_60)} features in top 60 set\n")

# Check for interaction features
interaction_features = []
for feat in features_60:
    if 'interaction' in feat.lower() or '_x_' in feat.lower() or '_minus_' in feat or '_log_' in feat or '_div_' in feat:
        interaction_features.append(feat)

print(f"🔍 Interaction Features Analysis:")
print(f"   Total features in top 60: {len(features_60)}")
print(f"   Interaction features: {len(interaction_features)}")
print(f"   Percentage: {len(interaction_features)/len(features_60)*100:.1f}%\n")

if interaction_features:
    print(f"✅ SUCCESS: {len(interaction_features)} interaction features in top 60!\n")
    print("📊 Sample interaction features:")
    for i, feat in enumerate(interaction_features[:10], 1):
        print(f"   {i}. {feat}")
    if len(interaction_features) > 10:
        print(f"   ... and {len(interaction_features) - 10} more")
else:
    print("❌ FAILURE: No interaction features found in top 60")
    sys.exit(1)

# Check stability analysis
if 'Stability analysis' in content or 'stable features' in content.lower():
    print("\n✅ Stability analysis is included in the report")
else:
    print("\n⚠️ Stability analysis not found in report")
