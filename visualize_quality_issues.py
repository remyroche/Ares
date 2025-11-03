"""
Quick visualization of the critical quality score issues
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
data = pd.read_parquet('data_cache/sr_ml_training/sr_quality_training_data.parquet')

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('🚨 Quality Score Critical Issues Analysis', fontsize=16, fontweight='bold', y=0.98)

# ========== ROW 1: Component Distributions ==========
# 1. Bounce Strength (ISSUE: Saturated)
ax1 = fig.add_subplot(gs[0, 0])
bounce = data['bounce_strength']
ax1.hist(bounce, bins=50, edgecolor='black', color='red', alpha=0.7)
ax1.axvline(bounce.mean(), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {bounce.mean():.3f}')
ax1.axvline(1.0, color='orange', linestyle='--', linewidth=2, label='Max (1.0)')
ax1.set_title('❌ ISSUE #1: Bounce Strength SATURATED', fontweight='bold', color='red')
ax1.set_xlabel('Bounce Strength')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.text(0.5, 0.95, f'{(bounce >= 0.95).sum()} samples (50%) at ≥0.95!', 
         transform=ax1.transAxes, ha='center', va='top', 
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

# 2. Hold Strength (OK)
ax2 = fig.add_subplot(gs[0, 1])
hold = data['hold_strength']
ax2.hist(hold, bins=50, edgecolor='black', color='green', alpha=0.7)
ax2.axvline(hold.mean(), color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {hold.mean():.3f}')
ax2.set_title('✅ Hold Strength: Good Variance', fontweight='bold', color='green')
ax2.set_xlabel('Hold Strength')
ax2.set_ylabel('Frequency')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.text(0.5, 0.95, f'Good spread: std={hold.std():.3f}', 
         transform=ax2.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

# 3. Trade Profit (ISSUE: Negative)
ax3 = fig.add_subplot(gs[0, 2])
profit = data['trade_profit']
ax3.hist(profit, bins=50, edgecolor='black', color='orange', alpha=0.7)
ax3.axvline(profit.mean(), color='red', linestyle='--', linewidth=3, label=f'Mean: {profit.mean():.3f}')
ax3.axvline(0, color='black', linestyle='-', linewidth=1, label='Zero')
ax3.set_title('❌ ISSUE #2: Trade Profit NEGATIVE', fontweight='bold', color='red')
ax3.set_xlabel('Trade Profit')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.text(0.5, 0.95, f'{(profit < 0).sum()} samples ({(profit < 0).sum()/len(profit)*100:.1f}%) losing!', 
         transform=ax3.transAxes, ha='center', va='top',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))

# ========== ROW 2: Effective Contribution ==========
# 4. Effective contribution of each component
ax4 = fig.add_subplot(gs[1, :])

# Calculate actual contributions
bounce_contrib = bounce * 0.35
hold_contrib = hold * 0.35
profit_contrib = np.maximum(profit, 0) * 0.30  # max(profit, 0) as per formula

components = ['Bounce\n(35%)', 'Hold\n(35%)', 'Trade Profit\n(30%)']
means = [bounce_contrib.mean(), hold_contrib.mean(), profit_contrib.mean()]
stds = [bounce_contrib.std(), hold_contrib.std(), profit_contrib.std()]

x_pos = np.arange(len(components))
bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7, 
               color=['red', 'green', 'orange'], edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
             f'{mean:.3f}\n±{std:.3f}',
             ha='center', va='bottom', fontweight='bold', fontsize=11)

ax4.set_ylabel('Contribution to Quality Score', fontsize=12, fontweight='bold')
ax4.set_title('🧩 Effective Component Contributions (Mean ± Std)', fontsize=14, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(components, fontsize=11)
ax4.grid(True, alpha=0.3, axis='y')
ax4.axhline(0, color='black', linewidth=1)

# Add dominance annotation
ax4.text(1, 0.3, '← DOMINATED BY HOLD ▶', 
         ha='center', va='center', fontsize=14, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

# ========== ROW 3: Correlations and Quality Distribution ==========
# 5. Top feature correlations
ax5 = fig.add_subplot(gs[2, 0:2])

feature_cols = [c for c in data.columns if c.startswith('feature_')]
correlations = data[feature_cols].corrwith(data['quality_score']).abs().sort_values(ascending=False).head(15)

y_pos = np.arange(len(correlations))
bars = ax5.barh(y_pos, correlations.values, color='steelblue', edgecolor='black')

# Color code by strength
for i, (bar, corr) in enumerate(zip(bars, correlations.values)):
    if corr > 0.3:
        bar.set_color('green')
    elif corr > 0.2:
        bar.set_color('orange')
    else:
        bar.set_color('red')

ax5.set_yticks(y_pos)
ax5.set_yticklabels([f.replace('feature_', '') for f in correlations.index], fontsize=9)
ax5.set_xlabel('|Correlation| with Quality Score', fontsize=11, fontweight='bold')
ax5.set_title('🔗 Top 15 Features (only 3 strong!)', fontsize=12, fontweight='bold')
ax5.axvline(0.3, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Strong (>0.3)')
ax5.axvline(0.2, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Moderate (>0.2)')
ax5.legend()
ax5.grid(True, alpha=0.3, axis='x')
ax5.invert_yaxis()

# 6. Quality Score Distribution
ax6 = fig.add_subplot(gs[2, 2])

quality = data['quality_score']
ax6.hist(quality, bins=50, edgecolor='black', color='purple', alpha=0.7)
ax6.axvline(quality.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {quality.mean():.3f}')
ax6.axvline(quality.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {quality.median():.3f}')
ax6.set_xlabel('Quality Score')
ax6.set_ylabel('Frequency')
ax6.set_title('📊 Quality Distribution', fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3)

# Add stats box
stats_text = f'Samples: {len(quality):,}\nStd: {quality.std():.3f}\nAt 1.0: {(quality >= 1.0).sum()} ({(quality >= 1.0).sum()/len(quality)*100:.1f}%)'
ax6.text(0.98, 0.95, stats_text,
         transform=ax6.transAxes, ha='right', va='top',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
         fontsize=9, fontfamily='monospace')

plt.savefig('analysis_output/quality_issues_summary.png', dpi=300, bbox_inches='tight')
print("\n✅ Critical issues visualization saved to: analysis_output/quality_issues_summary.png")
print("\n🚨 SUMMARY OF ISSUES:")
print(f"   1. Bounce strength saturated: {(bounce >= 0.95).sum()} samples (50%) at max")
print(f"   2. Trade profit negative: {(profit < 0).sum()} samples ({(profit < 0).sum()/len(profit)*100:.1f}%) losing")
print(f"   3. Hold dominates: Mean contribution {hold_contrib.mean():.3f} vs bounce {bounce_contrib.mean():.3f}")
print(f"   4. Weak features: Only {(correlations > 0.3).sum()} features with strong correlation")
print("\n💡 See QUALITY_SCORE_INVESTIGATION_FINDINGS.md for detailed fixes!")

