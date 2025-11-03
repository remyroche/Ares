"""
Diagnose Why the SR Quality Model Failed

Critical issues to investigate:
1. R² = -0.003 (worse than predicting mean)
2. Only 215 samples (too small)
3. 34% win rate with 2:1 R/R (barely breakeven)
4. Model learned nothing useful
"""

import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr, pearsonr

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Diagnose model failure."""
    
    logger.info("="*80)
    logger.info("🔬 DIAGNOSING MODEL FAILURE")
    logger.info("="*80)
    
    # Load the training data
    data_path = 'data_cache/sr_ml_training/sr_quality_SIMPLIFIED_20251102_202022.parquet'
    
    try:
        data = pd.read_parquet(data_path)
        logger.info(f"\n✅ Loaded data: {len(data)} samples")
    except Exception as e:
        logger.error(f"❌ Cannot load data: {e}")
        return
    
    # ==========================================================================
    # ISSUE 1: Sample Size
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🔍 ISSUE 1: SAMPLE SIZE")
    logger.info("="*80)
    
    feature_cols = [c for c in data.columns if c.startswith('feature_')]
    n_samples = len(data)
    n_features = len(feature_cols)
    
    ratio = n_samples / n_features
    
    logger.info(f"\n  Samples: {n_samples}")
    logger.info(f"  Features: {n_features}")
    logger.info(f"  Ratio: {ratio:.1f} samples per feature")
    
    logger.info(f"\n  📊 Sample Size Guidelines:")
    logger.info(f"     Minimum: {n_features * 10} samples (10 per feature)")
    logger.info(f"     Good: {n_features * 50} samples (50 per feature)")
    logger.info(f"     Excellent: {n_features * 100}+ samples (100+ per feature)")
    
    if ratio < 10:
        logger.error(f"  ❌ CRITICAL: {ratio:.1f} samples/feature is WAY too low!")
        logger.error(f"     Need at least {n_features * 10} samples (have {n_samples})")
        logger.error(f"     This GUARANTEES poor model performance!")
    elif ratio < 50:
        logger.warning(f"  ⚠️  WARNING: {ratio:.1f} samples/feature is marginal")
        logger.warning(f"     Recommend {n_features * 50}+ samples for reliable training")
    else:
        logger.info(f"  ✅ {ratio:.1f} samples/feature is adequate")
    
    # ==========================================================================
    # ISSUE 2: Target Distribution
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🔍 ISSUE 2: TARGET DISTRIBUTION")
    logger.info("="*80)
    
    target = data['realized_pnl_pct']
    
    logger.info(f"\n  realized_pnl_pct statistics:")
    logger.info(f"     Mean: {target.mean()*100:.4f}%")
    logger.info(f"     Std:  {target.std()*100:.4f}%")
    logger.info(f"     Min:  {target.min()*100:.2f}%")
    logger.info(f"     Max:  {target.max()*100:.2f}%")
    
    # Check for problems
    wins = (target > 0).sum()
    losses = (target < 0).sum()
    zeros = (target == 0).sum()
    
    win_rate = wins / len(target)
    
    logger.info(f"\n  Outcome distribution:")
    logger.info(f"     Wins: {wins} ({wins/len(target)*100:.1f}%)")
    logger.info(f"     Losses: {losses} ({losses/len(target)*100:.1f}%)")
    logger.info(f"     Zeros: {zeros} ({zeros/len(target)*100:.1f}%)")
    logger.info(f"     Win rate: {win_rate*100:.1f}%")
    
    # Check if target has variance
    if target.std() == 0:
        logger.error(f"  ❌ CRITICAL: Target has NO VARIANCE!")
        logger.error(f"     All values are identical - cannot train!")
    elif target.std() < 0.001:
        logger.warning(f"  ⚠️  WARNING: Very low target variance ({target.std()*100:.4f}%)")
        logger.warning(f"     Model will struggle to learn patterns")
    else:
        logger.info(f"  ✅ Target has variance: {target.std()*100:.2f}%")
    
    # Check win rate vs R/R ratio
    logger.info(f"\n  📊 Win Rate Analysis (2:1 R/R):")
    logger.info(f"     Breakeven: 33.3% (need at least this)")
    logger.info(f"     Actual: {win_rate*100:.1f}%")
    
    if win_rate < 0.333:
        logger.error(f"  ❌ LOSING STRATEGY: Win rate too low for 2:1 R/R!")
        logger.error(f"     Expected value is negative")
    elif win_rate < 0.40:
        logger.warning(f"  ⚠️  MARGINAL: Win rate barely profitable")
    else:
        logger.info(f"  ✅ Profitable strategy (win rate > 40%)")
    
    # ==========================================================================
    # ISSUE 3: Feature-Target Correlation
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🔍 ISSUE 3: FEATURE-TARGET CORRELATION")
    logger.info("="*80)
    
    logger.info(f"\n  Testing if features have ANY predictive power...")
    
    correlations = []
    
    for col in feature_cols:
        try:
            # Pearson correlation
            pearson_r, pearson_p = pearsonr(data[col], target)
            
            # Spearman correlation (rank-based, more robust)
            spearman_r, spearman_p = spearmanr(data[col], target)
            
            correlations.append({
                'feature': col,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'abs_corr': abs(spearman_r)
            })
        except:
            pass
    
    corr_df = pd.DataFrame(correlations).sort_values('abs_corr', ascending=False)
    
    logger.info(f"\n  Top 10 most correlated features:")
    logger.info(f"  {'Feature':<40} {'Spearman ρ':>12} {'p-value':>10}")
    logger.info(f"  {'-'*65}")
    
    for idx, row in corr_df.head(10).iterrows():
        sig = "***" if row['spearman_p'] < 0.001 else ("**" if row['spearman_p'] < 0.01 else ("*" if row['spearman_p'] < 0.05 else ""))
        logger.info(f"  {row['feature']:<40} {row['spearman_r']:>12.4f} {row['spearman_p']:>10.4f} {sig}")
    
    # Check if ANY features are predictive
    significant_features = corr_df[corr_df['spearman_p'] < 0.05]
    
    logger.info(f"\n  📊 Significant features (p<0.05): {len(significant_features)}/{len(feature_cols)}")
    
    if len(significant_features) == 0:
        logger.error(f"  ❌ CRITICAL: NO features are significantly correlated with target!")
        logger.error(f"     Features have NO predictive power!")
        logger.error(f"     Model CANNOT learn anything!")
    elif len(significant_features) < 3:
        logger.warning(f"  ⚠️  WARNING: Only {len(significant_features)} significant features")
        logger.warning(f"     Very weak signal - model will struggle")
    else:
        logger.info(f"  ✅ {len(significant_features)} features show predictive signal")
    
    # Best possible R² from correlation
    best_corr = corr_df['abs_corr'].max()
    theoretical_r2 = best_corr ** 2
    
    logger.info(f"\n  📊 Theoretical maximum R²:")
    logger.info(f"     Best correlation: {best_corr:.4f}")
    logger.info(f"     Max possible R²: {theoretical_r2:.4f}")
    logger.info(f"     Actual R²: -0.003")
    
    if theoretical_r2 < 0.05:
        logger.error(f"  ❌ FUNDAMENTAL PROBLEM: Even best feature has R²<0.05!")
        logger.error(f"     Features are NOT predictive of realized_pnl_pct")
        logger.error(f"     Need better features or different approach!")
    
    # ==========================================================================
    # ISSUE 4: Data Quality
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🔍 ISSUE 4: DATA QUALITY")
    logger.info("="*80)
    
    # Check for missing values
    missing = data[feature_cols].isnull().sum().sum()
    logger.info(f"\n  Missing values: {missing}")
    
    # Check for zero-variance features
    zero_var = (data[feature_cols].std() == 0).sum()
    logger.info(f"  Zero-variance features: {zero_var}")
    
    # Check for outliers in target
    q1, q99 = target.quantile([0.01, 0.99])
    outliers = ((target < q1) | (target > q99)).sum()
    logger.info(f"  Outliers in target: {outliers} ({outliers/len(target)*100:.1f}%)")
    
    # ==========================================================================
    # ISSUE 5: Overfitting vs Underfitting
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🔍 ISSUE 5: OVERFITTING vs UNDERFITTING")
    logger.info("="*80)
    
    # With 215 samples and 19 features, and R² ≈ 0...
    logger.info(f"\n  Given:")
    logger.info(f"     Samples: {n_samples}")
    logger.info(f"     Features: {n_features}")
    logger.info(f"     R²: -0.003")
    
    logger.info(f"\n  Diagnosis:")
    
    if ratio < 10:
        logger.error(f"  ❌ SEVERE UNDERFITTING (not enough data)")
        logger.error(f"     Model cannot learn patterns with so few samples")
    else:
        logger.warning(f"  ⚠️  Likely underfitting OR features not predictive")
    
    # ==========================================================================
    # ROOT CAUSE ANALYSIS
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("💡 ROOT CAUSE ANALYSIS")
    logger.info("="*80)
    
    logger.info(f"\n  Most Likely Causes (in order of probability):")
    
    logger.info(f"\n  1. ❌ INSUFFICIENT DATA (11.3 samples/feature)")
    logger.info(f"     • Have: 215 samples")
    logger.info(f"     • Need: 950+ samples (50 per feature)")
    logger.info(f"     • Impact: Cannot learn patterns")
    logger.info(f"     • Fix: Collect 5-10x more data")
    
    logger.info(f"\n  2. ❌ FEATURES NOT PREDICTIVE")
    logger.info(f"     • Best correlation: {best_corr:.4f}")
    logger.info(f"     • Max possible R²: {theoretical_r2:.4f}")
    logger.info(f"     • Impact: No signal in features")
    logger.info(f"     • Fix: Add better features (price action, regime, etc.)")
    
    logger.info(f"\n  3. ⚠️  WEAK WIN RATE (34% with 2:1 R/R)")
    logger.info(f"     • Breakeven: 33.3%")
    logger.info(f"     • Actual: 34%")
    logger.info(f"     • Impact: Barely profitable strategy")
    logger.info(f"     • Fix: Improve SR detection or relax SL/TP")
    
    logger.info(f"\n  4. ⚠️  HIGH NOISE IN TARGET")
    logger.info(f"     • Target std: {target.std()*100:.2f}%")
    logger.info(f"     • Target mean: {target.mean()*100:.2f}%")
    logger.info(f"     • Noise ratio: {target.std()/abs(target.mean()):.1f}x")
    logger.info(f"     • Impact: Signal-to-noise ratio too low")
    logger.info(f"     • Fix: Aggregate multiple trades, use risk-adjusted metrics")
    
    # ==========================================================================
    # RECOMMENDATIONS
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🚀 RECOMMENDATIONS")
    logger.info("="*80)
    
    logger.info(f"\n  IMMEDIATE ACTIONS:")
    
    logger.info(f"\n  1. 🎯 COLLECT MORE DATA (Priority 1)")
    logger.info(f"     Current: 215 samples")
    logger.info(f"     Target: 1,000+ samples")
    logger.info(f"     How:")
    logger.info(f"       • Increase date range (2024-01-01 to 2024-12-01)")
    logger.info(f"       • Sample more frequently (daily instead of weekly)")
    logger.info(f"       • Add more symbols (BTC, ETH, SOL, etc.)")
    logger.info(f"       • Add more timeframes (1h, 4h, 1d)")
    
    logger.info(f"\n  2. 🔧 TEST TRADING PARAMETERS")
    logger.info(f"     Current: SL=0.5%, TP=1.0% (2:1 R/R)")
    logger.info(f"     Win rate: 34% (barely breakeven)")
    logger.info(f"     Try:")
    logger.info(f"       • 1:1 R/R: SL=1.0%, TP=1.0% (need 50% win rate)")
    logger.info(f"       • 1.5:1 R/R: SL=0.75%, TP=1.0% (need 40% win rate)")
    logger.info(f"       • Wider SL: SL=1.0%, TP=2.0% (need 33% win rate)")
    
    logger.info(f"\n  3. 📊 ADD BETTER FEATURES")
    logger.info(f"     Current features have low correlation (max {best_corr:.4f})")
    logger.info(f"     Add:")
    logger.info(f"       • Price action context (trend strength, candle patterns)")
    logger.info(f"       • Market regime (volatility state, trend state)")
    logger.info(f"       • Order flow (volume profile, delta)")
    logger.info(f"       • Multi-timeframe confirmation")
    logger.info(f"       • Recent SR performance (hit rate in last N bars)")
    
    logger.info(f"\n  4. 🎲 TRY ALTERNATIVE TARGETS")
    logger.info(f"     Current target (single trade P&L) is very noisy")
    logger.info(f"     Try:")
    logger.info(f"       • Average of next 3 trades (reduces noise)")
    logger.info(f"       • Sharpe ratio (risk-adjusted)")
    logger.info(f"       • Binary: profitable vs unprofitable (classification)")
    logger.info(f"       • Hit rate in next 10 days (simpler target)")
    
    logger.info(f"\n  5. 📈 USE SIMPLER MODEL FIRST")
    logger.info(f"     With 215 samples, LightGBM is overkill")
    logger.info(f"     Try:")
    logger.info(f"       • Linear regression (establishes baseline)")
    logger.info(f"       • Ridge/Lasso (with regularization)")
    logger.info(f"       • Random forest with max_depth=3")
    
    # ==========================================================================
    # QUICK TEST: Can ANY model work with this data?
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🧪 QUICK TEST: Baseline Model Performance")
    logger.info("="*80)
    
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    
    X = data[feature_cols].fillna(0)
    y = target
    
    logger.info(f"\n  Testing simple models with cross-validation...")
    
    # Test 1: Predict mean (baseline)
    mean_prediction = np.full(len(y), y.mean())
    baseline_r2 = 1 - (np.sum((y - mean_prediction)**2) / np.sum((y - y.mean())**2))
    logger.info(f"\n  Baseline (predict mean): R² = {baseline_r2:.4f}")
    
    # Test 2: Linear regression
    try:
        ridge = Ridge(alpha=1.0)
        scores_ridge = cross_val_score(ridge, X, y, cv=3, scoring='r2')
        logger.info(f"  Ridge regression: R² = {scores_ridge.mean():.4f} ± {scores_ridge.std():.4f}")
    except Exception as e:
        logger.error(f"  Ridge failed: {e}")
    
    # Test 3: Simple random forest
    try:
        rf = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        scores_rf = cross_val_score(rf, X, y, cv=3, scoring='r2')
        logger.info(f"  Random Forest (shallow): R² = {scores_rf.mean():.4f} ± {scores_rf.std():.4f}")
    except Exception as e:
        logger.error(f"  Random Forest failed: {e}")
    
    logger.info(f"\n  💡 Interpretation:")
    if scores_ridge.mean() < 0 and scores_rf.mean() < 0:
        logger.error(f"  ❌ FUNDAMENTAL PROBLEM: Even simple models fail!")
        logger.error(f"     This confirms features are NOT predictive of target")
        logger.error(f"     Need to collect different features or more data")
    
    # ==========================================================================
    # FINAL DIAGNOSIS
    # ==========================================================================
    
    logger.info("\n" + "="*80)
    logger.info("🎯 FINAL DIAGNOSIS")
    logger.info("="*80)
    
    logger.info(f"\n  The model failed because:")
    
    logger.info(f"\n  1. ❌ INSUFFICIENT DATA")
    logger.info(f"     • 215 samples ÷ 19 features = 11 samples/feature")
    logger.info(f"     • Need 950+ samples (50 per feature minimum)")
    logger.info(f"     • With current data, model CANNOT learn")
    
    logger.info(f"\n  2. ❌ WEAK PREDICTIVE SIGNAL")
    logger.info(f"     • Best feature correlation: {best_corr:.4f}")
    logger.info(f"     • Max theoretical R²: {theoretical_r2:.4f}")
    logger.info(f"     • Features don't predict profit well")
    
    logger.info(f"\n  3. ⚠️  NOISY TARGET")
    logger.info(f"     • Single trade P&L is very noisy")
    logger.info(f"     • Noise/signal ratio: {target.std()/abs(target.mean()):.1f}x")
    logger.info(f"     • Hard for model to find patterns")
    
    logger.info(f"\n  4. ⚠️  MARGINAL WIN RATE")
    logger.info(f"     • 34% win rate with 2:1 R/R barely profitable")
    logger.info(f"     • Strategy itself may be weak")
    
    logger.info("\n" + "="*80)
    logger.info("✅ RECOMMENDED FIXES (in priority order)")
    logger.info("="*80)
    
    logger.info(f"\n  🥇 PRIORITY 1: Collect 5-10x more data")
    logger.info(f"     Target: 1,000-2,000 samples")
    logger.info(f"     • Extend date range to full year")
    logger.info(f"     • Add multiple symbols")
    logger.info(f"     • Sample more frequently")
    
    logger.info(f"\n  🥈 PRIORITY 2: Test different trading parameters")
    logger.info(f"     Try 1:1 R/R (SL=1%, TP=1%)")
    logger.info(f"     • Easier to hit TP → higher win rate")
    logger.info(f"     • More balanced risk/reward")
    
    logger.info(f"\n  🥉 PRIORITY 3: Add better features")
    logger.info(f"     Current features are weak predictors")
    logger.info(f"     • Add price action features")
    logger.info(f"     • Add regime indicators")
    logger.info(f"     • Add recent SR performance")
    
    logger.info(f"\n  4️⃣  Consider alternative targets")
    logger.info(f"     Single trade P&L is noisy")
    logger.info(f"     • Use average of multiple trades")
    logger.info(f"     • Use binary classification (win/loss)")
    logger.info(f"     • Use hit rate prediction")
    
    logger.info("\n" + "="*80)


if __name__ == '__main__':
    main()

