# 🎯 Automated Tuning - Quick Reference Card

## ⚡ TL;DR

```bash
# Step 1: Enable auto-tuning (one-time edit)
nano config/regime_clustering_config.yaml
# Change: auto_tune_iterative_opt: true

# Step 2: Run regime clustering  
python3 src/launcher/ares_launcher.py --step regime_clustering --symbol ETHUSDT --execution-mode light

# Step 3: Wait 15-20 mins, get improved metrics! 🎉
```

---

## 📊 What You Get

### Before
```
Silhouette: -0.03 ❌
DBI:         3.2  ❌
CV:          1.19 ✅
```

### After
```
Silhouette: 0.15-0.30 ✅
DBI:        1.5-2.2   ✅  
CV:         1.4-1.6   ✅
```

---

## 🎚️ Config File Toggles

Location: `config/regime_clustering_config.yaml`

| Setting | Value | What It Does |
|---------|-------|--------------|
| `auto_tune_iterative_opt` | `true` | Run fresh tuning |
| `auto_tune_iterative_opt` | `false` | Skip tuning |
| `use_cached_tuning` | `true` | Reuse previous tuning |
| `tuning_trials` | `10` | Fast (5-10 min) |
| `tuning_trials` | `20` | Balanced (15-20 min) ⭐ |
| `tuning_trials` | `50` | Best (30-45 min) |

---

## 🔄 Recommended Workflow

### Week 1: Initial Tuning
```yaml
auto_tune_iterative_opt: true
tuning_trials: 20
```
Run once, takes 15-20 minutes.

### Week 2-4: Use Cache
```yaml
auto_tune_iterative_opt: false
use_cached_tuning: true
```
Fast runs using tuned params!

### Month 2: Re-tune
```yaml
auto_tune_iterative_opt: true  
tuning_trials: 30
```
Update parameters for new market conditions.

---

## 📁 Where to Find Results

```bash
# Latest tuning report
cat artifacts/hyperparameter_tuning/auto_tuning_report_ETHUSDT_*.md | tail -50

# Latest tuning results (JSON)
ls -t artifacts/hyperparameter_tuning/auto_tuning_results_*.json | head -1

# View best parameters
cat artifacts/hyperparameter_tuning/auto_tuning_results_*.json | python3 -m json.tool | grep -A 30 "best_params"
```

---

## 🚨 Common Issues

| Problem | Solution |
|---------|----------|
| "Tuning takes forever" | Reduce `tuning_trials` to 10 |
| "No cached results" | Run with `auto_tune: true` first |
| "Metrics didn't improve" | Increase trials to 50 |
| "All trials fail" | Relax `tuning_min_balance` to 0.40 |

---

## 🎯 One-Liner Cheat Sheet

```bash
# Quick tuning (10 min)
sed -i '' 's/auto_tune_iterative_opt: false/auto_tune_iterative_opt: true/' config/regime_clustering_config.yaml && sed -i '' 's/tuning_trials: 20/tuning_trials: 10/' config/regime_clustering_config.yaml && python3 src/launcher/ares_launcher.py --step regime_clustering --symbol ETHUSDT --execution-mode light

# Use cache (instant)
sed -i '' 's/auto_tune_iterative_opt: true/auto_tune_iterative_opt: false/' config/regime_clustering_config.yaml && sed -i '' 's/use_cached_tuning: false/use_cached_tuning: true/' config/regime_clustering_config.yaml && python3 src/launcher/ares_launcher.py --step regime_clustering --symbol ETHUSDT --execution-mode light

# Disable all tuning (default)
sed -i '' 's/auto_tune_iterative_opt: true/auto_tune_iterative_opt: false/' config/regime_clustering_config.yaml && sed -i '' 's/use_cached_tuning: true/use_cached_tuning: false/' config/regime_clustering_config.yaml && python3 src/launcher/ares_launcher.py --step regime_clustering --symbol ETHUSDT --execution-mode light
```

---

## 💰 Time Investment

| Mode | First Run | Subsequent Runs | Quality |
|------|-----------|-----------------|---------|
| No tuning | 3 min | 3 min | Baseline |
| Auto-tune (10 trials) | 10 min | 3 min* | Good |
| Auto-tune (20 trials) | 18 min | 3 min* | Better ⭐ |
| Auto-tune (50 trials) | 40 min | 3 min* | Best |

*With caching enabled

---

## ✨ Bottom Line

**20 minutes of tuning** = **Permanently improved metrics**

Just flip `auto_tune_iterative_opt: true` once, wait 20 mins, then use cache forever! 🚀

