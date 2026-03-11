# ML Experiment Discipline

This document defines the **minimum protocol for machine learning experiments** in this repository.

All experiments must prioritize:

1. statistical validity
2. out-of-sample robustness
3. economic relevance
4. reproducibility

Experiments that violate these rules are considered **invalid research results**.

---

# 1. Dataset Versioning

Every experiment must reference a **specific dataset version**.

Required metadata:

dataset_version  
feature_pipeline  
universe_definition  
bar_frequency  
target_definition  

Datasets must be **immutable** once published.

---

# 2. Train / Validation / Test Splits

All ML experiments must use **time-based splits**.

Example:

train        2010-2018  
validation   2019-2020  
test         2021-2023  

Rules:

- The **test set must never be used for hyperparameter tuning**
- Validation is used only for **model selection**
- Test is used **once** for final evaluation

---

# 3. Walk-Forward Evaluation

Models must be evaluated using **rolling or expanding windows**.

Example:

train: 2010–2016 → test: 2017  
train: 2010–2017 → test: 2018  
train: 2010–2018 → test: 2019  

Single split backtests are insufficient.

---

# 4. Feature Causality

Features must only use **information available at time t**.

Valid:

feature_t = f(data ≤ t)

Invalid:

feature_t = f(data ≥ t)

Common leakage sources:

- future prices
- forward-filled labels
- global normalization
- improperly aligned rolling statistics

---

# 5. Hyperparameter Search

Hyperparameters must be tuned **only on validation data**.

Procedure:

1. train models on training data
2. evaluate on validation
3. select best configuration
4. evaluate once on test

Repeated evaluation on test is prohibited.

---

# 6. Reproducibility

Every experiment must record:

experiment_id  
dataset_version  
feature_set  
model_type  
hyperparameters  
random_seed  
training_window  
evaluation_window  
git_commit  

Re-running an experiment must reproduce identical results.

---

# 7. Random Seeds

All stochastic components must use fixed seeds.

Examples:

numpy  
torch  
sklearn  
model initialization

---

# 8. Economic Evaluation

Model evaluation must include **trading performance metrics**.

Required metrics:

Sharpe ratio  
maximum drawdown  
turnover  
transaction cost sensitivity  

Statistical metrics alone are insufficient.

---

# 9. Experiment Tracking

All experiments must be logged.

Required fields:

experiment_id  
timestamp  
dataset_version  
features  
model  
parameters  
metrics  

Results must be reproducible from stored metadata.

---

# 10. Research Discipline

Avoid excessive researcher degrees of freedom.

Guidelines:

- change one major variable per experiment
- log every experiment
- prefer simpler models
- verify robustness across multiple periods
