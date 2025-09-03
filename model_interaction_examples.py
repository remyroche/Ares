"""
Examples of ML Models Interacting with Each Other
=================================================

This module demonstrates various techniques for having ML models interact,
including stacking, meta-learning, and sequential model training.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import seaborn as sns


# 1. BASIC STACKING EXAMPLE
# ========================
def basic_stacking_example():
    """
    Demonstrates basic stacking where first-level models' predictions
    are used as features for a second-level meta-model.
    """
    print("=== BASIC STACKING EXAMPLE ===\n")
    
    # Generate synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define base models
    base_models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'svm': SVC(probability=True, random_state=42)
    }
    
    # Train base models and collect predictions
    base_predictions_train = np.zeros((len(X_train), len(base_models)))
    base_predictions_test = np.zeros((len(X_test), len(base_models)))
    
    for idx, (name, model) in enumerate(base_models.items()):
        print(f"Training base model: {name}")
        model.fit(X_train, y_train)
        
        # Get probability predictions (using positive class probability)
        base_predictions_train[:, idx] = model.predict_proba(X_train)[:, 1]
        base_predictions_test[:, idx] = model.predict_proba(X_test)[:, 1]
        
        # Individual model performance
        individual_acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name} accuracy: {individual_acc:.4f}")
    
    # Train meta-model using base model predictions
    print("\nTraining meta-model...")
    meta_model = LogisticRegression(random_state=42)
    meta_model.fit(base_predictions_train, y_train)
    
    # Make final predictions
    final_predictions = meta_model.predict(base_predictions_test)
    stacking_acc = accuracy_score(y_test, final_predictions)
    print(f"\nStacking accuracy: {stacking_acc:.4f}")
    
    return base_models, meta_model, X_test, y_test, base_predictions_test


# 2. ADVANCED STACKING WITH ORIGINAL FEATURES
# ===========================================
def stacking_with_original_features():
    """
    Demonstrates stacking where both original features and base model
    predictions are used as inputs to the meta-model.
    """
    print("\n\n=== STACKING WITH ORIGINAL FEATURES ===\n")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Base models
    base_models = {
        'rf': RandomForestClassifier(n_estimators=50, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    # Train base models
    base_predictions_train = []
    base_predictions_test = []
    
    for name, model in base_models.items():
        model.fit(X_train, y_train)
        base_predictions_train.append(model.predict_proba(X_train)[:, 1].reshape(-1, 1))
        base_predictions_test.append(model.predict_proba(X_test)[:, 1].reshape(-1, 1))
    
    # Combine original features with base model predictions
    X_train_stacked = np.hstack([X_train] + base_predictions_train)
    X_test_stacked = np.hstack([X_test] + base_predictions_test)
    
    # Train meta-model on combined features
    meta_model = RandomForestClassifier(n_estimators=100, random_state=42)
    meta_model.fit(X_train_stacked, y_train)
    
    # Evaluate
    final_predictions = meta_model.predict(X_test_stacked)
    accuracy = accuracy_score(y_test, final_predictions)
    print(f"Stacking with original features accuracy: {accuracy:.4f}")
    
    # Feature importance analysis
    feature_importance = meta_model.feature_importances_
    original_importance = feature_importance[:X.shape[1]].mean()
    model_pred_importance = feature_importance[X.shape[1]:].mean()
    
    print(f"\nAverage importance of original features: {original_importance:.4f}")
    print(f"Average importance of model predictions: {model_pred_importance:.4f}")


# 3. SEQUENTIAL MODEL TRAINING (BOOSTING-LIKE)
# ============================================
def sequential_model_training():
    """
    Demonstrates sequential training where each model learns from
    the errors of the previous model.
    """
    print("\n\n=== SEQUENTIAL MODEL TRAINING ===\n")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Store models and their weights
    sequential_models = []
    model_weights = []
    
    # Current targets (initially the true labels)
    current_targets = y_train.copy()
    
    # Train models sequentially
    n_iterations = 3
    for i in range(n_iterations):
        print(f"\nTraining model {i+1}...")
        
        # Train model on current targets
        model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42+i)
        model.fit(X_train, current_targets)
        
        # Get predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        # Calculate error rate
        error_rate = np.mean(train_pred != y_train)
        model_weight = 0.5 * np.log((1 - error_rate) / (error_rate + 1e-10))
        
        sequential_models.append(model)
        model_weights.append(model_weight)
        
        # Update targets to focus on misclassified examples
        misclassified_mask = train_pred != y_train
        current_targets = y_train.copy()
        
        # Individual model performance
        acc = accuracy_score(y_test, test_pred)
        print(f"Model {i+1} accuracy: {acc:.4f}, weight: {model_weight:.4f}")
    
    # Make ensemble predictions
    ensemble_pred = np.zeros(len(X_test))
    for model, weight in zip(sequential_models, model_weights):
        ensemble_pred += weight * (2 * model.predict(X_test) - 1)
    
    final_predictions = (ensemble_pred > 0).astype(int)
    ensemble_acc = accuracy_score(y_test, final_predictions)
    print(f"\nSequential ensemble accuracy: {ensemble_acc:.4f}")


# 4. MULTI-STAGE PIPELINE
# =======================
def multi_stage_pipeline():
    """
    Demonstrates a multi-stage pipeline where models are trained
    for different purposes and their outputs are combined.
    """
    print("\n\n=== MULTI-STAGE PIPELINE ===\n")
    
    # Generate data with some structure
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=5, n_clusters_per_class=2,
                              n_classes=3, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Stage 1: Difficulty assessment model
    # Train a model to predict if a sample is "easy" or "hard" to classify
    print("Stage 1: Training difficulty assessment model...")
    
    # Create difficulty labels based on k-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    difficulty_scores = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        fold_model = RandomForestClassifier(n_estimators=50, random_state=42)
        fold_model.fit(X_train[train_idx], y_train[train_idx])
        
        # Get prediction probabilities for validation set
        val_probs = fold_model.predict_proba(X_train[val_idx])
        # Difficulty score = entropy of prediction
        difficulty_scores[val_idx] = -np.sum(val_probs * np.log(val_probs + 1e-10), axis=1)
    
    # Create binary difficulty labels (easy=0, hard=1)
    difficulty_threshold = np.percentile(difficulty_scores, 70)
    difficulty_labels = (difficulty_scores > difficulty_threshold).astype(int)
    
    # Train difficulty predictor
    difficulty_model = RandomForestClassifier(n_estimators=100, random_state=42)
    difficulty_model.fit(X_train, difficulty_labels)
    
    # Stage 2: Specialized models for easy and hard cases
    print("\nStage 2: Training specialized models...")
    
    # Predict difficulty for all samples
    train_difficulty = difficulty_model.predict(X_train)
    test_difficulty = difficulty_model.predict(X_test)
    
    # Train specialized models
    easy_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    hard_model = GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Train on respective subsets
    easy_mask_train = train_difficulty == 0
    hard_mask_train = train_difficulty == 1
    
    if np.sum(easy_mask_train) > 0:
        easy_model.fit(X_train[easy_mask_train], y_train[easy_mask_train])
    
    if np.sum(hard_mask_train) > 0:
        hard_model.fit(X_train[hard_mask_train], y_train[hard_mask_train])
    
    # Stage 3: Make predictions using appropriate models
    print("\nStage 3: Making predictions with specialized models...")
    
    final_predictions = np.zeros(len(X_test))
    easy_mask_test = test_difficulty == 0
    hard_mask_test = test_difficulty == 1
    
    if np.sum(easy_mask_test) > 0:
        final_predictions[easy_mask_test] = easy_model.predict(X_test[easy_mask_test])
    
    if np.sum(hard_mask_test) > 0:
        final_predictions[hard_mask_test] = hard_model.predict(X_test[hard_mask_test])
    
    # Evaluate
    pipeline_acc = accuracy_score(y_test, final_predictions)
    print(f"\nMulti-stage pipeline accuracy: {pipeline_acc:.4f}")
    print(f"Easy samples: {np.sum(easy_mask_test)}, Hard samples: {np.sum(hard_mask_test)}")
    
    # Compare with single model
    single_model = RandomForestClassifier(n_estimators=150, random_state=42)
    single_model.fit(X_train, y_train)
    single_acc = accuracy_score(y_test, single_model.predict(X_test))
    print(f"Single model accuracy: {single_acc:.4f}")


# 5. HIERARCHICAL MODEL STRUCTURE
# ================================
def hierarchical_model_structure():
    """
    Demonstrates hierarchical modeling where models are organized
    in a tree structure for multi-class classification.
    """
    print("\n\n=== HIERARCHICAL MODEL STRUCTURE ===\n")
    
    # Generate 6-class problem
    X, y = make_classification(n_samples=1500, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=6, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define hierarchy: 
    # Level 1: Group classes into 2 super-classes
    # Level 2: Classify within each super-class
    
    # Create super-class labels (group classes 0,1,2 vs 3,4,5)
    y_train_super = (y_train >= 3).astype(int)
    y_test_super = (y_test >= 3).astype(int)
    
    # Level 1: Train super-class classifier
    print("Level 1: Training super-class classifier...")
    super_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    super_classifier.fit(X_train, y_train_super)
    
    # Predict super-classes
    train_super_pred = super_classifier.predict(X_train)
    test_super_pred = super_classifier.predict(X_test)
    
    # Level 2: Train sub-classifiers for each super-class
    print("\nLevel 2: Training sub-classifiers...")
    
    # Sub-classifier for super-class 0 (original classes 0,1,2)
    mask_0_train = train_super_pred == 0
    mask_0_test = test_super_pred == 0
    
    if np.sum(mask_0_train) > 0:
        sub_classifier_0 = RandomForestClassifier(n_estimators=100, random_state=42)
        # Map classes 0,1,2 to 0,1,2
        y_sub_0 = y_train[mask_0_train]
        sub_classifier_0.fit(X_train[mask_0_train], y_sub_0)
    
    # Sub-classifier for super-class 1 (original classes 3,4,5)
    mask_1_train = train_super_pred == 1
    mask_1_test = test_super_pred == 1
    
    if np.sum(mask_1_train) > 0:
        sub_classifier_1 = RandomForestClassifier(n_estimators=100, random_state=42)
        # Keep classes 3,4,5 as is
        y_sub_1 = y_train[mask_1_train]
        sub_classifier_1.fit(X_train[mask_1_train], y_sub_1)
    
    # Make hierarchical predictions
    final_predictions = np.zeros(len(X_test))
    
    if np.sum(mask_0_test) > 0:
        final_predictions[mask_0_test] = sub_classifier_0.predict(X_test[mask_0_test])
    
    if np.sum(mask_1_test) > 0:
        final_predictions[mask_1_test] = sub_classifier_1.predict(X_test[mask_1_test])
    
    # Evaluate
    hierarchical_acc = accuracy_score(y_test, final_predictions)
    print(f"\nHierarchical model accuracy: {hierarchical_acc:.4f}")
    
    # Compare with flat classifier
    flat_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
    flat_classifier.fit(X_train, y_train)
    flat_acc = accuracy_score(y_test, flat_classifier.predict(X_test))
    print(f"Flat classifier accuracy: {flat_acc:.4f}")


# 6. MODEL CASCADE (CONFIDENCE-BASED ROUTING)
# ===========================================
def model_cascade():
    """
    Demonstrates a cascade approach where simpler models handle
    easy cases and complex models handle difficult cases.
    """
    print("\n\n=== MODEL CASCADE ===\n")
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                              n_redundant=5, n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Define cascade of models (simple to complex)
    cascade_models = [
        ("Simple Logistic", LogisticRegression(random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    # Train all models
    print("Training cascade models...")
    for name, model in cascade_models:
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        print(f"{name}: {acc:.4f}")
    
    # Cascade prediction with confidence thresholds
    confidence_thresholds = [0.8, 0.7]  # High confidence for simple models
    
    print("\nCascade predictions...")
    final_predictions = np.full(len(X_test), -1)  # Initialize with -1
    
    for i, ((name, model), threshold) in enumerate(zip(cascade_models[:-1], confidence_thresholds)):
        # Get prediction probabilities
        probs = model.predict_proba(X_test)
        max_probs = np.max(probs, axis=1)
        
        # Make predictions for high-confidence samples
        high_conf_mask = (max_probs >= threshold) & (final_predictions == -1)
        
        if np.sum(high_conf_mask) > 0:
            final_predictions[high_conf_mask] = model.predict(X_test[high_conf_mask])
            print(f"{name} handled {np.sum(high_conf_mask)} samples")
    
    # Final model handles all remaining samples
    remaining_mask = final_predictions == -1
    if np.sum(remaining_mask) > 0:
        final_predictions[remaining_mask] = cascade_models[-1][1].predict(X_test[remaining_mask])
        print(f"{cascade_models[-1][0]} handled {np.sum(remaining_mask)} samples")
    
    # Evaluate cascade
    cascade_acc = accuracy_score(y_test, final_predictions)
    print(f"\nCascade accuracy: {cascade_acc:.4f}")


# Run all examples
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run demonstrations
    basic_stacking_example()
    stacking_with_original_features()
    sequential_model_training()
    multi_stage_pipeline()
    hierarchical_model_structure()
    model_cascade()
    
    print("\n\n=== SUMMARY ===")
    print("""
    Model interaction techniques demonstrated:
    
    1. Basic Stacking: Use predictions from multiple models as features for a meta-model
    2. Enhanced Stacking: Combine original features with model predictions
    3. Sequential Training: Models learn from errors of previous models
    4. Multi-Stage Pipeline: Different models for different aspects of the problem
    5. Hierarchical Structure: Tree-like organization of models
    6. Model Cascade: Route samples based on prediction confidence
    
    Benefits of model interaction:
    - Improved accuracy through ensemble effects
    - Better handling of different data patterns
    - Reduced overfitting through diverse models
    - Ability to leverage strengths of different algorithms
    - More interpretable decision-making process
    """)