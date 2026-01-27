
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.special import expit
import sys
import os

# Add src to path
sys.path.append(os.getcwd())

from src.utils.huber_regressor_for_trees import prepare_huber_teacher_outputs, HuberTeacherConfig
from src.training.steps.labeling.layer3.model_training import train_lgbm_model

def test_lgbm_inference_reconstruction():
    print(">>> Testing LGBM Inference Reconstruction...")

    # 1. Generate Mock Data
    np.random.seed(42)
    n_samples = 500
    n_features = 20
    X = pd.DataFrame(np.random.randn(n_samples, n_features), columns=[f"feat_{i}" for i in range(n_features)])
    # Target: binary classification
    y = (X['feat_0'] * 2 + X['feat_1'] - X['feat_2'] + np.random.randn(n_samples) > 0).astype(int)
    w = np.ones(n_samples)

    # 2. Prepare Huber Teacher
    print("   Running prepare_huber_teacher_outputs...")
    huber_config = HuberTeacherConfig(epsilons=(1.1,), alphas=(1e-2,), n_time_splits=2)
    huber_out = prepare_huber_teacher_outputs(
        X_train=X,
        y_train=y,
        sample_weight=w,
        config=huber_config,
        is_classification=True
    )

    # 3. Train LGBM Student
    print("   Running train_lgbm_model...")
    lgbm_res = train_lgbm_model(
        X_train=X,
        y_train=y,
        model_name="test_lgbm",
        task_type="classification",
        huber_output=huber_out,
        sample_weight=w,
        fast_mode=True
    )

    # 4. Check Artifacts
    print("   Checking returned artifacts...")
    keys = lgbm_res.keys()
    print(f"   Keys found: {list(keys)}")

    missing_artifacts = []
    required = ['teacher_model', 'teacher_scaler', 'teacher_features', 'student_features', 'predict_contract']
    for req in required:
        if req not in keys:
            missing_artifacts.append(req)

    if missing_artifacts:
        print(f"❌ Missing critical artifacts: {missing_artifacts}")
    else:
        print("✅ All critical artifacts present.")

    # 5. Simulate Inference (Reconstruction)
    # Goal: Reconstruct 'cate' (OOF) using the artifacts

    # A. Get OOF prediction from training
    oof_pred = lgbm_res['cate']

    # B. Reconstruct manually
    print("   Attempting reconstruction...")
    try:
        if missing_artifacts:
            print("   Skipping reconstruction due to missing artifacts.")
            return

        teacher = lgbm_res['teacher_model']
        t_scaler = lgbm_res['teacher_scaler']
        s_scaler = lgbm_res['scaler'] # Student scaler (standardscaler)
        student = lgbm_res['model']

        t_feats = lgbm_res['teacher_features']
        s_feats = lgbm_res['student_features']

        # 1. Teacher Prediction
        X_teacher = X[t_feats] # Ensure column order
        X_t_scaled = t_scaler.transform(X_teacher)
        teacher_pred = teacher.predict(X_t_scaled)

        # 2. Student Prediction
        X_student = X[s_feats]
        X_s_scaled = s_scaler.transform(X_student)

        # LGBM predict with raw_score=True gives margins
        student_margin = student.predict(X_s_scaled, raw_score=True)

        # 3. Combine
        total_margin = teacher_pred + student_margin

        # 4. Transform
        reconstructed_prob = expit(total_margin)

        # Compare
        # Note: train_lgbm_model might use internal calibration or fold splitting which affects 'cate'
        # 'cate' in train_lgbm_model is:
        # split_idx = 0.9
        # model fit on train
        # pred on all X -> raw_score
        # cate = sigmoid(raw + init_score)
        # So reconstruction should match EXACTLY if we use the same inputs.

        diff = np.abs(reconstructed_prob - oof_pred)
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        print(f"   Max Diff: {max_diff:.6e}")
        print(f"   Mean Diff: {mean_diff:.6e}")

        if max_diff < 1e-5:
            print("✅ Reconstruction Successful! Inference logic matches OOF.")
        else:
            print("❌ Reconstruction Mismatch.")

    except Exception as e:
        print(f"❌ Reconstruction Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lgbm_inference_reconstruction()
