#!/usr/bin/env python3
"""
Test pour la correction du problème SHAP avec NumPy
"""
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings('ignore')

# Patch temporaire pour np.bool déprécié
def apply_numpy_shap_fix():
    """Applique un patch temporaire pour corriger l'incompatibilité SHAP-NumPy"""
    try:
        # Vérifier si np.bool existe déjà
        if not hasattr(np, 'bool'):
            # Créer un alias vers bool pour compatibilité avec SHAP
            np.bool = bool
            print("✅ [PATCH] np.bool alias créé pour compatibilité SHAP")
            return True
        else:
            print("ℹ️ [PATCH] np.bool existe déjà")
            return True
    except Exception as e:
        print(f"❌ [PATCH] Erreur lors de l'application du patch: {e}")
        return False

def test_shap_with_numpy_fix():
    """Test SHAP avec le patch NumPy"""
    print("🔍 [TEST] Test SHAP avec patch NumPy")
    
    # 1. Créer des données de test
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    
    # 2. Créer un modèle calibré
    base_rf = RandomForestClassifier(n_estimators=10, random_state=42)
    calibrated_model = CalibratedClassifierCV(base_rf, cv=3)
    calibrated_model.fit(X, y)
    
    print(f"✅ [TEST] Modèle calibré créé: {type(calibrated_model)}")
    
    # 3. Extraire le modèle de base
    base_model = calibrated_model.estimators_[0]
    print(f"✅ [TEST] Modèle de base extrait: {type(base_model)}")
    
    # 4. Appliquer le patch NumPy
    patch_success = apply_numpy_shap_fix()
    
    if not patch_success:
        print("❌ [TEST] ÉCHEC: Impossible d'appliquer le patch NumPy")
        return False
    
    # 5. Tester SHAP avec le patch
    try:
        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(X[:10])
        print("✅ [TEST] SUCCÈS: SHAP fonctionne avec le patch NumPy!")
        return True
    except Exception as e:
        print(f"❌ [TEST] ÉCHEC: SHAP ne fonctionne toujours pas: {e}")
        return False

def test_shap_alternative_explainer():
    """Test avec un explainer alternatif si TreeExplainer échoue"""
    print("🔍 [TEST] Test avec explainer alternatif (KernelExplainer)")
    
    # 1. Créer des données de test
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
    
    # 2. Créer un modèle calibré
    base_rf = RandomForestClassifier(n_estimators=10, random_state=42)
    calibrated_model = CalibratedClassifierCV(base_rf, cv=3)
    calibrated_model.fit(X, y)
    
    # 3. Extraire le modèle de base
    base_model = calibrated_model.estimators_[0]
    
    # 4. Tester avec KernelExplainer (plus lent mais universel)
    try:
        explainer = shap.KernelExplainer(base_model.predict_proba, X[:20])
        shap_values = explainer.shap_values(X[:5])
        print("✅ [TEST] SUCCÈS: KernelExplainer fonctionne comme alternative!")
        return True
    except Exception as e:
        print(f"❌ [TEST] ÉCHEC: KernelExplainer ne fonctionne pas: {e}")
        return False

if __name__ == "__main__":
    print("🧪 [TEST] Test de la correction SHAP avec NumPy")
    print("=" * 60)
    
    # Test 1: Patch NumPy
    success1 = test_shap_with_numpy_fix()
    
    print("\n" + "=" * 60)
    
    # Test 2: Explainer alternatif
    success2 = test_shap_alternative_explainer()
    
    print("\n" + "=" * 60)
    print("📊 [RÉSULTATS]")
    print(f"   • Patch NumPy: {'✅ SUCCÈS' if success1 else '❌ ÉCHEC'}")
    print(f"   • Explainer alternatif: {'✅ SUCCÈS' if success2 else '❌ ÉCHEC'}")
    
    if success1 or success2:
        print("🎉 [GLOBAL] Au moins une solution fonctionne!")
    else:
        print("💥 [GLOBAL] Toutes les solutions ont échoué!")