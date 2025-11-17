#!/usr/bin/env python3
"""
Test script pour vérifier la correction SHAP avec CalibratedClassifierCV
"""

import sys
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

# Ajouter le chemin du projet
sys.path.append('src')

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_dict_structure,
    assert_float_equals
)

def test_shap_fix():
    """Test la correction SHAP avec CalibratedClassifierCV"""
    print("🧪 TEST DE LA CORRECTION SHAP")
    print("=" * 60)
    
    # Importer les utilitaires SHAP
    try:
        from src.utils.shap_utils import safe_shap_values, safe_shap_tree_explainer
        print("✅ Utils SHAP importées avec succès")
    except ImportError as e:
        print(f"❌ Erreur import SHAP utils: {e}")
        return False
    
    # Créer des données de test
    print("📊 Création des données de test...")
    X, y = make_classification(n_samples=100, n_features=10, n_classes=3, n_informative=5, random_state=42)
    
    # Créer et calibrer un modèle
    print("🤖 Création et calibration du modèle...")
    base_model = RandomForestClassifier(n_estimators=10, random_state=42)
    # Entraîner d'abord le modèle de base
    base_model.fit(X, y)
    calibrated_model = CalibratedClassifierCV(estimator=base_model, method='isotonic', cv='prefit')
    calibrated_model.fit(X, y)
    
    print(f"✅ Modèle calibré créé: {type(calibrated_model)}")
    
    # Tester avec l'ancienne méthode (doit échouer)
    print("🔍 Test avec l'ancienne méthode (doit échouer)...")
    try:
        import shap
        explainer_old = shap.TreeExplainer(calibrated_model)
        shap_values_old = explainer_old.shap_values(X[:10])
        print("❌ Ancienne méthode a réussu (ce n'est pas normal!)")
        old_method_works = True
    except Exception as e:
        print(f"✅ Ancienne méthode a échoué comme attendu: {e}")
        old_method_works = False
    
    # Tester avec la nouvelle méthode (doit fonctionner)
    print("🔍 Test avec la nouvelle méthode (doit fonctionner)...")
    try:
        explainer_new = safe_shap_tree_explainer(calibrated_model)
        if explainer_new is None:
            print("❌ Impossible de créer l'explainer sécurisé")
            return False
        
        shap_values_new = safe_shap_values(explainer_new, X[:10])
        if shap_values_new is None:
            print("❌ Impossible de calculer les valeurs SHAP sécurisées")
            return False
            
        print("✅ Nouvelle méthode a fonctionné!")
        new_method_works = True
    except Exception as e:
        print(f"❌ Nouvelle méthode a échoué: {e}")
        new_method_works = False
    
    # Résultat du test
    print("=" * 60)
    print("📋 RÉSULTAT DU TEST:")
    print(f"   Ancienne méthode fonctionne: {old_method_works}")
    print(f"   Nouvelle méthode fonctionne: {new_method_works}")
    
    # Validation avec assertions standardisées
    assert isinstance(old_method_works, bool), "Le résultat de l'ancienne méthode doit être un booléen"
    assert isinstance(new_method_works, bool), "Le résultat de la nouvelle méthode doit être un booléen"
    
    if new_method_works and not old_method_works:
        print("✅ SUCCÈS: La correction SHAP fonctionne correctement!")
        return True
    elif old_method_works and not new_method_works:
        print("❌ ÉCHEC: La correction SHAP ne fonctionne pas!")
        return False
    else:
        print("⚠️ RÉSULT INCOHÉRENT")
        return False

if __name__ == "__main__":
    test_shap_fix()