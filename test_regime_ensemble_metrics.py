#!/usr/bin/env python3
"""
Script de test pour vérifier l'implémentation des nouvelles métriques dans regime_ensemble_training
"""

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Ajouter le chemin du projet au PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

# Import des assertions standardisées
from tests.utils.assertions import (
    assert_float_equals,
    assert_dict_structure,
    assert_list_structure
)

def test_advanced_metrics():
    """Test the implementation of advanced metrics."""
    try:
        # Importer le composant
        from src.training.steps.market_analysis.components.regime_ensemble_training import RegimeEnsembleTrainingComponent
        
        print("✅ Import réussi de RegimeEnsembleTrainingComponent")
        
        # Créer une instance du composant
        component = RegimeEnsembleTrainingComponent()
        print("✅ Instance du composant créée")
        
        # Créer des données de test
        n_samples = 100
        n_classes = 4
        
        # Générer des étiquettes vraies et prédictions
        np.random.seed(42)
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)
        
        # Générer des probabilités prédictives
        y_pred_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
        
        print(f"✅ Données de test créées: {n_samples} échantillons, {n_classes} classes")
        
        # Tester la méthode _calculate_advanced_metrics
        advanced_metrics = component._calculate_advanced_metrics(y_true, y_pred, y_pred_proba)
        
        print("✅ Métriques avancées calculées avec succès")
        print("\n📊 Métriques calculées:")
        
        # Afficher les métriques principales
        if 'confusion_matrix' in advanced_metrics:
            print("  - Matrice de confusion (absolue et normalisée)")
            assert_dict_structure(
                advanced_metrics['confusion_matrix'],
                ['absolute', 'normalized'],
                message="La matrice de confusion doit contenir les clés 'absolute' et 'normalized'"
            )
        
        if 'per_class_metrics' in advanced_metrics:
            print("  - Métriques par classe (macro et weighted)")
            assert_dict_structure(
                advanced_metrics['per_class_metrics'],
                ['macro', 'weighted'],
                message="Les métriques par classe doivent contenir 'macro' et 'weighted'"
            )
        
        if 'macro_f1' in advanced_metrics:
            macro_f1 = advanced_metrics['macro_f1']
            assert isinstance(macro_f1, (int, float)), "Macro F1 doit être numérique"
            assert 0 <= macro_f1 <= 1, f"Macro F1 doit être entre 0 et 1, valeur: {macro_f1}"
            print(f"  - Macro F1: {macro_f1:.4f}")
        
        if 'balanced_accuracy' in advanced_metrics:
            balanced_acc = advanced_metrics['balanced_accuracy']
            assert isinstance(balanced_acc, (int, float)), "Balanced Accuracy doit être numérique"
            assert 0 <= balanced_acc <= 1, f"Balanced Accuracy doit être entre 0 et 1, valeur: {balanced_acc}"
            print(f"  - Balanced Accuracy: {balanced_acc:.4f}")
        
        if 'cohens_kappa' in advanced_metrics:
            kappa_data = advanced_metrics['cohens_kappa']
            assert_dict_structure(
                kappa_data,
                ['score', 'interpretation'],
                message="Cohen's Kappa doit contenir 'score' et 'interpretation'"
            )
            kappa_score = kappa_data['score']
            assert isinstance(kappa_score, (int, float)), "Cohen's Kappa score doit être numérique"
            print(f"  - Cohen's Kappa: {kappa_score:.4f} ({kappa_data['interpretation']})")
        
        if 'roc_auc_scores' in advanced_metrics:
            print(f"  - ROC-AUC Scores: {len(advanced_metrics['roc_auc_scores'])} classes")
        
        if 'pr_auc_scores' in advanced_metrics:
            print(f"  - PR-AUC Scores: {len(advanced_metrics['pr_auc_scores'])} classes")
        
        if 'probabilistic_calibration' in advanced_metrics:
            log_loss = advanced_metrics['probabilistic_calibration']['log_loss']
            brier_score = advanced_metrics['probabilistic_calibration']['brier_score']
            print(f"  - Log Loss: {log_loss:.4f}")
            print(f"  - Brier Score: {brier_score:.4f}")
        
        if 'temporal_metrics' in advanced_metrics:
            print("  - Métriques temporelles")
            temporal = advanced_metrics['temporal_metrics']
            if 'detection_delay' in temporal:
                delay = temporal['detection_delay']['mean_lag']
                print(f"    - Detection Delay (Mean Lag): {delay:.4f}")
            if 'regime_persistence' in temporal:
                persistence = temporal['regime_persistence']['persistence_ratio']
                print(f"    - Persistence Ratio: {persistence:.4f}")
            if 'transition_accuracy' in temporal:
                trans_acc = temporal['transition_accuracy']
                print(f"    - Transition Accuracy: {trans_acc:.4f}")
        
        if 'segmentation_metrics' in advanced_metrics:
            print("  - Métriques de segmentation")
            segmentation = advanced_metrics['segmentation_metrics']
            if 'adjusted_rand_index' in segmentation:
                ari = segmentation['adjusted_rand_index']
                print(f"    - Adjusted Rand Index: {ari:.4f}")
        
        if 'change_point_metrics' in advanced_metrics:
            print("  - Métriques de détection de points de changement")
            change_point = advanced_metrics['change_point_metrics']
            if 'precision' in change_point:
                cp_precision = change_point['precision']
                print(f"    - Change-Point Precision: {cp_precision:.4f}")
        
        if 'sequence_metrics' in advanced_metrics:
            print("  - Métriques de séquence")
            sequence = advanced_metrics['sequence_metrics']
            if 'hamming_loss' in sequence:
                hamming = sequence['hamming_loss']
                print(f"    - Hamming Loss: {hamming:.4f}")
        
        print("\n✅ Test des métriques avancées réussi!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test des métriques avancées: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_visualization_methods():
    """Test the visualization methods."""
    try:
        from src.training.steps.market_analysis.components.regime_ensemble_training import RegimeEnsembleTrainingComponent
        
        # Créer une instance du composant
        component = RegimeEnsembleTrainingComponent()
        
        # Créer des données de test
        n_samples = 100
        n_classes = 4
        
        # Générer des étiquettes vraies et prédictions
        np.random.seed(42)
        y_true = np.random.randint(0, n_classes, n_samples)
        y_pred = np.random.randint(0, n_classes, n_samples)
        y_pred_proba = np.random.dirichlet(np.ones(n_classes), n_samples)
        
        print("📊 Test des méthodes de visualisation")
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs("test_outputs", exist_ok=True)
        
        # Tester la visualisation de la matrice de confusion
        cm_viz_path = component._generate_confusion_matrix_visualization(
            y_true, y_pred, "test_outputs"
        )
        if cm_viz_path:
            print(f"✅ Visualisation de la matrice de confusion: {cm_viz_path}")
        
        # Tester la visualisation des courbes ROC
        roc_viz_path = component._generate_roc_curves_visualization(
            y_true, y_pred_proba, "test_outputs"
        )
        if roc_viz_path:
            print(f"✅ Visualisation des courbes ROC: {roc_viz_path}")
        
        # Tester la visualisation des courbes PR
        pr_viz_path = component._generate_precision_recall_curves_visualization(
            y_true, y_pred_proba, "test_outputs"
        )
        if pr_viz_path:
            print(f"✅ Visualisation des courbes PR: {pr_viz_path}")
        
        # Tester la visualisation temporelle
        temporal_viz_path = component._generate_temporal_regime_visualization(
            y_true, y_pred, "test_outputs"
        )
        if temporal_viz_path:
            print(f"✅ Visualisation temporelle: {temporal_viz_path}")
        
        print("\n✅ Test des méthodes de visualisation réussi!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test des visualisations: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Fonction principale du script de test."""
    print("🚀 Démarrage du test des métriques avancées pour Regime Ensemble Training")
    print("=" * 80)
    
    # Test des métriques avancées
    print("\n1. Test des métriques avancées")
    print("-" * 40)
    metrics_success = test_advanced_metrics()
    
    # Test des méthodes de visualisation
    print("\n2. Test des méthodes de visualisation")
    print("-" * 40)
    viz_success = test_visualization_methods()
    
    # Résumé
    print("\n" + "=" * 80)
    print("📋 RÉSUMÉ DES TESTS")
    print("=" * 80)
    print(f"Métriques avancées: {'✅ Succès' if metrics_success else '❌ Échec'}")
    print(f"Visualisations: {'✅ Succès' if viz_success else '❌ Échec'}")
    
    if metrics_success and viz_success:
        print("\n🎉 TOUS LES TESTS RÉUSSIS!")
        print("Les nouvelles métriques ont été correctement implémentées.")
        return 0
    else:
        print("\n⚠️ CERTAINS TESTS ONT ÉCHOUÉ")
        print("Veuillez vérifier l'implémentation.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)