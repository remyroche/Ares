#!/usr/bin/env python3
"""
Script de test pour vérifier la correction de sérialisation JSON des types numpy.

Ce script reproduit le problème original et vérifie que notre solution fonctionne correctement.
"""

import json
import numpy as np
import sys
import os

# Ajouter le chemin du projet au sys.path pour importer nos modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from training.steps.market_analysis.components.artifact_manager import ArtifactManager
from training.steps.market_analysis.components.regime_artifact_schema import RegimeArtifactExtractor


def test_numpy_json_serialization():
    """Teste la sérialisation JSON avec des types numpy."""
    print("🧪 TEST: Test de sérialisation JSON avec types numpy")
    
    # Créer un ArtifactManager pour le test
    artifact_manager = ArtifactManager(
        base_dir="test_artifacts",
        symbol="TEST",
        exchange="test",
        timeframe="1h"
    )
    
    # Test 1: Créer un dictionnaire avec des clés int64 (problème original)
    print("\n1️⃣ Test 1: Création d'un dictionnaire avec des clés int64...")
    try:
        # Créer des clés int64 avec numpy
        int64_keys = np.array([0, 1, 2], dtype=np.int64)
        int64_values = np.array([10, 20, 30], dtype=np.int64)
        
        # Créer le dictionnaire problématique
        problematic_dict = dict(zip(int64_keys, int64_values))
        
        # Tenter de sérialiser avec le sérialiseur par défaut
        try:
            json_str = json.dumps(problematic_dict)
            print("❌ ÉCHEC: Le dictionnaire avec clés int64 a été sérialisé (ne devrait pas arriver)")
            print(f"   JSON: {json_str}")
        except Exception as e:
            print(f"✅ SUCCÈS: Erreur attendue lors de la sérialisation: {e}")
        
        # Tenter avec notre sérialiseur corrigé
        try:
            # Prétraiter d'abord avec notre nouvelle fonction
            preprocessed_dict = artifact_manager._preprocess_for_json(problematic_dict)
            json_str = json.dumps(preprocessed_dict, default=artifact_manager._json_serializer)
            print("✅ SUCCÈS: Le dictionnaire a été sérialisé avec notre correcteur")
            print(f"   JSON: {json_str}")
        except Exception as e:
            print(f"❌ ÉCHEC: Notre correcteur a échoué: {e}")
            
    except Exception as e:
        print(f"❌ ÉCHEC du test 1: {e}")
    
    # Test 2: Simuler le cas réel du régime_artifact_schema
    print("\n2️⃣ Test 2: Simulation du cas régime_artifact_schema...")
    try:
        # Simuler les données comme dans le code original
        cluster_assignments = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2])
        unique_regimes = np.array([0, 1, 2])
        regime_counts = np.array([4, 3, 3])
        
        # Créer le dictionnaire comme dans le code original (ligne 416)
        regime_distribution = dict(zip(unique_regimes.astype(int), regime_counts.astype(int)))
        
        # Créer l'artefact
        artifact_data = {
            'cluster_assignments': cluster_assignments,
            'n_regimes': len(unique_regimes),
            'regime_distribution': regime_distribution,
            'clustering_method': 'test',
            'clustering_params': {},
            'metadata': {}
        }
        
        # Tenter de sérialiser avec notre sérialiseur
        try:
            # Prétraiter d'abord avec notre nouvelle fonction
            preprocessed_data = artifact_manager._preprocess_for_json(artifact_data)
            json_str = json.dumps(preprocessed_data, default=artifact_manager._json_serializer)
            print("✅ SUCCÈS: L'artefact régime a été sérialisé avec succès")
            print(f"   Clés dans regime_distribution: {list(preprocessed_data['regime_distribution'].keys())}")
            print(f"   Types des clés: {[type(k) for k in preprocessed_data['regime_distribution'].keys()]}")
        except Exception as e:
            print(f"❌ ÉCHEC: Échec de la sérialisation: {e}")
            
    except Exception as e:
        print(f"❌ ÉCHEC du test 2: {e}")
    
    print("\n🎯 TEST TERMINÉ")


if __name__ == "__main__":
    test_numpy_json_serialization()