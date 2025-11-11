import os
import shutil

def integrate_csv_generator():
    """
    Intègre le générateur CSV robuste dans le pipeline existant.
    """
    
    # Chemin vers le générateur CSV robuste
    source_file = './create_robust_csv_generator.py'
    
    # Chemin vers le répertoire du pipeline où le générateur sera intégré
    target_dir = './src/training/steps/market_analysis/rolling_hmm_clustering'
    target_file = os.path.join(target_dir, 'hpo_csv_generator.py')
    
    # Copier le fichier
    shutil.copy(source_file, target_file)
    
    print(f"Générateur CSV robuste intégré dans : {target_file}")
    
    # Créer un script d'utilisation
    usage_script = os.path.join(target_dir, 'use_hpo_csv_generator.py')
    with open(usage_script, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Script d'utilisation du générateur CSV HPO pour le HMM roulant.
"""

import sys
import os

# Ajouter le répertoire courant au chemin Python
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from hpo_csv_generator import generate_hpo_results_csv, identify_final_configuration

def main():
    """
    Fonction principale pour utiliser le générateur CSV HPO.
    """
    # Fichier JSON des résultats HPO
    json_file = './artifacts/rolling_hmm_hpo_results.json'
    
    # Générer le fichier CSV des résultats HPO
    csv_file = generate_hpo_results_csv(json_file)
    print(f"Fichier CSV des résultats HPO généré : {csv_file}")
    
    # Identifier la configuration finale et l'ajouter au CSV
    final_csv_file = identify_final_configuration(json_file, csv_file)
    print(f"Fichier CSV avec configuration finale : {final_csv_file}")

if __name__ == "__main__":
    main()
''')
    
    print(f"Script d'utilisation créé : {usage_script}")
    
    # Rendre le script d'utilisation exécutable
    os.chmod(usage_script, 0o755)
    
    print("\nPour utiliser le générateur CSV robuste :")
    print(f"1. cd {target_dir}")
    print("2. python3 use_hpo_csv_generator.py")

if __name__ == "__main__":
    integrate_csv_generator()
