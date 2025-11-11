#!/usr/bin/env python3
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
