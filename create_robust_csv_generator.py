import pandas as pd
import json
import numpy as np
from datetime import datetime

def generate_hpo_results_csv(json_file, output_file=None):
    """
    Génère un fichier CSV à partir des résultats HPO du HMM roulant.
    
    Args:
        json_file (str): Chemin vers le fichier JSON des résultats HPO
        output_file (str, optional): Chemin vers le fichier CSV de sortie. 
                                 Si None, génère un nom automatiquement.
    
    Returns:
        str: Chemin vers le fichier CSV généré
    """
    
    # Charger le fichier JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extraire les résultats des essais
    coarse_results = data['data'].get('coarse_results', [])
    fine_results = data['data'].get('fine_results', [])
    refinement_results = data['data'].get('refinement_results', [])
    
    # Combiner tous les résultats
    all_results = coarse_results + fine_results + refinement_results
    
    # Préparer les données pour le CSV
    csv_data = []
    for i, result in enumerate(all_results, 1):
        params = result['params']
        metrics = result['quality_metrics']
        
        # Créer une ligne pour le CSV
        row = {
            'Trial': i,
            'Rank': None,  # Sera rempli plus tard
            'K': params.get('n_components', 'N/A'),
            'Base_Alpha': 'N/A',  # Non applicable pour le HMM roulant
            'Kappa': params.get('kappa', 'N/A'),
            'N_Mixtures': params.get('n_components', 'N/A'),
            'Learning_Rate': 'N/A',  # Non applicable pour le HMM roulant
            'SVI_Iterations': 'N/A',  # Non applicable pour le HMM roulant
            'ELBO': 'N/A',  # Non applicable pour le HMM roulant
            'Quality_Score': metrics.get('quality_score', 'N/A'),
            'Silhouette_Score': metrics.get('silhouette_score', 'N/A'),
            'Davies_Bouldin_Index': metrics.get('davies_bouldin_score', 'N/A'),
            'Calinski_Harabasz_Index': metrics.get('calinski_harabasz_score', 'N/A'),
            'Within_CV': metrics.get('within_regime_cv', 'N/A'),
            'Between_CV': metrics.get('between_regime_cv', 'N/A'),
            'Within_CV_Std': metrics.get('within_regime_cv_std', 'N/A'),
            'Between_CV_Std': metrics.get('between_regime_cv_std', 'N/A'),
            'Temporal_Smoothness': metrics.get('temporal_smoothness', 'N/A'),
            'Regime_Persistence': metrics.get('regime_persistence', 'N/A'),
            'Balance_Score': metrics.get('balance_score', 'N/A'),
            'N_Regimes': metrics.get('n_regimes', 'N/A'),
            'Noise_Ratio': metrics.get('noise_ratio', 'N/A'),
            'Predictive_Power': metrics.get('predictive_power', 'N/A'),
            'Mean_Return': metrics.get('mean_return', 'N/A'),
            'Volatility': metrics.get('volatility', 'N/A'),
            'Sharpe_Ratio': metrics.get('sharpe_ratio', 'N/A'),
            'Max_Drawdown': metrics.get('max_drawdown', 'N/A'),
            'Hit_Rate': metrics.get('hit_rate', 'N/A'),
            'Log_Likelihood': metrics.get('log_likelihood', 'N/A'),
            'Refit_Stability_ARI': metrics.get('refit_stability_ari', 'N/A'),
            'State_Occupancy_Entropy': metrics.get('state_occupancy_entropy', 'N/A'),
            'Min_Regime_Size': metrics.get('min_regime_size', 'N/A'),
            'Max_Regime_Size': metrics.get('max_regime_size', 'N/A'),
            'Regime_Size_Std': metrics.get('regime_size_std', 'N/A')
        }
        csv_data.append(row)
    
    # Créer un DataFrame
    df = pd.DataFrame(csv_data)
    
    # Trier par score de qualité (décroissant)
    df = df.sort_values('Quality_Score', ascending=False)
    
    # Ajouter les rangs
    df['Rank'] = range(1, len(df) + 1)
    
    # Générer le nom du fichier de sortie si non spécifié
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f'./outcomes/hpo_results_ETHUSDT_{timestamp}.csv'
    
    # Sauvegarder le fichier CSV
    df.to_csv(output_file, index=False)
    
    return output_file

def identify_final_configuration(json_file, csv_file):
    """
    Identifie la configuration finale sauvegardée comme artefact et l'ajoute au CSV.
    
    Args:
        json_file (str): Chemin vers le fichier JSON des résultats HPO
        csv_file (str): Chemin vers le fichier CSV à modifier
    
    Returns:
        str: Chemin vers le fichier CSV modifié
    """
    
    # Charger le fichier JSON
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extraire les meilleurs paramètres
    best_params = data['data']['best_params']
    
    # Charger le fichier CSV
    df = pd.read_csv(csv_file)
    
    # Ajouter une colonne pour indiquer si l'essai utilise la configuration finale
    df['Final_Configuration'] = 'No'
    
    # Marquer l'essai qui utilise la configuration finale
    for idx, row in df.iterrows():
        trial_num = row['Trial']
        
        # Vérifier si les paramètres correspondent
        if (str(row['K']) == str(best_params.get('n_components', 'N/A')) and
            str(row['Kappa']) == str(best_params.get('kappa', 'N/A'))):
            df.at[idx, 'Final_Configuration'] = 'Yes'
    
    # Générer le nom du fichier de sortie
    base_name = csv_file.replace('.csv', '')
    output_file = f'{base_name}_with_final_config.csv'
    
    # Sauvegarder le fichier CSV modifié
    df.to_csv(output_file, index=False)
    
    return output_file

def main():
    """
    Fonction principale pour générer le fichier CSV des résultats HPO.
    """
    # Fichier JSON des résultats HPO
    json_file = './artifacts/rolling_hmm_hpo_results.json'
    
    # Générer le fichier CSV des résultats HPO
    csv_file = generate_hpo_results_csv(json_file)
    print(f"Fichier CSV des résultats HPO généré : {csv_file}")
    
    # Identifier la configuration finale et l'ajouter au CSV
    final_csv_file = identify_final_configuration(json_file, csv_file)
    print(f"Fichier CSV avec configuration finale : {final_csv_file}")
    
    # Afficher un résumé
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    best_params = data['data']['best_params']
    print("\nMeilleurs paramètres trouvés :")
    print(f"EWMA Config: {best_params.get('ewma_config_idx', 'N/A')}")
    print(f"n_components: {best_params.get('n_components', 'N/A')}")
    print(f"min_covar: {best_params.get('min_covar', 'N/A')}")
    print(f"kappa: {best_params.get('kappa', 'N/A')}")

if __name__ == "__main__":
    main()
