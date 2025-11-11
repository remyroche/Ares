import re

def fix_cluster_quality_assessor():
    """
    Modifie le script cluster_quality_assessor.py pour qu'il génère correctement le fichier CSV pour le HMM roulant.
    """
    
    # Chemin vers le script à modifier
    script_path = './src/training/steps/market_analysis/clusters/cluster_quality_assessor.py'
    
    # Lire le contenu du script
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Remplacer les lignes qui extraient les paramètres pour le HMM roulant
    # Ancien code :
    # params.get('K', 'N/A'),
    # params.get('base_alpha', 'N/A'),
    # params.get('kappa', 'N/A'),
    # params.get('n_mixtures', 'N/A'),
    
    # Nouveau code :
    # params.get('n_components', 'N/A'),
    # 'N/A',  # Non applicable pour le HMM roulant
    # params.get('kappa', 'N/A'),
    # params.get('n_components', 'N/A'),
    
    # Remplacer la section qui extrait les paramètres
    old_pattern = r"params\.get\('K', 'N/A'\),\s*params\.get\('base_alpha', 'N/A'\),\s*params\.get\('kappa', 'N/A'\),\s*params\.get\('n_mixtures', 'N/A'\),"
    new_code = """params.get('n_components', 'N/A'),
                    'N/A',  # Non applicable pour le HMM roulant
                    params.get('kappa', 'N/A'),
                    params.get('n_components', 'N/A'),"""
    
    content = re.sub(old_pattern, new_code, content)
    
    # Remplacer la section qui extrait les paramètres SVI
    # Ancien code :
    # params.get('learning_rate', 'N/A'),
    # params.get('svi_iterations', 'N/A'),
    # trial.get('final_elbo', 'N/A'),
    
    # Nouveau code :
    # 'N/A',  # Non applicable pour le HMM roulant
    # 'N/A',  # Non applicable pour le HMM roulant
    # 'N/A',  # Non applicable pour le HMM roulant
    
    old_svi_pattern = r"params\.get\('learning_rate', 'N/A'\),\s*params\.get\('svi_iterations', 'N/A'\),\s*trial\.get\('final_elbo', 'N/A'\),"
    new_svi_code = """'N/A',  # Non applicable pour le HMM roulant
                    'N/A',  # Non applicable pour le HMM roulant
                    'N/A',  # Non applicable pour le HMM roulant"""
    
    content = re.sub(old_svi_pattern, new_svi_code, content)
    
    # Écrire le contenu modifié dans le fichier
    with open(script_path, 'w') as f:
        f.write(content)
    
    print(f"Script {script_path} modifié avec succès pour le HMM roulant")

if __name__ == "__main__":
    fix_cluster_quality_assessor()
