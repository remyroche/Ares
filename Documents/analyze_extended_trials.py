import pandas as pd
import numpy as np
from typing import List, Dict, Tuple

def parse_extended_trial_data(data_text: str) -> pd.DataFrame:
    """Parse les données étendues des essais à partir du texte fourni"""
    lines = data_text.strip().split('\n')
    
    # Extraire les en-têtes
    headers = lines[0].split('\t')
    print(f"En-têtes trouvés: {headers}")
    print(f"Nombre d'en-têtes: {len(headers)}")
    
    # Parser les données
    data_rows = []
    for i, line in enumerate(lines[1:], 1):
        if line.strip():
            values = line.split('\t')
            print(f"Ligne {i}: {len(values)} valeurs, attendu: {len(headers)}")
            if len(values) >= len(headers):
                # Ignorer les deux premières colonnes qui semblent être des index
                if len(values) > len(headers):
                    values = values[2:]  # Supprimer les deux premiers éléments (index)
                if len(values) == len(headers):
                    data_rows.append(values)
                else:
                    print(f"  Ignorée: nombre incorrect de valeurs après ajustement")
            else:
                print(f"  Ignorée: pas assez de valeurs")
    
    print(f"Nombre total de lignes parsées: {len(data_rows)}")
    
    # Créer le DataFrame
    if data_rows:
        df = pd.DataFrame(data_rows, columns=headers)
        
        # Convertir les types de données
        numeric_columns = ['EWMA_Short', 'EWMA_Long', 'States', 'Kappa', 'Min_Cov',
                           'CV_Ratio', 'Silhouette', 'Temporal_Smoothness', 'Balance', 'Quality_Score']
        
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    else:
        return pd.DataFrame()

def analyze_top_performers_extended(df: pd.DataFrame, top_n: int = 10) -> Dict:
    """Analyse étendue des meilleurs performeurs"""
    results = {}
    
    # Top par Quality_Score
    results['top_quality'] = df.nlargest(top_n, 'Quality_Score')
    
    # Top par Silhouette (meilleure séparation des clusters)
    results['top_silhouette'] = df.nlargest(top_n, 'Silhouette')
    
    # Top par Temporal_Smoothness (meilleure cohérence temporelle)
    results['top_temporal_smoothness'] = df.nlargest(top_n, 'Temporal_Smoothness')
    
    # Top par Balance (meilleur équilibre)
    results['top_balance'] = df.nlargest(top_n, 'Balance')
    
    # Top par CV_Ratio (plus faible coefficient de variation)
    results['lowest_cv'] = df.nsmallest(top_n, 'CV_Ratio')
    
    # Meilleur score composite (Quality_Score + Silhouette + Temporal_Smoothness + Balance)
    df['composite_score'] = (df['Quality_Score'] + df['Silhouette'] + 
                            df['Temporal_Smoothness'] + df['Balance']) / 4
    results['top_composite'] = df.nlargest(top_n, 'composite_score')
    
    return results

def analyze_parameter_patterns(df: pd.DataFrame) -> Dict:
    """Analyse les patterns de paramètres"""
    patterns = {}
    
    # Analyse par EWMA_Short
    ewma_short_analysis = df.groupby('EWMA_Short').agg({
        'Quality_Score': ['mean', 'std', 'count'],
        'Silhouette': 'mean',
        'Temporal_Smoothness': 'mean',
        'Balance': 'mean'
    }).round(4)
    patterns['ewma_short_analysis'] = ewma_short_analysis
    
    # Analyse par EWMA_Long
    ewma_long_analysis = df.groupby('EWMA_Long').agg({
        'Quality_Score': ['mean', 'std', 'count'],
        'Silhouette': 'mean',
        'Temporal_Smoothness': 'mean',
        'Balance': 'mean'
    }).round(4)
    patterns['ewma_long_analysis'] = ewma_long_analysis
    
    # Analyse par States
    states_analysis = df.groupby('States').agg({
        'Quality_Score': ['mean', 'std', 'count'],
        'Silhouette': 'mean',
        'Temporal_Smoothness': 'mean',
        'Balance': 'mean'
    }).round(4)
    patterns['states_analysis'] = states_analysis
    
    # Analyse par Kappa ranges
    kappa_ranges = pd.cut(df['Kappa'], bins=[0, 1, 3, 6, 12], labels=['0-1', '1-3', '3-6', '6-12'])
    df_kappa = df.copy()
    df_kappa['kappa_range'] = kappa_ranges
    kappa_analysis = df_kappa.groupby('kappa_range').agg({
        'Quality_Score': ['mean', 'std', 'count'],
        'Silhouette': 'mean',
        'Temporal_Smoothness': 'mean',
        'Balance': 'mean'
    }).round(4)
    patterns['kappa_analysis'] = kappa_analysis
    
    return patterns

def main():
    # Données fournies
    extended_data = """EWMA_Short	EWMA_Long	States	Kappa	Min_Cov	CV_Ratio	Silhouette	Temporal_Smoothness	Balance	Quality_Score
2	1	8	20	6	0.2	0.0025	0.9951	0.5000	0.7567	0.7787	0.735
3	2	12	20	5	0.5	0.005	0.4730	0.5000	0.8508	0.7575	0.472
4	3	8	24	5	3.0	0.01	1.0000	0.5000	0.8272	0.7326	0.734
5	4	12	20	4	2.0	0.005	0.6224	0.5000	0.8637	0.8178	0.552
6	5	8	16	4	2.5	0.001	0.9890	0.5000	0.8163	0.7735	0.731
7	6	12	16	6	2.5	0.005	0.6608	0.5000	0.8249	0.8281	0.572
8	7	8	20	4	0.2	0.01	0.9656	0.5000	0.7929	0.7773	0.720
9	8	8	16	5	2.0	0.005	0.0744	0.5000	0.8303	0.7305	0.271
10	9	12	16	6	0.2	0.0025	0.6994	0.5000	0.8021	0.8291	0.591
11	10	8	20	4	1.5	0.005	0.9849	0.5000	0.8107	0.7730	0.729
12	11	12	20	5	0.5	0.001	0.4730	0.5000	0.8508	0.7575	0.472
13	12	12	20	4	0.5	0.01	0.6139	0.5000	0.8517	0.8178	0.547
14	13	12	16	6	0.2	0.01	0.7021	0.5000	0.8021	0.8291	0.592
15	14	8	20	4	0.8	0.0025	0.9619	0.5000	0.8025	0.7750	0.718
16	15	8	16	6	0.2	0.0025	0.9729	0.5000	0.7532	0.7765	0.724
17	16	8	20	6	1.0	0.0025	0.9948	0.5000	0.7727	0.7766	0.735
18	17	8	16	6	0.5	0.005	0.9526	0.5000	0.7599	0.7751	0.713
19	18	12	16	4	0.8	0.005	0.0881	0.5000	0.8872	0.7941	0.283
20	19	8	24	4	0.8	0.001	0.8904	0.5000	0.8013	0.7783	0.682
21	20	12	20	4	1.0	0.01	0.6172	0.5000	0.8576	0.8174	0.549
22	21	8	16	5	0.2	0.005	0.3442	0.5000	0.8085	0.7316	0.406
23	22	8	20	5	0.5	0.001	0.6487	0.5000	0.8156	0.7306	0.558
24	23	12	20	6	2.0	0.0025	0.9710	0.5000	0.8255	0.7966	0.724
25	24	12	20	4	0.5	0.005	0.6138	0.5000	0.8517	0.8179	0.547
26	25	12	20	6	3.0	0.0025	0.9685	0.5000	0.8316	0.7950	0.723
27	26	8	20	5	3.0	0.01	0.5143	0.5000	0.8373	0.7298	0.491
28	27	12	20	6	2.5	0.01	0.9694	0.5000	0.8292	0.7961	0.723
29	28	8	24	6	0.5	0.01	0.8729	0.5000	0.7579	0.7736	0.673
30	29	8	16	6	0.2	0.005	0.9727	0.5000	0.7532	0.7766	0.723
31	30	8	20	6	2.0	0.001	0.9989	0.5000	0.7861	0.7737	0.736
32	31	8	20	6	2.5	0.005	1.0000	0.5000	0.7901	0.7731	0.737
33	32	8	24	5	0.2	0.001	0.8576	0.5000	0.7997	0.7341	0.663
34	33	8	24	6	1.5	0.005	0.8535	0.5000	0.7752	0.7704	0.663
35	34	12	20	6	3.0	0.001	0.9685	0.5000	0.8316	0.7950	0.723
36	35	12	16	4	2.0	0.01	0.1194	0.5000	0.8936	0.7919	0.298
37	36	8	20	5	2.0	0.01	0.5642	0.5000	0.8316	0.7299	0.515
38	37	12	16	5	1.5	0.005	0.6823	0.5000	0.8549	0.7603	0.577
39	38	8	16	5	0.5	0.01	0.0441	0.5000	0.8134	0.7313	0.256
40	39	8	16	6	0.5	0.01	0.9513	0.5000	0.7599	0.7751	0.713
41	40	12	20	5	2.5	0.0025	0.5046	0.5000	0.8657	0.7564	0.488
42	41	8	24	6	2.5	0.0025	0.8282	0.5000	0.7848	0.7679	0.651
43	42	12	16	5	0.8	0.0025	0.5815	0.5000	0.8495	0.7612	0.527
44	43	12	16	5	2.0	0.0025	0.6580	0.5000	0.8582	0.7603	0.565
45	44	8	24	4	4.0	0.005	0.8890	0.5000	0.8243	0.7739	0.681
46	45	8	20	5	1.5	0.0025	0.5440	0.5000	0.8281	0.7300	0.505
47	46	12	20	6	2.5	0.005	0.9693	0.5000	0.8292	0.7960	0.723
48	47	8	20	4	4.0	0.005	0.9940	0.5000	0.8248	0.7698	0.734
49	48	12	20	6	0.5	0.001	0.9573	0.5000	0.8099	0.7987	0.718
50	49	8	24	6	3.0	0.0025	0.8172	0.5000	0.7880	0.7671	0.645
51	50	12	20	5	2.0	0.005	0.4640	0.5000	0.8627	0.7568	0.468
52	51	8	16	6	0.2	0.01	0.9727	0.5000	0.7532	0.7766	0.723
53	52	12	16	4	2.0	0.005	0.1194	0.5000	0.8936	0.7919	0.298
54	53	8	16	5	0.8	0.001	0.0956	0.5000	0.8172	0.7311	0.281
55	54	12	16	4	3.0	0.0025	0.1279	0.5000	0.8963	0.7910	0.302
56	55	12	20	4	4.0	0.0025	0.6220	0.5000	0.8706	0.8178	0.551
57	56	8	16	4	2.0	0.001	0.9930	0.5000	0.8132	0.7740	0.733
58	57	8	20	5	0.5	0.01	0.6487	0.5000	0.8156	0.7306	0.558
59	58	8	20	6	0.5	0.001	0.9954	0.5000	0.7631	0.7778	0.735
60	59	12	20	5	0.8	0.0025	0.4467	0.5000	0.8546	0.7574	0.459
61	60	8	24	5	3.0	0.01	1.0000	0.5000	0.8272	0.7326	0.734
62	61	8	20	6	2.5	0.005	1.0000	0.5000	0.7901	0.7731	0.737
63	62	8	20	6	0.2	0.0025	0.9951	0.5000	0.7567	0.7787	0.735
64	63	8	20	6	2.0	0.001	0.9989	0.5000	0.7861	0.7737	0.736
65	64	8	20	6	0.5	0.001	0.9954	0.5000	0.7631	0.7778	0.735
66	65	8	20	6	1.0	0.0025	0.9948	0.5000	0.7727	0.7766	0.735
67	66	8	24	5	3.0	0.01	1.0000	0.5000	0.8272	0.7326	0.734
68	67	8	20	4	3.0	0.005	0.9923	0.5000	0.8210	0.7704	0.733
69	68	8	20	4	4.5	0.005	0.9968	0.5000	0.8262	0.7694	0.735
70	69	8	20	4	1.5	0.01	0.9849	0.5000	0.8107	0.7730	0.729
71	70	8	20	4	3.0	0.01	0.9923	0.5000	0.8210	0.7704	0.733
72	71	8	20	4	4.5	0.01	0.9968	0.5000	0.8262	0.7694	0.735
73	72	12	16	5	3.0	0.01	0.4883	0.5000	0.8621	0.7601	0.480
74	73	8	20	4	0.5	0.005	0.9633	0.5000	0.7974	0.7763	0.719
75	74	8	20	4	3.6	0.01	0.9938	0.5000	0.8233	0.7703	0.734
76	75	8	24	5	2.8	0.01	1.0000	0.5000	0.8264	0.7328	0.734
77	76	8	24	5	6.6	0.01	1.0000	0.5000	0.8388	0.7325	0.734
78	77	12	20	4	6.6	0.005	0.6153	0.5000	0.8762	0.8175	0.548
79	78	8	20	5	11.6	0.0025	0.6640	0.5000	0.8542	0.7279	0.565
80	79	8	20	5	11.6	0.005	0.6640	0.5000	0.8542	0.7279	0.565
81	80	12	20	5	6.6	0.005	0.4746	0.5000	0.8746	0.7550	0.473
82	81	12	16	5	6.6	0.0025	0.1148	0.5000	0.8703	0.7603	0.293
83	82	8	20	3	11.6	0.005	0.3516	0.5000	0.8896	0.9309	0.425
84	83	12	20	3	11.6	0.005	0.6242	0.5000	0.9090	0.9625	0.565
85	84	8	20	4	11.6	0.005	1.0000	0.5000	0.8394	0.7679	0.736
86	85	12	16	5	6.6	0.01	0.1148	0.5000	0.8703	0.7603	0.293
87	86	8	16	5	4.1	0.01	0.3654	0.5000	0.8393	0.7299	0.416
88	87	8	16	3	1.6	0.005	0.6215	0.5000	0.8682	0.9326	0.560
89	88	12	16	4	11.6	0.005	0.1388	0.5000	0.9059	0.7863	0.308
90	89	8	16	5	6.6	0.005	0.7137	0.5000	0.8450	0.7291	0.590
91	90	8	16	3	9.1	0.0025	0.5616	0.5000	0.8850	0.9354	0.531
92	91	12	20	3	11.6	0.0025	0.6242	0.5000	0.9090	0.9625	0.565
93	92	12	20	4	11.6	0.005	0.6094	0.5000	0.8818	0.8173	0.545
94	93	8	20	4	11.6	0.0025	1.0000	0.5000	0.8394	0.7679	0.736
95	94	12	16	3	1.6	0.0025	0.6979	0.5000	0.8880	0.9557	0.600
96	95	12	20	5	1.6	0.0025	0.4685	0.5000	0.8606	0.7571	0.470
97	96	12	20	5	9.1	0.005	0.4690	0.5000	0.8768	0.7550	0.470
98	97	8	20	3	4.1	0.0025	0.4179	0.5000	0.8797	0.9309	0.458
99	98	8	16	3	9.1	0.005	0.5616	0.5000	0.8850	0.9354	0.531
100	99	8	20	4	6.6	0.01	0.9996	0.5000	0.8311	0.7689	0.736"""
    
    # Parser les données
    df = parse_extended_trial_data(extended_data)
    print(f"Total des essais analysés: {len(df)}")
    print(f"\nStatistiques descriptives des scores de qualité:")
    print(df['Quality_Score'].describe())
    
    # Analyser les meilleurs performeurs
    results = analyze_top_performers_extended(df)
    
    print("\n" + "="*80)
    print("TOP 10 PAR QUALITY_SCORE")
    print("="*80)
    for i, (_, row) in enumerate(results['top_quality'].iterrows(), 1):
        print(f"{i}. EWMA={row['EWMA_Short']}+{row['EWMA_Long']} | States={row['States']} | "
              f"κ={row['Kappa']:.1f} | Quality={row['Quality_Score']:.3f} | "
              f"Silhouette={row['Silhouette']:.3f} | Temporal={row['Temporal_Smoothness']:.3f} | "
              f"Balance={row['Balance']:.3f}")
    
    print("\n" + "="*80)
    print("TOP 10 PAR SILHOUETTE (MEILLEURE SÉPARATION DES CLUSTERS)")
    print("="*80)
    for i, (_, row) in enumerate(results['top_silhouette'].iterrows(), 1):
        print(f"{i}. EWMA={row['EWMA_Short']}+{row['EWMA_Long']} | States={row['States']} | "
              f"κ={row['Kappa']:.1f} | Silhouette={row['Silhouette']:.3f} | "
              f"Quality={row['Quality_Score']:.3f} | Temporal={row['Temporal_Smoothness']:.3f} | "
              f"Balance={row['Balance']:.3f}")
    
    print("\n" + "="*80)
    print("TOP 10 PAR TEMPORAL_SMOOTHNESS (MEILLEURE COHÉRENCE TEMPORELLE)")
    print("="*80)
    for i, (_, row) in enumerate(results['top_temporal_smoothness'].iterrows(), 1):
        print(f"{i}. EWMA={row['EWMA_Short']}+{row['EWMA_Long']} | States={row['States']} | "
              f"κ={row['Kappa']:.1f} | Temporal={row['Temporal_Smoothness']:.3f} | "
              f"Quality={row['Quality_Score']:.3f} | Silhouette={row['Silhouette']:.3f} | "
              f"Balance={row['Balance']:.3f}")
    
    print("\n" + "="*80)
    print("TOP 10 PAR BALANCE")
    print("="*80)
    for i, (_, row) in enumerate(results['top_balance'].iterrows(), 1):
        print(f"{i}. EWMA={row['EWMA_Short']}+{row['EWMA_Long']} | States={row['States']} | "
              f"κ={row['Kappa']:.1f} | Balance={row['Balance']:.3f} | "
              f"Quality={row['Quality_Score']:.3f} | Silhouette={row['Silhouette']:.3f} | "
              f"Temporal={row['Temporal_Smoothness']:.3f}")
    
    print("\n" + "="*80)
    print("TOP 10 PAR SCORE COMPOSITE (MOYENNE DES 4 MÉTRIQUES)")
    print("="*80)
    for i, (_, row) in enumerate(results['top_composite'].iterrows(), 1):
        print(f"{i}. EWMA={row['EWMA_Short']}+{row['EWMA_Long']} | States={row['States']} | "
              f"κ={row['Kappa']:.1f} | Composite={row['composite_score']:.3f} | "
              f"Quality={row['Quality_Score']:.3f} | Silhouette={row['Silhouette']:.3f} | "
              f"Temporal={row['Temporal_Smoothness']:.3f} | Balance={row['Balance']:.3f}")
    
    # Analyser les patterns de paramètres
    patterns = analyze_parameter_patterns(df)
    
    print("\n" + "="*80)
    print("ANALYSE DES PARAMÈTRES")
    print("="*80)
    
    print("\nAnalyse par EWMA_Short:")
    print(patterns['ewma_short_analysis'])
    
    print("\nAnalyse par EWMA_Long:")
    print(patterns['ewma_long_analysis'])
    
    print("\nAnalyse par States:")
    print(patterns['states_analysis'])
    
    print("\nAnalyse par Kappa ranges:")
    print(patterns['kappa_analysis'])
    
    # Identifier les configurations optimales
    print("\n" + "="*80)
    print("CONFIGURATIONS OPTIMALES RECOMMANDÉES")
    print("="*80)
    
    if len(results['top_composite']) > 0:
        # Meilleure configuration globale
        best_overall = results['top_composite'].iloc[0]
        print(f"\n🏆 MEILLEURE CONFIGURATION GLOBALE:")
        print(f"   EWMA={best_overall['EWMA_Short']}+{best_overall['EWMA_Long']} | States={best_overall['States']} | "
              f"κ={best_overall['Kappa']:.1f} | Min_Cov={best_overall['Min_Cov']:.4f}")
        print(f"   Scores: Quality={best_overall['Quality_Score']:.3f} | Silhouette={best_overall['Silhouette']:.3f} | "
              f"Temporal={best_overall['Temporal_Smoothness']:.3f} | Balance={best_overall['Balance']:.3f}")
    
    if len(results['top_quality']) > 0:
        # Meilleure pour la qualité pure
        best_quality = results['top_quality'].iloc[0]
        print(f"\n🎯 MEILLEURE POUR LA QUALITÉ PURE:")
        print(f"   EWMA={best_quality['EWMA_Short']}+{best_quality['EWMA_Long']} | States={best_quality['States']} | "
              f"κ={best_quality['Kappa']:.1f} | Min_Cov={best_quality['Min_Cov']:.4f}")
        print(f"   Quality_Score={best_quality['Quality_Score']:.3f}")
    
    if len(results['top_silhouette']) > 0:
        # Meilleure pour la séparation des clusters
        best_silhouette = results['top_silhouette'].iloc[0]
        print(f"\n🔍 MEILLEURE POUR LA SÉPARATION DES CLUSTERS:")
        print(f"   EWMA={best_silhouette['EWMA_Short']}+{best_silhouette['EWMA_Long']} | States={best_silhouette['States']} | "
              f"κ={best_silhouette['Kappa']:.1f} | Min_Cov={best_silhouette['Min_Cov']:.4f}")
        print(f"   Silhouette={best_silhouette['Silhouette']:.3f}")
    
    if len(results['top_temporal_smoothness']) > 0:
        # Meilleure pour la cohérence temporelle
        best_temporal = results['top_temporal_smoothness'].iloc[0]
        print(f"\n⏰ MEILLEURE POUR LA COHÉRENCE TEMPORELLE:")
        print(f"   EWMA={best_temporal['EWMA_Short']}+{best_temporal['EWMA_Long']} | States={best_temporal['States']} | "
              f"κ={best_temporal['Kappa']:.1f} | Min_Cov={best_temporal['Min_Cov']:.4f}")
        print(f"   Temporal_Smoothness={best_temporal['Temporal_Smoothness']:.3f}")
    
    return df, results, patterns

if __name__ == "__main__":
    df, results, patterns = main()