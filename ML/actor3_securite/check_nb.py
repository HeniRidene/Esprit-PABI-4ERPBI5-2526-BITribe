import json

with open('actor3_securite.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

checks = {
    'os.chdir': False,
    'drop_duplicates': False,
    'zone_sk <= 10': False,
    'Final clean shape': False,
    'nb_mortels': False,
    'gravite_index': False,
    'crime_rate_scaled': False,
    'GridSearchCV': False,
    'SVC': False,
    'cross_val_predict': False,
    'ROC': False,
    'F1/AUC table (res_df)': False,
    'AgglomerativeClustering': False,
    'davies_bouldin_score': False,
    'dendrogram': False,
    'PCA': False,
    'Zone risk table (High/Med/Low)': False,
    'IsolationForest': False,
    'Score distribution plot': False,
    'Anomalies by zone': False,
    'Precision@96=0.375': False,
    'densité 19.9': False,
    'gravité 0.1407': False,
    'zone map interpretation': False,
    'dict.fromkeys': False,
    'int() cast (astype(int))': False,
    'colormap list comprehension': False,
}

full_text = ''
for cell in nb['cells']:
    src = ''.join(cell['source'])
    full_text += src + '\n'

check_map = {
    'os.chdir': 'os.chdir',
    'drop_duplicates': 'drop_duplicates',
    'zone_sk <= 10': 'zone_sk <= 10',
    'Final clean shape': 'Final clean shape',
    'nb_mortels': 'nb_mortels',
    'gravite_index': 'gravite_index',
    'crime_rate_scaled': 'crime_rate_scaled',
    'GridSearchCV': 'GridSearchCV',
    'SVC': 'SVC(',
    'cross_val_predict': 'cross_val_predict',
    'ROC': 'roc_curve',
    'F1/AUC table (res_df)': 'res_df',
    'AgglomerativeClustering': 'AgglomerativeClustering',
    'davies_bouldin_score': 'davies_bouldin_score',
    'dendrogram': 'dendrogram(',
    'PCA': 'PCA(',
    'Zone risk table (High/Med/Low)': "'High'",
    'IsolationForest': 'IsolationForest',
    'Score distribution plot': 'plt.hist',
    'Anomalies by zone': 'by_zone',
    'Precision@96=0.375': '0.375',
    'densité 19.9': '19.9',
    'gravité 0.1407': '0.1407',
    'zone map interpretation': 'zone map',
    'dict.fromkeys': 'dict.fromkeys',
    'int() cast (astype(int))': 'astype(int)',
    'colormap list comprehension': 'colors_map[int(',
}

for k, pattern in check_map.items():
    checks[k] = pattern in full_text

missing = [k for k, v in checks.items() if not v]
present = [k for k, v in checks.items() if v]

print(f"\n=== PASSED ({len(present)}/{len(checks)}) ===")
for k in present:
    print(f"  [OK] {k}")
print(f"\n=== MISSING ({len(missing)}) ===")
for k in missing:
    print(f"  [!!] {k}")
