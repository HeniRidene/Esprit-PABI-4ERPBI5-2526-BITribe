import json
import subprocess

# Load notebook
with open('advanced_nlp.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Append lines to the specific cell
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        if len(cell['source']) > 0 and '--- 3.4 Plot: Actual vs LSTM vs Prophet' in cell['source'][0]:
            # Ensure no duplicates
            content = "".join(cell['source'])
            if "model.save" not in content:
                cell['source'].append("\n# Save model and scaler\n")
                cell['source'].append("model.save('outputs/lstm_model.keras')\n")
                cell['source'].append("import joblib\n")
                cell['source'].append("joblib.dump(scaler, 'outputs/scaler.pkl')\n")
                cell['source'].append("print('Saved → outputs/lstm_model.keras and outputs/scaler.pkl')\n")

# Save the notebook
with open('advanced_nlp.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully. Running notebook...")

# Run notebook
subprocess.run([
    "python", "-m", "jupyter", "nbconvert", 
    "--to", "notebook", 
    "--execute", 
    "--inplace", 
    "advanced_nlp.ipynb"
], check=True)
