import json

with open('advanced_nlp.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_lines = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        # Ignore lines with magic commands like %matplotlib inline or !pip install
        for line in cell['source']:
            if line.startswith('%') or line.startswith('!'):
                code_lines.append(f"#{line}")
            else:
                code_lines.append(line)
        code_lines.append('\n\n')

# Append save logic
code_lines.append("\n# Save model and scaler\n")
code_lines.append("try:\n")
code_lines.append("    model.save('outputs/lstm_model.keras')\n")
code_lines.append("    import joblib\n")
code_lines.append("    joblib.dump(scaler, 'outputs/scaler.pkl')\n")
code_lines.append("    print('Saved → outputs/lstm_model.keras and outputs/scaler.pkl')\n")
code_lines.append("except Exception as e:\n")
code_lines.append("    print('Failed to save model:', e)\n")

with open('runnable.py', 'w', encoding='utf-8') as f:
    f.writelines(code_lines)
