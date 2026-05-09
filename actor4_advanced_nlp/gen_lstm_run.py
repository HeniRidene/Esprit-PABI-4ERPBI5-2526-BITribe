import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open('lstm_congestion.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_lines = []
code_lines.append("import matplotlib\nmatplotlib.use('Agg')\n\n")

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        for line in cell['source']:
            if line.startswith('%') or line.startswith('!'):
                code_lines.append(f"#{line}")
            elif 'display(' in line:
                code_lines.append(line.replace('display(', 'print('))
            else:
                # Replace unicode arrows with ASCII
                line = line.replace('\u2192', '->')
                code_lines.append(line)
        code_lines.append('\n\n')

with open('run_lstm.py', 'w', encoding='utf-8') as f:
    f.writelines(code_lines)

print("Script generated: run_lstm.py")
