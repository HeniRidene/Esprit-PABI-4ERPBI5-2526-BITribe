import json

def analyze_nb(filename):
    print(f"=== {filename} ===")
    with open(filename, 'r', encoding='utf-8') as f:
        nb = json.load(f)
        
    for i, cell in enumerate(nb['cells']):
        cell_type = cell.get('cell_type', 'unknown')
        source = "".join(cell.get('source', []))
        outputs = cell.get('outputs', [])
        
        has_image = False
        img_types = []
        for out in outputs:
            if 'data' in out:
                for k in out['data'].keys():
                    if 'image' in k:
                        has_image = True
                        img_types.append(k)
        
        preview = source.split('\n')[0][:50] if source else ''
        print(f"Cell {i} [{cell_type}]: {preview}")
        if cell_type == 'code':
            print(f"  Outputs: {len(outputs)} images: {has_image} {img_types}")

for nb in ['advanced_nlp.ipynb', 'lstm_congestion.ipynb']:
    analyze_nb(nb)
