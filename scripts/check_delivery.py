"""Dependency-free release check used before static deployment."""
import ast,hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
config=json.loads((root/'app/config.json').read_text())
assert hashlib.sha256((root/'app/model.onnx').read_bytes()).hexdigest()==config['model_sha256']
count=0
for path in root.glob('0*.ipynb'):
    notebook=json.loads(path.read_text());executed=0
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            ast.parse(''.join(cell['source']))
            assert cell['execution_count'] is not None,(path,'Unexecuted cell')
            executed+=1
            assert not any(output.get('output_type')=='error' for output in cell.get('outputs',[])),path
    assert executed>3;count+=1
assert count==5
for relative,record in json.loads((root/'app/vendor/provenance.json').read_text())['files'].items():
    assert hashlib.sha256((root/'app/vendor'/relative).read_bytes()).hexdigest()==record['sha256']
print('Five executed notebooks, model hash and vendor provenance pass.')
