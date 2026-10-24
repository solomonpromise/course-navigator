"""Copy verified notebooks into the static download directory before serving."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[1]
target=root/'app/notebooks';target.mkdir(exist_ok=True)
for path in sorted(root.glob('0*.ipynb')):
    shutil.copy2(path,target/path.name)
print('Prepared five notebook downloads.')
