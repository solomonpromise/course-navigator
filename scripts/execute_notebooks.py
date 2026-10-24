"""Execute every teaching notebook in a fresh kernel; stop on the first error."""
from pathlib import Path
import json,time
import nbformat
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[1]
records=[]
for path in sorted(ROOT.glob('0*.ipynb')):
    print('EXECUTING',path.name,flush=True)
    notebook=nbformat.read(path,as_version=4)
    start=time.perf_counter()
    NotebookClient(notebook,timeout=600,kernel_name='coursenav',resources={'metadata':{'path':str(ROOT)}}).execute()
    notebook.metadata.kernelspec={'display_name':'Python 3 (Course Navigator)','language':'python','name':'python3'}
    nbformat.write(notebook,path)
    records.append(dict(notebook=path.name,code_cells=sum(c.cell_type=='code' for c in notebook.cells),seconds=time.perf_counter()-start,errors=0))
    print('PASS',path.name,round(records[-1]['seconds'],1),'seconds',flush=True)
(ROOT/'docs').mkdir(exist_ok=True)
(ROOT/'docs/notebook-execution.json').write_text(json.dumps(records,indent=2))
