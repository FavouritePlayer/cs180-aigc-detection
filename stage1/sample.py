"""Reproducible domain-stratified sample from the official validation split."""
import argparse, hashlib, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root); rows=[]
for domain in ['indoors','outdoor']:
    files=sorted(root.glob(f'val/{domain}/**/*.png'))
    assert len(files)>=100,(domain,len(files))
    # Hash ordering avoids reliance on NumPy/Python RNG versions.
    selected=sorted(files,key=lambda f:hashlib.sha256(('cs180-stage1-v1:'+str(f.relative_to(root))).encode()).hexdigest())[:100]
    for f in selected:
        rows.append(dict(domain=domain,path=str(f.relative_to(root)),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
Path(a.output).write_text(json.dumps(rows,indent=2)+'\n')
