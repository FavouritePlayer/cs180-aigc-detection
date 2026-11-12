"""Verify downloaded checkpoint bytes against public LFS SHA-256 metadata."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--expected',required=True);p.add_argument('--out',required=True);a=p.parse_args()
results=[]
for row in json.loads(Path(a.expected).read_text()):
    f=Path(a.root)/row['path'];h=hashlib.sha256()
    with f.open('rb') as s:
        for b in iter(lambda:s.read(8*1024*1024),b''):h.update(b)
    actual=h.hexdigest();assert actual==row['sha256'],f'Checkpoint hash mismatch: {f}'
    assert f.stat().st_size==row['bytes'],f'Checkpoint size mismatch: {f}'
    results.append(dict(**row,verified=True));print('Verified',row['path'])
Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
