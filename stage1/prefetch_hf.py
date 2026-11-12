"""Fetch only inference files, using validated range downloads for large weights."""
import argparse,time
from pathlib import Path
import requests
from download import download
p=argparse.ArgumentParser();p.add_argument('repo');p.add_argument('dest');p.add_argument('--fp16',action='store_true');a=p.parse_args()
dest=Path(a.dest);files=requests.get('https://huggingface.co/api/models/'+a.repo,timeout=30).json()['siblings']
for item in files:
    name=item['rfilename']
    small=name.endswith('.json') or name.endswith('merges.txt') or name=='vocab.txt'
    weight=name.endswith('.fp16.safetensors') if a.fp16 else name=='pytorch_model.bin'
    if not (small or weight):continue
    path=dest/name;path.parent.mkdir(parents=True,exist_ok=True);url=f'https://huggingface.co/{a.repo}/resolve/main/{name}'
    if weight:download(url,path,workers=32)
    else:
        for attempt in range(5):
            try:
                r=requests.get(url,timeout=30);r.raise_for_status();path.write_bytes(r.content);break
            except Exception:
                if attempt==4:raise
                time.sleep(2**attempt)
    print('Fetched',name,flush=True)
