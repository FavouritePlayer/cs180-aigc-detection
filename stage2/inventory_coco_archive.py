"""Inventory a provided COCO indoor ZIP for deterministic matched sampling.

The JSON contains the archive's captions and is intended for local use, not Git.
"""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--out',required=True);a=p.parse_args();archive=Path(a.archive);sha=hashlib.sha256()
with archive.open('rb') as f:
 for b in iter(lambda:f.read(8*1024*1024),b''):sha.update(b)
assert sha.hexdigest()=='c847a0655de51d16724852da1dbe155b0998b8941f80c8a47b2577b8dc71a089','Different archive version'
rows=[]
with zipfile.ZipFile(archive) as z:
 source=list(csv.DictReader(io.StringIO(z.read('coco_indoor_all/metadata.csv').decode('utf-8-sig'))));assert len(source)==16452
 names={i.filename for i in z.infolist() if not i.is_dir()};expected={'coco_indoor_all/metadata.csv'}|{'coco_indoor_all/'+r['path'] for r in source};assert names==expected
 for r in source:
  b=z.read('coco_indoor_all/'+r['path']);assert len(b)==int(r['out_bytes'])
  with Image.open(io.BytesIO(b)) as im:
   rgb=im.convert('RGB');rgb.load();r.update(width=rgb.width,height=rgb.height,decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
  assert (r['generator']=='real')==(r['label_A']=='0');assert r['path'].split('/')[1]==r['generator']
  r.update(sha256=hashlib.sha256(b).hexdigest(),bytes=len(b));rows.append(r)
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2)+'\n');print('Inventoried',len(rows),'images; archive and image CRCs verified during read')
