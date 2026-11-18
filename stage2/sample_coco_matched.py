"""Freeze 50 six-way COCO indoor caption groups without using model scores."""
import argparse,hashlib,json,collections
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('inventory');p.add_argument('--archive-sha256',required=True);p.add_argument('--out',required=True);a=p.parse_args()
rows=json.loads(Path(a.inventory).read_text());groups=collections.defaultdict(list)
for r in rows:groups[(r['caption_id'],r['instance'])].append(r)
non_test_captions={r['caption_id'] for r in rows if r['split']!='test'};non_test_hashes={r['decoded_rgb_sha256'] for r in rows if r['split']!='test'}
generators={'real','sd21','sdxl','sd3','dalle3','midjourney'};seed='cs180-coco-indoor-matched-pilot-v1'
eligible=[(k,v) for k,v in groups.items() if len(v)==6 and {r['generator'] for r in v}==generators and all(r['split']=='test' for r in v) and k[0] not in non_test_captions and all(r['decoded_rgb_sha256'] not in non_test_hashes for r in v) and len({r['decoded_rgb_sha256'] for r in v})==6]
eligible.sort(key=lambda kv:hashlib.sha256((seed+'|'+kv[0][0]+'|'+kv[0][1]).encode()).hexdigest())
selected=[];captions=set();hashes=set()
for key,group in eligible:
 if key[0] in captions or any(r['decoded_rgb_sha256'] in hashes for r in group):continue
 for r in sorted(group,key=lambda x:x['generator']):
  selected.append({'id':'i'+hashlib.sha256(r['path'].encode()).hexdigest()[:16],'file':r['path'],'label':'real' if r['generator']=='real' else 'ai','generator':r['generator'],'group_id':key[0]+'_'+key[1],'caption_id':key[0],'instance':key[1],'split':'test','sha256':r['sha256'],'decoded_rgb_sha256':r['decoded_rgb_sha256'],'width':r['width'],'height':r['height'],'bytes':r['bytes'],'archive_sha256':a.archive_sha256,'sampling_seed':seed})
 captions.add(key[0]);hashes.update(r['decoded_rgb_sha256'] for r in group)
 if len(captions)==50:break
assert len(selected)==300 and len({r['id'] for r in selected})==300
Path(a.out).write_text(json.dumps(selected,indent=2)+'\n')
print('Frozen',len(selected),'images;',len(captions),'caption groups;',len(eligible),'eligible groups')
